# -*- coding: utf-8 -*-
"""AI 代理核心逻辑：私人厨师（设计报告 §6 / §8 agents/personal_chief.py）

主脑：Agnes 多模态（agnes-2.5-flash，OpenAI 兼容，image_url 直接读图）
工具：tavily_search（可选，未配置 TAVILY_API_KEY 时降级为纯推理）
持久化：LangGraph Checkpointer（SqliteSaver，thread_id 会话维度，方案 B）
流式：astream_events(v2) 监听 on_chat_model_stream，产出 token 级增量
"""
from datetime import datetime
from typing import AsyncIterator

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch

from app.common.logger import get_logger
from app.config import (
    AGNES_API_KEY,
    AGNES_BASE_URL,
    AGNES_MODEL,
    DEEPSEEK_API_KEY,
    TAVILY_API_KEY,
    get_checkpointer,
)

logger = get_logger(__name__)

# 系统提示词：私人厨师五步闭环（设计报告 §6.4）
PRIVATE_CHEF_PROMPT = """你是一名专业、贴心的私人厨师。用户会提供一张食材图片（多模态输入，你可以直接查看图片），请严格按照以下流程工作：

1. 食材整理：直接读取用户消息中的图片，识别食材名称、大致用量、状态判断，
   整理成一份「当前可用的食材清单」。
2. 搜索食谱：以食材清单为检索关键词，调用 tavily_search 查找可行、真实的食谱；
   优先选择食材完全匹配的食谱，其次考虑可通过少量替换实现的。
3. 打分排名：从营养均衡度、制作难度两个维度对候选食谱分别打分（1-10 分），
   计算综合分并给出排名，简要说明打分理由。
4. 建议报告：将排序后的食谱整理成一份结构化建议报告，
   包含每道菜的评分与理由、所需补充的食材/调料、预估用时，帮助用户做出决策。

约束：以图片中实际可见的食材为准，不臆测；搜索不到合适食谱时如实说明。"""

# Agent 单例（构建开销大，进程内复用）
_agent = None


def _build_agent():
    """构建私人厨师 Agent（无参工厂，供 API 复用）"""
    tools = []
    if TAVILY_API_KEY:
        logger.info("已启用 tavily_search（食谱检索）")
        tools.append(TavilySearch(max_results=5, topic="general", search_depth="basic"))
    else:
        logger.info("未配置 TAVILY_API_KEY，以纯推理模式运行（无联网搜索）")

    if AGNES_API_KEY:
        model = init_chat_model(
            model=AGNES_MODEL,
            model_provider="openai",          # Agnes 完全兼容 OpenAI Chat Completions
            api_key=AGNES_API_KEY,
            base_url=AGNES_BASE_URL,
            temperature=0.7,
        )
        logger.info("主脑模型：Agnes %s（多模态，直接读图）", AGNES_MODEL)
    elif DEEPSEEK_API_KEY:
        model = init_chat_model(
            model="deepseek-chat",
            api_key=DEEPSEEK_API_KEY,
        )
        logger.info("主脑模型：deepseek-chat（纯文本兜底，图片需另行处理）")
    else:
        raise RuntimeError(
            "未配置主模型：请在 .env 中填写 AGNES_API_KEY（推荐）或 DEEPSEEK_API_KEY"
        )

    return create_agent(
        model=model,
        tools=tools,
        system_prompt=PRIVATE_CHEF_PROMPT,
        name="chief",
        checkpointer=get_checkpointer(),   # 方案 B：SqliteSaver 持久化（thread_id 会话维度）
    )


def get_agent():
    """获取（并缓存）Agent 实例"""
    global _agent
    if _agent is None:
        _agent = _build_agent()
    return _agent


def reset_agent() -> None:
    """重置 Agent 缓存。

    生命周期管理：main.py lifespan 退出时调用（配合 close_checkpointer），
    避免缓存 Agent 持有已关闭的 Checkpointer 连接（TestClient 多次启停场景）。
    """
    global _agent
    _agent = None


def build_user_content(question: str, image_url: str | None) -> str | list:
    """构建发给多模态主脑的用户内容：有图 → [text, image_url]；无图 → 纯文本"""
    if image_url:
        return [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    return question


def _thread_config(thread_id: str) -> dict:
    """生成带 thread_id 的 LangGraph 调用配置"""
    return {"configurable": {"thread_id": thread_id}}


async def stream_chat(
    question: str, image_url: str | None = None, thread_id: str = "default"
) -> AsyncIterator[str]:
    """流式对话：逐段产出文本增量（同一 thread_id 自动带上多轮记忆）。

    通过 astream_events(v2) 监听 on_chat_model_stream 拿到 token 级增量；
    工具调用（Tavily）期间模型无文本输出，不会产生噪音。
    """
    agent = get_agent()
    content = build_user_content(question, image_url)
    async for event in agent.astream_events(
        {"messages": [{"role": "user", "content": content}]},
        version="v2",
        config=_thread_config(thread_id),
    ):
        if event.get("event") != "on_chat_model_stream":
            continue
        chunk = event.get("data", {}).get("chunk")
        text = getattr(chunk, "content", None)
        if isinstance(text, str) and text:
            yield text


def _message_to_item(index: int, msg, created_at: str) -> dict:
    """把 checkpoint 中的 BaseMessage 转为前端展示结构。

    处理：role 映射、content 可能是 str 或 [text/image_url] 块列表、
    过滤图片 URL 提取、过滤 ToolMessage。
    """
    role_map = {"human": "user", "ai": "assistant"}
    mtype = getattr(msg, "type", "")
    if mtype not in role_map:
        return None          # ToolMessage / SystemMessage 等不展示
    content = msg.content
    text_parts: list[str] = []
    image_url = ""
    if isinstance(content, str):
        text_parts.append(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", "") or "")
                elif btype == "image_url":
                    iu = block.get("image_url") or {}
                    image_url = (
                        iu.get("url", "") if isinstance(iu, dict) else str(iu)
                    )
            elif isinstance(block, str):
                text_parts.append(block)
    text = "".join(text_parts)
    return {
        "id": getattr(msg, "id", None) or index,
        "role": role_map[mtype],
        "content": text,
        "image_url": image_url,
        "created_at": created_at,
    }


async def get_history(thread_id: str, limit: int = 50) -> tuple[list[dict], int]:
    """从 Checkpointer 读取会话历史（按对话顺序正序），返回 (items, total)。

    读最新 checkpoint 的 state.messages 派生，与对话内容完全一致（§4.2）。
    不存在的 thread 返回空列表。
    """
    agent = get_agent()
    try:
        snapshot = await agent.aget_state(_thread_config(thread_id))
    except Exception:
        return [], 0

    messages = []
    if snapshot is not None:
        values = getattr(snapshot, "values", None) or {}
        messages = values.get("messages", []) or []

    # created_at：用 checkpoint 创建时间（StateSnapshot.created_at 为 datetime）
    ts = getattr(snapshot, "created_at", None) or datetime.now()
    created_at = ts.strftime("%Y-%m-%d %H:%M:%S") if hasattr(ts, "strftime") else str(ts)

    items = []
    for i, m in enumerate(messages):
        item = _message_to_item(i, m, created_at)
        if item:
            items.append(item)
    # 保留最后 limit 条（checkpoint 内已是时间正序）
    total = len(items)
    return items[-limit:], total


async def clear_history(thread_id: str) -> bool:
    """清空会话：删除该 thread_id 的全部 checkpoints（§4.3）"""
    saver = get_checkpointer()
    try:
        await saver.adelete_thread(thread_id)
        return True
    except Exception:
        return False
