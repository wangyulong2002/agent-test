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
from langchain_core.messages import HumanMessage, RemoveMessage
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
# 主模型引用（用于历史摘要压缩，无图）
_model = None

# 历史管理参数（防止上下文过长导致幻觉）：方案 2 + 方案 3
HISTORY_COMPRESS_THRESHOLD = 18   # 历史消息超过该数量则压缩最旧一段
HISTORY_KEEP = 12                 # 压缩后保留最近 N 条消息
IMAGE_KEEP = 2                    # 仅保留最近 N 条含图消息，更早的降级为文字


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
            temperature=0.5,
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
    global _model
    _model = model


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


def _msg_text(m) -> str:
    """提取消息纯文本（兼容 str 与 [text/image_url] 块列表）。"""
    c = getattr(m, "content", None)
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for b in c:
            if isinstance(b, dict):
                if b.get("type") == "text":
                    parts.append(b.get("text", "") or "")
                elif b.get("type") == "image_url":
                    parts.append("[图片]")
            elif isinstance(b, str):
                parts.append(b)
        return "".join(parts)
    return "" if c is None else str(c)


def _has_image(m) -> bool:
    """消息是否含 image_url 块。"""
    c = getattr(m, "content", None)
    return isinstance(c, list) and any(
        isinstance(b, dict) and b.get("type") == "image_url" for b in c
    )


async def _summarize_messages(msgs: list) -> str:
    """将一批历史消息压缩为一段中文摘要（调用主模型，纯文本、无图）。

    摘要保留：用户食材/图片内容、已确认食谱、用户偏好或禁忌、未完成需求。
    工具消息不进摘要（仅保留其结论已被 ai 消息覆盖的信息）。
    """
    lines = []
    for m in msgs:
        mtype = getattr(m, "type", "")
        role = "用户" if mtype == "human" else ("厨师" if mtype == "ai" else "工具")
        lines.append(f"{role}: {_msg_text(m)}")
    prompt = (
        "请把以下多轮私厨对话压缩成一段简洁中文摘要，重点保留：用户提到的食材与图片内容、"
        "已确认的食谱/建议、用户偏好或禁忌、尚未完成的需求。不要编造信息。\n\n"
        + "\n".join(lines)
    )
    resp = await _model.ainvoke([{"role": "user", "content": prompt}])
    content = resp.content
    if isinstance(content, list):
        content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return "【历史摘要】" + (content if isinstance(content, str) else str(content))


async def _limit_images(agent, thread_id: str, messages: list, image_keep: int = IMAGE_KEEP) -> None:
    """方案 3：只保留最近 image_keep 条含图消息，更早的含图消息降级为文字占位。

    多模态图片是上下文大头，限制进入历史的图片数量可显著缓解长对话溢出/幻觉。
    """
    img_ids = [m.id for m in messages if getattr(m, "type", "") == "human" and _has_image(m)]
    if len(img_ids) <= image_keep:
        return
    downgrade_ids = set(img_ids[:-image_keep])
    updates = []
    for m in messages:
        if m.id in downgrade_ids:
            note = _msg_text(m) + "\n（原含食材图片，已移除以节省上下文，图片内容见历史摘要）"
            updates.append(HumanMessage(id=m.id, content=note))
    if updates:
        await agent.aupdate_state(_thread_config(thread_id), {"messages": updates})


async def _compress_history(
    agent, thread_id: str, messages: list,
    keep: int = HISTORY_KEEP, threshold: int = HISTORY_COMPRESS_THRESHOLD,
) -> None:
    """方案 2：历史超过阈值时，把最旧一段压缩成摘要并移除原文，防止上下文溢出与幻觉。

    用最旧一段的首条 human/ai 消息承载摘要（原地替换，保持其在 kept 之前的位置），
    其余旧消息通过 RemoveMessage 删除。
    """
    if len(messages) <= threshold:
        return
    to_compress = messages[:-keep]
    anchor = next(
        (m for m in to_compress if getattr(m, "type", "") in ("human", "ai")),
        to_compress[0],
    )
    summary = await _summarize_messages(to_compress)
    updates = [HumanMessage(id=anchor.id, content=summary)]
    for m in to_compress:
        if m.id != anchor.id:
            updates.append(RemoveMessage(id=m.id))
    await agent.aupdate_state(_thread_config(thread_id), {"messages": updates})


async def _manage_history(agent, thread_id: str) -> None:
    """每轮对话前整理历史：方案 3（限额图片）→ 方案 2（超阈值摘要压缩）。"""
    try:
        snap = await agent.aget_state(_thread_config(thread_id))
    except Exception:
        return
    msgs = getattr(snap, "values", {}).get("messages", []) or []
    if not msgs:
        return
    await _limit_images(agent, thread_id, msgs)
    # 图片降级可能改变了状态，重读后再压缩
    try:
        snap = await agent.aget_state(_thread_config(thread_id))
    except Exception:
        return
    msgs = getattr(snap, "values", {}).get("messages", []) or []
    await _compress_history(agent, thread_id, msgs)


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
    await _manage_history(agent, thread_id)   # 方案 2+3：压缩旧历史、限额图片，防幻觉
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
