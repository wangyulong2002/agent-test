import base64
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# 加载项目根目录 .env（OLLAMA / TAVILY / LANGSMITH 等配置）
load_dotenv()

from langchain.agents import create_agent  # noqa: E402
# from langchain.agents.middleware import AgentMiddleware  # noqa: E402  # 已注释：纯文本中间件不再需要
from langchain.chat_models import init_chat_model  # noqa: E402
from langchain.tools import tool  # noqa: E402
# from langchain_core.messages import HumanMessage  # noqa: E402  # 已注释：仅视觉识别用到
from langchain_tavily import TavilySearch  # noqa: E402

# LangSmith 可观测（langchain 1.x 自动读取 LANGSMITH_TRACING / LANGSMITH_API_KEY）
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true").lower() == "true"
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")

# ============ 模型配置 ============
# 主脑：Agnes AI 多模态模型（agnes-2.5-flash），兼容 OpenAI 格式、支持 image_url 输入，
#       直接在入口把食材图片作为图片块交给主脑，无需单独的视觉模型。
#       Base URL / Key / 模型名均可经 .env 的 AGNES_* 变量覆盖。
# 文档：https://wiki.agnes-ai.com  |  Base URL：https://apihub.agnes-ai.com/v1
AGNES_API_KEY = os.getenv("AGNES_API_KEY", "")
AGNES_BASE_URL = os.getenv("AGNES_BASE_URL", "https://apihub.agnes-ai.com/v1")
AGNES_MODEL = os.getenv("AGNES_MODEL", "agnes-2.5-flash")
# 后备主模型（DeepSeek 纯文本，仅在未配置 AGNES_API_KEY 时启用）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# —— 以下视觉模型相关配置已注释（原「本地 Ollama 视觉模型」方案已废弃）——
# OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# VISION_MODEL = os.getenv("VISION_MODEL", "qwen3.5:2b")
# VISION_TIMEOUT = int(os.getenv("VISION_TIMEOUT", "240"))   # 视觉请求超时（秒），冷启动需较大值）


# ============ 系统提示词主体：私人厨师 ============
PRIVATE_CHEF_PROMPT = """你是一名专业、贴心的私人厨师。用户会直接提供一张食材图片（多模态输入，你可以直接读取图片内容），请严格按照以下流程工作：

1. 食材整理：直接读取用户消息中的图片，识别其中的食材名称、大致用量、状态判断，
   整理成一份「当前可用的食材清单」。
2. 搜索食谱：以食材清单为检索关键词，调用 tavily_search 查找可行、真实的食谱；
   优先选择食材完全匹配的食谱，其次考虑可通过少量替换实现的。
3. 打分排名：从营养均衡度、制作难度两个维度对候选食谱分别打分（1-10 分），
   计算综合分并给出排名，简要说明打分理由。
4. 建议报告：将排序后的食谱整理成一份结构化建议报告，
   包含每道菜的评分与理由、所需补充的食材/调料、预估用时，帮助用户做出决策。

约束：以图片中实际可见的食材为准，不臆测；搜索不到合适食谱时如实说明。"""


def _guess_mime(path: str) -> str:
    """根据文件后缀猜测 mime 类型"""
    ext = os.path.splitext(path)[1].lower()
    return {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
        "gif": "image/gif",
    }.get(ext, "image/jpeg")


# ============ 以下为「纯文本中间件 + 本地视觉模型识别」方案，已全部注释 ============
# 主脑已改为直接支持图片的多模态模型，不再需要：
#   - 把 image_url 块「就地识别为文本」的中间件
#   - 调用本地 Ollama 视觉模型的链路
#   - describe_image 工具
#
# 如需恢复旧方案，取消下列注释，并在 _build_agent 中重新挂上 TextOnlyMiddleware()。

# # ============ 纯文本消息中间件（关键防御） ============
# # DeepSeek 是纯文本模型，任何 image_url 类型内容块都会导致 400。
# # 该中间件在每次调用主脑前，把消息中的 image_url 图片块「就地识别为文本」，
# # 确保发往 DeepSeek 的消息永远是纯文本，且主脑直接拿到识别结果。
# # 覆盖所有入口：CLI、LangGraph Studio、后端 API 等。
# _IMAGE_PLACEHOLDER = (
#     "[图片：无法自动识别该图片，请直接使用用户消息中给出的食材识别文本，不要向用户索要图片。]"
# )
#
#
# def _image_ref_from_block(block: dict) -> str:
#     """从图片内容块中提取可识别的图片地址（http URL 或 data URI）。
#
#     兼容 LangGraph Studio / langchain 1.x 的全部图片块格式：
#     - 旧版 OpenAI 风格：{"type": "image_url", "image_url": {"url": ...}}
#     - langchain 1.x image 块：{"type": "image", "url"|"base64"|"file_id": ...}
#     - langchain 1.x file 块：{"type": "file", "url"|"base64"|"file_id": ...}
#     - langchain 1.x v1 格式：{"type": "image", "source_type": "url"|"base64"|"id", ...}
#     - OpenAI file 嵌套：{"type": "file", "file": {"file_data"|"file_id": ...}}
#     """
#     btype = block.get("type")
#     if btype == "image_url":
#         iu = block.get("image_url")
#         if isinstance(iu, dict):
#             return iu.get("url", "") or ""
#         if isinstance(iu, str):
#             return iu
#         return ""
#     if btype in ("image", "file"):
#         # langchain 1.x v1 格式（source_type 显式声明数据来源）
#         source_type = block.get("source_type")
#         if source_type == "url":
#             return block.get("url", "") or ""
#         if source_type == "base64":
#             data = block.get("data", "") or ""
#             mime = block.get("mime_type") or "image/jpeg"
#             if data.startswith("data:"):
#                 return data
#             return f"data:{mime};base64,{data}"
#         if source_type == "id":
#             return f"<file_id:{block.get('file_id', '')}>"
#         # 普通格式
#         url = block.get("url")
#         if url:
#             return url
#         b64 = block.get("base64") or block.get("base64_data")
#         if b64:
#             mime = block.get("mime_type") or "image/jpeg"
#             if b64.startswith("data:"):
#                 return b64
#             return f"data:{mime};base64,{b64}"
#         # OpenAI file 嵌套格式
#         f = block.get("file")
#         if isinstance(f, dict):
#             fd = f.get("file_data")
#             if fd:
#                 mime = block.get("mime_type") or "image/jpeg"
#                 if fd.startswith("data:"):
#                     return fd
#                 return f"data:{mime};base64,{fd}"
#             if f.get("file_id"):
#                 return f"<file_id:{f.get('file_id')}>"
#         # v1 base64 简写：只有 data 字段
#         data = block.get("data")
#         if data:
#             mime = block.get("mime_type") or "image/jpeg"
#             if data.startswith("data:"):
#                 return data
#             return f"data:{mime};base64,{data}"
#         file_id = block.get("file_id")
#         if file_id:
#             return f"<file_id:{file_id}>"
#         return ""
#     return ""
#
#
# def _strip_image_blocks(messages: list) -> list:
#     """把消息内容中的图片等非文本块替换为文本，返回清洗后的新消息列表。
#
#     图片块（image_url / image）会被就地调用本地视觉模型识别，识别结果作为文本交给主脑；
#     其他未知块（如 input_audio 等）只保留 text 字段。
#     """
#     cleaned = []
#     for m in messages:
#         content = m.content
#         if isinstance(content, list):
#             new_blocks = []
#             for block in content:
#                 if isinstance(block, dict):
#                     btype = block.get("type")
#                     if btype == "text":
#                         new_blocks.append(block)
#                     elif btype in ("image_url", "image", "file"):
#                         # file 块可能是非图片文件（PDF/Word 等），仅当 mime 为图片或未声明时走视觉识别
#                         if btype == "file":
#                             mime = str(block.get("mime_type", "") or "")
#                             if mime and not mime.startswith("image/"):
#                                 print(f"⚠️ [中间件] 丢弃非图片 file 块 mime={mime}")
#                                 continue
#                         image_ref = _image_ref_from_block(block)
#                         print(
#                             f"🔎 [中间件] 收到图片块 type={btype}, "
#                             f"ref={str(image_ref)[:80]!r} (长度 {len(image_ref)})"
#                         )
#                         if image_ref and not image_ref.startswith("<file_id:"):
#                             t0 = time.time()
#                             try:
#                                 recognized = _recognize_image(image_ref)
#                                 print(
#                                     f"✅ [中间件] 图片识别完成（{time.time()-t0:.1f}s）："
#                                     f"{str(recognized)[:100]!r}"
#                                 )
#                             except Exception as e:
#                                 recognized = f"[视觉识别异常] {type(e).__name__}: {e}"
#                                 print(f"⚠️ [中间件] 图片识别异常：{recognized}")
#                             new_blocks.append(
#                                 {"type": "text", "text": f"[图片识别结果]\n{recognized}"}
#                             )
#                         else:
#                             print(f"⚠️ [中间件] 图片无可识别数据（file_id/空 ref），使用占位文本")
#                             new_blocks.append({"type": "text", "text": _IMAGE_PLACEHOLDER})
#                     else:
#                         # 其他未知块（如 input_audio 等）：只保留 text 字段，其余丢弃
#                         text = block.get("text", "")
#                         if text:
#                             new_blocks.append({"type": "text", "text": text})
#                         else:
#                             print(f"⚠️ [中间件] 丢弃未知块 type={btype!r}（无 text）")
#                 elif isinstance(block, str):
#                     new_blocks.append({"type": "text", "text": block})
#             if not new_blocks:  # 纯图片消息 → 保证非空
#                 new_blocks = [{"type": "text", "text": _IMAGE_PLACEHOLDER}]
#             content = new_blocks
#         cleaned.append(m.model_copy(update={"content": content}))
#     return cleaned
#
#
# def _dump_messages(messages: list) -> None:
#     """打印主脑收到的每条消息概览，便于定位 Studio 传图的实际结构。"""
#     print("🧩 [中间件] 主脑收到消息：")
#     for i, m in enumerate(messages):
#         content = m.content
#         if isinstance(content, list):
#             kinds = [b.get("type") if isinstance(b, dict) else type(b).__name__ for b in content]
#             preview = "".join(
#                 str(b.get("text", ""))[:60] if isinstance(b, dict) and b.get("text")
#                 else str(b)[:60]
#                 for b in content
#             )
#             print(f"  [{i}] role={getattr(m, 'role', '?')} blocks={kinds} preview={preview[:90]!r}")
#         else:
#             print(f"  [{i}] role={getattr(m, 'role', '?')} content(str)={str(content)[:90]!r}")
#
#
# class TextOnlyMiddleware(AgentMiddleware):
#     """中间件：发往主脑模型的消息强制纯文本化（把 image_url 就地识别为文本）。"""
#
#     def wrap_model_call(self, request, handler):
#         _dump_messages(request.messages)
#         request.messages = _strip_image_blocks(request.messages)
#         return handler(request)
#
#     async def awrap_model_call(self, request, handler):
#         import asyncio
#
#         _dump_messages(request.messages)
#         # 关键：LangGraph 服务端走异步路径，不能在事件循环里同步调用视觉模型
#         # （会抛 BlockingError，导致图片识别失败）。用线程池执行阻塞识别。
#         request.messages = await asyncio.to_thread(_strip_image_blocks, request.messages)
#         return await handler(request)
#
#
# def _ollama_available(model_name: str) -> bool:
#     """探测本地 Ollama 服务与指定模型是否可用（快速失败，不阻塞）"""
#     try:
#         import json
#         import urllib.request
#
#         req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags", method="GET")
#         with urllib.request.urlopen(req, timeout=2) as resp:
#             models = json.loads(resp.read())["models"]
#         names = {m["name"] for m in models}
#         if model_name not in names:
#             print(f"ℹ️  本地 Ollama 未找到模型 {model_name}，请先 ollama pull {model_name}")
#             return False
#         return True
#     except Exception:
#         print(f"ℹ️  本地 Ollama 不可用（请求 {model_name}），请先启动 Ollama 并拉取模型")
#         return False
#
#
# def _build_vision_chain() -> list:
#     """构建视觉模型链：仅本地 VISION_MODEL（默认与主模型一致，可单独配置）"""
#     if _ollama_available(VISION_MODEL):
#         local = init_chat_model(
#             model=VISION_MODEL,
#             model_provider="openai",
#             base_url=OLLAMA_BASE_URL + "/v1",   # Ollama OpenAI 兼容端点
#             api_key="ollama",                    # Ollama 不校验 key
#             timeout=VISION_TIMEOUT,              # 本地模型冷启动可能较慢
#         )
#         return [(f"本地 {VISION_MODEL}", local)]
#     return []
#
#
# VISION_CHAIN = _build_vision_chain()
#
#
# def _invoke_vision(msg) -> str:
#     """按视觉模型链逐个调用；当前模型异常时切下一个，全部失败返回友好提示。
#
#     本地视觉模型偶发超时/不可用，直接抛出会让整个 Agent 流程崩溃。
#     方案：指数退避重试 → 全部失败才返回可读错误，主脑 Agent 能继续处理并如实告知用户。
#     """
#     max_attempts = 3
#     errors = []
#     for name, model in VISION_CHAIN:
#         for attempt in range(1, max_attempts + 1):
#             try:
#                 return model.invoke([msg]).content
#             except Exception as e:  # 超时/网络/本地不可用等，直接切下一个模型
#                 errors.append(f"{name}: {type(e).__name__}: {e}")
#                 if attempt < max_attempts:
#                     wait = 2 ** attempt          # 2s → 4s 指数退避
#                     print(f"⏳ {name} 调用失败，{wait}s 后第 {attempt + 1}/{max_attempts} 次重试...")
#                     time.sleep(wait)
#                 break
#     # 全部模型失败：返回简短提示，避免把大段错误堆栈塞给主脑
#     detail = " | ".join(errors[-2:])
#     return (
#         "[视觉识别失败] 本地视觉模型不可用或未正确支持图片输入。"
#         f"已配置视觉模型：{VISION_MODEL}。"
#         f"最近一次错误：{detail}"
#     )
#
#
# def _recognize_image(image_ref: str, question: str = "识别图中的食材和大致用量，尽量具体") -> str:
#     """识别图片（http(s) URL 或 data URI），返回文字描述。
#
#     由入口与消息中间件统一调用，主脑模型不再直接接触图片。
#     """
#     msg = HumanMessage(
#         content=[
#             {"type": "image_url", "image_url": {"url": image_ref}},
#             {"type": "text", "text": question},
#         ]
#     )
#     return _invoke_vision(msg)
#
#
# @tool
# def describe_image(image_path: str, question: str = "识别图中的食材和大致用量，尽量具体") -> str:
#     """读取本地图片或网络图片 URL，用视觉模型分析后返回文字描述。
#
#     主模型不具备识图能力，需要识别图片内容（食材、用量等）时必须调用本工具。
#
#     Args:
#         image_path: 本地图片文件的绝对路径，或图片的 http(s) URL
#         question: 针对图片提出的问题
#     """
#     image_path = image_path.strip()
#     # URL 直接透传给视觉模型；本地路径读取后转 base64 data URI
#     if image_path.startswith(("http://", "https://")):
#         image_url = image_path
#     else:
#         try:
#             with open(image_path, "rb") as f:
#                 img_b64 = base64.b64encode(f.read()).decode("utf-8")
#         except (FileNotFoundError, IsADirectoryError, OSError) as e:
#             return f"图片读取失败：{e}。请检查路径是否正确（本地路径或 http(s) URL 均可）。"
#         image_url = f"data:{_guess_mime(image_path)};base64,{img_b64}"
#     return _recognize_image(image_url, question)


def _build_agent():
    """实际构建 Agent 的纯函数（无参）。

    拆出此函数的目的是避开 langgraph_api 对 factory 形参的限制：
    langgraph_api 会把任何 1 参 factory 当作 ``(config)`` 工厂，把 config 字典
    塞给唯一参数，导致 ``build_chief(enable_search=...)`` 这种语义被破坏。
    改成无参工厂后，服务端只会无参调用，行为可预期。
    """
    enable_search = bool(os.getenv("TAVILY_API_KEY"))
    # 1. Tavily 网页搜索工具（需要 TAVILY_API_KEY）
    search = None
    if enable_search:
        search = TavilySearch(
            max_results=5,
            topic="general",
            search_depth="basic",
        )
    # 2. 主脑模型：Agnes AI 多模态模型（直接读图，兼容 OpenAI 格式）
    #    未配置 AGNES_API_KEY 时回退到 DeepSeek 纯文本（此时图片需自行处理）。
    if AGNES_API_KEY:
        model = init_chat_model(
            model=AGNES_MODEL,
            model_provider="openai",          # Agnes 完全兼容 OpenAI Chat Completions
            api_key=AGNES_API_KEY,
            base_url=AGNES_BASE_URL,
            temperature=0.7,
        )
        model_label = f"Agnes {AGNES_MODEL}"
    elif DEEPSEEK_API_KEY:
        model = init_chat_model(
            model="deepseek-chat",
            api_key=DEEPSEEK_API_KEY,
        )
        model_label = "deepseek-chat"
    else:
        raise RuntimeError(
            "未配置任何主模型：请在 .env 中填写 AGNES_API_KEY（推荐）或 DEEPSEEK_API_KEY"
        )
    # 3. 组装 Agent（不再挂载 TextOnlyMiddleware，多模态模型直接接收图片）
    return create_agent(
        model=model,
        tools=([search] if search else []),
        system_prompt=PRIVATE_CHEF_PROMPT,
        name="chief",
    )


def build_chief(*args, **kwargs):
    """兼容旧调用：CLI 仍可传 ``enable_search``，langgraph_api 会无参调用。

    任意额外参数被忽略（langgraph_api 会注入 config/runtime，这些对构建无意义）。
    """
    # CLI 旧用法 build_chief(enable_search=True/False) 仍兼容
    return _build_agent()


def chief_graph():
    """零参 graph 工厂，专供 langgraph.json 作为 entry point。

    必须保持零参：langgraph_api 会把任何单参 factory 当作 ``(config)`` 工厂，
    把 config 字典塞给参数，破坏形参语义（曾导致 deepseek-chat 被替换为 qwen）。
    """
    return _build_agent()


def _parse_args(args: list[str]) -> tuple[str, str | None]:
    """解析命令行参数：支持 @image:本地路径 或 http(s) URL 标记图片。

    返回 (text_question, image_path)。
    例如：
      @image:/tmp/0.jpg 帮我分析一下 -> ("帮我分析一下", "/tmp/0.jpg")
      https://xxx.jpg 帮我看看       -> ("帮我看看", "https://xxx.jpg")
    """
    image_path: str | None = None
    text_parts: list[str] = []

    for token in args:
        stripped = token.strip()
        if stripped.lower().startswith("@image:"):
            image_path = stripped[len("@image:"):].strip()
        elif stripped.startswith(("http://", "https://")):
            image_path = stripped
        else:
            text_parts.append(stripped)

    text = " ".join(text_parts)
    if not text:
        text = "帮我看看这张食材图片，分析食材和用量，整理可用食材清单，搜索可行食谱并打分排名，最后给我一份建议报告。"
    return text, image_path


def main():
    text, image_path = _parse_args(sys.argv[1:])

    # 默认示例：未提供任何图片时，使用网络示例图片
    if not image_path:
        image_path = "https://pic.wenwen.soso.com/pqpic/wenwenpic/0/20190730074154-1379515267_jpeg_498_365_43845/0"
        print(f"ℹ️  未检测到图片参数，将使用示例图片：{image_path[:70]}...")
    else:
        print(f"🖼️  检测到图片输入：{image_path}")

    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        print("⚠️  未配置 TAVILY_API_KEY，将以纯推理模式运行（无联网搜索食谱）。")
        print("    启用搜索：https://app.tavily.com 免费注册，把 Key 填入 .env 的 TAVILY_API_KEY")
    if LANGSMITH_API_KEY:
        print(f"✅ LangSmith 追踪已开启（LANGSMITH_TRACING={LANGSMITH_TRACING}）")
    else:
        print("ℹ️  未配置 LANGSMITH_API_KEY，跳过 LangSmith 追踪（不影响运行）。")

    agent = build_chief()
    model_label = (
        f"Agnes {AGNES_MODEL}" if AGNES_API_KEY
        else "deepseek-chat" if DEEPSEEK_API_KEY
        else "未配置"
    )
    print("\n👨‍🍳 私人厨师 Agent 已就绪（主脑：" + model_label + "）\nQ: " + text[:80] + "...\n")

    # 主脑已替换为支持图片的多模态模型：直接在入口把图片作为 image_url 内容块交给主脑，
    # 不再单独调用视觉模型识别。网络图片直接用 URL；本地图片转 data URI。
    # 注意：Agnes 的 image_url 输入需「可公网访问」的图片 URL，本地 data URI 可能不被接受；
    #       本地图片请改用公开可访问的 URL，或先上传到图床/对象存储。
    if image_path:
        print("🖼️  已将图片作为多模态输入直接交给主脑模型")
        if image_path.startswith(("http://", "https://")):
            img_url = image_path
        else:
            try:
                with open(image_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")
            except (FileNotFoundError, IsADirectoryError, OSError) as e:
                print(f"❌ 图片读取失败：{e}")
                sys.exit(1)
            img_url = f"data:{_guess_mime(image_path)};base64,{img_b64}"

        user_prompt = [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": img_url}},
        ]
    else:
        user_prompt = text

    try:
        result = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
        print("=== 建议报告 ===")
        print(result["messages"][-1].content)
    except Exception as e:
        print(f"\n❌ 运行出错：{type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
