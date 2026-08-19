# -*- coding: utf-8 -*-
"""应用配置：.env 加载、路径常量、Checkpointer 单例（设计报告 §1/§5/§6.1）

路径约定：数据库 data/app.db（SqliteSaver checkpoint 落盘）、静态产物 app/static/。
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# 项目根目录（agent-test/）
BASE_DIR = Path(__file__).resolve().parent.parent
# 应用目录（app/）
APP_DIR = Path(__file__).resolve().parent

# 目录
DATA_DIR = BASE_DIR / "data"               # SQLite 数据目录
DB_PATH = DATA_DIR / "app.db"              # 数据库文件（设计报告 §5）
STATIC_DIR = APP_DIR / "static"            # Next.js 编译产物（设计报告 §8）

# 加载 .env（AGNES / TAVILY / LANGSMITH / OSS 等配置）
load_dotenv(BASE_DIR / ".env")

# ============ 主脑模型：Agnes 多模态（设计报告 §6.1） ============
AGNES_API_KEY = os.getenv("AGNES_API_KEY", "")
AGNES_BASE_URL = os.getenv("AGNES_BASE_URL", "https://apihub.agnes-ai.com/v1")
AGNES_MODEL = os.getenv("AGNES_MODEL", "agnes-2.5-flash")

# 兜底主模型（纯文本，未配置 Agnes 时启用）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# ============ 搜索工具（可选，设计报告 §6.2） ============
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# ============ OSS 对象存储（预签名直传，设计报告 §4.4） ============
OSS_ACCESS_KEY_ID = os.getenv("OSS_ACCESS_KEY_ID", "")
OSS_ACCESS_KEY_SECRET = os.getenv("OSS_ACCESS_KEY_SECRET", "")
OSS_ENDPOINT = os.getenv("OSS_ENDPOINT", "")       # 例：oss-cn-hangzhou.aliyuncs.com
OSS_BUCKET = os.getenv("OSS_BUCKET", "")
OSS_EXPIRE_SECONDS = int(os.getenv("OSS_EXPIRE_SECONDS", "3600"))
OSS_ENABLED = bool(
    OSS_ACCESS_KEY_ID and OSS_ACCESS_KEY_SECRET and OSS_ENDPOINT and OSS_BUCKET
)


def ensure_dirs() -> None:
    """确保 data/、static/ 目录存在"""
    for d in (DATA_DIR, STATIC_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ============ LangGraph Checkpointer（设计报告 §5，方案 B） ============
# AsyncSqliteSaver.from_conn_string 返回 async context manager，
# 在 FastAPI lifespan 中 __aenter__ 打开连接并缓存为进程级单例。
_checkpointer: AsyncSqliteSaver | None = None
_checkpointer_cm = None


async def init_checkpointer() -> AsyncSqliteSaver:
    """打开 AsyncSqliteSaver 连接并缓存（幂等），由 main.py lifespan 调用。

    - 必须用 AsyncSqliteSaver：langgraph-checkpoint-sqlite 3.x 的同步
      SqliteSaver 不支持 async 方法，async 运行时（astream_events /
      aget_state）会抛 NotImplementedError
    - 落盘：data/app.db（checkpoints 系列表，setup 幂等建表）
    """
    global _checkpointer, _checkpointer_cm
    if _checkpointer is None:
        ensure_dirs()
        _checkpointer_cm = AsyncSqliteSaver.from_conn_string(str(DB_PATH))
        _checkpointer = await _checkpointer_cm.__aenter__()
        await _checkpointer.setup()
    return _checkpointer


def get_checkpointer() -> AsyncSqliteSaver:
    """同步获取已初始化的 saver（必须在 init_checkpointer 之后调用）"""
    if _checkpointer is None:
        raise RuntimeError("Checkpointer 未初始化：请先 await init_checkpointer()")
    return _checkpointer


async def close_checkpointer() -> None:
    """关闭 Checkpointer 连接（进程退出时），由 main.py lifespan 调用"""
    global _checkpointer, _checkpointer_cm
    if _checkpointer_cm is not None:
        await _checkpointer_cm.__aexit__(None, None, None)
        _checkpointer = None
        _checkpointer_cm = None
