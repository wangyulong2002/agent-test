# -*- coding: utf-8 -*-
"""FastAPI 入口：注册路由、CORS、静态目录（设计报告 §8 app/main.py）"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.agents import personal_chief
from app.api.v1 import chat, oss
from app.common.logger import get_logger
from app.config import (
    STATIC_DIR,
    close_checkpointer,
    ensure_dirs,
    init_checkpointer,
)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时初始化 Checkpointer（设计报告 §5），退出时关闭连接"""
    ensure_dirs()
    await init_checkpointer()
    logger.info("Checkpointer 就绪：data/app.db")
    try:
        yield
    finally:
        # 关闭 Checkpointer 连接后重置 Agent 缓存，避免持有已关闭连接的实例
        await close_checkpointer()
        personal_chief.reset_agent()


app = FastAPI(
    title="私厨 AI",
    description="图片识别生成菜谱（流式对话）",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS：Next.js 前端联调需要（设计报告 §10 联调注意）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 路由
app.include_router(chat.router)
app.include_router(oss.router)

# 静态托管：Next.js 编译产物（最后挂载，避免抢占 /chat /oss 等 API 路由）
ensure_dirs()
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
