# -*- coding: utf-8 -*-
"""对话 API（设计报告 §4.1~4.3 / §8 api/v1/chat.py）

- POST   /chat/stream            流式对话（SSE，thread_id 会话维度）
- GET    /chat/messages          获取会话历史（从 Checkpointer 派生）
- DELETE /chat/messages          清空会话（删除 thread 的 checkpoints）
"""
import json

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from app.agents import personal_chief
from app.common.logger import get_logger
from app.models import schemas

router = APIRouter(prefix="/chat", tags=["chat"])
logger = get_logger(__name__)


def _sse(payload: dict) -> str:
    """序列化 SSE 帧（data: <json> + 空行）"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/stream")
async def stream(body: schemas.ChatRequest):
    """流式对话：SSE 返回，每帧 {"delta": ..., "done": bool}"""
    thread_id = body.thread_id.strip()
    if not thread_id:
        return schemas.err(40001, "thread_id 不能为空")
    message = body.message.strip()
    if not message:
        return schemas.err(40001, "消息不能为空")
    image_url = (body.image_url or "").strip() or None
    if image_url and not image_url.startswith(("http://", "https://")):
        return schemas.err(40001, "image_url 必须为 http(s) 公网图片地址")

    async def event_gen():
        collected: list[str] = []
        try:
            # Checkpointer 自动带上该 thread 的多轮记忆（§6.3），无需手工拼历史
            async for delta in personal_chief.stream_chat(message, image_url, thread_id):
                collected.append(delta)
                yield _sse({"delta": delta, "done": False})
            yield _sse({"delta": "", "done": True, "data": {"thread_id": thread_id}})
        except Exception as e:
            # 设计报告 §6.6：异常以 SSE error 帧返回并写入日志
            logger.error("AI 流式对话异常（thread=%s）", thread_id, exc_info=True)
            yield _sse(
                {
                    "delta": "",
                    "done": True,
                    "error": f"AI 服务异常，请稍后再试（{type(e).__name__}）",
                }
            )

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@router.get("/messages")
async def list_messages(
    thread_id: str = Query(..., min_length=1, max_length=64, description="会话标识"),
    limit: int = Query(50, ge=1, le=200),
):
    """获取会话历史（按对话顺序正序，从 Checkpointer 派生）"""
    items, total = await personal_chief.get_history(thread_id, limit)
    return schemas.ok({"total": total, "items": items})


@router.delete("/messages")
async def delete_messages(
    thread_id: str = Query(..., min_length=1, max_length=64, description="会话标识"),
):
    """清空会话历史（删除该 thread_id 的全部 checkpoints）"""
    ok_del = await personal_chief.clear_history(thread_id)
    if not ok_del:
        return schemas.err(50001, "清空会话失败，请稍后再试")
    return schemas.ok({"deleted": True, "thread_id": thread_id})
