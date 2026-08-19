# -*- coding: utf-8 -*-
"""Pydantic 数据模型 + 统一响应包装（设计报告 §4 / §8 models/schemas.py）"""
from typing import Any, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """POST /chat/stream 请求体（设计报告 §4.1）"""

    thread_id: str = Field(..., max_length=64, description="会话标识（空串由业务层返回 40001）")
    message: str = Field(..., max_length=2000, description="用户消息（空串由业务层返回 40001）")
    image_url: Optional[str] = Field(
        None, description="图片公网 URL（可选，多模态主脑直接读图）"
    )


class OSSSignRequest(BaseModel):
    """POST /oss/sign 请求体（设计报告 §4.4）"""

    filename: str = Field(..., description="文件名，如 dish_xxx.jpg")
    content_type: str = Field("image/jpeg", description="文件 MIME 类型")
    size: int = Field(0, ge=0, description="文件大小（字节）")


def ok(data: Any = None) -> dict:
    """成功响应：{"code": 0, "message": "ok", "data": ...}"""
    return {"code": 0, "message": "ok", "data": data}


def err(code: int, message: str) -> dict:
    """失败响应：{"code": <错误码>, "message": ..., "data": None}"""
    return {"code": code, "message": message, "data": None}
