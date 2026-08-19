# -*- coding: utf-8 -*-
"""OSS 上传签名 URL（设计报告 §4.4 / §8 api/v1/oss.py）

- POST /oss/sign：为前端签发预签名 PUT 上传 URL + 公网访问 URL
实现：阿里云 OSS QueryStringAuthentication（标准库 hmac，无第三方依赖）。
对接其他对象存储（腾讯 COS / 七牛 / MinIO）时替换 _sign_upload_url 即可。
"""
import base64
import hashlib
import hmac
import time
from urllib.parse import quote

from fastapi import APIRouter

from app.common.logger import get_logger
from app.config import (
    OSS_ACCESS_KEY_ID,
    OSS_ACCESS_KEY_SECRET,
    OSS_BUCKET,
    OSS_ENABLED,
    OSS_ENDPOINT,
    OSS_EXPIRE_SECONDS,
)
from app.models import schemas

router = APIRouter(prefix="/oss", tags=["oss"])
logger = get_logger(__name__)


def _sign_upload_url(
    filename: str, content_type: str, expires: int = 3600
) -> tuple[str, str]:
    """生成阿里云 OSS PUT 预签名 URL，返回 (upload_url, public_url)"""
    # 对象名：uploads/<时间戳>-<文件名>（保留原始 / 分隔符，空格等由 URL 构造时统一转义）
    key = f"uploads/{int(time.time())}-{filename}"
    host = f"{OSS_BUCKET}.{OSS_ENDPOINT}"                 # 例：bucket.oss-cn-hangzhou.aliyuncs.com
    # 阿里云 OSS 签名规范：CanonicalizedResource 必须为 /{bucket}/{object}；
    # URL 中对象名的 / 是路径分隔符，不能转义（quote(key, safe='/')）。
    # 已用 oss2 官方 SDK 离线对比验证：签名与 URL 完全一致。
    resource = f"/{OSS_BUCKET}/{key}"
    expiration = int(time.time()) + expires

    # StringToSign = VERB + \n + Content-MD5(空) + \n + Content-Type + \n + Expires + \n + CanonicalizedResource
    string_to_sign = f"PUT\n\n{content_type}\n{expiration}\n{resource}"
    signature = base64.b64encode(
        hmac.new(
            OSS_ACCESS_KEY_SECRET.encode(),
            string_to_sign.encode(),
            hashlib.sha1,
        ).digest()
    ).decode()

    query = (
        f"OSSAccessKeyId={quote(OSS_ACCESS_KEY_ID)}"
        f"&Expires={expiration}"
        f"&Signature={quote(signature)}"
    )
    upload_url = f"https://{host}/{quote(key, safe='/')}?{query}"
    public_url = f"https://{host}/{quote(key, safe='/')}"
    return upload_url, public_url


@router.post("/sign")
async def sign(req: schemas.OSSSignRequest):
    """签发 OSS 预签名上传 URL（前端 PUT 直传）"""
    if not OSS_ENABLED:
        return schemas.err(
            50001,
            "OSS 未配置：请在 .env 中填写 OSS_ACCESS_KEY_ID / OSS_ACCESS_KEY_SECRET / OSS_ENDPOINT / OSS_BUCKET",
        )
    upload_url, public_url = _sign_upload_url(
        req.filename, req.content_type, expires=OSS_EXPIRE_SECONDS
    )
    logger.info("签发 OSS 上传签名：%s（%d 字节）", public_url, req.size)
    return schemas.ok(
        {
            "upload_url": upload_url,
            "public_url": public_url,
            "expires_in": OSS_EXPIRE_SECONDS,
        }
    )
