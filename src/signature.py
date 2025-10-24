#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Z.AI API 签名功能
"""

import os
import hmac
import hashlib
from typing import Dict, Any, Optional
from functools import lru_cache
from jose import jwt
from jose.exceptions import JWTError
from .helpers import debug_log, perf_track


@lru_cache(maxsize=128)
@perf_track("jwt_decode", log_result=True, threshold_ms=5)
def decode_jwt_payload(token: str) -> Dict[str, Any]:
    """解码 JWT payload，出现异常时返回空字典。"""
    try:
        if not token:
            return {}
        payload = jwt.decode(
            token,
            key="",
            options={
                "verify_signature": False,
                "verify_exp": False,
                "verify_nbf": False,
                "verify_iat": False,
                "verify_aud": False,
            },
        )
        return payload
    except JWTError as e:
        debug_log(f"JWT解码错误: {e}")
        return {}
    except Exception as e:
        debug_log(f"解码JWT payload失败: {e}")
        return {}


@lru_cache(maxsize=128)
def extract_user_id_from_token(token: str) -> str:
    """从 JWT token 中提取用户 ID，失败时返回 guest。"""
    payload = decode_jwt_payload(token) if token else {}
    for key in ("id", "user_id", "uid", "sub"):
        val = payload.get(key)
        if isinstance(val, (str, int)) and str(val):
            return str(val)
    return "guest"


def generate_signature(
    message_text: str,
    request_id: str,
    timestamp_ms: int,
    user_id: str,
    secret: str | None = None,
) -> str:
    """生成 Z.AI API 双层 HMAC-SHA256 签名。"""
    import base64

    message_bytes = (message_text or "").encode("utf-8")
    message_base64 = base64.b64encode(message_bytes).decode("utf-8")

    canonical_prefix = f"requestId,{request_id},timestamp,{timestamp_ms},user_id,{user_id}"
    canonical_string = f"{canonical_prefix}|{message_base64}|{timestamp_ms}"
    window_index = timestamp_ms // (5 * 60 * 1000)

    if secret is None:
        secret_env = os.getenv("ZAI_SIGNING_SECRET")
        if not secret_env:
            raise ValueError("签名密钥未配置，请设置环境变量 ZAI_SIGNING_SECRET")
        if all(c in "0123456789abcdefABCDEF" for c in secret_env):
            root_key = bytes.fromhex(secret_env)
        else:
            root_key = secret_env.encode("utf-8")
    else:
        if isinstance(secret, bytes):
            root_key = secret
        elif len(secret) % 2 == 0 and all(c in "0123456789abcdefABCDEF" for c in secret):
            root_key = bytes.fromhex(secret)
        else:
            root_key = secret.encode("utf-8")

    derived_hex = hmac.new(root_key, str(window_index).encode("utf-8"), hashlib.sha256).hexdigest()
    signature = hmac.new(derived_hex.encode("utf-8"), canonical_string.encode("utf-8"), hashlib.sha256).hexdigest()
    return signature


class SignatureGenerator:
    """签名生成器封装。"""

    def __init__(self, secret: Optional[str] = None):
        if secret is None:
            secret = os.getenv("ZAI_SIGNING_SECRET")
            if not secret:
                raise ValueError("签名密钥未配置，请设置 ZAI_SIGNING_SECRET")
        self.secret = secret

    def extract_user_id(self, token: str) -> str:
        return extract_user_id_from_token(token)

    def generate_signature(
        self,
        message_text: str,
        request_id: str,
        timestamp_ms: int,
        user_id: str,
    ) -> str:
        return generate_signature(message_text, request_id, timestamp_ms, user_id, self.secret)

    def generate(
        self,
        token: str,
        request_id: str,
        timestamp: int,
        user_content: str,
    ) -> Dict[str, str]:
        user_id = self.extract_user_id(token)
        signature = self.generate_signature(user_content, request_id, timestamp, user_id)
        return {"signature": signature, "timestamp": timestamp}