#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Z.AI API 签名功能
"""

import os
import time
import hmac
import hashlib
from typing import Dict, Any, Optional
from jose import jwt
from jose.exceptions import JWTError
from helpers import debug_log


def decode_jwt_payload(token: str) -> Dict[str, Any]:
    """
    使用python-jose库解码JWT token的payload部分（带完整异常处理）
    
    python-jose提供更专业的JWT处理，支持更多加密算法和标准
    
    Args:
        token: JWT token字符串
        
    Returns:
        解码后的payload字典，失败时返回空字典
    """
    try:
        if not token:
            return {}
        
        # 使用python-jose库解码JWT，不验证签名（因为我们只需要读取payload）
        # jose库的decode不需要密钥即可解码（当options禁用验证时）
        payload = jwt.decode(
            token,
            key="",  # 空密钥，因为不验证签名
            options={
                "verify_signature": False,
                "verify_exp": False,
                "verify_nbf": False,
                "verify_iat": False,
                "verify_aud": False
            }
        )
        return payload
    except JWTError as e:
        debug_log(f"JWT解码错误: {e}")
        return {}
    except Exception as e:
        debug_log(f"解码JWT payload失败: {e}")
        return {}


def extract_user_id_from_token(token: str) -> str:
    """
    从JWT token中提取user_id（尝试多个常见字段）
    
    Args:
        token: JWT token字符串
        
    Returns:
        提取到的user_id，失败时返回"guest"
    """
    payload = decode_jwt_payload(token) if token else {}
    
    # 尝试多个可能的user_id字段
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
    secret: str = None
) -> str:
    """
    生成Z.AI API双层HMAC-SHA256签名
    
    Layer1: derived_key = HMAC(secret, window_index)
    Layer2: signature = HMAC(derived_key, canonical_string)
    canonical_string = "requestId,<id>,timestamp,<ts>,user_id,<uid>|<msg>|<ts>"
    
    Args:
        message_text: 最近一次user content
        request_id: 请求ID
        timestamp_ms: 时间戳（毫秒）
        user_id: 用户ID
        secret: 签名密钥，默认从环境变量ZAI_SIGNING_SECRET读取，否则使用"junjie"
        
    Returns:
        签名字符串
    """
    # 处理空值
    t = message_text or ""
    r = str(timestamp_ms)
    e = f"requestId,{request_id},timestamp,{timestamp_ms},user_id,{user_id}"
    i = f"{e}|{t}|{r}"
    
    # 时间窗口索引（5分钟为一个窗口）
    window_index = timestamp_ms // (5 * 60 * 1000)
    
    # 获取secret密钥
    if secret is None:
        secret = os.getenv("ZAI_SIGNING_SECRET", "junjie") or "junjie"
    root_key = (secret or "junjie").encode("utf-8")
    
    # 双层HMAC签名
    # Layer1: 派生密钥
    derived_hex = hmac.new(root_key, str(window_index).encode("utf-8"), hashlib.sha256).hexdigest()
    # Layer2: 最终签名
    signature = hmac.new(derived_hex.encode("utf-8"), i.encode("utf-8"), hashlib.sha256).hexdigest()
    
    return signature


def zs(e: str, t: str, timestamp: int, secret: str = None) -> Dict[str, str]:
    """
    生成Z.AI API签名（兼容旧版接口）
    
    Args:
        e: 签名字符串，格式为 "requestId,{requestId},timestamp,{timestamp},user_id,{user_id}"
        t: 最近一次user content
        timestamp: 时间戳（毫秒）
        secret: 签名密钥，默认从环境变量读取
        
    Returns:
        包含签名和时间戳的字典
    """
    # 处理空值
    t = t or ""
    r = str(timestamp)
    i = f"{e}|{t}|{r}"
    
    # 时间窗口索引
    n = timestamp // (5 * 60 * 1000)
    
    # 获取secret密钥
    if secret is None:
        secret = os.getenv("ZAI_SIGNING_SECRET", "junjie") or "junjie"
    key = (secret or "junjie").encode('utf-8')
    
    # 双层HMAC签名
    o = hmac.new(key, str(n).encode('utf-8'), hashlib.sha256).hexdigest()
    signature = hmac.new(o.encode('utf-8'), i.encode('utf-8'), hashlib.sha256).hexdigest()

    return {
        "signature": signature,
        "timestamp": timestamp
    }


def generate_zs_signature(
    token: str, 
    request_id: str, 
    timestamp: int, 
    user_content: str, 
    secret: str = None
) -> Dict[str, str]:
    """
    生成Z.AI API签名的便捷函数（推荐使用）
    
    Args:
        token: JWT token
        request_id: 请求ID
        timestamp: 时间戳（毫秒）
        user_content: 最近一次user content
        secret: 签名密钥，默认从环境变量读取
        
    Returns:
        包含签名和时间戳的字典
    """
    # 使用增强的user_id提取逻辑
    user_id = extract_user_id_from_token(token)
    
    # 构建签名字符串
    e = f"requestId,{request_id},timestamp,{timestamp},user_id,{user_id}"
    
    # 调用zs函数生成签名
    return zs(e, user_content or "", timestamp, secret)


class SignatureGenerator:
    """
    Z.AI API签名生成器类
    
    封装签名生成逻辑，便于复用和测试
    """
    
    def __init__(self, secret: Optional[str] = None):
        """
        初始化签名生成器
        
        Args:
            secret: 签名密钥，如果不提供则从环境变量ZAI_SIGNING_SECRET读取，默认为"junjie"
        """
        self.secret = secret or os.getenv("ZAI_SIGNING_SECRET", "junjie") or "junjie"
    
    def decode_jwt_payload(self, token: str) -> Dict[str, Any]:
        """
        解码JWT token的payload部分
        
        Args:
            token: JWT token字符串
            
        Returns:
            解码后的payload字典，失败时返回空字典
        """
        return decode_jwt_payload(token)
    
    def extract_user_id(self, token: str) -> str:
        """
        从JWT token中提取user_id
        
        Args:
            token: JWT token字符串
            
        Returns:
            提取到的user_id，失败时返回"guest"
        """
        return extract_user_id_from_token(token)
    
    def generate_signature(
        self, 
        message_text: str, 
        request_id: str, 
        timestamp_ms: int, 
        user_id: str
    ) -> str:
        """
        生成Z.AI API双层HMAC-SHA256签名
        
        Args:
            message_text: 最近一次user content
            request_id: 请求ID
            timestamp_ms: 时间戳（毫秒）
            user_id: 用户ID
            
        Returns:
            签名字符串
        """
        return generate_signature(message_text, request_id, timestamp_ms, user_id, self.secret)
    
    def generate(
        self, 
        token: str, 
        request_id: str, 
        timestamp: int, 
        user_content: str
    ) -> Dict[str, str]:
        """
        生成Z.AI API签名（主要方法）
        
        Args:
            token: JWT token
            request_id: 请求ID
            timestamp: 时间戳（毫秒）
            user_content: 最近一次user content
            
        Returns:
            包含signature和timestamp的字典
        """
        # 提取user_id
        user_id = self.extract_user_id(token)
        
        # 构建签名字符串
        e = f"requestId,{request_id},timestamp,{timestamp},user_id,{user_id}"
        
        # 调用zs函数生成签名
        return zs(e, user_content or "", timestamp, self.secret)
    
    def generate_with_user_id(
        self,
        user_id: str,
        request_id: str,
        timestamp: int,
        user_content: str
    ) -> Dict[str, str]:
        """
        使用已知的user_id生成签名
        
        Args:
            user_id: 用户ID
            request_id: 请求ID
            timestamp: 时间戳（毫秒）
            user_content: 最近一次user content
            
        Returns:
            包含signature和timestamp的字典
        """
        e = f"requestId,{request_id},timestamp,{timestamp},user_id,{user_id}"
        return zs(e, user_content or "", timestamp, self.secret)

