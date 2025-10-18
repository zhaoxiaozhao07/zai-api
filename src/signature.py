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
from functools import lru_cache
from jose import jwt
from jose.exceptions import JWTError
from .helpers import debug_log, perf_track


@lru_cache(maxsize=128)
@perf_track("jwt_decode", log_result=True, threshold_ms=5)
def decode_jwt_payload(token: str) -> Dict[str, Any]:
    """
    使用python-jose库解码JWT token的payload部分（带完整异常处理）
    
    python-jose提供更专业的JWT处理，支持更多加密算法和标准
    
    注意：此函数使用LRU缓存，可以缓存最近128个token的解码结果，显著提升性能
    
    Args:
        token: JWT token字符串
        
    Returns:
        解码后的payload字典，失败时返回空字典（注意：返回的是元组包装的字典以支持缓存）
    """
    try:
        if not token:
            return _make_cacheable_dict({})
        
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
        # 转换为可缓存的格式（使用frozendict或元组）
        return _make_cacheable_dict(payload)
    except JWTError as e:
        debug_log(f"JWT解码错误: {e}")
        return _make_cacheable_dict({})
    except Exception as e:
        debug_log(f"解码JWT payload失败: {e}")
        return _make_cacheable_dict({})


def _make_cacheable_dict(d: dict) -> Dict[str, Any]:
    """
    将字典转换为可缓存的格式（实际上dict本身在lru_cache中可以返回）
    
    Args:
        d: 原始字典
        
    Returns:
        原始字典（dict对象作为返回值是可以的）
    """
    return d


@lru_cache(maxsize=128)
def extract_user_id_from_token(token: str) -> str:
    """
    从JWT token中提取user_id（尝试多个常见字段）
    
    注意：此函数使用LRU缓存，避免重复解码同一token
    
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
    
    ⚠️ 2025-10-17 更新：基于浏览器断点调试分析的最新算法
    
    算法流程（从DARET26p.js逆向分析）：
    1. 使用TextEncoder编码消息为UTF-8字节
    2. 使用btoa进行Base64编码：btoa(String.fromCharCode(...bytes))
    3. 构建第一部分：a = "requestId,<id>,timestamp,<ts>,user_id,<uid>"
    4. 构建canonical string：g = a + "|" + base64(message) + "|" + timestamp
    5. 计算时间窗口：S = Math.floor(timestamp / (5 * 60 * 1000))
    6. 第一层HMAC：v = hmac_sha256(secret, S)  -> 返回hex字符串
    7. 第二层HMAC：signature = hmac_sha256(v_hex_as_utf8, g)
    
    注意：第二层HMAC使用第一层的hex字符串（作为UTF-8字符串）作为密钥
    
    Args:
        message_text: 最近一次user content
        request_id: 请求ID
        timestamp_ms: 时间戳（毫秒）
        user_id: 用户ID
        secret: 签名密钥，默认从环境变量ZAI_SIGNING_SECRET读取
        
    Returns:
        签名字符串
    """
    import base64
    
    # 1. 处理消息文本
    message = message_text or ""
    
    # 2. Base64编码消息（模拟JavaScript的btoa）
    message_bytes = message.encode("utf-8")
    message_base64 = base64.b64encode(message_bytes).decode("utf-8")
    
    # 3. 构建第一部分
    a = f"requestId,{request_id},timestamp,{timestamp_ms},user_id,{user_id}"
    
    # 4. 构建canonical string
    canonical_string = f"{a}|{message_base64}|{timestamp_ms}"
    
    # 5. 时间窗口索引（5分钟为一个窗口）
    window_index = timestamp_ms // (5 * 60 * 1000)
    
    # 6. 获取secret密钥
    # ✅ 2025-10-18 密钥更新：
    # 通过暴力搜索所有W0数组索引，成功找到当前有效的签名密钥！
    # 新密钥索引: 635 (在W0数组中的实际位置: 635-222=413)
    # 密钥长度: 31 bytes
    # 密钥 (hex): 6b65792d40404040292929282928283929292d787878782626262525252525
    # 验证状态: ✅ 已通过两组测试数据验证
    #
    # 历史记录：
    # - 旧密钥（已失效）: c4de923f8fd2eb92 (索引 309)
    # - 候选密钥（未验证）: 531:1f9eb337101b0ef3c20e08, 323:a87d52a8f0e91bc0b078
    #
    # 用户可通过环境变量 ZAI_SIGNING_SECRET 覆盖默认密钥
    if secret is None:
        secret_env = os.getenv("ZAI_SIGNING_SECRET")
        if secret_env:
            # 如果环境变量提供了hex格式的密钥，尝试解码
            try:
                # 支持任意长度的hex密钥
                if all(c in '0123456789abcdefABCDEF' for c in secret_env):
                    root_key = bytes.fromhex(secret_env)
                else:
                    root_key = secret_env.encode("utf-8")
            except:
                root_key = secret_env.encode("utf-8")
        else:
            # 使用最新验证的密钥（2025-10-18 通过暴力搜索找到）
            # 索引: 635, 长度: 31 bytes
            # 用户可通过环境变量 ZAI_SIGNING_SECRET 设置正确的密钥
            root_key = bytes.fromhex("6b65792d40404040292929282928283929292d787878782626262525252525")
    else:
        # 用户提供的密钥，尝试判断是hex还是字符串
        if isinstance(secret, bytes):
            root_key = secret
        elif len(secret) % 2 == 0 and all(c in '0123456789abcdefABCDEF' for c in secret):
            # 如果是偶数长度且全是hex字符，认为是hex格式
            root_key = bytes.fromhex(secret)
        else:
            root_key = secret.encode("utf-8")
    
    # 7. 第一层HMAC：生成派生密钥（返回hex字符串）
    derived_hex = hmac.new(root_key, str(window_index).encode("utf-8"), hashlib.sha256).hexdigest()
    
    # 8. 第二层HMAC：使用hex字符串作为UTF-8密钥
    # 注意：derived_hex是一个hex字符串，我们将其作为UTF-8字符串使用
    signature = hmac.new(derived_hex.encode("utf-8"), canonical_string.encode("utf-8"), hashlib.sha256).hexdigest()
    
    return signature


def zs(e: str, t: str, timestamp: int, secret: str = None) -> Dict[str, str]:
    """
    生成Z.AI API签名（兼容旧版接口）
    
    ⚠️ 2025-10-17 更新：基于浏览器断点调试分析
    
    Args:
        e: 签名字符串，格式为 "requestId,{requestId},timestamp,{timestamp},user_id,{user_id}"
        t: 最近一次user content
        timestamp: 时间戳（毫秒）
        secret: 签名密钥，默认从环境变量读取
        
    Returns:
        包含签名和时间戳的字典
    """
    import base64
    
    # 1. 处理消息文本
    message = t or ""
    
    # 2. Base64编码消息
    message_bytes = message.encode("utf-8")
    message_base64 = base64.b64encode(message_bytes).decode("utf-8")
    
    # 3. 构建canonical string
    canonical_string = f"{e}|{message_base64}|{timestamp}"
    
    # 4. 时间窗口索引
    window_index = timestamp // (5 * 60 * 1000)
    
    # 5. 获取secret密钥
    # ✅ 2025-10-18 密钥更新：
    # 通过暴力搜索所有W0数组索引，成功找到当前有效的签名密钥！
    # 新密钥索引: 635 (在W0数组中的实际位置: 635-222=413)
    # 密钥长度: 31 bytes
    # 密钥 (hex): 6b65792d40404040292929282928283929292d787878782626262525252525
    # 验证状态: ✅ 已通过两组测试数据验证
    #
    # 历史记录：
    # - 旧密钥（已失效）: c4de923f8fd2eb92 (索引 309)
    # - 候选密钥（未验证）: 531:1f9eb337101b0ef3c20e08, 323:a87d52a8f0e91bc0b078
    #
    # 用户可通过环境变量 ZAI_SIGNING_SECRET 覆盖默认密钥
    if secret is None:
        secret_env = os.getenv("ZAI_SIGNING_SECRET")
        if secret_env:
            # 如果环境变量提供了hex格式的密钥，尝试解码
            try:
                # 支持任意长度的hex密钥
                if all(c in '0123456789abcdefABCDEF' for c in secret_env):
                    root_key = bytes.fromhex(secret_env)
                else:
                    root_key = secret_env.encode("utf-8")
            except:
                root_key = secret_env.encode("utf-8")
        else:
            # 使用最新验证的密钥（2025-10-18 通过暴力搜索找到）
            # 索引: 635, 长度: 31 bytes
            # 用户可通过环境变量 ZAI_SIGNING_SECRET 设置正确的密钥
            root_key = bytes.fromhex("6b65792d40404040292929282928283929292d787878782626262525252525")
    else:
        # 用户提供的密钥，尝试判断是hex还是字符串
        if isinstance(secret, bytes):
            root_key = secret
        elif len(secret) % 2 == 0 and all(c in '0123456789abcdefABCDEF' for c in secret):
            # 如果是偶数长度且全是hex字符，认为是hex格式
            root_key = bytes.fromhex(secret)
        else:
            root_key = secret.encode("utf-8")
    
    # 6. 双层HMAC签名
    # Layer1: 派生密钥（返回hex字符串）
    derived_hex = hmac.new(root_key, str(window_index).encode('utf-8'), hashlib.sha256).hexdigest()
    # Layer2: 最终签名（使用hex字符串作为UTF-8密钥）
    signature = hmac.new(derived_hex.encode('utf-8'), canonical_string.encode('utf-8'), hashlib.sha256).hexdigest()

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

