#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
请求头管理器模块 - 封装所有 HTTP 请求头相关逻辑

从 zai_transformer.py 拆分出来，提供线程安全的 header 模板缓存机制
"""

import asyncio
from datetime import datetime
from typing import Dict, Optional

from browserforge.headers import HeaderGenerator
from furl import furl

from .config import settings
from .helpers import info_log
from .signature import decode_jwt_payload


class HeaderManager:
    """
    请求头管理器（单例模式）
    
    封装所有请求头相关逻辑，包括：
    - Header 模板生成与缓存
    - 动态 Headers 生成
    - 查询参数构建
    """
    
    _instance: Optional['HeaderManager'] = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._header_template_cache: Optional[Dict[str, str]] = None
        self._cache_lock = asyncio.Lock()
        self._header_generator: Optional[HeaderGenerator] = None
        self._cached_timezone = None
        self._initialized = True
    
    def _get_header_generator(self) -> HeaderGenerator:
        """获取或创建 HeaderGenerator 实例"""
        if self._header_generator is None:
            self._header_generator = HeaderGenerator(
                browser=['chrome'],
                os=['windows'],
                device=['desktop']
            )
        return self._header_generator
    
    def _get_timezone(self, tz_name: str = "Asia/Shanghai"):
        """获取时区对象（带缓存）"""
        if self._cached_timezone is None:
            try:
                # 使用 dateutil（与原有代码保持一致）
                from dateutil import tz as dateutil_tz
                self._cached_timezone = dateutil_tz.gettz(tz_name)
            except ImportError:
                try:
                    import zoneinfo
                    self._cached_timezone = zoneinfo.ZoneInfo(tz_name)
                except Exception:
                    import pytz
                    self._cached_timezone = pytz.timezone(tz_name)
        return self._cached_timezone
    
    async def get_header_template(self) -> Dict[str, str]:
        """
        获取缓存的 header 模板（仅在首次调用时生成，线程安全）
        
        Returns:
            header 模板字典
        """
        # 快速路径：如果已经缓存了，直接返回
        if self._header_template_cache is not None:
            return self._header_template_cache.copy()
        
        # 使用异步锁保护缓存初始化
        async with self._cache_lock:
            # 双重检查
            if self._header_template_cache is not None:
                return self._header_template_cache.copy()
            
            header_gen = self._get_header_generator()
            
            # 使用 BrowserForge 生成基础 headers（仅一次）
            base_headers = header_gen.generate()
            
            # 设置特定于 Z.AI 的 headers
            base_headers["Origin"] = "https://chat.z.ai"
            base_headers["Content-Type"] = "application/json"
            base_headers["X-Fe-Version"] = settings.ZAI_FE_VERSION
            
            # 设置 Fetch 相关 headers（用于 CORS 请求）
            base_headers["Sec-Fetch-Dest"] = "empty"
            base_headers["Sec-Fetch-Mode"] = "cors"
            base_headers["Sec-Fetch-Site"] = "same-origin"
            
            # 确保 Accept-Encoding 包含 zstd
            if "Accept-Encoding" in base_headers:
                if "zstd" not in base_headers["Accept-Encoding"]:
                    base_headers["Accept-Encoding"] = base_headers["Accept-Encoding"] + ", zstd"
            else:
                base_headers["Accept-Encoding"] = "gzip, deflate, br, zstd"
            
            # 确保 Accept 头适合 API 请求
            base_headers["Accept"] = "*/*"
            
            # 保持连接
            base_headers["Connection"] = "keep-alive"
            
            self._header_template_cache = base_headers
            info_log("✅ Header模板已缓存", 
                     user_agent=base_headers.get("User-Agent", "")[:50],
                     has_sec_ch_ua=("sec-ch-ua" in base_headers or "Sec-Ch-Ua" in base_headers))
        
        return self._header_template_cache.copy()
    
    async def clear_header_template(self):
        """清除缓存的 header 模板，强制下次调用时重新生成（线程安全）"""
        async with self._cache_lock:
            self._header_template_cache = None
            info_log("🔄 Header模板缓存已清除")
    
    async def get_dynamic_headers(self, chat_id: str = "", user_agent: str = "") -> Dict[str, str]:
        """
        使用缓存的 header 模板生成 headers（性能优化，线程安全）
        
        Args:
            chat_id: 对话 ID，用于生成 Referer
            user_agent: 可选的指定 User-Agent
            
        Returns:
            完整的 HTTP headers 字典
        """
        # 使用缓存的模板（避免每次调用 BrowserForge）
        headers = await self.get_header_template()
        
        # 仅更新需要变化的字段
        if chat_id:
            headers["Referer"] = f"https://chat.z.ai/c/{chat_id}"
        else:
            headers["Referer"] = "https://chat.z.ai/"
        
        # 如果指定了 user_agent，覆盖模板中的 User-Agent
        if user_agent:
            headers["User-Agent"] = user_agent
        
        return headers
    
    def build_query_params(
        self,
        timestamp: int, 
        request_id: str, 
        token: str,
        user_agent: str,
        chat_id: str = "",
        user_id: str = ""
    ) -> Dict[str, str]:
        """构建查询参数，模拟真实的浏览器请求"""
        if not user_id:
            try:
                payload = decode_jwt_payload(token)
                user_id = payload['id']
            except Exception:
                user_id = "guest-user-" + str(abs(hash(token)) % 1000000)
        
        # 使用 furl 构建 URL
        if chat_id:
            url = furl("https://chat.z.ai").add(path=["c", chat_id])
            pathname = f"/c/{chat_id}"
        else:
            url = furl("https://chat.z.ai")
            pathname = "/"
        
        tz = self._get_timezone("Asia/Shanghai")
        
        # 构建完整的查询参数
        query_params = {
            "timestamp": str(timestamp),
            "requestId": request_id,
            "user_id": user_id,
            "version": "0.0.1",
            "platform": "web",
            "token": token,
            "user_agent": user_agent,
            "language": "zh-CN",
            "languages": "zh-CN,zh",
            "timezone": "Asia/Shanghai",
            "cookie_enabled": "true",
            "screen_width": "2048",
            "screen_height": "1152",
            "screen_resolution": "2048x1152",
            "viewport_height": "654",
            "viewport_width": "1038",
            "viewport_size": "1038x654",
            "color_depth": "24",
            "pixel_ratio": "1.25",
            "current_url": str(url),
            "pathname": pathname,
            "search": "",
            "hash": "",
            "host": "chat.z.ai",
            "hostname": "chat.z.ai",
            "protocol": "https:",
            "referrer": "",
            "title": "Z.ai Chat - Free AI powered by GLM-4.6 & GLM-4.5",
            "timezone_offset": "-480",
            "local_time": datetime.now(tz=tz).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "utc_time": datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT"),
            "is_mobile": "false",
            "is_touch": "false",
            "max_touch_points": "10",
            "browser_name": "Chrome",
            "os_name": "Windows",
            "signature_timestamp": str(timestamp),
        }
        
        return query_params


# 全局单例实例
header_manager = HeaderManager()
