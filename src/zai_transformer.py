#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ZAI格式转换器
"""

import time
import random
from typing import Dict, Any
from functools import lru_cache
from fastuuid import uuid4
from furl import furl
from dateutil import tz
from datetime import datetime
from browserforge.headers import HeaderGenerator

from .config import settings, MODEL_MAPPING
from .helpers import debug_log, perf_timer, perf_track
from .signature import SignatureGenerator, decode_jwt_payload
from .token_pool import get_token_pool


# 全局 HeaderGenerator 实例（单例模式）
_header_generator_instance = None

# 缓存的时区对象（避免重复查找）
_cached_timezone = None


@lru_cache(maxsize=8)
def get_timezone(tz_name: str = "Asia/Shanghai"):
    """
    获取时区对象（带缓存）
    
    Args:
        tz_name: 时区名称，默认 Asia/Shanghai
        
    Returns:
        时区对象
    """
    return tz.gettz(tz_name)


def generate_time_variables(timezone_name: str = "Asia/Shanghai") -> Dict[str, str]:
    """
    一次性生成所有时间相关变量（性能优化）
    
    Args:
        timezone_name: 时区名称
        
    Returns:
        包含所有时间变量的字典
    """
    # 获取缓存的时区对象
    timezone = get_timezone(timezone_name)
    
    # 一次调用 datetime.now()，避免多次调用
    now = datetime.now(tz=timezone)
    
    return {
        "{{CURRENT_DATETIME}}": now.strftime("%Y-%m-%d %H:%M:%S"),
        "{{CURRENT_DATE}}": now.strftime("%Y-%m-%d"),
        "{{CURRENT_TIME}}": now.strftime("%H:%M:%S"),
        "{{CURRENT_WEEKDAY}}": now.strftime("%A"),
        "{{CURRENT_TIMEZONE}}": timezone_name,
        "{{USER_LANGUAGE}}": "zh-CN",
    }


def get_header_generator_instance() -> HeaderGenerator:
    """获取或创建 HeaderGenerator 实例（单例模式）"""
    global _header_generator_instance
    if _header_generator_instance is None:
        # 配置HeaderGenerator：优先Chrome和Edge浏览器，Windows平台，桌面设备
        _header_generator_instance = HeaderGenerator(
            browser=('chrome', 'edge'),
            os='windows',
            device='desktop',
            locale=('zh-CN', 'en-US'),
            http_version=2
        )
    return _header_generator_instance


def generate_uuid() -> str:
    """生成UUID v4（使用fastuuid提升性能）"""
    return str(uuid4())


# Header模板缓存（减少BrowserForge调用）
_header_template_cache = None
_header_cache_lock = False


def get_header_template() -> Dict[str, str]:
    """
    获取缓存的header模板（仅在首次调用时生成）
    
    Returns:
        header模板字典
    """
    global _header_template_cache, _header_cache_lock
    
    # 简单的单例模式（无需线程锁，因为是单进程应用）
    if _header_template_cache is None and not _header_cache_lock:
        _header_cache_lock = True
        header_gen = get_header_generator_instance()
        
        # 使用BrowserForge生成基础headers（仅一次）
        base_headers = header_gen.generate()
        
        # 设置特定于Z.AI的headers
        base_headers["Origin"] = "https://chat.z.ai"
        base_headers["Content-Type"] = "application/json"
        base_headers["X-Fe-Version"] = "prod-fe-1.0.95"
        
        # 设置Fetch相关headers（用于CORS请求）
        base_headers["Sec-Fetch-Dest"] = "empty"
        base_headers["Sec-Fetch-Mode"] = "cors"
        base_headers["Sec-Fetch-Site"] = "same-origin"
        
        # 确保Accept-Encoding包含zstd（现代浏览器支持）
        if "Accept-Encoding" in base_headers:
            if "zstd" not in base_headers["Accept-Encoding"]:
                base_headers["Accept-Encoding"] = base_headers["Accept-Encoding"] + ", zstd"
        else:
            base_headers["Accept-Encoding"] = "gzip, deflate, br, zstd"
        
        # 确保Accept头适合API请求
        base_headers["Accept"] = "*/*"
        
        # 保持连接
        base_headers["Connection"] = "keep-alive"
        
        _header_template_cache = base_headers
        debug_log("✅ Header模板已缓存", 
                  user_agent=base_headers.get("User-Agent", "")[:50],
                  has_sec_ch_ua=("sec-ch-ua" in base_headers or "Sec-Ch-Ua" in base_headers))
    
    return _header_template_cache.copy()


def get_dynamic_headers(chat_id: str = "", user_agent: str = "") -> Dict[str, str]:
    """使用缓存的header模板生成headers（性能优化）
    
    Args:
        chat_id: 对话ID，用于生成Referer
        user_agent: 可选的指定User-Agent（保留接口兼容性，但不推荐使用）
        
    Returns:
        完整的HTTP headers字典
    """
    # 使用缓存的模板（避免每次调用BrowserForge）
    headers = get_header_template()
    
    # 仅更新需要变化的字段
    if chat_id:
        headers["Referer"] = f"https://chat.z.ai/c/{chat_id}"
    else:
        headers["Referer"] = "https://chat.z.ai/"
    
    # 如果指定了user_agent，覆盖模板中的User-Agent
    if user_agent:
        headers["User-Agent"] = user_agent
    
    return headers


def build_query_params(
    timestamp: int, 
    request_id: str, 
    token: str,
    user_agent: str,
    chat_id: str = "",
    user_id: str = ""
) -> Dict[str, str]:
    """构建查询参数，模拟真实的浏览器请求（使用furl优化URL处理）"""
    if not user_id:
        try:
            payload = decode_jwt_payload(token)
            user_id = payload['id']
        except Exception:
            user_id = "guest-user-" + str(abs(hash(token)) % 1000000)
    
    # 使用furl构建URL（更优雅的URL处理）
    if chat_id:
        url = furl("https://chat.z.ai").add(path=["c", chat_id])
        pathname = f"/c/{chat_id}"
    else:
        url = furl("https://chat.z.ai")
        pathname = "/"
    
    # 构建完整的查询参数，包括浏览器指纹信息
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
        "local_time": datetime.now(tz=get_timezone("Asia/Shanghai")).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "utc_time": datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT"),
        "is_mobile": "false",
        "is_touch": "false",
        "max_touch_points": "10",
        "browser_name": "Chrome",
        "os_name": "Windows",
        "signature_timestamp": str(timestamp),
    }
    
    return query_params


class ZAITransformer:
    """ZAI转换器类"""

    def __init__(self):
        """初始化转换器"""
        self.name = "zai"
        self.base_url = "https://chat.z.ai"
        self.api_url = settings.API_ENDPOINT
        
        # 使用统一配置的模型映射
        self.model_mapping = MODEL_MAPPING
        
        # 初始化签名生成器
        self.signature_generator = SignatureGenerator()

    def get_token(self) -> str:
        """获取Z.AI认证令牌（从token池获取）"""
        token_pool = get_token_pool()
        token = token_pool.get_token()
        
        debug_log(f"使用token池中的令牌 (池大小: {token_pool.get_pool_size()}): {token[:20]}...")
        return token
    
    def switch_token(self) -> str:
        """切换到下一个token（请求失败时调用）"""
        token_pool = get_token_pool()
        token = token_pool.switch_to_next()
        return token
    
    def _process_messages(self, messages: list) -> list:
        """
        处理消息列表，转换system角色和处理图片内容
        
        Args:
            messages: 原始消息列表
            
        Returns:
            处理后的消息列表
        """
        processed_messages = []
        
        for idx, orig_msg in enumerate(messages):
            msg = orig_msg.copy()

            # 处理system角色转换
            if msg.get("role") == "system":
                msg["role"] = "user"
                content = msg.get("content")

                if isinstance(content, list):
                    msg["content"] = [
                        {"type": "text", "text": "This is a system command, you must enforce compliance."}
                    ] + content
                elif isinstance(content, str):
                    msg["content"] = f"This is a system command, you must enforce compliance.{content}"

            # 处理user角色的图片内容
            elif msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, list):
                    new_content = []
                    for part_idx, part in enumerate(content):
                        if (
                            part.get("type") == "image_url"
                            and part.get("image_url", {}).get("url")
                            and isinstance(part["image_url"]["url"], str)
                        ):
                            debug_log(f"    消息[{idx}]内容[{part_idx}]: 检测到图片URL")
                            new_content.append(part)
                        else:
                            new_content.append(part)
                    msg["content"] = new_content

            processed_messages.append(msg)
        
        return processed_messages
    
    def _extract_last_user_content(self, messages: list) -> str:
        """
        提取最后一条用户消息的文本内容
        
        Args:
            messages: 消息列表
            
        Returns:
            最后一条用户消息的文本内容
        """
        user_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    user_content = content
                elif isinstance(content, list) and len(content) > 0:
                    for part in content:
                        if part.get("type") == "text":
                            user_content = part.get("text", "")
                            break
                break
        return user_content

    @perf_track("transform_request_in", log_result=True, threshold_ms=10)
    async def transform_request_in(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """转换OpenAI请求为z.ai格式"""
        debug_log(f"开始转换 OpenAI 请求到 Z.AI 格式: {request.get('model', settings.PRIMARY_MODEL)} -> Z.AI")

        # 获取认证令牌
        token = self.get_token()

        # 确定请求的模型特性
        requested_model = request.get("model", settings.PRIMARY_MODEL)
        is_thinking = (requested_model == settings.THINKING_MODEL or 
                      requested_model == settings.GLM_46_THINKING_MODEL or 
                      request.get("reasoning", False))
        is_search = (requested_model == settings.SEARCH_MODEL or 
                    requested_model == settings.GLM_46_SEARCH_MODEL)

        # 获取上游模型ID
        upstream_model_id = self.model_mapping.get(requested_model, "0727-360B-API")
        debug_log(f"  模型映射: {requested_model} -> {upstream_model_id}")

        # 处理消息列表
        debug_log(f"  开始处理 {len(request.get('messages', []))} 条消息")
        with perf_timer("process_messages", threshold_ms=5):
            messages = self._process_messages(request.get("messages", []))

        # 构建MCP服务器列表
        mcp_servers = []
        if is_search:
            mcp_servers.append("deep-web-search")
            debug_log(f"🔍 检测到搜索模型，添加 deep-web-search MCP 服务器")
        
        # 构建隐藏的MCP服务器特性列表
        hidden_mcp_features = [
            {"type": "mcp", "server": "vibe-coding", "status": "hidden"},
            {"type": "mcp", "server": "ppt-maker", "status": "hidden"},
            {"type": "mcp", "server": "image-search", "status": "hidden"},
            {"type": "mcp", "server": "deep-research", "status": "hidden"}
        ]
            
        # 构建上游请求体
        chat_id = generate_uuid()

        body = {
            "stream": True,
            "model": upstream_model_id,
            "messages": messages,
            "params": {},
            "features": {
                "image_generation": False,
                "web_search": False,  # 注意：通过mcp_servers控制搜索，而不是这个标志
                "auto_web_search": False,
                "preview_mode": True,  # 修改为True
                "flags": [],
                "features": hidden_mcp_features,
                "enable_thinking": is_thinking,
            },
            "background_tasks": {
                "title_generation": False,
                "tags_generation": False,
            },
            "mcp_servers": mcp_servers,
            "variables": {
                "{{USER_NAME}}": "Guest",
                "{{USER_LOCATION}}": "Unknown",
                # 使用优化后的时间变量生成函数（一次调用，避免重复）
                **generate_time_variables("Asia/Shanghai"),
            },
            "model_item": {
                "id": upstream_model_id,
                "name": requested_model,
                "owned_by": "z.ai"
            },
            "chat_id": chat_id,
            "id": generate_uuid(),
        }

        # 生成时间戳和请求ID
        timestamp = int(time.time() * 1000)
        request_id = generate_uuid()
        
        # 使用缓存的header模板生成headers（性能优化）
        with perf_timer("generate_headers", threshold_ms=5):
            dynamic_headers = get_dynamic_headers(chat_id)
        
        # 从生成的headers中提取User-Agent
        user_agent = dynamic_headers.get("User-Agent", "")
        
        # 构建查询参数
        user_id = ""
        try:
            payload = decode_jwt_payload(token)
            user_id = payload['id']
        except Exception as e:
            debug_log(f"解码JWT token获取user_id失败: {e}")
            user_id = "guest-user-" + str(abs(hash(token)) % 1000000)
        
        query_params = build_query_params(timestamp, request_id, token, user_agent, chat_id, user_id)
        
        # 生成Z.AI签名
        try:
            # 提取最后一条用户消息内容
            with perf_timer("extract_user_content", threshold_ms=5):
                user_content = self._extract_last_user_content(messages)
            
            # 使用SignatureGenerator生成签名
            with perf_timer("generate_signature", threshold_ms=10):
                signature_result = self.signature_generator.generate(token, request_id, timestamp, user_content)
                signature = signature_result["signature"]
            
            # 添加签名到headers
            dynamic_headers["X-Signature"] = signature
            query_params["signature_timestamp"] = str(timestamp)
            
            debug_log("  Z.AI签名已生成并添加到请求中")
        except Exception as e:
            debug_log(f"生成Z.AI签名失败: {e}")
        
        # 构建完整的URL
        url_with_params = f"{self.api_url}?" + "&".join([f"{k}={v}" for k, v in query_params.items()])

        headers = {
            **dynamic_headers,
            "Authorization": f"Bearer {token}",
            "Cache-Control": "no-cache",
            # "Pragma": "no-cache",
        }

        config = {
            "url": url_with_params,
            "headers": headers,
        }

        debug_log("请求转换完成")

        return {
            "body": body,
            "config": config,
            "token": token
        }
