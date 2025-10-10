#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ZAI格式转换器
"""

import time
import random
from typing import Dict, Any
from fastuuid import uuid4
from furl import furl
from dateutil import tz
from datetime import datetime
from browserforge.headers import HeaderGenerator

from config import settings, MODEL_MAPPING
from helpers import debug_log
from signature import SignatureGenerator, decode_jwt_payload


# 全局 HeaderGenerator 实例（单例模式）
_header_generator_instance = None


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


def get_dynamic_headers(chat_id: str = "", user_agent: str = "") -> Dict[str, str]:
    """使用BrowserForge生成动态、真实的浏览器headers
    
    Args:
        chat_id: 对话ID，用于生成Referer
        user_agent: 可选的指定User-Agent，如果提供则基于此生成headers
        
    Returns:
        完整的HTTP headers字典
    """
    header_gen = get_header_generator_instance()
    
    # 使用BrowserForge生成基础headers
    # 如果提供了user_agent，则基于它生成；否则让BrowserForge自动选择
    if user_agent:
        base_headers = header_gen.generate(user_agent=user_agent)
    else:
        base_headers = header_gen.generate()
    
    # BrowserForge生成的headers已经包含了大部分真实的浏览器headers
    # 现在我们需要覆盖或添加Z.AI特定的headers
    
    # 设置Referer
    if chat_id:
        base_headers["Referer"] = f"https://chat.z.ai/c/{chat_id}"
    else:
        base_headers["Referer"] = "https://chat.z.ai/"
    
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
    
    debug_log("BrowserForge生成headers", 
              user_agent=base_headers.get("User-Agent", "")[:50],
              has_sec_ch_ua=("sec-ch-ua" in base_headers or "Sec-Ch-Ua" in base_headers))
    
    return base_headers


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
    
    query_params = {
        "timestamp": str(timestamp),
        "requestId": request_id,
        "user_id": user_id,
        "token": token,
        "current_url": str(url),  # furl自动处理URL编码
        "pathname": pathname,
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
        """获取Z.AI认证令牌（从配置读取）"""
        token = settings.ZAI_TOKEN
        if not token:
            debug_log("❌ 未配置ZAI_TOKEN")
            raise Exception("未配置ZAI_TOKEN，请在.env文件中设置")
        
        debug_log(f"使用配置的令牌: {token[:20]}...")
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
        is_search = requested_model == settings.SEARCH_MODEL

        # 获取上游模型ID
        upstream_model_id = self.model_mapping.get(requested_model, "0727-360B-API")
        debug_log(f"  模型映射: {requested_model} -> {upstream_model_id}")

        # 处理消息列表
        debug_log(f"  开始处理 {len(request.get('messages', []))} 条消息")
        messages = self._process_messages(request.get("messages", []))

        # 构建MCP服务器列表
        mcp_servers = []
        if is_search:
            mcp_servers.append("deep-web-search")
            debug_log(f"🔍 检测到搜索模型，添加 deep-web-search MCP 服务器")
            
        # 构建上游请求体
        chat_id = generate_uuid()

        body = {
            "stream": True,
            "model": upstream_model_id,
            "messages": messages,
            "params": {},
            "features": {
                "image_generation": False,
                "web_search": is_search,
                "auto_web_search": is_search,
                "preview_mode": False,
                "flags": [],
                "features": [],
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
                # 使用dateutil提供更精确的时区处理
                "{{CURRENT_DATETIME}}": datetime.now(tz=tz.gettz("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S"),
                "{{CURRENT_DATE}}": datetime.now(tz=tz.gettz("Asia/Shanghai")).strftime("%Y-%m-%d"),
                "{{CURRENT_TIME}}": datetime.now(tz=tz.gettz("Asia/Shanghai")).strftime("%H:%M:%S"),
                "{{CURRENT_WEEKDAY}}": datetime.now(tz=tz.gettz("Asia/Shanghai")).strftime("%A"),
                "{{CURRENT_TIMEZONE}}": "Asia/Shanghai",
                "{{USER_LANGUAGE}}": "zh-CN",
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
        
        # 使用BrowserForge生成动态headers（不指定user_agent让其自动选择更真实的配置）
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
            user_content = self._extract_last_user_content(messages)
            
            # 使用SignatureGenerator生成签名
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
            "Pragma": "no-cache",
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
