#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ZAI格式转换器
"""

import asyncio
import time
import random
from typing import Dict, Any, Tuple, List, Optional
from functools import lru_cache
from fastuuid import uuid4
from dateutil import tz
from datetime import datetime
from fastapi import HTTPException

from .config import settings, MODEL_MAPPING
from .helpers import error_log, info_log, debug_log, perf_timer, perf_track
from .signature import SignatureGenerator, decode_jwt_payload
from .token_pool import get_token_pool
from .image_handler import process_image_content
from .header_manager import header_manager
from .message_processor import message_processor


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


def generate_uuid() -> str:
    """生成UUID v4（使用fastuuid提升性能）"""
    return str(uuid4())


class ZAITransformer:
    """ZAI转换器类"""

    def __init__(self):
        """初始化转换器"""
        self.name = "zai"
        self.base_url = "https://chat.z.ai"
        # 不再在初始化时固定api_url，而是在transform_request_in中动态获取
        # self.api_url = settings.API_ENDPOINT

        # 使用统一配置的模型映射
        self.model_mapping = MODEL_MAPPING
        
        # 初始化签名生成器
        self.signature_generator = SignatureGenerator()

    async def get_token(self, http_client=None) -> str:
        """
        获取Z.AI认证令牌（从token池获取）

        Args:
            http_client: 外部传入的HTTP客户端（用于匿名Token获取）

        Returns:
            str: 可用的Token
        """
        token_pool = await get_token_pool()
        token = await token_pool.get_token(http_client=http_client)

        token_preview = f"{token[:20]}...{token[-20:]}" if len(token) > 40 else token
        debug_log(f"使用token池中的令牌 (池大小: {token_pool.get_pool_size()}): {token_preview}")
        return token
    
    async def switch_token(self, http_client=None) -> str:
        """
        切换到下一个token（请求失败时调用）

        Args:
            http_client: 保留参数兼容性

        Returns:
            str: 下一个Token

        Raises:
            ValueError: 所有配置Token都不可用时抛出
        """
        token_pool = await get_token_pool()
        token = await token_pool.switch_to_next()

        # 如果返回 None，说明所有配置 Token 都不可用
        if token is None:
            raise ValueError("[ERROR] 所有配置Token不可用")

        return token
    
    async def refresh_header_template(self):
        """刷新header模板（清除缓存并重新生成）"""
        await header_manager.clear_header_template()
        info_log("🔄 Header模板已刷新，下次请求将使用新的header")
    
    def _has_image_content(self, messages: List[Dict]) -> bool:
        """
        检测消息中是否包含图像
        
        Args:
            messages: 消息列表
            
        Returns:
            bool: 如果消息中包含图像内容则返回True
        """
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, list):
                    for part in content:
                        if (part.get("type") == "image_url" and
                            part.get("image_url", {}).get("url")):
                            return True
        return False
    
    def _process_messages(self, messages: list, is_vision_model: bool = False) -> Tuple[list, list]:
        """
        处理消息列表，转换system角色和提取图片内容
        
        Args:
            messages: 原始消息列表
            is_vision_model: 是否是视觉模型（GLM-4.5V/GLM-4.6V），视觉模型保留图片在messages中
            
        Returns:
            (处理后的消息列表, 图片URL列表)
        """
        processed_messages = []
        image_urls = []
        
        for idx, orig_msg in enumerate(messages):
            # 惰性拷贝：仅在需要修改时才拷贝
            msg = orig_msg
            needs_copy = False

            # 处理system角色转换
            if orig_msg.get("role") == "system":
                msg = orig_msg.copy()
                msg["role"] = "user"
                content = msg.get("content")

                if isinstance(content, list):
                    msg["content"] = [
                        {"type": "text", "text": "This is a system command, you must enforce compliance."}
                    ] + content
                elif isinstance(content, str):
                    msg["content"] = f"This is a system command, you must enforce compliance.{content}"

            # 处理user角色的图片内容
            elif orig_msg.get("role") == "user":
                content = orig_msg.get("content")
                if isinstance(content, list):
                    new_content = []
                    has_changes = False
                    for part_idx, part in enumerate(content):
                        if (
                            part.get("type") == "image_url"
                            and part.get("image_url", {}).get("url")
                            and isinstance(part["image_url"]["url"], str)
                        ):
                            image_url = part["image_url"]["url"]
                            debug_log(f"    消息[{idx}]内容[{part_idx}]: 检测到图片URL")
                            image_urls.append(image_url)
                            has_changes = True
                            
                            # 视觉模型：保留图片在消息中，但会在上传后修改URL格式
                            if is_vision_model:
                                new_content.append(part)
                            # 非视觉模型：移除图片内容，只保留文本
                        elif part.get("type") == "text":
                            new_content.append(part)
                    
                    # 仅在有改动时才拷贝并更新消息
                    if has_changes:
                        msg = orig_msg.copy()
                        # 如果new_content只有文本，提取为字符串
                        if len(new_content) == 1 and new_content[0].get("type") == "text":
                            msg["content"] = new_content[0].get("text", "")
                        elif new_content:
                            msg["content"] = new_content
                        else:
                            msg["content"] = ""

            processed_messages.append(msg)
        
        return processed_messages, image_urls
    
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
    async def transform_request_in(self, request: Dict[str, Any], client=None, upstream_url: str = None) -> Dict[str, Any]:
        """
        转换OpenAI请求为z.ai格式

        Args:
            request: OpenAI格式的请求
            client: HTTP客户端（用于图像上传和Token获取）
            upstream_url: 上游API地址（如果为None则使用默认配置）

        Returns:
            转换后的请求字典
        """
        info_log(f"开始转换 OpenAI 请求到 Z.AI 格式: {request.get('model', settings.PRIMARY_MODEL)} -> Z.AI")

        # 获取认证令牌
        token = await self.get_token(http_client=client)

        messages = request.get("messages", [])

        # 确定请求的模型特性
        requested_model = request.get("model", settings.PRIMARY_MODEL)
        is_thinking = (requested_model == settings.THINKING_MODEL or
                      requested_model == settings.GLM_46_THINKING_MODEL or
                      requested_model == settings.GLM_47_THINKING_MODEL or  # 只有 GLM-4.7-Thinking 启用 thinking
                      requested_model == settings.GLM_5_THINKING_MODEL or  # GLM-5-Think 启用 thinking
                      requested_model == settings.GLM_45V_MODEL or  # glm4.5v 视觉模型也是 thinking 模型
                      requested_model == settings.GLM_46V_MODEL or  # glm4.6v 视觉模型也是 thinking 模型
                      request.get("reasoning", False))
        is_vision_model = (requested_model == settings.GLM_45V_MODEL or
                          requested_model == settings.GLM_46V_MODEL)
        is_simplified_model = (requested_model == settings.GLM_47_MODEL or
                               requested_model == settings.GLM_47_THINKING_MODEL or
                               requested_model == settings.GLM_5_MODEL or
                               requested_model == settings.GLM_5_THINKING_MODEL)

        # 获取上游模型ID
        upstream_model_id = self.model_mapping.get(requested_model, "0727-360B-API")
        debug_log(f"  模型映射: {requested_model} -> {upstream_model_id}")

        # 处理消息列表并提取图像
        debug_log(f"  开始处理 {len(request.get('messages', []))} 条消息")
        with perf_timer("process_messages", threshold_ms=5):
            messages, image_urls = self._process_messages(request.get("messages", []), is_vision_model=is_vision_model)

        # 构建MCP服务器列表
        mcp_servers = []
        # GLM-4.6V 添加 VLM 专有服务器（支持图片搜索、识别、处理）
        if requested_model == settings.GLM_46V_MODEL:
            mcp_servers.extend(["vlm-image-search", "vlm-image-recognition", "vlm-image-processing"])
            debug_log(f"🔍 检测到 GLM-4.6V 模型，添加 VLM MCP 服务器")
        
        # 构建隐藏的MCP服务器特性列表
        hidden_mcp_features = [
            {"type": "mcp", "server": "vibe-coding", "status": "hidden"},
            {"type": "mcp", "server": "ppt-maker", "status": "hidden"},
            {"type": "mcp", "server": "image-search", "status": "hidden"},
            {"type": "mcp", "server": "deep-research", "status": "hidden"}
        ]

        # 处理图像上传
        files_list = []
        uploaded_files_map = {}  # 用于视觉模型(GLM-4.5V/GLM-4.6V)：原始URL -> 文件信息的映射
        
        if image_urls and client:
            info_log(f"检测到 {len(image_urls)} 张图像，开始上传")
            for idx, image_url in enumerate(image_urls):
                try:
                    file_obj = await process_image_content(image_url, token, client)
                    if file_obj:
                        # 非视觉模型：添加到files列表
                        if not is_vision_model:
                            files_list.append(file_obj)
                        else:
                            # 视觉模型：保存映射关系，稍后修改messages中的URL
                            uploaded_files_map[image_url] = file_obj
                        info_log(f"图像 [{idx+1}/{len(image_urls)}] 上传成功", file_id=file_obj.get("id"))
                    else:
                        error_log(f"图像 [{idx+1}/{len(image_urls)}] 上传失败")
                except Exception as e:
                    error_log(f"图像 [{idx+1}/{len(image_urls)}] 处理错误: {e}")
        elif image_urls:
            info_log(f"检测到 {len(image_urls)} 张图像，但未提供HTTP客户端，跳过上传")
        
        # GLM-4.5V特殊处理：修改messages中的图片URL格式
        # 生成当前用户消息ID（用于关联files）
        current_user_message_id = generate_uuid() if is_vision_model else None
        
        if is_vision_model and uploaded_files_map:
            info_log(f"[Vision] 开始修改消息中的图片URL格式")
            for msg in messages:
                if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                    for part in msg["content"]:
                        if part.get("type") == "image_url":
                            original_url = part.get("image_url", {}).get("url")
                            if original_url in uploaded_files_map:
                                file_info = uploaded_files_map[original_url]
                                # 提取file信息
                                file_data = file_info.get("file", {})
                                file_id = file_data.get("id", "")
                                # GLM-4.5V格式的URL只需要file_id
                                part["image_url"]["url"] = file_id
                                debug_log(f"[GLM-4.5V] 图片URL已转换", 
                                         original=original_url[:50], 
                                         new=file_id)
                                
                                # 添加ref_user_msg_id到文件信息（用于files数组）
                                file_info["ref_user_msg_id"] = current_user_message_id
                                files_list.append(file_info)
            
        # 构建上游请求体
        chat_id = generate_uuid()

        # GLM-4.5V 对 features/background_tasks 的要求与常规模型略有不同，
        # 尽量对齐实际抓包格式，避免上游返回 "Oops" 错误。
        if is_vision_model:
            features = {
                "image_generation": False,
                "web_search": False,
                "auto_web_search": False,
                "preview_mode": True,
                "flags": [],
                "enable_thinking": True,
            }
            background_tasks = {
                "title_generation": True,
                "tags_generation": True,
            }
        elif is_simplified_model:
            # GLM-4.7 / GLM-5 系列采用简化格式，enable_thinking 根据具体模型决定
            simplified_enable_thinking = (requested_model == settings.GLM_47_THINKING_MODEL or
                                          requested_model == settings.GLM_5_THINKING_MODEL)
            features = {
                "image_generation": False,
                "web_search": False,
                "auto_web_search": False,
                "preview_mode": True,
                "flags": [],
                "enable_thinking": simplified_enable_thinking,
            }
            background_tasks = {
                "title_generation": True,
                "tags_generation": True,
            }
        else:
            features = {
                "image_generation": False,
                "web_search": False,
                "auto_web_search": False,
                "preview_mode": True,
                "flags": [],
                "features": hidden_mcp_features,
                "enable_thinking": is_thinking,
            }
            background_tasks = {
                "title_generation": False,
                "tags_generation": False,
            }

        body = {
            "stream": True,
            "model": upstream_model_id,
            "messages": messages,
            "params": {},
            "features": features,
            "background_tasks": background_tasks,
            "mcp_servers": mcp_servers,
            "variables": {
                "{{USER_NAME}}": "Guest",
                "{{USER_LOCATION}}": "Unknown",
                # 使用优化后的时间变量生成函数（一次调用，避免重复）
                **generate_time_variables("Asia/Shanghai"),
            },
            "chat_id": chat_id,
            "id": generate_uuid(),
        }

        # 与抓包保持一致：GLM-4.5V/4.6V/4.7 需要 current_user_message_id/parent_id，且不上送 model_item
        if is_vision_model:
            body["current_user_message_id"] = current_user_message_id
            body["current_user_message_parent_id"] = None
        elif is_simplified_model:
            # GLM-4.7 / GLM-5 使用简化格式：添加 current_user_message_id/parent_id 但不添加 model_item
            body["current_user_message_id"] = generate_uuid()
            body["current_user_message_parent_id"] = None
        else:
            body["model_item"] = {
                "id": upstream_model_id,
                "name": requested_model,
                "owned_by": "openai"
            }
        
        # 如果有上传的文件，添加到body中
        if files_list:
            body["files"] = files_list
            debug_log(f"添加 {len(files_list)} 个文件到请求body")
        
        # GLM-4.5V需要添加current_user_message_id字段
        if is_vision_model and current_user_message_id:
            body["current_user_message_id"] = current_user_message_id
            debug_log(f"[GLM-4.5V] 添加current_user_message_id: {current_user_message_id}")

        # 生成时间戳和请求ID
        timestamp = int(time.time() * 1000)
        request_id = generate_uuid()
        
        # 提取最后一条用户消息内容（用于签名和请求体）
        with perf_timer("extract_user_content", threshold_ms=5):
            user_content = message_processor.extract_last_user_content(messages)
        
        # 将用户内容添加到请求体中（新要求）
        body["signature_prompt"] = user_content
        
        # 使用缓存的header模板生成headers（性能优化）
        with perf_timer("generate_headers", threshold_ms=5):
            dynamic_headers = await header_manager.get_dynamic_headers(chat_id)
        
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
        
        query_params = header_manager.build_query_params(timestamp, request_id, token, user_agent, chat_id, user_id)
        
        # 生成Z.AI签名
        try:
            # 使用SignatureGenerator生成签名
            with perf_timer("generate_signature", threshold_ms=10):
                signature_result = self.signature_generator.generate(token, request_id, timestamp, user_content)
                signature = signature_result["signature"]

            # 添加签名到headers
            dynamic_headers["X-Signature"] = signature
            query_params["signature_timestamp"] = str(timestamp)

            debug_log("  Z.AI签名已生成并添加到请求中")
        except Exception as e:
            error_log(f"生成Z.AI签名失败: {e}")

        # 使用传入的上游URL或默认配置
        api_url = upstream_url if upstream_url else settings.API_ENDPOINT
        debug_log(f"  使用上游地址: {api_url}")

        # 构建完整的URL
        url_with_params = f"{api_url}?" + "&".join([f"{k}={v}" for k, v in query_params.items()])

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

        info_log("请求转换完成")

        return {
            "body": body,
            "config": config,
            "token": token,
            "is_thinking": is_thinking,  # 标记是否为thinking模型，响应处理时用于判断是否输出reasoning_content
            "is_vision_model": is_vision_model,  # 标记是否为V系列视觉模型（4.5v/4.6v），用于区分多阶段思考格式
        }
