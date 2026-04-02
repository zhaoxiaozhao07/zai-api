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
from urllib.parse import urlsplit
import httpx
from fastuuid import uuid4
from dateutil import tz
from datetime import datetime
from fastapi import HTTPException

from .config import settings
from .helpers import error_log, info_log, debug_log, perf_timer, perf_track, get_timezone
from .signature import SignatureGenerator, decode_jwt_payload
from .token_pool import get_token_pool
from .image_handler import (
    process_image_content,
    fetch_image_bytes_and_meta,
    compute_image_asset_key,
    minimal_file_data_from_uploaded,
    build_reusable_file_payload,
)
from .header_manager import header_manager
from .message_processor import message_processor
from .conversation_state import conversation_state

PUBLIC_GLM_46V_MODEL = "glm-4.6v"
PUBLIC_GLM_5V_TURBO_MODEL = "glm-5v-turbo"
PUBLIC_GLM_5_MODEL = "glm-5"
PUBLIC_GLM_5_TURBO_MODEL = "glm-5-turbo"
PUBLIC_GLM_47_MODEL = "glm-4.7"
LEGACY_GLM_46V_ALIASES = frozenset({"GLM-4.6V"})
LEGACY_GLM_5V_TURBO_ALIASES = frozenset({"GLM-5v-Turbo"})
LEGACY_GLM_5_ALIASES = frozenset({"GLM-5", "GLM-5-Think"})
LEGACY_GLM_5_TURBO_ALIASES = frozenset({"GLM-5-Turbo"})



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

        # 初始化签名生成器
        self.signature_generator = SignatureGenerator()

    async def get_token(
        self,
        http_client=None,
        preferred_token: Optional[str] = None,
        rotate_for_new_session: bool = False,
    ) -> str:
        """
        获取Z.AI认证令牌（从token池获取）

        Args:
            http_client: 外部传入的HTTP客户端（用于匿名Token获取）

        Returns:
            str: 可用的Token
        """
        token_pool = await get_token_pool()
        if preferred_token:
            token = preferred_token
        elif rotate_for_new_session:
            token = await token_pool.get_token_for_new_session()
        else:
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

    def _resolve_base_url(self, upstream_url: Optional[str] = None) -> str:
        """从 completions 地址解析出站点根地址。"""
        if upstream_url:
            target_url = upstream_url
        elif settings.API_ENDPOINT:
            target_url = settings.API_ENDPOINT
        else:
            target_url = f"{self.base_url}/api/v2/chat/completions"
        parsed = urlsplit(target_url)
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}"
        return self.base_url

    def _extract_user_name(self, token: str) -> str:
        """从 JWT 中提取展示用户名，失败时返回 Guest。"""
        payload = decode_jwt_payload(token) if token else {}

        name = payload.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()

        email = payload.get("email")
        if isinstance(email, str) and email.strip():
            return email.split("@", 1)[0].strip() or "Guest"

        return "Guest"

    async def _create_upstream_chat(
        self,
        client,
        token: str,
        model: str,
        enable_thinking: bool,
        user_message: str,
        upstream_url: Optional[str] = None,
    ) -> str:
        """在调用 chat/completions 前先创建服务端 chat。"""
        if client is None:
            raise HTTPException(status_code=500, detail="缺少 HTTP 客户端，无法创建上游 chat")

        init_content = (user_message or "").strip() or "..."
        if len(init_content) > 500:
            init_content = init_content[:500] + "..."

        message_id = generate_uuid()
        timestamp_s = int(time.time())
        body = {
            "chat": {
                "id": "",
                "title": "新聊天",
                "models": [model],
                "params": {},
                "history": {
                    "messages": {
                        message_id: {
                            "id": message_id,
                            "parentId": None,
                            "childrenIds": [],
                            "role": "user",
                            "content": init_content,
                            "timestamp": timestamp_s,
                            "models": [model],
                        }
                    },
                    "currentId": message_id,
                },
                "tags": [],
                "flags": [],
                "features": [
                    {
                        "type": "tool_selector",
                        "server": "tool_selector_h",
                        "status": "hidden",
                    }
                ],
                "mcp_servers": [],
                "enable_thinking": enable_thinking,
                "auto_web_search": False,
                "message_version": 1,
                "extra": {},
                "timestamp": int(time.time() * 1000),
            }
        }

        dynamic_headers = await header_manager.get_dynamic_headers()
        headers = {
            **dynamic_headers,
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        }

        create_chat_url = f"{self._resolve_base_url(upstream_url)}/api/v1/chats/new"
        response = None
        last_exception: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                response = await client.post(create_chat_url, headers=headers, json=body)
                break
            except httpx.HTTPError as exc:
                last_exception = exc
                if attempt >= 3:
                    raise HTTPException(status_code=502, detail=f"Create upstream chat failed: {exc.__class__.__name__}") from exc
                await asyncio.sleep(min(0.5 * attempt, 1.5))

        if response is None:
            raise HTTPException(status_code=502, detail=f"Create upstream chat failed: {type(last_exception).__name__ if last_exception else 'unknown'}")

        if response.status_code != 200:
            error_body = response.text[:500]
            error_log(
                "创建上游 chat 失败",
                status_code=response.status_code,
                error_detail=error_body,
            )
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Create upstream chat failed: {error_body}",
            )

        chat_data = response.json()
        chat_id = chat_data.get("id") or chat_data.get("chat", {}).get("id")
        if not isinstance(chat_id, str) or not chat_id.strip():
            raise HTTPException(status_code=502, detail="上游 chat 创建成功但未返回 chat_id")

        return chat_id

    def _is_glm_46v_model(self, model: Any) -> bool:
        if not isinstance(model, str):
            return False
        return model in {
            PUBLIC_GLM_46V_MODEL,
            settings.GLM_46V_MODEL,
            *LEGACY_GLM_46V_ALIASES,
        }

    def _is_glm_5v_turbo_model(self, model: Any) -> bool:
        if not isinstance(model, str):
            return False
        return model in {
            PUBLIC_GLM_5V_TURBO_MODEL,
            settings.GLM_5V_TURBO_MODEL,
            *LEGACY_GLM_5V_TURBO_ALIASES,
        }

    def _is_glm_5_family_model(self, model: Any) -> bool:
        if not isinstance(model, str):
            return False
        return model in {
            PUBLIC_GLM_5_MODEL,
            PUBLIC_GLM_5_TURBO_MODEL,
            settings.GLM_5_MODEL,
            settings.GLM_5_THINKING_MODEL,
            settings.GLM_5_TURBO_MODEL,
            *LEGACY_GLM_5_ALIASES,
            *LEGACY_GLM_5_TURBO_ALIASES,
        }

    def _is_glm_47_model(self, model: Any) -> bool:
        return isinstance(model, str) and model == PUBLIC_GLM_47_MODEL

    def _normalize_requested_model(self, model: Any) -> str:
        if not isinstance(model, str) or not model.strip():
            return PUBLIC_GLM_5_MODEL

        normalized_model = model.strip()
        if self._is_glm_46v_model(normalized_model):
            return PUBLIC_GLM_46V_MODEL
        if self._is_glm_5v_turbo_model(normalized_model):
            return PUBLIC_GLM_5V_TURBO_MODEL
        if normalized_model in {
            PUBLIC_GLM_5_TURBO_MODEL,
            settings.GLM_5_TURBO_MODEL,
            *LEGACY_GLM_5_TURBO_ALIASES,
        }:
            return PUBLIC_GLM_5_TURBO_MODEL
        if self._is_glm_5_family_model(normalized_model):
            return PUBLIC_GLM_5_MODEL
        if self._is_glm_47_model(normalized_model):
            return PUBLIC_GLM_47_MODEL
        return normalized_model

    def _resolve_glm_5_thinking(self, request: Dict[str, Any]) -> bool:
        if "thinking" in request:
            thinking = request.get("thinking")
            return isinstance(thinking, dict) and thinking.get("type") == "enabled"

        if "reasoning_effort" in request:
            reasoning_effort = request.get("reasoning_effort")
            if isinstance(reasoning_effort, str):
                return reasoning_effort.lower() != "none"
            return bool(reasoning_effort)

        if "enable_thinking" in request:
            return request.get("enable_thinking") is True

        original_model = request.get("_original_model")
        if original_model in {
            settings.GLM_5_THINKING_MODEL,
            settings.GLM_5_TURBO_MODEL,
            "GLM-5-Think",
            "GLM-5-Turbo",
            PUBLIC_GLM_5_TURBO_MODEL,
        }:
            return True

        normalized_model = self._normalize_requested_model(request.get("model"))
        if normalized_model == PUBLIC_GLM_5_TURBO_MODEL:
            return True

        return False

    def _is_glm_5v_turbo_request(self, request: Dict[str, Any]) -> bool:
        original_model = request.get("_original_model")
        if self._is_glm_5v_turbo_model(original_model):
            return True

        normalized_model = self._normalize_requested_model(request.get("model"))
        return normalized_model == PUBLIC_GLM_5V_TURBO_MODEL

    def _extract_image_refs(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        refs: List[Dict[str, Any]] = []
        for msg_index, msg in enumerate(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for part_index, part in enumerate(content):
                if not isinstance(part, dict) or part.get("type") != "image_url":
                    continue
                image_url = part.get("image_url", {}).get("url") if isinstance(part.get("image_url"), dict) else None
                if not isinstance(image_url, str) or not image_url.strip():
                    continue
                refs.append(
                    {
                        "message_index": msg_index,
                        "part_index": part_index,
                        "url": image_url,
                    }
                )
        return refs

    async def _resolve_vision_assets(
        self,
        *,
        client: httpx.AsyncClient,
        token: str,
        messages: List[Dict[str, Any]],
        cached_session,
        current_user_message_id: str,
        include_latest_message_images: bool,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], List[str]]:
        refs = self._extract_image_refs(messages)
        if not refs and not cached_session:
            return [], {}, []

        session_asset_keys = list(getattr(cached_session, "vision_asset_keys", []) or [])
        existing_assets: Dict[str, Dict[str, Any]] = {}
        if cached_session and cached_session.chat_id:
            for asset in conversation_state.get_session_assets(cached_session.chat_id, session_asset_keys):
                asset_key = asset.get("asset_key")
                if isinstance(asset_key, str) and asset_key:
                    existing_assets[asset_key] = asset

        assets_by_key: Dict[str, Dict[str, Any]] = dict(existing_assets)
        latest_message_lookup = {
            ref["url"]
            for ref in refs
            if ref.get("message_index") == len(messages) - 1
        }

        for ref in refs:
            image_url = ref["url"]
            image_bytes, content_type, _, filename = await fetch_image_bytes_and_meta(image_url, client)
            asset_key = compute_image_asset_key(image_bytes)
            ref["asset_key"] = asset_key

            existing_asset = assets_by_key.get(asset_key)
            if existing_asset:
                existing_asset["last_used_at"] = time.time()
                continue

            if cached_session and cached_session.chat_id:
                uploaded = await process_image_content(image_url, token, client)
            else:
                uploaded = await process_image_content(image_url, token, client)
            if not uploaded:
                continue

            uploaded_file = uploaded.get("file", {})
            assets_by_key[asset_key] = {
                "asset_key": asset_key,
                "file_id": uploaded.get("id") or uploaded_file.get("id") or "",
                "name": uploaded.get("name") or uploaded_file.get("filename") or filename,
                "size": uploaded.get("size") or uploaded_file.get("meta", {}).get("size", 0),
                "file_data": minimal_file_data_from_uploaded(uploaded_file),
                "created_at": time.time(),
                "last_used_at": time.time(),
                "content_type": uploaded_file.get("content_type") or content_type,
            }

        ordered_asset_keys: List[str] = []
        for asset_key in session_asset_keys:
            if asset_key in assets_by_key and asset_key not in ordered_asset_keys:
                ordered_asset_keys.append(asset_key)
        for ref in refs:
            asset_key = ref.get("asset_key")
            if isinstance(asset_key, str) and asset_key and asset_key not in ordered_asset_keys:
                ordered_asset_keys.append(asset_key)

        reusable_assets: List[Dict[str, Any]] = []
        latest_image_asset_keys: List[str] = []
        for asset_key in ordered_asset_keys:
            asset = assets_by_key.get(asset_key)
            if not asset:
                continue
            payload = build_reusable_file_payload(asset, ref_user_msg_id=current_user_message_id)
            reusable_assets.append(payload)
            if include_latest_message_images:
                latest_image_asset_keys.append(asset_key)
            else:
                if asset_key in latest_message_lookup:
                    latest_image_asset_keys.append(asset_key)

        return reusable_assets, assets_by_key, latest_image_asset_keys

    def _merge_text_and_image_assets_into_latest_message(
        self,
        messages: List[Dict[str, Any]],
        image_asset_keys: List[str],
        assets_by_key: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not messages:
            return messages

        merged_messages = list(messages)
        latest_message = dict(merged_messages[-1])
        latest_content = latest_message.get("content")

        text_parts: List[Dict[str, Any]] = []
        if isinstance(latest_content, list):
            for part in latest_content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part)
        elif isinstance(latest_content, str) and latest_content:
            text_parts.append({"type": "text", "text": latest_content})

        image_parts: List[Dict[str, Any]] = []
        for asset_key in image_asset_keys:
            asset = assets_by_key.get(asset_key)
            if not asset:
                continue
            file_id = asset.get("file_id")
            if not isinstance(file_id, str) or not file_id:
                continue
            image_parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": file_id,
                    },
                }
            )

        merged_content: List[Dict[str, Any]] = []
        merged_content.extend(text_parts)
        merged_content.extend(image_parts)
        latest_message["content"] = merged_content if merged_content else latest_message.get("content", "")
        merged_messages[-1] = latest_message
        return merged_messages

    @perf_track("transform_request_in", log_result=True, threshold_ms=10)
    async def transform_request_in(self, request: Dict[str, Any], client=None, upstream_url: Optional[str] = None) -> Dict[str, Any]:
        """
        转换OpenAI请求为z.ai格式

        Args:
            request: OpenAI格式的请求
            client: HTTP客户端（用于图像上传和Token获取）
            upstream_url: 上游API地址（如果为None则使用默认配置）

        Returns:
            转换后的请求字典
        """
        info_log(f"开始转换 OpenAI 请求到 Z.AI 格式: {request.get('model', PUBLIC_GLM_5_MODEL)} -> Z.AI")

        requested_model = self._normalize_requested_model(request.get("model", PUBLIC_GLM_5_MODEL))
        original_messages = request.get("messages", [])
        history_key, cached_session = conversation_state.resolve_session(requested_model, original_messages)
        has_prior_history = len(original_messages) > 1

        if cached_session:
            token = await self.get_token(http_client=client, preferred_token=cached_session.token)
            parent_request_id = cached_session.last_request_id
            chat_id = cached_session.chat_id
            source_messages = original_messages[-1:] if original_messages else []
            info_log(
                "[CONVERSATION] 命中会话缓存，复用上游 chat_id",
                chat_id=chat_id,
                history_key=history_key,
            )
        else:
            token = await self.get_token(http_client=client, rotate_for_new_session=True)
            parent_request_id = None
            chat_id = None
            if has_prior_history:
                bootstrap_content = conversation_state.build_bootstrap_content(original_messages)
                source_messages = [{"role": "user", "content": bootstrap_content}]
                info_log(
                    "[CONVERSATION] 未命中会话缓存，使用历史bootstrap创建新会话",
                    history_key=history_key,
                    history_messages=len(original_messages) - 1,
                )
            else:
                source_messages = original_messages

        messages = source_messages

        is_vision_model = requested_model in {PUBLIC_GLM_46V_MODEL, PUBLIC_GLM_5V_TURBO_MODEL}
        is_glm_5v_turbo_model = requested_model == PUBLIC_GLM_5V_TURBO_MODEL
        is_glm_5_model = requested_model == PUBLIC_GLM_5_MODEL
        is_glm_5_turbo_model = requested_model == PUBLIC_GLM_5_TURBO_MODEL
        is_glm_47_model = requested_model == PUBLIC_GLM_47_MODEL
        is_thinking = is_vision_model
        if is_glm_5_model or is_glm_5_turbo_model or is_glm_47_model:
            is_thinking = self._resolve_glm_5_thinking(request)
        is_simplified_model = is_glm_5_model or is_glm_5_turbo_model or is_glm_47_model or is_glm_5v_turbo_model

        if is_glm_5v_turbo_model:
            upstream_model_id = "GLM-5v-Turbo"
        elif is_vision_model:
            upstream_model_id = PUBLIC_GLM_46V_MODEL
        elif is_glm_47_model:
            upstream_model_id = PUBLIC_GLM_47_MODEL
        elif is_glm_5_turbo_model:
            upstream_model_id = "GLM-5-Turbo"
        elif is_glm_5_model:
            upstream_model_id = PUBLIC_GLM_5_MODEL
        else:
            upstream_model_id = PUBLIC_GLM_5_MODEL
        debug_log(f"  模型映射: {requested_model} -> {upstream_model_id}")

        debug_log(f"开始处理 {len(messages)} 条消息")
        with perf_timer("process_messages", threshold_ms=5):
            messages, image_urls = message_processor.process_messages(messages, is_vision_model=is_vision_model)

        mcp_servers = []
        if is_vision_model:
            mcp_servers.extend(["vlm-image-search", "vlm-image-recognition", "vlm-image-processing"])
            debug_log(f"检测到视觉模型 {requested_model}，添加 VLM MCP 服务器")

        current_user_message_id = generate_uuid() if (is_vision_model or is_simplified_model) else None

        files_list = []
        vision_assets_for_store: List[Dict[str, Any]] = []

        if is_vision_model and client:
            include_latest_message_images = not cached_session and has_prior_history
            vision_source_messages = original_messages if (not cached_session and has_prior_history) else source_messages
            replace_all_user_message_images = not cached_session and has_prior_history
            vision_files, vision_assets_map, latest_image_asset_keys = await self._resolve_vision_assets(
                client=client,
                token=token,
                messages=vision_source_messages,
                cached_session=cached_session,
                current_user_message_id=current_user_message_id or generate_uuid(),
                include_latest_message_images=include_latest_message_images,
            )
            files_list.extend(vision_files)
            vision_assets_for_store = list(vision_assets_map.values())

            replace_asset_keys = latest_image_asset_keys
            if not replace_asset_keys and len(original_messages) == 1 and len(vision_files) == 1:
                single_file_id = vision_files[0].get("id")
                if isinstance(single_file_id, str) and single_file_id.strip():
                    fallback_asset_key = next(iter(vision_assets_map.keys()), None)
                    if isinstance(fallback_asset_key, str) and fallback_asset_key:
                        replace_asset_keys = [fallback_asset_key]

            stripped_messages = message_processor.replace_image_urls_with_file_ids(
                messages,
                replace_asset_keys,
                vision_assets_map,
                latest_message_only=not replace_all_user_message_images,
            )
            if stripped_messages is not None:
                messages = stripped_messages
            elif replace_asset_keys:
                messages = self._merge_text_and_image_assets_into_latest_message(messages, replace_asset_keys, vision_assets_map)
        elif image_urls and client:
            info_log(f"检测到 {len(image_urls)} 张图像，开始上传")
            for idx, image_url in enumerate(image_urls):
                try:
                    file_obj = await process_image_content(image_url, token, client)
                    if file_obj:
                        if not is_vision_model:
                            if is_simplified_model and current_user_message_id:
                                file_obj["ref_user_msg_id"] = current_user_message_id
                            files_list.append(file_obj)
                        info_log(f"图像 [{idx+1}/{len(image_urls)}] 上传成功", file_id=file_obj.get("id"))
                    else:
                        error_log(f"图像 [{idx+1}/{len(image_urls)}] 上传失败")
                except Exception as e:
                    error_log(f"图像 [{idx+1}/{len(image_urls)}] 处理错误: {e}")
        elif image_urls:
            info_log(f"检测到 {len(image_urls)} 张图像，但未提供HTTP客户端，跳过上传")

        with perf_timer("extract_user_content", threshold_ms=5):
            user_content = message_processor.extract_last_user_content(messages)

        if not chat_id:
            chat_id = await self._create_upstream_chat(
                client=client,
                token=token,
                model=upstream_model_id,
                enable_thinking=is_thinking,
                user_message=user_content,
                upstream_url=upstream_url,
            )

        if is_vision_model:
            vision_mode_flags = {
                "vlm_tools_enable": True,
                "vlm_web_search_enable": False,
                "vlm_website_mode": True,
            }
            features = {
                "image_generation": False,
                "web_search": False,
                "auto_web_search": False,
                "preview_mode": True,
                "flags": [],
                "enable_thinking": True,
                **vision_mode_flags,
            }
        else:
            vision_mode_flags = {}
            features = {
                "image_generation": False,
                "web_search": False,
                "auto_web_search": False,
                "preview_mode": True,
                "flags": [],
                "enable_thinking": is_thinking,
            }

        background_tasks = {
            "title_generation": True,
            "tags_generation": True,
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
                "{{USER_NAME}}": self._extract_user_name(token),
                "{{USER_LOCATION}}": "Unknown",
                **generate_time_variables("Asia/Shanghai"),
            },
            "chat_id": chat_id,
            "id": generate_uuid(),
            "extra": dict(vision_mode_flags),
        }

        body["current_user_message_id"] = current_user_message_id or generate_uuid()
        body["current_user_message_parent_id"] = parent_request_id
        
        if files_list:
            body["files"] = files_list
            debug_log(f"添加 {len(files_list)} 个文件到请求body")

        timestamp = int(time.time() * 1000)
        request_id = generate_uuid()
        body["signature_prompt"] = user_content

        with perf_timer("generate_headers", threshold_ms=5):
            dynamic_headers = await header_manager.get_dynamic_headers(chat_id)
        
        user_agent = dynamic_headers.get("User-Agent", "")
        
        user_id = ""
        try:
            payload = decode_jwt_payload(token)
            user_id = payload['id']
        except Exception as e:
            debug_log(f"解码JWT token获取user_id失败: {e}")
            user_id = "guest-user-" + str(abs(hash(token)) % 1000000)
        
        query_params = header_manager.build_query_params(timestamp, request_id, token, user_agent, chat_id, user_id)
        
        try:
            with perf_timer("generate_signature", threshold_ms=10):
                signature_result = self.signature_generator.generate(token, request_id, timestamp, user_content)
                signature = signature_result["signature"]

            dynamic_headers["X-Signature"] = signature
            query_params["signature_timestamp"] = str(timestamp)

            debug_log("Z.AI签名已生成并添加到请求中")
        except Exception as e:
            error_log(f"生成Z.AI签名失败: {e}")

        api_url = upstream_url if upstream_url else settings.API_ENDPOINT
        debug_log(f"使用上游地址: {api_url}")

        url_with_params = f"{api_url}?" + "&".join([f"{k}={v}" for k, v in query_params.items()])

        headers = {
            **dynamic_headers,
            "Authorization": f"Bearer {token}",
            "Cache-Control": "no-cache",
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
            "is_thinking": is_thinking,
            "is_vision_model": is_vision_model,
            "conversation": {
                "history_key": history_key,
                "cache_hit": cached_session is not None,
                "vision_assets": vision_assets_for_store,
            },
        }
