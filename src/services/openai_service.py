"""Service layer orchestrating OpenAI-compatible chat completions."""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import AsyncIterator, Dict, Optional, Tuple

import httpx
from fastapi import HTTPException

from ..helpers import (
    info_log,
    debug_log,
    error_log,
    bind_request_context,
    reset_request_context,
    request_stage_log,
    v_series_debug_log,
)
from ..schemas import OpenAIRequest
from ..config import settings
from ..zai_transformer import ZAITransformer
from ..token_pool import get_token_pool
from ..conversation_state import conversation_state
from .network_manager import network_manager
from .response_parser import response_parser
from .chunk_builder import chunk_builder


class ChatCompletionService:
    """Encapsulate chat completion workflow independent of FastAPI layer."""

    def __init__(self) -> None:
        self.transformer = ZAITransformer()
        self.parser = response_parser
        self.chunk = chunk_builder

    def _normalize_request_model(self, request_dict: dict) -> dict:
        """统一对外模型名，保留旧名称兼容。"""
        model = request_dict.get("model")
        glm_46v_aliases = {
            settings.GLM_46V_MODEL,
            "GLM-4.6V",
            "glm-4.6v",
        }
        glm_5v_turbo_aliases = {
            settings.GLM_5V_TURBO_MODEL,
            "GLM-5v-Turbo",
            "glm-5v-turbo",
        }
        glm_5_aliases = {
            settings.GLM_5_MODEL,
            settings.GLM_5_THINKING_MODEL,
            "glm-5",
            "GLM-5",
            "GLM-5-Think",
        }
        glm_5_turbo_aliases = {
            settings.GLM_5_TURBO_MODEL,
            "glm-5-turbo",
            "GLM-5-Turbo",
        }
        glm_47_aliases = {
            "glm-4.7",
        }

        if model in glm_46v_aliases:
            normalized_request = dict(request_dict)
            normalized_request["_original_model"] = model
            normalized_request["model"] = "glm-4.6v"
            return normalized_request

        if model in glm_5v_turbo_aliases:
            normalized_request = dict(request_dict)
            normalized_request["_original_model"] = model
            normalized_request["model"] = "glm-5v-turbo"
            return normalized_request

        if model in glm_5_aliases:
            normalized_request = dict(request_dict)
            normalized_request["_original_model"] = model
            normalized_request["model"] = "glm-5"
            return normalized_request

        if model in glm_5_turbo_aliases:
            normalized_request = dict(request_dict)
            normalized_request["_original_model"] = model
            normalized_request["model"] = "glm-5-turbo"
            normalized_request["enable_thinking"] = True
            return normalized_request

        if model in glm_47_aliases:
            normalized_request = dict(request_dict)
            normalized_request["_original_model"] = model
            normalized_request["model"] = "glm-4.7"
            return normalized_request

        return request_dict

    async def prepare_request(self, request: OpenAIRequest) -> Tuple[dict, dict]:
        """准备请求数据。"""
        request_dict = request.model_dump()
        normalized_request_dict = self._normalize_request_model(request_dict)
        return normalized_request_dict, normalized_request_dict

    async def build_transformed(self, request_dict: dict, client: httpx.AsyncClient, upstream: str) -> dict:
        request_stage_log("transform_in", "开始转换请求格式: OpenAI -> Z.AI", upstream=upstream)
        return await self.transformer.transform_request_in(
            request_dict,
            client=client,
            upstream_url=upstream,
        )

    async def get_request_context(self) -> Tuple[httpx.AsyncClient, Optional[str], str]:
        client, proxy = await network_manager.get_request_client()
        upstream = await network_manager.get_next_upstream()
        bind_request_context(proxy=proxy, upstream=upstream)
        debug_log("[REQUEST] 获取请求上下文", proxy=proxy or "直连", upstream=upstream)
        return client, proxy, upstream

    async def ensure_authorization(self, authorization: str) -> None:
        if settings.SKIP_AUTH_TOKEN:
            return

        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

        api_key = authorization[7:]
        if api_key != settings.AUTH_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid API key")

    def _store_conversation_success(
        self,
        request_dict_for_transform: dict,
        transformed: dict,
        assistant_message: Dict[str, str],
    ) -> None:
        try:
            conversation_state.store_next_turn(
                model=request_dict_for_transform.get("model", ""),
                request_messages=request_dict_for_transform.get("messages", []),
                assistant_message=assistant_message,
                chat_id=transformed["body"]["chat_id"],
                token=transformed.get("token", ""),
                last_request_id=transformed["body"]["id"],
                vision_assets=transformed.get("conversation", {}).get("vision_assets", []),
            )
        except Exception as exc:
            debug_log("[CONVERSATION] 会话状态写入失败", error=str(exc))

    def _invalidate_transformed_conversation(self, transformed: dict) -> None:
        chat_id = transformed.get("body", {}).get("chat_id")
        if chat_id:
            conversation_state.invalidate_chat(chat_id)

    async def handle_non_stream_request(
        self,
        request: OpenAIRequest,
        transformed: dict,
        request_client: httpx.AsyncClient,
        current_proxy: Optional[str],
        current_upstream: str,
        request_dict_for_transform: dict,
        json_lib,
    ) -> dict:
        bind_request_context(mode="non_stream")
        request_stage_log("non_stream_pipeline", "进入非流式处理流程")
        final_content = ""
        reasoning_content = ""
        latest_full_thinking = ""
        usage_info = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        retry_count = 0
        last_error = None
        last_status_code = None

        while retry_count <= settings.MAX_RETRIES:
            try:
                if retry_count > 0:
                    delay = self.calculate_backoff_delay(retry_count, last_status_code)
                    info_log(
                        "[RETRY] 非流式请求重试",
                        retry_count=retry_count,
                        delay=f"{delay:.2f}s",
                    )
                    await asyncio.sleep(delay)

                client = request_client
                headers = transformed["config"]["headers"].copy()

                attempt = retry_count + 1
                request_stage_log(
                    "upstream_request",
                    "向上游发起非流式请求",
                    attempt=attempt,
                    upstream=current_upstream,
                    proxy=current_proxy or "direct",
                )
                request_start_time = time.perf_counter()
                async with client.stream(
                    "POST",
                    transformed["config"]["url"],
                    json=transformed["body"],
                    headers=headers,
                ) as response:
                    ttfb = (time.perf_counter() - request_start_time) * 1000
                    debug_log("非流式上游TTFB", ttfb_ms=f"{ttfb:.2f}ms")

                    if response.status_code != 200:
                        last_status_code = response.status_code
                        error_text = await response.aread()
                        error_msg = error_text.decode("utf-8", errors="ignore")
                        error_log(
                            "上游返回错误",
                            status_code=response.status_code,
                            error_detail=error_msg[:200],
                        )

                        should_retry, transformed, request_client, current_proxy, current_upstream = await self._handle_retryable_error(
                            response.status_code,
                            retry_count,
                            transformed,
                            request_dict_for_transform,
                            request_client,
                            current_proxy,
                            current_upstream,
                        )
                        if should_retry:
                            retry_count += 1
                            continue

                        raise HTTPException(
                            status_code=response.status_code,
                            detail=f"Upstream error: {error_msg[:500]}",
                        )

                    request_stage_log(
                        "upstream_response",
                        "Z.AI 响应成功，开始聚合非流式数据",
                        status="success",
                        attempt=attempt,
                    )

                    thinking_content = ""
                    answer_content = ""
                    latest_full_edit = ""
                    latest_usage = None
                    is_thinking_model = transformed.get("is_thinking", False)

                    async for line in response.aiter_lines():
                        if not line:
                            continue

                        line = line.strip()
                        if not line.startswith("data:"):
                            try:
                                maybe_err = json_lib.loads(line)
                                if isinstance(maybe_err, dict) and (
                                    "error" in maybe_err or "code" in maybe_err or "message" in maybe_err
                                ):
                                    msg = (
                                        (maybe_err.get("error") or {}).get("message")
                                        if isinstance(maybe_err.get("error"), dict)
                                        else maybe_err.get("message")
                                    ) or "上游返回错误"
                                    raise HTTPException(status_code=500, detail=msg)
                            except json.JSONDecodeError:
                                pass
                            continue

                        data_str = line[5:].strip()
                        if not data_str or data_str.lower() == "[done]":
                            continue

                        try:
                            chunk = json_lib.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        if chunk.get("type") != "chat:completion":
                            continue

                        data = chunk.get("data", {})
                        phase = data.get("phase")
                        delta_content = data.get("delta_content", "")
                        edit_content = data.get("edit_content", "")
                        error_info = data.get("error")

                        if error_info:
                            error_detail = error_info.get("detail") or error_info.get("content") or "Unknown error"
                            error_log("[UPSTREAM_ERROR] 上游返回错误", error_detail=error_detail)
                            raise HTTPException(status_code=502, detail=f"Upstream error: {error_detail}")

                        if data.get("usage"):
                            latest_usage = data["usage"]

                        if phase == "tool_call":
                            if data.get("edit_index") is not None and edit_content:
                                image_urls = self.parser.extract_image_urls(edit_content)
                                if image_urls:
                                    markdown_images = self.parser.format_images_as_markdown(image_urls)
                                    if markdown_images:
                                        answer_content += "\n\n" + markdown_images + "\n\n"
                                else:
                                    thinking_content += edit_content
                            continue

                        if phase == "thinking":
                            if delta_content:
                                cleaned = self.parser.clean_thinking(delta_content)
                                if cleaned:
                                    thinking_content += cleaned
                            continue

                        if phase == "answer":
                            has_edit_index = data.get("edit_index") is not None
                            if has_edit_index and edit_content:
                                image_urls = self.parser.extract_image_urls(edit_content)
                                if image_urls:
                                    markdown_images = self.parser.format_images_as_markdown(image_urls)
                                    if markdown_images:
                                        answer_content += "\n\n" + markdown_images + "\n\n"
                                if "</details>" in edit_content:
                                    latest_full_edit = edit_content
                                else:
                                    cleaned = self.parser.clean_thinking(edit_content)
                                    if cleaned:
                                        new_thinking = self.parser.diff_new_content(thinking_content, cleaned)
                                        if new_thinking:
                                            thinking_content += new_thinking
                            if delta_content:
                                answer_content += delta_content
                            continue

                        if phase == "other":
                            if data.get("usage"):
                                latest_usage = data["usage"]
                            tail_text = None
                            if delta_content:
                                tail_text = delta_content
                            elif edit_content:
                                tail_text = self.parser.clean_thinking(edit_content)

                            if tail_text:
                                image_urls = self.parser.extract_image_urls(tail_text)
                                if image_urls:
                                    markdown_images = self.parser.format_images_as_markdown(image_urls)
                                    if markdown_images:
                                        answer_content += "\n\n" + markdown_images + "\n\n"
                                else:
                                    answer_content += tail_text
                            continue

                    if latest_full_edit:
                        thinking_part, answer_part = self.parser.split_edit_content(latest_full_edit)
                        if thinking_part:
                            thinking_content = thinking_part
                        if answer_part and not answer_content:
                            answer_content = answer_part

                    thinking_content = thinking_content.strip()
                    answer_content = answer_content.strip()
                    usage_info = latest_usage or {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    }

                    request_stage_log(
                        "non_stream_completed",
                        "非流式响应完成",
                        has_thinking=bool(thinking_content),
                        completion_tokens=usage_info.get("completion_tokens"),
                        prompt_tokens=usage_info.get("prompt_tokens"),
                    )

                    message = {
                        "role": "assistant",
                        "content": answer_content,
                    }
                    if thinking_content and is_thinking_model:
                        message["reasoning_content"] = thinking_content

                    self._store_conversation_success(
                        request_dict_for_transform=request_dict_for_transform,
                        transformed=transformed,
                        assistant_message=message,
                    )

                    return {
                        "id": transformed["body"]["chat_id"],
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "message": message,
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": usage_info,
                    }

            except HTTPException:
                raise
            except Exception as exc:
                error_log("非流式处理发生异常", error=str(exc))
                last_error = str(exc)
                retry_count += 1

        reset_request_context("mode")
        raise HTTPException(status_code=500, detail=f"非流式请求失败: {last_error}")

    def calculate_backoff_delay(
        self,
        retry_count: int,
        status_code: Optional[int] = None,
        base_delay: float = 1.5,
        max_delay: float = 8.0,
    ) -> float:
        linear_delay = base_delay * retry_count

        if status_code == 429:
            linear_delay *= 1.5
        elif status_code in [502, 503, 504]:
            linear_delay *= 1.2

        linear_delay = min(linear_delay, max_delay)
        jitter = linear_delay * 0.2
        return max(linear_delay + (2 * jitter * (time.time() % 1) - jitter), 0.5)

    async def stream_response(
        self,
        request: OpenAIRequest,
        transformed: dict,
        request_client: httpx.AsyncClient,
        current_proxy: Optional[str],
        current_upstream: str,
        request_dict_for_transform: dict,
        json_lib,
    ) -> AsyncIterator[str]:
        bind_request_context(mode="stream")
        request_stage_log("stream_pipeline", "进入流式处理流程")
        retry_count = 0
        last_error = None
        last_status_code = None

        while retry_count <= settings.MAX_RETRIES:
            try:
                if retry_count > 0:
                    delay = self.calculate_backoff_delay(retry_count, last_status_code)
                    info_log(
                        "[RETRY] 流式请求重试",
                        retry_count=retry_count,
                        delay=f"{delay:.2f}s",
                        last_status=last_status_code,
                    )
                    await asyncio.sleep(delay)

                client = request_client
                headers = transformed["config"]["headers"].copy()

                attempt = retry_count + 1
                request_stage_log(
                    "upstream_request",
                    "向上游发起流式请求",
                    attempt=attempt,
                    upstream=current_upstream,
                    proxy=current_proxy or "direct",
                )
                request_start_time = time.perf_counter()
                async with client.stream(
                    "POST",
                    transformed["config"]["url"],
                    json=transformed["body"],
                    headers=headers,
                ) as response:
                    ttfb = (time.perf_counter() - request_start_time) * 1000
                    debug_log("上游TTFB (首字节时间)", ttfb_ms=f"{ttfb:.2f}ms")

                    if response.status_code != 200:
                        error_text = await response.aread()
                        error_msg = error_text.decode("utf-8", errors="ignore")
                        error_log(
                            "上游返回错误",
                            status_code=response.status_code,
                            error_detail=error_msg[:200],
                        )

                        should_retry, transformed, request_client, current_proxy, current_upstream = await self._handle_retryable_error(
                            response.status_code,
                            retry_count,
                            transformed,
                            request_dict_for_transform,
                            request_client,
                            current_proxy,
                            current_upstream,
                        )
                        if should_retry:
                            retry_count += 1
                            last_status_code = response.status_code
                            last_error = f"{response.status_code}: {error_msg}"
                            continue

                        error_response = {
                            "error": {
                                "message": f"Upstream error: {response.status_code}",
                                "type": "upstream_error",
                                "code": response.status_code,
                                "details": error_msg[:500],
                            }
                        }
                        yield f"data: {json_lib.dumps(error_response)}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    request_stage_log(
                        "upstream_stream_ready",
                        "Z.AI 响应成功，开始处理 SSE 流",
                        status="success",
                        attempt=attempt,
                    )

                    has_thinking = False
                    thinking_accumulator = ""
                    answer_accumulator = ""
                    latest_usage = None
                    is_thinking_model = transformed.get("is_thinking", False)
                    is_vision_model = transformed.get("is_vision_model", False)

                    def _log_v_output(output_type: str, content: str):
                        if is_vision_model and content:
                            v_series_debug_log(
                                "V系列输出响应",
                                output_type=output_type,
                                content=content,
                            )

                    async for line in response.aiter_lines():
                        if not line or not line.strip():
                            continue

                        if not line.startswith("data:"):
                            continue

                        chunk_str = line[5:].strip()
                        if not chunk_str or chunk_str == "[DONE]":
                            if chunk_str == "[DONE]":
                                yield "data: [DONE]\n\n"
                            continue

                        try:
                            chunk_data = json_lib.loads(chunk_str)
                        except json.JSONDecodeError:
                            continue

                        if chunk_data.get("type") != "chat:completion":
                            yield f"data: {chunk_str}\n\n"
                            continue

                        data = chunk_data.get("data", {})
                        delta_content = data.get("delta_content")
                        edit_content = data.get("edit_content")
                        phase = data.get("phase")
                        is_done = phase == "done" or data.get("done")
                        error_info = data.get("error")

                        if is_vision_model:
                            v_series_debug_log(
                                "V系列上游响应",
                                phase=phase,
                                delta_content=delta_content,
                                edit_content=edit_content,
                                has_edit_index=data.get("edit_index") is not None,
                                edit_index=data.get("edit_index"),
                            )

                        if error_info:
                            error_detail = error_info.get("detail") or error_info.get("content") or "Unknown error"
                            error_log(f"[UPSTREAM_ERROR] 上游返回错误: {error_detail}")
                            
                            if not has_thinking:
                                has_thinking = True
                                yield self.chunk.build_role_chunk(json_lib, transformed, request)
                            
                            error_message = f"\n\n[系统提示: {error_detail}]"
                            yield self.chunk.build_content_chunk(json_lib, transformed, request, error_message)
                            
                            if is_done:
                                finish_chunk = self.chunk.build_finish_chunk(json_lib, transformed, request)
                                yield finish_chunk
                                yield "data: [DONE]\n\n"
                                await self._mark_token_success(transformed)
                                request_stage_log("stream_completed", "流式响应完成（带错误）", has_error=True)
                                return
                            continue

                        if phase == "tool_call":
                            if data.get("edit_index") is not None and edit_content:
                                if not has_thinking:
                                    has_thinking = True
                                    yield self.chunk.build_role_chunk(json_lib, transformed, request)
                                
                                image_urls = self.parser.extract_image_urls(edit_content)
                                if image_urls:
                                    markdown_images = self.parser.format_images_as_markdown(image_urls)
                                    if markdown_images:
                                        yield self.chunk.build_content_chunk(json_lib, transformed, request, "\n\n" + markdown_images + "\n\n")
                                        answer_accumulator += markdown_images
                                
                                if "<glm_block" in edit_content:
                                    glm_block_match = re.search(r'<glm_block[^>]*>.*?</glm_block>', edit_content, re.DOTALL)
                                    if glm_block_match:
                                        before_glm_block = edit_content[:glm_block_match.start()].strip()
                                        if before_glm_block:
                                            new_answer = self.parser.diff_new_content(answer_accumulator, before_glm_block)
                                            if new_answer:
                                                _log_v_output("tool_call_content", new_answer)
                                                yield self.chunk.build_content_chunk(json_lib, transformed, request, new_answer)
                                                answer_accumulator += new_answer
                                else:
                                    cleaned = self.parser.clean_thinking(edit_content)
                                    if cleaned:
                                        new_answer = self.parser.diff_new_content(answer_accumulator, cleaned)
                                        if new_answer:
                                            _log_v_output("tool_call_plain_content", new_answer)
                                            yield self.chunk.build_content_chunk(json_lib, transformed, request, new_answer)
                                            answer_accumulator += new_answer
                            continue

                        if phase == "thinking":
                            if delta_content:
                                if not has_thinking:
                                    has_thinking = True
                                    yield self.chunk.build_role_chunk(json_lib, transformed, request)
                                
                                if not is_thinking_model:
                                    cleaned = self.parser.clean_thinking(delta_content)
                                    if cleaned:
                                        _log_v_output("thinking_content", cleaned)
                                        yield self.chunk.build_content_chunk(json_lib, transformed, request, cleaned)
                                        answer_accumulator += cleaned

                                else:
                                    raw_new = self.parser.diff_new_content(thinking_accumulator, delta_content)
                                    if raw_new:
                                        cleaned_raw = self.parser.clean_thinking(raw_new)
                                        if cleaned_raw:
                                            _log_v_output("reasoning_content", cleaned_raw)
                                            yield self.chunk.build_reasoning_chunk(
                                                json_lib,
                                                transformed,
                                                request,
                                                cleaned_raw,
                                            )
                                            thinking_accumulator += raw_new

                                    cleaned_full = self.parser.clean_thinking(delta_content)
                                    if cleaned_full:
                                        new_reasoning = self.parser.diff_new_content(
                                            self.parser.clean_thinking(thinking_accumulator),
                                            cleaned_full,
                                        )
                                        if new_reasoning:
                                            _log_v_output("reasoning_content_fallback", new_reasoning)
                                            yield self.chunk.build_reasoning_chunk(
                                                json_lib,
                                                transformed,
                                                request,
                                                new_reasoning,
                                            )
                            continue

                        if phase == "answer":
                            has_edit_index = data.get("edit_index") is not None
                            if has_edit_index and edit_content:
                                if not has_thinking:
                                    has_thinking = True
                                    yield self.chunk.build_role_chunk(json_lib, transformed, request)
                                
                                image_urls = self.parser.extract_image_urls(edit_content)
                                if image_urls:
                                    markdown_images = self.parser.format_images_as_markdown(image_urls)
                                    if markdown_images:
                                        yield self.chunk.build_content_chunk(json_lib, transformed, request, "\n\n" + markdown_images + "\n\n")
                                        answer_accumulator += markdown_images
                                
                                if "</details>" in edit_content:
                                    thinking_part, answer_part = self.parser.split_edit_content(edit_content)
                                    
                                    if thinking_part:
                                        new_thinking = self.parser.diff_new_content(thinking_accumulator, thinking_part)
                                        if new_thinking:
                                            if not is_thinking_model:
                                                _log_v_output("answer_thinking_content", new_thinking)
                                                yield self.chunk.build_content_chunk(json_lib, transformed, request, new_thinking)
                                                answer_accumulator += new_thinking
                                            else:
                                                _log_v_output("answer_reasoning_content", new_thinking)
                                                yield self.chunk.build_reasoning_chunk(json_lib, transformed, request, new_thinking)
                                            thinking_accumulator += new_thinking
                                    
                                    if answer_part:
                                        new_answer = self.parser.diff_new_content(answer_accumulator, answer_part)
                                        if new_answer:
                                            _log_v_output("answer_content", new_answer)
                                            yield self.chunk.build_content_chunk(json_lib, transformed, request, new_answer)
                                            answer_accumulator += new_answer
                                else:
                                    cleaned = self.parser.clean_thinking(edit_content)
                                    if cleaned:
                                        new_answer = self.parser.diff_new_content(answer_accumulator, cleaned)
                                        if new_answer:
                                            _log_v_output("answer_plain_content", new_answer)
                                            yield self.chunk.build_content_chunk(json_lib, transformed, request, new_answer)
                                            answer_accumulator += new_answer
                                continue

                            if delta_content:
                                if not has_thinking:
                                    has_thinking = True
                                    yield self.chunk.build_role_chunk(json_lib, transformed, request)
                                
                                _log_v_output("delta_content", delta_content)
                                yield self.chunk.build_content_chunk(json_lib, transformed, request, delta_content)
                                answer_accumulator += delta_content
                            continue

                        if phase == "other":
                            if data.get("usage"):
                                latest_usage = data["usage"]

                            has_edit_index = data.get("edit_index") is not None
                            tail_text = None
                            if edit_content:
                                tail_text = edit_content
                            elif delta_content:
                                tail_text = delta_content

                            if tail_text:
                                if not has_thinking:
                                    has_thinking = True
                                    yield self.chunk.build_role_chunk(json_lib, transformed, request)
                                
                                if has_edit_index:
                                    image_urls = self.parser.extract_image_urls(tail_text)
                                    if image_urls:
                                        markdown_images = self.parser.format_images_as_markdown(image_urls)
                                        if markdown_images:
                                            yield self.chunk.build_content_chunk(json_lib, transformed, request, "\n\n" + markdown_images + "\n\n")
                                            answer_accumulator += markdown_images
                                    else:
                                        cleaned = self.parser.clean_thinking(tail_text)
                                        if cleaned:
                                            _log_v_output("other_tail_content", cleaned)
                                            yield self.chunk.build_content_chunk(json_lib, transformed, request, cleaned)
                                            answer_accumulator += cleaned
                                else:
                                    _log_v_output("other_content", tail_text)
                                    yield self.chunk.build_content_chunk(json_lib, transformed, request, tail_text)
                            continue

                        if data.get("usage"):
                            latest_usage = data["usage"]

                        if is_done:
                            assistant_message = {
                                "role": "assistant",
                                "content": answer_accumulator.strip(),
                            }
                            if is_thinking_model:
                                reasoning_content = self.parser.clean_thinking(thinking_accumulator).strip()
                                if reasoning_content:
                                    assistant_message["reasoning_content"] = reasoning_content

                            self._store_conversation_success(
                                request_dict_for_transform=request_dict_for_transform,
                                transformed=transformed,
                                assistant_message=assistant_message,
                            )

                            finish_chunk = self.chunk.build_finish_chunk(json_lib, transformed, request, usage=latest_usage)
                            yield finish_chunk
                            if latest_usage:
                                yield self.chunk.build_usage_chunk(json_lib, transformed, request, latest_usage)
                            yield "data: [DONE]\n\n"
                            await self._mark_token_success(transformed)
                            request_stage_log("stream_completed", "流式响应完成", has_error=False)
                            return

                    assistant_message = {
                        "role": "assistant",
                        "content": answer_accumulator.strip(),
                    }
                    if is_thinking_model:
                        reasoning_content = self.parser.clean_thinking(thinking_accumulator).strip()
                        if reasoning_content:
                            assistant_message["reasoning_content"] = reasoning_content

                    self._store_conversation_success(
                        request_dict_for_transform=request_dict_for_transform,
                        transformed=transformed,
                        assistant_message=assistant_message,
                    )

                    finish_chunk = self.chunk.build_finish_chunk(json_lib, transformed, request, usage=latest_usage)
                    yield finish_chunk
                    if latest_usage:
                        yield self.chunk.build_usage_chunk(json_lib, transformed, request, latest_usage)
                    yield "data: [DONE]\n\n"

                    await self._mark_token_success(transformed)
                    request_stage_log(
                        "stream_completed",
                        "流式响应完成",
                        has_error=False,
                    )
                    return

            except Exception as exc:
                error_log("流处理错误", error=str(exc))
                retry_count += 1
                last_error = str(exc)
                last_status_code = None

                if network_manager.has_proxy_pool() and "connect" in str(exc).lower():
                    await network_manager.switch_proxy_on_failure()

                if retry_count > settings.MAX_RETRIES:
                    error_response = {
                        "error": {
                            "message": f"Stream processing failed: {last_error}",
                            "type": "stream_error",
                        }
                    }
                    yield f"data: {json_lib.dumps(error_response)}\n\n"
                    yield "data: [DONE]\n\n"
                    error_log("[REQUEST] 流式响应错误")
                    return
            finally:
                reset_request_context("mode")

    async def _mark_token_success(self, transformed: dict) -> None:
        token_pool = await get_token_pool()
        current_token = transformed.get("token", "")
        if current_token:
            token_pool.mark_token_success(current_token)

    async def _handle_retryable_error(
        self,
        status_code: int,
        retry_count: int,
        transformed: dict,
        request_dict_for_transform: dict,
        request_client: httpx.AsyncClient,
        current_proxy: Optional[str],
        current_upstream: str,
    ) -> Tuple[bool, dict, httpx.AsyncClient, Optional[str], str]:
        retryable_codes = [400, 401, 405, 429, 502, 503, 504]
        if status_code not in retryable_codes or retry_count >= settings.MAX_RETRIES:
            return False, transformed, request_client, current_proxy, current_upstream

        token_pool = await get_token_pool()
        current_token = transformed.get("token", "")

        if current_token:
            token_pool.mark_token_failure(current_token)
        self._invalidate_transformed_conversation(transformed)
        info_log(f"[CONFIG] 配置Token错误 {status_code}，切换Token")

        await self.transformer.switch_token()
        await self.transformer.refresh_header_template()
        current_upstream = await network_manager.get_next_upstream()
        transformed = await self.transformer.transform_request_in(
            request_dict_for_transform,
            client=request_client,
            upstream_url=current_upstream,
        )
        info_log("[OK] 已切换到下一个配置Token")

        if network_manager.has_upstream_pool() and network_manager.upstream_strategy == "failover":
            await network_manager.switch_upstream_on_failure()
            current_upstream = await network_manager.get_next_upstream()
            transformed = await self.transformer.transform_request_in(
                request_dict_for_transform,
                client=request_client,
                upstream_url=current_upstream,
            )
            info_log("[FAILOVER] Token错误，已切换上游")

        if status_code in [502, 503, 504]:
            if network_manager.has_proxy_pool() and network_manager.proxy_strategy == "failover":
                await network_manager.switch_proxy_on_failure()
            if network_manager.has_upstream_pool() and network_manager.upstream_strategy == "failover":
                await network_manager.switch_upstream_on_failure()
                current_upstream = await network_manager.get_next_upstream()
                transformed = await self.transformer.transform_request_in(
                    request_dict_for_transform,
                    client=request_client,
                    upstream_url=current_upstream,
                )
                info_log("[FAILOVER] 网络错误，已切换上游")

        return True, transformed, request_client, current_proxy, current_upstream

    async def mark_token_success_if_configured(self, transformed: dict) -> None:
        token_pool = await get_token_pool()
        current_token = transformed.get("token", "")
        if current_token:
            token_pool.mark_token_success(current_token)


chat_completion_service = ChatCompletionService()
