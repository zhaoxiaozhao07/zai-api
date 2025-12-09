"""Service layer orchestrating OpenAI-compatible chat completions."""

from __future__ import annotations

import asyncio
import json
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
)
from ..schemas import OpenAIRequest
from ..config import settings
from ..zai_transformer import ZAITransformer
from ..token_pool import get_token_pool
from .network_manager import network_manager


class ChatCompletionService:
    """Encapsulate chat completion workflow independent of FastAPI layer."""

    def __init__(self) -> None:
        self.transformer = ZAITransformer()

    async def prepare_request(self, request: OpenAIRequest) -> Tuple[dict, dict]:
        request_dict = request.model_dump()
        return self._prepare_messages(request, request_dict)

    def _prepare_messages(self, request: OpenAIRequest, request_dict: dict) -> Tuple[dict, dict]:
        messages = [
            msg.model_dump() if hasattr(msg, "model_dump") else msg
            for msg in request.messages
        ]
        
        # 移除 tools 和 tool_choice 参数（上游不支持）
        transformed_dict = request_dict.copy()
        if "tools" in transformed_dict:
            del transformed_dict["tools"]
            info_log("[请求处理] 已移除 tools 参数（上游不支持）")
        if "tool_choice" in transformed_dict:
            del transformed_dict["tool_choice"]
            info_log("[请求处理] 已移除 tool_choice 参数（上游不支持）")
        
        transformed_dict["messages"] = messages
        return request_dict, transformed_dict

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
                    debug_log("⏱️ 非流式上游TTFB", ttfb_ms=f"{ttfb:.2f}ms")

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

                    # 内容累积变量
                    # 上游会先以多条 `phase=thinking` 的 data 推理，再在 `phase=answer` / `phase=other` 中给出
                    # 完整的 edit_content（含 <details> 推理 + 最终回答）以及增量的 delta_content
                    # 为了构造非流式一次性返回，我们：
                    # 1）持续累积 thinking/answer 的 delta_content，用于兜底
                    # 2）优先从带 </details> 的 edit_content 中解析出完整 reasoning_content + answer
                    thinking_content = ""  # 累积 thinking 阶段的 delta_content（兜底）
                    answer_content = ""    # 累积 answer 阶段的 delta_content（兜底）
                    latest_full_edit = ""  # 记录最后一个包含 </details> 的 edit_content（完整思考+正文）
                    latest_usage = None

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
                            except (json.JSONDecodeError, HTTPException):
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

                        # 收集 usage 信息
                        if data.get("usage"):
                            latest_usage = data["usage"]

                        # tool_call 阶段：如果带有 edit_index，优先检测图片，否则放入 thinking
                        if phase == "tool_call":
                            if data.get("edit_index") is not None and edit_content:
                                # 先检测是否有图片
                                image_urls = self._extract_image_urls(edit_content)
                                if image_urls:
                                    # 有图片，转换为markdown格式放入 answer_content
                                    markdown_images = self._format_images_as_markdown(image_urls)
                                    if markdown_images:
                                        answer_content += "\n\n" + markdown_images + "\n\n"
                                else:
                                    # 无图片，放入 thinking_content
                                    thinking_content += edit_content
                            continue

                        # thinking 阶段：累积 delta_content（只用于兜底，优先用 answer/other 阶段的完整 edit_content）
                        if phase == "thinking":
                            if delta_content:
                                thinking_content += delta_content
                            continue

                        # answer 阶段：处理 edit_content（包含完整 thinking+正文）和 delta_content
                        if phase == "answer":
                            # 如果有 edit_content 且包含完整的 </details>，优先用它来解析完整推理+回答
                            if edit_content and "</details>" in edit_content:
                                latest_full_edit = edit_content
                            
                            # 累积 answer 的 delta_content
                            if delta_content:
                                answer_content += delta_content
                            continue

                        # other 阶段：可能有 usage 信息，也可能有最后的 edit_content 片段
                        # 如果带有 edit_index，说明是工具调用相关内容，应放入 thinking
                        if phase == "other":
                            # 收集 usage
                            if data.get("usage"):
                                latest_usage = data["usage"]

                            # 判断是否是工具调用相关内容（带 edit_index）
                            has_edit_index = data.get("edit_index") is not None

                            # 获取内容
                            tail_text = None
                            if edit_content:
                                tail_text = edit_content
                            elif delta_content:
                                tail_text = delta_content

                            if tail_text:
                                if has_edit_index:
                                    # 带 edit_index 的内容：先检测是否有图片
                                    image_urls = self._extract_image_urls(tail_text)
                                    if image_urls:
                                        # 有图片，转换为markdown格式放入 answer_content
                                        markdown_images = self._format_images_as_markdown(image_urls)
                                        if markdown_images:
                                            answer_content += "\n\n" + markdown_images + "\n\n"
                                    else:
                                        # 无图片，放入 thinking_content
                                        thinking_content += tail_text
                                else:
                                    # 普通内容放入 answer_content
                                    answer_content += tail_text
                            continue

                    # 如果上游在 answer/other 阶段给出了完整的 edit_content（含 </details>），
                    # 则优先用它来拆分出 reasoning_content 和 最终回答，避免仅依赖增量导致内容不完整或截断
                    # 注意：latest_full_edit 中可能不包含 <details> 开头（例如只剩属性残片），
                    # 这里统一交给 _split_edit_content + _clean_thinking 做清洗，移除 true" duration="1" 等残留。
                    if latest_full_edit:
                        thinking_part, answer_part = self._split_edit_content(latest_full_edit)
                        if thinking_part:
                            thinking_content = thinking_part
                        if answer_part:
                            answer_content = answer_part

                    # 清理内容
                    thinking_content = thinking_content.strip()
                    answer_content = answer_content.strip()

                    # 使用 usage 信息
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

                    # 构建消息对象
                    message = {
                        "role": "assistant",
                        "content": answer_content,
                    }
                    # 如果有 thinking 内容，添加 reasoning_content 字段
                    if thinking_content:
                        message["reasoning_content"] = thinking_content

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
                    debug_log("⏱️ 上游TTFB (首字节时间)", ttfb_ms=f"{ttfb:.2f}ms")

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
                    # 累积已输出的 reasoning/content，用于做增量 diff，避免覆盖式 delta 导致截断
                    thinking_accumulator = ""
                    answer_accumulator = ""

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

                        # 检测上游返回的错误（如内容安全警告）
                        if error_info:
                            error_detail = error_info.get("detail") or error_info.get("content") or "Unknown error"
                            error_log(f"[UPSTREAM_ERROR] 上游返回错误: {error_detail}")
                            
                            if not has_thinking:
                                has_thinking = True
                                yield self._build_role_chunk(json_lib, transformed, request)
                            
                            error_message = f"\n\n[系统提示: {error_detail}]"
                            yield self._build_content_chunk(json_lib, transformed, request, error_message)
                            
                            if is_done:
                                finish_chunk = self._build_finish_chunk(json_lib, transformed, request)
                                yield finish_chunk
                                yield "data: [DONE]\n\n"
                                await self._mark_token_success(transformed)
                                request_stage_log("stream_completed", "流式响应完成（带错误）", has_error=True)
                                return
                            continue

                        # tool_call 阶段：如果带有 edit_index，优先检测图片，否则放入思考
                        if phase == "tool_call":
                            if data.get("edit_index") is not None and edit_content:
                                if not has_thinking:
                                    has_thinking = True
                                    yield self._build_role_chunk(json_lib, transformed, request)
                                
                                # 先检测是否有图片（包括 image_reference 工具返回的图片）
                                image_urls = self._extract_image_urls(edit_content)
                                if image_urls:
                                    # 有图片，转换为markdown格式放入正文
                                    markdown_images = self._format_images_as_markdown(image_urls)
                                    if markdown_images:
                                        yield self._build_content_chunk(json_lib, transformed, request, "\n\n" + markdown_images + "\n\n")
                                        answer_accumulator += markdown_images
                                else:
                                    # 无图片，计算新增的思考内容
                                    new_thinking = self._diff_new_content(thinking_accumulator, edit_content)
                                    if new_thinking:
                                        cleaned = self._clean_thinking(new_thinking)
                                        if cleaned:
                                            yield self._build_reasoning_chunk(json_lib, transformed, request, cleaned)
                                            thinking_accumulator += new_thinking
                            continue

                        # thinking 阶段：流式输出 reasoning_content（使用增量 diff，防止覆盖式 delta 截断）
                        if phase == "thinking":
                            if delta_content:
                                if not has_thinking:
                                    has_thinking = True
                                    yield self._build_role_chunk(json_lib, transformed, request)
                                
                                # 先做“原样增量”：直接把本次 delta 里的新增部分先输出
                                raw_new = self._diff_new_content(thinking_accumulator, delta_content)
                                if raw_new:
                                    cleaned_raw = self._clean_thinking(raw_new)
                                    if cleaned_raw:
                                        yield self._build_reasoning_chunk(
                                            json_lib,
                                            transformed,
                                            request,
                                            cleaned_raw,
                                        )
                                        thinking_accumulator += raw_new

                                # 再做一次基于清洗后的兜底增量，防止上游覆盖式 delta 导致遗漏
                                cleaned_full = self._clean_thinking(delta_content)
                                if cleaned_full:
                                    new_reasoning = self._diff_new_content(
                                        self._clean_thinking(thinking_accumulator),
                                        cleaned_full,
                                    )
                                    if new_reasoning:
                                        yield self._build_reasoning_chunk(
                                            json_lib,
                                            transformed,
                                            request,
                                            new_reasoning,
                                        )
                                        # 这里不再修改 thinking_accumulator，避免与原始增量状态不一致
                            continue

                        # answer 阶段：处理 edit_content（包含完整thinking）和 delta_content
                        if phase == "answer":
                            # 如果有 edit_content 且包含完整 thinking，忽略（因为已在 thinking 阶段输出）
                            if edit_content and "</details>" in edit_content:
                                # edit_content 包含完整的 thinking，但我们已经通过 delta 输出了
                                pass
                            
                            # 流式输出 answer 的 delta_content
                            if delta_content:
                                if not has_thinking:
                                    has_thinking = True
                                    yield self._build_role_chunk(json_lib, transformed, request)
                                
                                yield self._build_content_chunk(json_lib, transformed, request, delta_content)
                                answer_accumulator += delta_content
                            continue

                        # other 阶段：可能有 usage 信息，也可能携带正文的最后一小段（edit_content 或 delta_content）
                        # 如果带有 edit_index，说明是工具调用相关内容，应放入思考
                        if phase == "other":
                            # 1) 先处理 usage
                            if data.get("usage"):
                                yield self._build_usage_chunk(json_lib, transformed, request, data["usage"])

                            # 2) 判断是否是工具调用相关内容（带 edit_index）
                            has_edit_index = data.get("edit_index") is not None
                            
                            # 3) 处理内容
                            tail_text = None
                            if edit_content:
                                tail_text = edit_content
                            elif delta_content:
                                tail_text = delta_content

                            if tail_text:
                                if not has_thinking:
                                    has_thinking = True
                                    yield self._build_role_chunk(json_lib, transformed, request)
                                
                                if has_edit_index:
                                    # 带 edit_index 的内容：先检测是否有图片
                                    image_urls = self._extract_image_urls(tail_text)
                                    if image_urls:
                                        # 有图片，转换为markdown格式放入正文
                                        markdown_images = self._format_images_as_markdown(image_urls)
                                        if markdown_images:
                                            yield self._build_content_chunk(json_lib, transformed, request, "\n\n" + markdown_images + "\n\n")
                                            answer_accumulator += markdown_images
                                    else:
                                        # 无图片，放入 reasoning_content（思考）
                                        new_thinking = self._diff_new_content(thinking_accumulator, tail_text)
                                        if new_thinking:
                                            cleaned = self._clean_thinking(new_thinking)
                                            if cleaned:
                                                yield self._build_reasoning_chunk(json_lib, transformed, request, cleaned)
                                                thinking_accumulator += new_thinking
                                else:
                                    # 普通内容放入 content（正文）
                                    yield self._build_content_chunk(json_lib, transformed, request, tail_text)
                            continue

                        # 输出 usage 信息
                        if data.get("usage"):
                            yield self._build_usage_chunk(json_lib, transformed, request, data["usage"])

                        # 检查是否为 done 状态
                        if is_done:
                            finish_chunk = self._build_finish_chunk(json_lib, transformed, request)
                            yield finish_chunk
                            yield "data: [DONE]\n\n"
                            await self._mark_token_success(transformed)
                            request_stage_log("stream_completed", "流式响应完成", has_error=False)
                            return

                    finish_chunk = self._build_finish_chunk(json_lib, transformed, request)
                    yield finish_chunk
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
        
        # Token 错误处理：标记失败并切换
        if current_token:
            token_pool.mark_token_failure(current_token)
        info_log(f"[TOKEN] Token错误 {status_code}，切换Token")

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

    def _build_role_chunk(self, json_lib, transformed: dict, request: OpenAIRequest) -> str:
        return f"data: {json_lib.dumps({
            'choices': [{
                'delta': {'role': 'assistant'},
                'finish_reason': None,
                'index': 0,
                'logprobs': None,
            }],
            'created': int(time.time()),
            'id': transformed['body']['chat_id'],
            'model': request.model,
            'object': 'chat.completion.chunk',
            'system_fingerprint': 'fp_zai_001',
        })}\n\n"

    def _build_content_chunk(
        self,
        json_lib,
        transformed: dict,
        request: OpenAIRequest,
        content: str,
    ) -> str:
        return f"data: {json_lib.dumps({
            'choices': [{
                'delta': {'content': content},
                'finish_reason': None,
                'index': 0,
                'logprobs': None,
            }],
            'created': int(time.time()),
            'id': transformed['body']['chat_id'],
            'model': request.model,
            'object': 'chat.completion.chunk',
            'system_fingerprint': 'fp_zai_001',
        })}\n\n"

    def _build_reasoning_chunk(
        self,
        json_lib,
        transformed: dict,
        request: OpenAIRequest,
        reasoning_content: str,
    ) -> str:
        """构建包含 reasoning_content 的流式响应块"""
        return f"data: {json_lib.dumps({
            'choices': [{
                'delta': {'reasoning_content': reasoning_content},
                'finish_reason': None,
                'index': 0,
                'logprobs': None,
            }],
            'created': int(time.time()),
            'id': transformed['body']['chat_id'],
            'model': request.model,
            'object': 'chat.completion.chunk',
            'system_fingerprint': 'fp_zai_001',
        })}\n\n"

    def _build_usage_chunk(self, json_lib, transformed: dict, request: OpenAIRequest, usage) -> str:
        return f"data: {json_lib.dumps({
            'choices': [{
                'delta': {},
                'finish_reason': None,
                'index': 0,
                'logprobs': None,
            }],
            'created': int(time.time()),
            'id': transformed['body']['chat_id'],
            'model': request.model,
            'object': 'chat.completion.chunk',
            'system_fingerprint': 'fp_zai_001',
            'usage': usage,
        })}\n\n"

    def _build_finish_chunk(self, json_lib, transformed: dict, request: OpenAIRequest) -> str:
        return f"data: {json_lib.dumps({
            'choices': [{
                'delta': {},
                'finish_reason': 'stop',
                'index': 0,
                'logprobs': None,
            }],
            'created': int(time.time()),
            'id': transformed['body']['chat_id'],
            'model': request.model,
            'object': 'chat.completion.chunk',
            'system_fingerprint': 'fp_zai_001',
        })}\n\n"

    def _extract_search_info(self, reasoning_content: str, edit_content: str) -> str:
        """从 edit_content 中提取搜索信息"""
        if edit_content and "<glm_block" in edit_content and "search" in edit_content:
            try:
                import re
                decoded = edit_content
                try:
                    decoded = edit_content.encode("utf-8").decode("unicode_escape").encode("latin1").decode("utf-8")
                except Exception:
                    try:
                        import codecs
                        decoded = codecs.decode(edit_content, "unicode_escape")
                    except Exception:
                        pass

                queries_match = re.search(r'"queries":\s*\[(.*?)\]', decoded)
                if queries_match:
                    queries_str = queries_match.group(1)
                    queries = re.findall(r'"([^"]+)"', queries_str)
                    if queries:
                        search_info = "🔍 **搜索：** " + "　".join(queries[:5])
                        reasoning_content += f"\n\n{search_info}\n\n"
                        debug_log("[搜索信息] 提取到搜索查询", queries=queries)
            except Exception as exc:
                debug_log("[搜索信息] 提取失败", error=str(exc))
        return reasoning_content

    def _extract_image_urls(self, content: str) -> list:
        """从上游响应内容中提取图片URL
        
        处理格式示例：
        1. {"image_url":{"url":"https://qc4n.bigmodel.cn/xxx.png?..."}}
        2. {"img_url": "https://bigmodel-us3-prod-agent.cn-wlcb.ufileos.com/xxx.jpg", ...}
        
        Returns:
            list: 提取到的图片URL列表
        """
        import re
        
        if not content:
            return []
        
        image_urls = []
        
        # === 格式1: image_url 类型（bigmodel.cn 域名）===
        # 匹配 "type":"image_url" 后面的 "url":"xxx" 模式
        if '"type\\":\\"image_url\\"' in content or '"type":"image_url"' in content:
            # 模式1：转义JSON格式 \"url\":\"xxx\"
            pattern1 = r'\"url\":\s*\"(https?://[^\"\\]+(?:\\.[^\"\\]+)*[^\"\\]*)\"'
            # 模式2：普通JSON格式 "url":"xxx"
            pattern2 = r'"url":\s*"(https?://[^"]+)"'
            
            matches = re.findall(pattern1, content)
            for url in matches:
                clean_url = url.replace('\\/', '/').replace('\\"', '"')
                if clean_url and 'bigmodel.cn' in clean_url:
                    image_urls.append(clean_url)
            
            if not image_urls:
                matches = re.findall(pattern2, content)
                for url in matches:
                    if url and 'bigmodel.cn' in url:
                        image_urls.append(url)
        
        # === 格式2: img_url 类型（ufileos.com 域名，image_reference 工具调用）===
        # 匹配 "img_url": "https://xxx.ufileos.com/xxx.jpg" 模式
        if 'img_url' in content or 'image_reference' in content:
            # 转义JSON格式: \"img_url\": \"xxx\"
            pattern_img = r'\\\"img_url\\\":\s*\\\"(https?://[^\\\"]+)\\\"'
            matches = re.findall(pattern_img, content)
            for url in matches:
                clean_url = url.replace('\\/', '/')
                if clean_url and ('ufileos.com' in clean_url or 'bigmodel' in clean_url):
                    image_urls.append(clean_url)
            
            # 普通JSON格式: "img_url": "xxx"
            if not image_urls:
                pattern_img2 = r'"img_url":\s*"(https?://[^"]+)"'
                matches = re.findall(pattern_img2, content)
                for url in matches:
                    if url and ('ufileos.com' in url or 'bigmodel' in url):
                        image_urls.append(url)
        
        return image_urls

    def _format_images_as_markdown(self, image_urls: list) -> str:
        """将图片URL列表格式化为markdown图片格式
        
        Args:
            image_urls: 图片URL列表
            
        Returns:
            str: markdown格式的图片字符串
        """
        if not image_urls:
            return ""
        
        markdown_images = []
        for i, url in enumerate(image_urls, 1):
            # 生成markdown图片格式
            markdown_images.append(f"![图片{i}]({url})")
        
        return "\n\n".join(markdown_images)

    def _clean_thinking(self, delta_content: str) -> str:
        """清理 thinking 内容，提取纯文本
        
        处理格式：
        - 移除 <details> 和 <summary> 标签
        - 移除 markdown 引用符号 "> "
        - 保留纯文本内容
        """
        import re
        
        if not delta_content:
            return ""
        
        # 0. 先丢弃可能出现在 <details> 之前的属性残片，如：true" duration="2" view="" last_tool_call_name="">
        #   这类内容通常出现在 edit_content 开头，但并不是思考正文的一部分
        #   策略：如果首行包含 duration= 或 last_tool_call_name 等字段，且以 "> 或 "> 结尾，则视为属性串，丢弃整行
        first_newline = delta_content.find("\n")
        if first_newline != -1:
            first_line = delta_content[:first_newline].strip()
            # 如果这一行包含典型的 <details> 属性字段，且以 > 或 "> 结尾，判定为属性残片
            if re.search(r'(duration=|last_tool_call_name|view=)', first_line) and re.search(r'[">]$', first_line):
                delta_content = delta_content[first_newline + 1 :]

        # 1. 移除 <details> 开始标签（包括所有属性）
        delta_content = re.sub(r'<details[^>]*>', '', delta_content)
        
        # 2. 移除 </details> 结束标签
        delta_content = re.sub(r'</details>', '', delta_content)
        
        # 3. 移除 <summary> 标签及其内容（如 "Thinking..." 或 "Thought for X seconds"）
        delta_content = re.sub(r'<summary[^>]*>.*?</summary>', '', delta_content, flags=re.DOTALL)
        
        # 4. 移除行首的引用标记 "> "（markdown 格式）
        delta_content = re.sub(r'^>\s*', '', delta_content, flags=re.MULTILINE)
        delta_content = re.sub(r'\n>\s*', '\n', delta_content)
        
        # 5. 移除多余的空行（3个及以上连续换行符）
        delta_content = re.sub(r'\n{3,}', '\n\n', delta_content)
        
        # 6. 去除首尾空白
        return delta_content.strip()

    def _split_edit_content(self, edit_content: str) -> Tuple[str, str]:
        """拆分 edit_content，返回 (thinking_part, answer_part)
        
        处理格式：
        <details type="reasoning" done="false/true" ...>
        <summary>Thinking...</summary>
        > 思考内容
        </details>
        回答内容
        """
        if not edit_content:
            return "", ""

        thinking_part = ""
        answer_part = ""

        # 查找 </details> 标签位置
        if "</details>" in edit_content:
            # 分割 thinking 和 answer 部分
            parts = edit_content.split("</details>", 1)
            thinking_part = parts[0] + "</details>"  # 保留完整的 details 标签用于清理
            answer_part = parts[1] if len(parts) > 1 else ""
        else:
            # 没有 details 标签，整个内容当作答案
            answer_part = edit_content

        # 清理 thinking 内容（移除标签，保留纯文本）
        if thinking_part:
            thinking_part = self._clean_thinking(thinking_part)
        
        # 清理 answer 内容（移除可能的标签）
        answer_part = answer_part.strip()
        if answer_part:
            # 移除开头的换行符
            answer_part = answer_part.lstrip('\n')
            # 移除可能包含的 think 标签
            answer_part = answer_part.replace("<think>", "").replace("</think>", "")
        
        return thinking_part, answer_part

    def _diff_new_content(self, existing: str, incoming: str) -> str:
        """计算 incoming 相比 existing 的新增部分（用于流式增量输出）"""
        incoming = incoming or ""
        if not incoming:
            return ""

        existing = existing or ""
        if not existing:
            return incoming

        if incoming == existing:
            return ""

        # 如果 incoming 是 existing 的扩展，返回新增部分
        if incoming.startswith(existing):
            return incoming[len(existing):]

        # 寻找最长公共前缀以计算增量
        max_overlap = min(len(existing), len(incoming))
        for overlap in range(max_overlap, 0, -1):
            if existing[-overlap:] == incoming[:overlap]:
                return incoming[overlap:]

        # 如果 existing 完全包含在 incoming 中
        if existing in incoming:
            return incoming.replace(existing, "", 1)

        # 无法确定增量，返回完整内容
        return incoming


chat_completion_service = ChatCompletionService()

