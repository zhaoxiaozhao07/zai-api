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
from ..toolify.detector import StreamingFunctionCallDetector
from ..toolify_handler import (
    should_enable_toolify,
    prepare_toolify_request,
    parse_toolify_response,
    format_toolify_response_for_stream,
)
from ..toolify_config import get_toolify
from ..zai_transformer import ZAITransformer
from ..token_pool import get_token_pool
from .network_manager import network_manager


class ChatCompletionService:
    """Encapsulate chat completion workflow independent of FastAPI layer."""

    def __init__(self) -> None:
        self.transformer = ZAITransformer()

    async def prepare_request(self, request: OpenAIRequest) -> Tuple[dict, dict, bool]:
        request_dict = request.model_dump()
        return self._prepare_messages(request, request_dict)

    def _prepare_messages(self, request: OpenAIRequest, request_dict: dict) -> Tuple[dict, dict, bool]:
        enable_toolify = should_enable_toolify(request_dict)
        messages = [
            msg.model_dump() if hasattr(msg, "model_dump") else msg
            for msg in request.messages
        ]

        if enable_toolify:
            info_log("[TOOLIFY] 工具调用功能已启用")
            messages, _ = prepare_toolify_request(request_dict, messages)
            transformed_dict = request_dict.copy()
            transformed_dict.pop("tools", None)
            transformed_dict.pop("tool_choice", None)
            transformed_dict["messages"] = messages
        else:
            transformed_dict = request_dict

        return request_dict, transformed_dict, enable_toolify

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
        enable_toolify: bool,
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

                    final_content = ""
                    reasoning_content = ""

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

                        if data.get("usage"):
                            try:
                                usage_info = data["usage"]
                            except Exception:  # pragma: no cover
                                pass

                        if phase == "tool_call":
                            reasoning_content = self._extract_search_info(reasoning_content, edit_content)
                            continue

                        if phase == "thinking" and delta_content:
                            reasoning_content += self._clean_thinking(delta_content)
                        elif phase == "answer":
                            final_content += self._extract_answer(delta_content, edit_content)

                    final_content = (final_content or "").strip()
                    reasoning_content = (reasoning_content or "").strip()

                    # 清理上游可能自带的think标签（避免重复）
                    reasoning_content = reasoning_content.replace("<think>", "").replace("</think>", "")
                    final_content = final_content.replace("<think>", "").replace("</think>", "")

                    if enable_toolify and final_content:
                        debug_log("[TOOLIFY] 检查非流式响应中的工具调用")
                        tool_response = parse_toolify_response(final_content, request.model)
                        if tool_response:
                            info_log("[TOOLIFY] 非流式响应中检测到工具调用")
                            request_stage_log(
                                "non_stream_toolify",
                                "非流式响应中检测到工具调用",
                                finish_reason="tool_calls",
                            )
                            return {
                                "id": transformed["body"]["chat_id"],
                                "object": "chat.completion",
                                "created": int(time.time()),
                                "model": request.model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "message": tool_response,
                                        "finish_reason": "tool_calls",
                                    }
                                ],
                                "usage": usage_info,
                            }

                    request_stage_log(
                        "non_stream_completed",
                        "非流式响应完成",
                        completion_tokens=usage_info.get("completion_tokens"),
                        prompt_tokens=usage_info.get("prompt_tokens"),
                    )

                    # 构建消息对象，将 reasoning_content 作为单独的字段
                    message = {
                        "role": "assistant",
                        "content": final_content,
                    }
                    # 只有当存在 reasoning_content 时才添加该字段
                    if reasoning_content:
                        message["reasoning_content"] = reasoning_content

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
        enable_toolify: bool,
    ) -> AsyncIterator[str]:
        bind_request_context(mode="stream")
        request_stage_log("stream_pipeline", "进入流式处理流程")
        retry_count = 0
        last_error = None
        last_status_code = None

        toolify_detector = None
        if enable_toolify:
            toolify_instance = get_toolify()
            if toolify_instance:
                toolify_detector = StreamingFunctionCallDetector(toolify_instance.trigger_signal)
                debug_log("[TOOLIFY] 流式工具调用检测器已初始化")

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

                    async for line in response.aiter_lines():
                        if not line or not line.strip():
                            continue

                        if not line.startswith("data:"):
                            continue

                        chunk_str = line[5:].strip()
                        if not chunk_str or chunk_str == "[DONE]":
                            if chunk_str == "[DONE]" and toolify_detector:
                                parsed_tools, remaining_content = toolify_detector.finalize()
                                if remaining_content:
                                    # 清理可能包含的think标签
                                    remaining_content = remaining_content.replace("<think>", "").replace("</think>", "")
                                    
                                    if remaining_content:
                                        if not has_thinking:
                                            has_thinking = True
                                            yield self._build_role_chunk(json_lib, transformed, request)
                                        yield self._build_content_chunk(json_lib, transformed, request, remaining_content)

                                if parsed_tools:
                                    for chunk in format_toolify_response_for_stream(
                                        parsed_tools,
                                        request.model,
                                        transformed["body"]["chat_id"],
                                    ):
                                        yield chunk
                                    request_stage_log(
                                        "stream_toolify_completed",
                                        "流式响应（早期工具调用检测）完成",
                                    )
                                    return

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
                        edit_index = data.get("edit_index")
                        phase = data.get("phase")
                        is_done = phase == "done" or data.get("done")
                        error_info = data.get("error")

                        # 详细调试：记录每个chunk的原始数据
                        debug_log(f"[RAW_CHUNK] phase={phase}, delta={bool(delta_content)}, edit={bool(edit_content)}, edit_index={edit_index}, usage={bool(data.get('usage'))}, done={data.get('done')}, error={bool(error_info)}")
                        if delta_content:
                            debug_log(f"[RAW_DELTA] len={len(delta_content)}, content={delta_content[:100]}")
                        if edit_content:
                            edit_len = len(edit_content)
                            edit_preview = edit_content[:200] if edit_len <= 500 else f"{edit_content[:100]}...{edit_content[-100:]}"
                            debug_log(f"[RAW_EDIT] len={edit_len}, content={edit_preview}")
                            # 对于包含 edit_content 的关键 chunk，记录完整 JSON
                            if edit_len > 10 or is_done:
                                debug_log(f"[RAW_JSON] {chunk_str[:1000]}")

                        # 检测上游返回的错误（如内容安全警告）
                        if error_info:
                            error_detail = error_info.get("detail") or error_info.get("content") or "Unknown error"
                            error_log(f"[UPSTREAM_ERROR] 上游返回错误: {error_detail}")
                            
                            # 如果还没有发送任何内容，发送错误信息
                            if not has_thinking:
                                has_thinking = True
                                yield self._build_role_chunk(json_lib, transformed, request)
                            
                            # 发送错误提示给客户端
                            error_message = f"\n\n[系统提示: {error_detail}]"
                            yield self._build_content_chunk(json_lib, transformed, request, error_message)
                            
                            # 如果同时标记为 done，结束流
                            if is_done:
                                finish_chunk = self._build_finish_chunk(json_lib, transformed, request)
                                yield finish_chunk
                                yield "data: [DONE]\n\n"
                                await self._mark_token_success(transformed)
                                request_stage_log(
                                    "stream_completed",
                                    "流式响应完成（带错误）",
                                    has_error=True,
                                )
                                return
                            continue

                        # 调试日志：记录phase和是否有内容
                        if delta_content or edit_content:
                            debug_log(f"[PHASE] phase={phase}, has_delta={bool(delta_content)}, has_edit={bool(edit_content)}")

                        # 处理thinking阶段：通过 reasoning_content 字段流式输出
                        if phase == "thinking":
                            if delta_content:
                                if not has_thinking:
                                    has_thinking = True
                                    yield self._build_role_chunk(json_lib, transformed, request)
                                
                                # 清理thinking内容中的HTML标记、引用符号和think标签
                                cleaned_content = self._clean_thinking(delta_content)
                                
                                # 通过 reasoning_content 字段输出
                                if cleaned_content:
                                    yield self._build_reasoning_chunk(json_lib, transformed, request, cleaned_content)
                            
                            # 检查 edit_content 是否包含完整的 thinking + answer
                            if edit_content and "</details>" in edit_content:
                                debug_log("[THINKING_EDIT] 检测到 edit_content 包含 </details>，可能包含 answer")
                                # 提取 </details> 后的内容作为 answer
                                answer_content = edit_content.split("</details>")[-1].strip()
                                if answer_content:
                                    # 清理可能的 think 标签
                                    answer_content = answer_content.replace("<think>", "").replace("</think>", "")
                                    if answer_content:
                                        # 输出 answer 内容
                                        if not has_thinking:
                                            has_thinking = True
                                            yield self._build_role_chunk(json_lib, transformed, request)
                                        
                                        yield self._build_content_chunk(json_lib, transformed, request, answer_content)
                                        debug_log(f"[THINKING_EDIT] 输出 answer 内容: {answer_content[:50]}...")
                            
                            continue

                        # 清理非thinking阶段内容中可能自带的think标签（避免重复）
                        if delta_content:
                            # 移除上游返回的<think>和</think>标签
                            delta_content = delta_content.replace("<think>", "").replace("</think>", "")
                            if not delta_content:  # 如果清理后为空，跳过
                                continue

                        # 跳过tool_call阶段的内容
                        if phase == "tool_call":
                            continue

                        # 在answer阶段，处理edit_content（可能包含完整thinking + answer开头）
                        if phase == "answer" and edit_content and not delta_content:
                            # 这个chunk只有edit_content，可能包含完整thinking + answer开头
                            # 提取</details>后的内容作为answer开头
                            if "</details>" in edit_content:
                                answer_start = edit_content.split("</details>")[-1].strip()
                                if answer_start:
                                    # 清理可能的think标签
                                    answer_start = answer_start.replace("<think>", "").replace("</think>", "")
                                    if answer_start:
                                        delta_content = answer_start
                                        debug_log(f"[EDIT_CONTENT] 从edit_content提取answer开头: {answer_start[:50]}...")
                            # 如果没有</details>或提取失败，跳过这个chunk
                            if not delta_content:
                                continue

                        if enable_toolify and toolify_detector:
                            yielded, should_continue, processed, has_thinking = self._process_toolify_detection(
                                toolify_detector,
                                delta_content,
                                has_thinking,
                                transformed,
                                request,
                                json_lib,
                            )
                            for chunk in yielded:
                                yield chunk
                            if should_continue:
                                continue
                            # 清理处理后的内容中可能包含的think标签
                            if processed:
                                processed = processed.replace("<think>", "").replace("</think>", "")
                            delta_content = processed

                        # 输出answer阶段的内容
                        if delta_content and phase == "answer":
                            if not has_thinking:
                                has_thinking = True
                                yield self._build_role_chunk(json_lib, transformed, request)

                            yield self._build_content_chunk(json_lib, transformed, request, delta_content)

                        # 处理 phase=other 时的 edit_content（可能包含最后一段答案）
                        if phase == "other" and edit_content:
                            # 清理可能的think标签
                            cleaned_edit = edit_content.replace("<think>", "").replace("</think>", "")
                            if cleaned_edit:
                                if not has_thinking:
                                    has_thinking = True
                                    yield self._build_role_chunk(json_lib, transformed, request)
                                
                                yield self._build_content_chunk(json_lib, transformed, request, cleaned_edit)
                                debug_log(f"[OTHER] 输出 phase=other 的 edit_content: {cleaned_edit[:50]}...")

                        if data.get("usage"):
                            yield self._build_usage_chunk(json_lib, transformed, request, data["usage"])

                        # 处理完当前 chunk 的所有内容后，检查是否为 done 状态
                        if is_done:
                            debug_log("[DONE] 检测到 done 标志，流结束")
                            
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
        if current_token and not token_pool.is_anonymous_token(current_token):
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
        is_anonymous = token_pool.is_anonymous_token(current_token)

        if is_anonymous:
            info_log(f"[ANONYMOUS] 检测到匿名Token错误 {status_code}，清理缓存并重新获取")
            await self.transformer.clear_anonymous_token_cache()
            await self.transformer.refresh_header_template()
            await network_manager.cleanup_current_client(current_proxy)
            new_client, new_proxy = await network_manager.get_request_client()
            request_client = new_client
            current_proxy = new_proxy

            if network_manager.has_upstream_pool() and network_manager.upstream_strategy == "failover":
                await network_manager.switch_upstream_on_failure()
                info_log("[FAILOVER] 匿名Token错误，尝试切换上游地址")

            current_upstream = await network_manager.get_next_upstream()
            transformed = await self.transformer.transform_request_in(
                request_dict_for_transform,
                client=request_client,
                upstream_url=current_upstream,
            )
            info_log("[OK] 已获取新的匿名Token并重新生成请求")
        else:
            token_pool.mark_token_failure(current_token)
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

    def _process_toolify_detection(
        self,
        toolify_detector,
        delta_content: str,
        has_thinking: bool,
        transformed: dict,
        request: OpenAIRequest,
        json_lib,
    ) -> Tuple[list, bool, str, bool]:
        chunks_to_yield = []

        if not toolify_detector or not delta_content:
            return chunks_to_yield, False, delta_content, has_thinking

        debug_log("[TOOLIFY] 调用工具检测器")
        is_tool_detected, content_to_yield = toolify_detector.process_chunk(delta_content)

        if is_tool_detected:
            if content_to_yield:
                if not has_thinking:
                    has_thinking = True
                    chunks_to_yield.append(self._build_role_chunk(json_lib, transformed, request))
                chunks_to_yield.append(
                    self._build_content_chunk(json_lib, transformed, request, content_to_yield)
                )

            return chunks_to_yield, True, "", has_thinking

        return chunks_to_yield, False, content_to_yield, has_thinking

    async def mark_token_success_if_configured(self, transformed: dict) -> None:
        token_pool = await get_token_pool()
        current_token = transformed.get("token", "")
        if current_token and not token_pool.is_anonymous_token(current_token):
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
                        debug_log("[非流式] 提取到搜索信息", queries=queries)
            except Exception as exc:
                debug_log("[非流式] 提取搜索信息失败", error=str(exc))
        return reasoning_content

    def _clean_thinking(self, delta_content: str) -> str:
        import re
        
        # 清理details标签的开头
        if delta_content.startswith("<details"):
            if "</summary>" in delta_content:
                # 提取</summary>后的内容
                delta_content = delta_content.split("</summary>")[-1].strip()
        
        # 移除可能出现的summary标签
        delta_content = re.sub(r'<summary[^>]*>.*?</summary>', '', delta_content, flags=re.DOTALL)
        
        # 移除details标签
        delta_content = re.sub(r'</?details[^>]*>', '', delta_content)
        
        # 移除引用标记 "> " (markdown引用格式)
        delta_content = re.sub(r'^>\s*', '', delta_content, flags=re.MULTILINE)
        delta_content = re.sub(r'\n>\s*', '\n', delta_content)
        
        # 移除多余的换行符
        delta_content = re.sub(r'\n{3,}', '\n\n', delta_content)
        
        return delta_content.strip()

    def _extract_answer(self, delta_content: str, edit_content: str) -> str:
        result = ""
        if edit_content and "</details>\n" in edit_content:
            content_after = edit_content.split("</details>\n")[-1]
            if content_after:
                result = content_after
        else:
            result = delta_content or ""
        
        return result


chat_completion_service = ChatCompletionService()

