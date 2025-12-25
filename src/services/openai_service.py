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
from .network_manager import network_manager
from .response_parser import response_parser
from .chunk_builder import chunk_builder


class ChatCompletionService:
    """Encapsulate chat completion workflow independent of FastAPI layer."""

    def __init__(self) -> None:
        self.transformer = ZAITransformer()
        # 使用拆分后的解析器和构建器
        self.parser = response_parser
        self.chunk = chunk_builder

    async def prepare_request(self, request: OpenAIRequest) -> Tuple[dict, dict]:
        """准备请求数据。"""
        request_dict = request.model_dump()
        return request_dict, request_dict

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
                    
                    # 检查是否为thinking模型，非thinking模型不返回 reasoning_content
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
                                image_urls = self.parser.extract_image_urls(edit_content)
                                if image_urls:
                                    # 有图片，转换为markdown格式放入 answer_content
                                    markdown_images = self.parser.format_images_as_markdown(image_urls)
                                    if markdown_images:
                                        answer_content += "\n\n" + markdown_images + "\n\n"
                                else:
                                    # 无图片，放入 thinking_content
                                    thinking_content += edit_content
                            continue

                        # thinking 阶段：清洗后累积 delta_content
                        if phase == "thinking":
                            if delta_content:
                                cleaned = self.parser.clean_thinking(delta_content)
                                if cleaned:
                                    thinking_content += cleaned
                            continue


                        # answer 阶段：处理 edit_content 和 delta_content
                        # 关键区分：
                        # - 带 edit_index + edit_content：思考内容的完整版（替换之前的增量）
                        # - delta_content：实际正文内容（无论是否有 edit_index 都应累积）
                        if phase == "answer":
                            has_edit_index = data.get("edit_index") is not None
                            
                            # 带 edit_index 的 edit_content 是思考内容的完整版
                            if has_edit_index and edit_content:
                                # 先提取图片 URL（如有）
                                image_urls = self.parser.extract_image_urls(edit_content)
                                if image_urls:
                                    markdown_images = self.parser.format_images_as_markdown(image_urls)
                                    if markdown_images:
                                        answer_content += "\n\n" + markdown_images + "\n\n"
                                
                                # 记录完整思考内容（用于后续处理）
                                if "</details>" in edit_content:
                                    latest_full_edit = edit_content
                                else:
                                    # 清洗并累积思考内容
                                    cleaned = self.parser.clean_thinking(edit_content)
                                    if cleaned:
                                        new_thinking = self.parser.diff_new_content(thinking_content, cleaned)
                                        if new_thinking:
                                            thinking_content += new_thinking
                            
                            # delta_content 是正文内容（独立处理，不使用 elif）
                            # 上游可能在同一数据包中同时返回 edit_content 和 delta_content
                            if delta_content:
                                answer_content += delta_content
                            continue


                        # other 阶段：可能有 usage 信息，也可能有最后的 edit_content 片段
                        # 注意：other 阶段的内容通常是正文的结尾部分，应放入 answer_content
                        if phase == "other":
                            # 收集 usage
                            if data.get("usage"):
                                latest_usage = data["usage"]

                            # 获取内容（优先使用 delta_content，因为它是增量正文）
                            tail_text = None
                            if delta_content:
                                tail_text = delta_content
                            elif edit_content:
                                # edit_content 可能包含格式标签，需要清理
                                tail_text = self.parser.clean_thinking(edit_content)

                            if tail_text:
                                # 先检测是否有图片
                                image_urls = self.parser.extract_image_urls(tail_text)
                                if image_urls:
                                    markdown_images = self.parser.format_images_as_markdown(image_urls)
                                    if markdown_images:
                                        answer_content += "\n\n" + markdown_images + "\n\n"
                                else:
                                    # other 阶段的内容都放入 answer_content（因为是正文结尾）
                                    answer_content += tail_text
                            continue

                    # 如果上游在 answer/other 阶段给出了完整的 edit_content（含 </details>），
                    # 则优先用它来拆分出 reasoning_content 和 最终回答，避免仅依赖增量导致内容不完整或截断
                    # 注意：latest_full_edit 中可能不包含 <details> 开头（例如只剩属性残片），
                    # 这里统一交给 _split_edit_content + _clean_thinking 做清洗，移除 true" duration="1" 等残留。
                    if latest_full_edit:
                        thinking_part, answer_part = self.parser.split_edit_content(latest_full_edit)
                        if thinking_part:
                            thinking_content = thinking_part
                        # 对于 answer_part，如果已经通过 delta_content 累积了内容，
                        # 则保留累积内容，因为 delta_content 通常更完整
                        # 只有当 answer_content 为空时才使用解析出的 answer_part
                        if answer_part and not answer_content:
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
                    # 对于非thinking模型，不返回 reasoning_content，thinking内容已在累积过程中处理
                    # 注意：不再将 thinking_content 合并到 answer_content，因为非 thinking 模型
                    # 的 thinking 阶段内容应该被忽略或已在累积过程中正确处理
                    
                    message = {
                        "role": "assistant",
                        "content": answer_content,
                    }
                    # 只有thinking模型才添加 reasoning_content 字段
                    if thinking_content and is_thinking_model:
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
                    
                    # 累积 usage 信息，用于最终的 finish chunk
                    latest_usage = None
                    
                    # 检查是否为thinking模型，非thinking模型不输出 reasoning_content
                    is_thinking_model = transformed.get("is_thinking", False)
                    # 检查是否为 V 系列视觉模型（4.5v/4.6v），用于区分多阶段思考格式
                    is_vision_model = transformed.get("is_vision_model", False)

                    # V系列输出响应日志辅助函数
                    def _log_v_output(output_type: str, content: str):
                        """记录V系列模型输出给客户端的内容（写入文件，仅debug模式）"""
                        if is_vision_model and content:
                            v_series_debug_log(
                                "V系列输出响应",
                                output_type=output_type,
                                content=content,  # 保留完整内容
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

                        # V系列模型上游响应调试日志（写入文件）
                        if is_vision_model:
                            v_series_debug_log(
                                "V系列上游响应",
                                phase=phase,
                                delta_content=delta_content,  # 保留完整内容
                                edit_content=edit_content,    # 保留完整内容
                                has_edit_index=data.get("edit_index") is not None,
                                edit_index=data.get("edit_index"),
                            )

                        # 检测上游返回的错误（如内容安全警告）
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

                        # tool_call 阶段：如果带有 edit_index，优先检测图片，否则放入思考
                        if phase == "tool_call":
                            if data.get("edit_index") is not None and edit_content:
                                if not has_thinking:
                                    has_thinking = True
                                    yield self.chunk.build_role_chunk(json_lib, transformed, request)
                                
                                # 先检测是否有图片（包括 image_reference 工具返回的图片）
                                image_urls = self.parser.extract_image_urls(edit_content)
                                if image_urls:
                                    # 有图片，转换为markdown格式放入正文
                                    markdown_images = self.parser.format_images_as_markdown(image_urls)
                                    if markdown_images:
                                        yield self.chunk.build_content_chunk(json_lib, transformed, request, "\n\n" + markdown_images + "\n\n")
                                        answer_accumulator += markdown_images
                                
                                # 检测是否包含 <glm_block> 标签
                                # 如果包含，需要拆分：glm_block 前的内容是正文，glm_block 本身是工具调用
                                if "<glm_block" in edit_content:
                                    # 拆分 glm_block 前的内容（正文）和 glm_block 本身
                                    glm_block_match = re.search(r'<glm_block[^>]*>.*?</glm_block>', edit_content, re.DOTALL)
                                    if glm_block_match:
                                        # glm_block 前的内容是正文
                                        before_glm_block = edit_content[:glm_block_match.start()].strip()
                                        if before_glm_block:
                                            new_answer = self.parser.diff_new_content(answer_accumulator, before_glm_block)
                                            if new_answer:
                                                _log_v_output("tool_call_content", new_answer)
                                                yield self.chunk.build_content_chunk(json_lib, transformed, request, new_answer)
                                                answer_accumulator += new_answer
                                        # glm_block 本身不输出（工具调用细节）
                                else:
                                    # 没有 glm_block，整个内容是正文的追加
                                    cleaned = self.parser.clean_thinking(edit_content)
                                    if cleaned:
                                        new_answer = self.parser.diff_new_content(answer_accumulator, cleaned)
                                        if new_answer:
                                            _log_v_output("tool_call_plain_content", new_answer)
                                            yield self.chunk.build_content_chunk(json_lib, transformed, request, new_answer)
                                            answer_accumulator += new_answer
                            continue


                        # thinking 阶段处理
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
                                    # thinking模型：流式输出 reasoning_content（使用增量 diff，防止覆盖式 delta 截断）
                                    # 先做"原样增量"：直接把本次 delta 里的新增部分先输出
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

                                    # 再做一次基于清洗后的兜底增量，防止上游覆盖式 delta 导致遗漏
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
                                            # 这里不再修改 thinking_accumulator，避免与原始增量状态不一致
                            continue

                        # answer 阶段：处理 edit_content 和 delta_content
                        # 关键区分：
                        # - 带 edit_index + edit_content：思考内容的完整版（替换之前的增量），应放入 reasoning_content
                        # - 不带 edit_index + delta_content：实际正文内容，应放入 content
                        if phase == "answer":
                            has_edit_index = data.get("edit_index") is not None
                            
                            # 带 edit_index 的 edit_content 可能包含 思考+正文 混合内容（V 系列模型）
                            # 或者只是正文的追加内容（thinking 系列模型）
                            # 通过检测 </details> 标签来区分
                            if has_edit_index and edit_content:
                                if not has_thinking:
                                    has_thinking = True
                                    yield self.chunk.build_role_chunk(json_lib, transformed, request)
                                
                                # 先提取图片 URL（如有）
                                image_urls = self.parser.extract_image_urls(edit_content)
                                if image_urls:
                                    markdown_images = self.parser.format_images_as_markdown(image_urls)
                                    if markdown_images:
                                        yield self.chunk.build_content_chunk(json_lib, transformed, request, "\n\n" + markdown_images + "\n\n")
                                        answer_accumulator += markdown_images
                                
                                # 只有包含 </details> 标签时才拆分思考和正文（V 系列模型的多阶段思考）
                                if "</details>" in edit_content:
                                    thinking_part, answer_part = self.parser.split_edit_content(edit_content)
                                    
                                    # 处理思考部分
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
                                    
                                    # 处理正文部分
                                    if answer_part:
                                        new_answer = self.parser.diff_new_content(answer_accumulator, answer_part)
                                        if new_answer:
                                            _log_v_output("answer_content", new_answer)
                                            yield self.chunk.build_content_chunk(json_lib, transformed, request, new_answer)
                                            answer_accumulator += new_answer
                                else:
                                    # 没有 details 标签，整个内容当作正文处理（thinking 系列模型）
                                    cleaned = self.parser.clean_thinking(edit_content)  # 清理可能的其他标签
                                    if cleaned:
                                        new_answer = self.parser.diff_new_content(answer_accumulator, cleaned)
                                        if new_answer:
                                            _log_v_output("answer_plain_content", new_answer)
                                            yield self.chunk.build_content_chunk(json_lib, transformed, request, new_answer)
                                            answer_accumulator += new_answer
                                continue


                            
                            # 不带 edit_index 的 delta_content 是正文内容
                            if delta_content:
                                if not has_thinking:
                                    has_thinking = True
                                    yield self.chunk.build_role_chunk(json_lib, transformed, request)
                                
                                _log_v_output("delta_content", delta_content)
                                yield self.chunk.build_content_chunk(json_lib, transformed, request, delta_content)
                                answer_accumulator += delta_content
                            continue




                        # other 阶段：可能有 usage 信息，也可能携带正文的最后一小段（edit_content 或 delta_content）
                        # 如果带有 edit_index，说明是工具调用相关内容，应放入思考
                        if phase == "other":
                            # 1) 先累积 usage 信息（稍后在 finish chunk 之后输出）
                            if data.get("usage"):
                                latest_usage = data["usage"]

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
                                    yield self.chunk.build_role_chunk(json_lib, transformed, request)
                                
                                if has_edit_index:
                                    # 带 edit_index 的内容：先检测是否有图片
                                    image_urls = self.parser.extract_image_urls(tail_text)
                                    if image_urls:
                                        # 有图片，转换为markdown格式放入正文
                                        markdown_images = self.parser.format_images_as_markdown(image_urls)
                                        if markdown_images:
                                            yield self.chunk.build_content_chunk(json_lib, transformed, request, "\n\n" + markdown_images + "\n\n")
                                            answer_accumulator += markdown_images
                                    else:
                                        # 无图片的内容处理
                                        # phase="other" + has_edit_index 的内容是正文末尾的补充（如最后几个字）
                                        # 不论是 V 系列还是 thinking 系列，都应输出到正文 content
                                        # 注意：不使用 diff 计算，因为这些是纯增量内容，直接输出即可
                                        cleaned = self.parser.clean_thinking(tail_text)
                                        if cleaned:
                                            _log_v_output("other_tail_content", cleaned)
                                            yield self.chunk.build_content_chunk(json_lib, transformed, request, cleaned)
                                            answer_accumulator += cleaned


                                else:
                                    # 普通内容放入 content（正文）
                                    _log_v_output("other_content", tail_text)
                                    yield self.chunk.build_content_chunk(json_lib, transformed, request, tail_text)
                            continue

                        # 累积 usage 信息（用于最终输出）
                        if data.get("usage"):
                            latest_usage = data["usage"]

                        # 检查是否为 done 状态
                        if is_done:
                            # 1. 发送带 usage 的 finish chunk
                            finish_chunk = self.chunk.build_finish_chunk(json_lib, transformed, request, usage=latest_usage)
                            yield finish_chunk
                            # 2. 如果有 usage 信息，发送独立的 usage chunk
                            if latest_usage:
                                yield self.chunk.build_usage_chunk(json_lib, transformed, request, latest_usage)
                            # 3. 发送 [DONE]
                            yield "data: [DONE]\n\n"
                            await self._mark_token_success(transformed)
                            request_stage_log("stream_completed", "流式响应完成", has_error=False)
                            return

                    # 流正常结束，发送 finish chunk 和 usage chunk
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

        # 标记Token失败并切换
        if current_token:
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

    async def mark_token_success_if_configured(self, transformed: dict) -> None:
        token_pool = await get_token_pool()
        current_token = transformed.get("token", "")
        if current_token:
            token_pool.mark_token_success(current_token)



chat_completion_service = ChatCompletionService()


