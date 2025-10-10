"""
OpenAI API endpoints - 优化版本
"""

import time
import json
import random
import asyncio
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse
import httpx

from .config import settings
from .schemas import OpenAIRequest, ModelsResponse, Model
from .helpers import debug_log, get_logger
from .zai_transformer import ZAITransformer

router = APIRouter()

# 获取结构化日志记录器
logger = get_logger("openai_api")

# 全局转换器实例
transformer = ZAITransformer()

# 全局httpx客户端配置（连接池复用）
_http_client = None
_client_lock = asyncio.Lock()


async def get_http_client() -> httpx.AsyncClient:
    """
    获取或创建全局httpx客户端（单例模式，支持连接池复用）
    使用asyncio特性进行并发控制
    """
    global _http_client
    
    async with _client_lock:
        if _http_client is None or _http_client.is_closed:
            proxy = settings.HTTPS_PROXY or settings.HTTP_PROXY
            
            # 配置连接池和超时
            limits = httpx.Limits(
                max_connections=100,  # 最大连接数
                max_keepalive_connections=20,  # 保持活动的连接数
                keepalive_expiry=30.0  # 保持活动超时时间
            )
            
            timeout = httpx.Timeout(
                connect=10.0,  # 连接超时
                read=60.0,     # 读取超时
                write=10.0,    # 写入超时
                pool=5.0       # 连接池超时
            )
            
            _http_client = httpx.AsyncClient(
                timeout=timeout,
                limits=limits,
                proxy=proxy,
                http2=True,  # 启用HTTP/2支持
                follow_redirects=True
            )
            
            if proxy:
                debug_log("使用代理创建HTTP客户端", proxy=proxy)
        
        return _http_client


async def close_http_client():
    """关闭全局httpx客户端"""
    global _http_client
    
    async with _client_lock:
        if _http_client is not None and not _http_client.is_closed:
            await _http_client.aclose()
            _http_client = None
            debug_log("HTTP客户端已关闭")


def calculate_backoff_delay(retry_count: int, status_code: int = None, base_delay: float = 3.0, max_delay: float = 30.0) -> float:
    """
    计算退避延迟时间（带随机抖动和针对不同错误的策略）
    
    Args:
        retry_count: 当前重试次数（从1开始）
        status_code: HTTP状态码，用于区分不同错误类型
        base_delay: 基础延迟时间（秒）
        max_delay: 最大延迟时间（秒）
    
    Returns:
        计算后的延迟时间（秒）
    """
    # 基础指数退避：base_delay * 2^(retry_count-1)
    exponential_delay = base_delay * (2 ** (retry_count - 1))
    
    # 针对不同错误类型调整延迟
    if status_code == 429:  # Rate limit - 更长的退避时间
        exponential_delay *= 2.0  # 双倍延迟
        debug_log("[BACKOFF] 检测到429限流错误，使用更长退避时间")
    elif status_code in [502, 503, 504]:  # 服务端错误 - 适中延迟
        exponential_delay *= 1.5
        debug_log("[BACKOFF] 检测到服务端错误，使用适中退避时间")
    elif status_code in [400, 401]:  # 认证/请求错误 - 标准延迟
        debug_log("[BACKOFF] 检测到认证/请求错误，使用标准退避时间")
    
    # 限制最大延迟
    exponential_delay = min(exponential_delay, max_delay)
    
    # 添加随机抖动因子（±25%），避免多个请求同时重试造成雪崩
    jitter = exponential_delay * 0.25  # 25%的抖动范围
    jittered_delay = exponential_delay + random.uniform(-jitter, jitter)
    
    # 确保延迟不会小于基础延迟的一半
    final_delay = max(jittered_delay, base_delay * 0.5)
    
    debug_log(
        "[BACKOFF] 计算退避延迟",
        retry_count=retry_count,
        status_code=status_code,
        exponential=f"{exponential_delay:.2f}s",
        final=f"{final_delay:.2f}s"
    )
    
    return final_delay


@router.get("/v1/models")
async def list_models():
    """List available models"""
    current_time = int(time.time())
    response = ModelsResponse(
        data=[
            Model(id=settings.PRIMARY_MODEL, created=current_time, owned_by="z.ai"),
            Model(id=settings.THINKING_MODEL, created=current_time, owned_by="z.ai"),
            Model(id=settings.SEARCH_MODEL, created=current_time, owned_by="z.ai"),
            Model(id=settings.AIR_MODEL, created=current_time, owned_by="z.ai"),
            Model(id=settings.GLM_46_MODEL, created=current_time, owned_by="z.ai"),
            Model(id=settings.GLM_46_THINKING_MODEL, created=current_time, owned_by="z.ai"),
        ]
    )
    return response


@router.post("/v1/chat/completions")
async def chat_completions(request: OpenAIRequest, authorization: str = Header(...)):
    """Handle chat completion requests with ZAI transformer - 支持流式和非流式"""
    role = request.messages[0].role if request.messages else "unknown"
    debug_log(
        "收到客户端请求",
        model=request.model,
        stream=request.stream,
        message_count=len(request.messages)
    )
    
    try:
        # Validate API key
        if not settings.SKIP_AUTH_TOKEN:
            if not authorization.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
            
            api_key = authorization[7:]
            if api_key != settings.AUTH_TOKEN:
                raise HTTPException(status_code=401, detail="Invalid API key")
            
        # 转换请求
        request_dict = request.model_dump()
        debug_log("开始转换请求格式: OpenAI -> Z.AI")
        
        # 初始转换
        transformed = await transformer.transform_request_in(request_dict)
        
        # 根据stream参数决定返回流式或非流式响应
        if not request.stream:
            debug_log("使用非流式模式")
            return await handle_non_stream_request(request, transformed)

        # 调用上游API
        async def stream_response():
            """流式响应生成器（包含重试机制和智能退避策略）"""
            nonlocal transformed  # 声明使用外部作用域的transformed变量
            retry_count = 0
            last_error = None
            last_status_code = None  # 记录上次失败的状态码

            while retry_count <= settings.MAX_RETRIES:
                try:
                    # 智能退避重试策略（带随机抖动和错误类型区分）
                    if retry_count > 0:
                        # 使用智能退避策略计算延迟
                        delay = calculate_backoff_delay(
                            retry_count=retry_count,
                            status_code=last_status_code,
                            base_delay=3.0,
                            max_delay=40.0
                        )
                        debug_log(
                            f"[RETRY] 重试请求 ({retry_count}/{settings.MAX_RETRIES})",
                            retry_count=retry_count,
                            max_retries=settings.MAX_RETRIES,
                            delay=f"{delay:.2f}s",
                            last_status=last_status_code
                        )
                        await asyncio.sleep(delay)

                    # 获取全局HTTP客户端（复用连接池）
                    client = await get_http_client()
                    
                    # 发起流式请求
                    async with client.stream(
                        "POST",
                        transformed["config"]["url"],
                        json=transformed["body"],
                        headers=transformed["config"]["headers"],
                    ) as response:
                        # 检查响应状态码
                        if response.status_code != 200:
                            error_text = await response.aread()
                            error_msg = error_text.decode('utf-8', errors='ignore')
                            debug_log(
                                "上游返回错误",
                                status_code=response.status_code,
                                error_detail=error_msg[:200]
                            )
                            
                            # 可重试的错误
                            retryable_codes = [400, 401, 429, 502, 503, 504]
                            if response.status_code in retryable_codes and retry_count < settings.MAX_RETRIES:
                                # 记录状态码和错误信息，用于智能退避策略
                                last_status_code = response.status_code
                                last_error = f"{response.status_code}: {error_msg}"
                                retry_count += 1
                                
                                # 如果是401认证失败或400错误，尝试切换token
                                if response.status_code in [400, 401]:
                                    debug_log("[SWITCH] 检测到认证/请求错误，尝试切换token")
                                    new_token = transformer.switch_token()
                                    # 使用新token重新生成请求
                                    transformed = await transformer.transform_request_in(request_dict)
                                    debug_log(f"[OK] 已切换token并重新生成请求")
                                
                                continue
                            
                            error_response = {
                                "error": {
                                    "message": f"Upstream error: {response.status_code}",
                                    "type": "upstream_error",
                                    "code": response.status_code,
                                    "details": error_msg[:500]
                                }
                            }
                            yield f"data: {json.dumps(error_response)}\n\n"
                            yield "data: [DONE]\n\n"
                            return

                        # 200 成功，处理响应
                        debug_log("Z.AI 响应成功，开始处理 SSE 流", status="success")

                        # 处理状态
                        has_thinking = False
                        first_thinking_chunk = True

                        # 处理SSE流 - 使用 aiter_lines() 自动处理编码和行分割
                        buffer = ""
                        line_count = 0

                        async for line in response.aiter_lines():
                            line_count += 1
                            if not line:
                                continue

                            # 累积到buffer处理完整的数据行
                            buffer += line + "\n"

                            # 检查是否有完整的data行
                            while "\n" in buffer:
                                current_line, buffer = buffer.split("\n", 1)
                                if not current_line.strip():
                                    continue

                                if current_line.startswith("data:"):
                                    chunk_str = current_line[5:].strip()
                                    if not chunk_str or chunk_str == "[DONE]":
                                        if chunk_str == "[DONE]":
                                            yield "data: [DONE]\n\n"
                                        continue

                                    try:
                                        chunk_data = json.loads(chunk_str)

                                        if chunk_data.get("type") == "chat:completion":
                                            data = chunk_data.get("data", {})
                                            phase = data.get("phase")

                                            # 处理思考内容
                                            if phase == "thinking":
                                                if not has_thinking:
                                                    has_thinking = True
                                                    role_chunk = {
                                                        "choices": [{
                                                            "delta": {"role": "assistant"},
                                                            "finish_reason": None,
                                                            "index": 0,
                                                            "logprobs": None,
                                                        }],
                                                        "created": int(time.time()),
                                                        "id": transformed["body"]["chat_id"],
                                                        "model": request.model,
                                                        "object": "chat.completion.chunk",
                                                        "system_fingerprint": "fp_zai_001",
                                                    }
                                                    yield f"data: {json.dumps(role_chunk)}\n\n"

                                                delta_content = data.get("delta_content", "")
                                                if delta_content:
                                                    if delta_content.startswith("<details"):
                                                        content = (
                                                            delta_content.split("</summary>\n>")[-1].strip()
                                                            if "</summary>\n>" in delta_content
                                                            else delta_content
                                                        )
                                                    else:
                                                        content = delta_content
                                                    
                                                    if first_thinking_chunk:
                                                        formatted_content = f"<think>{content}"
                                                        first_thinking_chunk = False
                                                    else:
                                                        formatted_content = content

                                                    thinking_chunk = {
                                                        "choices": [{
                                                            "delta": {
                                                                "role": "assistant",
                                                                "content": formatted_content,
                                                            },
                                                            "finish_reason": None,
                                                            "index": 0,
                                                            "logprobs": None,
                                                        }],
                                                        "created": int(time.time()),
                                                        "id": transformed["body"]["chat_id"],
                                                        "model": request.model,
                                                        "object": "chat.completion.chunk",
                                                        "system_fingerprint": "fp_zai_001",
                                                    }
                                                    yield f"data: {json.dumps(thinking_chunk)}\n\n"

                                            # 处理答案内容
                                            elif phase == "answer":
                                                edit_content = data.get("edit_content", "")
                                                delta_content = data.get("delta_content", "")

                                                if not has_thinking:
                                                    has_thinking = True
                                                    role_chunk = {
                                                        "choices": [{
                                                            "delta": {"role": "assistant"},
                                                            "finish_reason": None,
                                                            "index": 0,
                                                            "logprobs": None,
                                                        }],
                                                        "created": int(time.time()),
                                                        "id": transformed["body"]["chat_id"],
                                                        "model": request.model,
                                                        "object": "chat.completion.chunk",
                                                        "system_fingerprint": "fp_zai_001",
                                                    }
                                                    yield f"data: {json.dumps(role_chunk)}\n\n"

                                                # 处理思考结束和答案开始
                                                if edit_content and "</details>\n" in edit_content:
                                                    if has_thinking and not first_thinking_chunk:
                                                        sig_chunk = {
                                                            "choices": [{
                                                                "delta": {
                                                                    "role": "assistant",
                                                                    "content": "</think>",
                                                                },
                                                                "finish_reason": None,
                                                                "index": 0,
                                                                "logprobs": None,
                                                            }],
                                                            "created": int(time.time()),
                                                            "id": transformed["body"]["chat_id"],
                                                            "model": request.model,
                                                            "object": "chat.completion.chunk",
                                                            "system_fingerprint": "fp_zai_001",
                                                        }
                                                        yield f"data: {json.dumps(sig_chunk)}\n\n"

                                                    content_after = edit_content.split("</details>\n")[-1]
                                                    if content_after:
                                                        content_chunk = {
                                                            "choices": [{
                                                                "delta": {
                                                                    "role": "assistant",
                                                                    "content": content_after,
                                                                },
                                                                "finish_reason": None,
                                                                "index": 0,
                                                                "logprobs": None,
                                                            }],
                                                            "created": int(time.time()),
                                                            "id": transformed["body"]["chat_id"],
                                                            "model": request.model,
                                                            "object": "chat.completion.chunk",
                                                            "system_fingerprint": "fp_zai_001",
                                                        }
                                                        yield f"data: {json.dumps(content_chunk)}\n\n"

                                                elif delta_content:
                                                    if not has_thinking:
                                                        has_thinking = True
                                                        role_chunk = {
                                                            "choices": [{
                                                                "delta": {"role": "assistant"},
                                                                "finish_reason": None,
                                                                "index": 0,
                                                                "logprobs": None,
                                                            }],
                                                            "created": int(time.time()),
                                                            "id": transformed["body"]["chat_id"],
                                                            "model": request.model,
                                                            "object": "chat.completion.chunk",
                                                            "system_fingerprint": "fp_zai_001",
                                                        }
                                                        yield f"data: {json.dumps(role_chunk)}\n\n"

                                                    content_chunk = {
                                                        "choices": [{
                                                            "delta": {"content": delta_content},
                                                            "finish_reason": None,
                                                            "index": 0,
                                                            "logprobs": None,
                                                        }],
                                                        "created": int(time.time()),
                                                        "id": transformed["body"]["chat_id"],
                                                        "model": request.model,
                                                        "object": "chat.completion.chunk",
                                                        "system_fingerprint": "fp_zai_001",
                                                    }
                                                    yield f"data: {json.dumps(content_chunk)}\n\n"

                                            # 处理完成
                                            if data.get("usage"):
                                                finish_chunk = {
                                                    "choices": [{
                                                        "delta": {},
                                                        "finish_reason": "stop",
                                                        "index": 0,
                                                        "logprobs": None,
                                                    }],
                                                    "usage": data["usage"],
                                                    "created": int(time.time()),
                                                    "id": transformed["body"]["chat_id"],
                                                    "model": request.model,
                                                    "object": "chat.completion.chunk",
                                                    "system_fingerprint": "fp_zai_001",
                                                }
                                                yield f"data: {json.dumps(finish_chunk)}\n\n"
                                                yield "data: [DONE]\n\n"

                                    except json.JSONDecodeError as e:
                                        debug_log(
                                            "JSON解析错误",
                                            error=str(e),
                                            json_length=len(chunk_str),
                                            json_preview=chunk_str[:100] if len(chunk_str) > 100 else chunk_str
                                        )
                                    except Exception as e:
                                        debug_log("处理chunk错误", error=str(e))

                        debug_log("SSE 流处理完成", line_count=line_count)
                        yield "data: [DONE]\n\n"
                        return

                except Exception as e:
                    debug_log("流处理错误", error=str(e))
                    retry_count += 1
                    last_error = str(e)

                    if retry_count > settings.MAX_RETRIES:
                        error_response = {
                            "error": {
                                "message": f"Stream processing failed: {last_error}",
                                "type": "stream_error"
                            }
                        }
                        yield f"data: {json.dumps(error_response)}\n\n"
                        yield "data: [DONE]\n\n"
                        return

        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        debug_log("处理请求时发生错误", error=str(e))
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


def create_openai_response(chat_id: str, model: str, content: str, reasoning_content: str = "", usage: dict = None) -> dict:
    """创建OpenAI格式的响应对象"""
    response = {
        "id": chat_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }
        ],
        "usage": usage or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }
    
    # 如果有推理内容，添加到message中
    if reasoning_content:
        response["choices"][0]["message"]["reasoning_content"] = reasoning_content
    
    return response


async def handle_non_stream_request(request: OpenAIRequest, transformed: dict) -> dict:
    """
    处理非流式请求
    
    说明：上游始终以 SSE 形式返回（transform_request_in 固定 stream=True），
    因此这里需要聚合 aiter_lines() 的 data: 块，提取 usage、思考内容与答案内容，
    并最终产出一次性 OpenAI 格式响应。
    """
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
            # 智能退避重试策略
            if retry_count > 0:
                delay = calculate_backoff_delay(
                    retry_count=retry_count,
                    status_code=last_status_code,
                    base_delay=3.0,
                    max_delay=40.0
                )
                debug_log(
                    f"[RETRY] 非流式请求重试 ({retry_count}/{settings.MAX_RETRIES})",
                    retry_count=retry_count,
                    delay=f"{delay:.2f}s"
                )
                await asyncio.sleep(delay)
            
            # 获取全局HTTP客户端
            client = await get_http_client()
            
            # 发起流式请求（上游始终返回SSE流）
            async with client.stream(
                "POST",
                transformed["config"]["url"],
                json=transformed["body"],
                headers=transformed["config"]["headers"],
            ) as response:
                # 检查响应状态码
                if response.status_code != 200:
                    error_text = await response.aread()
                    error_msg = error_text.decode('utf-8', errors='ignore')
                    debug_log(
                        "上游返回错误",
                        status_code=response.status_code,
                        error_detail=error_msg[:200]
                    )
                    
                    # 可重试的错误
                    retryable_codes = [400, 401, 429, 502, 503, 504]
                    if response.status_code in retryable_codes and retry_count < settings.MAX_RETRIES:
                        last_status_code = response.status_code
                        last_error = f"{response.status_code}: {error_msg}"
                        retry_count += 1
                        
                        # 如果是401认证失败或400错误，尝试切换token
                        if response.status_code in [400, 401]:
                            debug_log("[SWITCH] 检测到认证/请求错误，尝试切换token")
                            new_token = transformer.switch_token()
                            # 使用新token重新生成请求
                            request_dict = request.model_dump()
                            transformed = await transformer.transform_request_in(request_dict)
                            debug_log(f"[OK] 已切换token并重新生成请求")
                        
                        continue
                    
                    # 不可重试的错误，直接抛出
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Upstream error: {error_msg[:500]}"
                    )
                
                # 200 成功，聚合SSE流数据
                debug_log("Z.AI 响应成功，开始聚合非流式数据", status="success")
                
                # 重置聚合变量
                final_content = ""
                reasoning_content = ""
                
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    
                    line = line.strip()
                    
                    # 仅处理以 data: 开头的 SSE 行
                    if not line.startswith("data:"):
                        # 尝试解析为错误 JSON
                        try:
                            maybe_err = json.loads(line)
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
                    if not data_str or data_str in ("[DONE]", "DONE", "done"):
                        continue
                    
                    # 解析 SSE 数据块
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    
                    if chunk.get("type") != "chat:completion":
                        continue
                    
                    data = chunk.get("data", {})
                    phase = data.get("phase")
                    delta_content = data.get("delta_content", "")
                    edit_content = data.get("edit_content", "")
                    
                    # 记录用量（通常在最后块中出现）
                    if data.get("usage"):
                        try:
                            usage_info = data["usage"]
                        except Exception:
                            pass
                    
                    # 思考阶段聚合（去除 <details><summary>... 包裹头）
                    if phase == "thinking":
                        if delta_content:
                            if delta_content.startswith("<details"):
                                cleaned = (
                                    delta_content.split("</summary>\n>")[-1].strip()
                                    if "</summary>\n>" in delta_content
                                    else delta_content
                                )
                            else:
                                cleaned = delta_content
                            reasoning_content += cleaned
                    
                    # 答案阶段聚合
                    elif phase == "answer":
                        # 当 edit_content 同时包含思考结束标记与答案时，提取答案部分
                        if edit_content and "</details>\n" in edit_content:
                            content_after = edit_content.split("</details>\n")[-1]
                            if content_after:
                                final_content += content_after
                        elif delta_content:
                            final_content += delta_content
                
                # 清理并返回
                final_content = (final_content or "").strip()
                reasoning_content = (reasoning_content or "").strip()
                
                # 若没有聚合到答案，但有思考内容，则保底返回思考内容
                if not final_content and reasoning_content:
                    final_content = reasoning_content
                
                debug_log(
                    "非流式响应聚合完成",
                    content_length=len(final_content),
                    reasoning_length=len(reasoning_content),
                    usage=usage_info
                )
                
                # 返回标准OpenAI格式响应
                return create_openai_response(
                    transformed["body"]["chat_id"],
                    request.model,
                    final_content,
                    reasoning_content,
                    usage_info
                )
        
        except HTTPException:
            raise
        except Exception as e:
            debug_log("非流式处理错误", error=str(e))
            retry_count += 1
            last_error = str(e)
            
            if retry_count > settings.MAX_RETRIES:
                raise HTTPException(
                    status_code=500,
                    detail=f"Non-stream processing failed after {settings.MAX_RETRIES} retries: {last_error}"
                )
    
    # 不应该到达这里
    raise HTTPException(status_code=500, detail="Unexpected error in non-stream processing")
