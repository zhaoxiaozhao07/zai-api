"""
OpenAI API endpoints - 优化版本
"""

import time
import json
import asyncio
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse
import httpx

from config import settings
from schemas import OpenAIRequest, ModelsResponse, Model
from helpers import debug_log, get_logger
from zai_transformer import ZAITransformer

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
    """Handle chat completion requests with ZAI transformer - 优化异步处理"""
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

        # 调用上游API
        async def stream_response():
            """流式响应生成器（包含重试机制和指数退避）"""
            nonlocal transformed  # 声明使用外部作用域的transformed变量
            retry_count = 0
            last_error = None

            while retry_count <= settings.MAX_RETRIES:
                try:
                    # 指数退避重试策略
                    if retry_count > 0:
                        # 计算退避延迟：基础2秒 * 2^(retry_count-1)，最大30秒
                        delay = min(2.0 * (2 ** (retry_count - 1)), 30.0)
                        debug_log(
                            f"重试请求 ({retry_count}/{settings.MAX_RETRIES})",
                            retry_count=retry_count,
                            max_retries=settings.MAX_RETRIES,
                            delay=delay
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
                                retry_count += 1
                                last_error = f"{response.status_code}: {error_msg}"
                                
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
