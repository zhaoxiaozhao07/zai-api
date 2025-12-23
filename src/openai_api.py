"""
OpenAI API endpoints
"""

import time
import json
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional

from .config import settings
from .schemas import OpenAIRequest, ModelsResponse, Model
from .helpers import (
    error_log,
    info_log,
    debug_log,
    get_logger,
    bind_request_context,
    reset_request_context,
    request_stage_log,
)
from .services.openai_service import chat_completion_service

# 尝试导入orjson（性能优化），如果不可用则fallback到标准json
try:
    import orjson
    
    # 创建orjson兼容层
    class JSONEncoder:
        @staticmethod
        def dumps(obj, **kwargs):
            # orjson.dumps返回bytes，需要decode
            return orjson.dumps(obj).decode('utf-8')
        
        @staticmethod
        def loads(s, **kwargs):
            # orjson.loads可以接受str或bytes
            return orjson.loads(s)
    
    json_lib = JSONEncoder()
    info_log("[OK] 使用 orjson 进行 JSON 序列化/反序列化（性能优化）")
except ImportError:
    json_lib = json
    info_log("[WARN] orjson 未安装，使用标准 json 库")

router = APIRouter()

logger = get_logger("openai_api")

service = chat_completion_service


@router.get("/v1/models")
async def list_models():
    """List available models"""
    current_time = int(time.time())
    response = ModelsResponse(
        data=[
            Model(id=settings.PRIMARY_MODEL, created=current_time, owned_by="z.ai"),
            Model(id=settings.THINKING_MODEL, created=current_time, owned_by="z.ai"),
            Model(id=settings.AIR_MODEL, created=current_time, owned_by="z.ai"),
            Model(id=settings.GLM_45V_MODEL, created=current_time, owned_by="z.ai"),
            Model(id=settings.GLM_46_MODEL, created=current_time, owned_by="z.ai"),
            Model(id=settings.GLM_46_THINKING_MODEL, created=current_time, owned_by="z.ai"),
            Model(id=settings.GLM_46V_MODEL, created=current_time, owned_by="z.ai"),
            Model(id=settings.GLM_47_MODEL, created=current_time, owned_by="z.ai"),
            Model(id=settings.GLM_47_THINKING_MODEL, created=current_time, owned_by="z.ai"),
        ]
    )
    return response


@router.post("/v1/chat/completions")
async def chat_completions(request: OpenAIRequest, authorization: str = Header(...)):
    """处理 chat completion 请求，支持流式和非流式"""
    role = request.messages[0].role if request.messages else "unknown"
    request_stage_log(
        "received",
        "收到客户端请求",
        model=request.model,
        stream=request.stream,
        entry_role=role,
        message_count=len(request.messages),
        tools_count=len(request.tools) if request.tools else 0,
    )
    
    # 输出客户端请求体
    request_body = request.model_dump()
    debug_log("客户端请求体详情", request_body=json_lib.dumps(request_body))
    
    try:
        # Validate API key
        if not settings.SKIP_AUTH_TOKEN:
            if not authorization.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
            
            api_key = authorization[7:]
            if api_key != settings.AUTH_TOKEN:
                raise HTTPException(status_code=401, detail="Invalid API key")
        
        # 获取请求上下文（HTTP 客户端 / 代理 / 上游）
        request_client, current_proxy, current_upstream = await service.get_request_context()
        request_stage_log(
            "route_selected",
            "已获取请求路由配置",
            upstream=current_upstream,
            proxy=current_proxy or "direct",
        )

        # 准备请求并转换
        request_dict, request_dict_for_transform = await service.prepare_request(request)
        chat_id = request_dict_for_transform.get("chat_id")
        bind_request_context(
            request_id=chat_id,
            upstream=current_upstream,
            proxy=current_proxy,
            model=request.model,
        )
        request_stage_log(
            "context_ready",
            "请求上下文已建立",
            request_id=chat_id,
            upstream=current_upstream,
            proxy=current_proxy or "direct",
            mode="stream" if request.stream else "non_stream",
        )

        transformed = await service.build_transformed(
            request_dict_for_transform,
            client=request_client,
            upstream=current_upstream,
        )
        request_stage_log(
            "transformed",
            "请求已转换为上游所需格式",
            upstream=current_upstream,
        )

        # 根据 stream 参数决定返回流式或非流式响应
        if not request.stream:
            request_stage_log(
                "non_stream_mode",
                "使用非流式模式",
                upstream=current_upstream,
                proxy=current_proxy or "direct",
            )
            try:
                result = await service.handle_non_stream_request(
                    request,
                    transformed,
                    request_client,
                    current_proxy,
                    current_upstream,
                    request_dict_for_transform,
                    json_lib,
                )
                request_stage_log("non_stream_ready", "非流式结果已生成")
                return result
            finally:
                reset_request_context("request_id", "upstream", "proxy", "model")

        async def stream_response():
            request_stage_log(
                "stream_mode",
                "使用流式模式",
                upstream=current_upstream,
                proxy=current_proxy or "direct",
            )
            try:
                request_stage_log("stream_dispatch", "开始推送流式响应数据")
                async for chunk in service.stream_response(
                    request,
                    transformed,
                    request_client,
                    current_proxy,
                    current_upstream,
                    request_dict_for_transform,
                    json_lib,
                ):
                                                    yield chunk
                request_stage_log("stream_finished", "流式响应生成器完成")
            finally:
                request_stage_log("stream_cleanup", "流式上下文清理")
                reset_request_context("request_id", "upstream", "proxy", "model")

        streaming_response = StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        request_stage_log("stream_ready", "流式响应已交给 FastAPI", media_type="text/event-stream")
        return streaming_response

    except HTTPException:
        reset_request_context("request_id", "upstream", "proxy", "model")
        error_log("[REQUEST] 处理HTTPException")
        raise
    except Exception as e:
        reset_request_context("request_id", "upstream", "proxy", "model")
        error_log("处理请求时发生错误", error=str(e))
        error_log("[REQUEST] 处理异常")
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
