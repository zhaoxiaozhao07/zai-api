"""
OpenAI API endpoints - 优化版本（集成 Toolify 工具调用功能）
"""

import time
import json
import random
import asyncio
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse
import httpx
from typing import Optional, Dict

from .config import settings
from .schemas import OpenAIRequest, ModelsResponse, Model
from .helpers import debug_log, get_logger, perf_timer
from .zai_transformer import ZAITransformer
from .toolify_handler import should_enable_toolify, prepare_toolify_request, parse_toolify_response
from .toolify.detector import StreamingFunctionCallDetector
from .toolify_config import get_toolify

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
    debug_log("✅ 使用 orjson 进行 JSON 序列化/反序列化（性能优化）")
except ImportError:
    json_lib = json
    debug_log("⚠️ orjson 未安装，使用标准 json 库")

router = APIRouter()

# 获取结构化日志记录器
logger = get_logger("openai_api")

# 全局转换器实例
transformer = ZAITransformer()

# ================== 客户端池管理 ==================
# 为每个代理维护一个长期客户端（复用连接）
_proxy_clients: Dict[str, httpx.AsyncClient] = {}
_default_client: Optional[httpx.AsyncClient] = None  # 无代理时的默认客户端
_client_lock = asyncio.Lock()  # 保护客户端池的并发访问

# 优化的连接池配置
CONNECTION_POOL_CONFIG = {
    "limits": httpx.Limits(
        max_keepalive_connections=20,  # 保持连接数
        max_connections=100,            # 最大连接数
        keepalive_expiry=30,           # 连接保活时间
    ),
    "timeout": httpx.Timeout(
        connect=10.0,    # 连接超时
        read=30.0,       # 读取超时
        write=30.0,      # 写入超时
        pool=10.0,       # 连接池获取超时
    ),
    "http2": True,  # 启用HTTP/2
}

async def get_or_create_client(proxy_url: Optional[str] = None) -> httpx.AsyncClient:
    """
    获取或创建HTTP客户端（简单的池化复用，线程安全）
    
    Args:
        proxy_url: 代理地址，如果为None则使用直连
    
    Returns:
        httpx.AsyncClient: 复用的HTTP客户端
    """
    global _proxy_clients, _default_client
    
    async with _client_lock:  # 保护并发访问
        # 直连情况
        if proxy_url is None:
            if _default_client is None:
                debug_log("[CLIENT] 创建默认客户端（无代理）")
                _default_client = httpx.AsyncClient(**CONNECTION_POOL_CONFIG)
            return _default_client
        
        # 代理情况
        if proxy_url not in _proxy_clients:
            debug_log(f"[CLIENT] 为代理创建新客户端: {proxy_url}")
            _proxy_clients[proxy_url] = httpx.AsyncClient(
                proxy=proxy_url,
                **CONNECTION_POOL_CONFIG
            )
        
        return _proxy_clients[proxy_url]

async def cleanup_clients():
    """清理所有缓存的HTTP客户端"""
    global _proxy_clients, _default_client
    
    # 关闭所有代理客户端
    for proxy_url, client in _proxy_clients.items():
        debug_log(f"[CLIENT] 关闭代理客户端: {proxy_url}")
        await client.aclose()
    _proxy_clients.clear()
    
    # 关闭默认客户端
    if _default_client:
        debug_log("[CLIENT] 关闭默认客户端")
        await _default_client.aclose()
        _default_client = None
    
    debug_log("[CLIENT] 所有客户端已清理")

# ================== 代理池管理 ==================
_proxy_list = []  # 代理列表
_proxy_index = 0  # 当前代理索引
_proxy_lock = asyncio.Lock()  # 代理切换锁

def init_proxy_pool():
    """初始化代理池"""
    global _proxy_list
    
    # 优先使用 HTTPS_PROXY_LIST，如果没有则使用 HTTP_PROXY_LIST
    _proxy_list = settings.HTTPS_PROXY_LIST or settings.HTTP_PROXY_LIST
    
    if _proxy_list:
        debug_log(f"[PROXY] 初始化代理池，共 {len(_proxy_list)} 个代理，策略: {settings.PROXY_STRATEGY}")
        for i, proxy in enumerate(_proxy_list):
            debug_log(f"  代理 {i+1}: {proxy}")

def get_next_proxy() -> Optional[str]:
    """
    获取下一个代理（根据策略）
    
    Returns:
        Optional[str]: 代理地址，如果没有可用代理则返回 None
    """
    global _proxy_index
    
    if not _proxy_list:
        return None
    
    if settings.PROXY_STRATEGY == "round-robin":
        # 轮询策略：每次调用都切换到下一个代理
        proxy = _proxy_list[_proxy_index]
        _proxy_index = (_proxy_index + 1) % len(_proxy_list)
        debug_log(f"[PROXY] Round-robin选择代理 {_proxy_index}: {proxy}")
        return proxy
    else:
        # failover 策略：始终使用当前索引的代理
        proxy = _proxy_list[_proxy_index]
        debug_log(f"[PROXY] Failover使用代理 {_proxy_index}: {proxy}")
        return proxy

async def switch_proxy_on_failure():
    """
    失败时切换代理（仅 failover 策略需要）
    """
    global _proxy_index
    
    if not _proxy_list or settings.PROXY_STRATEGY != "failover":
        return
    
    async with _proxy_lock:
        old_index = _proxy_index
        _proxy_index = (_proxy_index + 1) % len(_proxy_list)
        debug_log(f"[PROXY] 代理失败，从 {old_index} 切换到 {_proxy_index}: {_proxy_list[_proxy_index]}")

# 初始化代理池
init_proxy_pool()

async def get_request_client() -> httpx.AsyncClient:
    """
    为当前请求获取合适的HTTP客户端（简化版）
    使用客户端池复用连接，每个代理维护一个长期客户端
    
    Returns:
        httpx.AsyncClient: 复用的HTTP客户端
    """
    # 获取当前应该使用的代理
    proxy = get_next_proxy()
    
    # 获取或创建对应的客户端
    return await get_or_create_client(proxy)

def calculate_backoff_delay(retry_count: int, status_code: int = None, base_delay: float = 1.5, max_delay: float = 8.0) -> float:
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
    # 使用线性增长而非指数增长：base_delay * retry_count
    # 这样：第1次=1.5s, 第2次=3s, 第3次=4.5s, 第4次=6s
    linear_delay = base_delay * retry_count
    
    # 针对不同错误类型调整延迟
    if status_code == 429:  # Rate limit - 更长的退避时间
        linear_delay *= 1.5  # 1.5倍延迟
        debug_log("[BACKOFF] 检测到429限流错误，使用更长退避时间")
    elif status_code in [502, 503, 504]:  # 服务端错误 - 适中延迟
        linear_delay *= 1.2  # 1.2倍延迟
        debug_log("[BACKOFF] 检测到服务端错误，使用适中退避时间")
    elif status_code in [400, 401, 405]:  # 认证/请求错误 - 标准延迟
        debug_log(f"[BACKOFF] 检测到{status_code}认证/请求错误，使用标准退避时间")
    
    # 限制最大延迟
    linear_delay = min(linear_delay, max_delay)
    
    # 添加随机抖动因子（±20%），避免多个请求同时重试造成雪崩
    jitter = linear_delay * 0.2  # 20%的抖动范围
    jittered_delay = linear_delay + random.uniform(-jitter, jitter)
    
    # 确保延迟不会小于0.5秒
    final_delay = max(jittered_delay, 0.5)
    
    debug_log(
        "[BACKOFF] 计算退避延迟",
        retry_count=retry_count,
        status_code=status_code,
        linear=f"{linear_delay:.2f}s",
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
            Model(id=settings.GLM_45V_MODEL, created=current_time, owned_by="z.ai"),
            Model(id=settings.GLM_46_MODEL, created=current_time, owned_by="z.ai"),
            Model(id=settings.GLM_46_THINKING_MODEL, created=current_time, owned_by="z.ai"),
            Model(id=settings.GLM_46_SEARCH_MODEL, created=current_time, owned_by="z.ai"),
        ]
    )
    return response


@router.post("/v1/chat/completions")
async def chat_completions(request: OpenAIRequest, authorization: str = Header(...)):
    """Handle chat completion requests with ZAI transformer - 支持流式和非流式以及工具调用"""
    role = request.messages[0].role if request.messages else "unknown"
    debug_log(
        "收到客户端请求",
        model=request.model,
        stream=request.stream,
        message_count=len(request.messages),
        tools_count=len(request.tools) if request.tools else 0
    )
    
    # 获取复用的HTTP客户端
    request_client = None
    
    try:
        # Validate API key
        if not settings.SKIP_AUTH_TOKEN:
            if not authorization.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
            
            api_key = authorization[7:]
            if api_key != settings.AUTH_TOKEN:
                raise HTTPException(status_code=401, detail="Invalid API key")
        
        # 获取复用的HTTP客户端
        request_client = await get_request_client()
        debug_log("[REQUEST] 获取复用HTTP客户端")
            
        # 转换请求
        request_dict = request.model_dump()
        
        # 检查是否需要启用工具调用
        enable_toolify = should_enable_toolify(request_dict)
        
        # 准备消息列表
        messages = [msg.model_dump() if hasattr(msg, 'model_dump') else msg for msg in request.messages]
        
        # 如果启用工具调用，预处理消息并注入提示词
        if enable_toolify:
            debug_log("[TOOLIFY] 工具调用功能已启用")
            messages, _ = prepare_toolify_request(request_dict, messages)
            # 从请求中移除 tools 和 tool_choice（已转换为提示词）
            request_dict_for_transform = request_dict.copy()
            request_dict_for_transform.pop("tools", None)
            request_dict_for_transform.pop("tool_choice", None)
            request_dict_for_transform["messages"] = messages
        else:
            request_dict_for_transform = request_dict
        
        debug_log("开始转换请求格式: OpenAI -> Z.AI")
        
        # 初始转换（传入request_client用于图像上传和Token获取）
        transformed = await transformer.transform_request_in(request_dict_for_transform, client=request_client)
        
        # 根据stream参数决定返回流式或非流式响应
        if not request.stream:
            debug_log("使用非流式模式")
            result = await handle_non_stream_request(request, transformed, enable_toolify, request_client)
            return result

        # 调用上游API（流式模式）
        async def stream_response():
            """流式响应生成器（包含重试机制、智能退避策略和工具调用检测）"""
            nonlocal transformed  # 声明使用外部作用域的transformed变量
            nonlocal request_client  # 声明使用外部作用域的request_client变量
            retry_count = 0
            last_error = None
            last_status_code = None  # 记录上次失败的状态码
            
            # 如果启用工具调用，创建检测器
            toolify_detector = None
            if enable_toolify:
                toolify_instance = get_toolify()
                if toolify_instance:
                    toolify_detector = StreamingFunctionCallDetector(toolify_instance.trigger_signal)
                    debug_log("[TOOLIFY] 流式工具调用检测器已初始化")

            while retry_count <= settings.MAX_RETRIES:
                try:
                    # 智能退避重试策略（带随机抖动和错误类型区分）
                    if retry_count > 0:
                        # 使用智能退避策略计算延迟（线性增长：1.5s, 3s, 4.5s, 6s...）
                        delay = calculate_backoff_delay(
                            retry_count=retry_count,
                            status_code=last_status_code
                        )
                        debug_log(
                            f"[RETRY] 重试请求 ({retry_count}/{settings.MAX_RETRIES})",
                            retry_count=retry_count,
                            max_retries=settings.MAX_RETRIES,
                            delay=f"{delay:.2f}s",
                            last_status=last_status_code
                        )
                        await asyncio.sleep(delay)

                    # 使用请求专用的HTTP客户端
                    client = request_client
                    
                    # 发起流式请求（带性能追踪）
                    # 使用转换后的headers（包含Accept-Encoding）
                    headers = transformed["config"]["headers"].copy()
                    
                    request_start_time = time.perf_counter()
                    async with client.stream(
                        "POST",
                        transformed["config"]["url"],
                        json=transformed["body"],
                        headers=headers,
                    ) as response:
                        # 记录首字节时间（TTFB）
                        ttfb = (time.perf_counter() - request_start_time) * 1000
                        debug_log(f"⏱️ 上游TTFB (首字节时间)", ttfb_ms=f"{ttfb:.2f}ms")
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
                            retryable_codes = [400, 401, 405, 429, 502, 503, 504]
                            if response.status_code in retryable_codes and retry_count < settings.MAX_RETRIES:
                                # 记录状态码和错误信息
                                last_status_code = response.status_code
                                last_error = f"{response.status_code}: {error_msg}"
                                retry_count += 1
                                
                                # 检查当前是否使用匿名Token
                                from .token_pool import get_token_pool
                                token_pool = get_token_pool()
                                current_token = transformed.get("token", "")
                                is_anonymous = token_pool.is_anonymous_token(current_token)
                                
                                if is_anonymous:
                                    # 匿名Token：任何错误都清理缓存并重新获取
                                    debug_log(f"[ANONYMOUS] 检测到匿名Token错误 {response.status_code}，清理缓存并重新获取")
                                    await transformer.clear_anonymous_token_cache()
                                    
                                    # 清理客户端缓存，强制下次重新创建（匿名Token可能导致连接状态异常）
                                    debug_log("[CLIENT] 清理HTTP客户端缓存，下次请求将创建新客户端")
                                    await cleanup_clients()
                                    
                                    # 重新获取客户端
                                    request_client = await get_request_client()
                                    
                                    # 使用新的匿名Token重新生成请求
                                    transformed = await transformer.transform_request_in(request_dict_for_transform, client=request_client)
                                    debug_log("[OK] 已获取新的匿名Token并重新生成请求")
                                else:
                                    # 配置Token：按原有逻辑处理
                                    if response.status_code in [400, 401, 405]:
                                        debug_log(f"[CONFIG] 配置Token错误 {response.status_code}，切换到下一个Token")
                                        new_token = await transformer.switch_token()
                                        transformer.refresh_header_template()
                                        transformed = await transformer.transform_request_in(request_dict_for_transform, client=request_client)
                                        debug_log(f"[OK] 已切换到下一个配置Token")
                                    
                                    # 网络错误时尝试切换代理（仅限 failover 策略）
                                    if response.status_code in [502, 503, 504] and _proxy_list:
                                        await switch_proxy_on_failure()
                                
                                continue
                            
                            error_response = {
                                "error": {
                                    "message": f"Upstream error: {response.status_code}",
                                    "type": "upstream_error",
                                    "code": response.status_code,
                                    "details": error_msg[:500]
                                }
                            }
                            yield f"data: {json_lib.dumps(error_response)}\n\n"
                            yield "data: [DONE]\n\n"
                            return

                        # 200 成功，处理响应
                        debug_log("Z.AI 响应成功，开始处理 SSE 流", status="success")

                        # 处理状态
                        has_thinking = False
                        first_thinking_chunk = True
                        accumulated_content = ""  # 累积内容用于工具调用检测
                        yielded_chunks_count = 0  # 统计yield的chunk数量
                        usage_info = None  # 暂存usage信息

                        # 处理SSE流 - 使用 aiter_lines() 自动处理行分割
                        buffer = ""
                        line_count = 0
                        debug_log("开始迭代SSE流...")

                        async for line in response.aiter_lines():
                            line_count += 1
                            if not line:
                                continue
                            
                            debug_log(f"[RAW-LINE-{line_count}] 长度: {len(line)}, 开头: {repr(line[:50]) if len(line) > 0 else 'EMPTY'}")

                            # 累积到buffer处理完整的数据行
                            buffer += line + "\n"

                            # 检查是否有完整的data行
                            while "\n" in buffer:
                                current_line, buffer = buffer.split("\n", 1)
                                if not current_line.strip():
                                    continue

                                if current_line.startswith("data:"):
                                    chunk_str = current_line[5:].strip()
                                    debug_log(f"[SSE-RAW] 收到数据行，长度: {len(chunk_str)}, 预览: {chunk_str[:100] if chunk_str else 'empty'}")
                                    if not chunk_str or chunk_str == "[DONE]":
                                            if chunk_str == "[DONE]":
                                                debug_log("[SSE-RAW] 收到 [DONE] 信号")
                                                # 流结束，检查是否有工具调用
                                                if toolify_detector:
                                                    debug_log(f"[TOOLIFY] 流结束，检测器状态: {toolify_detector.state}, 缓冲区长度: {len(toolify_detector.content_buffer)}")
                                                    debug_log(f"[TOOLIFY] 缓冲区内容: {repr(toolify_detector.content_buffer[:500])}")
                                                    parsed_tools, remaining_content = toolify_detector.finalize()
                                                    debug_log(f"[TOOLIFY] finalize()结果: tools={parsed_tools}, remaining={repr(remaining_content[:100]) if remaining_content else 'empty'}")
                                                    
                                                    # 先输出剩余的内容（如果有）
                                                    if remaining_content:
                                                        debug_log(f"[TOOLIFY] 输出缓冲区剩余内容: {len(remaining_content)}字符")
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
                                                            yield f"data: {json_lib.dumps(role_chunk)}\n\n"
                                                        
                                                        content_chunk = {
                                                            "choices": [{
                                                                "delta": {"content": remaining_content},
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
                                                        yield f"data: {json_lib.dumps(content_chunk)}\n\n"
                                                    
                                                    # 然后处理工具调用
                                                    if parsed_tools:
                                                        debug_log("[TOOLIFY] 流结束时检测到工具调用")
                                                        from .toolify_handler import format_toolify_response_for_stream
                                                        tool_chunks = format_toolify_response_for_stream(
                                                            parsed_tools, 
                                                            request.model, 
                                                            transformed["body"]["chat_id"]
                                                        )
                                                        for chunk in tool_chunks:
                                                            yield chunk
                                                        # 检测到工具调用后提前结束
                                                        debug_log("[REQUEST] 流式响应（早期工具调用检测）完成")
                                                        return
                                                    else:
                                                        debug_log("[TOOLIFY] finalize()未检测到工具调用")
                                                yield "data: [DONE]\n\n"
                                            continue

                                    try:
                                        chunk_data = json_lib.loads(chunk_str)
                                        debug_log(f"[SSE-JSON] 解析成功，type: {chunk_data.get('type')}")

                                        if chunk_data.get("type") == "chat:completion":
                                            data = chunk_data.get("data", {})
                                            phase = data.get("phase")
                                            debug_log(f"[SSE] 处理块: type={chunk_data.get('type')}, phase={phase}, has_delta={bool(data.get('delta_content'))}, has_usage={bool(data.get('usage'))}")

                                            # 处理tool_call阶段（提取搜索信息）
                                            if phase == "tool_call":
                                                edit_content = data.get("edit_content", "")
                                                
                                                # 提取搜索查询信息
                                                if edit_content and "<glm_block" in edit_content and "search" in edit_content:
                                                    # 尝试从edit_content中提取搜索查询
                                                    try:
                                                        import re
                                                        # 先尝试直接解码Unicode
                                                        decoded = edit_content
                                                        try:
                                                            # 解码\uXXXX格式的Unicode字符
                                                            decoded = edit_content.encode('utf-8').decode('unicode_escape').encode('latin1').decode('utf-8')
                                                        except:
                                                            # 如果解码失败，尝试其他方法
                                                            try:
                                                                import codecs
                                                                decoded = codecs.decode(edit_content, 'unicode_escape')
                                                            except:
                                                                pass
                                                        
                                                        # 提取queries数组
                                                        queries_match = re.search(r'"queries":\s*\[(.*?)\]', decoded)
                                                        if queries_match:
                                                            queries_str = queries_match.group(1)
                                                            # 提取所有引号内的内容
                                                            queries = re.findall(r'"([^"]+)"', queries_str)
                                                            if queries:
                                                                search_info = "🔍 **搜索：** " + "　".join(queries[:5])  # 最多显示5个查询
                                                                
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
                                                                    yield f"data: {json_lib.dumps(role_chunk)}\n\n"
                                                                
                                                                search_chunk = {
                                                                    "choices": [{
                                                                        "delta": {"content": f"\n\n{search_info}\n\n"},
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
                                                                yielded_chunks_count += 1
                                                                debug_log(f"[YIELD] search info chunk #{yielded_chunks_count}, queries: {queries}")
                                                                yield f"data: {json_lib.dumps(search_chunk)}\n\n"
                                                    except Exception as e:
                                                        debug_log(f"[SSE-TOOL_CALL] 提取搜索信息失败: {e}")
                                                
                                                # 如果不是usage信息，跳过其他tool_call内容
                                                if not data.get("usage"):
                                                    debug_log(f"[SSE-TOOL_CALL] 跳过tool_call阶段的其他内容")
                                                    continue
                                            
                                            # 处理思考内容
                                            if phase == "thinking":
                                                delta_content = data.get("delta_content", "")
                                                debug_log(f"[SSE-THINKING] delta_content长度: {len(delta_content) if delta_content else 0}, 内容: {repr(delta_content[:50]) if delta_content else 'EMPTY'}")
                                                
                                                # 工具调用检测
                                                if toolify_detector and delta_content:
                                                    debug_log(f"[SSE-THINKING] 调用工具检测器")
                                                    is_tool_detected, content_to_yield = toolify_detector.process_chunk(delta_content)
                                                    
                                                    if is_tool_detected:
                                                        debug_log(f"[TOOLIFY] 在thinking阶段检测到工具调用触发信号，缓冲区长度: {len(toolify_detector.content_buffer)}")
                                                        debug_log(f"[TOOLIFY] 检测时缓冲区内容: {repr(toolify_detector.content_buffer[:200])}")
                                                        debug_log(f"[TOOLIFY] 检测器当前状态: {toolify_detector.state}")
                                                        # 如果之前有输出内容，先发送
                                                        if content_to_yield:
                                                            debug_log(f"[TOOLIFY] 输出触发信号前的内容: {repr(content_to_yield[:100])}")
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
                                                                yield f"data: {json_lib.dumps(role_chunk)}\n\n"
                                                            
                                                            content_chunk = {
                                                                "choices": [{
                                                                    "delta": {"content": content_to_yield},
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
                                                            yield f"data: {json_lib.dumps(content_chunk)}\n\n"
                                                        debug_log(f"[TOOLIFY] 跳过本次delta处理，等待更多内容")
                                                        continue
                                                    
                                                    # 使用检测器处理后的内容
                                                    delta_content = content_to_yield
                                                
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
                                                    yield f"data: {json_lib.dumps(role_chunk)}\n\n"

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
                                                    yielded_chunks_count += 1
                                                    debug_log(f"[YIELD] thinking chunk #{yielded_chunks_count}, content长度: {len(formatted_content)}")
                                                    yield f"data: {json_lib.dumps(thinking_chunk)}\n\n"

                                            # 处理答案内容
                                            elif phase == "answer":
                                                edit_content = data.get("edit_content", "")
                                                delta_content = data.get("delta_content", "")
                                                debug_log(f"[SSE-ANSWER] delta_content长度: {len(delta_content) if delta_content else 0}, edit_content长度: {len(edit_content) if edit_content else 0}")
                                                
                                                # 工具调用检测
                                                if toolify_detector and delta_content:
                                                    debug_log(f"[SSE-ANSWER] 调用工具检测器")
                                                    is_tool_detected, content_to_yield = toolify_detector.process_chunk(delta_content)
                                                    
                                                    if is_tool_detected:
                                                        debug_log(f"[TOOLIFY] 在answer阶段检测到工具调用触发信号，缓冲区长度: {len(toolify_detector.content_buffer)}")
                                                        debug_log(f"[TOOLIFY] 检测时缓冲区内容: {repr(toolify_detector.content_buffer[:200])}")
                                                        debug_log(f"[TOOLIFY] 检测器当前状态: {toolify_detector.state}")
                                                        # 如果之前有输出内容，先发送
                                                        if content_to_yield:
                                                            debug_log(f"[TOOLIFY] 输出触发信号前的内容: {repr(content_to_yield[:100])}")
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
                                                                yield f"data: {json_lib.dumps(role_chunk)}\n\n"
                                                            
                                                            content_chunk = {
                                                                "choices": [{
                                                                    "delta": {"content": content_to_yield},
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
                                                            yield f"data: {json_lib.dumps(content_chunk)}\n\n"
                                                        debug_log(f"[TOOLIFY] 跳过本次delta处理，等待更多内容")
                                                        continue
                                                    
                                                    # 使用检测器处理后的内容
                                                    delta_content = content_to_yield

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
                                                    yield f"data: {json_lib.dumps(role_chunk)}\n\n"

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
                                                        yield f"data: {json_lib.dumps(sig_chunk)}\n\n"

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
                                                        yield f"data: {json_lib.dumps(content_chunk)}\n\n"

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
                                                        yield f"data: {json_lib.dumps(role_chunk)}\n\n"

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
                                                    yielded_chunks_count += 1
                                                    debug_log(f"[YIELD] answer chunk #{yielded_chunks_count}, content长度: {len(delta_content)}")
                                                    yield f"data: {json_lib.dumps(content_chunk)}\n\n"

                                            # 暂存usage信息，但不立即结束（可能后面还有answer内容）
                                            if data.get("usage"):
                                                debug_log("[SSE] 收到usage信息，暂存但继续处理")
                                                # 暂存usage信息
                                                usage_info = data["usage"]
                                                # 不要在这里return，继续处理后续chunks

                                    except json.JSONDecodeError as e:
                                        debug_log(
                                            "JSON解析错误",
                                            error=str(e),
                                            json_length=len(chunk_str),
                                            json_preview=chunk_str[:100] if len(chunk_str) > 100 else chunk_str
                                        )
                                    except Exception as e:
                                        debug_log("处理chunk错误", error=str(e))

                        debug_log("SSE 流处理完成", line_count=line_count, yielded_chunks=yielded_chunks_count)
                        
                        # 流自然结束，检查是否有工具调用
                        if toolify_detector:
                            debug_log(f"[TOOLIFY] 流自然结束 - 检测器状态: {toolify_detector.state}, 缓冲区长度: {len(toolify_detector.content_buffer)}")
                            parsed_tools, remaining_content = toolify_detector.finalize()
                            debug_log(f"[TOOLIFY] 流自然结束 - finalize()结果: tools={parsed_tools}, remaining={repr(remaining_content[:100]) if remaining_content else 'empty'}")
                            
                            # 先输出剩余的内容（如果有）
                            if remaining_content:
                                debug_log(f"[TOOLIFY] 输出缓冲区剩余内容: {len(remaining_content)}字符")
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
                                    yield f"data: {json_lib.dumps(role_chunk)}\n\n"
                                
                                content_chunk = {
                                    "choices": [{
                                        "delta": {"content": remaining_content},
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
                                yield f"data: {json_lib.dumps(content_chunk)}\n\n"
                            
                            # 然后处理工具调用
                            if parsed_tools:
                                debug_log("[TOOLIFY] 流自然结束 - 检测到工具调用，输出工具调用结果")
                                from .toolify_handler import format_toolify_response_for_stream
                                tool_chunks = format_toolify_response_for_stream(
                                    parsed_tools, 
                                    request.model, 
                                    transformed["body"]["chat_id"]
                                )
                                for chunk in tool_chunks:
                                    yield chunk
                                # 流式响应完成
                                debug_log("[REQUEST] 流式响应（工具调用）完成")
                                return
                        
                        debug_log("[SSE] 流自然结束 - 没有工具调用，输出finish chunk")
                        # 输出最后的finish chunk（包含usage信息）
                        finish_chunk = {
                            "choices": [{
                                "delta": {},
                                "finish_reason": "stop",
                                "index": 0,
                                "logprobs": None,
                            }],
                            "usage": usage_info if 'usage_info' in locals() else {
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "total_tokens": 0
                            },
                            "created": int(time.time()),
                            "id": transformed["body"]["chat_id"],
                            "model": request.model,
                            "object": "chat.completion.chunk",
                            "system_fingerprint": "fp_zai_001",
                        }
                        yield f"data: {json_lib.dumps(finish_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        # 流式响应正常完成
                        debug_log("[REQUEST] 流式响应完成")
                        return

                except Exception as e:
                    debug_log("流处理错误", error=str(e))
                    retry_count += 1
                    last_error = str(e)
                    
                    # 网络连接错误时尝试切换代理（仅限 failover 策略）
                    if _proxy_list and "connect" in str(e).lower():
                        await switch_proxy_on_failure()

                    if retry_count > settings.MAX_RETRIES:
                        error_response = {
                            "error": {
                                "message": f"Stream processing failed: {last_error}",
                                "type": "stream_error"
                            }
                        }
                        yield f"data: {json_lib.dumps(error_response)}\n\n"
                        yield "data: [DONE]\n\n"
                        # 流式响应错误完成
                        debug_log("[REQUEST] 流式响应错误")
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
        debug_log("[REQUEST] 处理HTTPException")
        raise
    except Exception as e:
        debug_log("处理请求时发生错误", error=str(e))
        debug_log("[REQUEST] 处理异常")
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


async def handle_non_stream_request(request: OpenAIRequest, transformed: dict, enable_toolify: bool = False, request_client: httpx.AsyncClient = None) -> dict:
    """
    处理非流式请求
    
    Args:
        request: OpenAI请求对象
        transformed: 转换后的请求
        enable_toolify: 是否启用工具调用
        request_client: 请求专用的HTTP客户端
    
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
                    status_code=last_status_code
                )
                debug_log(
                    f"[RETRY] 非流式请求重试 ({retry_count}/{settings.MAX_RETRIES})",
                    retry_count=retry_count,
                    delay=f"{delay:.2f}s"
                )
                await asyncio.sleep(delay)
            
            # 使用提供的客户端，或获取复用的客户端
            if request_client:
                client = request_client
            else:
                client = await get_request_client()
            
            # 发起流式请求（上游始终返回SSE流，带性能追踪）
            # 使用转换后的headers（包含Accept-Encoding）
            headers = transformed["config"]["headers"].copy()
            
            request_start_time = time.perf_counter()
            async with client.stream(
                "POST",
                transformed["config"]["url"],
                json=transformed["body"],
                headers=headers,
            ) as response:
                # 记录非流式请求的TTFB
                ttfb = (time.perf_counter() - request_start_time) * 1000
                debug_log(f"⏱️ 非流式上游TTFB", ttfb_ms=f"{ttfb:.2f}ms")
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
                    retryable_codes = [400, 401, 405, 429, 502, 503, 504]
                    if response.status_code in retryable_codes and retry_count < settings.MAX_RETRIES:
                        last_status_code = response.status_code
                        last_error = f"{response.status_code}: {error_msg}"
                        retry_count += 1
                        
                        # 检查当前是否使用匿名Token
                        from .token_pool import get_token_pool
                        token_pool = get_token_pool()
                        current_token = transformed.get("token", "")
                        is_anonymous = token_pool.is_anonymous_token(current_token)
                        
                        if is_anonymous:
                            # 匿名Token：任何错误都清理缓存并重新获取
                            debug_log(f"[ANONYMOUS-NONSTREAM] 检测到匿名Token错误 {response.status_code}，清理缓存并重新获取")
                            await transformer.clear_anonymous_token_cache()
                            
                            # 清理客户端缓存，强制下次重新创建
                            debug_log("[CLIENT-NONSTREAM] 清理HTTP客户端缓存，下次请求将创建新客户端")
                            await cleanup_clients()
                            
                            # 重新获取客户端
                            if request_client:
                                request_client = await get_request_client()
                                client = request_client
                            else:
                                client = await get_request_client()
                            
                            # 使用新的匿名Token重新生成请求
                            request_dict = request.model_dump()
                            transformed = await transformer.transform_request_in(request_dict, client=client)
                            debug_log("[OK-NONSTREAM] 已获取新的匿名Token并重新生成请求")
                        else:
                            # 配置Token：按原有逻辑处理
                            if response.status_code in [400, 401, 405]:
                                debug_log(f"[CONFIG-NONSTREAM] 配置Token错误 {response.status_code}，切换到下一个Token")
                                new_token = await transformer.switch_token()
                                transformer.refresh_header_template()
                                request_dict = request.model_dump()
                                transformed = await transformer.transform_request_in(request_dict, client=client)
                                debug_log(f"[OK-NONSTREAM] 已切换到下一个配置Token")
                            
                            # 网络错误时尝试切换代理（仅限 failover 策略）
                            if response.status_code in [502, 503, 504] and _proxy_list:
                                await switch_proxy_on_failure()
                        
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
                    if not data_str or data_str in ("[DONE]", "DONE", "done"):
                        continue
                    
                    # 解析 SSE 数据块
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
                    
                    # 记录用量（通常在最后块中出现）
                    if data.get("usage"):
                        try:
                            usage_info = data["usage"]
                        except Exception:
                            pass
                    
                    # 处理tool_call阶段（提取搜索信息）
                    if phase == "tool_call":
                        edit_content = data.get("edit_content", "")
                        
                        # 提取搜索查询信息并添加到最终内容
                        if edit_content and "<glm_block" in edit_content and "search" in edit_content:
                            try:
                                import re
                                # 先尝试直接解码Unicode
                                decoded = edit_content
                                try:
                                    decoded = edit_content.encode('utf-8').decode('unicode_escape').encode('latin1').decode('utf-8')
                                except:
                                    try:
                                        import codecs
                                        decoded = codecs.decode(edit_content, 'unicode_escape')
                                    except:
                                        pass
                                
                                # 提取queries数组
                                queries_match = re.search(r'"queries":\s*\[(.*?)\]', decoded)
                                if queries_match:
                                    queries_str = queries_match.group(1)
                                    queries = re.findall(r'"([^"]+)"', queries_str)
                                    if queries:
                                        search_info = "🔍 **搜索：** " + "　".join(queries[:5])
                                        final_content += f"\n\n{search_info}\n\n"
                                        debug_log(f"[非流式] 提取到搜索信息: {queries}")
                            except Exception as e:
                                debug_log(f"[非流式] 提取搜索信息失败: {e}")
                        
                        continue
                    
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
                
                # 工具调用检测（非流式模式）
                if enable_toolify and final_content:
                    debug_log("[TOOLIFY] 检查非流式响应中的工具调用")
                    tool_response = parse_toolify_response(final_content, request.model)
                    if tool_response:
                        debug_log("[TOOLIFY] 非流式响应中检测到工具调用")
                        # 返回包含tool_calls的响应
                        return {
                            "id": transformed["body"]["chat_id"],
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "message": tool_response,
                                    "finish_reason": "tool_calls"
                                }
                            ],
                            "usage": usage_info
                        }
                
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
            
            # 网络连接错误时尝试切换代理（仅限 failover 策略）
            if _proxy_list and "connect" in str(e).lower():
                await switch_proxy_on_failure()
            
            if retry_count > settings.MAX_RETRIES:
                raise HTTPException(
                    status_code=500,
                    detail=f"Non-stream processing failed after {settings.MAX_RETRIES} retries: {last_error}"
                )
    
    # 不应该到达这里
    raise HTTPException(status_code=500, detail="Unexpected error in non-stream processing")
