"""
Utility functions for the application
"""

import sys
import time
import logging
import structlog
from structlog import contextvars as struct_context
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Any, Optional
from .config import settings


# 配置structlog
def configure_structlog():
    """配置structlog日志系统"""
    processors = [
        struct_context.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=False),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    # 根据日志级别选择渲染器
    if settings.LOG_LEVEL == "debug":
        # 调试模式：使用彩色控制台输出
        processors.append(structlog.dev.ConsoleRenderer())
        log_level = logging.DEBUG
    elif settings.LOG_LEVEL == "info":
        # 信息模式：使用彩色控制台输出
        processors.append(structlog.dev.ConsoleRenderer())
        log_level = logging.INFO
    else:  # false
        # 禁用模式：使用JSON格式输出（但实际不会输出）
        processors.append(structlog.processors.JSONRenderer())
        log_level = logging.CRITICAL  # 只输出致命错误
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


# 初始化structlog
configure_structlog()

# 获取全局logger实例
_logger = structlog.get_logger()


def bind_request_context(**kwargs) -> None:
    """绑定结构化日志上下文，忽略空值。"""
    filtered = {k: v for k, v in kwargs.items() if v is not None}
    if filtered:
        struct_context.bind_contextvars(**filtered)


def reset_request_context(*keys: str) -> None:
    """清理指定上下文字段，未传入则清空全部。"""
    if keys:
        struct_context.unbind_contextvars(*keys)
    else:
        struct_context.clear_contextvars()


def error_log(message: str, *args, **kwargs) -> None:
    """
    错误日志记录函数（所有级别都输出）
    
    Args:
        message: 日志消息
        *args: 消息格式化参数（兼容旧版）
        **kwargs: 额外的结构化上下文字段
    """
    # 格式化消息（兼容旧版用法）
    if args:
        formatted_message = message % args
    else:
        formatted_message = message
    
    # 使用structlog记录错误日志
    _logger.error(formatted_message, **kwargs)


def info_log(message: str, *args, **kwargs) -> None:
    """
    信息日志记录函数（info和debug级别输出）
    
    Args:
        message: 日志消息
        *args: 消息格式化参数（兼容旧版）
        **kwargs: 额外的结构化上下文字段
    """
    if settings.LOG_LEVEL in ["info", "debug"]:
        # 格式化消息（兼容旧版用法）
        if args:
            formatted_message = message % args
        else:
            formatted_message = message
        
        # 使用structlog记录信息日志
        _logger.info(formatted_message, **kwargs)


def debug_log(message: str, *args, **kwargs) -> None:
    """
    调试日志记录函数（仅debug级别输出）
    
    Args:
        message: 日志消息
        *args: 消息格式化参数（兼容旧版）
        **kwargs: 额外的结构化上下文字段
    """
    if settings.LOG_LEVEL == "debug":
        # 格式化消息（兼容旧版用法）
        if args:
            formatted_message = message % args
        else:
            formatted_message = message
        
        # 使用structlog记录调试日志
        _logger.debug(formatted_message, **kwargs)


def request_stage_log(stage: str, message: str, **kwargs) -> None:
    """
    Log info-level request stage transitions without dumping payload data.

    Args:
        stage: Logical stage identifier (e.g. "received", "upstream_request").
        message: Human readable description for terminal viewers.
        **kwargs: Extra structured fields to enrich the log.
    """
    normalized_stage = (stage or "unknown").strip().lower().replace(" ", "_")
    info_log(f"[REQUEST] {message}", stage=normalized_stage, **kwargs)


def get_logger(name: str = None):
    """
    获取一个structlog logger实例
    
    Args:
        name: logger名称（可选）
        
    Returns:
        structlog BoundLogger实例
    """
    if name:
        return structlog.get_logger(name)
    return _logger


# ============================================================================
# 性能追踪工具
# ============================================================================

@contextmanager
def perf_timer(operation_name: str, log_result: bool = True, threshold_ms: float = 0):
    """
    性能计时上下文管理器
    
    Args:
        operation_name: 操作名称
        log_result: 是否记录结果到日志
        threshold_ms: 仅记录超过此阈值的操作（毫秒），0表示记录所有
        
    Yields:
        包含elapsed_ms的字典，可在上下文中使用
        
    Example:
        with perf_timer("token_decode") as timer:
            result = decode_token(token)
        print(f"耗时: {timer['elapsed_ms']:.2f}ms")
    """
    timer_dict = {"elapsed_ms": 0, "elapsed_s": 0}
    start_time = time.perf_counter()
    
    try:
        yield timer_dict
    finally:
        elapsed_s = time.perf_counter() - start_time
        elapsed_ms = elapsed_s * 1000
        timer_dict["elapsed_ms"] = elapsed_ms
        timer_dict["elapsed_s"] = elapsed_s
        
        if log_result and elapsed_ms >= threshold_ms:
            debug_log(
                f"⏱️ {operation_name}",
                elapsed_ms=f"{elapsed_ms:.2f}ms",
                elapsed_s=f"{elapsed_s:.4f}s"
            )


def perf_track(operation_name: Optional[str] = None, log_result: bool = True, threshold_ms: float = 0):
    """
    性能追踪装饰器
    
    Args:
        operation_name: 操作名称，默认使用函数名
        log_result: 是否记录结果到日志
        threshold_ms: 仅记录超过此阈值的操作（毫秒），0表示记录所有
        
    Example:
        @perf_track("decode_jwt")
        def decode_token(token):
            ...
    """
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or func.__name__
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            with perf_timer(op_name, log_result, threshold_ms) as timer:
                result = func(*args, **kwargs)
            return result
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            with perf_timer(op_name, log_result, threshold_ms) as timer:
                result = await func(*args, **kwargs)
            return result
        
        # 根据函数类型返回相应的wrapper
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator

