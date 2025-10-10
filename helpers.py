"""
Utility functions for the application
"""

import sys
import logging
import structlog
from config import settings


# 配置structlog
def configure_structlog():
    """配置structlog日志系统"""
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=False),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    # 根据调试模式选择渲染器
    if settings.DEBUG_LOGGING:
        # 开发模式：使用彩色控制台输出
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        # 生产模式：使用JSON格式输出
        processors.append(structlog.processors.JSONRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.DEBUG if settings.DEBUG_LOGGING else logging.INFO
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


# 初始化structlog
configure_structlog()

# 获取全局logger实例
_logger = structlog.get_logger()


def debug_log(message: str, *args, **kwargs) -> None:
    """
    结构化日志记录函数，支持额外的上下文信息
    
    Args:
        message: 日志消息
        *args: 消息格式化参数（兼容旧版）
        **kwargs: 额外的结构化上下文字段
    """
    if settings.DEBUG_LOGGING:
        # 格式化消息（兼容旧版用法）
        if args:
            formatted_message = message % args
        else:
            formatted_message = message
        
        # 使用structlog记录日志
        _logger.debug(formatted_message, **kwargs)


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

