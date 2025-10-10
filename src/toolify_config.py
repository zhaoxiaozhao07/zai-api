"""
Toolify 配置模块
管理工具调用功能的配置和实例
"""

import logging
from typing import Optional
from .config import settings
from .toolify import ToolifyCore

logger = logging.getLogger(__name__)

# 全局 Toolify 实例
_toolify_instance: Optional[ToolifyCore] = None


def get_toolify() -> Optional[ToolifyCore]:
    """
    获取 Toolify 实例（单例模式）
    
    Returns:
        ToolifyCore实例，如果功能未启用则返回None
    """
    global _toolify_instance
    
    if not settings.ENABLE_TOOLIFY:
        logger.debug("[TOOLIFY] 工具调用功能已禁用")
        return None
    
    if _toolify_instance is None:
        _toolify_instance = ToolifyCore(enable_function_calling=True)
        logger.info("[TOOLIFY] 工具调用功能已启用并初始化")
    
    return _toolify_instance


def is_toolify_enabled() -> bool:
    """检查 Toolify 功能是否启用"""
    return settings.ENABLE_TOOLIFY

