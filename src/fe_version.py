#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Z.AI前端版本自动获取模块

从 https://chat.z.ai 页面自动提取最新的 X-FE-Version 值。
该模块会缓存版本号（默认 30 分钟 TTL），避免频繁请求。
"""

from __future__ import annotations

import re
import time
from typing import Optional

import httpx

from .helpers import info_log, error_log, debug_log

# 获取版本的源URL
FE_VERSION_SOURCE_URL = "https://chat.z.ai"

# 缓存TTL（秒）- 默认30分钟
CACHE_TTL_SECONDS = 1800

# 版本号正则表达式（匹配 prod-fe-x.x.x 格式）
_version_pattern = re.compile(r"prod-fe-\d+\.\d+\.\d+")

# 缓存变量
_cached_version: str = ""
_cached_at: float = 0.0


def _extract_version(page_content: str) -> Optional[str]:
    """
    从页面内容中提取版本号
    
    Args:
        page_content: 页面HTML内容
        
    Returns:
        提取到的版本号，如果未找到则返回 None
    """
    if not page_content:
        return None
    
    matches = _version_pattern.findall(page_content)
    if not matches:
        return None
    
    # 选择最高版本号（按字典序排序）
    return max(matches)


def _should_use_cache(force_refresh: bool) -> bool:
    """
    判断是否应该使用缓存的版本号
    
    Args:
        force_refresh: 是否强制刷新
        
    Returns:
        True 表示可以使用缓存，False 表示需要重新获取
    """
    if force_refresh:
        return False
    if not _cached_version:
        return False
    if _cached_at <= 0:
        return False
    return (time.time() - _cached_at) < CACHE_TTL_SECONDS


def get_latest_fe_version(force_refresh: bool = False) -> str:
    """
    获取最新的 X-FE-Version 值
    
    查找顺序：
        1. 缓存的版本号（如果在 TTL 内）
        2. 从 chat.z.ai 获取最新版本
    
    Args:
        force_refresh: 是否强制刷新缓存
        
    Returns:
        版本号字符串
        
    Raises:
        Exception: 如果无法从远程获取版本号
    """
    global _cached_version, _cached_at
    
    # 检查是否可以使用缓存
    if _should_use_cache(force_refresh):
        debug_log("[FE-VERSION] 使用缓存的版本号", version=_cached_version)
        return _cached_version
    
    # 准备请求头（使用通用的 User-Agent）
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    
    try:
        # 从远程获取版本号
        debug_log("[FE-VERSION] 正在从远程获取版本号", url=FE_VERSION_SOURCE_URL)
        with httpx.Client(timeout=10.0, follow_redirects=True) as client:
            response = client.get(FE_VERSION_SOURCE_URL, headers=headers)
            response.raise_for_status()
            
            # 提取版本号
            version = _extract_version(response.text)
            if version:
                # 检查是否有更新
                if version != _cached_version:
                    info_log("[FE-VERSION] 检测到版本更新", new_version=version, old_version=_cached_version or "无")
                
                # 更新缓存
                _cached_version = version
                _cached_at = time.time()
                return version
            
            # 未找到版本号
            error_log("[FE-VERSION] 无法在页面中找到版本号")
            raise Exception("无法在页面中找到 X-FE-Version")
            
    except Exception as exc:
        error_log("[FE-VERSION] 获取版本号失败", error=str(exc), url=FE_VERSION_SOURCE_URL)
        raise Exception(f"获取 X-FE-Version 失败: {exc}")


def refresh_fe_version() -> str:
    """
    强制刷新缓存的版本号
    
    Returns:
        最新的版本号字符串
    """
    return get_latest_fe_version(force_refresh=True)


def get_fe_version_with_fallback(fallback: Optional[str] = None) -> Optional[str]:
    """
    获取版本号（带降级处理）
    
    如果无法从远程获取版本号，则返回 fallback 值。
    
    Args:
        fallback: 降级版本号（通常从环境变量读取）
        
    Returns:
        版本号字符串，如果获取失败且无 fallback 则返回 None
    """
    try:
        return get_latest_fe_version()
    except Exception as e:
        if fallback:
            info_log("[FE-VERSION] 使用降级版本号", fallback=fallback, error=str(e))
            return fallback
        error_log("[FE-VERSION] 无法获取版本号且无降级值", error=str(e))
        return None

