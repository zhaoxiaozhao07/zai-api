#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Token池管理模块 - 支持配置Token和匿名Token的智能降级
"""

import os
import asyncio
import random
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta
from .helpers import debug_log
from .config import settings


def get_zai_dynamic_headers(chat_id: str = "") -> Dict[str, str]:
    """
    生成 Z.AI 特定的动态浏览器 headers
    使用 browserforge 生成真实的浏览器指纹
    
    Args:
        chat_id: 聊天 ID，用于生成正确的 Referer
        
    Returns:
        Dict[str, str]: 包含 Z.AI 特定配置的 headers
    """
    try:
        from browserforge.headers import HeaderGenerator
        
        # 使用 browserforge 生成真实的浏览器headers
        generator = HeaderGenerator()
        base_headers = generator.generate()
        
        # 提取关键信息
        user_agent = base_headers.get("user-agent", "")
        sec_ch_ua = base_headers.get("sec-ch-ua", "")
        
    except Exception as e:
        debug_log(f"[WARN] browserforge生成headers失败，使用备用方案: {e}")
        # 备用方案：使用简单的Chrome UA
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"
        sec_ch_ua = '"Not_A Brand";v="8", "Chromium";v="139", "Google Chrome";v="139"'
    
    # Z.AI 特定的 headers
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "Connection": "keep-alive",
        "Cache-Control": "no-cache",
        "User-Agent": user_agent,
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "X-FE-Version": "prod-fe-1.0.79",
        "Origin": "https://chat.z.ai",
    }
    
    # 添加浏览器特定的 sec-ch-ua headers
    if sec_ch_ua:
        headers["sec-ch-ua"] = sec_ch_ua
        headers["sec-ch-ua-mobile"] = "?0"
        headers["sec-ch-ua-platform"] = '"Windows"'
    
    # 根据 chat_id 设置 Referer
    if chat_id:
        headers["Referer"] = f"https://chat.z.ai/c/{chat_id}"
    else:
        headers["Referer"] = "https://chat.z.ai/"
    
    return headers


class TokenPool:
    """Token池管理类 - 支持配置Token和匿名Token的智能降级"""
    
    def __init__(self):
        """初始化Token池"""
        self.tokens: List[str] = []
        self.current_index: int = 0
        self.current_token: Optional[str] = None
        self.anonymous_mode: bool = False
        
        # 匿名Token缓存
        self._anonymous_token: Optional[str] = None
        self._anonymous_token_expires_at: Optional[datetime] = None
        self._fetch_lock = asyncio.Lock()
        
        # Token轮询锁 - 防止并发切换导致的竞态条件
        self._switch_lock = asyncio.Lock()
        
        self._load_tokens()
    
    def _load_tokens(self):
        """从.env和tokens.txt加载token并去重（ZAI_TOKEN现在是可选的）"""
        token_set = set()
        
        # 1. 从环境变量ZAI_TOKEN加载（现在是可选的）
        zai_token = os.getenv("ZAI_TOKEN", "").strip()
        if not zai_token:
            debug_log("[INFO] 未配置ZAI_TOKEN，将启用匿名Token模式")
            self.anonymous_mode = True
        else:
            # 处理多个token（逗号分割）
            env_tokens = [token.strip() for token in zai_token.split(",") if token.strip()]
            token_set.update(env_tokens)
            debug_log(f"从环境变量ZAI_TOKEN加载了 {len(env_tokens)} 个token")
        
        # 2. 从tokens.txt加载（可选）
        tokens_file = Path("tokens.txt")
        if tokens_file.exists():
            try:
                with open(tokens_file, 'r', encoding='utf-8') as f:
                    file_tokens = [line.strip() for line in f if line.strip()]
                    file_tokens_count = len(file_tokens)
                    token_set.update(file_tokens)
                    debug_log(f"从tokens.txt加载了 {file_tokens_count} 个token")
            except Exception as e:
                debug_log(f"[WARN] 读取tokens.txt失败: {e}")
        else:
            debug_log("tokens.txt文件不存在，跳过加载")
        
        # 去重后的token列表
        self.tokens = list(token_set)
        
        if self.tokens:
            # 有配置的token
            self.current_token = self.tokens[0]
            self.current_index = 0
            self.anonymous_mode = False
            debug_log(f"[OK] Token池初始化完成，共 {len(self.tokens)} 个唯一token")
        else:
            # 没有配置token，启用匿名模式
            self.anonymous_mode = True
            debug_log("[INFO] Token池为空，启用纯匿名Token模式")
    
    async def _fetch_anonymous_token(self, http_client=None) -> Optional[str]:
        """
        获取匿名Token（带简单重试机制）
        
        Args:
            http_client: 外部传入的HTTP客户端（可选）
            
        Returns:
            Optional[str]: 匿名Token，失败返回None
        """
        if not settings.ENABLE_GUEST_TOKEN:
            debug_log("[WARN] 匿名Token功能已禁用（ENABLE_GUEST_TOKEN=false）")
            return None
        
        max_retries = 3
        last_status_code = None
        
        for attempt in range(max_retries):
            try:
                headers = get_zai_dynamic_headers()
                
                # 使用外部传入的HTTP客户端，如果没有则创建临时客户端
                if http_client:
                    client = http_client
                    should_close = False
                else:
                    # 临时创建一个简单的客户端
                    import httpx
                    client = httpx.AsyncClient(timeout=httpx.Timeout(10.0))
                    should_close = True
                
                try:
                    response = await client.get(
                        settings.ZAI_AUTH_ENDPOINT,
                        headers=headers
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        token = data.get("token", "")
                        if token:
                            debug_log(f"[OK] 成功获取匿名Token: {token[:20]}...")
                            return token
                    else:
                        last_status_code = response.status_code
                        debug_log(f"[WARN] 获取匿名Token失败，状态码: {response.status_code}")
                        
                finally:
                    # 如果是临时创建的客户端，需要关闭
                    if should_close:
                        await client.aclose()
                        
            except Exception as e:
                debug_log(f"[ERROR] 获取匿名Token异常 (尝试 {attempt + 1}/{max_retries}): {e}")
            
            # 智能退避重试策略
            if attempt < max_retries - 1:
                base_delay = 1.0
                max_delay = 5.0
                # 指数退避加随机抖动
                exponential_delay = min(base_delay * (2 ** attempt), max_delay)
                jitter = exponential_delay * 0.25
                final_delay = exponential_delay + random.uniform(-jitter, jitter)
                debug_log(f"[RETRY] 等待 {final_delay:.2f} 秒后重试...")
                await asyncio.sleep(final_delay)
        
        # 全部失败
        debug_log("[ERROR] 匿名Token获取失败")
        return None
    
    async def _get_cached_anonymous_token(self) -> Optional[str]:
        """
        获取缓存的匿名Token（如果未过期）
        
        Returns:
            Optional[str]: 缓存的Token，如果过期或不存在返回None
        """
        if self._anonymous_token and self._anonymous_token_expires_at:
            if datetime.utcnow() < self._anonymous_token_expires_at:
                debug_log(f"[CACHE] 使用缓存的匿名Token（剩余 {(self._anonymous_token_expires_at - datetime.utcnow()).seconds}秒）")
                return self._anonymous_token
            else:
                debug_log("[CACHE] 匿名Token缓存已过期")
        return None
    
    async def _cache_anonymous_token(self, token: str):
        """
        缓存匿名Token
        
        Args:
            token: 要缓存的Token
        """
        self._anonymous_token = token
        cache_minutes = settings.GUEST_TOKEN_CACHE_MINUTES
        self._anonymous_token_expires_at = datetime.utcnow() + timedelta(minutes=cache_minutes)
        debug_log(f"[CACHE] 匿名Token已缓存，有效期 {cache_minutes} 分钟")
    
    async def get_anonymous_token(self, http_client=None) -> Optional[str]:
        """
        获取匿名Token（优先使用缓存，失效时自动创建新Token）
        使用异步锁防止并发请求
        
        Args:
            http_client: 外部传入的HTTP客户端（可选）
            
        Returns:
            Optional[str]: 匿名Token
        """
        # 先检查缓存
        cached_token = await self._get_cached_anonymous_token()
        if cached_token:
            return cached_token
        
        # 使用锁防止并发获取
        async with self._fetch_lock:
            # 双重检查：可能其他协程已经获取了
            cached_token = await self._get_cached_anonymous_token()
            if cached_token:
                return cached_token
            
            # 获取新的匿名Token
            debug_log("[FETCH] 开始获取新的匿名Token...")
            new_token = await self._fetch_anonymous_token(http_client)
            
            if new_token:
                await self._cache_anonymous_token(new_token)
                return new_token
            else:
                # 获取失败，直接返回None
                debug_log("[ERROR] 无法获取匿名Token")
                return None
    
    async def clear_anonymous_token_cache(self):
        """
        清理匿名Token缓存（当Token失效时调用）
        线程安全版本，使用异步锁保护
        """
        async with self._fetch_lock:  # 使用同一个锁，防止清理和获取的竞态条件
            debug_log("[CACHE] 安全清理匿名Token缓存")
            old_token = self._anonymous_token[:20] + "..." if self._anonymous_token else "None"
            self._anonymous_token = None
            self._anonymous_token_expires_at = None
            debug_log(f"[CACHE] 匿名Token缓存已清理，原Token: {old_token}")
    
    async def get_token(self, http_client=None) -> str:
        """
        获取Token（智能降级策略）
        优先级：配置Token → 匿名Token → 缓存Token
        
        Args:
            http_client: 外部传入的HTTP客户端（可选）
            
        Returns:
            str: 可用的Token
            
        Raises:
            ValueError: 所有Token获取方式都失败时抛出
        """
        # 策略1: 如果有配置的Token，优先使用
        if self.tokens and self.current_token:
            debug_log(f"[TOKEN] 使用配置Token (索引: {self.current_index}/{len(self.tokens)})")
            return self.current_token
        
        # 策略2: 没有配置Token或Token失效，使用匿名Token
        if settings.ENABLE_GUEST_TOKEN:
            debug_log("[TOKEN] 配置Token不可用，尝试获取匿名Token...")
            anonymous_token = await self.get_anonymous_token(http_client)
            if anonymous_token:
                return anonymous_token
        
        # 策略3: 所有方式都失败
        raise ValueError("[ERROR] 无法获取任何可用的Token，请检查配置或网络连接")
    
    
    async def switch_to_next(self) -> str:
        """
        切换到下一个配置Token（轮询）
        使用异步锁确保并发安全，防止竞态条件
        
        Returns:
            str: 下一个Token
        """
        async with self._switch_lock:  # 使用异步锁保护，确保原子操作
            if len(self.tokens) <= 1:
                debug_log("[WARN] 只有一个token，无法切换")
                return self.current_token if self.current_token else ""
            
            # 原子操作：切换到下一个token
            self.current_index = (self.current_index + 1) % len(self.tokens)
            self.current_token = self.tokens[self.current_index]
            
            debug_log(f"[SWITCH] 切换到下一个token (索引: {self.current_index}/{len(self.tokens)}): {self.current_token[:20]}...")
            return self.current_token
    
    async def switch_to_anonymous(self, http_client=None) -> Optional[str]:
        """
        切换到匿名Token模式
        
        Args:
            http_client: 外部传入的HTTP客户端（可选）
            
        Returns:
            Optional[str]: 匿名Token
        """
        debug_log("[SWITCH] 切换到匿名Token模式")
        return await self.get_anonymous_token(http_client)
    
    def get_pool_size(self) -> int:
        """获取配置Token池大小"""
        return len(self.tokens)
    
    def is_anonymous_mode(self) -> bool:
        """检查是否处于匿名模式"""
        return self.anonymous_mode
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取Token池状态信息
        
        Returns:
            Dict: 包含Token池状态的字典
        """
        status = {
            "configured_tokens": len(self.tokens),
            "current_index": self.current_index,
            "anonymous_mode": self.anonymous_mode,
            "guest_token_enabled": settings.ENABLE_GUEST_TOKEN,
        }
        
        if self._anonymous_token:
            status["anonymous_token_cached"] = True
            if self._anonymous_token_expires_at:
                remaining = (self._anonymous_token_expires_at - datetime.utcnow()).total_seconds()
                status["cache_remaining_seconds"] = max(0, int(remaining))
        else:
            status["anonymous_token_cached"] = False
        
        return status
    
    def is_anonymous_token(self, token: str) -> bool:
        """
        检测是否为匿名Token
        
        Args:
            token: 要检测的Token
            
        Returns:
            bool: 如果是匿名Token返回True
        """
        # 线程安全的读取，不需要锁
        return token == self._anonymous_token if self._anonymous_token else False
    
    async def reload(self):
        """重新加载token池"""
        debug_log("[RELOAD] 重新加载token池")
        async with self._switch_lock:  # 重新加载时也需要异步锁
            self._load_tokens()


# 全局token池实例（单例模式）
_token_pool_instance: Optional[TokenPool] = None


def get_token_pool() -> TokenPool:
    """获取全局token池实例（线程安全的单例模式）"""
    global _token_pool_instance
    if _token_pool_instance is None:
        _token_pool_instance = TokenPool()
    return _token_pool_instance
