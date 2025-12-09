#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Token池管理模块 - 仅支持配置Token
"""

import os
import asyncio
import time
import threading
from typing import List, Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass
from .helpers import error_log, info_log, debug_log
from .config import settings


@dataclass
class TokenStatus:
    """Token 运行时状态"""
    token: str
    is_available: bool = True
    failure_count: int = 0
    last_failure_time: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0

    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests


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
        "X-FE-Version": settings.ZAI_FE_VERSION,
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
    """Token池管理类 - 仅支持配置Token"""

    def __init__(self):
        """初始化Token池"""
        self.tokens: List[str] = []
        self.current_index: int = 0
        self.current_token: Optional[str] = None

        # Token轮询锁 - 防止并发切换导致的竞态条件
        self._switch_lock = asyncio.Lock()

        # Token 状态跟踪
        self.token_statuses: Dict[str, TokenStatus] = {}
        self.failure_threshold: int = 3  # 失败 3 次后禁用
        self._status_lock = threading.Lock()  # 状态修改锁（同步锁，用于非异步方法）

        self._load_tokens()
    
    def _load_tokens(self):
        """从.env和tokens.txt加载token并去重（ZAI_TOKEN是必需的）"""
        token_set = set()
        
        # 1. 从环境变量ZAI_TOKEN加载
        zai_token = os.getenv("ZAI_TOKEN", "").strip()
        if zai_token:
            # 处理多个token（逗号分割）
            env_tokens = [token.strip() for token in zai_token.split(",") if token.strip()]
            token_set.update(env_tokens)
            info_log(
                "加载环境变量 Token",
                tokens=len(env_tokens),
            )
        
        # 2. 从tokens.txt加载（可选）
        tokens_file = Path("tokens.txt")
        if tokens_file.exists():
            try:
                with open(tokens_file, 'r', encoding='utf-8') as f:
                    file_tokens = [line.strip() for line in f if line.strip()]
                    file_tokens_count = len(file_tokens)
                    token_set.update(file_tokens)
                    info_log(
                        "加载 tokens.txt",
                        tokens=file_tokens_count,
                    )
            except Exception as e:
                error_log(f"[WARN] 读取tokens.txt失败: {e}")
        else:
            info_log("tokens.txt 文件不存在，跳过加载")
        
        # 去重后的token列表
        self.tokens = list(token_set)

        if self.tokens:
            info_log("Token 池初始化完成", total=len(self.tokens))
            # 初始化 Token 状态（会设置 current_token 和 current_index）
            self._init_token_statuses()
        else:
            # 没有配置token，报错
            error_log("[ERROR] 未配置任何Token，请设置 ZAI_TOKEN 环境变量或在 tokens.txt 中配置")
            raise ValueError("未配置任何Token，请设置 ZAI_TOKEN 环境变量")

    def _init_token_statuses(self):
        """初始化 Token 状态"""
        for token in self.tokens:
            self.token_statuses[token] = TokenStatus(token=token)

        # 设置第一个可用的 Token 为当前 Token
        if self.tokens:
            self.current_token = self.tokens[0]
            self.current_index = 0

        debug_log(f"[TOKEN] 初始化了 {len(self.token_statuses)} 个 Token 状态")
    
    async def get_token(self, http_client=None) -> str:
        """
        获取Token（仅使用配置Token）

        Args:
            http_client: 外部传入的HTTP客户端（保留参数兼容性）

        Returns:
            str: 可用的Token

        Raises:
            ValueError: 所有Token都不可用时抛出
        """
        async with self._switch_lock:
            if (self.tokens and self.current_token and
                self.current_token in self.token_statuses and
                self.token_statuses[self.current_token].is_available):
                debug_log(f"[TOKEN] 使用配置Token (索引: {self.current_index}/{len(self.tokens)})")
                return self.current_token

        # 尝试切换到下一个可用的Token
        next_token = await self.switch_to_next()
        if next_token:
            return next_token

        raise ValueError("[ERROR] 所有配置Token都不可用，请检查Token配置")
    
    async def switch_to_next(self) -> Optional[str]:
        """
        切换到下一个可用的配置Token（轮询）
        使用异步锁确保并发安全

        Returns:
            str: 下一个Token，如果所有Token都不可用则返回None
        """
        async with self._switch_lock:
            # 尝试恢复失败的 Token
            self.try_recover_tokens()

            # 从当前位置开始，查找下一个可用的 Token
            attempts = 0
            while attempts < len(self.tokens):
                self.current_index = (self.current_index + 1) % len(self.tokens)
                next_token = self.tokens[self.current_index]

                # 检查是否可用
                if next_token in self.token_statuses and self.token_statuses[next_token].is_available:
                    self.current_token = next_token
                    status = self.token_statuses[next_token]

                    available_count = sum(1 for s in self.token_statuses.values() if s.is_available)
                    info_log(
                        f"[SWITCH] 切换Token (成功率: {status.success_rate:.1%}, "
                        f"可用: {available_count}/{len(self.tokens)})"
                    )
                    return next_token

                attempts += 1

            # 所有 Token 都不可用
            error_log("[ERROR] 所有配置Token都不可用")
            return None
    
    def get_pool_size(self) -> int:
        """获取配置Token池大小"""
        return len(self.tokens)
    
    def mark_token_success(self, token: str):
        """
        标记 Token 使用成功

        Args:
            token: 成功的 Token
        """
        if token in self.token_statuses:
            with self._status_lock:
                status = self.token_statuses[token]
                status.total_requests += 1
                status.successful_requests += 1
                status.failure_count = 0
                status.last_failure_time = 0.0  # 重置失败时间

                # 如果之前被禁用，现在恢复
                was_disabled = not status.is_available
                if was_disabled:
                    status.is_available = True

            # 日志输出在锁外，避免阻塞
            if was_disabled:
                info_log(f"[RECOVER] Token恢复可用: {token[:20]}...")
            else:
                debug_log(
                    f"[TOKEN] 成功 (成功率: {status.success_rate:.1%}, "
                    f"总请求: {status.total_requests})"
                )

    def mark_token_failure(self, token: str):
        """
        标记 Token 使用失败

        Args:
            token: 失败的 Token
        """
        if token in self.token_statuses:
            with self._status_lock:
                status = self.token_statuses[token]
                status.total_requests += 1
                status.failure_count += 1
                status.last_failure_time = time.time()

                # 失败次数达到阈值，禁用 Token
                should_disable = status.failure_count >= self.failure_threshold
                if should_disable:
                    status.is_available = False

                failure_count = status.failure_count
                success_rate = status.success_rate

            # 日志输出在锁外，避免阻塞
            if should_disable:
                error_log(
                    f"[DISABLE] Token 已禁用: {token[:20]}... "
                    f"(连续失败 {failure_count} 次)"
                )
            else:
                info_log(
                    f"[TOKEN] 失败 (失败计数: {failure_count}/{self.failure_threshold}, "
                    f"成功率: {success_rate:.1%})"
                )

    def try_recover_tokens(self):
        """
        尝试恢复失败的 Token
        恢复条件：失败时间超过 30 分钟

        注意：这里只是标记为可用，实际验证在下次请求时进行
        如果恢复的 Token 仍然失败，会再次被禁用
        """
        current_time = time.time()
        recovery_interval = 1800  # 30 分钟
        recovered_tokens = []

        with self._status_lock:
            for status in self.token_statuses.values():
                if (not status.is_available and
                    current_time - status.last_failure_time > recovery_interval):
                    status.is_available = True
                    status.failure_count = 0
                    recovered_tokens.append(status.token)

        # 日志输出在锁外，避免阻塞
        if recovered_tokens:
            for token in recovered_tokens:
                info_log(f"[RECOVER] Token 已恢复: {token[:20]}...")
            info_log(f"[RECOVERY] 恢复了 {len(recovered_tokens)} 个 Token")

    def get_status(self) -> Dict[str, Any]:
        """
        获取Token池状态信息

        Returns:
            Dict: 包含Token池状态的字典
        """
        # 统计可用 Token
        available_count = sum(1 for s in self.token_statuses.values() if s.is_available)

        status = {
            "configured_tokens": len(self.tokens),
            "available_tokens": available_count,
            "current_index": self.current_index,
        }

        return status
    
    async def reload(self):
        """重新加载token池"""
        info_log("[RELOAD] 重新加载token池")
        async with self._switch_lock:
            self._load_tokens()


# 全局token池实例（单例模式）
_token_pool_instance: Optional[TokenPool] = None
_token_pool_lock = asyncio.Lock()


async def get_token_pool() -> TokenPool:
    """获取全局token池实例（异步线程安全的单例模式）"""
    global _token_pool_instance
    if _token_pool_instance is None:
        async with _token_pool_lock:
            # 双重检查锁定，防止并发创建多个实例
            if _token_pool_instance is None:
                _token_pool_instance = TokenPool()
    return _token_pool_instance
