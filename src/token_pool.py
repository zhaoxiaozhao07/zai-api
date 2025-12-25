#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Token池管理模块 - 管理配置Token的轮询和状态跟踪
"""

import os
import asyncio
import random
import time
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from .helpers import error_log, info_log, debug_log, bind_request_context, reset_request_context
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


class TokenPool:
    """Token池管理类 - 管理配置Token的轮询和状态跟踪"""
    
    # 类级别单例状态（封装全局变量）
    _instance: Optional['TokenPool'] = None
    _instance_lock: Optional[asyncio.Lock] = None  # 延迟初始化
    
    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        """获取实例锁（延迟初始化，避免模块导入时的事件循环问题）"""
        if cls._instance_lock is None:
            cls._instance_lock = asyncio.Lock()
        return cls._instance_lock
    
    @classmethod
    async def get_instance(cls) -> 'TokenPool':
        """
        获取全局 TokenPool 实例（异步线程安全的单例模式）
        
        封装了原来的全局变量 `_token_pool_instance` 和 `_token_pool_lock`
        
        Returns:
            TokenPool: 全局唯一的 TokenPool 实例
        """
        if cls._instance is None:
            async with cls._get_lock():
                # 双重检查锁定，防止并发创建多个实例
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    @classmethod
    async def reset_instance(cls):
        """重置单例实例（用于测试）"""
        async with cls._get_lock():
            cls._instance = None

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
        self._in_fallback: bool = False  # 降级模式标志
        # 注意：移除了 threading.Lock，依赖 Python GIL 保证单个字段赋值的原子性

        self._load_tokens()
    
    def _load_tokens(self):
        """从.env和tokens.txt加载token并去重（ZAI_TOKEN是必须的）"""
        token_set = set()
        
        # 1. 从环境变量ZAI_TOKEN加载（必须配置）
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
            # 有配置的token
            info_log("Token 池初始化完成", total=len(self.tokens))

            # 初始化 Token 状态（会设置 current_token 和 current_index）
            self._init_token_statuses()
        else:
            # 没有配置token，报错
            error_log("[ERROR] 未配置任何Token，请设置ZAI_TOKEN环境变量或在tokens.txt中添加Token")

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
        获取Token
        优先级：可用的配置Token

        Args:
            http_client: 外部传入的HTTP客户端（保留参数兼容性）

        Returns:
            str: 可用的Token

        Raises:
            ValueError: 所有Token都不可用时抛出
        """
        # 如果有配置的Token且当前Token可用，使用它
        async with self._switch_lock:
            if (self.tokens and self.current_token and
                self.current_token in self.token_statuses and
                self.token_statuses[self.current_token].is_available):
                debug_log(f"[TOKEN] 使用配置Token (索引: {self.current_index}/{len(self.tokens)})")
                return self.current_token

        # 所有Token都不可用
        raise ValueError("[ERROR] 无法获取任何可用的Token，请检查配置或网络连接")
    
    
    async def switch_to_next(self) -> Optional[str]:
        """
        切换到下一个可用的配置Token（轮询）
        使用异步锁确保并发安全

        Returns:
            str: 下一个Token，如果所有Token都不可用则返回None
        """
        async with self._switch_lock:
            # 如果在降级模式，尝试恢复配置 Token
            if self._in_fallback:
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

            # 所有 Token 都不可用，标记降级
            self._in_fallback = True
            info_log("[FALLBACK] 所有配置Token不可用")
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
        if token not in self.token_statuses:
            return
            
        status = self.token_statuses[token]
        status.total_requests += 1
        status.successful_requests += 1
        status.failure_count = 0
        status.last_failure_time = 0.0  # 重置失败时间

        # 如果之前被禁用，现在恢复
        was_disabled = not status.is_available
        if was_disabled:
            status.is_available = True

        # 退出降级模式
        was_in_fallback = self._in_fallback
        if was_disabled and was_in_fallback:
            self._in_fallback = False

        # 日志输出
        if was_disabled:
            info_log(f"[RECOVER] Token恢复可用: {token[:20]}...")
            if was_in_fallback:
                info_log("[RECOVERY] 配置Token已恢复，退出降级模式")
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
        if token not in self.token_statuses:
            return
            
        status = self.token_statuses[token]
        status.total_requests += 1
        status.failure_count += 1
        status.last_failure_time = time.time()

        # 失败次数达到阈值，禁用 Token
        should_disable = status.failure_count >= self.failure_threshold
        if should_disable:
            status.is_available = False

        # 日志输出
        if should_disable:
            error_log(
                f"[DISABLE] Token 已禁用: {token[:20]}... "
                f"(连续失败 {status.failure_count} 次)"
            )
        else:
            info_log(
                f"[TOKEN] 失败 (失败计数: {status.failure_count}/{self.failure_threshold}, "
                f"成功率: {status.success_rate:.1%})"
            )

    def try_recover_tokens(self):
        """
        尝试恢复失败的 Token（在降级模式下定期调用）
        恢复条件：失败时间超过 30 分钟

        注意：这里只是标记为可用，实际验证在下次请求时进行
        如果恢复的 Token 仍然失败，会再次被禁用
        """
        if not self._in_fallback:
            return

        current_time = time.time()
        recovery_interval = 1800  # 30 分钟
        recovered_tokens = []

        for status in self.token_statuses.values():
            if (not status.is_available and
                current_time - status.last_failure_time > recovery_interval):
                status.is_available = True
                status.failure_count = 0
                recovered_tokens.append(status.token)

        # 如果恢复了至少一个 Token，退出降级模式
        if recovered_tokens:
            self._in_fallback = False

        # 日志输出
        if recovered_tokens:
            for token in recovered_tokens:
                info_log(f"[RECOVER] Token 已恢复: {token[:20]}...")
            info_log(f"[RECOVERY] 恢复了 {len(recovered_tokens)} 个 Token，退出降级模式")

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
        async with self._switch_lock:  # 重新加载时也需要异步锁
            self._load_tokens()

# 保留兼容性的全局函数（内部调用类方法）
async def get_token_pool() -> TokenPool:
    """
    获取全局 token 池实例（异步线程安全的单例模式）
    
    注意：这是为了向后兼容保留的函数，内部调用 TokenPool.get_instance()
    """
    return await TokenPool.get_instance()
