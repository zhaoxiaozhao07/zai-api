"""Generalised HTTP client, proxy and upstream management helpers."""

from __future__ import annotations

import asyncio
from typing import Optional, Dict, Tuple

import httpx

from ..helpers import info_log, debug_log, error_log
from ..config import settings


_CONNECTION_POOL_CONFIG: Dict[str, object] = {
    "limits": httpx.Limits(
        max_keepalive_connections=20,
        max_connections=100,
        keepalive_expiry=30,
    ),
    "timeout": httpx.Timeout(
        connect=10.0,
        read=120.0,
        write=30.0,
        pool=10.0,
    ),
    "http2": True,
}


class NetworkManager:
    """Manage shared HTTP clients plus upstream/proxy selection."""

    def __init__(self) -> None:
        self._proxy_clients: Dict[str, httpx.AsyncClient] = {}
        self._default_client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()

        self._proxy_lock = asyncio.Lock()
        self._proxy_list = settings.PROXY_LIST
        self._proxy_index = 0

        self._upstream_lock = asyncio.Lock()
        self._upstream_list = settings.UPSTREAM_LIST
        self._upstream_index = 0

        if self._proxy_list:
            info_log(
                "[PROXY] 初始化代理池",
                count=len(self._proxy_list),
                strategy=settings.PROXY_STRATEGY,
            )

        if self._upstream_list:
            info_log(
                "[UPSTREAM] 初始化上游地址池",
                count=len(self._upstream_list),
                strategy=settings.UPSTREAM_STRATEGY,
            )
            for idx, upstream in enumerate(self._upstream_list):
                debug_log("[UPSTREAM] 地址", index=idx, url=upstream)

    async def get_or_create_client(self, proxy_url: Optional[str] = None) -> httpx.AsyncClient:
        async with self._client_lock:
            if proxy_url is None:
                if self._default_client is None:
                    info_log("[CLIENT] 创建默认客户端（无代理）")
                    self._default_client = httpx.AsyncClient(**_CONNECTION_POOL_CONFIG)
                return self._default_client

            if proxy_url not in self._proxy_clients:
                info_log("[CLIENT] 为代理创建新客户端", proxy=proxy_url)
                self._proxy_clients[proxy_url] = httpx.AsyncClient(
                    proxy=proxy_url,
                    **_CONNECTION_POOL_CONFIG,
                )

            return self._proxy_clients[proxy_url]

    async def cleanup_clients(self) -> None:
        async with self._client_lock:
            clients_to_close = list(self._proxy_clients.values())
            self._proxy_clients.clear()
            default = self._default_client
            self._default_client = None

        for client in clients_to_close:
            try:
                await client.aclose()
            except Exception as exc:  # pragma: no cover - 问题记录即可
                error_log("[CLIENT] 关闭代理客户端失败", error=str(exc))

        if default:
            try:
                await default.aclose()
                info_log("[CLIENT] 默认客户端已关闭")
            except Exception as exc:  # pragma: no cover
                error_log("[CLIENT] 关闭默认客户端失败", error=str(exc))

        info_log("[CLIENT] 所有客户端已清理")

    async def cleanup_current_client(self, proxy_url: Optional[str] = None) -> None:
        client_to_close = None

        async with self._client_lock:
            if proxy_url is None:
                if self._default_client:
                    client_to_close = self._default_client
                    self._default_client = None
                    info_log("[CLIENT] 标记默认客户端待清理")
            else:
                if proxy_url in self._proxy_clients:
                    client_to_close = self._proxy_clients.pop(proxy_url)
                    info_log("[CLIENT] 标记代理客户端待清理", proxy=proxy_url)

        if client_to_close:
            try:
                await client_to_close.aclose()
                info_log("[CLIENT] 客户端已关闭，下次请求将创建新连接")
            except Exception as exc:  # pragma: no cover
                error_log("[CLIENT] 关闭客户端失败", error=str(exc))

    async def get_request_client(self) -> Tuple[httpx.AsyncClient, Optional[str]]:
        proxy = await self.get_next_proxy()
        client = await self.get_or_create_client(proxy)
        return client, proxy

    async def get_next_proxy(self) -> Optional[str]:
        if not self._proxy_list:
            return None

        if settings.PROXY_STRATEGY == "round-robin":
            async with self._proxy_lock:
                proxy = self._proxy_list[self._proxy_index]
                self._proxy_index = (self._proxy_index + 1) % len(self._proxy_list)
                debug_log("[PROXY] Round-robin选择代理", index=self._proxy_index, proxy=proxy)
                return proxy

        async with self._proxy_lock:
            proxy = self._proxy_list[self._proxy_index]
            debug_log("[PROXY] Failover使用代理", index=self._proxy_index, proxy=proxy)
            return proxy

    async def switch_proxy_on_failure(self) -> None:
        if not self._proxy_list or settings.PROXY_STRATEGY != "failover":
            return

        async with self._proxy_lock:
            old_index = self._proxy_index
            self._proxy_index = (self._proxy_index + 1) % len(self._proxy_list)
            info_log(
                "[PROXY] 代理失败切换",
                old_index=old_index,
                new_index=self._proxy_index,
                proxy=self._proxy_list[self._proxy_index],
            )

    async def get_next_upstream(self) -> str:
        if not self._upstream_list:
            return settings.API_ENDPOINT

        if settings.UPSTREAM_STRATEGY == "round-robin":
            async with self._upstream_lock:
                upstream = self._upstream_list[self._upstream_index]
                self._upstream_index = (self._upstream_index + 1) % len(self._upstream_list)
                debug_log("[UPSTREAM] Round-robin选择上游", index=self._upstream_index, url=upstream)
                return upstream

        async with self._upstream_lock:
            upstream = self._upstream_list[self._upstream_index]
            debug_log("[UPSTREAM] Failover使用上游", index=self._upstream_index, url=upstream)
            return upstream

    async def switch_upstream_on_failure(self) -> None:
        if not self._upstream_list or settings.UPSTREAM_STRATEGY != "failover":
            return

        async with self._upstream_lock:
            old_index = self._upstream_index
            self._upstream_index = (self._upstream_index + 1) % len(self._upstream_list)
            info_log(
                "[UPSTREAM] 上游失败切换",
                old_index=old_index,
                new_index=self._upstream_index,
                upstream=self._upstream_list[self._upstream_index],
            )

    def has_proxy_pool(self) -> bool:
        return bool(self._proxy_list)

    def has_upstream_pool(self) -> bool:
        return bool(self._upstream_list)

    @property
    def proxy_strategy(self) -> str:
        return settings.PROXY_STRATEGY

    @property
    def upstream_strategy(self) -> str:
        return settings.UPSTREAM_STRATEGY


network_manager = NetworkManager()

