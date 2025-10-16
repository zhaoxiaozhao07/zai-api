"""
FastAPI application configuration module
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional
from pydantic_settings import BaseSettings

# 加载.env文件,覆盖电脑自身环境变量，哪怕为空也要加载
load_dotenv(override=True)


def _load_proxy_list() -> list:
    """
    从环境变量和proxys.txt文件加载代理列表，并去重合并

    Returns:
        list: 去重后的代理列表
    """
    proxy_set = set()

    # 1. 从环境变量加载代理（支持HTTP_PROXY和HTTPS_PROXY，优先HTTPS_PROXY）
    https_proxy_raw = os.getenv("HTTPS_PROXY", "")
    http_proxy_raw = os.getenv("HTTP_PROXY", "")

    # 处理HTTPS代理（优先）
    if https_proxy_raw:
        env_proxies = [p.strip() for p in https_proxy_raw.split(",") if p.strip()]
        proxy_set.update(env_proxies)

    # 处理HTTP代理
    if http_proxy_raw:
        env_proxies = [p.strip() for p in http_proxy_raw.split(",") if p.strip()]
        proxy_set.update(env_proxies)

    # 2. 从proxys.txt文件加载代理（可选）
    proxys_file = Path("proxys.txt")
    if proxys_file.exists():
        try:
            with open(proxys_file, 'r', encoding='utf-8') as f:
                file_proxies = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
                proxy_set.update(file_proxies)
                if file_proxies:
                    print(f"[INFO] 从proxys.txt加载了 {len(file_proxies)} 个代理")
        except Exception as e:
            print(f"[ERROR] 读取proxys.txt失败: {e}")

    # 去重后的代理列表
    proxy_list = list(proxy_set)

    if proxy_list:
        print(f"[INFO] 代理池初始化完成，共 {len(proxy_list)} 个唯一代理")
        # for i, proxy in enumerate(proxy_list, 1):
        #     print(f"  代理 {i}: {proxy}")

    return proxy_list


def _load_upstream_list() -> list:
    """
    从环境变量和upstreams.txt文件加载上游地址列表,并去重合并

    Returns:
        list: 去重后的上游地址列表
    """
    upstream_set = set()

    # 1. 从环境变量加载上游地址(支持逗号分隔多个)
    api_endpoint_raw = os.getenv("API_ENDPOINT", "")

    if api_endpoint_raw:
        env_upstreams = [u.strip() for u in api_endpoint_raw.split(",") if u.strip()]
        upstream_set.update(env_upstreams)

    # 2. 从upstreams.txt文件加载上游地址(可选)
    upstreams_file = Path("upstreams.txt")
    if upstreams_file.exists():
        try:
            with open(upstreams_file, 'r', encoding='utf-8') as f:
                file_upstreams = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
                upstream_set.update(file_upstreams)
                if file_upstreams:
                    print(f"[INFO] 从upstreams.txt加载了 {len(file_upstreams)} 个上游地址")
        except Exception as e:
            print(f"[ERROR] 读取upstreams.txt失败: {e}")

    # 去重后的上游地址列表
    upstream_list = list(upstream_set)

    if upstream_list:
        print(f"[INFO] 上游地址池初始化完成,共 {len(upstream_list)} 个唯一地址")
        # for i, upstream in enumerate(upstream_list, 1):
        #     print(f"  上游 {i}: {upstream}")

    return upstream_list


class Settings(BaseSettings):
    """Application settings"""

    # API Configuration - 上游地址配置(支持多个地址)
    # 从环境变量和upstreams.txt文件加载上游地址列表,并去重合并
    _upstream_list = _load_upstream_list()

    # 统一的上游地址列表(合并去重后的结果)
    UPSTREAM_LIST: list = _upstream_list

    # 为了向后兼容,保留单地址配置(使用列表中的第一个)
    API_ENDPOINT: str = _upstream_list[0] if _upstream_list else "https://chat.z.ai/api/chat/completions"

    # 上游地址策略: failover(失败切换) 或 round-robin(轮询)
    UPSTREAM_STRATEGY: str = os.getenv("UPSTREAM_STRATEGY", "round-robin").lower()

    AUTH_TOKEN: str = os.getenv("AUTH_TOKEN", "sk-your-api-key")
    
    # Z.AI Token Configuration
    ZAI_TOKEN: str = os.getenv("ZAI_TOKEN", "")
    ZAI_SIGNING_SECRET: str = os.getenv("ZAI_SIGNING_SECRET", "junjie")
    ZAI_FE_VERSION: str = os.getenv("ZAI_FE_VERSION", "prod-fe-1.0.99")
    
    # Anonymous Token Configuration - 匿名Token配置
    ENABLE_GUEST_TOKEN: bool = os.getenv("ENABLE_GUEST_TOKEN", "true").lower() == "true"
    GUEST_TOKEN_CACHE_MINUTES: int = int(os.getenv("GUEST_TOKEN_CACHE_MINUTES", "30"))
    ZAI_AUTH_ENDPOINT: str = os.getenv("ZAI_AUTH_ENDPOINT", "https://chat.z.ai/api/v1/auths/")
    
    # Model Configuration
    PRIMARY_MODEL: str = os.getenv("PRIMARY_MODEL", "GLM-4.5")
    THINKING_MODEL: str = os.getenv("THINKING_MODEL", "GLM-4.5-Thinking")
    SEARCH_MODEL: str = os.getenv("SEARCH_MODEL", "GLM-4.5-Search")
    AIR_MODEL: str = os.getenv("AIR_MODEL", "GLM-4.5-Air")
    GLM_45V_MODEL: str = os.getenv("GLM_45V_MODEL", "GLM-4.5V")
    GLM_46_MODEL: str = os.getenv("GLM_46_MODEL", "GLM-4.6")
    GLM_46_THINKING_MODEL: str = os.getenv("GLM_46_THINKING_MODEL", "GLM-4.6-Thinking")
    GLM_46_SEARCH_MODEL: str = os.getenv("GLM_46_SEARCH_MODEL", "GLM-4.6-Search")
    
    # Server Configuration
    LISTEN_PORT: int = int(os.getenv("LISTEN_PORT", "8080"))
    
    # Logging Configuration - 支持三个等级：false, info, debug
    _log_level_str: str = os.getenv("LOG_LEVEL", "info").lower()
    LOG_LEVEL: str = _log_level_str if _log_level_str in ["false", "info", "debug"] else "info"
    
    # 向后兼容旧的DEBUG_LOGGING配置
    DEBUG_LOGGING: bool = LOG_LEVEL == "debug"
    
    # Feature Configuration
    SKIP_AUTH_TOKEN: bool = os.getenv("SKIP_AUTH_TOKEN", "false").lower() == "true"
    
    # Request Configuration
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    
    # Proxy Configuration - 代理配置（支持多个代理）
    # 从环境变量和proxys.txt文件加载代理列表，并去重合并
    _proxy_list = _load_proxy_list()
    
    # 统一的代理列表（合并去重后的结果）
    PROXY_LIST: list = _proxy_list
    
    # 为了向后兼容，保留原有的配置方式
    HTTP_PROXY_LIST: list = _proxy_list
    HTTPS_PROXY_LIST: list = _proxy_list
    
    # 保留单代理配置的兼容性
    HTTP_PROXY: Optional[str] = _proxy_list[0] if _proxy_list else None
    HTTPS_PROXY: Optional[str] = _proxy_list[0] if _proxy_list else None
    
    # 代理策略：failover（失败切换）或 round-robin（轮询）
    PROXY_STRATEGY: str = os.getenv("PROXY_STRATEGY", "failover").lower()
    
    # Toolify Configuration - 工具调用功能配置
    ENABLE_TOOLIFY: bool = os.getenv("ENABLE_TOOLIFY", "true").lower() == "true"
    TOOLIFY_CUSTOM_PROMPT: Optional[str] = os.getenv("TOOLIFY_CUSTOM_PROMPT")
    
    # class Config:
    #     env_file = ".env"


settings = Settings()

# Model Mapping Configuration - ZAI API模型映射
MODEL_MAPPING = {
    settings.PRIMARY_MODEL: "0727-360B-API",
    settings.THINKING_MODEL: "0727-360B-API",
    settings.SEARCH_MODEL: "0727-360B-API",
    settings.AIR_MODEL: "0727-106B-API",
    settings.GLM_45V_MODEL: "glm-4.5v",
    settings.GLM_46_MODEL: "GLM-4-6-API-V1",
    settings.GLM_46_THINKING_MODEL: "GLM-4-6-API-V1",
    settings.GLM_46_SEARCH_MODEL: "GLM-4-6-API-V1",
}

