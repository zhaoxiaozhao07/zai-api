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
                    print(f"[PROXY] 从proxys.txt加载了 {len(file_proxies)} 个代理")
        except Exception as e:
            print(f"[WARN] 读取proxys.txt失败: {e}")
    
    # 去重后的代理列表
    proxy_list = list(proxy_set)
    
    if proxy_list:
        print(f"[OK] 代理池初始化完成，共 {len(proxy_list)} 个唯一代理")
        # for i, proxy in enumerate(proxy_list, 1):
        #     print(f"  代理 {i}: {proxy}")
    
    return proxy_list


class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    API_ENDPOINT: str = os.getenv("API_ENDPOINT", "https://chat.z.ai/api/chat/completions")
    AUTH_TOKEN: str = os.getenv("AUTH_TOKEN", "sk-your-api-key")
    
    # Z.AI Token Configuration
    ZAI_TOKEN: str = os.getenv("ZAI_TOKEN", "")
    ZAI_SIGNING_SECRET: str = os.getenv("ZAI_SIGNING_SECRET", "junjie")
    
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
    DEBUG_LOGGING: bool = os.getenv("DEBUG_LOGGING", "true").lower() == "true"
    
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

