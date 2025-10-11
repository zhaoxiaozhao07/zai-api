"""
FastAPI application configuration module
"""

import os
from dotenv import load_dotenv
from typing import Optional
from pydantic_settings import BaseSettings

# 加载.env文件,覆盖电脑自身环境变量，哪怕为空也要加载
load_dotenv(override=True)


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
    
    # Proxy Configuration
    HTTP_PROXY: Optional[str] = os.getenv("HTTP_PROXY")
    HTTPS_PROXY: Optional[str] = os.getenv("HTTPS_PROXY")
    
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

