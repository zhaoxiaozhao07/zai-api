#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main application entry point - OpenAI format conversion
"""

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.openai_api import router as openai_router

# Create FastAPI app
app = FastAPI(
    title="OpenAI Compatible API Server",
    description="OpenAI-compatible API server for Z.AI chat service",
    version="1.0.0-dev",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Include API router
app.include_router(openai_router)


@app.options("/")
async def handle_options():
    """Handle OPTIONS requests"""
    return Response(status_code=200)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "OpenAI Compatible API Server",
        "version": "1.0.0-dev",
        "description": "OpenAI格式转换功，含工具调用"
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    import os

    # 根据 CPU 核心数自动设置 workers（默认 4，可通过环境变量覆盖）
    workers = int(os.getenv("UVICORN_WORKERS", "4"))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.LISTEN_PORT,
        workers=workers,
        loop="uvloop",
        http="httptools",
        reload=False,
        log_level="info",
    )

