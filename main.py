#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main application entry point - Simplified version for OpenAI format conversion only
"""

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.openai_api import router as openai_router

# Create FastAPI app
app = FastAPI(
    title="OpenAI Compatible API Server (Simplified)",
    description="A simplified OpenAI-compatible API server for Z.AI chat service",
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
        "message": "OpenAI Compatible API Server (Simplified)",
        "version": "1.0.0-dev",
        "description": "仅包含OpenAI格式转换功能，不含工具调用"
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.LISTEN_PORT,
        reload=False,
    )

