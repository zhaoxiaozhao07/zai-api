#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
响应块构建器模块 - 封装所有 SSE 流式响应块构建逻辑

从 openai_service.py 拆分出来，提供统一的响应块构建接口
"""

import time
from typing import Dict, Optional, Any

from ..schemas import OpenAIRequest


class ChunkBuilder:
    """响应块构建器类，封装所有 SSE 块构建逻辑"""

    def build_role_chunk(self, json_lib, transformed: dict, request: OpenAIRequest) -> str:
        """构建角色初始化 chunk（第一个 SSE 块）"""
        return f"data: {json_lib.dumps({
            'id': 'chatcmpl-' + transformed['body']['chat_id'],
            'object': 'chat.completion.chunk',
            'created': int(time.time()),
            'model': request.model,
            'choices': [{
                'index': 0,
                'delta': {
                    'role': 'assistant',
                    'content': None,
                    'reasoning_content': None,
                },
                'logprobs': None,
                'finish_reason': None,
            }],
        })}\n\n"

    def build_content_chunk(
            self,
            json_lib,
            transformed: dict,
            request: OpenAIRequest,
            content: str,
    ) -> str:
        """构建正文内容 chunk"""
        return f"data: {json_lib.dumps({
            'id': 'chatcmpl-' + transformed['body']['chat_id'],
            'object': 'chat.completion.chunk',
            'created': int(time.time()),
            'model': request.model,
            'choices': [{
                'index': 0,
                'delta': {
                    'role': 'assistant',
                    'content': content,
                    'reasoning_content': None,
                },
                'logprobs': None,
                'finish_reason': None,
            }],
        })}\n\n"

    def build_reasoning_chunk(
            self,
            json_lib,
            transformed: dict,
            request: OpenAIRequest,
            reasoning_content: str,
    ) -> str:
        """构建包含 reasoning_content 的流式响应块（思考/推理内容）"""
        return f"data: {json_lib.dumps({
            'id': 'chatcmpl-' + transformed['body']['chat_id'],
            'object': 'chat.completion.chunk',
            'created': int(time.time()),
            'model': request.model,
            'choices': [{
                'index': 0,
                'delta': {
                    'role': 'assistant',
                    'content': None,
                    'reasoning_content': reasoning_content,
                },
                'logprobs': None,
                'finish_reason': None,
            }],
        })}\n\n"

    def build_usage_chunk(
            self, json_lib, transformed: dict, request: OpenAIRequest, usage: Dict[str, Any]
    ) -> str:
        """构建 usage 信息 chunk（携带 token 使用统计）"""
        normalized_usage = {
            'prompt_tokens': usage.get('prompt_tokens', 0),
            'total_tokens': usage.get('total_tokens', 0),
            'completion_tokens': usage.get('completion_tokens', 0),
            'estimated_cost': usage.get('estimated_cost'),
            'prompt_tokens_details': usage.get('prompt_tokens_details', {
                'cached_tokens': 0,
                'cache_write_tokens': None,
            }),
        }
        return f"data: {json_lib.dumps({
            'id': 'chatcmpl-' + transformed['body']['chat_id'],
            'object': 'chat.completion.chunk',
            'created': int(time.time()),
            'model': request.model,
            'choices': [],
            'usage': normalized_usage,
        })}\n\n"

    def build_finish_chunk(
            self,
            json_lib,
            transformed: dict,
            request: OpenAIRequest,
            usage: Optional[Dict] = None,
            finish_reason: str = 'stop'
    ) -> str:
        """构建结束 chunk（包含 finish_reason 和可选的 usage）"""
        chunk_usage = None
        if usage:
            chunk_usage = {
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0),
                'estimated_cost': usage.get('estimated_cost'),
                'prompt_tokens_details': usage.get('prompt_tokens_details', {
                    'cached_tokens': 0,
                    'cache_write_tokens': None,
                }),
            }
        return f"data: {json_lib.dumps({
            'id': 'chatcmpl-' + transformed['body']['chat_id'],
            'object': 'chat.completion.chunk',
            'created': int(time.time()),
            'model': request.model,
            'choices': [{
                'index': 0,
                'delta': {
                    'role': 'assistant',
                    'content': None,
                    'reasoning_content': None,
                    'tool_calls': None,
                },
                'logprobs': None,
                'finish_reason': finish_reason,
            }],
            'usage': chunk_usage,
        })}\n\n"


# 全局单例实例
chunk_builder = ChunkBuilder()
