"""
Toolify 请求和响应处理模块
处理工具调用相关的请求预处理和响应解析
"""

import json
import logging
import uuid
from typing import Dict, Any, List, Optional

from .toolify_config import get_toolify, is_toolify_enabled
from .toolify.prompt import generate_function_prompt, safe_process_tool_choice
from .toolify.parser import parse_function_calls_xml
from .config import settings

logger = logging.getLogger(__name__)


def should_enable_toolify(request_dict: Dict[str, Any]) -> bool:
    """
    判断是否应该为当前请求启用工具调用功能
    
    Args:
        request_dict: 请求字典
        
    Returns:
        是否启用工具调用
    """
    if not is_toolify_enabled():
        return False
    
    # 检查请求中是否包含tools
    has_tools = request_dict.get("tools") and len(request_dict.get("tools", [])) > 0
    
    return has_tools


def prepare_toolify_request(request_dict: Dict[str, Any], messages: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], bool]:
    """
    准备带工具调用的请求
    
    Args:
        request_dict: 原始请求字典
        messages: 消息列表
        
    Returns:
        (处理后的消息列表, 是否启用了工具调用)
    """
    toolify = get_toolify()
    if not toolify:
        return messages, False
    
    tools = request_dict.get("tools")
    if not tools or len(tools) == 0:
        return messages, False
    
    logger.info(f"[TOOLIFY] 检测到 {len(tools)} 个工具定义，启用工具调用功能")
    
    # 预处理消息（转换tool和tool_calls）
    processed_messages = toolify.preprocess_messages(messages)
    logger.debug(f"[TOOLIFY] 消息预处理完成: {len(messages)} -> {len(processed_messages)}")
    
    # 生成工具调用提示词
    function_prompt, trigger_signal = generate_function_prompt(
        tools,
        toolify.trigger_signal,
        settings.TOOLIFY_CUSTOM_PROMPT
    )
    
    # 处理 tool_choice（传入 tools 以支持验证）
    tool_choice = request_dict.get("tool_choice")
    tool_choice_prompt = safe_process_tool_choice(tool_choice, tools)
    if tool_choice_prompt:
        function_prompt += tool_choice_prompt
    
    # 在消息开头注入系统提示词
    system_message = {"role": "system", "content": function_prompt}
    processed_messages.insert(0, system_message)
    
    logger.debug(f"[TOOLIFY] 已注入工具调用系统提示词，消息数: {len(processed_messages)}")
    
    return processed_messages, True


def parse_toolify_response(content: str, model: str) -> Optional[Dict[str, Any]]:
    """
    解析响应中的工具调用
    
    Args:
        content: 响应内容
        model: 模型名称
        
    Returns:
        如果检测到工具调用，返回包含tool_calls的响应字典；否则返回None
    """
    toolify = get_toolify()
    if not toolify:
        return None
    
    logger.debug(f"[TOOLIFY] 开始解析响应中的工具调用，内容长度: {len(content)}")
    
    # 解析 XML 格式的工具调用
    parsed_tools = parse_function_calls_xml(content, toolify.trigger_signal)
    
    if not parsed_tools:
        logger.debug("[TOOLIFY] 未检测到工具调用")
        return None
    
    logger.info(f"[TOOLIFY] 检测到 {len(parsed_tools)} 个工具调用")
    
    # 转换为 OpenAI 格式
    tool_calls = toolify.convert_parsed_tools_to_openai_format(parsed_tools)
    
    return {
        "tool_calls": tool_calls,
        "content": None,
        "role": "assistant"
    }


def format_toolify_response_for_stream(parsed_tools: List[Dict[str, Any]], model: str, chat_id: str) -> List[str]:
    """
    格式化工具调用为流式响应块
    
    Args:
        parsed_tools: 解析出的工具列表
        model: 模型名称
        chat_id: 会话ID
        
    Returns:
        SSE格式的响应块列表
    """
    toolify = get_toolify()
    if not toolify:
        return []
    
    tool_calls = toolify.convert_parsed_tools_to_openai_format(parsed_tools)
    chunks: List[str] = []
    
    # 初始块 - 发送角色和tool_calls
    initial_chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(uuid.uuid4().time_low),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls
            },
            "finish_reason": None
        }],
    }
    chunks.append(f"data: {json.dumps(initial_chunk)}\n\n")
    
    # 结束块
    final_chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(uuid.uuid4().time_low),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "tool_calls"
        }],
    }
    chunks.append(f"data: {json.dumps(final_chunk)}\n\n")
    chunks.append("data: [DONE]\n\n")
    
    return chunks

