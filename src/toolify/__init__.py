"""
Toolify 插件 - 为 LLM 提供工具调用能力
从 Toolify 项目提取的核心功能模块
"""

from .core import ToolifyCore
from .parser import parse_function_calls_xml, remove_think_blocks
from .detector import StreamingFunctionCallDetector
from .prompt import generate_function_prompt

__all__ = [
    'ToolifyCore',
    'parse_function_calls_xml',
    'remove_think_blocks',
    'StreamingFunctionCallDetector',
    'generate_function_prompt',
]

