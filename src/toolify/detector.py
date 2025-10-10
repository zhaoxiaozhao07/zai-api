"""
Toolify 流式检测器
用于在流式响应中检测工具调用
"""

import logging
from typing import Optional, List, Dict, Any
from .parser import parse_function_calls_xml

logger = logging.getLogger(__name__)


class StreamingFunctionCallDetector:
    """
    增强型流式函数调用检测器，支持动态触发信号，避免在<think>标签内误判
    
    核心特性：
    1. 避免在<think>块内触发工具调用检测
    2. 正常输出<think>块内容给用户
    3. 支持嵌套think标签
    """
    
    def __init__(self, trigger_signal: str):
        self.trigger_signal = trigger_signal
        self.reset()
    
    def reset(self):
        """重置检测器状态"""
        self.content_buffer = ""
        self.state = "detecting"  # detecting, tool_parsing
        self.in_think_block = False
        self.think_depth = 0
        self.signal = self.trigger_signal
        self.signal_len = len(self.signal)
    
    def process_chunk(self, delta_content: str) -> tuple[bool, str]:
        """
        处理流式内容块
        
        Args:
            delta_content: 新的内容块
            
        Returns:
            (is_tool_call_detected, content_to_yield): 是否检测到工具调用，以及应该输出的内容
        """
        if not delta_content:
            return False, ""
        
        self.content_buffer += delta_content
        content_to_yield = ""
        
        if self.state == "tool_parsing":
            return False, ""
        
        if delta_content:
            logger.debug(f"[TOOLIFY] 处理块: {repr(delta_content[:50])}{'...' if len(delta_content) > 50 else ''}, 缓冲区长度: {len(self.content_buffer)}, think状态: {self.in_think_block}")
        
        i = 0
        while i < len(self.content_buffer):
            # 更新think状态
            skip_chars = self._update_think_state(i)
            if skip_chars > 0:
                for j in range(skip_chars):
                    if i + j < len(self.content_buffer):
                        content_to_yield += self.content_buffer[i + j]
                i += skip_chars
                continue
            
            # 在非think块中检测触发信号
            if not self.in_think_block and self._can_detect_signal_at(i):
                if self.content_buffer[i:i+self.signal_len] == self.signal:
                    logger.debug(f"[TOOLIFY] 在非think块中检测到触发信号! 信号: {self.signal[:20]}...")
                    logger.debug(f"[TOOLIFY] 触发信号位置: {i}, think状态: {self.in_think_block}, think深度: {self.think_depth}")
                    self.state = "tool_parsing"
                    self.content_buffer = self.content_buffer[i:]
                    return True, content_to_yield
            
            # 如果剩余内容不足以判断，保留在缓冲区
            remaining_len = len(self.content_buffer) - i
            if remaining_len < self.signal_len or remaining_len < 8:
                break
            
            content_to_yield += self.content_buffer[i]
            i += 1
        
        self.content_buffer = self.content_buffer[i:]
        return False, content_to_yield
    
    def _update_think_state(self, pos: int):
        """更新think标签状态，支持嵌套"""
        remaining = self.content_buffer[pos:]
        
        if remaining.startswith('<think>'):
            self.think_depth += 1
            self.in_think_block = True
            logger.debug(f"[TOOLIFY] 进入think块，深度: {self.think_depth}")
            return 7
        
        elif remaining.startswith('</think>'):
            self.think_depth = max(0, self.think_depth - 1)
            self.in_think_block = self.think_depth > 0
            logger.debug(f"[TOOLIFY] 退出think块，深度: {self.think_depth}")
            return 8
        
        return 0
    
    def _can_detect_signal_at(self, pos: int) -> bool:
        """检查是否可以在指定位置检测信号"""
        return (pos + self.signal_len <= len(self.content_buffer) and 
                not self.in_think_block)
    
    def finalize(self) -> Optional[List[Dict[str, Any]]]:
        """流结束时的最终处理"""
        if self.state == "tool_parsing":
            return parse_function_calls_xml(self.content_buffer, self.trigger_signal)
        return None

