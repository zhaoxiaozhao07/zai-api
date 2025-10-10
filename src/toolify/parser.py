"""
Toolify XML 解析器
解析模型响应中的工具调用XML格式
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def remove_think_blocks(text: str) -> str:
    """
    临时移除所有 <think>...</think> 块用于XML解析
    支持嵌套think标签
    注意：此函数仅用于临时解析，不影响返回给用户的原始内容
    """
    while '<think>' in text and '</think>' in text:
        start_pos = text.find('<think>')
        if start_pos == -1:
            break
        
        pos = start_pos + 7
        depth = 1
        
        while pos < len(text) and depth > 0:
            if text[pos:pos+7] == '<think>':
                depth += 1
                pos += 7
            elif text[pos:pos+8] == '</think>':
                depth -= 1
                pos += 8
            else:
                pos += 1
        
        if depth == 0:
            text = text[:start_pos] + text[pos:]
        else:
            break
    
    return text


def parse_function_calls_xml(xml_string: str, trigger_signal: str) -> Optional[List[Dict[str, Any]]]:
    """
    增强型XML解析函数，支持动态触发信号
    
    1. 保留 <think>...</think> 块（它们应正常返回给用户）
    2. 解析时临时移除think块，防止干扰XML解析
    3. 查找触发信号的最后一次出现
    4. 从最后一个触发信号开始解析function_calls
    
    Args:
        xml_string: 包含XML的响应字符串
        trigger_signal: 触发信号字符串
        
    Returns:
        解析出的工具调用列表，格式为 [{"name": "tool_name", "args": {...}}, ...]
        如果没有找到工具调用，返回None
    """
    logger.debug(f"[TOOLIFY] 开始解析XML，输入长度: {len(xml_string) if xml_string else 0}")
    logger.debug(f"[TOOLIFY] 使用触发信号: {trigger_signal[:20]}...")
    
    if not xml_string or trigger_signal not in xml_string:
        logger.debug(f"[TOOLIFY] 输入为空或不包含触发信号")
        return None
    
    # 临时移除think块用于解析
    cleaned_content = remove_think_blocks(xml_string)
    logger.debug(f"[TOOLIFY] 移除think块后内容长度: {len(cleaned_content)}")
    
    # 查找所有触发信号位置
    signal_positions = []
    start_pos = 0
    while True:
        pos = cleaned_content.find(trigger_signal, start_pos)
        if pos == -1:
            break
        signal_positions.append(pos)
        start_pos = pos + 1
    
    if not signal_positions:
        logger.debug(f"[TOOLIFY] 在清理后的内容中未找到触发信号")
        return None
    
    logger.debug(f"[TOOLIFY] 找到 {len(signal_positions)} 个触发信号位置: {signal_positions}")
    
    # 使用最后一个触发信号位置
    last_signal_pos = signal_positions[-1]
    content_after_signal = cleaned_content[last_signal_pos:]
    logger.debug(f"[TOOLIFY] 从最后触发信号开始的内容: {repr(content_after_signal[:100])}")
    
    # 查找function_calls标签
    calls_content_match = re.search(r"<function_calls>([\s\S]*?)</function_calls>", content_after_signal)
    if not calls_content_match:
        logger.warning(f"[TOOLIFY] 未找到function_calls标签！内容: {repr(content_after_signal[:300])}")
        # 检查是否有不完整的function_calls开始标签
        if "<function_calls" in content_after_signal:
            logger.warning(f"[TOOLIFY] 发现不完整的function_calls开始标签，但没有结束标签")
        return None
    
    calls_content = calls_content_match.group(1)
    logger.debug(f"[TOOLIFY] function_calls内容: {repr(calls_content)}")
    
    # 解析所有function_call块
    results = []
    call_blocks = re.findall(r"<function_call>([\s\S]*?)</function_call>", calls_content)
    logger.debug(f"[TOOLIFY] 找到 {len(call_blocks)} 个function_call块")
    
    for i, block in enumerate(call_blocks):
        logger.debug(f"[TOOLIFY] 处理function_call #{i+1}: {repr(block)}")
        
        # 提取tool名称
        tool_match = re.search(r"<tool>(.*?)</tool>", block)
        if not tool_match:
            logger.debug(f"[TOOLIFY] 块 #{i+1} 中未找到tool标签")
            continue
        
        name = tool_match.group(1).strip()
        args = {}
        
        # 提取args块
        args_block_match = re.search(r"<args>([\s\S]*?)</args>", block)
        if args_block_match:
            args_content = args_block_match.group(1)
            # 支持包含连字符的参数标签名（如-i, -A）；匹配任何非空格、非'>'、非'/'字符
            arg_matches = re.findall(r"<([^\s>/]+)>([\s\S]*?)</\1>", args_content)

            def _coerce_value(v: str):
                """尝试将字符串值转换为JSON对象"""
                try:
                    return json.loads(v)
                except Exception:
                    pass
                return v

            for k, v in arg_matches:
                args[k] = _coerce_value(v)
        
        result = {"name": name, "args": args}
        results.append(result)
        logger.debug(f"[TOOLIFY] 添加工具调用: {result}")
    
    logger.debug(f"[TOOLIFY] 最终解析结果: {results}")
    return results if results else None

