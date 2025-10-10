"""
Toolify 核心功能模块
提供工具调用的主要功能：请求处理、响应解析、格式转换
"""

import uuid
import json
import secrets
import string
import logging
from typing import List, Dict, Any, Optional
from collections import OrderedDict
import time
import threading

logger = logging.getLogger(__name__)


def generate_random_trigger_signal() -> str:
    """生成随机的、自闭合的触发信号，如 <Function_AB1c_Start/>"""
    chars = string.ascii_letters + string.digits
    random_str = ''.join(secrets.choice(chars) for _ in range(4))
    return f"<Function_{random_str}_Start/>"


class ToolCallMappingManager:
    """
    工具调用映射管理器（带TTL和大小限制）
    
    功能：
    1. 自动过期清理 - 条目在指定时间后自动删除
    2. 大小限制 - 防止内存无限增长
    3. LRU驱逐 - 达到大小限制时删除最少使用的条目
    4. 线程安全 - 支持并发访问
    5. 周期性清理 - 后台线程定期清理过期条目
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600, cleanup_interval: int = 300):
        """
        初始化映射管理器
        
        Args:
            max_size: 最大存储条目数
            ttl_seconds: 条目生存时间（秒）
            cleanup_interval: 清理间隔（秒）
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cleanup_interval = cleanup_interval
        
        self._data: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.RLock()
        
        self._cleanup_thread = threading.Thread(target=self._periodic_cleanup, daemon=True)
        self._cleanup_thread.start()
        
        logger.debug(f"[TOOLIFY] 工具调用映射管理器已启动 - 最大条目: {max_size}, TTL: {ttl_seconds}s")
    
    def store(self, tool_call_id: str, name: str, args: dict, description: str = "") -> None:
        """存储工具调用映射"""
        with self._lock:
            current_time = time.time()
            
            if tool_call_id in self._data:
                del self._data[tool_call_id]
                del self._timestamps[tool_call_id]
            
            while len(self._data) >= self.max_size:
                oldest_key = next(iter(self._data))
                del self._data[oldest_key]
                del self._timestamps[oldest_key]
                logger.debug(f"[TOOLIFY] 因大小限制移除最旧条目: {oldest_key}")
            
            self._data[tool_call_id] = {
                "name": name,
                "args": args,
                "description": description,
                "created_at": current_time
            }
            self._timestamps[tool_call_id] = current_time
            
            logger.debug(f"[TOOLIFY] 存储工具调用映射: {tool_call_id} -> {name}")
    
    def get(self, tool_call_id: str) -> Optional[Dict[str, Any]]:
        """获取工具调用映射（更新LRU顺序）"""
        with self._lock:
            current_time = time.time()
            
            if tool_call_id not in self._data:
                logger.debug(f"[TOOLIFY] 未找到工具调用映射: {tool_call_id}")
                return None
            
            if current_time - self._timestamps[tool_call_id] > self.ttl_seconds:
                logger.debug(f"[TOOLIFY] 工具调用映射已过期: {tool_call_id}")
                del self._data[tool_call_id]
                del self._timestamps[tool_call_id]
                return None
            
            result = self._data[tool_call_id]
            self._data.move_to_end(tool_call_id)
            
            logger.debug(f"[TOOLIFY] 找到工具调用映射: {tool_call_id} -> {result['name']}")
            return result
    
    def cleanup_expired(self) -> int:
        """清理过期条目，返回清理数量"""
        with self._lock:
            current_time = time.time()
            expired_keys = []
            
            for key, timestamp in self._timestamps.items():
                if current_time - timestamp > self.ttl_seconds:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._data[key]
                del self._timestamps[key]
            
            if expired_keys:
                logger.debug(f"[TOOLIFY] 清理了 {len(expired_keys)} 个过期条目")
            
            return len(expired_keys)
    
    def _periodic_cleanup(self) -> None:
        """后台周期性清理线程"""
        while True:
            try:
                time.sleep(self.cleanup_interval)
                self.cleanup_expired()
            except Exception as e:
                logger.error(f"[TOOLIFY] 后台清理线程异常: {e}")


class ToolifyCore:
    """Toolify 核心类 - 管理工具调用功能"""
    
    def __init__(self, enable_function_calling: bool = True):
        """
        初始化 Toolify 核心
        
        Args:
            enable_function_calling: 是否启用函数调用功能
        """
        self.enable_function_calling = enable_function_calling
        self.mapping_manager = ToolCallMappingManager()
        self.trigger_signal = generate_random_trigger_signal()
        
        logger.info(f"[TOOLIFY] 核心已初始化 - 功能启用: {enable_function_calling}")
        logger.debug(f"[TOOLIFY] 触发信号: {self.trigger_signal}")
    
    def store_tool_call_mapping(self, tool_call_id: str, name: str, args: dict, description: str = ""):
        """存储工具调用ID与调用内容的映射"""
        self.mapping_manager.store(tool_call_id, name, args, description)
    
    def get_tool_call_mapping(self, tool_call_id: str) -> Optional[Dict[str, Any]]:
        """获取工具调用ID对应的调用内容"""
        return self.mapping_manager.get(tool_call_id)
    
    def format_tool_result_for_ai(self, tool_call_id: str, result_content: str) -> str:
        """格式化工具调用结果供AI理解"""
        logger.debug(f"[TOOLIFY] 格式化工具调用结果: tool_call_id={tool_call_id}")
        tool_info = self.get_tool_call_mapping(tool_call_id)
        if not tool_info:
            logger.debug(f"[TOOLIFY] 未找到工具调用映射，使用默认格式")
            return f"Tool execution result:\n<tool_result>\n{result_content}\n</tool_result>"
        
        formatted_text = f"""Tool execution result:
- Tool name: {tool_info['name']}
- Execution result:
<tool_result>
{result_content}
</tool_result>"""
        
        logger.debug(f"[TOOLIFY] 格式化完成，工具名: {tool_info['name']}")
        return formatted_text
    
    def format_assistant_tool_calls_for_ai(self, tool_calls: List[Dict[str, Any]]) -> str:
        """将助手的工具调用格式化为AI可读的字符串格式"""
        logger.debug(f"[TOOLIFY] 格式化助手工具调用. 数量: {len(tool_calls)}")
        
        xml_calls_parts = []
        for tool_call in tool_calls:
            function_info = tool_call.get("function", {})
            name = function_info.get("name", "")
            arguments_json = function_info.get("arguments", "{}")
            
            try:
                args_dict = json.loads(arguments_json)
            except (json.JSONDecodeError, TypeError):
                args_dict = {"raw_arguments": arguments_json}

            args_parts = []
            for key, value in args_dict.items():
                json_value = json.dumps(value, ensure_ascii=False)
                args_parts.append(f"<{key}>{json_value}</{key}>")
            
            args_content = "\n".join(args_parts)
            
            xml_call = f"<function_call>\n<tool>{name}</tool>\n<args>\n{args_content}\n</args>\n</function_call>"
            xml_calls_parts.append(xml_call)

        all_calls = "\n".join(xml_calls_parts)
        final_str = f"{self.trigger_signal}\n<function_calls>\n{all_calls}\n</function_calls>"
        
        logger.debug("[TOOLIFY] 助手工具调用格式化成功")
        return final_str
    
    def preprocess_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        预处理消息，转换工具类型消息为AI可理解格式
        
        Args:
            messages: OpenAI格式的消息列表
            
        Returns:
            处理后的消息列表
        """
        processed_messages = []
        
        for message in messages:
            if isinstance(message, dict):
                # 处理 tool 角色消息
                if message.get("role") == "tool":
                    tool_call_id = message.get("tool_call_id")
                    content = message.get("content")
                    
                    if tool_call_id and content:
                        formatted_content = self.format_tool_result_for_ai(tool_call_id, content)
                        processed_message = {
                            "role": "user",
                            "content": formatted_content
                        }
                        processed_messages.append(processed_message)
                        logger.debug(f"[TOOLIFY] 转换tool消息为user消息: tool_call_id={tool_call_id}")
                    else:
                        logger.debug(f"[TOOLIFY] 跳过无效tool消息: tool_call_id={tool_call_id}")
                
                # 处理 assistant 角色的 tool_calls
                elif message.get("role") == "assistant" and "tool_calls" in message and message["tool_calls"]:
                    tool_calls = message.get("tool_calls", [])
                    formatted_tool_calls_str = self.format_assistant_tool_calls_for_ai(tool_calls)
                    
                    # 与原始内容合并
                    original_content = message.get("content") or ""
                    final_content = f"{original_content}\n{formatted_tool_calls_str}".strip()

                    processed_message = {
                        "role": "assistant",
                        "content": final_content
                    }
                    # 复制其他字段（除了tool_calls）
                    for key, value in message.items():
                        if key not in ["role", "content", "tool_calls"]:
                            processed_message[key] = value

                    processed_messages.append(processed_message)
                    logger.debug(f"[TOOLIFY] 转换assistant的tool_calls为content")
                else:
                    processed_messages.append(message)
            else:
                processed_messages.append(message)
        
        return processed_messages
    
    def convert_parsed_tools_to_openai_format(self, parsed_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        将解析出的工具调用转换为OpenAI格式的tool_calls
        
        Args:
            parsed_tools: 解析出的工具列表 [{"name": "tool_name", "args": {...}}, ...]
            
        Returns:
            OpenAI格式的tool_calls列表
        """
        tool_calls = []
        for tool in parsed_tools:
            tool_call_id = f"call_{uuid.uuid4().hex}"
            self.store_tool_call_mapping(
                tool_call_id,
                tool["name"],
                tool["args"],
                f"调用工具 {tool['name']}"
            )
            tool_calls.append({
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "arguments": json.dumps(tool["args"])
                }
            })
        
        logger.debug(f"[TOOLIFY] 转换了 {len(tool_calls)} 个工具调用")
        return tool_calls

