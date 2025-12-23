"""
Toolify 提示词生成器
生成工具调用的系统提示词

更新自 Toolify_new 版本，添加参数校验和增强功能
"""

import json
import logging
from typing import List, Dict, Any, Optional
from fastapi import HTTPException

logger = logging.getLogger(__name__)


def get_function_call_prompt_template(trigger_signal: str, custom_template: Optional[str] = None) -> str:
    """
    基于动态触发信号生成提示词模板
    
    Args:
        trigger_signal: 触发信号字符串
        custom_template: 自定义模板（可选）
        
    Returns:
        提示词模板字符串
    """
    if custom_template:
        logger.info("[TOOLIFY] 使用配置中的自定义提示词模板")
        return custom_template.format(
            trigger_signal=trigger_signal,
            tools_list="{tools_list}"
        )
    
    return f"""
你可以访问以下可用工具来帮助解决问题：

{{tools_list}}

**重要上下文说明：**
1. 如果需要，你可以在单次响应中调用多个工具。
2. 对话上下文中可能已包含之前函数调用的工具执行结果。请仔细查看对话历史，避免不必要的重复工具调用。
3. 当工具执行结果出现在上下文中时，它们将使用 <tool_result>...</tool_result> 这样的XML标签格式化，便于识别。
4. 这是你可以使用的唯一工具调用格式，任何偏差都将导致失败。

当你需要使用工具时，你**必须**严格遵循以下格式。不要在工具调用语法的第一行和第二行包含任何额外的文本、解释或对话：

1. 开始工具调用时，在新行上准确输出：
{trigger_signal}
不要有前导或尾随空格，完全按照上面显示的输出。触发信号必须单独占一行，且只出现一次。

2. 从第二行开始，**立即**紧跟完整的 <function_calls> XML块。

3. 对于多个工具调用，在同一个 <function_calls> 包装器中包含多个 <function_call> 块。

4. 在结束标签 </function_calls> 后不要添加任何文本或解释。

严格的参数键规则：
- 你必须使用**完全相同**的参数键（区分大小写和标点符号）。不要重命名、添加或删除字符。
- 如果键以连字符开头（例如 -i, -C），你必须在标签名中保留连字符。例如：<-i>true</-i>, <-C>2</-C>。
- 永远不要将 "-i" 转换为 "i" 或将 "-C" 转换为 "C"。不要复数化、翻译或给参数键起别名。
- <tool> 标签必须包含列表中某个工具的确切名称。任何其他工具名称都是无效的。
- <args> 必须包含该工具的所有必需参数。

正确示例（多个工具调用，包括带连字符的键）：
...响应内容（可选）...
{trigger_signal}
<function_calls>
    <function_call>
        <tool>Grep</tool>
        <args>
            <-i>true</-i>
            <-C>2</-C>
            <path>.</path>
        </args>
    </function_call>
    <function_call>
        <tool>search</tool>
        <args>
            <keywords>["Python Document", "how to use python"]</keywords>
        </args>
    </function_call>
</function_calls>

错误示例（额外文本 + 错误键名 — 不要这样做）：
...响应内容（可选）...
{trigger_signal}
我将为你调用工具。
<function_calls>
    <function_call>
        <tool>Grep</tool>
        <args>
            <i>true</i>
            <C>2</C>
            <path>.</path>
        </args>
    </function_call>
</function_calls>

现在请准备好严格遵循以上规范。
"""


def generate_function_prompt(tools: List[Dict[str, Any]], trigger_signal: str, custom_template: Optional[str] = None) -> tuple[str, str]:
    """
    基于客户端请求中的工具定义生成注入的系统提示词
    
    Args:
        tools: 工具定义列表（OpenAI格式）
        trigger_signal: 触发信号
        custom_template: 自定义模板（可选）
        
    Returns:
        (prompt_content, trigger_signal): 提示词内容和触发信号
        
    Raises:
        HTTPException: 如果工具定义校验失败（例如 required 字段引用了未定义的参数）
    """
    tools_list_str = []
    for i, tool in enumerate(tools):
        func = tool.get("function", {})
        name = func.get("name", "")
        description = func.get("description", "")

        # 读取 JSON Schema 字段 + 基础类型校验
        schema: Dict[str, Any] = func.get("parameters", {}) or {}

        props_raw = schema.get("properties", {})
        if props_raw is None:
            props_raw = {}
        if not isinstance(props_raw, dict):
            raise HTTPException(
                status_code=400,
                detail=f"Tool '{name}': 'properties' must be an object, got {type(props_raw).__name__}"
            )
        props: Dict[str, Any] = props_raw

        required_raw = schema.get("required", [])
        if required_raw is None:
            required_raw = []
        if not isinstance(required_raw, list):
            raise HTTPException(
                status_code=400,
                detail=f"Tool '{name}': 'required' must be a list, got {type(required_raw).__name__}"
            )

        non_string_required = [k for k in required_raw if not isinstance(k, str)]
        if non_string_required:
            raise HTTPException(
                status_code=400,
                detail=f"Tool '{name}': 'required' entries must be strings, got {non_string_required}"
            )

        required_list: List[str] = required_raw

        # 验证 required 中的字段必须在 properties 中定义
        missing_keys = [key for key in required_list if key not in props]
        if missing_keys:
            raise HTTPException(
                status_code=400,
                detail=f"Tool '{name}': required parameters {missing_keys} are not defined in properties"
            )

        # 简要摘要行：name (type)
        params_summary = ", ".join([
            f"{p_name} ({(p_info or {}).get('type', 'any')})" for p_name, p_info in props.items()
        ]) or "None"

        # 构建详细参数规范
        detail_lines: List[str] = []
        for p_name, p_info in props.items():
            p_info = p_info or {}
            p_type = p_info.get("type", "any")
            is_required = "Yes" if p_name in required_list else "No"
            p_desc = p_info.get("description")
            enum_vals = p_info.get("enum")
            default_val = p_info.get("default")
            examples_val = p_info.get("examples") or p_info.get("example")

            # 常见约束和提示
            constraints: Dict[str, Any] = {}
            for key in [
                "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                "minLength", "maxLength", "pattern", "format",
                "minItems", "maxItems", "uniqueItems"
            ]:
                if key in p_info:
                    constraints[key] = p_info.get(key)

            # 数组项类型提示
            if p_type == "array":
                items = p_info.get("items") or {}
                if isinstance(items, dict):
                    itype = items.get("type")
                    if itype:
                        constraints["items.type"] = itype

            # 组合详细行
            detail_lines.append(f"- {p_name}:")
            detail_lines.append(f"  - type: {p_type}")
            detail_lines.append(f"  - required: {is_required}")
            if p_desc:
                detail_lines.append(f"  - description: {p_desc}")
            if enum_vals is not None:
                try:
                    detail_lines.append(f"  - enum: {json.dumps(enum_vals, ensure_ascii=False)}")
                except Exception:
                    detail_lines.append(f"  - enum: {enum_vals}")
            if default_val is not None:
                try:
                    detail_lines.append(f"  - default: {json.dumps(default_val, ensure_ascii=False)}")
                except Exception:
                    detail_lines.append(f"  - default: {default_val}")
            if examples_val is not None:
                try:
                    detail_lines.append(f"  - examples: {json.dumps(examples_val, ensure_ascii=False)}")
                except Exception:
                    detail_lines.append(f"  - examples: {examples_val}")
            if constraints:
                try:
                    detail_lines.append(f"  - constraints: {json.dumps(constraints, ensure_ascii=False)}")
                except Exception:
                    detail_lines.append(f"  - constraints: {constraints}")

        detail_block = "\n".join(detail_lines) if detail_lines else "(无参数详情)"

        desc_block = f"```\n{description}\n```" if description else "None"

        tools_list_str.append(
            f"{i + 1}. <tool name=\"{name}\">\n"
            f"   描述:\n{desc_block}\n"
            f"   参数摘要: {params_summary}\n"
            f"   必需参数: {', '.join(required_list) if required_list else 'None'}\n"
            f"   参数详情:\n{detail_block}"
        )
    
    prompt_template = get_function_call_prompt_template(trigger_signal, custom_template)
    prompt_content = prompt_template.replace("{tools_list}", "\n\n".join(tools_list_str))
    
    return prompt_content, trigger_signal


def safe_process_tool_choice(tool_choice, tools: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    安全处理tool_choice字段，避免类型错误
    
    Args:
        tool_choice: tool_choice参数（可能是字符串或对象）
        tools: 可用工具列表（用于验证）
        
    Returns:
        附加的提示词内容
        
    Raises:
        HTTPException: 如果 tool_choice 指定的工具不在 tools 列表中
    """
    try:
        if tool_choice is None:
            return ""
        
        if isinstance(tool_choice, str):
            if tool_choice == "none":
                return "\n\n**重要提示：** 本轮你被禁止使用任何工具。请像普通聊天助手一样响应，直接回答用户的问题。"
            elif tool_choice == "auto":
                # 默认行为，无额外约束
                return ""
            elif tool_choice == "required":
                return "\n\n**重要提示：** 本轮你必须调用至少一个工具。不要在不使用工具的情况下响应。"
            else:
                logger.warning(f"[TOOLIFY] 未知的tool_choice字符串值: {tool_choice}")
                return ""
        
        # 处理 ToolChoice 对象: {"type": "function", "function": {"name": "xxx"}}
        elif hasattr(tool_choice, 'function'):
            function_dict = tool_choice.function
            if not isinstance(function_dict, dict):
                raise HTTPException(status_code=400, detail="tool_choice.function must be an object")
            
            required_tool_name = function_dict.get("name")
            if not required_tool_name or not isinstance(required_tool_name, str):
                raise HTTPException(status_code=400, detail="tool_choice.function.name must be a non-empty string")
            
            # 验证指定的工具是否存在
            if tools:
                tool_names = [t.get("function", {}).get("name") for t in tools if isinstance(t, dict)]
                if required_tool_name not in tool_names:
                    raise HTTPException(
                        status_code=400,
                        detail=f"tool_choice specifies tool '{required_tool_name}' which is not in the tools list. Available tools: {tool_names}"
                    )
            
            return f"\n\n**重要提示：** 本轮你必须**仅**使用名为 `{required_tool_name}` 的工具。生成必要的参数并按指定的XML格式输出。"
        
        # 处理字典形式的 tool_choice
        elif isinstance(tool_choice, dict):
            function_info = tool_choice.get("function", {})
            if isinstance(function_info, dict):
                required_tool_name = function_info.get("name")
                if required_tool_name and isinstance(required_tool_name, str):
                    # 验证指定的工具是否存在
                    if tools:
                        tool_names = [t.get("function", {}).get("name") for t in tools if isinstance(t, dict)]
                        if required_tool_name not in tool_names:
                            raise HTTPException(
                                status_code=400,
                                detail=f"tool_choice specifies tool '{required_tool_name}' which is not in the tools list. Available tools: {tool_names}"
                            )
                    return f"\n\n**重要提示：** 本轮你必须**仅**使用名为 `{required_tool_name}` 的工具。生成必要的参数并按指定的XML格式输出。"
            logger.debug(f"[TOOLIFY] 无法解析tool_choice字典: {tool_choice}")
            return ""
        
        else:
            logger.warning(f"[TOOLIFY] 不支持的tool_choice类型: {type(tool_choice)}")
            return ""
    
    except HTTPException:
        # 重新抛出 HTTPException 以保留状态码
        raise
    except Exception as e:
        logger.error(f"[TOOLIFY] 处理tool_choice时出错: {e}")
        return ""
