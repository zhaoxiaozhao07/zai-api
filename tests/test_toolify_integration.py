"""
测试 Toolify 工具调用功能集成
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openai import OpenAI

# 测试工具定义
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：北京、上海"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位",
                        "default": "celsius"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "在网络上搜索信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "最大结果数量",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    }
                },
                "required": ["query"]
            }
        }
    }
]


def test_non_stream_with_tools():
    """测试非流式请求与工具调用"""
    print("=" * 60)
    print("测试 1: 非流式请求 + 工具调用")
    print("=" * 60)
    
    client = OpenAI(
        base_url="http://localhost:8080/v1",
        api_key="sk-123456"
    )
    
    try:
        response = client.chat.completions.create(
            model="GLM-4.5",
            messages=[
                {"role": "user", "content": "北京今天天气怎么样？"}
            ],
            tools=tools,
            tool_choice="auto"
        )
        
        print(f"\n✅ 请求成功！")
        print(f"Finish reason: {response.choices[0].finish_reason}")
        
        if response.choices[0].message.tool_calls:
            print(f"\n🔧 检测到 {len(response.choices[0].message.tool_calls)} 个工具调用:")
            for tool_call in response.choices[0].message.tool_calls:
                print(f"  - ID: {tool_call.id}")
                print(f"  - 工具: {tool_call.function.name}")
                print(f"  - 参数: {tool_call.function.arguments}")
        else:
            print(f"\n📝 普通回复: {response.choices[0].message.content}")
        
        print(f"\n使用情况:")
        print(f"  - 输入 tokens: {response.usage.prompt_tokens}")
        print(f"  - 输出 tokens: {response.usage.completion_tokens}")
        print(f"  - 总计 tokens: {response.usage.total_tokens}")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


def test_stream_with_tools():
    """测试流式请求与工具调用"""
    print("\n" + "=" * 60)
    print("测试 2: 流式请求 + 工具调用")
    print("=" * 60)
    
    client = OpenAI(
        base_url="http://localhost:8080/v1",
        api_key="sk-123456"
    )
    
    try:
        stream = client.chat.completions.create(
            model="GLM-4.5",
            messages=[
                {"role": "user", "content": "帮我搜索Python教程"}
            ],
            tools=tools,
            stream=True
        )
        
        print(f"\n✅ 开始接收流式响应...")
        has_tool_calls = False
        content_parts = []
        
        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                
                if delta.content:
                    content_parts.append(delta.content)
                    print(delta.content, end="", flush=True)
                
                if delta.tool_calls:
                    has_tool_calls = True
                    print(f"\n\n🔧 检测到工具调用:")
                    for tool_call in delta.tool_calls:
                        if tool_call.function:
                            print(f"  - 工具: {tool_call.function.name}")
                            print(f"  - 参数: {tool_call.function.arguments}")
        
        if not has_tool_calls and content_parts:
            print(f"\n\n📝 完整内容: {''.join(content_parts)}")
        
        print(f"\n✅ 流式响应完成")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


def test_without_tools():
    """测试不带工具的普通请求"""
    print("\n" + "=" * 60)
    print("测试 3: 普通请求（不带工具）")
    print("=" * 60)
    
    client = OpenAI(
        base_url="http://localhost:8080/v1",
        api_key="sk-123456"
    )
    
    try:
        response = client.chat.completions.create(
            model="GLM-4.5",
            messages=[
                {"role": "user", "content": "你好，请介绍一下你自己"}
            ]
        )
        
        print(f"\n✅ 请求成功！")
        print(f"Finish reason: {response.choices[0].finish_reason}")
        print(f"回复: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


def test_tool_choice_none():
    """测试 tool_choice=none"""
    print("\n" + "=" * 60)
    print("测试 4: tool_choice=none（禁止使用工具）")
    print("=" * 60)
    
    client = OpenAI(
        base_url="http://localhost:8080/v1",
        api_key="sk-123456"
    )
    
    try:
        response = client.chat.completions.create(
            model="GLM-4.5",
            messages=[
                {"role": "user", "content": "北京天气如何？"}
            ],
            tools=tools,
            tool_choice="none"  # 明确禁止使用工具
        )
        
        print(f"\n✅ 请求成功！")
        print(f"Finish reason: {response.choices[0].finish_reason}")
        
        if response.choices[0].message.tool_calls:
            print(f"❌ 错误：不应该有工具调用！")
        else:
            print(f"✅ 正确：没有工具调用")
            print(f"回复: {response.choices[0].message.content[:200]}...")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "🚀" * 30)
    print("Toolify 工具调用功能集成测试")
    print("🚀" * 30)
    print("\n⚠️  请确保服务器正在运行：python main.py")
    print("⚠️  请确保 ENABLE_TOOLIFY=true 已设置")
    
    input("\n按 Enter 键开始测试...")
    
    # 执行测试
    test_non_stream_with_tools()
    test_stream_with_tools()
    test_without_tools()
    test_tool_choice_none()
    
    print("\n" + "=" * 60)
    print("✅ 所有测试完成！")
    print("=" * 60)

