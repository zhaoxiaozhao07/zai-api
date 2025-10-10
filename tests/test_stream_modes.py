#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试流式和非流式请求
用于验证OpenAI兼容API服务器的两种响应模式
"""

import httpx
import json
import sys
import io
from typing import Optional

# 设置stdout编码为utf-8，避免Windows下的编码问题
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# 配置
API_BASE_URL = "http://localhost:8080"  # 默认端口8080
API_KEY = "sk-123456"  # 根据实际情况修改，如果SKIP_AUTH_TOKEN=true则无需修改

# 测试消息
TEST_MESSAGES = [
    {
        "role": "user",
        "content": "你好，请简单介绍一下Python语言的特点。"
    }
]

# 可用模型列表
AVAILABLE_MODELS = [
    "glm-4.5",
    "glm-4.5-thinking",
    "glm-4.5-search",
    "glm-4.5-air",
    "glm-4.6",
    "glm-4.6-thinking",
]


def print_section(title: str):
    """打印分隔线"""
    print("\n" + "=" * 80)
    print(f" {title} ")
    print("=" * 80 + "\n")


def test_non_stream(model: str = "glm-4.5", messages: list = None) -> Optional[dict]:
    """
    测试非流式请求
    
    Args:
        model: 模型名称
        messages: 消息列表
    
    Returns:
        响应数据或None
    """
    print_section(f"测试非流式请求 - 模型: {model}")
    
    if messages is None:
        messages = TEST_MESSAGES
    
    url = f"{API_BASE_URL}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,  # 非流式
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    try:
        print(f"📤 发送请求到: {url}")
        print(f"📝 请求内容: {json.dumps(payload, ensure_ascii=False, indent=2)}")
        print("\n⏳ 等待响应...")
        
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, json=payload, headers=headers)
        
        print(f"\n✅ 响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"📦 响应数据:")
            print(json.dumps(data, ensure_ascii=False, indent=2))
            
            # 提取并显示关键信息
            if "choices" in data and len(data["choices"]) > 0:
                message = data["choices"][0].get("message", {})
                content = message.get("content", "")
                reasoning_content = message.get("reasoning_content", "")
                
                print(f"\n💬 助手回复:")
                print(f"{content}")
                
                if reasoning_content:
                    print(f"\n🧠 推理内容:")
                    print(f"{reasoning_content}")
                
                # 显示使用情况
                if "usage" in data:
                    usage = data["usage"]
                    print(f"\n📊 Token使用情况:")
                    print(f"  - 提示词: {usage.get('prompt_tokens', 0)}")
                    print(f"  - 完成: {usage.get('completion_tokens', 0)}")
                    print(f"  - 总计: {usage.get('total_tokens', 0)}")
            
            return data
        else:
            print(f"❌ 请求失败: {response.text}")
            return None
            
    except httpx.TimeoutException:
        print("❌ 请求超时")
        return None
    except Exception as e:
        print(f"❌ 发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_stream(model: str = "glm-4.5", messages: list = None):
    """
    测试流式请求
    
    Args:
        model: 模型名称
        messages: 消息列表
    """
    print_section(f"测试流式请求 - 模型: {model}")
    
    if messages is None:
        messages = TEST_MESSAGES
    
    url = f"{API_BASE_URL}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,  # 流式
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    try:
        print(f"📤 发送请求到: {url}")
        print(f"📝 请求内容: {json.dumps(payload, ensure_ascii=False, indent=2)}")
        print("\n⏳ 开始接收流式数据...\n")
        
        full_content = ""
        chunk_count = 0
        
        with httpx.Client(timeout=60.0) as client:
            with client.stream("POST", url, json=payload, headers=headers) as response:
                print(f"✅ 响应状态码: {response.status_code}\n")
                
                if response.status_code != 200:
                    error_text = response.read().decode('utf-8')
                    print(f"❌ 请求失败: {error_text}")
                    return
                
                print("💬 助手回复 (流式):")
                print("-" * 80)
                
                for line in response.iter_lines():
                    if not line:
                        continue
                    
                    line = line.strip()
                    
                    if line.startswith("data:"):
                        chunk_str = line[5:].strip()
                        
                        if chunk_str == "[DONE]":
                            print("\n" + "-" * 80)
                            print("✅ 流式传输完成")
                            break
                        
                        if not chunk_str:
                            continue
                        
                        try:
                            chunk = json.loads(chunk_str)
                            chunk_count += 1
                            
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                
                                if content:
                                    print(content, end="", flush=True)
                                    full_content += content
                                
                                # 检查完成原因
                                finish_reason = chunk["choices"][0].get("finish_reason")
                                if finish_reason:
                                    print(f"\n\n✅ 完成原因: {finish_reason}")
                                
                                # 显示使用情况
                                if "usage" in chunk:
                                    usage = chunk["usage"]
                                    print(f"\n📊 Token使用情况:")
                                    print(f"  - 提示词: {usage.get('prompt_tokens', 0)}")
                                    print(f"  - 完成: {usage.get('completion_tokens', 0)}")
                                    print(f"  - 总计: {usage.get('total_tokens', 0)}")
                            
                            elif "error" in chunk:
                                print(f"\n❌ 错误: {chunk['error']}")
                                break
                        
                        except json.JSONDecodeError as e:
                            print(f"\n⚠️ JSON解析错误: {e}")
                            continue
        
        print(f"\n📈 统计信息:")
        print(f"  - 接收到的数据块数: {chunk_count}")
        print(f"  - 完整内容长度: {len(full_content)} 字符")
        
    except httpx.TimeoutException:
        print("❌ 请求超时")
    except Exception as e:
        print(f"❌ 发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


def test_health_check():
    """测试健康检查端点"""
    print_section("健康检查")
    
    try:
        url = f"{API_BASE_URL}/health"
        print(f"📤 检查服务健康状态: {url}")
        
        with httpx.Client(timeout=5.0) as client:
            response = client.get(url)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 服务状态: {data.get('status', 'unknown')}")
            return True
        else:
            print(f"❌ 健康检查失败: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 无法连接到服务: {str(e)}")
        return False


def test_models_list():
    """测试模型列表端点"""
    print_section("获取模型列表")
    
    try:
        url = f"{API_BASE_URL}/v1/models"
        headers = {
            "Authorization": f"Bearer {API_KEY}"
        }
        print(f"📤 获取模型列表: {url}")
        
        with httpx.Client(timeout=5.0) as client:
            response = client.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            models = data.get("data", [])
            print(f"✅ 可用模型数量: {len(models)}")
            print("\n📋 模型列表:")
            for model in models:
                print(f"  - {model.get('id')} (owned by: {model.get('owned_by')})")
            return models
        else:
            print(f"❌ 获取模型列表失败: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"❌ 发生错误: {str(e)}")
        return []


def main():
    """主测试函数"""
    print("\n" + "=" * 80)
    print(" OpenAI 兼容 API 服务器 - 流式与非流式测试 ")
    print("=" * 80)
    
    # 1. 健康检查
    if not test_health_check():
        print("\n❌ 服务未运行或无法访问，请先启动服务！")
        sys.exit(1)
    
    # 2. 获取模型列表
    models = test_models_list()
    
    # 3. 测试非流式请求
    test_non_stream(model="glm-4.5")
    
    # 4. 测试流式请求
    test_stream(model="glm-4.5")
    
    # 5. 测试思考模型（非流式）
    print_section("测试思考模型（非流式）")
    test_non_stream(
        model="glm-4.5-thinking",
        messages=[
            {
                "role": "user",
                "content": "计算 15 * 23 + 47，并解释计算步骤。"
            }
        ]
    )
    
    # 总结
    print_section("测试完成")
    print("✅ 所有测试已完成！")
    print("\n💡 提示:")
    print("  - 非流式模式: 一次性返回完整响应")
    print("  - 流式模式: 实时逐字返回响应内容")
    print("  - 思考模型: 会在响应中包含推理过程（reasoning_content）")


if __name__ == "__main__":
    main()

