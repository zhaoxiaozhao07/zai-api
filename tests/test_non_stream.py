#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简单的非流式请求测试
"""

import httpx
import json
import sys
import io

# Windows编码处理
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 配置
API_BASE_URL = "http://localhost:8080"
API_KEY = "sk-123456"  # 使用env_template.txt中的默认key

def test_non_stream():
    """测试非流式请求"""
    url = f"{API_BASE_URL}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": "GLM-4.5",
        "messages": [
            {
                "role": "user",
                "content": "你好，用一句话介绍一下自己。"
            }
        ],
        "stream": False,  # 非流式
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    print("=" * 80)
    print("测试非流式请求")
    print("=" * 80)
    print(f"\n[请求] URL: {url}")
    print(f"[请求] Payload:\n{json.dumps(payload, ensure_ascii=False, indent=2)}")
    print("\n等待响应...\n")
    
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, json=payload, headers=headers)
        
        print(f"[响应] 状态码: {response.status_code}")
        print(f"[响应] Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n[响应] 完整数据:\n{json.dumps(data, ensure_ascii=False, indent=2)}")
            
            # 提取内容
            if "choices" in data and len(data["choices"]) > 0:
                message = data["choices"][0].get("message", {})
                content = message.get("content", "")
                
                print(f"\n{'=' * 80}")
                print("助手回复:")
                print(f"{'=' * 80}")
                print(content)
                print(f"{'=' * 80}")
                
                # 使用统计
                if "usage" in data:
                    usage = data["usage"]
                    print(f"\nToken使用: 提示={usage.get('prompt_tokens', 0)}, "
                          f"完成={usage.get('completion_tokens', 0)}, "
                          f"总计={usage.get('total_tokens', 0)}")
                
                print("\n✓ 非流式请求测试成功！")
                return True
        else:
            print(f"\n[错误] 响应内容:\n{response.text}")
            return False
            
    except Exception as e:
        print(f"\n[错误] 发生异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_stream():
    """测试流式请求"""
    url = f"{API_BASE_URL}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": "GLM-4.5",
        "messages": [
            {
                "role": "user",
                "content": "你好，用一句话介绍一下自己。"
            }
        ],
        "stream": True,  # 流式
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    print("\n" + "=" * 80)
    print("测试流式请求")
    print("=" * 80)
    print(f"\n[请求] URL: {url}")
    print(f"[请求] Payload:\n{json.dumps(payload, ensure_ascii=False, indent=2)}")
    print("\n开始接收流式数据...\n")
    
    try:
        full_content = ""
        chunk_count = 0
        
        with httpx.Client(timeout=60.0) as client:
            with client.stream("POST", url, json=payload, headers=headers) as response:
                print(f"[响应] 状态码: {response.status_code}\n")
                
                if response.status_code != 200:
                    error_text = response.read().decode('utf-8')
                    print(f"[错误] {error_text}")
                    return False
                
                print(f"{'=' * 80}")
                print("助手回复 (流式):")
                print(f"{'=' * 80}")
                
                for line in response.iter_lines():
                    if not line:
                        continue
                    
                    line = line.strip()
                    
                    if line.startswith("data:"):
                        chunk_str = line[5:].strip()
                        
                        if chunk_str == "[DONE]":
                            print(f"\n{'=' * 80}")
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
                        
                        except json.JSONDecodeError:
                            continue
        
        print(f"\n\n接收数据块: {chunk_count} 个")
        print(f"内容长度: {len(full_content)} 字符")
        print("\n✓ 流式请求测试成功！")
        return True
        
    except Exception as e:
        print(f"\n[错误] 发生异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 测试非流式
    success1 = test_non_stream()
    
    # 测试流式
    success2 = test_stream()
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"非流式请求: {'✓ 成功' if success1 else '✗ 失败'}")
    print(f"流式请求:   {'✓ 成功' if success2 else '✗ 失败'}")
    print("=" * 80)

