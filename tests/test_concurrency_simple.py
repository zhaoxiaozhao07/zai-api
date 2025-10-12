#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简化版并发测试脚本
快速测试 API 并发性能
"""

import asyncio
import httpx
import time
import sys
import io

# Windows编码处理
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 配置
API_BASE_URL = "http://localhost:8080"
API_KEY = "sk-123456"


async def send_request(client, request_id):
    """发送单个请求"""
    url = f"{API_BASE_URL}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": "glm-4.5",
        "messages": [{"role": "user", "content": f"你好，这是第{request_id}个测试请求。"}],
        "stream": False,
        "max_tokens": 100
    }
    
    start_time = time.time()
    
    try:
        response = await client.post(url, json=payload, headers=headers)
        duration = time.time() - start_time
        
        return {
            "id": request_id,
            "success": response.status_code == 200,
            "duration": duration,
            "status": response.status_code
        }
    except Exception as e:
        duration = time.time() - start_time
        return {
            "id": request_id,
            "success": False,
            "duration": duration,
            "error": str(e)
        }


async def test_concurrent(num_requests=10, concurrency=5):
    """执行并发测试"""
    print(f"\n{'='*80}")
    print(f"🚀 并发测试: {num_requests}个请求，{concurrency}个并发")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # 使用信号量控制并发数
        semaphore = asyncio.Semaphore(concurrency)
        
        async def limited_request(request_id):
            async with semaphore:
                result = await send_request(client, request_id)
                print(f"✓ 请求 #{result['id']:02d} - "
                      f"{'成功' if result['success'] else '失败'} - "
                      f"耗时: {result['duration']:.3f}秒")
                return result
        
        # 创建所有任务
        tasks = [limited_request(i+1) for i in range(num_requests)]
        
        # 执行
        results = await asyncio.gather(*tasks)
    
    total_time = time.time() - start_time
    
    # 统计结果
    success_count = sum(1 for r in results if r['success'])
    durations = [r['duration'] for r in results if r['success']]
    
    print(f"\n{'='*80}")
    print("📊 测试结果:")
    print(f"{'='*80}")
    print(f"总请求数: {num_requests}")
    print(f"成功数: {success_count}")
    print(f"失败数: {num_requests - success_count}")
    print(f"成功率: {success_count/num_requests*100:.2f}%")
    
    if durations:
        print(f"\n平均响应时间: {sum(durations)/len(durations):.3f}秒")
        print(f"最快响应: {min(durations):.3f}秒")
        print(f"最慢响应: {max(durations):.3f}秒")
    
    print(f"\n总耗时: {total_time:.2f}秒")
    print(f"吞吐量: {success_count/total_time:.2f} 请求/秒")
    print(f"{'='*80}\n")


async def main():
    """主函数"""
    print("\n" + "="*80)
    print(" 简化版并发测试 ".center(80))
    print("="*80)
    
    # 健康检查
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{API_BASE_URL}/health")
            if response.status_code != 200:
                print("❌ 服务未运行，请先启动服务！")
                return
        print("✅ 服务运行正常\n")
    except Exception as e:
        print(f"❌ 无法连接到服务: {e}")
        return
    
    # 测试场景
    await test_concurrent(num_requests=5, concurrency=1)   # 顺序执行
    await test_concurrent(num_requests=10, concurrency=5)  # 低并发
    await test_concurrent(num_requests=20, concurrency=10) # 中并发
    
    print("✅ 所有测试完成！\n")


if __name__ == "__main__":
    asyncio.run(main())









