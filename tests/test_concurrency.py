#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
并发测试脚本
用于测试OpenAI兼容API服务器的并发处理能力
"""

import asyncio
import httpx
import json
import time
import sys
import io
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import statistics

# Windows编码处理
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 配置
API_BASE_URL = "http://localhost:8080"
API_KEY = "sk-123456"

# 测试消息模板
TEST_MESSAGES = [
    {
        "role": "user",
        "content": "你好，请简单介绍一下自己。"
    },
    {
        "role": "user",
        "content": "Python是什么？"
    },
    {
        "role": "user",
        "content": "解释一下机器学习的基本概念。"
    },
    {
        "role": "user",
        "content": "什么是RESTful API？"
    },
    {
        "role": "user",
        "content": "请列举3个常用的数据结构。"
    },
]


@dataclass
class RequestResult:
    """请求结果数据类"""
    request_id: int
    success: bool
    status_code: Optional[int]
    duration: float  # 响应时间（秒）
    error_message: Optional[str] = None
    response_data: Optional[Dict[str, Any]] = None
    tokens_used: Optional[int] = None


def print_section(title: str, width: int = 100):
    """打印分隔线"""
    print("\n" + "=" * width)
    print(f" {title} ".center(width))
    print("=" * width)


async def send_single_request(
    client: httpx.AsyncClient,
    request_id: int,
    model: str = "GLM-5",
    stream: bool = False,
    message_index: int = 0
) -> RequestResult:
    """
    发送单个请求
    
    Args:
        client: httpx异步客户端
        request_id: 请求ID
        model: 模型名称
        stream: 是否流式
        message_index: 消息索引
    
    Returns:
        RequestResult: 请求结果
    """
    url = f"{API_BASE_URL}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    # 选择测试消息
    messages = [TEST_MESSAGES[message_index % len(TEST_MESSAGES)]]
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "temperature": 0.7,
        "max_tokens": 200
    }
    
    start_time = time.time()
    
    try:
        if not stream:
            # 非流式请求
            response = await client.post(url, json=payload, headers=headers)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                tokens = data.get("usage", {}).get("total_tokens", 0)
                
                return RequestResult(
                    request_id=request_id,
                    success=True,
                    status_code=response.status_code,
                    duration=duration,
                    response_data=data,
                    tokens_used=tokens
                )
            else:
                return RequestResult(
                    request_id=request_id,
                    success=False,
                    status_code=response.status_code,
                    duration=duration,
                    error_message=response.text
                )
        else:
            # 流式请求
            async with client.stream("POST", url, json=payload, headers=headers) as response:
                if response.status_code != 200:
                    duration = time.time() - start_time
                    error_text = await response.aread()
                    return RequestResult(
                        request_id=request_id,
                        success=False,
                        status_code=response.status_code,
                        duration=duration,
                        error_message=error_text.decode('utf-8')
                    )
                
                # 接收流式数据
                chunk_count = 0
                async for line in response.aiter_lines():
                    if not line or not line.strip():
                        continue
                    
                    line = line.strip()
                    if line.startswith("data:"):
                        chunk_str = line[5:].strip()
                        if chunk_str == "[DONE]":
                            break
                        if chunk_str:
                            chunk_count += 1
                
                duration = time.time() - start_time
                return RequestResult(
                    request_id=request_id,
                    success=True,
                    status_code=200,
                    duration=duration,
                    response_data={"chunk_count": chunk_count}
                )
    
    except asyncio.TimeoutError:
        duration = time.time() - start_time
        return RequestResult(
            request_id=request_id,
            success=False,
            status_code=None,
            duration=duration,
            error_message="请求超时"
        )
    except Exception as e:
        duration = time.time() - start_time
        return RequestResult(
            request_id=request_id,
            success=False,
            status_code=None,
            duration=duration,
            error_message=str(e)
        )


async def run_concurrent_requests(
    num_requests: int,
    concurrency: int,
    model: str = "GLM-5",
    stream: bool = False,
    timeout: float = 60.0
) -> List[RequestResult]:
    """
    运行并发请求
    
    Args:
        num_requests: 总请求数
        concurrency: 并发数
        model: 模型名称
        stream: 是否流式
        timeout: 超时时间（秒）
    
    Returns:
        List[RequestResult]: 请求结果列表
    """
    print(f"\n🚀 开始执行并发测试...")
    print(f"   总请求数: {num_requests}")
    print(f"   并发数: {concurrency}")
    print(f"   模型: {model}")
    print(f"   模式: {'流式' if stream else '非流式'}")
    print(f"   超时时间: {timeout}秒")
    
    results = []
    
    # 创建异步HTTP客户端
    limits = httpx.Limits(max_keepalive_connections=concurrency, max_connections=concurrency * 2)
    timeout_config = httpx.Timeout(timeout)
    
    async with httpx.AsyncClient(limits=limits, timeout=timeout_config) as client:
        # 创建任务队列
        semaphore = asyncio.Semaphore(concurrency)
        
        async def limited_request(request_id: int, message_index: int):
            async with semaphore:
                return await send_single_request(
                    client, request_id, model, stream, message_index
                )
        
        # 创建所有任务
        tasks = [
            limited_request(i, i) 
            for i in range(num_requests)
        ]
        
        # 执行任务并显示进度
        total_start = time.time()
        
        # 使用 gather 执行所有任务
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(RequestResult(
                    request_id=i,
                    success=False,
                    status_code=None,
                    duration=0.0,
                    error_message=str(result)
                ))
            else:
                processed_results.append(result)
        
        total_duration = time.time() - total_start
        
        print(f"\n并发测试完成！总耗时: {total_duration:.2f}秒")
        
        return processed_results


def analyze_results(results: List[RequestResult]) -> Dict[str, Any]:
    """
    分析测试结果
    
    Args:
        results: 请求结果列表
    
    Returns:
        Dict: 分析统计数据
    """
    total = len(results)
    successful = sum(1 for r in results if r.success)
    failed = total - successful
    
    success_rate = (successful / total * 100) if total > 0 else 0
    
    # 计算响应时间统计
    durations = [r.duration for r in results if r.success]
    
    if durations:
        avg_duration = statistics.mean(durations)
        median_duration = statistics.median(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        
        if len(durations) > 1:
            std_duration = statistics.stdev(durations)
        else:
            std_duration = 0.0
    else:
        avg_duration = median_duration = min_duration = max_duration = std_duration = 0.0
    
    # 统计Token使用
    total_tokens = sum(r.tokens_used for r in results if r.tokens_used)
    
    # 统计错误类型
    error_types = {}
    for r in results:
        if not r.success and r.error_message:
            error_msg = r.error_message[:50]  # 截取前50个字符
            error_types[error_msg] = error_types.get(error_msg, 0) + 1
    
    return {
        "total_requests": total,
        "successful_requests": successful,
        "failed_requests": failed,
        "success_rate": success_rate,
        "avg_duration": avg_duration,
        "median_duration": median_duration,
        "min_duration": min_duration,
        "max_duration": max_duration,
        "std_duration": std_duration,
        "total_tokens": total_tokens,
        "error_types": error_types,
        "durations": durations
    }


def print_analysis(analysis: Dict[str, Any], test_name: str = "测试"):
    """
    打印分析结果
    
    Args:
        analysis: 分析数据
        test_name: 测试名称
    """
    print_section(f"{test_name} - 结果分析")
    
    print(f"\n📊 请求统计:")
    print(f"   总请求数: {analysis['total_requests']}")
    print(f"   成功数: {analysis['successful_requests']} ✅")
    print(f"   失败数: {analysis['failed_requests']} ❌")
    print(f"   成功率: {analysis['success_rate']:.2f}%")
    
    print(f"\n响应时间统计:")
    print(f"   平均响应时间: {analysis['avg_duration']:.3f}秒")
    print(f"   中位数响应时间: {analysis['median_duration']:.3f}秒")
    print(f"   最快响应: {analysis['min_duration']:.3f}秒")
    print(f"   最慢响应: {analysis['max_duration']:.3f}秒")
    print(f"   标准差: {analysis['std_duration']:.3f}秒")
    
    if analysis['total_tokens'] > 0:
        print(f"\n🔢 Token使用统计:")
        print(f"   总Token数: {analysis['total_tokens']}")
        print(f"   平均每请求: {analysis['total_tokens'] / analysis['successful_requests']:.0f}") if analysis['successful_requests'] > 0 else None
    
    if analysis['error_types']:
        print(f"\n❌ 错误类型统计:")
        for error, count in sorted(analysis['error_types'].items(), key=lambda x: x[1], reverse=True):
            print(f"   [{count}次] {error}")
    
    # 计算吞吐量
    if analysis['durations']:
        total_time = max(analysis['durations'])
        throughput = analysis['successful_requests'] / total_time if total_time > 0 else 0
        print(f"\n📈 性能指标:")
        print(f"   吞吐量: {throughput:.2f} 请求/秒")


def print_percentiles(durations: List[float]):
    """
    打印响应时间百分位数
    
    Args:
        durations: 响应时间列表
    """
    if not durations:
        return
    
    sorted_durations = sorted(durations)
    percentiles = [50, 75, 90, 95, 99]
    
    print(f"\n📊 响应时间百分位数:")
    for p in percentiles:
        index = int(len(sorted_durations) * p / 100)
        if index >= len(sorted_durations):
            index = len(sorted_durations) - 1
        value = sorted_durations[index]
        print(f"   P{p}: {value:.3f}秒")


async def test_basic_concurrency():
    """基础并发测试"""
    print_section("基础并发测试 - 非流式")
    
    # 测试配置
    test_configs = [
        {"num_requests": 5, "concurrency": 1, "name": "顺序执行 (1并发, 5请求)"},
        {"num_requests": 10, "concurrency": 5, "name": "低并发 (5并发, 10请求)"},
        {"num_requests": 20, "concurrency": 10, "name": "中并发 (10并发, 20请求)"},
    ]
    
    all_results = []
    
    for config in test_configs:
        print(f"\n{'─' * 100}")
        print(f"🧪 测试场景: {config['name']}")
        
        results = await run_concurrent_requests(
            num_requests=config['num_requests'],
            concurrency=config['concurrency'],
            stream=False
        )
        
        analysis = analyze_results(results)
        print_analysis(analysis, config['name'])
        print_percentiles(analysis['durations'])
        
        all_results.append({
            "config": config,
            "analysis": analysis
        })
        
        # 短暂延迟，避免过载
        await asyncio.sleep(1)
    
    return all_results


async def test_stream_concurrency():
    """流式请求并发测试"""
    print_section("流式请求并发测试")
    
    results = await run_concurrent_requests(
        num_requests=10,
        concurrency=5,
        stream=True
    )
    
    analysis = analyze_results(results)
    print_analysis(analysis, "流式请求并发测试")
    print_percentiles(analysis['durations'])
    
    return analysis


async def test_stress():
    """压力测试"""
    print_section("压力测试")
    
    print("\n⚠️  警告: 这将发送大量并发请求，请确保服务器能够承受！")
    print("   按 Ctrl+C 可以随时中断测试...")
    
    await asyncio.sleep(2)
    
    # 压力测试配置
    results = await run_concurrent_requests(
        num_requests=50,
        concurrency=20,
        stream=False,
        timeout=120.0
    )
    
    analysis = analyze_results(results)
    print_analysis(analysis, "压力测试")
    print_percentiles(analysis['durations'])
    
    return analysis


async def test_health_check():
    """健康检查"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{API_BASE_URL}/health")
            return response.status_code == 200
    except Exception:
        return False


def print_summary(all_tests: List[Dict[str, Any]]):
    """打印总结"""
    print_section("测试总结")
    
    print("\n📋 所有测试场景对比:\n")
    print(f"{'测试场景':<40} {'成功率':<12} {'平均响应':<12} {'吞吐量':<12}")
    print("─" * 100)
    
    for test in all_tests:
        if 'config' in test:
            name = test['config']['name']
            analysis = test['analysis']
        else:
            name = test.get('name', '未知测试')
            analysis = test['analysis']
        
        success_rate = f"{analysis['success_rate']:.2f}%"
        avg_duration = f"{analysis['avg_duration']:.3f}秒"
        
        if analysis['durations']:
            total_time = max(analysis['durations'])
            throughput = analysis['successful_requests'] / total_time if total_time > 0 else 0
            throughput_str = f"{throughput:.2f} req/s"
        else:
            throughput_str = "N/A"
        
        print(f"{name:<40} {success_rate:<12} {avg_duration:<12} {throughput_str:<12}")


async def main():
    """主函数"""
    print("\n" + "=" * 100)
    print(" OpenAI 兼容 API 服务器 - 并发性能测试 ".center(100))
    print("=" * 100)
    print(f"\n⏰ 测试开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 健康检查
    print("\n🔍 检查服务状态...")
    if not await test_health_check():
        print("❌ 服务未运行或无法访问，请先启动服务！")
        sys.exit(1)
    print("服务运行正常")
    
    all_tests = []
    
    try:
        # 1. 基础并发测试
        basic_results = await test_basic_concurrency()
        all_tests.extend(basic_results)
        
        # 2. 流式请求测试
        stream_analysis = await test_stream_concurrency()
        all_tests.append({"name": "流式请求并发测试", "analysis": stream_analysis})
        
        # 3. 压力测试（可选）
        print("\n" + "─" * 100)
        user_input = input("\n是否执行压力测试? (y/N): ").strip().lower()
        if user_input == 'y':
            stress_analysis = await test_stress()
            all_tests.append({"name": "压力测试", "analysis": stress_analysis})
        else:
            print("⏭️  跳过压力测试")
        
        # 打印总结
        print_summary(all_tests)
        
        print(f"\n⏰ 测试结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n所有测试完成！")
        
        print("\n💡 建议:")
        print("   1. 观察不同并发级别下的性能变化")
        print("   2. 关注成功率，确保服务稳定性")
        print("   3. 监控响应时间的分布情况")
        print("   4. 查看服务器资源使用情况（CPU、内存、网络）")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())









