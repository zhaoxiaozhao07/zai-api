# OpenAI 兼容 API 服务器

一个完全兼容 OpenAI API 格式的代理服务器，将 OpenAI 格式的请求转换为 Z.AI API 调用，支持完整的工具调用（Function Calling）功能。

## 功能特性

### 核心功能

- **OpenAI 格式兼容**：完全兼容 OpenAI API 请求和响应格式
- **Z.AI 后端集成**：自动转换并调用 Z.AI API
- **多模型支持**：支持 GLM-4.5、GLM-4.6、Thinking、Search、Air 等多个模型
- **流式和非流式**：同时支持流式（SSE）和非流式响应模式
- **匿名Token支持**：无需Z.AI账号，自动获取临时匿名Token，零配置部署
- **智能Token降级**：配置Token → 匿名Token → 缓存Token 三级降级策略
- Claude Code支持：支持通过Claude Code Router接入到CC中
- Code插件支持：支持Cline、Roo Code、Kilo Code等第三方插件

### 工具调用（Function Calling）

- **标准 OpenAI 工具调用**：完全兼容 OpenAI 的 tools 和 tool_choice 参数
- **多函数并行调用**：支持在一次响应中调用多个函数
- **智能 XML 解析**：自动检测和解析模型响应中的工具调用
- **流式实时检测**：在流式模式下实时检测工具调用触发
- **Think 标签过滤**：智能过滤思考过程，只返回工具调用结果
- **自定义提示词**：支持自定义工具调用提示词模板

### 其他功能

- **健康检查**：提供 `/health` 端点用于服务监控
- **CORS 支持**：内置跨域支持，方便前端集成
- **代理配置**：支持 HTTP/HTTPS 代理
- **Token 认证**：支持 API Key 认证和跳过认证选项
- **智能Token缓存**：30分钟缓存策略，减少API调用频率
- **重试机制**：请求失败自动重试
- **详细日志**：可配置的详细日志输出

## 快速开始

### 方式一：Docker 部署（推荐）

使用 Docker Compose 快速部署，这是最简单的方式：

#### 1. 配置环境

复制环境变量模板并配置：

```bash
cp env_template.txt .env
```

编辑 `.env` 文件，配置以下关键参数：

```env
# Z.AI API 配置
API_ENDPOINT=https://chat.z.ai/api/chat/completions
# ZAI_TOKEN=your_zai_token        # 可选：不配置将自动使用匿名Token
ZAI_SIGNING_SECRET=junjie

# 服务器配置
LISTEN_PORT=8080
AUTH_TOKEN=sk-123456

# 匿名Token配置（推荐保持默认值）
ENABLE_GUEST_TOKEN=true          # 启用匿名Token功能
GUEST_TOKEN_CACHE_MINUTES=55     # 缓存55分钟

# 功能开关
ENABLE_TOOLIFY=true              # 启用工具调用功能
DEBUG_LOGGING=false              # 生产环境建议关闭
```

### 零配置部署模式

如果你不想配置任何Token，可以直接使用匿名模式：

```env
# 最小化配置 - 仅需要这两个基本参数
LISTEN_PORT=8080
AUTH_TOKEN=sk-123456

# 其他参数会使用默认值，自动启用匿名Token模式
```

#### 2. 启动服务

```bash
# 构建并启动容器
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

服务将在 `http://localhost:8080` 启动。

#### 3. 更新服务

```bash
# 拉取最新代码后重新构建
docker-compose up -d --build
```

### 方式二：本地部署

如果你更喜欢在本地环境直接运行：

#### 1. 安装依赖

```bash
pip install -r requirements.txt
```

#### 2. 配置环境

复制环境变量模板并配置：

```bash
cp env_template.txt .env
```

编辑 `.env` 文件，配置以下关键参数（同上）。

#### 3. 启动服务

```bash
python main.py
```

服务将在 `http://localhost:8080` 启动。

### 4. 使用 API

使用 OpenAI SDK 或任何兼容 OpenAI 的客户端：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-123456"  # 与 .env 中的 AUTH_TOKEN 一致
)

# 普通对话
response = client.chat.completions.create(
    model="GLM-4.5",
    messages=[
        {"role": "user", "content": "你好"}
    ]
)

# 工具调用
response = client.chat.completions.create(
    model="GLM-4.5",
    messages=[
        {"role": "user", "content": "今天北京天气怎么样？"}
    ],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称"
                    }
                },
                "required": ["city"]
            }
        }
    }]
)
```

## 配置说明

### 环境变量

| 变量名                     | 说明                  | 默认值                                     |
| -------------------------- | --------------------- | ------------------------------------------ |
| `API_ENDPOINT`           | Z.AI API 地址         | `https://chat.z.ai/api/chat/completions` |
| `ZAI_TOKEN`              | Z.AI 认证 Token（可选）| -                                          |
| `ZAI_SIGNING_SECRET`     | 签名密钥              | `junjie`                                 |
| `AUTH_TOKEN`             | API Key（客户端认证） | `sk-your-api-key`                        |
| `LISTEN_PORT`            | 服务监听端口          | `8080`                                   |
| `ENABLE_GUEST_TOKEN`     | 启用匿名Token功能     | `true`                                   |
| `GUEST_TOKEN_CACHE_MINUTES` | 匿名Token缓存时间（分钟） | `55`                                     |
| `ZAI_AUTH_ENDPOINT`      | Z.AI认证API端点       | `https://chat.z.ai/api/v1/auths/`        |
| `PRIMARY_MODEL`          | 主模型名称            | `GLM-4.5`                                |
| `THINKING_MODEL`         | 思考模型名称          | `GLM-4.5-Thinking`                       |
| `SEARCH_MODEL`           | 搜索模型名称          | `GLM-4.5-Search`                         |
| `AIR_MODEL`              | Air 模型名称          | `GLM-4.5-Air`                            |
| `GLM_45V_MODEL`          | GLM-4.5V 视觉模型     | `GLM-4.5V`                               |
| `GLM_46_MODEL`           | GLM-4.6 模型名称      | `GLM-4.6`                                |
| `GLM_46_THINKING_MODEL`  | GLM-4.6 思考模型      | `GLM-4.6-Thinking`                       |
| `GLM_46_SEARCH_MODEL`    | GLM-4.6 搜索模型      | `GLM-4.6-Search`                         |
| `ENABLE_TOOLIFY`         | 启用工具调用功能      | `true`                                   |
| `TOOLIFY_CUSTOM_PROMPT`  | 自定义工具调用提示词  | -                                          |
| `DEBUG_LOGGING`          | 详细日志输出          | `true`                                   |
| `SKIP_AUTH_TOKEN`        | 跳过认证（用于测试）  | `false`                                  |
| `MAX_RETRIES`            | 请求重试次数          | `3`                                      |
| `HTTP_PROXY`             | HTTP 代理地址         | -                                          |
| `HTTPS_PROXY`            | HTTPS 代理地址        | -                                          |

## API 端点

| 端点                     | 方法 | 说明                    |
| ------------------------ | ---- | ----------------------- |
| `/`                    | GET  | 服务信息                |
| `/health`              | GET  | 健康检查                |
| `/v1/chat/completions` | POST | 聊天补全（兼容 OpenAI） |
| `/v1/models`           | GET  | 模型列表                |

## 工具调用示例

### 单函数调用

```python
tools = [{
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "执行数学计算",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "数学表达式"}
            },
            "required": ["expression"]
        }
    }
}]

response = client.chat.completions.create(
    model="GLM-4.5",
    messages=[{"role": "user", "content": "计算 123 * 456"}],
    tools=tools
)
```

### 多函数调用

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取天气信息",
            "parameters": {...}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_restaurants",
            "description": "搜索餐厅",
            "parameters": {...}
        }
    }
]

response = client.chat.completions.create(
    model="GLM-4.5",
    messages=[{"role": "user", "content": "北京今天天气怎么样？附近有什么好吃的？"}],
    tools=tools
)
```

### 指定工具选择策略

```
# 强制调用指定函数
response = client.chat.completions.create(
    model="GLM-4.5",
    messages=[{"role": "user", "content": "现在几点？"}],
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "get_current_time"}}
)

# 强制调用任意函数
response = client.chat.completions.create(
    model="GLM-4.5",
    messages=[{"role": "user", "content": "帮我查询信息"}],
    tools=tools,
    tool_choice="required"
)
```

### 流式工具调用

```
stream = client.chat.completions.create(
    model="GLM-4.5",
    messages=[{"role": "user", "content": "今天天气如何？"}],
    tools=tools,
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.tool_calls:
        print(chunk.choices[0].delta.tool_calls)
```

## 测试

运行集成测试：

```bash
# 工具调用集成测试
python tests/test_toolify_integration.py

# 流式模式测试
python tests/test_stream_modes.py

# 非流式测试
python tests/test_non_stream.py
```

## 故障排除

### 工具调用未生效

1. 确认 `ENABLE_TOOLIFY=true`
2. 检查请求中是否包含 `tools` 参数
3. 查看日志中的 `[TOOLIFY]` 信息

### 认证失败

1. 确认 `.env` 中的 `ZAI_TOKEN` 配置正确，或启用匿名模式
2. 检查客户端的 `api_key` 是否与 `AUTH_TOKEN` 一致
3. 如果使用匿名模式，确认 `ENABLE_GUEST_TOKEN=true`


### 模型不可用

1. 检查 Z.AI API 是否可访问
2. 确认模型名称配置正确
3. 查看 Z.AI Token 是否有效
4. 在匿名模式下，某些高级功能可能受限

### 查看详细日志

设置 `DEBUG_LOGGING=true` 并重启服务，查看完整的请求和响应日志。

## 支持的模型

| 模型名称         | Z.AI 后端模型  | 说明                     |
| ---------------- | -------------- | ----------------------- |
| GLM-4.5          | 0727-360B-API  | 主模型                   |
| GLM-4.5-Thinking | 0727-360B-API  | 思考模型                 |
| GLM-4.5-Search   | 0727-360B-API  | 搜索模型                 |
| GLM-4.5-Air      | 0727-106B-API  | 轻量级模型               |
| GLM-4.5V         | glm-4.5v       | 视觉模型                 |
| GLM-4.6          | GLM-4-6-API-V1 | 4.6 版本                 |
| GLM-4.6-Thinking | GLM-4-6-API-V1 | 4.6 思考版本             |
| GLM-4.6-Search   | GLM-4-6-API-V1 | 4.6 搜索版本             |

`所有模型均支持图像上传`

## 参考项目

本项目参考了以下开源项目：

1. [z.ai2api_python](https://github.com/ZyphrZero/z.ai2api_python)
2. [Toolify](https://github.com/funnycups/Toolify)
