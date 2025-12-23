# OpenAI 兼容 API 服务器

一个完全兼容 OpenAI API 格式的代理服务器，将 OpenAI 格式的请求转换为 Z.AI API 调用。（不再支持工具调用，时常失灵）

## 功能特性

### 核心功能

- **OpenAI 格式兼容**：完全兼容 OpenAI API 请求和响应格式
- **Z.AI 后端集成**：自动转换并调用 Z.AI API
- **多模型支持**：支持 GLM-4.7（最新旗舰模型）、GLM-4.6、GLM-4.5、Thinking、V系列等多个模型
- **流式和非流式**：同时支持流式（SSE）和非流式响应模式
- **Code插件支持**：支持Cline、Roo Code、Kilo Code等第三方插件

### 其他功能

- **健康检查**：提供 `/health` 端点用于服务监控
- **CORS 支持**：内置跨域支持，方便前端集成
- **代理配置**：支持 HTTP/HTTPS 代理
- **Token 认证**：支持 API Key 认证和跳过认证选项
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
ZAI_TOKEN=your_zai_token        # 必须配置
ZAI_SIGNING_SECRET=junjie

# 服务器配置
LISTEN_PORT=8080
AUTH_TOKEN=sk-123456

# 功能开关
DEBUG_LOGGING=false              # 生产环境建议关闭
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
```

## 配置说明

### 环境变量

| 变量名                     | 说明                  | 默认值                                      |
| -------------------------- | --------------------- |------------------------------------------|
| `API_ENDPOINT`           | Z.AI API 地址         | `https://chat.z.ai/api/chat/completions` |
| `ZAI_TOKEN`              | Z.AI 认证 Token（必须）| -                                        |
| `ZAI_SIGNING_SECRET`     | 签名密钥              | `junjie`                                 |
| `AUTH_TOKEN`             | API Key（客户端认证） | `sk-your-api-key`                        |
| `LISTEN_PORT`            | 服务监听端口          | `8080`                                   |
| `UVICORN_WORKERS`        | Uvicorn 工作进程数     | `4`                                      |
| `PRIMARY_MODEL`          | 主模型名称            | `GLM-4.5`                                |
| `THINKING_MODEL`         | 思考模型名称          | `GLM-4.5-Thinking`                       |
| `AIR_MODEL`              | Air 模型名称          | `GLM-4.5-Air`                            |
| `GLM_45V_MODEL`          | GLM-4.5V 视觉模型     | `GLM-4.5V`                               |
| `GLM_46_MODEL`           | GLM-4.6 模型名称      | `GLM-4.6`                                |
| `GLM_46_THINKING_MODEL`  | GLM-4.6 思考模型      | `GLM-4.6-Thinking`                       |
| `GLM_47_MODEL`           | GLM-4.7 旗舰模型（最新）| `GLM-4.7`                                |
| `GLM_47_THINKING_MODEL`  | GLM-4.7 思考模型      | `GLM-4.7-Thinking`                       |
| `DEBUG_LOGGING`          | 详细日志输出          | `true`                                   |
| `SKIP_AUTH_TOKEN`        | 跳过认证（用于测试）  | `false`                                  |
| `MAX_RETRIES`            | 请求重试次数          | `3`                                      |
| `HTTP_PROXY`             | HTTP 代理地址         | -                                        |
| `HTTPS_PROXY`            | HTTPS 代理地址        | -                                        |

## API 端点

| 端点                     | 方法 | 说明                    |
| ------------------------ | ---- | ----------------------- |
| `/`                    | GET  | 服务信息                |
| `/health`              | GET  | 健康检查                |
| `/v1/chat/completions` | POST | 聊天补全（兼容 OpenAI） |
| `/v1/models`           | GET  | 模型列表                |

## 故障排除

### 认证失败

1. 确认 `.env` 中的 `ZAI_TOKEN` 配置正确
2. 检查客户端的 `api_key` 是否与 `AUTH_TOKEN` 一致

### 模型不可用

1. 检查 Z.AI API 是否可访问
2. 确认模型名称配置正确
3. 查看 Z.AI Token 是否有效

### 查看详细日志

设置 `DEBUG_LOGGING=true` 并重启服务，查看完整的请求和响应日志。

## 支持的模型

| 模型名称         | Z.AI 后端模型  | 说明                     |
| ---------------- | -------------- | ----------------------- |
| GLM-4.7          | GLM-4-7-API    | 最新旗舰模型             |
| GLM-4.7-Thinking | GLM-4-7-API    | 4.7 思考版本             |
| GLM-4.6          | GLM-4-6-API-V1 | 4.6 版本                 |
| GLM-4.6-Thinking | GLM-4-6-API-V1 | 4.6 思考版本             |
| GLM-4.5          | 0727-360B-API  | 主模型                   |
| GLM-4.5-Thinking | 0727-360B-API  | 思考模型                 |
| GLM-4.5-Air      | 0727-106B-API  | 轻量级模型               |
| GLM-4.5V         | glm-4.5v       | 视觉模型                 |
| GLM-4.6V         | glm-4.6v       | 视觉旗舰模型             |

所有模型均支持图像上传

## 参考项目

本项目参考了以下开源项目：

1. [z.ai2api_python](https://github.com/ZyphrZero/z.ai2api_python)