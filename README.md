# OpenAI 兼容 API 服务器

一个完全兼容 OpenAI API 格式的代理服务器，将 OpenAI 格式的请求转换为 Z.AI API 调用。

## 功能特性

- **OpenAI 格式兼容**：兼容 OpenAI API 请求/响应格式，支持流式（SSE）和非流式模式
- **多模型支持**：支持 GLM-5、GLM-4.7、GLM-4.6、GLM-4.5、Thinking、V系列等多个模型
- **插件兼容**：支持 Cline、Roo Code、Kilo Code 等第三方插件
- **运维友好**：健康检查、CORS、代理配置、Token 认证、请求重试、可配置日志

## 快速开始

### 环境配置（通用）

```bash
cp env_template.txt .env
```

编辑 `.env` 文件，配置关键参数：

```env
ZAI_TOKEN=your_zai_token        # 必须配置
AUTH_TOKEN=sk-123456             # 客户端认证用
LISTEN_PORT=8080
DEBUG_LOGGING=false              # 生产环境建议关闭
```

完整配置项见 `env_template.txt`。

### Docker 部署（推荐）

```bash
docker-compose up -d             # 启动
docker-compose logs -f           # 查看日志
docker-compose down              # 停止
docker-compose up -d --build     # 更新重建
```

### 本地部署

```bash
pip install -r requirements.txt
python main.py
```

服务默认在 `http://localhost:8080` 启动。

### 使用示例

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-123456"  # 与 .env 中的 AUTH_TOKEN 一致
)

response = client.chat.completions.create(
    model="GLM-4.5",
    messages=[{"role": "user", "content": "你好"}]
)
```

## API 端点

| 端点                     | 方法 | 说明                    |
| ------------------------ | ---- | ----------------------- |
| `/`                    | GET  | 服务信息                |
| `/health`              | GET  | 健康检查                |
| `/v1/chat/completions` | POST | 聊天补全（兼容 OpenAI） |
| `/v1/models`           | GET  | 模型列表                |

## 支持的模型

| 模型名称         | Z.AI 后端模型  | 说明             |
| ---------------- | -------------- | ---------------- |
| GLM-5            | glm-5          | 最新旗舰模型     |
| GLM-5-Think      | glm-5          | GLM-5 思考版本   |
| GLM-4.7          | glm-4.7        | 旗舰模型         |
| GLM-4.7-Thinking | glm-4.7        | 4.7 思考版本     |
| GLM-4.6          | GLM-4-6-API-V1 | 4.6 版本         |
| GLM-4.6-Thinking | GLM-4-6-API-V1 | 4.6 思考版本     |
| GLM-4.5          | 0727-360B-API  | 主模型           |
| GLM-4.5-Thinking | 0727-360B-API  | 思考模型         |
| GLM-4.5-Air      | 0727-106B-API  | 轻量级模型       |
| GLM-4.5V         | glm-4.5v       | 视觉模型         |
| GLM-4.6V         | glm-4.6v       | 视觉旗舰模型     |


## 参考项目

- [z.ai2api_python](https://github.com/ZyphrZero/z.ai2api_python)