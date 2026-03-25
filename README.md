# OpenAI 兼容 API 服务器

一个完全兼容 OpenAI API 格式的代理服务器，将 OpenAI 格式的请求转换为 Z.AI API 调用。

## 功能特性

- **OpenAI 格式兼容**：兼容 OpenAI API 请求/响应格式，支持流式（SSE）和非流式模式
- **多模型支持**：支持 `glm-4.6v` 视觉模型、`glm-5` 与 `glm-4.7`
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
    model="glm-5",
    messages=[{"role": "user", "content": "你好"}]
)
```

如需为 `glm-5` 或 `glm-4.7` 显式开启思考模式，可以在请求体中使用以下任一字段：

- `thinking: {"type": "enabled"}`
- `reasoning_effort`（值不为 `none`）
- `enable_thinking: true`

当多个字段同时存在时，按以下优先级解析：

`thinking` > `reasoning_effort` > `enable_thinking`

## API 端点

| 端点                     | 方法 | 说明                    |
| ------------------------ | ---- | ----------------------- |
| `/`                    | GET  | 服务信息                |
| `/health`              | GET  | 健康检查                |
| `/v1/chat/completions` | POST | 聊天补全（兼容 OpenAI） |
| `/v1/models`           | GET  | 模型列表                |

| 模型名称 | Z.AI 后端模型 | 说明 |
| -------- | ------------- | ---- |
| glm-4.6v | glm-4.6v | 视觉旗舰模型 |
| glm-5 | glm-5 | 最新旗舰模型 |
| glm-4.7 | glm-4.7 | 与 `glm-5` 共享同一套思考开关规则 |

兼容性说明：

- 公开模型名以 `/v1/models` 返回结果为准：`glm-4.6v`、`glm-5`、`glm-4.7`
- 为兼容旧客户端，请求中仍接受 `GLM-4.6V`、`GLM-5`、`GLM-5-Think`
- 其中 `GLM-5-Think` 在未显式传入新开关字段时，会默认按思考模式处理


## 参考项目

- [z.ai2api_python](https://github.com/ZyphrZero/z.ai2api_python)
