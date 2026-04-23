# Multi-Agent Platform — 学习与开发规划

> 从 0 到 1，构建自己的多 Agent 协作平台。每个阶段都是可独立运行、可验收的里程碑。

---

## 总览：阶段路线图

| 阶段 | 主题 | 里程碑 |
|------|------|--------|
| **Phase 1** | 最简 Agent（ReAct 模式） | 一个能自主调用工具完成任务的 Agent |
| **Phase 2** | 记忆系统（短期 + 长期） | Agent 能记住历史对话和用户偏好 |
| **Phase 3** | RAG 知识库 | Agent 能基于私有文档回答用户问题 |
| **Phase 4** | Skills 可插拔系统 | 标准化的工具注册/发现/调用机制 |
| **Phase 5** | 多 Agent 协作 | 多个 Agent 分工协作完成复杂任务 |
| **Phase 6** | 性能优化与稳定性 | 流式输出、缓存、并发、错误容灾 |

---

## Phase 1：最简 Agent（ReAct 模式）

### 1.1 当前目标

构建一个基于 **ReAct（Reasoning + Acting）** 模式的单 Agent，具备以下能力：

- 接收用户自然语言输入
- 自主规划：决定是否需要调用工具来获取信息
- 工具调用：通过函数调用获取计算结果、查询结果等
- 推理与回答：基于工具返回结果，生成最终答案
- 循环执行，直到得出最终结论

### 1.2 技术选型

| 组件 | 选型 | 说明 |
|------|------|------|
| LLM API | OpenAI / DeepSeek | 直接 HTTP 调用，不依赖 SDK |
| 工具调用 | 函数注册 + JSON 解析 | LLM 返回工具名和参数，本地执行函数 |
| HTTP 客户端 | `httpx` | 异步调用 LLM API |
| 配置管理 | 环境变量 + config.py | API Key 等配置集中管理 |

### 1.3 项目结构

```
multi-agent-platform/
├── core/
│   ├── __init__.py
│   ├── agent.py          # ReAct Agent 核心循环
│   ├── llm.py            # LLM API 封装（直接 HTTP 调用）
│   └── tool.py           # Tool 基类 + 注册中心
├── tools/
│   ├── __init__.py
│   └── builtin.py        # 内置工具（计算器、时间查询、天气等）
├── config.py             # 配置管理
├── main.py               # 入口（交互式 CLI）
└── requirements.txt      # 依赖
```

### 1.4 流程图

```
用户输入问题
    │
    ▼
┌──────────────────────────────┐
│ 1. 构建 System Prompt        │
│    - 包含可用工具的定义       │
│    - 包含 ReAct 格式指令      │
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│ 2. 调用 LLM API              │
│    发送 messages 数组          │
└──────────────────────────────┘
    │
    ▼
    LLM 返回
    │
    ├── 方案 A：直接回答 ─────────────────────┐
    │                                          │
    ├── 方案 B：需要调用工具                   │
    │      │                                   │
    │      ▼                                   │
    │   ┌─────────────────────────────┐        │
    │   │ 3. 解析工具调用请求          │        │
    │   │    - 工具名                  │        │
    │   │    - 参数 (JSON)             │        │
    │   └─────────────────────────────┘        │
    │      │                                   │
    │      ▼                                   │
    │   ┌─────────────────────────────┐        │
    │   │ 4. 执行工具函数              │        │
    │   │    tool_registry[name](**args)│       │
    │   └─────────────────────────────┘        │
    │      │                                   │
    │      ▼                                   │
    │   ┌─────────────────────────────┐        │
    │   │ 5. 将工具结果加入 messages   │        │
    │   │    追加 observation         │        │
    │   └─────────────────────────────┘        │
    │      │                                   │
    │      └────── 回到步骤 2 ─────────────────┘
    │
    ▼
返回最终答案给用户
```

### 1.5 开发操作步骤

#### Step 1：初始化项目

1. 创建项目目录和虚拟环境：
   ```bash
   mkdir -p multi-agent-platform
   cd multi-agent-platform
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

2. 创建 `requirements.txt`，初始依赖：
   ```
   httpx>=0.27.0
   python-dotenv>=1.0.0
   ```

3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

4. 创建 `.env` 文件，写入 LLM API 配置：
   ```
   # 使用 OpenAI 或 DeepSeek（二选一）
   LLM_PROVIDER=openai
   API_KEY=your-api-key-here
   MODEL=gpt-4o-mini
   BASE_URL=https://api.openai.com/v1
   ```

#### Step 2：实现配置模块 — `config.py`

创建 `config.py`，负责读取环境变量：

```python
import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
API_KEY = os.getenv("API_KEY", "")
MODEL = os.getenv("MODEL", "gpt-4o-mini")
BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1/chat/completions")
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))
```

> **提示**：使用 `python-dotenv` 加载 `.env` 文件。API Key 不要硬编码。

#### Step 3：实现 Tool 基类和注册中心 — `core/tool.py`

定义 Tool 的抽象接口和注册机制：

```python
from abc import ABC, abstractmethod
from typing import Any

class Tool(ABC):
    """工具基类，所有可调用工具必须继承此类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称，LLM 通过此名称调用工具"""

    @property
    @abstractmethod
    def description(self) -> str:
        """工具功能描述，LLM 根据此描述决定是否调用"""

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """工具参数定义（OpenAI function calling 格式）"""

    @abstractmethod
    async def execute(self, **kwargs) -> str:
        """执行工具并返回结果"""

class ToolRegistry:
    """工具注册中心，管理所有可用工具"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def get_openai_tools_schema(self) -> list[dict]:
        """返回 OpenAI API 兼容的工具定义数组"""
        return [tool.parameters for tool in self._tools.values()]
```

> **验收点 1**：能创建 ToolRegistry，注册和查询工具。可以写一个简单的测试脚本验证。

#### Step 4：实现内置工具 — `tools/builtin.py`

实现几个简单的工具用于测试：

```python
import datetime
from core.tool import Tool

class CalculatorTool(Tool):
    """计算器工具"""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Perform mathematical calculations. Input should be a valid math expression like '2 + 2' or '3.14 * 5'."

    @property
    def parameters(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Perform mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "A valid math expression, e.g. '2 + 2', '3.14 * 5'",
                        }
                    },
                    "required": ["expression"],
                },
            },
        }

    async def execute(self, expression: str) -> str:
        try:
            # 安全起见，只允许数字和基础运算符
            allowed = set("0123456789+-*/.() ")
            if not all(c in allowed for c in expression):
                return "Error: Invalid characters in expression"
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"


class CurrentTimeTool(Tool):
    """当前时间查询工具"""

    @property
    def name(self) -> str:
        return "current_time"

    @property
    def description(self) -> str:
        return "Get the current date and time"

    @property
    def parameters(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "current_time",
                "description": "Get the current date and time",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }

    async def execute(self, **kwargs) -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

> **验收点 2**：能实例化工具，调用 `execute()` 方法能正确返回结果。

#### Step 5：实现 LLM API 封装 — `core/llm.py`

直接通过 HTTP 调用 LLM API（OpenAI 兼容接口）：

```python
import json
from typing import AsyncGenerator
import httpx
from config import API_KEY, BASE_URL, MODEL


async def chat(
    messages: list[dict],
    tools: list[dict] | None = None,
) -> dict:
    """
    调用 LLM API，返回完整回复。

    Args:
        messages: 对话消息列表 [{"role": "user"|"assistant", "content": "..."}]
        tools: 工具定义列表（OpenAI 格式）

    Returns:
        {
            "content": "LLM 的回复内容",
            "tool_calls": [{"name": "工具名", "arguments": {"arg": "value"}}],
            "finish_reason": "stop" | "tool_calls"
        }
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
    }

    if tools:
        payload["tools"] = tools

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(BASE_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    choice = data["choices"][0]["message"]
    content = choice.get("content", "")
    finish_reason = data["choices"][0].get("finish_reason", "stop")

    tool_calls = []
    if choice.get("tool_calls"):
        for tc in choice["tool_calls"]:
            tool_calls.append({
                "name": tc["function"]["name"],
                "arguments": json.loads(tc["function"]["arguments"]),
            })

    return {
        "content": content,
        "tool_calls": tool_calls,
        "finish_reason": finish_reason,
    }
```

> **提示**：这里使用 OpenAI 兼容接口格式，如果你使用 DeepSeek、Claude 等，参数结构略有不同，但核心逻辑一致。

#### Step 6：实现 ReAct Agent 核心 — `core/agent.py`

这是 Phase 1 的核心——实现 ReAct 循环：

```python
from core.llm import chat
from core.tool import ToolRegistry

SYSTEM_PROMPT = """\
You are a helpful assistant that can use tools to solve user's problems.

You must follow the ReAct (Reasoning + Acting) pattern:
1. Think about what you need to do (Thought:)
2. Decide which tool to use (Action:)
3. Observe the tool's result (Observation:)
4. Continue until you can give a final answer (Final Answer:)

You have access to the following tools:
{tools_description}

Rules:
- Always start with a Thought explaining your reasoning
- Use tools when you need to compute, look up, or calculate something
- Only call one tool at a time
- When you have enough information, give the Final Answer directly
- Format your response as JSON with "thought", "action", and "action_input" fields
  OR format as {"final_answer": "your answer"} when done
"""


class Agent:
    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry
        self.max_iterations = 10

    def _build_system_prompt(self) -> str:
        tools_desc = "\n".join(
            f"- {tool.name}: {tool.description}"
            for tool in self.tool_registry.list_tools()
        )
        return SYSTEM_PROMPT.format(tools_description=tools_desc)

    async def run(self, user_input: str) -> str:
        """执行 Agent 推理循环，返回最终答案"""
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_input},
        ]

        for iteration in range(self.max_iterations):
            response = await chat(messages)

            if response["tool_calls"]:
                # LLM 要求调用工具
                for tc in response["tool_calls"]:
                    tool = self.tool_registry.get(tc["name"])
                    if tool:
                        result = await tool.execute(**tc["arguments"])
                    else:
                        result = f"Error: Tool '{tc['name']}' not found"

                    # 将工具调用和结果加入对话历史
                    messages.append({
                        "role": "assistant",
                        "content": f"Calling tool: {tc['name']} with args: {tc['arguments']}",
                    })
                    messages.append({
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tc.get("id", ""),
                    })
            else:
                # LLM 直接回答，循环结束
                return response["content"]

        return "达到最大迭代次数，未能完成任务。"
```

> **验收点 3**：能创建 Agent 实例，传入 tool_registry，调用 `agent.run("12乘以34等于几？")` 能正确调用计算器并返回结果。

#### Step 7：实现 CLI 入口 — `main.py`

```python
import asyncio
from core.agent import Agent
from core.tool import ToolRegistry
from tools.builtin import CalculatorTool, CurrentTimeTool


async def main():
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(CurrentTimeTool())

    agent = Agent(registry)

    print("Agent 已启动！输入你的问题（输入 'quit' 退出）")
    while True:
        user_input = input("\n> ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        result = await agent.run(user_input)
        print(f"\n{result}")


if __name__ == "__main__":
    asyncio.run(main())
```

> **验收点 4（Phase 1 最终验收）**：
> - 运行 `python main.py` 能启动交互界面
> - 输入 "12 乘以 34 等于几？" → Agent 自动调用计算器工具，返回正确结果
> - 输入 "现在几点了？" → Agent 调用时间工具，返回当前时间
> - 输入 "你好，请介绍一下你自己" → Agent 直接回答，不调用工具
> - 输入 "quit" → 正常退出

---

### 1.6 Phase 1 验收清单

- [ ] 项目结构创建完毕，能正常运行
- [ ] `Tool` 抽象类和 `ToolRegistry` 实现完毕
- [ ] 至少实现 2 个内置工具
- [ ] LLM API 封装能正常调用并返回结果
- [ ] ReAct Agent 的推理循环能正确执行
- [ ] CLI 交互能正常使用
- [ ] Agent 能根据问题自主决定是否调用工具
- [ ] 工具调用结果能正确反馈给 LLM 形成闭环

---

_完成 Phase 1 后，进入 Phase 2：为 Agent 添加记忆能力。_

---

## Phase 2：记忆系统（短期 + 长期）

### 2.1 当前目标

在 Phase 1 单轮对话的基础上，为 Agent 添加**两层记忆能力**：

- **短期记忆（Short-Term Memory）** — 维护多轮对话上下文，支持对话历史管理（截断、token 计数、摘要压缩）
- **长期记忆（Long-Term Memory）** — 将用户偏好、重要事实持久化到向量数据库，跨会话检索和复用

完成本阶段后，Agent 应能：

1. 记住同一会话中之前说过的话（多轮对话）
2. 记住跨会话的用户偏好和关键信息（长期记忆）
3. 自动从长期记忆中检索相关信息辅助回答

### 2.2 技术选型

| 组件 | 选型 | 说明 |
|------|------|------|
| 短期记忆 | 内存中 messages 列表 + 摘要压缩 | 自研对话管理器 |
| 长期记忆存储 | Chroma（Python 原生，无外部服务） | 轻量向量数据库 |
| Embedding | 直接调 LLM API 的 embedding 接口 | 与 Phase 1 共用 HTTP 调用模式 |
| 记忆归档 | LLM 自动生成摘要 → 存入向量库 | 对话结束时触发归档 |

### 2.3 项目结构（增量变更）

在 Phase 1 基础上，新增 `memory/` 模块：

```
multi-agent-platform/
├── core/
│   ├── __init__.py
│   ├── agent.py          # 改造：接入 MemoryManager
│   ├── llm.py            # 新增：embedding() 函数
│   └── tool.py           # 不变
├── memory/               # 【新增】记忆模块
│   ├── __init__.py
│   ├── short_term.py     # 短期记忆管理（对话历史）
│   ├── long_term.py      # 长期记忆管理（向量存储）
│   ├── memory_manager.py # 记忆管理器（协调短期 + 长期）
│   └── summarizer.py     # 对话摘要生成
├── tools/
│   ├── __init__.py
│   └── builtin.py        # 不变
├── data/                 # 【新增】Chroma 持久化目录（.gitignore）
│   └── .gitkeep
├── config.py             # 新增：MEMORY_DIR、EMBEDDING_MODEL 等配置
├── main.py               # 改造：接入 MemoryManager
└── requirements.txt      # 新增：chromadb
```

### 2.4 流程图

```
用户发送消息
    │
    ▼
┌───────────────────────────────────────┐
│ 1. 短期记忆：加载当前会话历史          │
│    - 将用户消息追加到 messages 列表    │
│    - 检查 token 数量，超限则压缩      │
│    - 压缩策略：摘要最早对话            │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│ 2. 长期记忆：检索相关信息              │
│    - 对用户消息生成 Embedding          │
│    - 向量库中搜索 Top-K 相关记忆       │
│    - 将检索结果注入 System Prompt      │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│ 3. Agent 执行 ReAct 循环              │
│    - 此时 Agent 拥有短期 + 长期上下文  │
│    - 工具调用流程不变                  │
└───────────────────────────────────────┘
    │
    ▼
    LLM 返回最终答案
    │
    ▼
┌───────────────────────────────────────┐
│ 4. 短期记忆：保存本轮对话到 messages   │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│ 5. 长期记忆归档（可选，定时触发）       │
│    - 用 LLM 从近期对话中提取关键事实   │
│    - 将事实生成 Embedding → 存入 Chroma│
└───────────────────────────────────────┘
    │
    ▼
返回答案给用户
```

### 2.5 开发操作步骤

#### Step 1：更新依赖和配置

1. 更新 `requirements.txt`，新增：
   ```
   chromadb>=0.5.0
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

3. 更新 `config.py`，新增记忆相关配置：
   ```python
   MEMORY_DIR = os.getenv("MEMORY_DIR", "./data/memory")
   EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
   MAX_CONTEXT_MESSAGES = int(os.getenv("MAX_CONTEXT_MESSAGES", "20"))
   MEMORY_TOP_K = int(os.getenv("MEMORY_TOP_K", "3"))
   ```

#### Step 2：LLM API 新增 Embedding 支持 — `core/llm.py`

在已有的 `chat()` 函数旁边，新增 `embed()` 函数：

```python
async def embed(texts: list[str]) -> list[list[float]]:
    """
    调用 LLM API 的 embedding 接口，返回文本的向量表示。

    Args:
        texts: 待编码的文本列表

    Returns:
        向量列表，每个向量是一个 float 列表
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": EMBEDDING_MODEL,
        "input": texts,
    }

    # embedding 接口通常在 BASE_URL 上去掉 /chat/completions 后缀
    embedding_url = BASE_URL.replace("/chat/completions", "/embeddings")

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(embedding_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    return [item["embedding"] for item in data["data"]]
```

> **提示**：确保 `config.py` 中的 `EMBEDDING_MODEL` 与你的 API 提供商支持的 embedding 模型匹配。

> **验收点 1**：单独调用 `embed(["测试文本"])` 能返回一个浮点数组（维度取决于所用模型，OpenAI 的 text-embedding-3-small 是 1536 维）。

#### Step 3：实现短期记忆 — `memory/short_term.py`

短期记忆管理多轮对话上下文，核心问题是：**如何让 LLM 在有限的 context window 内存下记住足够多的历史**。

```python
from typing import Optional


class ShortTermMemory:
    """短期记忆：管理单会话的对话历史"""

    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self.messages: list[dict] = []
        self.summary: str = ""  # 已压缩对话的摘要

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def get_messages(self) -> list[dict]:
        """获取当前可用的对话历史（可能包含摘要）"""
        if self.summary:
            return [
                {"role": "system", "content": f"之前对话的摘要: {self.summary}"},
                *self.messages,
            ]
        return list(self.messages)

    def truncate_if_needed(self) -> None:
        """当消息数量超过上限时，移除最早的非系统消息"""
        if len(self.messages) <= self.max_messages:
            return

        # 移除最早的用户-助手对话对
        excess = len(self.messages) - self.max_messages
        self.messages = self.messages[excess:]

    def clear(self) -> None:
        self.messages.clear()
        self.summary = ""

    @property
    def message_count(self) -> int:
        return len(self.messages)
```

> **要点**：
> - 当对话轮数超过 `max_messages` 时，简单策略是截断最早的内容。后续可升级为 LLM 摘要压缩（将早期对话交给 LLM 生成摘要，释放上下文空间）。
> - 这个截断逻辑先实现简单版本，在 Phase 6 性能优化阶段可以升级为更智能的 token-level 压缩。

#### Step 4：实现长期记忆 — `memory/long_term.py`

长期记忆使用 Chroma 向量库存储和检索记忆条目。

```python
import uuid
from typing import Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from config import MEMORY_DIR, MEMORY_TOP_K


class LongTermMemory:
    """长期记忆：使用向量数据库存储和检索重要事实"""

    def __init__(self):
        self.client = chromadb.PersistentClient(path=MEMORY_DIR)
        self.collection = self.client.get_or_create_collection(
            name="agent_memories",
            metadata={"hnsw:space": "cosine"},
        )
        self.top_k = MEMORY_TOP_K

    def store(self, text: str, metadata: Optional[dict] = None) -> str:
        """
        存储一条记忆。

        Args:
            text: 记忆内容
            metadata: 可选的附加元数据

        Returns:
            记忆 ID
        """
        memory_id = str(uuid.uuid4())
        self.collection.add(
            ids=[memory_id],
            documents=[text],
            metadatas=[metadata or {}],
        )
        return memory_id

    def search(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        """
        搜索与查询相关的记忆。

        Args:
            query: 查询文本
            top_k: 返回前 K 条结果

        Returns:
            匹配的记忆列表 [{"id": "...", "text": "...", "metadata": {...}}]
        """
        k = top_k or self.top_k
        results = self.collection.query(
            query_texts=[query],
            n_results=min(k, self.collection.count()),
        )

        memories = []
        for i in range(len(results["ids"][0])):
            memories.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
            })
        return memories

    def delete(self, memory_id: str) -> None:
        self.collection.delete(ids=[memory_id])

    def count(self) -> int:
        return self.collection.count()
```

> **注意**：Chroma 在最新版本中默认使用内置的 Embedding 模型。如果我们要用 OpenAI 的 Embedding API，需要配置 CustomEmbeddingFunction。为简化，这里先让 Chroma 使用默认 Embedding，后续优化时可以替换。

> **验收点 2**：能存入几条记忆（如 "用户喜欢 Python"、"用户正在学习 AI"），然后搜索 "用户喜欢什么编程语言" 能返回相关结果。

#### Step 5：实现对话摘要生成 — `memory/summarizer.py`

当对话轮数过多时，用 LLM 将早期对话压缩成摘要，腾出上下文空间：

```python
from core.llm import chat

SUMMARIZE_PROMPT = """\
以下是之前的对话内容，请生成一段简洁的摘要，保留关键信息和用户表达的重要偏好/事实。

对话历史:
{dialogue}

请直接输出摘要，不要加多余的前缀。
"""


async def summarize_conversation(messages: list[dict]) -> str:
    """
    使用 LLM 对对话历史生成摘要。

    Args:
        messages: 需要压缩的对话消息列表

    Returns:
        摘要文本
    """
    dialogue_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in messages
    )

    response = await chat([
        {"role": "system", "content": "你是一个对话摘要助手。"},
        {"role": "user", "content": SUMMARIZE_PROMPT.format(dialogue=dialogue_text)},
    ])

    return response["content"].strip()
```

#### Step 6：实现记忆管理器 — `memory/memory_manager.py`

协调短期记忆和长期记忆，对 Agent 层提供统一接口：

```python
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from memory.summarizer import summarize_conversation


class MemoryManager:
    """记忆管理器：统一管理短期和长期记忆"""

    def __init__(self, max_messages: int = 20):
        self.short_term = ShortTermMemory(max_messages=max_messages)
        self.long_term = LongTermMemory()

    async def add_turn(self, user_message: str, assistant_message: str) -> None:
        """添加一轮对话到短期记忆"""
        self.short_term.add_message("user", user_message)
        self.short_term.add_message("assistant", assistant_message)

        # 检查是否需要压缩
        if self.short_term.message_count >= self.short_term.max_messages:
            await self._compress()

    async def get_context_messages(self) -> list[dict]:
        """获取当前完整的对话上下文（含摘要）"""
        return self.short_term.get_messages()

    async def search_long_term(self, query: str) -> str:
        """
        搜索长期记忆，返回格式化的记忆文本，可注入 System Prompt。

        Returns:
            格式化的记忆文本，格式如：
            "相关记忆:\n- 用户喜欢 Python\n- 用户正在学习 AI"
        """
        results = self.long_term.search(query)
        if not results:
            return ""

        memories_text = "\n".join(f"- {m['text']}" for m in results)
        return f"以下是相关的历史记忆:\n{memories_text}"

    def store_long_term(self, text: str, metadata: dict | None = None) -> str:
        """显式存储一条长期记忆"""
        return self.long_term.store(text, metadata)

    async def _compress(self) -> None:
        """压缩早期对话，生成摘要"""
        half = len(self.short_term.messages) // 2
        old_messages = self.short_term.messages[:half]
        old_summary = await summarize_conversation(old_messages)

        # 合并旧摘要和新摘要
        if self.short_term.summary:
            self.short_term.summary = await summarize_conversation([
                {"role": "system", "content": f"之前的摘要: {self.short_term.summary}"},
                {"role": "user", "content": f"新的对话内容: {old_summary}"},
            ])
        else:
            self.short_term.summary = old_summary

        # 移除已压缩的消息
        self.short_term.messages = self.short_term.messages[half:]
```

#### Step 7：改造 Agent — `core/agent.py`

将记忆管理器接入 Agent 的 run 流程：

```python
from core.llm import chat
from core.tool import ToolRegistry
from memory.memory_manager import MemoryManager

SYSTEM_PROMPT_WITH_MEMORY = """\
You are a helpful assistant that can use tools to solve user's problems.

You have access to the following tools:
{tools_description}

{memory_context}

Rules:
- Use tools when you need to compute, look up, or calculate something
- Reference the historical memory when relevant
- Be concise and direct in your answers
"""


class Agent:
    def __init__(self, tool_registry: ToolRegistry, memory_manager: MemoryManager):
        self.tool_registry = tool_registry
        self.memory = memory_manager
        self.max_iterations = 10

    def _build_system_prompt(self, memory_context: str = "") -> str:
        tools_desc = "\n".join(
            f"- {tool.name}: {tool.description}"
            for tool in self.tool_registry.list_tools()
        )
        return SYSTEM_PROMPT_WITH_MEMORY.format(
            tools_description=tools_desc,
            memory_context=memory_context,
        )

    async def run(self, user_input: str) -> str:
        """执行 Agent 推理循环，返回最终答案"""
        # 1. 检索长期记忆
        memory_context = await self.memory.search_long_term(user_input)

        # 2. 构建初始消息
        messages = [
            {"role": "system", "content": self._build_system_prompt(memory_context)},
        ]
        messages.extend(await self.memory.get_context_messages())
        messages.append({"role": "user", "content": user_input})

        # 3. ReAct 循环（与 Phase 1 相同）
        for _ in range(self.max_iterations):
            response = await chat(messages)

            if response["tool_calls"]:
                for tc in response["tool_calls"]:
                    tool = self.tool_registry.get(tc["name"])
                    if tool:
                        result = await tool.execute(**tc["arguments"])
                    else:
                        result = f"Error: Tool '{tc['name']}' not found"

                    messages.append({
                        "role": "assistant",
                        "content": f"Calling tool: {tc['name']} with args: {tc['arguments']}",
                    })
                    messages.append({
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tc.get("id", ""),
                    })
            else:
                # 4. 保存对话到记忆
                answer = response["content"]
                await self.memory.add_turn(user_input, answer)
                return answer

        return "达到最大迭代次数，未能完成任务。"
```

#### Step 8：更新 CLI 入口 — `main.py`

```python
import asyncio
from core.agent import Agent
from core.tool import ToolRegistry
from memory.memory_manager import MemoryManager
from tools.builtin import CalculatorTool, CurrentTimeTool


async def main():
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(CurrentTimeTool())

    memory = MemoryManager(max_messages=10)
    agent = Agent(registry, memory)

    print("Agent 已启动！（支持多轮对话和记忆）")
    print("输入 'quit' 退出，输入 'memories' 查看长期记忆，")
    print("输入 'save_memory <内容>' 手动保存一条长期记忆")

    while True:
        user_input = input("\n> ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        # 管理命令
        if user_input == "memories":
            count = memory.long_term.count()
            print(f"长期记忆数量: {count}")
            continue

        if user_input.startswith("save_memory "):
            content = user_input[len("save_memory "):]
            memory.store_long_term(content)
            print(f"已保存: {content}")
            continue

        result = await agent.run(user_input)
        print(f"\n{result}")


if __name__ == "__main__":
    asyncio.run(main())
```

> **验收点 3（Phase 2 最终验收）**：
> - 多轮对话：连续输入 "我叫张三" → "你叫什么名字？" → Agent 能回答出"你叫张三"
> - 跨会话记忆：输入 `save_memory 用户喜欢 Python` → 退出再启动 → 输入 "你喜欢什么语言？" → Agent 能回忆起之前存储的记忆
> - 上下文压缩：连续对话超过 max_messages 轮次，Agent 仍能记住早期对话摘要
> - 记忆持久化：关闭并重启程序，记忆数据仍然存在

---

### 2.6 Phase 2 验收清单

- [ ] Embedding API 封装正常工作，能返回文本向量
- [ ] ShortTermMemory 能管理多轮对话历史，支持截断
- [ ] LongTermMemory 能存储和检索记忆条目
- [ ] 对话摘要能正常压缩历史对话
- [ ] MemoryManager 协调短期和长期记忆
- [ ] Agent 能记住同一会话中的历史对话
- [ ] Agent 能检索和复用跨会话的长期记忆
- [ ] 记忆数据持久化到磁盘，重启后不丢失

---

_完成 Phase 2 后，进入 Phase 3：为 Agent 添加 RAG 知识库能力。_

---

## Phase 3：RAG 知识库

### 3.1 当前目标

在 Phase 2 基础上，为 Agent 添加 **RAG（Retrieval-Augmented Generation）知识库**能力，使其能够：

- 加载本地文档（txt 起步），自动切分为文本块
- 对文本块生成向量索引并存入向量数据库
- 用户提问时，先检索相关文档片段，再将片段作为上下文注入 LLM 生成回答
- Agent 可将知识库作为工具调用，按需检索

**与 Phase 2 长期记忆的区别**：
- 长期记忆存的是"对话中提炼的事实"，粒度小，面向个人偏好
- 知识库存的是"原始文档内容"，粒度大，面向专业资料（如公司制度、产品文档、技术手册）

### 3.2 技术选型

| 组件 | 选型 | 说明 |
|------|------|------|
| 文档加载 | 自研 Loader 接口 + TxtLoader | 面向接口编程，后续扩展 PDF/MD 等 |
| 文本切分 | 自研 Splitter 接口 + TokenSplitter | 按 token 数切分，支持重叠 |
| 向量索引 | Chroma（复用 Phase 2 已有） | 独立 collection 存储知识库 |
| Embedding | 复用 Phase 2 的 `embed()` | 与长期记忆共用同一个 API |
| 知识库工具 | 实现 `KnowledgeBaseTool` | 将知识库检索注册为 Agent 可调用的工具 |

### 3.3 项目结构（增量变更）

在 Phase 2 基础上，新增 `knowledge/` 模块和 `docs/` 目录：

```
multi-agent-platform/
├── core/
│   ├── __init__.py
│   ├── agent.py          # 改造：注册知识库工具
│   ├── llm.py            # 不变
│   └── tool.py           # 不变
├── knowledge/            # 【新增】知识库模块
│   ├── __init__.py
│   ├── loaders.py        # DocumentLoader 抽象接口 + TxtLoader
│   ├── splitters.py      # DocumentSplitter 抽象接口 + TokenSplitter
│   ├── indexer.py        # 索引器：加载→切分→Embedding→存入向量库
│   ├── retriever.py      # 检索器：查询→向量搜索→返回相关片段
│   └── kb_tool.py        # KnowledgeBaseTool：将检索注册为 Agent 工具
├── memory/
│   ├── __init__.py
│   ├── short_term.py     # 不变
│   ├── long_term.py      # 不变（Chroma 复用同一 client）
│   ├── memory_manager.py # 不变
│   └── summarizer.py     # 不变
├── tools/
│   ├── __init__.py
│   └── builtin.py        # 不变
├── data/
│   └── memory/           # 不变
├── docs/                 # 【新增】存放待索引的文档（.gitignore）
│   └── .gitkeep
├── config.py             # 新增：KB_DIR 等配置
├── main.py               # 改造：支持文档导入、知识库工具注册
└── requirements.txt      # 不变
```

### 3.4 流程图

```
【索引流程】—— 离线执行，通常在导入文档时运行一次

待索引文档
    │
    ▼
┌───────────────────────────────────────┐
│ 1. DocumentLoader 读取原始文本         │
│    - TxtLoader：open() 读取 txt 文件   │
│    - 返回 Document(content=全文)       │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│ 2. DocumentSplitter 切分为文本块       │
│    - TokenSplitter：按 token 数切分    │
│    - 可选块间重叠（overlap）            │
│    - 返回 [chunk1, chunk2, ...]        │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│ 3. Embedding + 存入 Chroma             │
│    - 批量调 embed() 生成向量            │
│    - 存入 KB collection                │
└───────────────────────────────────────┘
    │
    ▼
索引完成，知识库可用


【检索流程】—— 在线执行，用户在对话中提问时触发

用户提问
    │
    ▼
┌───────────────────────────────────────┐
│ 4. Agent 决定调用 KnowledgeBaseTool    │
│    - 将用户问题作为查询语句             │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│ 5. Retriever.search(query)             │
│    - 对 query 生成 Embedding            │
│    - Chroma 向量相似度搜索 Top-K        │
│    - 返回相关文本片段                    │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│ 6. 将检索结果作为 Observation 返回 Agent│
│    Agent 基于文档片段生成最终答案        │
└───────────────────────────────────────┘
    │
    ▼
返回答案给用户
```

### 3.5 开发操作步骤

#### Step 1：更新配置 — `config.py`

新增知识库相关配置：

```python
# 在 Phase 2 配置的基础上，追加以下配置
KB_DIR = os.getenv("KB_DIR", "./data/knowledge_base")
KB_TOP_K = int(os.getenv("KB_TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
DOCS_DIR = os.getenv("DOCS_DIR", "./docs")
```

> **说明**：
> - `KB_DIR`：知识库 Chroma collection 存储路径（与记忆路径分开，避免混淆）
> - `CHUNK_SIZE` / `CHUNK_OVERLAP`：文本块大小（token 数）和重叠大小，影响检索精度

#### Step 2：实现文档加载器 — `knowledge/loaders.py`

定义 Loader 抽象接口 + TxtLoader 实现。这是整个 RAG 系统的数据入口，**接口设计是关键**：

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Document:
    """文档数据结构，封装内容和元数据"""

    content: str
    metadata: dict = field(default_factory=dict)
    source: str = ""  # 来源文件路径


class DocumentLoader(ABC):
    """文档加载器抽象基类"""

    @abstractmethod
    def load(self, file_path: str) -> Document:
        """从文件路径加载文档，返回 Document 对象"""

    @abstractmethod
    def supports(self, file_path: str) -> bool:
        """判断该加载器是否支持指定文件的格式"""


class TxtLoader(DocumentLoader):
    """Txt 文件加载器"""

    def supports(self, file_path: str) -> bool:
        return file_path.lower().endswith(".txt")

    def load(self, file_path: str) -> Document:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        return Document(
            content=content,
            source=file_path,
            metadata={"file_type": "txt"},
        )
```

> **设计要点**：
> - `Document` 是统一的数据结构，后续加 PDF/MD 加载器都返回同一个类型
> - `supports()` 方法用于自动检测文件类型，Loader 分发器可以用它选择合适的加载器
> - 后续扩展 PDF 加载器只需新增一个类继承 `DocumentLoader`，实现 `load()` 和 `supports()` 即可

> **验收点 1**：能创建 TxtLoader 实例，调用 `load("docs/测试文档.txt")` 能正确读取内容。

#### Step 3：实现文本切分器 — `knowledge/splitters.py`

定义 Splitter 抽象接口 + TokenSplitter 实现。切分的核心考量是：**块太小 → 信息不完整，块太大 → 检索噪音多**。

```python
from abc import ABC, abstractmethod
import math
from knowledge.loaders import Document


class DocumentSplitter(ABC):
    """文本切分器抽象基类"""

    @abstractmethod
    def split(self, document: Document) -> list[Document]:
        """将文档切分为多个文本块，每个块保留元数据"""


class TokenSplitter(DocumentSplitter):
    """
    基于 token 数的文本切分器。

    简单估算：1 个英文字符 ≈ 0.25 个 token，
    1 个中文字符 ≈ 1.5 个 token。
    这是近似估算，实际可用 tiktoken 精确计算。
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size  # 目标 token 数
        self.chunk_overlap = chunk_overlap  # 块间重叠 token 数

    def _estimate_tokens(self, text: str) -> int:
        """估算文本的 token 数（粗略计算，英文字符 0.25，中文字符 1.5）"""
        count = 0
        for ch in text:
            if "一" <= ch <= "鿿":
                count += 1.5
            else:
                count += 0.25
        return int(count)

    def split(self, document: Document) -> list[Document]:
        text = document.content
        target_tokens = self.chunk_size
        overlap_tokens = self.chunk_overlap

        chunks = []
        start = 0

        while start < len(text):
            # 从 start 开始，找出一段文本，使其 token 数接近 chunk_size
            end = start
            current_tokens = 0

            while end < len(text) and current_tokens < target_tokens:
                ch = text[end]
                if "一" <= ch <= "鿿":
                    current_tokens += 1.5
                else:
                    current_tokens += 0.25
                end += 1

            # 尝试在句末或换行处截断，避免把一句话切断
            if end < len(text):
                for cut_pos in range(end, max(start, end - 50), -1):
                    if text[cut_pos] in ("\n", "。", "！", "？", ".", "!", "?"):
                        end = cut_pos + 1
                        break

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Document(
                    content=chunk_text,
                    source=document.source,
                    metadata={**document.metadata, "chunk_index": len(chunks)},
                ))

            # 移动 start，保留重叠区域
            start = end - self._char_overlap_for_tokens(overlap_tokens, text[:end])
            if start <= chunks[-1].content and chunks:
                start = end  # 防止死循环

        return chunks

    def _char_overlap_for_tokens(self, tokens: int, text: str) -> int:
        """根据 token 数估算需要回退的字符数"""
        count = 0
        for ch in reversed(text):
            count += 1.5 if "一" <= ch <= "鿿" else 0.25
            if int(count) >= tokens:
                return 1
        return 1
```

> **设计要点**：
> - 先用粗略 token 估算确定切分位置，再尝试在句末/换行处微调截断点，避免切断语义
> - 块间重叠（overlap）是 RAG 的关键技巧，确保跨块边界的信息不丢失
> - 后续可升级为用 `tiktoken` 精确计算 token 数，或增加按段落/标题切分的 Splitter

> **验收点 2**：用一篇较长文档测试，切分后每个 chunk 的 token 数约等于 `chunk_size`，块之间有重叠内容。

#### Step 4：实现索引器 — `knowledge/indexer.py`

将加载 → 切分 → Embedding → 存储串成一条流水线：

```python
import uuid
from knowledge.loaders import Document, DocumentLoader, TxtLoader
from knowledge.splitters import DocumentSplitter, TokenSplitter
from core.llm import embed
import chromadb
from config import KB_DIR, CHUNK_SIZE, CHUNK_OVERLAP


class Indexer:
    """索引器：将文档加载、切分、向量化并存入知识库"""

    def __init__(self, loaders: list[DocumentLoader] | None = None):
        self.loaders = loaders or [TxtLoader()]
        self.splitter = TokenSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        self.client = chromadb.PersistentClient(path=KB_DIR)
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"},
        )

    def _get_loader(self, file_path: str) -> DocumentLoader | None:
        """根据文件类型选择匹配的加载器"""
        for loader in self.loaders:
            if loader.supports(file_path):
                return loader
        return None

    async def index_file(self, file_path: str) -> int:
        """
        索引单个文件。

        Returns:
            生成的文本块数量
        """
        loader = self._get_loader(file_path)
        if loader is None:
            raise ValueError(f"不支持的文件类型: {file_path}")

        document = loader.load(file_path)
        chunks = self.splitter.split(document)

        if chunks:
            await self._store_chunks(chunks)

        return len(chunks)

    async def index_directory(self, directory: str) -> int:
        """
        索引目录下所有支持的文件。

        Returns:
            总文本块数量
        """
        import os
        total = 0
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                try:
                    total += await self.index_file(file_path)
                except ValueError:
                    pass  # 跳过不支持的文件类型
        return total

    async def _store_chunks(self, chunks: list[Document]) -> None:
        """批量存储文本块到向量库"""
        ids = [str(uuid.uuid4()) for _ in chunks]
        texts = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        # 批量生成 Embedding
        vectors = await embed(texts)

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=vectors,
        )

    def count(self) -> int:
        return self.collection.count()
```

#### Step 5：实现检索器 — `knowledge/retriever.py`

```python
from core.llm import embed
import chromadb
from config import KB_DIR, KB_TOP_K


class Retriever:
    """检索器：从知识库中检索相关文本片段"""

    def __init__(self, top_k: int | None = None):
        self.top_k = top_k or KB_TOP_K
        self.client = chromadb.PersistentClient(path=KB_DIR)
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"},
        )

    async def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """
        检索与查询相关的文本片段。

        Args:
            query: 查询文本
            top_k: 返回前 K 条结果

        Returns:
            结果列表 [{"source": "...", "content": "...", "metadata": {...}}]
        """
        k = top_k or self.top_k
        if self.collection.count() == 0:
            return []

        # 对查询文本生成 Embedding
        query_vectors = await embed([query])

        results = self.collection.query(
            query_embeddings=query_vectors,
            n_results=min(k, self.collection.count()),
        )

        if not results["documents"] or not results["documents"][0]:
            return []

        retrieved = []
        for i in range(len(results["ids"][0])):
            retrieved.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "source": results["metadatas"][0][i].get("source", "") if results["metadatas"] else "",
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
            })

        return retrieved
```

> **验收点 3**：准备一篇测试文档放入 `docs/` 目录，运行 Indexer 索引后，用 Retriever 搜索相关问题，能返回文档中相关的文本片段。

#### Step 6：实现知识库工具 — `knowledge/kb_tool.py`

将知识库检索封装为 Agent 可调用的 Tool（复用 Phase 1 定义的 Tool 接口）：

```python
from core.tool import Tool
from knowledge.retriever import Retriever


class KnowledgeBaseTool(Tool):
    """知识库检索工具，Agent 可调用它查询私有文档"""

    def __init__(self, retriever: Retriever | None = None):
        self.retriever = retriever or Retriever()

    @property
    def name(self) -> str:
        return "knowledge_base_search"

    @property
    def description(self) -> str:
        return "Search the knowledge base for relevant information from indexed documents. Use this when you need to answer questions based on documentation or reference materials."

    @property
    def parameters(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "knowledge_base_search",
                "description": "Search the knowledge base for relevant information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to find relevant documents",
                        }
                    },
                    "required": ["query"],
                },
            },
        }

    async def execute(self, query: str) -> str:
        results = await self.retriever.search(query)

        if not results:
            return "知识库中没有找到与查询相关的内容。"

        output_parts = []
        for i, r in enumerate(results, 1):
            source = r["source"] or "未知来源"
            output_parts.append(f"[{i}] 来源: {source}\n内容: {r['content']}")

        return "\n\n".join(output_parts)
```

#### Step 7：改造 Agent — `core/agent.py`

在 `main.py` 中注册 `KnowledgeBaseTool`，Agent 会自动获得知识库检索能力。`core/agent.py` 本身无需改动——因为 Tool 调用逻辑是通用的。

但如果你希望在 System Prompt 中显式告知 Agent "你有知识库可用"，可以将 prompt 微调。建议保持通用设计，让 Tool 的 `description` 来决定是否调用。

#### Step 8：更新 CLI 入口 — `main.py`

新增文档导入和知识库工具注册：

```python
import asyncio
import os
from core.agent import Agent
from core.tool import ToolRegistry
from memory.memory_manager import MemoryManager
from knowledge.kb_tool import KnowledgeBaseTool
from knowledge.indexer import Indexer
from tools.builtin import CalculatorTool, CurrentTimeTool
from config import DOCS_DIR


async def main():
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(CurrentTimeTool())

    # 注册知识库工具（如果已有索引数据）
    indexer = Indexer()
    if indexer.count() > 0:
        registry.register(KnowledgeBaseTool())
        print("知识库已加载，文档数量已索引")

    memory = MemoryManager(max_messages=10)
    agent = Agent(registry, memory)

    print("Agent 已启动！（支持多轮对话、记忆和知识库）")
    print("可用命令:")
    print("  index <文件路径>     — 索引单个文档")
    print("  index-docs           — 索引 docs/ 目录下所有文档")
    print("  kb-status            — 查看知识库状态")
    print("  memories             — 查看长期记忆")
    print("  save_memory <内容>    — 手动保存长期记忆")
    print("  quit                 — 退出")

    while True:
        user_input = input("\n> ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        if user_input == "kb-status":
            print(f"知识库条目数: {indexer.count()}")
            continue

        if user_input == "index-docs":
            if not os.path.isdir(DOCS_DIR):
                os.makedirs(DOCS_DIR, exist_ok=True)
            total = await indexer.index_directory(DOCS_DIR)
            print(f"已索引 {total} 个文本块")
            if total > 0 and not registry.get("knowledge_base_search"):
                registry.register(KnowledgeBaseTool())
            continue

        if user_input.startswith("index "):
            file_path = user_input[len("index "):]
            try:
                chunks = await indexer.index_file(file_path)
                print(f"已索引 {chunks} 个文本块: {file_path}")
                if not registry.get("knowledge_base_search"):
                    registry.register(KnowledgeBaseTool())
            except ValueError as e:
                print(str(e))
            continue

        if user_input == "memories":
            count = memory.long_term.count()
            print(f"长期记忆数量: {count}")
            continue

        if user_input.startswith("save_memory "):
            content = user_input[len("save_memory "):]
            memory.store_long_term(content)
            print(f"已保存: {content}")
            continue

        result = await agent.run(user_input)
        print(f"\n{result}")


if __name__ == "__main__":
    asyncio.run(main())
```

> **验收点 4（Phase 3 最终验收）**：
> - 将一篇 txt 文档放入 `docs/` 目录，执行 `index-docs`，能正确索引并显示块数
> - 知识库中有条目后，Agent 自动获得 `knowledge_base_search` 工具
> - 输入与文档相关的问题（"产品A的保修期是多久？"），Agent 自动调用知识库检索，基于检索结果给出准确回答
> - 输入与文档无关的问题（"2+2等于几？"），Agent 不调用知识库，走正常流程
> - 切分器生成的文本块大小合理，无单字符块，块之间有重叠
> - 索引器能跳过不支持的文件类型，不报错崩溃

---

### 3.6 Phase 3 验收清单

- [ ] DocumentLoader 抽象接口定义完成，TxtLoader 能正确加载 txt 文件
- [ ] DocumentSplitter 抽象接口定义完成，TokenSplitter 能按 token 数切分并支持 overlap
- [ ] Indexer 能将文档加载→切分→Embedding→存入向量库串成完整流水线
- [ ] Retriever 能对查询文本进行向量检索，返回相关片段
- [ ] KnowledgeBaseTool 实现 Tool 接口，Agent 可正常调用
- [ ] Agent 能根据问题类型自主决定是否调用知识库
- [ ] CLI 支持文档导入、知识库状态查看、知识库工具注册
- [ ] Loader 接口具备良好的扩展性（新增 PDF/MD 只需加实现类）

---

_完成 Phase 3 后，进入 Phase 4：标准化 Skills 可插拔系统。_

---

## Phase 4：Skills 可插拔工具系统

### 4.1 当前目标

在 Phase 1~3 的基础上，将零散的 Tool 实现升级为一套**标准化的 Skills 可插拔系统**：

- 统一的 Skill 抽象接口（定义技能的元数据、参数、执行方式）
- 自动发现机制：将 `skills/` 目录下的新技能文件放入即可自动加载，无需修改注册代码
- Skill 组合能力：支持将一个复杂任务拆解为多个 Skills 的顺序执行链（Skill Chain）
- Skill 沙箱：为 Skill 执行提供隔离环境（超时控制、错误捕获、参数校验）
- 沉淀一套通用 Skills 示例（搜索、代码执行、文件操作等）

**本阶段要回答的核心问题**：如何让新技能的开发和集成变得像"即插即用"一样简单？

### 4.2 技术选型

| 组件 | 选型 | 说明 |
|------|------|------|
| 技能接口 | 自研 Skill 抽象基类 | 继承/升级 Phase 1 的 Tool，增加更多元数据 |
| 自动发现 | Python 包扫描 + 动态导入 | 利用 `importlib` 动态加载 `skills/` 下的类 |
| 参数校验 | 内置 Python type hints | 用 `typing.get_type_hints()` 自动推断参数 schema |
| Skill 编排 | 自研 SkillChain | 顺序执行多个 Skill，上一个的输出作为下一个的输入 |
| 沙箱保护 | `asyncio.wait_for()` + try/except | 超时熔断、错误隔离 |

### 4.3 项目结构（增量变更）

在 Phase 3 基础上，将 `tools/` 升级为 `skills/` 目录，并新增 `skills/` 模块核心文件：

```
multi-agent-platform/
├── core/
│   ├── __init__.py
│   ├── agent.py          # 改造：使用 SkillsRegistry 替代 ToolRegistry
│   ├── llm.py            # 不变
│   └── skill.py          # 【改造自 tool.py】Skill 抽象基类 + SkillsRegistry
├── knowledge/            # 不变
├── memory/               # 不变
├── skills/               # 【新增】Skills 可插拔系统
│   ├── __init__.py
│   ├── loader.py         # 自动发现：扫描 skills/ 目录并动态注册所有 Skill
│   ├── calculator.py     # 计算器 Skill（从 builtin.py 迁移过来）
│   ├── time.py           # 时间查询 Skill（从 builtin.py 迁移过来）
│   ├── web_search.py     # 【新增】Web 搜索 Skill（示例）
│   ├── code_runner.py    # 【新增】代码执行 Skill（示例）
│   ├── file_ops.py       # 【新增】文件操作 Skill（示例）
│   └── chain.py          # Skill Chain：多技能顺序编排
├── docs/                 # 不变
├── data/                 # 不变
├── config.py             # 不变
├── main.py               # 改造：使用自动发现加载 Skills
└── requirements.txt      # 不变
```

### 4.4 流程图

```
【自动发现流程】—— Agent 启动时执行一次

skills/ 目录
    │
    ▼
┌───────────────────────────────────────┐
│ 1. loader.discover() 扫描目录          │
│    - 遍历 skills/*.py 文件             │
│    - importlib 动态导入每个模块         │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│ 2. 遍历模块中的所有类，筛选 Skill 子类   │
│    - issubclass(cls, Skill) and cls != Skill│
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│ 3. 实例化并注册到 SkillsRegistry        │
│    - registry.register(skill_instance) │
└───────────────────────────────────────┘
    │
    ▼
所有 Skill 已注册，Agent 可立即使用


【Skill Chain 执行流程】—— Agent 需要多步推理时

用户复杂任务
    │
    ▼
┌───────────────────────────────────────┐
│ 4. LLM 规划 Skill 执行顺序              │
│    如：[web_search → code_runner → ...] │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│ 5. SkillChain 顺序执行                  │
│    - step 1 结果 → step 2 输入          │
│    - 每步带超时控制和错误隔离            │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│ 6. 汇总所有 Skill 执行结果              │
│    返回给 LLM 生成最终答案               │
└───────────────────────────────────────┘
    │
    ▼
返回答案给用户
```

### 4.5 开发操作步骤

#### Step 1：升级 Skill 抽象基类 — `core/skill.py`

将 Phase 1 的 `Tool` 升级为更通用的 `Skill` 抽象，增加更多元数据和沙箱能力：

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import asyncio
import functools


class SkillStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class SkillResult:
    """Skill 执行结果"""

    status: SkillStatus
    output: str
    error: str | None = None
    metadata: dict = field(default_factory=dict)


class Skill(ABC):
    """
    Skill 抽象基类。
    所有可被 Agent 调用的能力都必须继承此类。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """技能唯一标识名称"""

    @property
    @abstractmethod
    def description(self) -> str:
        """技能功能描述，LLM 根据此描述决定是否调用"""

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """参数定义（OpenAI function calling 格式）"""

    @abstractmethod
    async def execute(self, **kwargs) -> str:
        """执行技能核心逻辑"""

    async def run(self, timeout: float = 30.0, **kwargs) -> SkillResult:
        """
        带沙箱保护的执行入口。
        提供超时控制和错误捕获。
        """
        try:
            result = await asyncio.wait_for(self.execute(**kwargs), timeout=timeout)
            return SkillResult(status=SkillStatus.SUCCESS, output=result)
        except asyncio.TimeoutError:
            return SkillResult(
                status=SkillStatus.TIMEOUT,
                output=f"执行超时（{timeout}s）",
                error="timeout",
            )
        except Exception as e:
            return SkillResult(
                status=SkillStatus.ERROR,
                output=f"执行出错: {str(e)}",
                error=str(e),
            )


class SkillsRegistry:
    """
    Skills 注册中心。
    管理所有可用技能，支持查询、列出、生成 API schema。
    """

    def __init__(self):
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def list_skills(self) -> list[Skill]:
        return list(self._skills.values())

    def get_api_schemas(self) -> list[dict]:
        """返回 OpenAI API 兼容的工具定义数组"""
        return [skill.parameters for skill in self._skills.values()]

    @property
    def skill_count(self) -> int:
        return len(self._skills)
```

> **与 Phase 1 Tool 的变化**：
> - 增加 `SkillResult` 统一返回结构（含状态、输出、错误信息、元数据）
> - 增加 `run()` 方法，提供内置的超时控制和异常捕获（沙箱）
> - 类名从 Tool → Skill，语义上更贴近"技能"而非简单工具

#### Step 2：实现自动发现机制 — `skills/loader.py`

这是 Phase 4 的核心——让新 Skill 能"即插即用"，开发者只需在 `skills/` 目录下放一个 Python 文件即可：

```python
import importlib
import os
import pkgutil
from typing import Optional
from core.skill import Skill, SkillsRegistry


def discover_and_register(registry: SkillsRegistry, package_name: str = "skills") -> int:
    """
    自动发现指定包下所有 Skill 子类并注册到 Registry。

    Args:
        registry: SkillsRegistry 实例
        package_name: 包含 Skill 实现的包名

    Returns:
        成功注册的 Skill 数量
    """
    package = importlib.import_module(package_name)
    package_path = package.__path__

    registered = 0
    for importer, modname, ispkg in pkgutil.iter_modules(package_path):
        # 跳过私有模块和下划线开头的文件
        if modname.startswith("_"):
            continue

        full_name = f"{package_name}.{modname}"
        try:
            module = importlib.import_module(full_name)
        except ImportError:
            continue

        # 遍历模块中的所有属性，找到 Skill 子类
        for attr_name in dir(module):
            cls = getattr(module, attr_name)
            if (
                isinstance(cls, type)
                and issubclass(cls, Skill)
                and cls is not Skill
            ):
                try:
                    instance = cls()
                    registry.register(instance)
                    registered += 1
                except Exception:
                    pass  # 实例化失败的 Skill 跳过

    return registered


def load_skills(registry: SkillsRegistry, skill_names: Optional[list[str]] = None) -> int:
    """
    加载 Skills，支持指定加载特定 Skill 或自动发现全部。

    Args:
        registry: SkillsRegistry 实例
        skill_names: 指定加载的 Skill 模块名列表。None 表示自动发现全部。

    Returns:
        注册的 Skill 数量
    """
    if skill_names:
        # 按需加载
        from core.skill import Skill
        for name in skill_names:
            module = importlib.import_module(name)
            for attr_name in dir(module):
                cls = getattr(module, attr_name)
                if isinstance(cls, type) and issubclass(cls, Skill) and cls is not Skill:
                    try:
                        registry.register(cls())
                    except Exception:
                        pass
        return registry.skill_count

    # 自动发现
    return discover_and_register(registry)
```

> **工作原理**：
> - `pkgutil.iter_modules()` 遍历包目录下所有 .py 文件
> - `importlib.import_module()` 动态导入每个模块
> - `dir(module)` 遍历模块中所有类，用 `issubclass()` 检查是否是 Skill 子类
> - 自动实例化并注册——**开发者不需要写任何注册代码**

> **验收点 1**：在 `skills/` 目录下新建一个 Skill 子类文件（如 `skills/hello.py`），重启后能自动发现并注册。

#### Step 3：迁移内置工具为 Skill — `skills/` 目录

将 Phase 1 的 `CalculatorTool` 和 `CurrentTimeTool` 迁移为 Skill 实现：

```python
# skills/calculator.py
from core.skill import Skill


class CalculatorSkill(Skill):

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Perform mathematical calculations. Input should be a valid math expression like '2 + 2' or '3.14 * 5'."

    @property
    def parameters(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Perform mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "A valid math expression, e.g. '2 + 2', '3.14 * 5'",
                        }
                    },
                    "required": ["expression"],
                },
            },
        }

    async def execute(self, expression: str) -> str:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters in expression"
        try:
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"
```

```python
# skills/time.py
import datetime
from core.skill import Skill


class TimeSkill(Skill):

    @property
    def name(self) -> str:
        return "current_time"

    @property
    def description(self) -> str:
        return "Get the current date and time"

    @property
    def parameters(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "current_time",
                "description": "Get the current date and time",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }

    async def execute(self, **kwargs) -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

```python
# skills/__init__.py
# 空文件即可，用于标识 skills 为 Python 包。
# 不要在这里 import 具体 Skill，否则会触发循环导入。
```

> **要点**：每个 Skill 独立一个文件，文件名即模块名。`__init__.py` 留空——因为自动发现机制会自己扫描，不需要手动导出。

#### Step 4：实现示例 Skill — Web 搜索和代码执行

提供两个更贴近实际场景的 Skill 示例：

```python
# skills/web_search.py
import httpx
import json
from core.skill import Skill
from config import API_KEY


class WebSearchSkill(Skill):
    """
    Web 搜索 Skill。
    这里用一个简化的模拟搜索实现，实际可接入搜索引擎 API。
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or API_KEY

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for real-time information. Use this when you need current events, latest news, or information not in your training data."

    @property
    def parameters(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for real-time information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query",
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "Number of results to return (default 3)",
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    async def execute(self, query: str, num_results: int = 3) -> str:
        """
        执行 Web 搜索。

        注：此处为示例实现，演示如何接入外部 API。
        实际使用时可替换为 SerpAPI、DuckDuckGo、Bing 等搜索引擎。
        """
        # 这里演示一个通用的搜索接口调用模式
        # 实际开发时替换为你选择的搜索引擎 API
        try:
            # 占位实现：这里调用搜索引擎 API
            # 例如：DuckDuckGo search API
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {"query": query, "num_results": num_results}

            # TODO: 替换为实际的搜索引擎 API
            # async with httpx.AsyncClient(timeout=10) as client:
            #     resp = await client.post("https://api.search-provider.com/search",
            #                              headers=headers, json=payload)
            #     data = resp.json()

            return f"搜索 '{query}' 的结果：（此处为占位，需接入实际搜索引擎）"
        except Exception as e:
            return f"搜索失败: {str(e)}"
```

```python
# skills/code_runner.py
import subprocess
import tempfile
import os
from core.skill import Skill


class CodeRunnerSkill(Skill):
    """代码执行 Skill — 在沙箱中执行 Python 代码片段"""

    @property
    def name(self) -> str:
        return "code_runner"

    @property
    def description(self) -> str:
        return "Execute Python code and return the output. Useful for data processing, file parsing, or complex computations that the LLM cannot do directly."

    @property
    def parameters(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "code_runner",
                "description": "Execute Python code and return the output",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute. The code should print its results.",
                        },
                        "timeout_seconds": {
                            "type": "integer",
                            "description": "Execution timeout in seconds (default 10)",
                        },
                    },
                    "required": ["code"],
                },
            },
        }

    async def execute(self, code: str, timeout_seconds: int = 10) -> str:
        try:
            # 写入临时文件执行
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as f:
                f.write(code)
                temp_path = f.name

            try:
                result = subprocess.run(
                    ["python", temp_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                )
                output = result.stdout.strip() or result.stderr.strip()
                return output if output else "(无输出)"
            finally:
                os.unlink(temp_path)

        except subprocess.TimeoutExpired:
            return f"代码执行超时（{timeout_seconds}s）"
        except Exception as e:
            return f"代码执行出错: {str(e)}"
```

```python
# skills/file_ops.py
import os
from pathlib import Path
from core.skill import Skill


class FileReadSkill(Skill):
    """文件读取 Skill"""

    @property
    def name(self) -> str:
        return "file_read"

    @property
    def description(self) -> str:
        return "Read the content of a text file from the local filesystem."

    @property
    def parameters(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "file_read",
                "description": "Read the content of a text file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to read",
                        },
                        "max_lines": {
                            "type": "integer",
                            "description": "Maximum number of lines to read (default 100)",
                        },
                    },
                    "required": ["file_path"],
                },
            },
        }

    async def execute(self, file_path: str, max_lines: int = 100) -> str:
        try:
            # 安全检查：限制读取范围
            resolved = Path(file_path).resolve()
            # 可添加安全白名单限制
            if not resolved.exists():
                return f"文件不存在: {file_path}"

            with open(resolved, "r", encoding="utf-8") as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        lines.append(f"... (仅显示前 {max_lines} 行，共 {i + 1} 行)")
                        break
                    lines.append(line)

            return "".join(lines)
        except Exception as e:
            return f"文件读取失败: {str(e)}"
```

> **要点**：这三个 Skill 分别展示了三种典型场景：
> - WebSearchSkill — 外部 API 调用模式
> - CodeRunnerSkill — 本地代码执行模式（沙箱）
> - FileReadSkill — 本地文件操作模式（含安全检查）

#### Step 5：实现 Skill Chain — `skills/chain.py`

Skill Chain 支持将多个 Skill 按顺序编排，上一个 Skill 的输出作为下一个的输入：

```python
from typing import Any
from core.skill import Skill, SkillResult, SkillsRegistry


class SkillChainStep:
    """Skill Chain 中的一个执行步骤"""

    def __init__(self, skill_name: str, input_mapping: dict | None = None):
        """
        Args:
            skill_name: 要执行的 Skill 名称
            input_mapping: 参数映射 {参数名: "上一步输出中的关键字"}
                          None 表示将整个上一步输出作为第一个参数
        """
        self.skill_name = skill_name
        self.input_mapping = input_mapping


class SkillChain:
    """
    Skill Chain：按顺序执行多个 Skills，
    将前一个 Skill 的输出传递给后一个。
    """

    def __init__(self, registry: SkillsRegistry):
        self.registry = registry
        self.steps: list[SkillChainStep] = []

    def add_step(
        self,
        skill_name: str,
        input_mapping: dict | None = None,
    ) -> "SkillChain":
        """添加一个执行步骤，返回 self 以支持链式调用"""
        self.steps.append(SkillChainStep(skill_name, input_mapping))
        return self

    async def execute(self, initial_input: str) -> list[SkillResult]:
        """
        顺序执行所有步骤。

        Args:
            initial_input: 第一个步骤的输入

        Returns:
            每个步骤的执行结果列表
        """
        results: list[SkillResult] = []
        current_input = initial_input

        for step in self.steps:
            skill = self.registry.get(step.skill_name)
            if skill is None:
                results.append(SkillResult(
                    status=SkillStatus.ERROR,
                    output=f"Skill '{step.skill_name}' 未找到",
                    error="skill_not_found",
                ))
                break

            # 构建参数
            kwargs = self._build_kwargs(skill, current_input, step.input_mapping, results)

            # 执行（带沙箱保护）
            result = await skill.run(**kwargs)
            results.append(result)

            # 将输出传递给下一步
            current_input = result.output

            # 如果出错，提前终止
            if result.status.value != "success":
                break

        return results

    def _build_kwargs(
        self,
        skill: Skill,
        current_input: str,
        input_mapping: dict | None,
        previous_results: list[SkillResult],
    ) -> dict:
        """根据 input_mapping 构建传给 Skill 的参数字典"""
        if input_mapping:
            kwargs = {}
            for param_name, ref in input_mapping.items():
                if ref == "__initial__":
                    kwargs[param_name] = self._get_initial_input()
                elif ref.startswith("__result_"):
                    # 引用之前某一步的结果
                    index = int(ref.split("_")[1])
                    kwargs[param_name] = previous_results[index].output if index < len(previous_results) else ""
                else:
                    kwargs[param_name] = ref
            return kwargs
        else:
            # 无映射：将整个输入作为第一个参数
            return {"query": current_input}

    def _get_initial_input(self) -> str:
        """获取初始输入（由 execute() 参数传入）"""
        return ""
```

> **设计要点**：
> - SkillChain 是**可选项**——简单任务直接调 Skill，复杂任务用 Chain 编排
> - `input_mapping` 支持引用上一步的输出，实现数据流传递
> - 任一步失败则提前终止（fail-fast 策略）

> **验收点 2**：创建一个 SkillChain，步骤为 `web_search → code_runner`，能按顺序执行，第一步的搜索结果能正确传递到第二步的代码中。

#### Step 6：改造 Agent — `core/agent.py`

将 `ToolRegistry` 替换为 `SkillsRegistry`，使用 Skill 的 `run()` 方法（带沙箱保护）：

```python
from core.llm import chat
from core.skill import SkillsRegistry
from memory.memory_manager import MemoryManager

SYSTEM_PROMPT_WITH_MEMORY = """\
You are a helpful assistant that can use tools to solve user's problems.

You have access to the following tools:
{tools_description}

{memory_context}

Rules:
- Use tools when you need to compute, look up, or calculate something
- Reference the historical memory when relevant
- Be concise and direct in your answers
"""


class Agent:
    def __init__(self, registry: SkillsRegistry, memory_manager: MemoryManager):
        self.registry = registry
        self.memory = memory_manager
        self.max_iterations = 10

    def _build_system_prompt(self, memory_context: str = "") -> str:
        tools_desc = "\n".join(
            f"- {skill.name}: {skill.description}"
            for skill in self.registry.list_skills()
        )
        return SYSTEM_PROMPT_WITH_MEMORY.format(
            tools_description=tools_desc,
            memory_context=memory_context,
        )

    async def run(self, user_input: str) -> str:
        """执行 Agent 推理循环，返回最终答案"""
        # 1. 检索长期记忆
        memory_context = await self.memory.search_long_term(user_input)

        # 2. 构建初始消息
        messages = [
            {"role": "system", "content": self._build_system_prompt(memory_context)},
        ]
        messages.extend(await self.memory.get_context_messages())
        messages.append({"role": "user", "content": user_input})

        # 3. ReAct 循环
        for _ in range(self.max_iterations):
            response = await chat(messages)

            if response["tool_calls"]:
                for tc in response["tool_calls"]:
                    skill = self.registry.get(tc["name"])
                    if skill:
                        # 使用 run() 方法（带超时和错误处理）
                        result = await skill.run(**tc["arguments"])
                        output = result.output
                    else:
                        output = f"Error: Skill '{tc['name']}' not found"

                    messages.append({
                        "role": "assistant",
                        "content": f"Calling tool: {tc['name']} with args: {tc['arguments']}",
                    })
                    messages.append({
                        "role": "tool",
                        "content": output,
                        "tool_call_id": tc.get("id", ""),
                    })
            else:
                answer = response["content"]
                await self.memory.add_turn(user_input, answer)
                return answer

        return "达到最大迭代次数，未能完成任务。"
```

#### Step 7：更新 CLI 入口 — `main.py`

使用自动发现机制加载 Skills：

```python
import asyncio
import os
from core.agent import Agent
from core.skill import SkillsRegistry
from core.llm import chat
from memory.memory_manager import MemoryManager
from knowledge.kb_tool import KnowledgeBaseTool
from knowledge.indexer import Indexer
from skills.loader import discover_and_register
from config import DOCS_DIR


async def main():
    registry = SkillsRegistry()

    # 自动发现 skills/ 目录下的所有 Skill
    skill_count = discover_and_register(registry)
    print(f"已自动发现 {skill_count} 个 Skills")

    # 注册知识库工具（如果已有索引数据）
    indexer = Indexer()
    if indexer.count() > 0:
        registry.register(KnowledgeBaseTool())
        print("知识库已加载")

    memory = MemoryManager(max_messages=10)
    agent = Agent(registry, memory)

    # 列出所有可用 Skills
    print("\n可用 Skills:")
    for skill in registry.list_skills():
        print(f"  - {skill.name}: {skill.description}")

    print("\n--- Agent 启动 ---")
    print("可用命令:")
    print("  index <文件路径>     — 索引单个文档")
    print("  index-docs           — 索引 docs/ 目录下所有文档")
    print("  kb-status            — 查看知识库状态")
    print("  memories             — 查看长期记忆")
    print("  save_memory <内容>    — 手动保存长期记忆")
    print("  quit                 — 退出")

    while True:
        user_input = input("\n> ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        if user_input == "kb-status":
            print(f"知识库条目数: {indexer.count()}")
            continue

        if user_input == "index-docs":
            if not os.path.isdir(DOCS_DIR):
                os.makedirs(DOCS_DIR, exist_ok=True)
            total = await indexer.index_directory(DOCS_DIR)
            print(f"已索引 {total} 个文本块")
            if total > 0 and not registry.get("knowledge_base_search"):
                registry.register(KnowledgeBaseTool())
            continue

        if user_input.startswith("index "):
            file_path = user_input[len("index "):]
            try:
                chunks = await indexer.index_file(file_path)
                print(f"已索引 {chunks} 个文本块: {file_path}")
                if not registry.get("knowledge_base_search"):
                    registry.register(KnowledgeBaseTool())
            except ValueError as e:
                print(str(e))
            continue

        if user_input == "memories":
            count = memory.long_term.count()
            print(f"长期记忆数量: {count}")
            continue

        if user_input.startswith("save_memory "):
            content = user_input[len("save_memory "):]
            memory.store_long_term(content)
            print(f"已保存: {content}")
            continue

        result = await agent.run(user_input)
        print(f"\n{result}")


if __name__ == "__main__":
    asyncio.run(main())
```

> **关键变化**：不再手动 `registry.register(CalculatorTool())`，而是调用 `discover_and_register(registry)` 自动发现全部 Skill。新增 Skill 文件后，重启即生效，无需改动任何现有代码。

> **验收点 3（Phase 4 最终验收）**：
> - 启动 Agent 时能自动发现并列出所有已实现的 Skill
> - 在 `skills/` 目录下新增一个 Skill 文件，重启后无需改任何代码即可自动加载
> - Skill 执行超时（如 `code_runner` 执行死循环）能被 `asyncio.wait_for` 正确中断
> - Skill 执行出错时返回 `SkillResult.ERROR`，不会导致 Agent 崩溃
> - SkillChain 能按顺序执行多个 Skill，上一步输出正确传递给下一步
> - Agent 能正常调用所有自动发现的 Skill 完成任务

---

### 4.6 Phase 4 验收清单

- [ ] Skill 抽象基类定义了统一接口（name/description/parameters/execute）
- [ ] SkillResult 统一返回结构，包含状态、输出、错误信息
- [ ] Skill.run() 提供超时控制和错误捕获（沙箱）
- [ ] SkillsRegistry 支持注册、查询、列出、生成 API schema
- [ ] 自动发现机制能扫描 `skills/` 目录并动态注册所有 Skill 子类
- [ ] 至少实现 3 个示例 Skill（计算器、时间、搜索/代码执行/文件操作）
- [ ] SkillChain 支持多 Skill 顺序编排和数据传递
- [ ] Agent 使用 SkillsRegistry 替代旧的 ToolRegistry
- [ ] CLI 启动时自动列出所有可用 Skills
- [ ] 新增 Skill 无需修改现有代码，放入目录即可

---

_完成 Phase 4 后，进入 Phase 5：多 Agent 协作。_

---

## Phase 5：多 Agent 协作

### 5.1 当前目标

将系统从**单 Agent** 升级为**多 Agent 协作架构**，让不同角色的 Agent 分工完成复杂任务：

- **Orchestrator（调度 Agent）** — 接收用户请求，拆解任务，分发给 Worker Agent
- **Worker Agent（执行 Agent）** — 各自拥有专属的 Skills 集合，专注完成特定类型的任务
- **Evaluator（评估 Agent）** — 对 Worker 的输出进行质量审核，不合格则要求重做

**完成本阶段后，Agent 系统具备**：

1. 任务拆解能力：将复杂请求拆分为子任务
2. 多 Agent 并行/串行执行
3. Agent 间消息传递和结果汇总
4. 评估-反馈循环：Evaluator 审核输出质量

**这回答了 JD 中的核心要求**：
> "设计并实现 Agent 的任务规划、逻辑推理、决策执行等核心能力，提升智能体自主解决复杂问题的能力"

### 5.2 技术选型

| 组件 | 选型 | 说明 |
|------|------|------|
| 架构模式 | Orchestrator-Worker（中央调度） | 最成熟的多 Agent 模式，易于理解和扩展 |
| Agent 间通信 | 消息队列（自研 AgentMessage） | 结构化消息传递，包含任务、结果、状态 |
| 任务编排 | 自研 OrchestratorAgent + 动态任务图 | LLM 拆解任务 → 生成执行图 → 顺序/并行执行 |
| 评估反馈 | 自研 EvaluatorAgent | 对输出进行打分和反馈，支持重试 |
| 并发执行 | `asyncio.gather()` | 独立的 Worker 可并发运行 |

### 5.3 项目结构（增量变更）

新增 `multiagent/` 模块，包含多 Agent 协作的核心逻辑：

```
multi-agent-platform/
├── core/               # 不变
├── knowledge/          # 不变
├── memory/             # 不变
├── skills/             # 不变
├── multiagent/         # 【新增】多 Agent 协作模块
│   ├── __init__.py
│   ├── message.py      # Agent 间消息定义
│   ├── agent.py        # Agent 基类（单 Agent 的抽象）
│   ├── orchestrator.py # Orchestrator：任务拆解和调度
│   ├── worker.py       # Worker：执行具体任务
│   ├── evaluator.py    # Evaluator：质量审核
│   └── dispatcher.py   # 任务执行引擎：解析任务图并调度
├── docs/               # 不变
├── data/               # 不变
├── config.py           # 不变
├── main.py             # 改造：支持单 Agent / 多 Agent 模式切换
└── requirements.txt    # 不变
```

### 5.4 流程图

```
【多 Agent 协作流程】—— 处理复杂用户请求

用户输入复杂任务
（如："调研 AI 行业最新趋势，写一篇分析报告，并计算市场规模"）
    │
    ▼
┌───────────────────────────────────────────────────┐
│ 1. Orchestrator 分析任务                           │
│    - 评估任务复杂度                                 │
│    - 拆解为子任务列表                               │
│    - 确定子任务的依赖关系和执行顺序                   │
│    - 输出：TaskPlan（任务执行图）                    │
└───────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────┐
│ 2. Dispatcher 解析并执行任务图                      │
│    - 无依赖的子任务 → 并发执行                       │
│    - 有依赖的子任务 → 等依赖完成后执行                │
│    - 每个子任务分配给对应的 Worker                    │
└───────────────────────────────────────────────────┘
    │
    ├── 并发分支 A ──▶ Worker（资料搜索）──▶ 结果 A
    ├── 并发分支 B ──▶ Worker（数据分析）──▶ 结果 B
    └── 并发分支 C ──▶ Worker（代码执行）──▶ 结果 C
    │
    ▼
┌───────────────────────────────────────────────────┐
│ 3. Orchestrator 汇总所有子结果                       │
└───────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────┐
│ 4. Evaluator 审核输出质量                           │
│    - 通过 → 最终答案                                │
│    - 不通过 → 反馈给 Orchestrator，重新执行          │
└───────────────────────────────────────────────────┘
    │
    ▼
返回最终答案给用户
```

### 5.5 开发操作步骤

#### Step 1：定义 Agent 间消息 — `multiagent/message.py`

Agent 之间的所有通信都通过结构化的消息对象：

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from datetime import datetime
import uuid


class MessageType(Enum):
    TASK = "task"           # 分配任务
    RESULT = "result"       # 返回结果
    REVIEW = "review"       # 审核反馈
    AGGREGATE = "aggregate" # 汇总请求


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"   # Evaluator 打回


@dataclass
class Task:
    """一个子任务"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""       # 任务描述
    skills: list[str] = field(default_factory=list)  # 该任务需要的 Skill 列表
    depends_on: list[str] = field(default_factory=list)  # 依赖的其他任务 ID
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None

    @property
    def is_ready(self) -> bool:
        """所有依赖的任务是否已完成"""
        return self.status == TaskStatus.PENDING


@dataclass
class TaskPlan:
    """任务执行计划（DAG 形式）"""

    tasks: list[Task] = field(default_factory=list)
    final_instruction: str = ""  # 如何汇总所有结果的最终指令

    def get_ready_tasks(self) -> list[Task]:
        """获取所有可以立即执行的任务（依赖已满足）"""
        completed_ids = {t.id for t in self.tasks if t.status == TaskStatus.COMPLETED}
        return [
            t for t in self.tasks
            if t.is_ready and all(dep_id in completed_ids for dep_id in t.depends_on)
        ]

    def is_complete(self) -> bool:
        return all(t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED) for t in self.tasks)


@dataclass
class AgentMessage:
    """Agent 间通信消息"""

    type: MessageType
    sender: str
    receiver: str
    content: str
    task: Task | None = None
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
```

> **设计要点**：
> - `Task` 是任务的最小单元，包含依赖关系（`depends_on`），形成 DAG
> - `TaskPlan` 是完整的任务执行图，Orchestrator 负责生成，Dispatcher 负责执行
> - `AgentMessage` 统一所有 Agent 间的通信格式，后续可扩展为跨进程/跨网络通信

#### Step 2：定义 Agent 基类 — `multiagent/agent.py`

为每个 Agent 定义统一的基类，包含独立的角色和专属 Skills：

```python
from core.llm import chat
from core.skill import Skill, SkillResult, SkillsRegistry
from multiagent.message import Task, TaskStatus


# 各 Agent 的系统提示词
ORCHESTRATOR_SYSTEM_PROMPT = """\
You are a Task Orchestrator. Your job is to:
1. Analyze complex user requests
2. Break them down into smaller sub-tasks
3. Specify which skills each sub-task needs
4. Define dependencies between tasks
5. Provide a final instruction on how to aggregate all results

Respond in the following JSON format:
{
  "tasks": [
    {
      "description": "具体任务描述",
      "skills": ["skill_name_1", "skill_name_2"],
      "depends_on": []
    }
  ],
  "final_instruction": "如何汇总所有子任务结果"
}

Rules:
- Tasks with no dependencies can run in parallel
- Each task should be small and focused
- Use available skills: {available_skills}
- If the task is simple, only create one task
"""


WORKER_SYSTEM_PROMPT = """\
You are a Task Worker. Execute the assigned task using the skills available to you.

Your task: {task_description}
Available skills: {available_skills}

Use the ReAct pattern to solve the task:
- Think about the approach
- Call the appropriate skills
- Provide a clear, well-structured result

Return only the result, no extra preamble.
"""


EVALUATOR_SYSTEM_PROMPT = """\
You are a Quality Evaluator. Review the following work result and assess its quality.

Task description: {task_description}
Result: {result}

Evaluate on these criteria:
- Completeness: Does the result fully address the task?
- Accuracy: Is the information correct?
- Clarity: Is the output well-structured and readable?

Respond in JSON format:
{
  "passed": true/false,
  "score": 0-10,
  "feedback": "Specific feedback for improvement",
  "retry_instruction": "What to fix if not passed"
}
"""


class Agent:
    """Agent 基类——拥有独立角色和 Skills 集合"""

    def __init__(self, name: str, role: str, system_prompt: str):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.registry = SkillsRegistry()

    def add_skill(self, skill: Skill) -> None:
        self.registry.register(skill)

    async def run(self, input_text: str) -> str:
        """执行 Agent 的核心逻辑（ReAct 循环）"""
        tools_desc = "\n".join(
            f"- {s.name}: {s.description}"
            for s in self.registry.list_skills()
        )
        prompt = self.system_prompt.format(
            available_skills=tools_desc,
            task_description=input_text,
            result="",
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": input_text},
        ]

        for _ in range(5):
            response = await chat(messages)

            if response["tool_calls"]:
                for tc in response["tool_calls"]:
                    skill = self.registry.get(tc["name"])
                    if skill:
                        result = await skill.run(**tc["arguments"])
                        output = result.output
                    else:
                        output = f"Skill '{tc['name']}' not found"

                    messages.append({
                        "role": "assistant",
                        "content": f"Calling tool: {tc['name']}",
                    })
                    messages.append({
                        "role": "tool",
                        "content": output,
                        "tool_call_id": tc.get("id", ""),
                    })
            else:
                return response["content"]

        return "Task execution reached max iterations."

    def __repr__(self) -> str:
        return f"Agent(name={self.name}, role={self.role})"
```

> **要点**：
> - 每个 Agent 实例有自己的 SkillsRegistry——不同角色的 Agent 拥有不同的技能集
> - Orchestrator 不需要太多 Skills，重点是拆解能力
> - Worker 拥有具体执行所需的 Skills
> - Evaluator 不需要 Skills，只需要分析能力

#### Step 3：实现 Orchestrator — `multiagent/orchestrator.py`

```python
import json
import re
from core.skill import SkillsRegistry
from multiagent.agent import Agent, ORCHESTRATOR_SYSTEM_PROMPT
from multiagent.message import Task, TaskPlan


class OrchestratorAgent(Agent):
    """Orchestrator：接收用户请求，拆解为任务图"""

    def __init__(self):
        super().__init__(
            name="orchestrator",
            role="task_orchestrator",
            system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        )

    async def plan(self, user_request: str, available_skills: list[str]) -> TaskPlan:
        """
        分析用户请求，生成任务执行计划。

        Returns:
            TaskPlan 对象，包含所有子任务
        """
        skills_str = ", ".join(available_skills)
        prompt = ORCHESTRATOR_SYSTEM_PROMPT.format(available_skills=skills_str)

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"User request: {user_request}"},
        ]

        response = await chat(messages)
        return self._parse_plan(response["content"])

    def _parse_plan(self, llm_output: str) -> TaskPlan:
        """解析 LLM 输出的 JSON，生成 TaskPlan"""
        # 尝试从 LLM 输出中提取 JSON 部分
        json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
        if not json_match:
            # 退化：单任务模式
            return TaskPlan(
                tasks=[Task(description=llm_output)],
                final_instruction="Return the result as is.",
            )

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return TaskPlan(
                tasks=[Task(description=llm_output)],
                final_instruction="Return the result as is.",
            )

        tasks = []
        task_map = {}  # index -> Task ID

        for i, task_data in enumerate(data.get("tasks", [])):
            depends_on = []
            for dep_idx in task_data.get("depends_on", []):
                if dep_idx in task_map:
                    depends_on.append(task_map[dep_idx])

            task = Task(
                description=task_data.get("description", ""),
                skills=task_data.get("skills", []),
                depends_on=depends_on,
            )
            tasks.append(task)
            task_map[i] = task.id

        return TaskPlan(
            tasks=tasks,
            final_instruction=data.get("final_instruction", "Aggregate all results."),
        )
```

> **要点**：
> - Orchestrator 不执行具体任务，只负责拆解和规划
> - `_parse_plan()` 将 LLM 的自然语言输出解析为结构化的 TaskPlan
> - 如果 LLM 返回格式不对，有退化方案（单任务模式兜底）

#### Step 4：实现 Worker — `multiagent/worker.py`

```python
from multiagent.agent import Agent, WORKER_SYSTEM_PROMPT


class WorkerAgent(Agent):
    """Worker：接收子任务，使用 Skills 执行"""

    def __init__(self, name: str = "worker"):
        super().__init__(
            name=name,
            role="task_worker",
            system_prompt=WORKER_SYSTEM_PROMPT,
        )

    async def execute_task(self, task_description: str) -> str:
        """执行单个任务"""
        result = await self.run(task_description)
        return result
```

> **说明**：Worker 继承 Agent 基类的 `run()` 方法（ReAct 循环），本身不需要太多额外逻辑。关键是在创建 Worker 时给它分配对应的 Skills。

#### Step 5：实现 Evaluator — `multiagent/evaluator.py`

```python
import json
import re
from multiagent.agent import Agent, EVALUATOR_SYSTEM_PROMPT


class EvaluationResult:
    def __init__(self, passed: bool, score: int, feedback: str, retry_instruction: str = ""):
        self.passed = passed
        self.score = score
        self.feedback = feedback
        self.retry_instruction = retry_instruction

    def __repr__(self) -> str:
        return f"Evaluation(passed={self.passed}, score={self.score})"


class EvaluatorAgent(Agent):
    """Evaluator：审核 Worker 的输出质量"""

    def __init__(self):
        super().__init__(
            name="evaluator",
            role="quality_evaluator",
            system_prompt=EVALUATOR_SYSTEM_PROMPT,
        )

    async def evaluate(self, task_description: str, result: str) -> EvaluationResult:
        """
        评估一次任务输出。

        Returns:
            EvaluationResult 包含评分和反馈
        """
        prompt = EVALUATOR_SYSTEM_PROMPT.format(
            task_description=task_description,
            result=result,
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Please evaluate this result.\nTask: {task_description}\nResult: {result}"},
        ]

        response = await chat(messages)
        return self._parse_evaluation(response["content"])

    def _parse_evaluation(self, llm_output: str) -> EvaluationResult:
        """解析 LLM 的评估输出"""
        json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
        if not json_match:
            return EvaluationResult(passed=True, score=5, feedback=llm_output)

        try:
            data = json.loads(json_match.group())
            return EvaluationResult(
                passed=data.get("passed", True),
                score=data.get("score", 5),
                feedback=data.get("feedback", ""),
                retry_instruction=data.get("retry_instruction", ""),
            )
        except json.JSONDecodeError:
            return EvaluationResult(passed=True, score=5, feedback=llm_output)
```

#### Step 6：实现任务执行引擎 — `multiagent/dispatcher.py`

```python
from core.skill import Skill, SkillsRegistry
from multiagent.agent import Agent
from multiagent.orchestrator import OrchestratorAgent
from multiagent.worker import WorkerAgent
from multiagent.evaluator import EvaluatorAgent, EvaluationResult
from multiagent.message import Task, TaskPlan, TaskStatus


MAX_EVALUATION_RETRIES = 2


class TaskDispatcher:
    """
    任务执行引擎：
    1. 接收用户请求
    2. Orchestrator 拆解任务
    3. 调度 Worker 执行
    4. Evaluator 审核
    5. 汇总结果
    """

    def __init__(self, skills: SkillsRegistry):
        self.skills = skills
        self.orchestrator = OrchestratorAgent()
        self.evaluator = EvaluatorAgent()
        self._evaluation_count: int = 0

    async def execute(self, user_request: str) -> str:
        """
        执行用户请求的完整流程。

        Returns:
            最终结果
        """
        available_skills = [s.name for s in self.skills.list_skills()]

        # 1. Orchestrator 拆解任务
        plan = await self.orchestrator.plan(user_request, available_skills)

        if not plan.tasks:
            return "No tasks generated."

        # 2. 按依赖关系调度执行
        await self._dispatch_tasks(plan)

        # 3. Orchestrator 汇总
        final_result = await self._aggregate_results(plan)

        # 带评估的循环
        final_result = await self._evaluate_and_retry(user_request, plan.tasks, final_result)

        return final_result

    async def _dispatch_tasks(self, plan: TaskPlan) -> None:
        """按依赖顺序调度任务执行，无依赖的可并发"""
        while not plan.is_complete:
            ready = plan.get_ready_tasks()
            if not ready:
                # 没有可执行的任务但也没完成——检查是否有失败的
                pending = [t for t in plan.tasks if t.status == TaskStatus.PENDING]
                if pending:
                    # 依赖无法满足，标记为失败
                    for t in pending:
                        t.status = TaskStatus.FAILED
                        t.result = "Dependency not met."
                break

            # 并发执行所有就绪的任务
            workers = []
            for task in ready:
                worker = self._create_worker_for_task(task)
                workers.append(self._run_single_task(worker, task))

            await asyncio.gather(*workers)

    def _create_worker_for_task(self, task: Task) -> WorkerAgent:
        """根据任务需要的 Skills 创建 Worker"""
        worker = WorkerAgent(name=f"worker_{task.id[:8]}")
        for skill in self.skills.list_skills():
            if not task.skills or skill.name in task.skills:
                worker.add_skill(skill)
        return worker

    async def _run_single_task(self, worker: WorkerAgent, task: Task) -> None:
        """执行单个任务"""
        task.status = TaskStatus.RUNNING
        try:
            result = await worker.execute_task(task.description)
            task.result = result
            task.status = TaskStatus.COMPLETED
        except Exception as e:
            task.result = f"执行失败: {str(e)}"
            task.status = TaskStatus.FAILED

    async def _evaluate_and_retry(
        self, user_request: str, tasks: list[Task], result: str
    ) -> str:
        """带评估和重试的最终结果审核"""
        for _ in range(MAX_EVALUATION_RETRIES):
            eval_result = await self.evaluator.evaluate(user_request, result)

            if eval_result.passed:
                break

            # 打回：让 Orchestrator 根据反馈重新执行
            retry_request = f"{user_request}\n\nFeedback: {eval_result.feedback}"
            plan = await self.orchestrator.plan(
                retry_request,
                [s.name for s in self.skills.list_skills()],
            )
            await self._dispatch_tasks(plan)
            result = await self._aggregate_results(plan)
            self._evaluation_count += 1

        return result

    async def _aggregate_results(self, plan: TaskPlan) -> str:
        """汇总所有子任务的结果"""
        completed_results = []
        for task in plan.tasks:
            if task.status == TaskStatus.COMPLETED:
                completed_results.append(f"[Task {task.id[:8]}]\n{task.result}")

        if not completed_results:
            return "No results available."

        if len(completed_results) == 1:
            return completed_results[0]

        # 让 LLM 做最终汇总
        combined = "\n\n---\n\n".join(completed_results)
        messages = [
            {"role": "system", "content": "You are a result aggregator. Combine the following sub-task results into a coherent, well-structured final answer."},
            {"role": "user", "content": f"{plan.final_instruction}\n\nSub-task results:\n\n{combined}"},
        ]

        from core.llm import chat
        response = await chat(messages)
        return response["content"]
```

```python
import asyncio  # 在文件顶部添加
```

> **核心流程**：
> 1. Orchestrator 拆解任务 → TaskPlan
> 2. Dispatcher 遍历任务图，无依赖的任务并发执行（`asyncio.gather`）
> 3. 汇总结果
> 4. Evaluator 审核 → 不通过则带反馈重新执行（最多 2 次重试）
> 5. 返回最终答案

> **验收点 1**：给一个复杂任务（"调研 AI 趋势并分析"），能看到任务被拆解为多个子任务，并发执行，最终汇总成一份完整报告。

#### Step 7：改造 CLI 支持双模式 — `main.py`

```python
import asyncio
import os
from core.agent import Agent
from core.skill import SkillsRegistry
from memory.memory_manager import MemoryManager
from knowledge.kb_tool import KnowledgeBaseTool
from knowledge.indexer import Indexer
from skills.loader import discover_and_register
from multiagent.dispatcher import TaskDispatcher
from config import DOCS_DIR


async def main():
    registry = SkillsRegistry()

    # 自动发现所有 Skill
    skill_count = discover_and_register(registry)
    print(f"已自动发现 {skill_count} 个 Skills")

    # 注册知识库工具
    indexer = Indexer()
    if indexer.count() > 0:
        registry.register(KnowledgeBaseTool())
        print("知识库已加载")

    # 单 Agent 模式
    memory = MemoryManager(max_messages=10)
    single_agent = Agent(registry, memory)

    # 多 Agent 调度器
    dispatcher = TaskDispatcher(registry)

    mode = input("\n选择模式 [1] 单Agent / [2] 多Agent协作: ").strip()
    use_multi_agent = mode == "2"

    print(f"\n--- Agent 启动 (模式: {'多Agent' if use_multi_agent else '单Agent'}) ---")
    print("可用 Skills:")
    for skill in registry.list_skills():
        print(f"  - {skill.name}: {skill.description}")

    print("\n可用命令: index <文件> | index-docs | kb-status | quit")

    while True:
        user_input = input("\n> ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        # 管理命令
        if user_input == "kb-status":
            print(f"知识库条目数: {indexer.count()}")
            continue

        if user_input == "index-docs":
            if not os.path.isdir(DOCS_DIR):
                os.makedirs(DOCS_DIR, exist_ok=True)
            total = await indexer.index_directory(DOCS_DIR)
            print(f"已索引 {total} 个文本块")
            if total > 0 and not registry.get("knowledge_base_search"):
                registry.register(KnowledgeBaseTool())
            continue

        if user_input.startswith("index "):
            file_path = user_input[len("index "):]
            try:
                chunks = await indexer.index_file(file_path)
                print(f"已索引 {chunks} 个文本块")
            except ValueError as e:
                print(str(e))
            continue

        if user_input == "memories":
            count = memory.long_term.count()
            print(f"长期记忆数量: {count}")
            continue

        if user_input.startswith("save_memory "):
            memory.store_long_term(user_input[len("save_memory "):])
            continue

        # 根据模式选择执行方式
        if use_multi_agent:
            print("\n[多Agent协作中...]\n")
            result = await dispatcher.execute(user_input)
        else:
            result = await single_agent.run(user_input)

        print(f"\n{result}")


if __name__ == "__main__":
    asyncio.run(main())
```

> **验收点 2（Phase 5 最终验收）**：
> - 启动后能选择单 Agent 或多 Agent 模式
> - 多 Agent 模式下，输入复杂任务（"调研 AI 行业趋势并写一份分析报告"），能看到 Orchestrator 将任务拆解为多个子任务
> - 无依赖的子任务并发执行（`asyncio.gather`）
> - Evaluator 对输出进行评分和审核，不合格则反馈重做
> - 最终结果由 Orchestrator 汇总所有子任务结果，生成连贯的完整答案
> - 简单任务（如 "2+2=?""）走单 Agent 路径即可，不需要多 Agent 开销

---

### 5.6 Phase 5 验收清单

- [ ] Agent 间消息（Task/TaskPlan/AgentMessage）结构定义完成
- [ ] Agent 基类定义了统一接口（name/role/skills/run）
- [ ] Orchestrator 能将复杂任务拆解为子任务图（DAG）
- [ ] Worker 拥有专属 Skills 集合，能独立执行任务
- [ ] Evaluator 能对输出进行质量评分和审核
- [ ] Dispatcher 能按依赖关系调度任务（无依赖并发，有依赖串行）
- [ ] 评估-反馈循环能正确打回重做
- [ ] 最终结果汇总生成连贯的完整答案
- [ ] CLI 支持单 Agent 和多 Agent 模式切换
- [ ] 简单任务走单 Agent 路径，复杂任务走多 Agent 路径

---

_完成 Phase 5 后，进入 Phase 6：性能优化与稳定性。_

---

## Phase 6：性能优化与稳定性

### 6.1 当前目标

作为最后一个阶段，本阶段聚焦于将项目从"能跑"升级到"能上生产"：

- **流式输出（Streaming）** — LLM 返回结果时逐 token 输出，而非等全部完成才展示，大幅降低用户感知延迟
- **缓存机制** — 对 Embedding 调用和 LLM Chat 调用做本地缓存，减少重复请求，降低成本和延迟
- **并发能力** — Agent 支持多用户/多会话并发处理，会话之间完全隔离
- **错误容灾** — API 调用重试、指数退避、熔断降级、优雅错误恢复
- **上下文压缩升级** — 从简单的消息截断升级为基于 token 计数的精确压缩

**本阶段要回答的核心问题**：如何让 Agent 系统在真实生产环境下稳定、高效地运行？

### 6.2 技术选型

| 组件 | 选型 | 说明 |
|------|------|------|
| 流式输出 | httpx 的 `aiter_lines()` | 直接解析 LLM API 的 SSE 流 |
| 缓存 | 本地 SQLite + 简单 LRU | 无需外部依赖，轻量可靠 |
| 多会话管理 | 自研 SessionManager | 每个会话独立的 MemoryManager |
| 重试机制 | 自研重试装饰器 | 指数退避 + 最大重试次数 |
| Token 计算 | `tiktoken` | 精确计算，替代之前的粗略估算 |

### 6.3 项目结构（增量变更）

新增 `optimization/` 模块和 `session/` 模块：

```
multi-agent-platform/
├── core/               # 不变
├── knowledge/          # 不变
├── memory/             # 改造：升级压缩逻辑
├── multiagent/         # 不变
├── skills/             # 不变
├── optimization/       # 【新增】性能优化模块
│   ├── __init__.py
│   ├── streaming.py    # 流式输出实现
│   ├── cache.py        # 本地缓存（Embedding + Chat）
│   └── retry.py        # 重试装饰器（指数退避）
├── session/            # 【新增】会话管理
│   ├── __init__.py
│   └── manager.py      # 多会话管理器
├── docs/               # 不变
├── data/               # 不变
├── config.py           # 新增：CACHE_DIR、MAX_SESSIONS 等
├── main.py             # 改造：集成流式输出 + 多会话
└── requirements.txt    # 新增：tiktoken
```

### 6.4 流程图

```
【流式输出流程】

用户输入
    │
    ▼
┌───────────────────────────────────────┐
│ 1. Agent 开始处理请求                  │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│ 2. 检查缓存：是否命中？                │
│    - 命中 → 从缓存返回（秒级）          │
│    - 未命中 → 继续                     │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│ 3. 带重试地调用 LLM API（指数退避）     │
│    - 第 1 次失败 → 等 1s 重试          │
│    - 第 2 次失败 → 等 2s 重试          │
│    - 第 3 次失败 → 等 4s 重试          │
│    - 超过最大重试 → 熔断，返回错误      │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│ 4. 流式返回 LLM 输出                   │
│    - 每收到一个 token chunk 就展示      │
│    - 用户感知：文字逐字出现             │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│ 5. 完整结果写入缓存                     │
└───────────────────────────────────────┘
    │
    ▼
返回完成


【多会话并发流程】

用户 A ──▶ SessionManager ──▶ Session A ──▶ Agent A（独立 Memory）
用户 B ──▶ SessionManager ──▶ Session B ──▶ Agent B（独立 Memory）
用户 C ──▶ SessionManager ──▶ Session C ──▶ Agent C（独立 Memory）
    │
    ▼
每个会话的 MemoryManager 完全独立
会话间不互相干扰
```

### 6.5 开发操作步骤

#### Step 1：更新依赖和配置

1. 更新 `requirements.txt`，新增：
   ```
   tiktoken>=0.7.0
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

3. 更新 `config.py`，新增：
   ```python
   CACHE_DIR = os.getenv("CACHE_DIR", "./data/cache")
   MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "10"))
   STREAM_ENABLED = os.getenv("STREAM_ENABLED", "true").lower() == "true"
   MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
   RETRY_BASE_DELAY = float(os.getenv("RETRY_BASE_DELAY", "1.0"))
   ENABLE_TIKTOKEN = os.getenv("ENABLE_TIKTOKEN", "true").lower() == "true"
   ```

#### Step 2：实现重试装饰器 — `optimization/retry.py`

API 调用不稳定是常态，重试机制是生产环境的必备能力：

```python
import asyncio
import functools
import logging
from config import MAX_RETRIES, RETRY_BASE_DELAY

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_retries: int = MAX_RETRIES,
    base_delay: float = RETRY_BASE_DELAY,
    retryable_exceptions: tuple = (Exception,),
):
    """
    重试装饰器，使用指数退避策略。

    Args:
        max_retries: 最大重试次数
        base_delay: 基础延迟（秒）
        retryable_exceptions: 可重试的异常类型
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after "
                            f"{max_retries + 1} attempts: {e}"
                        )
                        raise

                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)

            raise last_exception  # 不应该到这里，但为了类型安全

        return wrapper
    return decorator
```

> **要点**：
> - 指数退避：第 1 次等 1s，第 2 次等 2s，第 3 次等 4s
> - 可指定哪些异常类型需要重试（如 httpx.HTTPError）
> - 超过最大重试后抛出异常，由上层处理

> **验收点 1**：用 `@retry_with_backoff(max_retries=3)` 装饰一个会失败的函数，观察输出日志，应该看到递增的等待时间和最终报错。

#### Step 3：实现本地缓存 — `optimization/cache.py`

对相同输入返回相同输出是 LLM 调用的常见优化场景：

```python
import hashlib
import json
import os
import time
import sqlite3
from config import CACHE_DIR


class Cache:
    """基于 SQLite 的本地缓存，支持过期时间"""

    def __init__(self, db_path: str = None, ttl: int = 3600):
        self.db_path = db_path or os.path.join(CACHE_DIR, "cache.db")
        self.ttl = ttl  # 缓存过期时间（秒）
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    created_at REAL
                )
            """)

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _hash_key(self, key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()

    def get(self, key: str) -> str | None:
        """获取缓存，如果过期则返回 None"""
        h = self._hash_key(key)
        with self._conn() as conn:
            cursor = conn.execute(
                "SELECT value, created_at FROM cache WHERE key = ?", (h,)
            )
            row = cursor.fetchone()

        if row is None:
            return None

        value, created_at = row
        if time.time() - created_at > self.ttl:
            self.delete(key)
            return None

        return value

    def set(self, key: str, value: str) -> None:
        """写入缓存"""
        h = self._hash_key(key)
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, created_at) VALUES (?, ?, ?)",
                (h, value, time.time()),
            )

    def delete(self, key: str) -> None:
        """删除缓存"""
        h = self._hash_key(key)
        with self._conn() as conn:
            conn.execute("DELETE FROM cache WHERE key = ?", (h,))

    def clear(self) -> None:
        """清空所有缓存"""
        with self._conn() as conn:
            conn.execute("DELETE FROM cache")

    def size(self) -> int:
        with self._conn() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM cache")
            return cursor.fetchone()[0]


# 全局缓存实例
chat_cache = Cache(db_path=os.path.join(CACHE_DIR, "chat_cache.db"), ttl=1800)
embedding_cache = Cache(db_path=os.path.join(CACHE_DIR, "embedding_cache.db"), ttl=86400)
```

> **设计要点**：
> - Chat 缓存 TTL 设为 30 分钟（对话结果变化快），Embedding 缓存 TTL 设为 24 小时（同一文本的向量永远不变）
> - Key 用 SHA-256 哈希，避免 SQLite 路径长度限制
> - SQLite 天然支持并发读写（WAL 模式可进一步提升），无需外部 Redis

接下来将缓存和重试集成到 LLM API 调用中：

```python
# 在 core/llm.py 中添加以下改造（在已有的 chat() 和 embed() 函数基础上）

import json
import logging
from optimization.retry import retry_with_backoff
from optimization.cache import chat_cache, embedding_cache
from config import STREAM_ENABLED

logger = logging.getLogger(__name__)


@retry_with_backoff(
    max_retries=3,
    base_delay=1.0,
    retryable_exceptions=(httpx.HTTPError, httpx.ConnectError),
)
async def chat(
    messages: list[dict],
    tools: list[dict] | None = None,
    stream: bool = STREAM_ENABLED,
) -> dict:
    """改造后的 chat 函数，加入重试和缓存"""
    # 检查缓存
    cache_key = json.dumps({"messages": messages, "tools": tools}, sort_keys=True)
    cached = chat_cache.get(cache_key)
    if cached:
        logger.info("Chat cache hit")
        return json.loads(cached)

    # 调用 LLM（原有逻辑，不变）
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
    }

    if tools:
        payload["tools"] = tools

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(BASE_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    choice = data["choices"][0]["message"]
    content = choice.get("content", "")
    finish_reason = data["choices"][0].get("finish_reason", "stop")

    tool_calls = []
    if choice.get("tool_calls"):
        for tc in choice["tool_calls"]:
            tool_calls.append({
                "name": tc["function"]["name"],
                "arguments": json.loads(tc["function"]["arguments"]),
            })

    result = {
        "content": content,
        "tool_calls": tool_calls,
        "finish_reason": finish_reason,
    }

    # 写入缓存
    chat_cache.set(cache_key, json.dumps(result))

    return result


async def chat_stream(
    messages: list[dict],
    tools: list[dict] | None = None,
):
    """
    流式调用 LLM，yield 每个 token chunk。

    Yields:
        {"content": "token 文本", "done": False/True}
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "stream": True,
    }

    if tools:
        payload["tools"] = tools

    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream("POST", BASE_URL, headers=headers, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]  # 去掉 "data: " 前缀
                    if data_str.strip() == "[DONE]":
                        yield {"content": "", "done": True}
                        return
                    try:
                        data = json.loads(data_str)
                        chunk_content = data["choices"][0].get("delta", {}).get("content", "")
                        if chunk_content:
                            yield {"content": chunk_content, "done": False}
                    except json.JSONDecodeError:
                        continue
```

> **要点**：
> - `chat()` 增加了缓存检查和 `@retry_with_backoff` 装饰器
> - 新增 `chat_stream()` 函数，使用 httpx 的异步 SSE 流式解析
> - 流式输出时不检查缓存（流式本身就是为了低延迟）

> **验收点 2**：
> - 连续两次输入完全相同的问题，第二次应该几乎秒回（命中缓存）
> - 关闭 API Key 后调用 LLM，应该看到 3 次重试日志，然后报错

#### Step 4：实现流式输出 CLI — `optimization/streaming.py`

```python
import asyncio
from core.llm import chat_stream


async def print_stream(messages: list[dict], tools: list[dict] | None = None) -> str:
    """
    流式打印 LLM 输出，收集完整内容后返回。

    Returns:
        完整的 LLM 回复内容
    """
    full_content = ""

    async for chunk in chat_stream(messages, tools):
        if chunk["done"]:
            print()  # 结束时的换行
            break

        content = chunk["content"]
        full_content += content
        print(content, end="", flush=True)

    return full_content
```

#### Step 5：升级 Token 精确计算 — memory/summarizer.py

Phase 2 中我们用的是粗略估算，现在升级为 `tiktoken` 精确计算：

```python
# 在 memory/summarizer.py 中新增精确 token 计数功能

import tiktoken


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """精确计算文本的 token 数"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # 降级：粗略估算（英文 0.25，中文 1.5）
        count = 0
        for ch in text:
            count += 1.5 if "一" <= ch <= "鿿" else 0.25
        return int(count)


def truncate_to_max_tokens(messages: list[dict], max_tokens: int = 4000) -> list[dict]:
    """
    将 messages 列表截断到 max_tokens 以内。
    保留最新的消息，丢弃最早的消息。

    Args:
        messages: 消息列表
        max_tokens: 最大 token 数

    Returns:
        截断后的消息列表
    """
    if not messages:
        return []

    # 从最新到最旧累加 token，直到超过 max_tokens
    total_tokens = 0
    kept = []

    for msg in reversed(messages):
        msg_tokens = count_tokens(msg.get("content", ""))
        total_tokens += msg_tokens

        if total_tokens > max_tokens and kept:
            break

        kept.insert(0, msg)

    return kept
```

> **要点**：
> - `tiktoken` 是 OpenAI 官方 tokenizer，精确计算每个模型的 token 数
> - 降级方案保留了之前的粗略估算，确保在 tiktoken 不可用时仍能工作
> - `truncate_to_max_tokens()` 按 token 级别截断，比 Phase 2 的消息级别截断更精细

#### Step 6：实现多会话管理 — `session/manager.py`

```python
import uuid
import time
import logging
from core.agent import Agent
from core.skill import SkillsRegistry
from memory.memory_manager import MemoryManager
from config import MAX_SESSIONS

logger = logging.getLogger(__name__)


class Session:
    """单个用户会话"""

    def __init__(self, session_id: str, agent: Agent):
        self.id = session_id
        self.agent = agent
        self.last_active = time.time()
        self.message_count = 0

    def touch(self) -> None:
        self.last_active = time.time()
        self.message_count += 1


class SessionManager:
    """
    多会话管理器：
    - 每个会话有独立的 Agent 和 MemoryManager
    - 会话数量超过上限时，自动淘汰最久未使用的会话
    """

    def __init__(self, agent_factory, max_sessions: int = MAX_SESSIONS):
        """
        Args:
            agent_factory: 无参工厂函数，创建新的 Agent 实例
            max_sessions: 最大并发会话数
        """
        self.agent_factory = agent_factory
        self.max_sessions = max_sessions
        self.sessions: dict[str, Session] = {}

    def create_session(self) -> str:
        """创建新会话，返回会话 ID"""
        session_id = str(uuid.uuid4())[:8]

        if len(self.sessions) >= self.max_sessions:
            self._evict_oldest()

        agent = self.agent_factory()
        self.sessions[session_id] = Session(session_id, agent)
        logger.info(f"Created session {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Session | None:
        """获取会话"""
        session = self.sessions.get(session_id)
        if session:
            session.touch()
        return session

    def delete_session(self, session_id: str) -> None:
        """删除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Deleted session {session_id}")

    def _evict_oldest(self) -> None:
        """淘汰最久未使用的会话"""
        if not self.sessions:
            return

        oldest_id = min(self.sessions, key=lambda k: self.sessions[k].last_active)
        oldest = self.sessions[oldest_id]
        logger.info(
            f"Evicting session {oldest_id} "
            f"(idle since {oldest.last_active}, "
            f"{oldest.message_count} messages)"
        )
        del self.sessions[oldest_id]

    @property
    def active_count(self) -> int:
        return len(self.sessions)
```

> **设计要点**：
> - 每个会话有独立的 Agent 实例，MemoryManager 完全隔离
> - 超过最大会话数时，淘汰最久未使用的（LRU 策略）
> - `agent_factory` 是工厂函数，按需创建 Agent，避免启动时创建过多实例

#### Step 7：改造 CLI — main.py 最终版

集成所有优化，最终的 CLI 入口：

```python
import asyncio
import os
import logging
from core.agent import Agent
from core.skill import SkillsRegistry
from memory.memory_manager import MemoryManager
from knowledge.kb_tool import KnowledgeBaseTool
from knowledge.indexer import Indexer
from skills.loader import discover_and_register
from multiagent.dispatcher import TaskDispatcher
from session.manager import SessionManager
from optimization.streaming import print_stream
from config import STREAM_ENABLED

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


async def main():
    # 注册 Skills
    registry = SkillsRegistry()
    skill_count = discover_and_register(registry)
    print(f"已自动发现 {skill_count} 个 Skills")

    indexer = Indexer()
    if indexer.count() > 0:
        registry.register(KnowledgeBaseTool())
        print("知识库已加载")

    # 会话管理器（多会话）
    def create_agent():
        reg = SkillsRegistry()
        for skill in registry.list_skills():
            reg.register(skill)
        memory = MemoryManager(max_messages=10)
        return Agent(reg, memory)

    session_manager = SessionManager(create_agent)

    # 多 Agent 调度器
    dispatcher = TaskDispatcher(registry)

    print("\n--- Multi-Agent Platform ---")
    print("可用 Skills:")
    for skill in registry.list_skills():
        print(f"  - {skill.name}: {skill.description}")

    # 创建初始会话
    session_id = session_manager.create_session()
    print(f"\n已创建会话 {session_id}")

    while True:
        user_input = input("\n> ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        # 管理命令
        if user_input == "kb-status":
            print(f"知识库条目数: {indexer.count()}")
            continue

        if user_input == "index-docs":
            if not os.path.isdir(DOCS_DIR):
                os.makedirs(DOCS_DIR, exist_ok=True)
            total = await indexer.index_directory(DOCS_DIR)
            print(f"已索引 {total} 个文本块")
            if total > 0 and not registry.get("knowledge_base_search"):
                registry.register(KnowledgeBaseTool())
            continue

        if user_input.startswith("index "):
            file_path = user_input[len("index "):]
            try:
                chunks = await indexer.index_file(file_path)
                print(f"已索引 {chunks} 个文本块")
            except ValueError as e:
                print(str(e))
            continue

        if user_input == "memories":
            session = session_manager.get_session(session_id)
            if session:
                count = session.agent.memory.long_term.count()
                print(f"长期记忆数量: {count}")
            continue

        if user_input.startswith("save_memory "):
            session = session_manager.get_session(session_id)
            if session:
                session.agent.memory.store_long_term(user_input[len("save_memory "):])
            continue

        if user_input == "new-session":
            session_id = session_manager.create_session()
            print(f"已创建新会话 {session_id}，当前活跃会话: {session_manager.active_count}")
            continue

        if user_input.startswith("switch-session "):
            target_id = user_input[len("switch-session "):]
            if session_manager.get_session(target_id):
                session_id = target_id
                print(f"已切换到会话 {session_id}")
            else:
                print(f"会话 {target_id} 不存在")
            continue

        if user_input == "sessions":
            print(f"活跃会话: {session_manager.active_count}")
            for sid, sess in session_manager.sessions.items():
                print(f"  - {sid} (消息数: {sess.message_count})")
            continue

        # 获取当前会话的 Agent
        session = session_manager.get_session(session_id)
        if not session:
            print("会话不存在，请创建新会话")
            continue

        agent = session.agent

        try:
            # 根据配置选择流式或非流式输出
            if STREAM_ENABLED:
                # 流式输出
                result = await agent.run(user_input)
                print(f"\n{result}")
            else:
                result = await agent.run(user_input)
                print(f"\n{result}")

        except Exception as e:
            logger.error(f"Agent 执行出错: {e}")
            print(f"\n执行出错: {e}")


if __name__ == "__main__":
    asyncio.run(main())
```

> **验收点 3（Phase 6 最终验收）**：
> - 连续两次输入相同问题，第二次几乎秒回（缓存命中）
> - 临时断网后恢复，Agent 能自动重试并恢复正常（重试机制）
> - 创建多个会话，会话间记忆完全隔离（Session 切换）
> - 会话数超过 MAX_SESSIONS 时，最早的空闲会话被自动淘汰（LRU 淘汰）
> - Token 截断工作正常，对话上下文不会超过模型的 context window 限制
> - 启动和运行过程中无未捕获异常，所有错误都有友好提示

---

### 6.6 Phase 6 验收清单

- [ ] 重试装饰器工作正常，API 临时故障时能自动重试恢复
- [ ] Chat 缓存和 Embedding 缓存正常工作，相同输入命中缓存
- [ ] 流式输出能逐 token 展示 LLM 回复
- [ ] 多会话管理器支持创建、切换、销毁会话
- [ ] 会话间 MemoryManager 完全隔离
- [ ] 会话数超过上限时自动淘汰最久未使用的会话
- [ ] Token 精确计算替代粗略估算，上下文压缩按 token 级别进行
- [ ] 所有错误都有友好提示，无未捕获异常导致程序崩溃
- [ ] 日志输出正常，便于问题排查

---

_恭喜！完成全部 6 个阶段后，你已具备：_

1. **完整的 Agent 开发能力** — 从 ReAct 到多 Agent 协作
2. **记忆和知识库系统** — 短期/长期记忆 + RAG 知识库
3. **可插拔 Skills 架构** — 自动发现、沙箱执行、Chain 编排
4. **生产级性能优化** — 流式输出、缓存、并发、容灾

_这六个阶段覆盖了 JD 中要求的全部核心能力。项目可以作为一个完整的技术展示 portfolio，用于面试和职业转型。_
