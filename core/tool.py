# Tool 基类 + 注册中心
# 定义工具接口和工具管理
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
