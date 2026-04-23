# 内置工具（计算器、时间查询、天气等）
# 提供开箱即用的工具实现
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