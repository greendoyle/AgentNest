# ReAct Agent 核心循环
# 负责 Agent 的推理-行动-观察循环
from core.llm import chat
from core.tool import ToolRegistry
from tools.log import get_logger

logger = get_logger(__name__)
SYSTEM_PROMPT = """
You are a helpful assistant that can use tools to solve user's problems.

You must follow the ReAct (Reasoning + Acting) pattern:
1. Think about what you need to do (Thought:)
2. Decide which tool to use (Action:)
3. Observe the tool's result (Observation:)
4. Continue until you can give a final answer (Final Answer:)

Rules:
- Always start with a Thought explaining your reasoning
- Use tools when you need to compute, look up, or calculate something
- Only call one tool at a time
- When you have enough information, give the Final Answer directly
"""

class Agent:
    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry
        self.max_iterations = 10

    async def run(self, user_input: str) -> str:
        """执行 Agent 推理循环，返回最终答案"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]

        tools = self.tool_registry.get_openai_tools_schema()

        for iteration in range(self.max_iterations):
            logger.debug(f"消息轮次： {iteration + 1}")
            response = await chat(messages, tools)

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
                        "tool_call_id": tc["id"],
                    })
            else:
                # LLM 直接回答，循环结束
                return response["content"]

        return "达到最大迭代次数，未能完成任务。"