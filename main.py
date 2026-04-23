# 入口（交互式 CLI）
# 程序主入口，提供命令行交互界面
import asyncio
from core.agent import Agent
from core.tool import ToolRegistry
from tools.builtin import CalculatorTool, CurrentTimeTool
from tools.log import get_logger

logger = get_logger(__name__)

async def main():
    logger.info("程序启动")
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(CurrentTimeTool())

    agent = Agent(registry)

    print("Agent 已启动！输入你的问题（输入 'quit' 退出）")
    while True:
        user_input = input("\n> ").strip()
        logger.debug(f"用户输入: {user_input}")
        if user_input.lower() in ("quit", "exit", "q"):
            logger.info("程序退出")
            break
        if not user_input:
            continue

        result = await agent.run(user_input)
        logger.debug(f"Agent 输出: {result}")
        print(f"\n{result}")

if __name__ == "__main__":
    asyncio.run(main())