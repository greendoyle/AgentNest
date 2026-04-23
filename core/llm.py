# LLM API 封装（直接 HTTP 调用）
# 提供与大语言模型的通信接口
import json
import httpx
from config import API_KEY, BASE_URL, MODEL
from tools.log import get_logger

logger = get_logger(__name__)

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

    async with httpx.AsyncClient(timeout=180) as client:
        logger.debug(f"LLM 请求: {payload}")
        response = await client.post(BASE_URL, headers=headers, json=payload)
        data = response.json()
        if response.status_code != 200:
            logger.error(f"请求失败: {data.get('error').get('message')}")
            return {
                "content": "请求失败:" + data.get("error").get("message"),
                "tool_calls": [],
                "finish_reason": "error",
            }

    logger.debug(f"LLM 回复: {data}")
    choice = data["choices"][0]["message"]
    content = choice.get("content", "")
    finish_reason = data["choices"][0].get("finish_reason", "stop")

    tool_calls = []
    if choice.get("tool_calls"):
        for tc in choice["tool_calls"]:
            tool_calls.append({
                "name": tc["function"]["name"],
                "arguments": json.loads(tc["function"]["arguments"]),
                "id": tc.get("id", ""),
            })

    return {
        "content": content,
        "tool_calls": tool_calls,
        "finish_reason": finish_reason,
    }