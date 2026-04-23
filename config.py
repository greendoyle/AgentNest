# 配置管理
# 管理 API 密钥、模型参数等配置项
import os
from dotenv import load_dotenv

load_dotenv()

# 使用的 LLM 提供商（如 openai / anthropic 等）
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
# 对应提供商的 API 密钥
API_KEY = os.getenv("API_KEY", "")
# 要使用的模型名称
MODEL = os.getenv("MODEL", "gpt-4o-mini")
# API 请求的基础地址
BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1/chat/completions")
# 单次任务最大迭代次数
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))

# 测试config文件是否有效
# print("LLM_PROVIDER:", LLM_PROVIDER)
# print("API_KEY:", API_KEY)
# print("MODEL:", MODEL)
# print("BASE_URL:", BASE_URL)
# print("MAX_ITERATIONS:", MAX_ITERATIONS)