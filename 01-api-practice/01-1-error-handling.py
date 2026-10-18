# ==================================================
# 文件：01-1-error-handling.py
# 定位：基础API异常重试
# 作用：理解网络错误、限流、API错误的分级处理，以及重试机制
# ==================================================

# ---------- 导入依赖库 ----------
from openai import OpenAI, APIError, APIConnectionError, RateLimitError
# OpenAI：大模型客户端类，DeepSeek兼容OpenAI接口格式
# APIError：通用API业务错误（比如密钥错误、参数不合法）
# APIConnectionError：网络连接异常（断网、域名解析失败、超时）
# RateLimitError：调用频率超限（触发平台限流规则）

from dotenv import load_dotenv  # 加载 .env 配置文件，读取API密钥等环境变量
import os                       # 读取系统环境变量
import time                     # 重试等待延时，给服务器恢复时间


# 加载 .env 配置（全局只执行一次）
load_dotenv()

# 初始化大模型客户端，建立与DeepSeek的连接
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),    # 从环境变量读取API密钥
    base_url=os.getenv("DEEPSEEK_BASE_URL")   # DeepSeek接口地址
)


# ====================== 带重试的AI调用函数 ======================
def ask_ai(prompt: str, retries: int = 3) -> str | None:
    """
    带自动重试的大模型单轮问答
    :param prompt: 用户提问内容
    :param retries: 最大重试次数，默认3次
    :return: 成功返回回答文本；失败返回None
    """
    # 重试循环：最多尝试 retries 次
    for attempt in range(retries):
        try:
            # 调用大模型聊天接口
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            # 调用成功，直接返回回答内容，结束函数
            return response.choices[0].message.content

        # ---------- 分级异常处理 ----------
        except APIConnectionError:
            # 网络连接错误：通常是临时波动，重试大概率能恢复
            print(f"网络错误，第{attempt + 1}次重试...")
            time.sleep(2)  # 等待2秒后重试，避免频繁请求

        except RateLimitError:
            # 触发限流：单位时间调用次数太多，需要等更久
            print(f"触发限流，第{attempt + 1}次重试，等5秒...")
            time.sleep(5)  # 限流等待时间更长，给服务器恢复空间

        except APIError as e:
            # 通用API业务错误（比如密钥无效、格式错误）
            # 这类错误重试也没用，直接打印错误并返回
            print(f"API错误：{e}")
            return None

    # 所有重试次数用完仍未成功
    return None


# ====================== 测试代码 ======================
if __name__ == "__main__":
    # 调用函数提问
    answer = ask_ai("什么是AI Agent？")

    # 判断返回结果并输出
    if answer:
        print(answer)
    else:
        print("调用失败")
