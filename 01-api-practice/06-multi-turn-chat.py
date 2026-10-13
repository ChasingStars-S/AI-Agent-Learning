# ==================================================
# 06-multi-turn-chat.py
# 多轮对话：带记忆的聊天机器人
# 核心：messages 列表就是记忆，每轮都追加，每次都完整传给模型
# ==================================================

from openai import OpenAI  # 大模型客户端，DeepSeek兼容OpenAI接口格式
from dotenv import load_dotenv  # 读取 .env 配置文件，存放API密钥
import os  # 读取环境变量

# 加载 .env 文件里的配置（API密钥、接口地址）
load_dotenv()

# 创建大模型客户端，建立和DeepSeek的连接
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量读取密钥
    base_url=os.getenv("DEEPSEEK_BASE_URL")  # DeepSeek API地址
)

# ========== 对话历史（记忆） ==========
# 一开始只有系统提示，后面每说一句话都追加进去
messages = [
    {"role": "system", "content": "你是一个乐于助人的助手，回答简洁。"}
]

print("=== 多轮聊天开始（输入 退出 结束）===")

# 循环对话
while True:
    # 1. 用户输入
    user_input = input("你：")

    # 退出命令
    if user_input in ["退出", "exit", "quit"]:
        print("再见！")
        break

    # 2. 把用户的话追加到记忆里
    messages.append({"role": "user", "content": user_input})

    # 3. 请求大模型（传入完整记忆）
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,  # 每次都传完整历史
        temperature=0.7,
        stream=False,
    )

    # 4. 拿到回答（完整文本）
    answer = response.choices[0].message.content
    print(f"AI：{answer}")

    # 5. 把AI的回答也追加到记忆里
    messages.append({"role": "assistant", "content": answer})

    # 可选：打印当前记忆长度
    # print(f"（当前对话轮数：{len(messages)//2}）")

    # 流式输出
    # print("AI：", end="", flush=True)
    #
    # full_answer = ""  # 拼接完整回答，用于追加到记忆
    # for chunk in response:
    #     if chunk.choices[0].delta.content:
    #         content = chunk.choices[0].delta.content
    #         print(content, end="", flush=True)
    #         full_answer += content
    #
    # print()  # 回答结束换行
    #
    # # ==========完整回答追加到记忆，保证下一轮能记住 ==========
    # messages.append({"role": "assistant", "content": full_answer})

