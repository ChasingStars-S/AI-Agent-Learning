# 01-hello.py  第一个大模型调用脚本

# 导入需要的库
from openai import OpenAI
from dotenv import load_dotenv
import os

# # 加载.env文件里的密钥
# load_dotenv()
#
# # 创建客户端（连接DeepSeek）
# client = OpenAI(
#     api_key=os.getenv("DEEPSEEK_API_KEY"),      # 从.env读密钥
#     base_url=os.getenv("DEEPSEEK_BASE_URL")     # DeepSeek的API地址
# )
#
# # 调用大模型
# response = client.chat.completions.create(  # 调用聊天接口，发请求
#     model="deepseek-chat",          # 用哪个模型（deepseek-chat是主力模型）
#     messages=[                      # 对话消息列表，是一个数组
#         {"role": "system", "content": "你是一个 helpful 的助手"},   # 系统提示：设定AI的身份
#         {"role": "user", "content": "你好，用三句话介绍一下你自己"}   # 用户提问
#     ],
#     temperature=0.7                 # 创造性：0=保守，1=放飞自我，0.7中间值
# )
#
# # 拿到回答并打印
# answer = response.choices[0].message.content  # 从返回结果里取出回答文本
# print("AI回答：")
# print(answer)


#建立与AI的链接(ollama)
client = OpenAI(
    api_key='ollama',
    base_url="http://localhost:11434/v1"
)

#调用AI大模型
response = client.chat.completions.create(  # 调用聊天接口，发请求
    model="deepseek-r1:8b",                 # 用哪个模型（deepseek-chat是主力模型）
    messages=[                              # 对话消息列表，是一个数组
        {"role": "system", "content": "你是一个 helpful 的助手"},  # 系统提示：设定AI的身份
        {"role": "user", "content": "你好，用三句话介绍一下你自己"}   # 用户提问
    ],
    temperature=0.7,               # 创造性：0=保守，1=放飞自我，0.7中间值
    stream=False,                   # 关键参数：开启流式输出
    )

# 非流式输出 拿到回答并打印
answer = response.choices[0].message.content  # 从返回结果里取出回答文本
print("AI回答：")
print(answer)

# 流式输出：一个字一个字打印
# print("AI回答：", end="", flush=True) #`flush=True` 的作用：强制立刻打印，不缓冲。不加的话可能还是攒一堆才打印，看不出打字机效果。
# for chunk in response:
#     # 每次拿到一小段内容
#     if chunk.choices[0].delta.content:
#         print(chunk.choices[0].delta.content, end="", flush=True)
#
# print()  # 最后换行