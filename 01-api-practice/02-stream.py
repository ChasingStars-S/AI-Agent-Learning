# 02-stream.py  流式输出（打字机效果）

from openai import OpenAI

#建立与AI的链接(ollama)
client = OpenAI(
    api_key='ollama',
    base_url="http://localhost:11434/v1"
)

# 调用大模型，关键：stream=True 开启流式
stream = client.chat.completions.create(
    model="deepseek-r1:8b",
    messages=[
        {"role": "system", "content": "你是一个 helpful 的助手"},
        {"role": "user", "content": "写一篇300字的春天散文，语言优美"}
    ],
    temperature=0.7,
    stream=True   # 关键参数：开启流式输出
)

# 流式输出：一个字一个字打印
# print("AI回答：", end="", flush=True)
# for chunk in stream:
#     # 每次拿到一小段内容
#     if chunk.choices[0].delta.content:
#         print(chunk.choices[0].delta.content, end="", flush=True)

# print()  # 最后换行

# 加个计数器，看看一共返回了多少段
# print("AI回答：", end="", flush=True)
# count = 0
# for chunk in stream:
#     # 每次拿到一小段内容
#     if chunk.choices[0].delta.content:
#         print(chunk.choices[0].delta.content, end="", flush=True)
#         count += 1
# print()
# print(f"共返回了 {count} 段数据")

# 把流式结果存到变量里
print("AI回答：", end="", flush=True)
full_answer = ""   # 用一个字符串拼接完整回答
for chunk in stream:
    if chunk.choices[0].delta.content:
        content = chunk.choices[0].delta.content
        print(content, end="", flush=True)
        full_answer += content   # 拼接到变量里
print()
print("-----------")
print("完整回答已保存，长度：", len(full_answer), "个字符")
print("前50个字：", full_answer[:50])

