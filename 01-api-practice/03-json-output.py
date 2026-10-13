# ==================================================
# 03-json-output.py 强制输出JSON格式
# ==================================================
from openai import OpenAI

import json  # 用来解析json字符串


#建立与AI的链接(ollama)
client = OpenAI(
    api_key='ollama',
    base_url="http://localhost:11434/v1"
)

response = client.chat.completions.create(
    model="deepseek-r1:8b",
    messages=[
        {
            "role": "system",
            "content": "你是数据提取助手，**只返回标准JSON，不要任何额外解释、前言后语**。禁止输出markdown代码块，不要写```json标记。"
        },
        {
            "role": "user",
            "content": "提取人物信息：姓名李四，年龄25，职业软件工程师，爱好跑步、看书。输出字段：name, age, job, hobbies。hobbies是数组，age是数字。"
        }
    ],
    temperature=0.3, # 调低创造性，保证输出格式稳定
    response_format={"type": "json_object"} # ✅核心参数：强制模型输出JSON
)

# 获取AI返回的json字符串
json_str = response.choices[0].message.content
print("AI原始返回字符串：")
print(json_str)
print("-"*40)

# 【重点】json.loads()：把文本字符串转成Python字典
data = json.loads(json_str)
print("✅解析完成，得到Python字典：")
print(data)
print("单独取出姓名：", data["name"])
print("单独取出第一个爱好：", data["hobbies"][0])
