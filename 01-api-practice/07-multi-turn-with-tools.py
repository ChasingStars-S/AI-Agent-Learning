# ==================================================
# 07-multi-turn-with-tools.py
# 多轮对话 + 工具调用
# 特点：正常聊天是多轮；遇到需要工具的问题，自动调用工具，然后继续对话
# ==================================================

from openai import OpenAI  # 大模型客户端，DeepSeek兼容OpenAI接口格式
from dotenv import load_dotenv  # 读取 .env 配置文件，存放API密钥
import json  # 解析模型返回的工具调用参数（JSON字符串转字典）
import os  # 读取环境变量
from datetime import datetime  # Python内置时间库，用来获取本地时间

# 加载 .env 文件里的配置（API密钥、接口地址）
load_dotenv()

# 创建大模型客户端，建立和DeepSeek的连接
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量读取密钥
    base_url=os.getenv("DEEPSEEK_BASE_URL")  # DeepSeek API地址
)

# 本地工具函数
def calculator(expression: str) -> str:
    try:
        result = eval(expression)
        return f"计算结果：{result}"
    except:
        return "计算错误"


def get_current_time() -> str:
    now = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
    return f"当前时间是：{now}"


# 工具说明书
tools = [
    # 工具1：计算器
    {
        "type": "function",  # 固定值，表示这是一个函数工具
        "function": {
            "name": "calculator",  # 函数名，必须和上面Python函数名完全一致
            "description": "计算数学题、算术表达式。用户问计算相关问题时调用",  # 工具用途描述，模型靠这个判断什么时候用
            "parameters": {  # 参数定义
                "type": "object",
                "properties": {
                    "expression": {  # 参数名
                        "type": "string",  # 参数类型
                        "description": "要计算的数学表达式，比如 2+3、10*5+3"  # 参数说明
                    }
                },
                "required": ["expression"]  # 必填参数列表，缺了模型会追问用户
            }
        }
    },

    # 工具2：获取当前时间
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取现在的时间、几点了、今天几号。用户问时间日期时调用",
            "parameters": {
                "type": "object",
                "properties": {},  # 这个工具不需要参数，所以是空的
                "required": []  # 没有必填项
            }
        }
    }
]

# 对话记忆
messages = [
    {"role": "system", "content": "你是智能助手，可以聊天，可以用工具解答问题。"}
]

print("=== 多轮聊天（输入  退出  结束）===")

while True:
    user_input = input("你：")
    if user_input in ["退出", "exit"]:
        break

    messages.append({"role": "user", "content": user_input})

    # ===== 第一步：判断是否调用工具 =====
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    resp_msg = response.choices[0].message

    # ===== 如果需要调用工具 =====
    if resp_msg.tool_calls:
        tool_call = resp_msg.tool_calls[0]
        func_name = tool_call.function.name
        func_args = json.loads(tool_call.function.arguments)

        print(f"（调用工具：{func_name}）")

        # 执行函数
        if func_name == "calculator":
            result = calculator(**func_args)
        elif func_name == "get_current_time":
            result = get_current_time(**func_args)
        else:
            result = "未知工具"

        # 追加调用信息和结果
        messages.append(resp_msg)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result
        })

        # 第二次请求：拿到最终回答
        final_resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages
        )
        answer = final_resp.choices[0].message.content

    # ===== 不需要工具，直接回答 =====
    else:
        answer = resp_msg.content

    print(f"AI：{answer}")
    messages.append({"role": "assistant", "content": answer})
