# ==================================================
# 功能：演示大模型工具调用（Function Calling）完整流程
# 场景：通用入门，用「计算器 + 获取时间」两个简单工具
# 核心逻辑：大模型只做判断，不执行代码；本地Python执行函数，返回结果
# ==================================================
from click import prompt
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


# ==================================================
# 【第一部分：本地工具函数】
# 这些是你自己写的Python代码
# 大模型**不会运行这些代码**，它只会生成「调用指令」，由你本地执行
# ==================================================

def calculator(expression: str) -> str:
    """
    计算器工具：计算数学表达式
    :param expression: 数学表达式字符串，例如 "2+3*4"、"100/5"
    :return: 返回计算结果字符串
    """
    try:
        # eval执行字符串形式的数学运算（仅演示用，生产环境慎用eval）
        result = eval(expression)
        return f"计算结果：{result}"
    except:
        # 表达式不合法时返回错误信息
        return "计算错误，表达式不合法"


def get_current_time() -> str:
    """
    时间工具：获取当前系统时间
    不需要传入参数
    :return: 格式化后的时间字符串
    """
    # 按年月日时分秒格式获取当前时间
    now = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
    return f"当前时间是：{now}"


# ==================================================
# 【第二部分：工具说明书】
# 用JSON格式描述工具，告诉大模型：
#   有哪些工具、每个工具是干嘛的、需要传什么参数
# 这是Function Calling的核心，描述越准确，模型调用越准
# ==================================================
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

# ==================================================
# 【第三部分：对话消息列表】
# 保存完整对话上下文，每一轮都要追加，不能丢
# 角色：system（系统设定）、user（用户说的）、assistant（AI说的）、tool（工具返回结果）
# ==================================================
prompt = input("请输入你的问题：")
messages = [
    # system：设定AI的身份和行为规则
    {"role": "system", "content": "你是一个智能助手，可以使用工具解答问题。需要工具时自动调用，不需要就直接回答。"},
    # user：用户的提问
    {"role": "user", "content": prompt}
]

# ==================================================
# 【第四部分：第一次请求大模型】
# 目的：让大模型判断「要不要调用工具、调用哪个、传什么参数」
# ==================================================
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,  # 传入对话上下文
    tools=tools,  # 传入工具说明书
    tool_choice="auto"  # auto：让大模型自己决定要不要调用工具
)

# 取出模型返回的第一条回复消息
resp_msg = response.choices[0].message

# ==================================================
# 【第五部分：判断并执行工具调用】
# 如果模型返回了 tool_calls，说明它想调用工具
# ==================================================
if resp_msg.tool_calls:
    # 取出第一个工具调用（一次可能调用多个，这里演示单个）
    tool_call = resp_msg.tool_calls[0]

    func_name = tool_call.function.name  # 要调用的函数名
    func_args = json.loads(tool_call.function.arguments)  # 参数是JSON字符串，转成Python字典

    print(f"🔧 模型决定调用工具：{func_name}")
    print(f"📥 传入参数：{func_args}")

    # 根据函数名，执行对应的本地Python函数
    if func_name == "calculator":
        # **func_args：把字典解包成关键字参数传给函数
        result = calculator(**func_args)
    elif func_name == "get_current_time":
        result = get_current_time(**func_args)
    else:
        result = "未知工具"

    print(f"✅ 工具执行结果：{result}")

    # ==================================================
    # 【非常关键】把调用信息和结果追加回对话
    # 不追加的话，第二次请求大模型不知道之前发生了什么
    # ==================================================
    messages.append(resp_msg)  # 1. 把模型发起工具调用的消息加回去
    messages.append({  # 2. 把工具执行结果作为tool角色消息加回去
        "role": "tool",
        "tool_call_id": tool_call.id,  # 必须带ID，大模型靠它匹配对应哪次调用
        "content": result
    })

    # ==================================================
    # 【第六部分：第二次请求大模型】
    # 目的：把工具执行结果给大模型，让它整理成自然语言回答
    # ==================================================
    final_resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages
    )
    print("\n🤖 AI最终回答：")
    print(final_resp.choices[0].message.content)

else:
    # 模型判断不需要调用工具，直接返回文字回答
    print("\n🤖 AI回答：")
    print(resp_msg.content)
