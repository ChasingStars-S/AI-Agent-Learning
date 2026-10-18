# ==================================================
# 08-simple-react.py
# 手写极简ReAct循环
# ReAct = Reasoning（思考） + Acting（行动）
# 循环：思考 → 行动 → 观察 → 思考... 直到任务完成
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

# ====================== 工具函数 ======================
def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)
        return f"计算结果：{result}"
    except:
        return "计算错误，表达式不合法"


def get_current_time() -> str:
    """获取当前时间"""
    now = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
    return f"当前时间是：{now}"


# 工具字典：通过函数名快速找到对应函数
tools_map = {
    "calculator": calculator,
    "get_current_time": get_current_time
}


# ====================== 工具说明书 ======================
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学题、算术表达式。需要计算数字时调用",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取**实时**当前时间、日期。只要用户问现在几点、今天几号，必须调用这个工具获取最新时间，禁止使用历史数据。",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    }
]


# 系统提示：规定ReAct的思考格式
system_prompt = """
你是一个会思考的智能助手，按照以下步骤解决问题：
1. 先思考下一步该做什么
2. 需要工具就调用工具，拿到结果后继续思考
3. 有了最终答案就直接回答，不要继续调用工具

注意：
- 每次只思考一步，不要一次想多步
- 调用工具时只输出工具调用，不要输出最终答案
- 工具执行完拿到结果后，再继续思考
"""

# 对话历史（记忆）全局对话记忆，全程只初始化一次
messages = [
    {"role": "system", "content": system_prompt}
]

# ====================== ReAct循环核心 ======================
def react_run(user_question: str, max_steps: int = 5):
    """
    ReAct循环执行函数
    :param user_question: 用户的问题/目标
    :param max_steps: 最大循环步数，防止死循环
    """

    global messages  # 使用全局记忆

    print(f"🧑 用户问题：{user_question}")
    print("=" * 50)

    # 把用户问题追加到全局记忆
    messages.append({"role": "user", "content": user_question})

    # 循环执行
    for step in range(max_steps):
        print(f"\n--- 第 {step+1} 步 思考中 ---")

        # 1. 请求大模型：思考下一步
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        resp_msg = response.choices[0].message

        # 2. 判断：有没有调用工具
        if resp_msg.tool_calls:
            # 有工具调用 → 行动 + 观察
            tool_call = resp_msg.tool_calls[0]
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            print(f"🎯 决定行动：调用 {func_name}，参数：{func_args}")

            # 执行工具（行动）
            func = tools_map[func_name]
            observation = func(**func_args)  # 拿到观察结果

            print(f"👁 观察结果：{observation}")

            # 把思考和观察都加入记忆
            messages.append(resp_msg.model_dump())  # 转成纯dict
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": observation
            })

        else:
            # 没有工具调用 → 任务完成，给出最终答案
            print("\n✅ 最终答案：")
            print(resp_msg.content)
            # 把最终回答也追加到记忆里
            messages.append({"role": "assistant", "content": resp_msg.content})
            return resp_msg.content

    # 达到最大步数还没结束
    print("\n⚠️ 达到最大步数，任务未完成")
    return None


# ====================== 启动测试 ======================
if __name__ == "__main__":
    # 测试题：需要两次工具调用的问题
    # react_run("现在几点了？再过6小时是几点？")
    # print("\n" + "=" * 60 + "\n")

    # 控制台输入对话
    print("=== ReAct 智能助手（输入 退出 结束）===")
    print("💡 可以问需要计算、查时间的问题，我会一步步思考解决\n")

    while True:
        # 获取用户输入
        question = input("请提问：")

        # 退出命令
        if question in ["退出", "exit", "quit"]:
            print("再见！")
            break

        # 空输入跳过
        if not question.strip():
            continue

        # 执行ReAct循环
        react_run(question)
        print("\n" + "=" * 60 + "\n")
