# ==================================================
# 文件：11-langchain-tools.py
# 定位：LangChain工具调用基础
# 对应原生：04-function-call.py（Function Calling四步流程）
# 核心知识点：
#   1. @tool 装饰器：自动生成工具描述，不用手写JSON
#   2. bind_tools：绑定工具到模型，自动转换为标准格式
#   3. 手动四步流程：对应原生四步，理解框架封装了什么
#   4. ToolMessage：工具结果消息，对应原生 role=tool
#   5. create_react_agent（LangGraph版）：自动循环，替代旧版AgentExecutor
# ==================================================

from dotenv import load_dotenv
import os
from datetime import datetime

# ---------- 导入LangChain核心组件 ----------
from langchain_core.tools import tool
# @tool 装饰器：把普通Python函数转为LangChain标准工具
# 自动根据函数签名、文档字符串生成标准工具描述JSON
# 对应原生：手写工具函数 + 手写tools说明书

from langchain_openai import ChatOpenAI
# 聊天模型类

from langchain_core.messages import ToolMessage
# ToolMessage：工具执行结果消息对象
# 对应原生：{"role": "tool", "tool_call_id": "...", "content": "..."}

# ---------- 新版Agent：langchain 1.x 官方推荐从 langchain.agents 导入 ----------
from langchain.agents import create_agent
# create_agent：LangChain 1.x 新版ReAct Agent（替代已弃用的 create_react_agent）
# 优势：基于状态机构建，可视化、可控、支持复杂分支

# 加载配置
load_dotenv()

# 初始化大模型
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0
)


# ==================================================
# 第一部分：用@tool装饰器定义工具
# 对比原生：需要分别写「函数实现」和「tools说明书JSON」
# LangChain：函数+文档字符串 = 工具，自动生成描述schema
# ==================================================
@tool
def calculator(expression: str) -> str:
    """
    计算数学表达式的结果。
    当用户需要计算数学题、算术表达式时调用此工具。

    Args:
        expression: 数学表达式字符串，例如 "2+3*4"、"100/5"
    """
    try:
        result = eval(expression)
        return f"计算结果：{result}"
    except:
        return "计算错误，表达式不合法"


@tool
def get_current_time() -> str:
    """
    获取当前的实时日期和时间。
    当用户询问现在几点、今天几号、当前时间时调用此工具。
    禁止使用历史对话中的旧时间数据，必须每次调用此工具获取最新时间。
    """
    now = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
    return f"当前时间是：{now}"


# ==================================================
# 第二部分：手动工具调用流程
# 对应原生：04-function-call.py 四步流程
# ==================================================
def manual_tool_call():
    """
    手动走完整工具调用四步流程
    用来对比原生写法，理解框架每一步封装了什么
    """
    print("=== 1. 手动工具调用流程 ===")

    # ---------- 第一步：绑定工具到模型 ----------
    # bind_tools：把工具列表转换为模型可识别的标准格式
    # 对应原生：每次调用接口手动传 tools 参数
    llm_with_tools = llm.bind_tools([calculator, get_current_time])

    user_question = "12加36乘2等于多少？"
    print("用户问题：", user_question)

    # ---------- 第二步：第一次调用，判断是否调用工具 ----------
    # 对应原生：第一次请求，获取tool_calls
    response = llm_with_tools.invoke(user_question)

    print("模型是否调用工具：", bool(response.tool_calls))

    # 工具名 → 工具对象 的映射字典，用于按名字分发执行
    tools_map = {"calculator": calculator, "get_current_time": get_current_time}

    if response.tool_calls:
        # 取出第一个工具调用
        # 【注意】新版LangChain 1.x中 tool_call 是字典，用 ["键"] 取值，没有 .name 属性
        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]      # 工具名称
        tool_args = tool_call["args"]      # 工具参数字典
        tool_id = tool_call["id"]          # 调用ID，用于匹配工具结果
        print(f"调用工具：{tool_name}")
        print(f"工具参数：{tool_args}")

        # ---------- 第三步：执行工具 ----------
        # 从映射字典里找到对应工具，传入参数字典执行
        # 对应原生：根据函数名找函数 + json.loads解析参数 + **解包调用
        tool_obj = tools_map.get(tool_name)
        tool_result = tool_obj.invoke(tool_args)
        print(f"工具返回：{tool_result}")

        # ---------- 第四步：第二次调用，生成最终回答 ----------
        # 构造消息列表：用户问题 + 模型工具调用消息 + 工具结果消息
        # 对应原生：手动拼接messages列表
        messages = [
            ("user", user_question),
            response,  # 模型返回的工具调用消息（对应原生assistant消息）
            ToolMessage(
                tool_call_id=tool_id,  # 必须绑定调用ID，对应原生tool_call_id
                content=tool_result
            )
        ]

        # 第二次调用模型，拿到自然语言最终回答
        final_response = llm_with_tools.invoke(messages)
        print(f"\n最终回答：{final_response.content}")


# ==================================================
# 第三部分：多工具自动选择测试
# ==================================================
def multi_tool_test():
    """测试不同问题下，模型自动选择调用哪个工具，或者不调用"""
    print("\n=== 2. 多工具自动选择 ===")

    llm_with_tools = llm.bind_tools([calculator, get_current_time])

    questions = [
        "现在几点了？",
        "5的3次方是多少？",
        "你好"
    ]

    for q in questions:
        response = llm_with_tools.invoke(q)
        if response.tool_calls:
            # 新版tool_call是字典，用 ["name"] 取值
            print(f"问题：{q} → 调用工具：{response.tool_calls[0]['name']}")
        else:
            print(f"问题：{q} → 不调用工具，直接回答")


# ==================================================
# 第四部分：新版 create_agent 自动ReAct Agent
# 对应原生：08-simple-react.py 手写while循环
# 使用 langchain.agents.create_agent（1.x官方推荐）
# ==================================================
def auto_react_agent():
    """
    新版 create_agent 自动ReAct Agent
    内部自动完成：思考→行动→观察→再思考...循环直到结束
    """
    print("\n=== 3. 新版 create_agent 自动ReAct Agent ===")

    # 创建ReAct Agent（一行搞定，官方新版 create_agent）
    app = create_agent(
        model=llm,
        tools=[calculator, get_current_time],
    )

    # 执行任务，传入消息列表
    result = app.invoke({
        "messages": [
            ("user", "现在几点了？再过6小时是几点？")
        ]
    })

    # 从消息列表最后一条取出最终回答
    final_answer = result["messages"][-1].content

    print(f"\n最终答案：{final_answer}")

    # 打印执行过程（可选，观察每一步）
    print("\n--- 执行过程 ---")
    for i, msg in enumerate(result["messages"]):
        msg_type = msg.__class__.__name__
        content = str(msg.content)[:50]
        print(f"步骤{i+1} [{msg_type}]：{content}...")


# ==================================================
# 主入口
# ==================================================
if __name__ == "__main__":
    manual_tool_call()    # 手动流程，对应原生四步，对比学习
    multi_tool_test()     # 多工具自动选择
    auto_react_agent()   # LangGraph版自动Agent
