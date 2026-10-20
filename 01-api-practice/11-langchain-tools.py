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

# ---------- 新版Agent：从langgraph.prebuilt导入 ----------
from langgraph.prebuilt import create_react_agent
# LangGraph版ReAct Agent
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
# 对应资料：3. 工具的定义方式2：使用@tool装饰器
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
# 对应资料：1.3 工具调用的整体流程 + 2. 工具调用的四步
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

    if response.tool_calls:
        # 取出第一个工具调用
        tool_call = response.tool_calls[0]
        print(f"调用工具：{tool_call.name}")
        print(f"工具参数：{tool_call.args}")

        # ---------- 第三步：执行工具 ----------
        # 工具对象的 .invoke() 方法，直接传入参数字典执行
        # 对应原生：根据函数名找函数 + json.loads解析参数 + **解包调用
        tool_result = calculator.invoke(tool_call)
        print(f"工具返回：{tool_result}")

        # ---------- 第四步：第二次调用，生成最终回答 ----------
        # 构造消息列表：用户问题 + 模型工具调用消息 + 工具结果消息
        # 对应原生：手动拼接messages列表
        messages = [
            ("user", user_question),
            response,  # 模型返回的工具调用消息（对应原生assistant消息）
            ToolMessage(
                tool_call_id=tool_call.id,  # 必须绑定调用ID，对应原生tool_call_id
                content=tool_result
            )
        ]

        # 第二次调用模型，拿到自然语言最终回答
        final_response = llm_with_tools.invoke(messages)
        print(f"\n最终回答：{final_response.content}")


# ==================================================
# 第三部分：多工具自动选择测试
# 对应资料：4.3 案例3：多工具调用
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
            print(f"问题：{q} → 调用工具：{response.tool_calls[0].name}")
        else:
            print(f"问题：{q} → 不调用工具，直接回答")


# ==================================================
# 第四部分：LangGraph版自动ReAct Agent
# 对应资料：第07章 智能体 + ReAct模式
# 对应原生：08-simple-react.py 手写while循环
# 新版写法：使用 langgraph.prebuilt.create_react_agent
# ==================================================
def auto_react_agent():
    """
    LangGraph版自动ReAct Agent
    旧版用 AgentExecutor，新版直接用 create_react_agent
    内部自动完成：思考→行动→观察→再思考...循环直到结束
    """
    print("\n=== 3. LangGraph版自动ReAct Agent ===")

    # 创建ReAct Agent（一行搞定，不需要再创建AgentExecutor）
    app = create_react_agent(
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
