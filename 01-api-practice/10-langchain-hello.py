# ==================================================
# 文件：10-langchain-hello.py
# 定位：LangChain入门 · 第一个框架调用
# 核心知识点：
#   1. ChatOpenAI：框架封装的大模型客户端
#   2. ChatPromptTemplate：提示词模板，支持变量复用
#   3. StrOutputParser：输出解析器，自动提取纯文本
#   4. 链（Chain）：用 | 管道符串联组件，数据依次流转
# 对比原生API：框架封装了重复的样板代码，组件可插拔、可复用
# ==================================================

from dotenv import load_dotenv
import os

# ---------- 导入LangChain核心组件 ----------
from langchain_openai import ChatOpenAI
# ChatOpenAI：LangChain封装的聊天模型类
# 对应原生API：OpenAI客户端
# 作用：统一封装大模型交互逻辑，换模型只改初始化，业务代码不用动
# DeepSeek完全兼容OpenAI接口格式，直接用这个类对接即可

from langchain_core.prompts import ChatPromptTemplate
# ChatPromptTemplate：聊天提示词模板组件
# 对应原生API：手动拼接字符串、组装messages列表
# 作用：用 {} 做变量占位符，提示词可复用、易维护，不用硬编码每一条消息

from langchain_core.output_parsers import StrOutputParser

# StrOutputParser：字符串输出解析器
# 对应原生API：手动写 .choices[0].message.content 提取文本
# 作用：自动从模型返回的消息对象中提取纯文本，统一输出格式
# 扩展：还有JSON解析器、列表解析器等，对应不同输出格式


# 加载 .env 配置文件
load_dotenv()

# ====================== 1. 初始化大模型 ======================
# 对应原生：创建OpenAI客户端实例
llm = ChatOpenAI(
    model="deepseek-chat",  # 模型名称
    api_key=os.getenv("DEEPSEEK_API_KEY"),  # API密钥，从环境变量读取
    base_url=os.getenv("DEEPSEEK_BASE_URL"),  # 接口地址
    temperature=0.7  # 创造性温度
)


# ====================== 2. 基础调用（对应原生普通问答） ======================
def basic_chat():
    """
    最基础的直接调用方式
    消息格式和原生API完全一致：role + content
    适合快速验证、简单场景
    """
    print("=== 1. 基础调用 ===")

    # invoke：LangChain统一的执行方法，对应原生的 create()
    response = llm.invoke([
        {"role": "system", "content": "你是一个简洁的助手"},
        {"role": "user", "content": "1+1等于几"}
    ])

    # .content 获取文本内容，和原生类似
    print("回答：", response.content)


# ====================== 3. 提示词模板 + 链（LangChain核心思想） ======================
def template_chain():
    """
    提示词模板 + 链 的调用方式
    核心：用 | 管道符把多个组件按顺序串成一条流水线
    数据流向：模板填充变量 → 传给大模型 → 解析器提取文本
    """
    print("\n=== 2. 提示词模板 + 链 ===")

    # 定义提示词模板：{} 包裹的是变量占位符
    # from_messages：按消息列表格式创建模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个{style}的助手，回答要{length}"),
        ("user", "{question}")
    ])

    # 创建输出解析器：自动提取纯文本内容
    output_parser = StrOutputParser()

    # 【核心】用 | 管道符串成链（Chain）
    # 执行顺序从左到右：prompt 输出 → 输入 llm → llm 输出 → 输入 output_parser
    chain = prompt | llm | output_parser

    # 调用链的 invoke 方法，传入变量字典
    # 模板会自动把变量填充进去，再交给大模型，最后解析成字符串
    result = chain.invoke({
        "style": "幽默",
        "length": "简短",
        "question": "什么是Python"
    })

    # 直接拿到纯字符串结果，不用自己取 .content
    print("回答：", result)


# ====================== 4. 流式输出 ======================
def stream_chat():
    """
    流式逐字输出，对应原生的 stream=True
    用链的方式实现，同样支持模板、解析器
    """
    print("\n=== 3. 流式输出 ===")

    # 定义模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个文采很好的助手"),
        ("user", "写一句关于春天的话")
    ])

    # 串成链
    chain = prompt | llm | StrOutputParser()

    # stream() 方法开启流式调用，逐段返回文本
    # 对应原生：for chunk in stream: 遍历增量
    print("回答：", end="", flush=True)
    for chunk in chain.stream({"question": "写一句关于春天的话"}):
        print(chunk, end="", flush=True)
    print()


# ====================== 主入口 ======================
if __name__ == "__main__":
    # 依次执行三个演示
    basic_chat()
    template_chain()
    stream_chat()
