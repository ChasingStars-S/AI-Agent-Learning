# ==================================================
# 文件：09-llm-utils.py
# 定位：大模型调用工具类封装
# 核心优化点：
#   1. 参数化配置：模型、温度、重试次数支持外部传入覆盖
#   2. 入口校验：关键方法提前校验参数，避免深层报错
#   3. 异常容错：JSON解析、流式输出增加异常捕获，不直接崩溃
#   4. 多工具支持：支持一次调用多个工具，全部执行完再返回
#   5. 错误可视化：解析失败时打印模型原始输出，方便排查
# ==================================================

# ---------- 导入依赖库 ----------
from openai import OpenAI, APIError, APIConnectionError, RateLimitError
# OpenAI：大模型客户端类（DeepSeek兼容OpenAI接口格式）
# APIError：通用API业务错误
# APIConnectionError：网络连接异常
# RateLimitError：调用频率超限（限流）

from dotenv import load_dotenv  # 加载 .env 配置文件，读取API密钥等环境变量
import os                       # 读取系统环境变量
import json                     # JSON字符串与Python数据结构互转
import time                     # 重试等待延时


# 加载 .env 配置（全局只执行一次）
load_dotenv()


class LLMClient:
    """
    大模型客户端工具类
    封装所有与DeepSeek交互的逻辑，统一管理密钥、模型、重试策略
    对外提供：普通问答、流式输出、JSON结构化输出、工具调用 四大能力
    """

    def __init__(self, model: str = None, default_temperature: float = None, max_retries: int = None):
        """
        参数化初始化，支持外部覆盖配置，不传入则使用默认值
        :param model: 模型名称，默认 deepseek-chat
        :param default_temperature: 默认创造性温度，默认 0.7
        :param max_retries: 接口失败最大重试次数，默认 3
        """
        # 创建大模型客户端，从环境变量读取密钥与接口地址
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL")
        )

        # 配置项：外部传入则覆盖，否则使用默认值
        self.model = model if model else "deepseek-chat"
        self.default_temperature = default_temperature if default_temperature is not None else 0.7
        self.max_retries = max_retries if max_retries is not None else 3

    # ==================================================
    # 方法1：普通单轮问答
    # ==================================================
    def chat(self, prompt: str, system: str = "你是一个 helpful 的助手", temperature: float = None) -> str:
        """
        普通单轮问答，一次性返回完整回答文本
        :param prompt: 用户提问内容
        :param system: 系统提示词，设定AI身份与规则
        :param temperature: 创造性温度，不传则使用类默认值
        :return: AI完整回答字符串；调用失败返回"调用失败"
        """
        # 入口参数校验：空内容提前抛出友好提示，避免深层报错
        if not prompt.strip():
            raise ValueError("prompt 不能为空字符串")

        # 未传温度则使用类默认温度
        if temperature is None:
            temperature = self.default_temperature

        # 带重试循环：失败自动重试，最多 max_retries 次
        for attempt in range(self.max_retries):
            try:
                # 调用大模型聊天接口
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature
                )
                # 成功则直接返回回答内容
                return response.choices[0].message.content

            # 按异常类型分别处理
            except APIConnectionError:
                # 网络连接错误，等待2秒后重试
                print(f"网络错误，第{attempt+1}次重试...")
                time.sleep(2)
            except RateLimitError:
                # 触发限流，等待5秒给服务器恢复时间
                print(f"触发限流，第{attempt+1}次重试...")
                time.sleep(5)
            except APIError as e:
                # 其他API业务错误，打印错误信息并终止重试
                print(f"API错误：{e}")
                break

        # 所有重试均失败
        return "调用失败"

    # ==================================================
    # 方法2：流式输出（打字机效果）
    # ==================================================
    def chat_stream(self, prompt: str, system: str = "你是一个 helpful 的助手", temperature: float = None):
        """
        流式输出，返回生成器，逐字逐段产出文本
        :return: 生成器，每次yield一小段文本
        """
        # 入口参数校验
        if not prompt.strip():
            raise ValueError("prompt 不能为空字符串")

        if temperature is None:
            temperature = self.default_temperature

        try:
            # 开启流式模式 stream=True
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                stream=True  # 核心参数：开启流式，不一次性返回全部结果
            )

            # 遍历流式数据包，逐个产出文本
            for chunk in stream:
                # 只在有实际内容时产出，空数据包跳过
                if chunk.choices[0].delta.content:
                    # yield：生成器关键字，逐个返回，不一次性结束
                    yield chunk.choices[0].delta.content

        # 流式过程中出现异常，返回错误文本而不是直接崩溃
        except Exception as e:
            yield f"[调用出错：{str(e)}]"

    # ==================================================
    # 方法3：强制JSON结构化输出
    # ==================================================
    def chat_json(self, prompt: str, system: str = "你只输出标准JSON，不要其他文字") -> dict:
        """
        强制大模型返回标准JSON格式，自动解析为Python字典
        :param prompt: 用户指令/待提取的内容
        :param system: 系统提示，约束输出格式
        :return: 解析成功返回Python字典；解析失败返回None
        """
        # 入口参数校验
        if not prompt.strip():
            raise ValueError("prompt 不能为空字符串")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,  # 温度设为0，降低创造性，保证格式稳定
                response_format={"type": "json_object"}  # 强制模型输出JSON格式
            )

            # json.loads：把JSON格式字符串转成Python字典
            return json.loads(response.choices[0].message.content)

        # 捕获JSON解析错误，打印原始输出方便排查
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析失败：{e}")
            print(f"模型原始输出：{response.choices[0].message.content}")
            return None
        # 捕获其他所有调用错误
        except Exception as e:
            print(f"❌ 调用失败：{e}")
            return None

    # ==================================================
    # 方法4：工具调用（支持多工具并行执行）
    # ==================================================
    def run_with_tools(self, prompt: str, tools: list, tools_map: dict, system: str = "你可以使用工具解答问题") -> str:
        """
        带工具调用的问答
        支持模型一次返回多个工具调用，全部执行完毕后统一生成回答
        :param prompt: 用户提问
        :param tools: 工具说明书列表（给大模型看的JSON描述）
        :param tools_map: 本地执行函数字典 {函数名字符串: Python函数对象}
        :param system: 系统提示
        :return: 最终自然语言回答
        """
        # 入口参数校验
        if not prompt.strip():
            raise ValueError("prompt 不能为空字符串")
        if not tools or not isinstance(tools, list):
            raise ValueError("tools 必须是非空列表")

        # 对话消息列表
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]

        # ---------- 第一次请求大模型：判断是否调用工具、调用哪些工具 ----------
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto"  # 自动决定是否调用工具
        )
        resp_msg = response.choices[0].message

        # ---------- 判断：模型发起了工具调用 ----------
        if resp_msg.tool_calls:
            # 先把模型的工具调用请求追加进对话上下文
            messages.append(resp_msg.model_dump())  # 转成纯dict

            # 循环处理所有工具调用（原版本只处理第一个）
            for tool_call in resp_msg.tool_calls:
                func_name = tool_call.function.name

                # 解析工具参数，带容错
                try:
                    # 参数是JSON字符串，转成Python字典
                    func_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    result = "工具参数解析失败，格式不合法"
                else:
                    # 从工具字典中找到对应函数并执行
                    if func_name in tools_map:
                        try:
                            # **func_args：字典解包，作为关键字参数传入函数
                            result = tools_map[func_name](**func_args)
                        # 捕获工具执行异常，返回错误信息，不中断整体流程
                        except Exception as e:
                            result = f"工具执行出错：{str(e)}"
                    else:
                        result = f"未知工具：{func_name}"

                # 每个工具的执行结果，都作为tool消息追加进上下文
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,  # 必须携带ID，模型靠它匹配调用与结果
                    "content": result
                })

            # ---------- 所有工具执行完毕，第二次请求大模型 ----------
            # 模型结合所有工具结果，整理成自然语言最终回答
            final_resp = self.client.completions.create(
                model=self.model,
                messages=messages
            )
            return final_resp.choices[0].message.content

        # ---------- 不需要调用工具，直接返回回答 ----------
        else:
            return resp_msg.content


# ==================================================
# 控制台交互测试入口
# 直接运行本文件即可启动菜单式测试
# ==================================================
if __name__ == "__main__":
    # 实例化客户端（使用默认配置）
    llm = LLMClient()

    # ---------- 预置测试用工具函数 ----------
    def add(a, b):
        """测试用工具：计算两个数字的和"""
        return f"计算结果：{a + b}"

    # 工具说明书（给大模型看）
    test_tools = [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "计算两个数字的和",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer", "description": "第一个数字"},
                        "b": {"type": "integer", "description": "第二个数字"}
                    },
                    "required": ["a", "b"]
                }
            }
        }
    ]
    # 工具映射字典（本地执行用）
    test_tools_map = {"add": add}

    # ---------- 菜单界面 ----------
    print("=" * 50)
    print("       LLM 工具类测试控制台")
    print("=" * 50)

    while True:
        print("\n请选择测试功能：")
        print("1. 普通问答")
        print("2. 流式输出（打字机效果）")
        print("3. 强制JSON输出")
        print("4. 工具调用（加法计算）")
        print("0. 退出")

        choice = input("\n请输入编号：").strip()

        if choice == "0":
            print("再见！")
            break

        elif choice == "1":
            prompt = input("请输入你的问题：")
            print("\nAI回答：")
            print(llm.chat(prompt))

        elif choice == "2":
            prompt = input("请输入你的问题：")
            print("\nAI回答：", end="", flush=True)
            # 遍历生成器，逐字打印
            for text in llm.chat_stream(prompt):
                print(text, end="", flush=True)
            print()  # 回答结束换行

        elif choice == "3":
            prompt = input("请输入提取指令（比如：提取信息：姓名张三，年龄20，输出name,age）：")
            data = llm.chat_json(prompt)
            if data:
                print("\n✅解析成功，得到字典：")
                print(data)

        elif choice == "4":
            prompt = input("请输入加法问题（比如：12加34等于多少）：")
            result = llm.run_with_tools(prompt, test_tools, test_tools_map)
            print(f"\nAI回答：{result}")

        else:
            print("输入无效，请输入 0-4 的编号")
