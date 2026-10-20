# AI Agent 学习项目

> 从零开始系统学习大模型应用开发，从原生API到底层原理，再到LangChain框架、LangGraph Agent编排、RAG检索增强生成。

---

## 📚 学习路线

### 阶段1：原生API基础（01~09号文件）
不依赖任何框架，纯手写实现，理解大模型调用底层原理。

| 文件 | 内容 | 对应能力 |
|------|------|---------|
| 01 | 基础API调用 | 第一个请求、密钥配置 |
| 02 | 流式输出 | 打字机效果、delta增量 |
| 03 | JSON结构化输出 | 强制格式、数据提取 |
| 04 | 基础工具调用 | Function Calling四步流程 |
| 05 | 流式工具调用 | 工具判断+流式回答 |
| 06 | 多轮对话 | 上下文记忆机制 |
| 07 | 多轮+工具 | 连续对话中自动调用工具 |
| 08 | 手写ReAct | Agent循环原理 |
| 09 | 工具类封装 | 工程化、异常重试、参数化 |

### 阶段2：LangChain框架（10~18号文件）
学习工程化框架，理解通用能力封装，用搭积木的方式快速构建AI应用。

| 文件 | 内容 | 对应能力 |
|------|------|---------|
| 10 | LangChain入门 | 模型调用、提示模板、链(Chain) |
| 11 | 工具调用 | @tool装饰器、bind_tools、create_react_agent |
| 12 | 记忆系统 | 缓冲记忆、滑动窗口、摘要记忆 |
| 13 | 结构化输出 | JsonOutputParser、Pydantic强类型校验 |
| 14 | RAG基础 | 文档加载→分割→向量化→检索→生成 |
| 15 | RAG进阶 | 查询改写、语义分块、效果优化 |
| 16 | LangGraph入门 | 节点、边、状态、条件分支、Agent编排 |
| 17 | 中间件系统 | 日志、Token统计、限流、非侵入扩展 |
| 18 | LangSmith调试 | 调用追踪、性能监控 |

---

## 🛠 技术栈

- **大模型**：DeepSeek（兼容OpenAI接口格式）
- **框架**：LangChain 1.x + LangGraph 1.x
- **向量数据库**：FAISS
- **开发语言**：Python 3.13
- **工具链**：python-dotenv、pydantic、openai SDK

---

## 🚀 快速开始

### 1. 环境准备
```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows PowerShell
.\venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量
在项目根目录创建 `.env` 文件：
```env
DEEPSEEK_API_KEY=你的API密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
```

### 3. 运行示例
```bash
# 运行基础调用
python 01-api-practice/01-hello.py

# 运行LangChain示例
python 10-langchain-hello.py
```

---

## 📁 项目结构
```
ai-agent-learning/
├── 01-api-practice/          # 阶段1：原生API基础
│   ├── 01-hello.py
│   ├── 02-stream.py
│   └── ...
├── 10-langchain-hello.py      # 阶段2：LangChain
├── 11-langchain-tools.py
├── 12-langchain-memory.py
├── requirements.txt           # 依赖清单
├── .env                       # 环境变量（不提交Git）
└── README.md                  # 项目说明
```

---

## 🎯 核心知识点

1. **大模型调用原理**：messages消息结构、system/user/assistant/tool四种角色
2. **流式输出**：stream=True + delta.content增量返回
3. **Function Calling**：模型只生成调用指令，本地Python执行函数
4. **ReAct模式**：思考→行动→观察→再思考...循环直到完成任务
5. **记忆机制**：完整保存、滑动窗口、摘要压缩三种策略
6. **RAG流程**：解决大模型知识滞后和幻觉问题
7. **LangGraph**：用状态图编排Agent，替代黑盒式AgentExecutor

---

## 📖 参考资料
- [LangChain官方文档](https://python.langchain.com/)
- [LangGraph官方文档](https://langchain-ai.github.io/langgraph/)
- DeepSeek API文档

---

> 学习笔记持续更新中...
