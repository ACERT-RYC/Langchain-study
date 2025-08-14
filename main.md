1.LangChain介绍
LangChain 是一个用于开发大语言模型（LLM）驱动应用的开源框架（Python/JS 库）。它的核心目标是简化将 LLM（如 GPT、Llama 等）与外部数据源、计算逻辑和工具集成的过程，从而构建更强大、可定制化的 AI 应用。
官方文档（https://python.langchain.com）开始，逐步探索各个模块的高级用法。
2.LangChain 能做什么?
 场景 	 主要用途 
PromptTemplate	模板化提示词，批量生成不同内容
Chains	把多个步骤串起来，形成完整流程
Memory	保存上下文，实现多轮对话
Retriever + VectorStore	做RAG，先在知识库检索，再生成回答，解决幻觉问题，防止胡编乱造
Agent + Tools	根据用户需求自动选择合适工具，比如先查天气再写诗
3.对比示例(使用LangChain的优点)
直接调用API
import openai
response = openai.chatcompletion.create(
    mode1="gpt-3.5-turbo"
    messages=[{"role":"user","content":"请帮我写一篇面试自我介绍"}]
print(response.choices[o] .message.content)
from langchain.prompts import PromptTemplate
from 1angchain.chains import LLMchain
from langchain_openai import ChatopenAI
llm = ChatOpenAI()
prompt = PromptTemplate.from_template("请帮我写一篇{topic}")
chain = LLMchain(llm=llm, prompt=prompt)
result= chain.run({"topic"："面试自我介绍})
print(result)
对比可以知道，使用框架，我们可以灵活的进行不同需求的实现，还可以有更多的串联步骤
4.LangChain核心模块
一、核心六大模块
1. 模型（Models）
负责与各种语言模型交互
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Ollama

# OpenAI 模型
llm_openai = ChatOpenAI(model="gpt-4-turbo")

# Anthropic 模型
llm_claude = ChatAnthropic(model="claude-3-opus")

# 本地模型
llm_local = Ollama(model="llama3")
2. 提示（Prompts）
管理提示模板和优化
from langchain.prompts import ChatPromptTemplate

template = """
你是一位专业的{role}。请根据以下上下文回答问题：
{context}

问题：{question}
"""

prompt = ChatPromptTemplate.from_template(template)
formatted_prompt = prompt.format(
    role="医生",
    context="患者有发热、咳嗽症状，体温38.5℃",
    question="可能的诊断是什么？"
)
3. 数据连接（Indexes）
集成外部数据源
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 1. 加载文档
loader = WebBaseLoader("https://zh.wikipedia.org/wiki/人工智能")
docs = loader.load()

# 2. 分割文本
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

# 3. 向量化存储
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=OpenAIEmbeddings()
)

# 4. 创建检索器
retriever = vectorstore.as_retriever()
4. 记忆（Memory）
管理对话状态和历史
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 使用示例
memory.save_context(
    {"input": "你好，我叫小明"}, 
    {"output": "你好小明！有什么可以帮你的？"}
)
memory.save_context(
    {"input": "我的名字是什么？"}, 
    {"output": "你叫小明。"}
)
5. 链（Chains）
组合多个组件形成工作流
from langchain.chains import RetrievalQA

# 创建问答链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm_openai,
    chain_type="stuff",
    retriever=retriever,
    memory=memory
)

# 使用链
response = qa_chain.invoke({"query": "人工智能的主要应用领域有哪些？"})
print(response["result"])
6. 代理（Agents）
动态调用工具完成任务
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool

# 定义工具
def currency_converter(amount: float, from_currency: str, to_currency: str):
    """货币转换工具"""
    conversion_rates = {"USD": 1, "CNY": 7.2, "EUR": 0.93}
    return amount * conversion_rates[to_currency] / conversion_rates[from_currency]

tools = [
    Tool(
        name="CurrencyConverter",
        func=currency_converter,
        description="货币转换工具，支持USD, CNY, EUR"
    )
]

# 创建代理
agent = create_tool_calling_agent(
    llm=llm_openai,
    tools=tools,
    prompt=prompt
)

# 执行代理
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = agent_executor.invoke({
    "input": "请将100美元转换为人民币"
})
二、高级应用模式
1. 检索增强生成（RAG）
from langchain_core.runnables import RunnablePassthrough

# 定义RAG管道
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm_openai
    | StrOutputParser()
)
 
# 使用RAG
response = rag_chain.invoke("LangChain是什么？")
2. 多智能体协作
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.agents import initialize_multi_agent

# 定义不同角色的代理
researcher = create_openai_tools_agent(
    llm=llm_openai,
    tools=[web_search_tool],
    system_message="你是一名研究人员"
)

writer = create_openai_tools_agent(
    llm=llm_openai,
    tools=[drafting_tool],
    system_message="你是一名内容撰写者"
)

# 初始化多代理系统
team = initialize_multi_agent(
    agents=[researcher, writer],
    manager_llm=llm_openai
)

# 执行团队任务
result = team.invoke("撰写一篇关于量子计算的科普文章")
3. 结构化输出
from langchain_core.pydantic_v1 import BaseModel, Field

# 定义输出结构
class PatientInfo(BaseModel):
    name: str = Field(description="患者姓名")
    age: int = Field(description="患者年龄")
    symptoms: list[str] = Field(description="症状列表")
    diagnosis: str = Field(description="初步诊断")

# 创建带结构化输出的链
structured_chain = llm_openai.with_structured_output(PatientInfo)

# 使用
medical_text = "患者张三，45岁，主诉头痛、发热三天，体温38.2℃"
result = structured_chain.invoke(f"从以下文本提取患者信息：{medical_text}")
print(f"姓名: {result.name}, 诊断: {result.diagnosis}")
三、最佳实践
1. 性能优化技巧
# 流式响应
for chunk in qa_chain.stream({"query": "解释深度学习原理"}):
    print(chunk["result"], end="", flush=True)

# 批量处理
questions = ["什么是机器学习？", "监督学习和无监督学习的区别？"]
results = qa_chain.batch([{"query": q} for q in questions])

# 异步调用
async def run_chain():
    return await qa_chain.ainvoke({"query": "AI的未来发展趋势"})
2. 错误处理
from langchain_core.runnables import RunnableLambda

def handle_errors(input_dict):
    try:
        # 正常执行
        return qa_chain.invoke(input_dict)
    except Exception as e:
        # 错误处理
        return {"result": f"处理请求时出错: {str(e)}"}

safe_chain = RunnableLambda(handle_errors)
3. 部署与监控
# 使用LangServe部署API
from langchain.chains import RetrievalQA
from langserve import add_routes

app = FastAPI()
chain = RetrievalQA.from_chain_type(...)
add_routes(app, chain, path="/qa")

# 集成监控
from langchain.callbacks import LangChainTracer

tracer = LangChainTracer()
chain = RetrievalQA.from_chain_type(..., callbacks=[tracer])
四、实际应用场景
企业知识库问答
qa_system = RetrievalQA.from_chain_type(
    llm=llm_openai,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    chain_type="map_reduce"
)
自动化数据分析
agent = create_pandas_dataframe_agent(
    llm=llm_openai,
    df=pd.read_csv("sales_data.csv"),
    verbose=True
)
agent.run("分析第四季度销售趋势")
智能文档处理
from langchain.chains import AnalyzeDocumentChain
from langchain.chains.summarize import load_summarize_chain

summary_chain = load_summarize_chain(llm_openai, chain_type="map_reduce")
summarize_document = AnalyzeDocumentChain(combine_docs_chain=summary_chain)
summarize_document.run("大型法律文档.pdf")
LangChain 的强大之处在于这些模块可以灵活组合，构建从简单问答到复杂工作流的各种应用。
5.LangChain基本使用
● 安装并配置
● 实现第一个程序
● 使用PromptTemplate+LLMChain生成多主题文案
一、安装
pip install langchain
pip install openai
二、API KEY 配置
想要调用大模型，我们需要使用大模型对应的API KEY才能进行大模型的使用，例如要使用DeepSeek，需要去官网获取API KEY，并且要设置环境变量
1. 前往官网 https://www.deepseek.com/，点击API开放平台

2. 点击API keys，点击创建API KEY

3. 输入名称，点击创建

4. 创建成功后点击复制并使用

5. 可以使用配置环境变量来配置API KEY，也可以直接填写
OPENAI_API_KEY = "刚才复制的API KEY"
# 或者使用
os.environ["DeepSeek_API_KEY"] = "刚才复制的API KEY"
三、代码实现
# 导入需要的组件
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

OPENAI_API_KEY = ("你的API KEY")

# 模型初始化
model = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=OPENAI_API_KEY,
    base_url="https://api.deepseek.com/v1",  # 关键配置：指定 DeepSeek 的 API 端点
    streaming=False,  # 是否使用流式响应
    temperature=0.7  # 控制生成文本的随机性
)

# 创建处理链
prompt = ChatPromptTemplate.from_template("请根据下面的主题写一篇小红书营销的短文：{topic}")
output_parser = StrOutputParser()
chain = prompt | model | output_parser

# 调用链
result = chain.invoke({"topic": "康师傅绿茶"})
print(result)
运行结果如下：

四、多步链实现
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

OPENAI_API_KEY = ("sk-07955e2855364096933f0077e015996c")

# 模型初始化
model = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=OPENAI_API_KEY,
    base_url="https://api.deepseek.com/v1",  # 关键配置：指定 DeepSeek 的 API 端点
    streaming=False,  # 是否使用流式响应
    temperature=0.7  # 控制生成文本的随机性
)

# 创建输出解析器
output_parser = StrOutputParser()

# 创建提示词模板
extract_prompt = ChatPromptTemplate.from_template("请你从以下句子中提取关键词，用不超过五个字概括：{user_input}")
# 创建链，有一定顺序
extract_chain = extract_prompt | model | output_parser

# 创建第二条写作链
write_prompt = ChatPromptTemplate.from_template("请你以{keywords}为主题是写一篇短文")

# 链接多链
full_chain = RunnableParallel(
    keywords=extract_chain,  # 提取关键词
    user_input=RunnablePassthrough()  # 保留原始输入
) | write_prompt | model | output_parser  # 连接写作链

# 调用多步链
result = full_chain.invoke("故宫博物院收藏了大量明清时期的珍贵文物")
print("最终结果:", result)

# 测试各步骤输出（可选）
# test_input = "故宫博物院收藏了大量明清时期的珍贵文物"
# keywords = extract_chain.invoke(test_input)
# print("提取的关键词:", keywords)
# print("写作模板结果:", write_prompt.format(keywords=keywords, user_input=test_input))
五、上下文记忆
什么是上下文记忆？为什么需要它？
上下文记忆（Message Memory） 是对话系统中的核心机制，用于存储和利用历史对话信息，使AI能够理解上下文关系，实现连贯的多轮对话。使用上下文记忆可以实现
● 上下文理解：理解当前问题与历史对话的关系
● 连贯性维护：保持对话的连续性和一致性
● 个性化交互：基于用户历史提供个性化响应
● 状态管理：跟踪复杂任务的处理进度
● 减少重复：避免重复询问相同信息
LCEL实现消息记忆的核心组件
1. 记忆存储组件
from langchain.memory import (
 ConversationBufferMemory,  # 基础记忆
 ConversationSummaryMemory,  # 摘要记忆
 ConversationEntityMemory,   # 实体记忆
 CombinedMemory              # 组合记忆
)
2. 记忆感知链
from langchain_core.runnables import RunnableWithMessageHistory
3. 记忆存储后端
from langchain.storage import (
 LocalFileStore,        # 本地文件存储
 RedisStore,            # Redis存储
 SQLiteStore            # SQLite存储
)
带有基础记忆的对话链
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableWithMessageHistory

# 1. 初始化模型
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# 2. 创建提示模板（包含记忆占位符）
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的AI助手，当前时间是{time}"),
    MessagesPlaceholder(variable_name="history"),  # 记忆占位符
    ("human", "{input}") 
])

# 3. 创建记忆存储
memory = ConversationBufferMemory(
    return_messages=True,  # 返回消息对象而非字符串
    memory_key="history"   # 与提示中的变量名匹配
)

# 4. 创建基础链
chain = RunnablePassthrough.assign(
    time=lambda _: datetime.now().strftime("%Y-%m-%d %H:%M")
) | prompt | model

# 5. 添加记忆功能
memory_chain = RunnableWithMessageHistory(
    chain,
    lambda session_id: memory,  # 记忆获取函数
    input_messages_key="input",  # 输入消息键
    history_messages_key="history"  # 历史消息键
)

# 6. 使用链（指定会话ID）
result = memory_chain.invoke(
    {"input": "你好，我叫张三"},
    config={"configurable": {"session_id": "user_123"}}
)
print(result.content)

# 后续对话（自动包含历史）
result = memory_chain.invoke(
    {"input": "你还记得我叫什么吗？"},
    config={"configurable": {"session_id": "user_123"}}
)
print(result.content)  # 输出：当然记得，你叫张三！
六、few shot（少量示例）和 example selectors（示例选择器）
基本概念
Few-shot Prompting（少量示例提示）
Few-shot 是通过在 prompt 中提供一小部分示例，来帮助大模型理解任务，从而更好地完成生成任务。
比如你想让模型完成“中译英”的任务，加入几个“中->英”的示例能显著提升表现。
ExampleSelector（示例选择器）
是 LangChain 提供的一类工具，用于动态地从示例集中选出最相关的一些示例用于 few-shot prompting。LangChain 常见的选择器包括：
● SemanticSimilarityExampleSelector: 基于向量相似度选择示例
● LengthBasedExampleSelector: 基于示例长度选择
● MaxMarginalRelevanceExampleSelector: 保证信息多样性和相关性
代码演示：few-shot + 示例选择器
示例任务：中译英（中文翻译为英文）
安装依赖（如未安装）
pip install langchain openai tiktoken faiss-cpu
FewShotPromptTemplate + 静态示例
from langchain.prompts import PromptTemplate, FewShotPromptTemplate

# 准备几个示例
examples = [
    {"input": "你好", "output": "Hello"},
    {"input": "今天天气真好", "output": "The weather is nice today"},
]

# 单个示例的模板格式
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="输入: {input}\n输出: {output}"
)

# FewShotPromptTemplate 将示例 + 用户输入组合为完整 prompt
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,# 直接传入examples
    example_prompt=example_prompt,
    prefix="请将以下中文翻译为英文：",
    suffix="输入: {input}\n输出:",  # 用户输入填充处
    input_variables=["input"]
)

print(few_shot_prompt.format(input="我爱编程"))
加入 ExampleSelector：动态选择示例
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_huggingface import HuggingFaceEmbeddings

# 轻量级embedding模型
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 示例样本
examples = [
    {"input": "你好", "output": "Hello"},
    {"input": "今天天气真好", "output": "The weather is nice today"},
    {"input": "今天食物不错", "output": "The food is nice today"},
    {"input": "你多大了", "output": "How old are you"},
    {"input": "这是一个晴朗的早上", "output": "This is a nica day"},
]

# 示例格式模板
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="输入: {input}\n输出: {output}"
)

# 示例选择器（语义相似度）
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=examples,# 在示例选择器中传入所有examples
    embeddings=embedding_model,# 嵌入模型
    vectorstore_cls=FAISS,# 向量数据库
    k=2# 选择最相近的2个示例
)

# 构建 few-shot prompt（用 example_selector）
dynamic_few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,# 使用的是示例选择器
    example_prompt=example_prompt,
    suffix="输入: {input}\n输出:",
    input_variables=["input"]
)

# 测试输入
print(dynamic_few_shot_prompt.format(input="今天"))

小结
Few-Shot 和 Example Selector 主要用于提升 LLM 在问答、对话、信息提取等任务中的准确性和上下文适应能力。  
功能	模块	说明
Few-Shot Prompt	FewShotPromptTemplate	构造包含多个示例的 Prompt 模板
静态示例	examples=[...]	示例写死在模板中
动态示例选择	example_selector=...	根据输入动态选择相关示例
示例格式	PromptTemplate	每个示例的格式化模板
相似度选择器	SemanticSimilarityExampleSelector	基于语义相似度从示例集中选择
应用场景举例
✅ 场景一：智能客服 / 问答系统
  ○ Few-shot prompt 提供类似问题的示例（如“退款流程怎么走？”→“请点击订单详情并申请退款…”）
  ○ Example Selector 动态选择语义最相关的问题示例来增强 prompt，让模型给出更精准回复。
  ○ 实际好处：不用训练新模型，仅用 prompt 就能适配各种用户提问风格。
✅ 场景二：企业知识库问答 / 文档解析
  ○ 企业文档如操作手册、政策说明等结构复杂。
  ○ 使用 few-shot 提供“问题-回答”样例，结合示例选择器动态选取相关场景。
  ○ 比如输入：“如何申请年假？” → 选择最接近“休假”话题的 QA 示例，用于指导生成回答。
✅ 场景三：信息抽取任务（IE）
  ○ 比如从合同、发票中抽取字段：发票号、总金额、签署方等。
  ○ few-shot 提供字段抽取的输入输出示例，让模型学会抽取结构化数据。
  ○ 示例选择器根据合同类型自动匹配类似示例。
✅ 场景四：代码生成 / 自动补全
  ○ 提供类似代码片段或模板，让模型更好理解上下文语义。
  ○ Selector 根据用户当前输入代码片段选择历史相似的例子。
为什么不直接写死例子？
静态 few-shot 示例在实际中容易遇到问题：
  ○ 每个用户输入风格不同
  ○ 企业语境/部门/业务变化多
  ○ 示例写得多了 prompt 太长，太少又泛化不足
示例选择器的优势：动态、上下文相关、避免无效示例污染生成质量。
七、LangServer
LangServe 是一个由 LangChain 团队开发的 Python 库，专门用于快速部署 LangChain 应用作为 RESTful API 服务。它的核心目标是简化 LangChain 链（Chains）、智能体（Agents）或其他组件在生产环境中的部署过程，让开发者无需从头搭建复杂的 Web 服务框架（如 Flask/FastAPI）即可轻松发布 API。
核心功能与特点：
1. 无缝集成 LangChain
直接将已有的 LangChain 对象（如 Chain、Agent、Runnable）包装成 API 端点，无需重写逻辑。
2. 自动生成 API 文档
基于 OpenAPI 规范自动生成交互式文档（Swagger UI），方便测试和集成。
3. 支持异步和流式响应
适用于需要实时输出的场景（如逐词返回大模型生成结果）。
4. 输入/输出模型验证
自动校验请求数据的格式，确保符合链的输入要求。
5. 轻量级与易扩展
基于 FastAPI 构建，支持中间件、认证等高级功能。
典型使用场景：
● 将本地开发的 LangChain 应用（如问答系统、文本分析工具）快速发布为 Web API。
● 为前端应用（Web/移动端）提供大模型能力后端支持。
● 构建微服务架构的 AI 应用生态系统。
项目创建过程
LangServe安装
使用pip命令进行安装
pip install "langserve[all]"
要快速启动Langserve，需要安装langchain-cli
pip install -U langchain-cli
安装完成后在终端输入langchain，出现以下页面表示安装成功

项目创建
1. 使用langchain cli命令创建新应用(my-app为创建项目的名字)
langchain app new my-app
创建后项目目录结构如下：

2. 在add_routes中定义可运行的对象，转到server.py进行编辑
# Edit this to add the chain you want to add
add_routes(app, NotImplemented)
3. 使用poetry 用于添加第三方包（例如 langchain-openai、langchain-anthropic、langchain-mistral 等）
#安装pipx
pip install pipx

#加入到环境变量，需要重启PyCharm
pipx ensurepath

#安装poetry
pipx install poetry

#安装 langchain-openai 库，例如：poetry add [package-name]
poetry add langchain
poetry add langchain-openai
安装langchain时候会遇到如下错误

这是由于版本匹配问题，目前我们安装的是V0.3 Langchain框架，在使用poetry进行第三方包安装时候会出现版本问题，解决办法，将项目文件pyproject.toml文件中的pydantic = "<2"修改为pydantic = “>=2.7.4"

安装成功后pyproject.toml文件中会出现如下所示代码

4. 设置环境变量
export OPENAI_API_KEY="sk-..."
5. 启动项目
poetry run langchain serve --port=8100
八、多模态集成开发
LangChain 多模态实现目标
我们这里以图文问答为例：
● 输入：图片 + 用户问题
● 输出：基于图片内容的自然语言回答
多模态的关键在于：
● 图像向量化（使用模型如 BLIP2 / CLIP / MiniGPT-4）
● 文本提问嵌入
● 多模态融合模型执行推理
实现代码：图文问答链
步骤说明
1. 读取图片，传入视觉模型（如 BLIP2）获取图像描述
2. 构建文本提示模板
3. 用qwen-vl-plus模型生成答案
完整示例代码：
import base64  # 用于将图片编码为 base64 格式
import requests  # 用于发送 HTTP 请求调用 DeepSeek 接口
from PIL import Image  # 用于读取和处理图片
from io import BytesIO  # 用于创建内存缓冲区
from langchain_core.runnables import RunnableLambda  # LangChain 中用于链式封装的工具

DASHSCOPE_API_KEY = "你的api key" # 通义千问模型
DASHSCOPE_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

# ✅ 将图片转为 base64 格式，供 API 使用
def image_to_base64(image_path: str) -> str:
    # 打开图像，并转换为 RGB 模式（去除透明通道，统一格式）
    img = Image.open(image_path).convert("RGB")
    # 创建内存缓冲区，用于存储图像二进制数据
    buffered = BytesIO()
    # 将图像保存为 JPEG 格式写入内存
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    # 将图像数据编码为 base64 字符串并返回
    return f"data:image/jpeg;base64,{img_str}"  # 必须包含MIME前缀

def ask_qwen_multimodal(inputs: dict) -> str:
    """调用通义千问多模态模型处理图文问答"""
    image_source = inputs["image_url"]
    question = inputs["question"]

    # 处理图片源：本地路径转base64，URL直接使用
    if image_source.startswith(("http://", "https://")):
        image_data = image_source  # 网络图片直接使用URL
    else:
        image_data = image_to_base64(image_source)  # 本地图片转base64

    # 设置请求头
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }

    # 构造通义千问API请求体（符合DashScope规范）
    payload = {
        "model": "qwen-vl-plus",  # 可选：qwen-vl-plus/qwen-vl-max
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"image": image_data},  # 图片数据
                        {"text": question}  # 问题文本
                    ]
                }
            ]
        },
        "parameters": {
            "temperature": 0.6,
            "top_p": 0.8
        }
    }

    # 发送API请求
    response = requests.post(DASHSCOPE_API_URL, headers=headers, json=payload)

    # 错误处理
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        error_detail = response.json().get("message", "未知错误")
        raise Exception(f"API请求失败 [{response.status_code}]: {error_detail}") from e

    # 解析响应（通义千问返回结构）
    response_data = response.json()
    return response_data["output"]["choices"][0]["message"]["content"]


# ✅ 使用 LangChain 的 Runnable 封装成链式调用单元
multimodal_chain = RunnableLambda(ask_qwen_multimodal)

# ✅ 测试用例（支持本地路径和URL）
if __name__ == "__main__":
    # 测试配置（根据需要修改）
    test_inputs = {
        "image_url": "https://gitee.com/ACERT6/langchain/raw/master/true.jpg",
        "question": "图片中有什么动物？它们在做什么？"
    }

    # 执行调用
    try:
        result = multimodal_chain.invoke(test_inputs)
        print("\n✅ 通义千问回答：")
        print("-" * 50)
        print(result)
        print("-" * 50)
    except Exception as e:
        print(f"\n❌ 调用失败: {str(e)}")

总结
项目	描述
多模态核心	图像 + 文本融合建模
LangChain 作用	提供链式调用与结构化逻辑处理
LECL 亮点	极大提升多模态处理链的组合性和可读性
九、工具
LCEL 工具使用指南
LCEL 允许通过管道符 | 声明式组合链式组件，简化复杂工作流的构建。
1. 基础工具调用
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 定义自定义工具
@tool
def calculate_length(text: str) -> int:
    """计算输入字符串的长度"""
    return len(text)

# 创建 LCEL 链
chain = (
    ChatPromptTemplate.from_template("分析内容: {input}") 
    | ChatOpenAI(model="gpt-3.5-turbo") 
    | calculate_length  # 直接调用工具
)

result = chain.invoke({"input": "Hello, world!"})
print(result)  # 输出: 13
2. 动态工具选择 (Tool Routing)
使用 bind_tools 让模型智能选择工具：
from langchain_core.utils.function_calling import convert_to_openai_tool

# 定义多个工具
tools = [calculate_length, ...]  # 添加其他工具

# 创建支持工具调用的模型
model = ChatOpenAI().bind_tools(tools)

# 构建决策链
chain = (
    {"input": lambda x: x["input"]} 
    | ChatPromptTemplate.from_template("处理请求: {input}") 
    | model
    | (lambda msg: tool_map[msg.tool_calls[0]['name']](msg.tool_calls[0]['args']))
)
3. 组合工具流
from langchain_core.runnables import RunnablePassthrough

# 定义工具预处理链
preprocess = (
    RunnablePassthrough.assign(
        cleaned_input=lambda x: x["input"].strip().lower()
    ) 
    | {"data": RunnablePassthrough()}
)

# 完整工作流
full_chain = (
    preprocess 
    | calculate_length 
    | {"original": RunnablePassthrough(), "length": RunnablePassthrough()}
)

full_chain.invoke({"input": "  HELLO  "}) 
# 输出: {'original': '  HELLO  ', 'length': 5}
4. 错误处理
from langchain_core.runnables import RunnableLambda

def safe_tool_call(args):
    try:
        return calculate_length(args)
    except Exception as e:
        return f"Tool error: {str(e)}"

resilient_chain = (
    RunnablePassthrough() 
    | RunnableLambda(safe_tool_call)
)
5. 流式输出
# 流式返回工具结果
async for chunk in chain.astream({"input": "Stream me!"}):
    print(chunk, end="", flush=True)
关键技巧
1. 工具绑定
model_with_tools = ChatOpenAI().bind_tools([tool1, tool2])
2. 参数解析
from langchain_core.output_parsers import JsonOutputParser
parser = JsonOutputParser()
chain = model | parser
3. 调试模式
chain = ... | calculate_length
chain.get_graph().print_ascii()  # 打印拓扑图
4. 工具组合
multi_tool_chain = tool1 | tool2  # 前一个工具输出作为后一个输入
最佳实践
● 命名规范：工具函数名需清晰（如 get_weather_data）
● 类型提示：严格定义工具输入类型（如 text: str）
● 错误处理：在工具内部捕获异常，返回结构化错误信息
● 性能优化：对耗时工具使用 RunnableLambda 异步化
官方资源：  
● LCEL 文档  
6.常见函数使用
ChatPromptTemplate和PromptTemplate
在LangChain框架中，ChatPromptTemplate和PromptTemplate都是用于构建提示词（prompt）的工具，但它们的设计目标和使用场景有显著区别：
 PromptTemplate
● 用途：
为非聊天型模型（如OpenAI的text-davinci-003）生成单字符串提示。
这类模型接收一个完整的文本字符串作为输入（例如："问题：什么是AI？"）。
● 输出格式：
返回一个字符串（str）。
● 示例：
from langchain.prompts import PromptTemplate

template = "解释以下概念：{concept}"
prompt = PromptTemplate.from_template(template)
formatted_prompt = prompt.format(concept="机器学习")
# 输出: "解释以下概念：机器学习"

# 实质上是一个字符串的拼接
ChatPromptTemplate
● 用途：
为聊天型模型（如OpenAI的gpt-3.5-turbo、gpt-4）生成结构化消息列表。
这类模型需要输入一个包含角色（role）和内容（content）的消息序列（例如：[{"role": "system", "content": "..."}, ...]）。
● 输出格式：
返回一个消息对象列表（List[BaseMessage]），例如SystemMessage、HumanMessage等。
● 核心特点：
支持多角色消息（如系统指令、用户输入、AI回复历史），适合多轮对话场景。
● 示例：
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("你是一个AI助手，用简单语言回答问题。"),
    HumanMessagePromptTemplate.from_template("解释：{concept}")
])

messages = template.format_messages(concept="神经网络")
# 输出示例:
# [
#   SystemMessage(content="你是一个AI助手，用简单语言回答问题。"),
#   HumanMessage(content="解释：神经网络")
# ]
关键区别总结
特性	PromptTemplate	ChatPromptTemplate
适用模型	非聊天模型（文本补全）	聊天模型（对话式）
输出格式	字符串 (str)	消息对象列表 (List[BaseMessage])
是否支持多角色消息	❌ 单字符串	✅ 支持系统、用户、AI等多种角色消息
典型使用场景	简单问答、文本生成	多轮对话、带上下文的交互
选择建议
● 如果使用 传统文本补全模型（如text-davinci-003）→ 用 PromptTemplate。
● 如果使用 聊天模型（如gpt-3.5-turbo, gpt-4）→ 用 ChatPromptTemplate 构建消息序列。
核心方法对比
from_template() 方法
特性	PromptTemplate	ChatPromptTemplate
作用	从单个字符串模板创建实例	从单个字符串模板创建实例
输入	字符串模板（可包含变量占位符）	字符串模板（可包含变量占位符）
输出类型	PromptTemplate 实例	ChatPromptTemplate 实例
内部结构	单字符串模板	自动包装成用户消息（HumanMessage）
使用场景	简单单轮提示	仅需用户消息的简单聊天场景
示例	PromptTemplate.from_template("解释 {concept}")	ChatPromptTemplate.from_template("解释 {concept}")
# 等价于：
[HumanMessagePromptTemplate("解释 {concept}")]
关键区别：
虽然方法名相同，但 ChatPromptTemplate.from_template() 会将字符串自动转换为 HumanMessagePromptTemplate，而 PromptTemplate 保持原始字符串结构。
from_messages() 方法
特性	ChatPromptTemplate
作用	从消息组件列表创建聊天提示模板
输入	消息组件列表（支持多种格式）
输出	ChatPromptTemplate 实例
核心优势	支持多角色、结构化消息组合
消息格式	支持三种格式：
1. (role, template) 元组
2. MessagePromptTemplate 实例
3. BaseMessage 实例
示例	ChatPromptTemplate.from_messages(
[("system", "你是一个{expert_type}专家"),
MessagesPlaceholder("history"),
("human", "{user_input}")])
format_messages() 方法
特性	PromptTemplate	ChatPromptTemplate
作用	❌ 不存在（但有 format()）	✅ 格式化生成完整消息序列
输出类型	-	List[BaseMessage]
等价操作	format() → 返回字符串	格式化后生成结构化消息对象列表
输出示例	-	[SystemMessage(content="..."),
HumanMessage(content="...")]
输入要求	-	需提供模板中定义的所有变量
使用场景	-	准备直接输入聊天模型的最终提示
方法关联图


详细使用示例
from_template() 对比
# PromptTemplate
pt_template = PromptTemplate.from_template("解释 {topic}")
print(type(pt_template))  # <class 'langchain.prompts.PromptTemplate'>
print(pt_template.input_variables)  # ['topic']

# ChatPromptTemplate
chatpt_template = ChatPromptTemplate.from_template("解释 {topic}")
print(type(chatpt_template))  # <class 'langchain.prompts.ChatPromptTemplate'>
print(chatpt_template.input_variables)  # ['topic']
from_messages() 专属方法
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate

# 多角色消息模板
chat_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("你是{role}领域的专家"),
    HumanMessagePromptTemplate.from_template("用简单语言解释：{concept}"),
    AIMessagePromptTemplate.from_template("好的，我将这样解释：{draft}"),
    ("human", "请基于以上完善对{concept}的解释")
])

print(chat_template.input_variables) 
# ['role', 'concept', 'draft'] 自动提取所有变量
format_messages() vs format()
# PromptTemplate 只有 format()
pt = PromptTemplate.from_template("你好，{name}！")
result_pt = pt.format(name="小明")
print(type(result_pt))  # <class 'str'>
print(result_pt)  # "你好，小明！"

# ChatPromptTemplate 两者都有
chatpt = ChatPromptTemplate.from_template("你好，{name}！")
result_chat_str = chatpt.format(name="小红")
print(result_chat_str)  # "Human: 你好，小红！" (字符串表示)

result_chat_msgs = chatpt.format_messages(name="小红")
print(type(result_chat_msgs))  # <class 'list'>
print(result_chat_msgs)  
# [HumanMessage(content='你好，小红！')]
使用场景决策树
是否需要与聊天模型交互？
├─ 否 → 使用 PromptTemplate
│   ├─ 简单提示 → from_template()
│   └─ 格式化输出 → format()
│
└─ 是 → 使用 ChatPromptTemplate
    ├─ 单条用户消息 → from_template()
    ├─ 复杂对话结构 → from_messages()
    └─ 准备模型输入 → format_messages()
关键总结
1. from_template()  
  ○ 两者接口相似但实现不同
  ○ ChatPromptTemplate 会将输入自动转换为用户消息
2. from_messages()  
  ○ ChatPromptTemplate 专属方法
  ○ 支持构建多角色对话结构
  ○ 支持三种消息组件格式
3. format_messages()  
  ○ ChatPromptTemplate 的核心输出方法
  ○ 生成可直接输入聊天模型的结构化消息
  ○ 与 PromptTemplate.format() 有本质区别（对象列表 vs 字符串）
RunnableWithMessageHistory
RunnableWithMessageHistory 是 LangChain 中用于管理对话历史的核心组件，它允许你将对话历史无缝集成到 Runnable 链中。源码描述如下：
 """Runnable that manages chat message history for another Runnable.

    A chat message history is a sequence of messages that represent a conversation.

    RunnableWithMessageHistory wraps another Runnable and manages the chat message
    history for it; it is responsible for reading and updating the chat message
    history.

    The formats supported for the inputs and outputs of the wrapped Runnable
    are described below.

    RunnableWithMessageHistory must always be called with a config that contains
    the appropriate parameters for the chat message history factory.

    By default, the Runnable is expected to take a single configuration parameter
    called `session_id` which is a string. This parameter is used to create a new
    or look up an existing chat message history that matches the given session_id.

    In this case, the invocation would look like this:

    `with_history.invoke(..., config={"configurable": {"session_id": "bar"}})`
    ; e.g., ``{"configurable": {"session_id": "<SESSION_ID>"}}``.

    The configuration can be customized by passing in a list of
    ``ConfigurableFieldSpec`` objects to the ``history_factory_config`` parameter (see
    example below).

    In the examples, we will use a chat message history with an in-memory
    implementation to make it easy to experiment and see the results.

    For production use cases, you will want to use a persistent implementation
    of chat message history, such as ``RedisChatMessageHistory``.

    Parameters:
        get_session_history: Function that returns a new BaseChatMessageHistory.
            This function should either take a single positional argument
            `session_id` of type string and return a corresponding
            chat message history instance.
        input_messages_key: Must be specified if the base runnable accepts a dict
            as input. The key in the input dict that contains the messages.
        output_messages_key: Must be specified if the base Runnable returns a dict
            as output. The key in the output dict that contains the messages.
        history_messages_key: Must be specified if the base runnable accepts a dict
            as input and expects a separate key for historical messages.
        history_factory_config: Configure fields that should be passed to the
            chat history factory. See ``ConfigurableFieldSpec`` for more details.

    Example: Chat message history with an in-memory implementation for testing.

    .. code-block:: python

        from operator import itemgetter

        from langchain_openai.chat_models import ChatOpenAI

        from langchain_core.chat_history import BaseChatMessageHistory
        from langchain_core.documents import Document
        from langchain_core.messages import BaseMessage, AIMessage
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from pydantic import BaseModel, Field
        from langchain_core.runnables import (
            RunnableLambda,
            ConfigurableFieldSpec,
            RunnablePassthrough,
        )
        from langchain_core.runnables.history import RunnableWithMessageHistory


        class InMemoryHistory(BaseChatMessageHistory, BaseModel):
            \"\"\"In memory implementation of chat message history.\"\"\"

            messages: list[BaseMessage] = Field(default_factory=list)

            def add_messages(self, messages: list[BaseMessage]) -> None:
                \"\"\"Add a list of messages to the store\"\"\"
                self.messages.extend(messages)

            def clear(self) -> None:
                self.messages = []

        # Here we use a global variable to store the chat message history.
        # This will make it easier to inspect it to see the underlying results.
        store = {}

        def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
            if session_id not in store:
                store[session_id] = InMemoryHistory()
            return store[session_id]


        history = get_by_session_id("1")
        history.add_message(AIMessage(content="hello"))
        print(store)  # noqa: T201


    Example where the wrapped Runnable takes a dictionary input:

        .. code-block:: python

            from typing import Optional

            from langchain_community.chat_models import ChatAnthropic
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
            from langchain_core.runnables.history import RunnableWithMessageHistory


            prompt = ChatPromptTemplate.from_messages([
                ("system", "You're an assistant who's good at {ability}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ])

            chain = prompt | ChatAnthropic(model="claude-2")

            chain_with_history = RunnableWithMessageHistory(
                chain,
                # Uses the get_by_session_id function defined in the example
                # above.
                get_by_session_id,
                input_messages_key="question",
                history_messages_key="history",
            )

            print(chain_with_history.invoke(  # noqa: T201
                {"ability": "math", "question": "What does cosine mean?"},
                config={"configurable": {"session_id": "foo"}}
            ))

            # Uses the store defined in the example above.
            print(store)  # noqa: T201

            print(chain_with_history.invoke(  # noqa: T201
                {"ability": "math", "question": "What's its inverse"},
                config={"configurable": {"session_id": "foo"}}
            ))

            print(store)  # noqa: T201


    Example where the session factory takes two keys, user_id and conversation id):

        .. code-block:: python

            store = {}

            def get_session_history(
                user_id: str, conversation_id: str
            ) -> BaseChatMessageHistory:
                if (user_id, conversation_id) not in store:
                    store[(user_id, conversation_id)] = InMemoryHistory()
                return store[(user_id, conversation_id)]

            prompt = ChatPromptTemplate.from_messages([
                ("system", "You're an assistant who's good at {ability}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ])

            chain = prompt | ChatAnthropic(model="claude-2")

            with_message_history = RunnableWithMessageHistory(
                chain,
                get_session_history=get_session_history,
                input_messages_key="question",
                history_messages_key="history",
                history_factory_config=[
                    ConfigurableFieldSpec(
                        id="user_id",
                        annotation=str,
                        name="User ID",
                        description="Unique identifier for the user.",
                        default="",
                        is_shared=True,
                    ),
                    ConfigurableFieldSpec(
                        id="conversation_id",
                        annotation=str,
                        name="Conversation ID",
                        description="Unique identifier for the conversation.",
                        default="",
                        is_shared=True,
                    ),
                ],
            )

            with_message_history.invoke(
                {"ability": "math", "question": "What does cosine mean?"},
                config={"configurable": {"user_id": "123", "conversation_id": "1"}}
            )

    """
以下是对其功能和使用方法的详细解析：
核心功能
1. 对话历史管理
自动处理聊天消息的存储和检索，将历史对话注入到每次请求中
2. 会话隔离
通过 session_id 区分不同对话上下文
3. 灵活配置
支持自定义历史存储方式和输入/输出格式
关键参数解析
参数	类型	说明	必填
get_session_history	Callable[[str], BaseChatMessageHistory]	历史记录工厂函数	✅
input_messages_key	str	输入字典中用户消息的键名	字典输入时需指定
output_messages_key	str	输出字典中AI消息的键名	字典输出时需指定
history_messages_key	str	输入字典中历史消息的键名	使用历史变量时需指定
history_factory_config	List[ConfigurableFieldSpec]	自定义会话参数配置	可选
使用流程
定义历史存储实现
from langchain_core.chat_history import InMemoryChatMessageHistory

# 全局存储
store = {}
# 获取历史会话函数
def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    # 如果最开始没有存储过session_id和con_id，会进行初始化存储
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]
构建基础链
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是擅长{ability}的助手"),
    MessagesPlaceholder(variable_name="history"),  # 历史消息占位符
    ("human", "{question}")
])

chain = prompt | ChatOpenAI(model="gpt-3.5-turbo")
包装历史管理器
from langchain_core.runnables import RunnableWithMessageHistory

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_by_session_id,# 获取历史会话函数
    input_messages_key="question",# 输入键
    history_messages_key="history"# 历史键
)
调用时传递会话ID
# 第一次调用
response1 = chain_with_history.invoke(
    {"ability": "数学", "question": "余弦是什么？"},
    config={"configurable": {"session_id": "user123"}}# config配置
)

# 后续调用（自动包含历史）保证和之前session_id一样
response2 = chain_with_history.invoke(
    {"ability": "数学", "question": "它的反函数是什么？"},
    config={"configurable": {"session_id": "user123"}}
)
高级用法：多参数会话
from langchain_core.runnables import ConfigurableFieldSpec

def get_history(user_id: str, conv_id: str):
    key = (user_id, conv_id)
    if key not in store:
        store[key] = InMemoryHistory()
    return store[key]
# ConfigurableFieldSpec用于多个会话参数使用，例如不仅有用户user_id，还有会话的conv_id
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history=get_history,
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="用户ID",
            description="用户唯一标识"
        ),
        ConfigurableFieldSpec(
            id="conv_id",
            annotation=str,
            name="会话ID",
            description="对话唯一标识"
        )
    ]
)

# 调用时传递复合ID
chain_with_history.invoke(
    {"question": "解释勾股定理"},
    config={"configurable": {"user_id": "alice", "conv_id": "math_101"}}
)
工作流程剖析
1. 输入处理  

2. 历史检索
使用 session_id 从工厂函数获取历史存储对象
3. 执行链
将增强后的输入传递给底层 Runnable
4. 输出处理  

生产环境建议
1. 持久化存储
替换内存实现为持久化方案：
from langchain_community.chat_message_histories import RedisChatMessageHistory

def get_redis_history(session_id: str):
    return RedisChatMessageHistory(session_id, url="redis://localhost:6379")
2. 历史记录清理
实现自动过期机制：
# 在历史类中添加TTL逻辑
class ExpiringHistory(RedisChatMessageHistory):
    def __init__(self, ttl=3600, **kwargs):
        super().__init__(**kwargs)
        self.ttl = ttl
    
    def add_messages(self, messages):
        super().add_messages(messages)
        self.redis.expire(self.key, self.ttl)  # 设置过期时间
3. 安全隔离
使用加密 session_id 防止未授权访问
常见问题解决
问题1：历史消息未正确注入
✅ 检查点：
● 确认 history_messages_key 与 Prompt 中的变量名匹配
● 验证工厂函数返回正确的历史对象
问题2：会话数据混淆
✅ 解决方案：
# 确保每次调用使用独立session_id
import uuid
session_id = str(uuid.uuid4())  # 生成唯一ID
问题3：大历史导致性能下降
✅ 优化策略：
# 实现历史截断
def get_truncated_history(session_id: str, max_length=10):
    full_history = get_by_session_id(session_id)
    return full_history[-max_length:]  # 取最近N条
通过 RunnableWithMessageHistory 可以轻松实现：
● 多轮对话上下文保持
● 用户会话隔离
● 历史感知的AI响应
● 可扩展的存储方案
关键设计原则：将历史管理逻辑与业务链解耦，使开发者专注核心功能实现。
StructuredTool 的作用
● StructuredTool 是 LangChain 工具系统的高级实现，相比 Tool，它支持多参数的结构化输入，并且可以用 Pydantic 模型定义参数类型与校验规则。
● 适合需要多个输入字段、严格参数验证的工具（比如天气查询、数据库操作等）。
基本用法
1. 定义输入参数模型
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    city: str = Field(..., description="城市名称")
    date: str = Field(..., description="日期，格式 YYYY-MM-DD")
2.创建 StructuredTool
from langchain.tools import StructuredTool

def get_weather(city: str, date: str) -> str:
    return f"{date} {city} 晴天 26°C"

weather_tool = StructuredTool.from_function(
    func=get_weather,
    name="WeatherTool",
    description="获取指定城市和日期的天气",
    args_schema=WeatherInput
)
3.关键点：
● args_schema：绑定 Pydantic 输入模型
● 输入会自动解析和验证，不需要手动解析字符串
核心方法
方法名	说明	常用参数
StructuredTool.from_function()	从已有函数快速创建结构化工具	func, name, description, args_schema, return_direct
.run(input_dict)	同步执行工具	input_dict: dict，键名需匹配 args_schema
.arun(input_dict)	异步执行工具	同上
.bind(**kwargs)	创建绑定了默认参数的新工具	用于预设参数值
关键参数说明
1.from_function() 的参数
参数名	类型	说明
func	Callable	工具的执行函数
name	str	工具名称，Agent 用它来识别工具
description	str	工具用途描述，模型用它来判断是否调用
args_schema	BaseModel 子类	输入参数结构（Pydantic 模型）
return_direct	bool	是否直接返回结果而不交给 LLM 处理
infer_schema	bool	是否自动从函数签名推断参数模型（可选）
使用示例
1.在 Agent 中使用
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType

llm = ChatOpenAI(temperature=0)

agent = initialize_agent(
    tools=[weather_tool],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

agent.run({"input": "帮我查一下北京 2025-08-12 的天气"})
说明：
● AgentType.OPENAI_FUNCTIONS 或 STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION 是推荐的类型，因为它们原生支持结构化输入。
● Agent 会自动将自然语言解析为结构化参数传给 args_schema。
2.手动调用
result = weather_tool.run({"city": "上海", "date": "2025-08-15"})
print(result)  # 输出：2025-08-15 上海 晴天 26°C
异步版本
import asyncio

async def main():
    res = await weather_tool.arun({"city": "广州", "date": "2025-08-20"})
    print(res)

asyncio.run(main())
最佳实践
1. 始终用 args_schema 明确字段类型与描述，方便 Agent 准确调用。
2. 函数签名与参数模型保持一致，避免调用时出错。
3. 短描述 + 详细字段说明，提高 LLM 工具选择的准确率。
4. 如果有耗时操作（如网络请求），可以同时实现 _arun() 异步方法。
✅ 总结
● StructuredTool 是多参数、强类型的工具封装方式。
● 结合 args_schema 能显著提高工具调用的准确性和鲁棒性。
● 推荐与 AgentType.OPENAI_FUNCTIONS 搭配使用，以充分利用结构化参数解析能力。
