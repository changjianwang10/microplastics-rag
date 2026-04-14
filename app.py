# -*- coding: utf-8 -*-
"""
微塑料降解机理问答 Web 应用
基于 OpenRouter 免费 API 和 FAISS 本地向量库
"""

# ==================== 方法一：强制设置系统编码环境 ====================
import sys
import locale
import os

# 设置环境变量，强制 Python 使用 UTF-8 编码
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["LC_ALL"] = "C.UTF-8"
os.environ["LANG"] = "C.UTF-8"

# 尝试设置 locale
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except:
    pass

# 重新配置标准输出/错误的编码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ==================== 方法三：修补 pickle 加载时的编码问题 ====================
import pickle

# 保存原始的 pickle.load 函数
if not hasattr(pickle, "_original_load"):
    pickle._original_load = pickle.load

# 定义一个新的 load 函数，强制使用 UTF-8 编码
def _patched_load(file, *, fix_imports=True, encoding="utf-8", errors="strict", buffers=None):
    return pickle._original_load(file, fix_imports=fix_imports, encoding=encoding, errors=errors, buffers=buffers)

# 替换 pickle.load
pickle.load = _patched_load

# ==================== 正常导入其他库 ====================
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 加载本地 .env 文件（仅用于本地调试，Streamlit Cloud 会从 Secrets 注入）
load_dotenv()

# ==================== 配置部分（请根据需要修改） ====================
FAISS_INDEX_DIR = "./faiss_index"                           # FAISS 索引文件夹路径
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")        # 从环境变量读取 API Key

# LLM 模型（OpenRouter 免费模型）
LLM_MODEL = "qwen/qwen3.6-plus-preview:free"
# Embedding 模型（OpenRouter 免费 Embedding）
EMBEDDING_MODEL = "openrouter/free"

# 您的 Streamlit Cloud 应用域名（用于 OpenRouter 的 Referer 头）
APP_URL = "https://microplastics-rag-cjwang.streamlit.app"   # ⚠️ 实际域名
# ===================================================================

# 检查必要的环境变量
if not OPENROUTER_API_KEY:
    st.error("❌ 未检测到 OPENROUTER_API_KEY 环境变量。请在 Streamlit Cloud 的 Secrets 中配置。")
    st.stop()

# ==================== FAISS 向量库加载 ====================
def load_vectorstore():
    """加载 FAISS 向量数据库，并处理可能的中文编码问题"""
    try:
        # 检查索引目录是否存在
        if not Path(FAISS_INDEX_DIR).exists():
            st.error(f"❌ FAISS 索引目录不存在: {FAISS_INDEX_DIR}")
            st.info("请确保已将 faiss_index 文件夹上传到 GitHub 仓库，且路径正确。")
            st.stop()

        # 初始化 Embedding 模型（使用 OpenRouter 接口）
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": APP_URL,
                "X-Title": "Microplastics RAG QA"
            }
        )

        # 加载本地 FAISS 索引（allow_dangerous_deserialization=True 表示信任本地文件）
        vectorstore = FAISS.load_local(
            FAISS_INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vectorstore

    except Exception as e:
        st.error(f"❌ 加载 FAISS 索引失败: {e}")
        st.stop()

# ==================== 文档格式化 ====================
def format_docs(docs):
    """将检索到的文档列表合并为一个上下文字符串"""
    return "\n\n".join(doc.page_content for doc in docs)

# ==================== RAG 链构建 ====================
def get_rag_chain(vectorstore):
    """构建 RAG 问答链"""
    # 初始化 LLM
    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": APP_URL,
            "X-Title": "Microplastics RAG QA"
        },
        temperature=0.1,
        max_tokens=1500
    )

    # 提示词模板（专注降解机理）
    template = """你是一个专业的微塑料降解机理研究助手。请严格根据以下提供的文献上下文，回答关于降解机理的问题。

**重要规则**：
1. 只回答与"降解机理"直接相关的内容，包括：自由基产生路径、活性氧物种作用、聚合物链断裂过程、催化剂电子转移机制、中间产物鉴定、官能团变化等。
2. 如果上下文中没有明确的机理信息，请如实回答"根据已知文献，未提及该条件下的具体降解机理。"
3. 回答必须包含两部分，格式如下：

【机理描述】
（用专业语言清晰概括机理，可分段）

【原文依据】
- 文献：[文件名]
  关键句："[引用原文中支持机理的完整句子]"

上下文信息：
{context}

用户问题：{question}

请回答："""

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # 检索器（返回最相关的5个文档块）
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 构建 LCEL 链
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, retriever

# ==================== Streamlit 界面 ====================
st.set_page_config(page_title="微塑料降解机理问答", page_icon="🔬")
st.title("🔬 微塑料降解机理文献问答")
st.markdown("本助手基于 OpenRouter 免费大模型构建。请提问关于 **降解机理** 的问题。")

# 缓存初始化过程，避免每次提问都重新加载
@st.cache_resource
def init_rag_system():
    vectorstore = load_vectorstore()
    rag_chain, retriever = get_rag_chain(vectorstore)
    return rag_chain, retriever

# 尝试初始化系统
try:
    rag_chain, retriever = init_rag_system()
except Exception as e:
    st.error(f"系统初始化失败: {e}")
    st.stop()

# 用户输入框
user_question = st.text_input("请输入你的问题：", placeholder="例如：光催化条件下 PE 的降解机理是什么？")

if user_question:
    with st.spinner("正在检索文献，生成机理回答..."):
        try:
            # 机理增强检索：如果用户问题中未包含"机理"关键词，自动补全
            if "机理" not in user_question and "机制" not in user_question:
                enhanced_question = f"{user_question} 降解机理"
            else:
                enhanced_question = user_question

            # 执行问答
            answer = rag_chain.invoke(enhanced_question)

            # 单独检索文档用于显示原文依据
            retrieved_docs = retriever.invoke(enhanced_question)

            # 显示回答
            st.subheader("📝 机理回答")
            st.write(answer)

            # 显示原文依据
            if retrieved_docs:
                st.subheader("📚 原文依据")
                sources_dict = {}
                for doc in retrieved_docs:
                    source = doc.metadata.get("source", "未知文献")
                    if source not in sources_dict:
                        sources_dict[source] = []
                    sources_dict[source].append(doc.page_content)

                for source, contents in sources_dict.items():
                    with st.expander(f"📄 {source}"):
                        for i, content in enumerate(contents, 1):
                            st.markdown(f"**片段 {i}:**")
                            st.markdown(f"> {content}")
                            st.divider()
            else:
                st.info("未找到相关原文依据。")

        except Exception as e:
            st.error(f"处理过程中出错：{e}")