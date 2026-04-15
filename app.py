# -*- coding: utf-8 -*-
"""
微塑料降解机理问答 Web 应用 (强力编码修复版)
基于 OpenRouter 免费 API 和 FAISS 本地向量库
"""

# ==================== 强力编码环境设置 ====================
import sys
import locale
import os

# Python 3.7+ 强制 UTF-8 模式
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["LC_ALL"] = "C.UTF-8"
os.environ["LANG"] = "C.UTF-8"

try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except:
    pass

# 重配置标准输出/错误
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ==================== Pickle 编码补丁 ====================
import pickle

if not hasattr(pickle, "_original_load"):
    pickle._original_load = pickle.load

def _patched_load(file, *, fix_imports=True, encoding="utf-8", errors="strict", buffers=None):
    return pickle._original_load(file, fix_imports=fix_imports, encoding=encoding, errors=errors, buffers=buffers)

pickle.load = _patched_load

# ==================== 正常导入库 ====================
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# ==================== 配置部分（请修改） ====================
FAISS_INDEX_DIR = "./faiss_index"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
LLM_MODEL = "openrouter/free"
EMBEDDING_MODEL = "text-embedding-v4"

# ⚠️ 请替换为您的实际 Streamlit Cloud 域名
APP_URL = "https://microplastics-rag-cjwang.streamlit.app"
# =========================================================

# 辅助函数：确保字符串是有效的 Unicode
def safe_unicode(s):
    if isinstance(s, bytes):
        return s.decode('utf-8', errors='replace')
    if not isinstance(s, str):
        s = str(s)
    # 移除可能的无效字符（可选）
    return s.encode('utf-8', errors='replace').decode('utf-8')

# 安全写入函数
def safe_write(content):
    return st.write(safe_unicode(content))

# 检查 API Key
if not OPENROUTER_API_KEY:
    st.error("❌ 未检测到 OPENROUTER_API_KEY 环境变量。请在 Streamlit Cloud 的 Secrets 中配置。")
    st.stop()

# ==================== FAISS 加载 ====================
def load_vectorstore():
    try:
        if not Path(FAISS_INDEX_DIR).exists():
            st.error(f"❌ FAISS 索引目录不存在: {FAISS_INDEX_DIR}")
            st.stop()
        embeddings = DashScopeEmbeddings(
            model=EMBEDDING_MODEL,
            dashscope_api_key=DASHSCOPE_API_KEY
            #dimension=1024   # 尝试调整为与 v3 一致的维度
                        
        # embeddings = OpenAIEmbeddings(
        #     model=EMBEDDING_MODEL,
        #     api_key=OPENROUTER_API_KEY,
        #     base_url="https://openrouter.ai/api/v1",
        #     default_headers={
        #         "HTTP-Referer": APP_URL,
        #         "X-Title": "Microplastics RAG QA"
        #     }
        )
        vectorstore = FAISS.load_local(
            FAISS_INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vectorstore
    except Exception as e:
        st.error(f"❌ 加载 FAISS 索引失败: {safe_unicode(e)}")
        st.stop()

def format_docs(docs):
    """安全格式化文档，清洗可能存在的非 UTF-8 字符"""
    contents = []
    for doc in docs:
        content = safe_unicode(doc.page_content)
        contents.append(content)
    return "\n\n".join(contents)

# ==================== RAG 链 ====================
def get_rag_chain(vectorstore):
    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": APP_URL,
            "X-Title": "Microplastics RAG QA"
        },
        temperature=0.2,
        max_tokens=1500,
        streaming=False,   # 关闭流式，避免编码问题
        request_timeout=60
    )

    template = """你是一个专业的高级氧化降解微塑料机理研究助手。请严格按照以下规则分析文献上下文并回答用户问题。

**信息来源过滤规则**：
1. **允许使用的上下文部分**：只采用来自文献中以下标题或类似标题下的内容：Result、Results、Discussion、Conclusion、Analysis、Mechanistic pathway、Mechanism、Proposed mechanism。严禁使用 Introduction、Background、Literature review 等章节的信息。
2. 分清AOPs包括七种技术（光催化、电催化、过氧化氢（芬顿）、过硫酸盐、光电结合、电过硫酸盐结合、高碘酸盐）;塑料类型有七种（PE（HDPE、LDPE）、PP、PET、PLA、PVC、PA66、PS）,要知道这些塑料的全称。要要考虑上下意词，包含关系、同物异词等现象。
3. 如果上下文中不包含上述允许部分的任何内容，或者虽有这些部分但未明确提及降解机理，则你必须基于你自己的知识进行推理和回答。
4. 当基于自身知识回答时，你必须在机理描述末尾明确标注“(注：本部分回答基于模型自身知识推断，未在提供的文献上下文中找到直接依据。)”

**回答格式要求**：
请严格按以下格式输出回答：

【机理描述】
（用专业语言清晰概括降解机理，可分段。如果机理描述来自上下文，分点直接陈述并进行总结；如果基于自身知识，请在段落末尾附上标注。）

【原文依据】
- 文献：[片段出处的文件名]
  关键句：“[引用原文中支持机理的完整句子]”
  （如果本条依据来自允许部分，请在此处列出；如果完全没有原文依据，请写“无直接原文依据，以上机理描述基于模型知识推断。”）

上下文信息：
{context}

用户问题：{question}

请回答："""

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

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
st.markdown("本助手基于 OpenRouter 的大模型API构建。请提问关于 **降解机理** 的问题。")

@st.cache_resource
def init_rag_system():
    vectorstore = load_vectorstore()
    rag_chain, retriever = get_rag_chain(vectorstore)
    return rag_chain, retriever

try:
    rag_chain, retriever = init_rag_system()
except Exception as e:
    st.error(f"系统初始化失败: {safe_unicode(e)}")
    st.stop()

user_question = st.text_input("请输入你的问题：", placeholder="例如：光催化（高级氧化技术）条件下 PE （塑料类型）的降解机理？")

if user_question:
    with st.spinner("正在检索文献，生成机理回答..."):
        try:
            if "机理" not in user_question and "机制" not in user_question:
                enhanced_question = f"{user_question} 降解机理"
            else:
                enhanced_question = user_question

            answer = rag_chain.invoke(enhanced_question)
            retrieved_docs = retriever.invoke(enhanced_question)

            # 显示回答（使用安全输出）
            st.subheader("📝 机理回答")
            safe_write(answer)

            # 显示原文依据
            if retrieved_docs:
                st.subheader("📚 原文依据")
                sources_dict = {}
                for doc in retrieved_docs:
                    source = safe_unicode(doc.metadata.get("source", "未知文献"))
                    content = safe_unicode(doc.page_content)
                    if source not in sources_dict:
                        sources_dict[source] = []
                    sources_dict[source].append(content)

                for source, contents in sources_dict.items():
                    with st.expander(f"📄 {source}"):
                        for i, content in enumerate(contents, 1):
                            st.markdown(f"**片段 {i}:**")
                            st.markdown(f"> {content}")
                            st.divider()
            else:
                st.info("未找到相关原文依据。")

        except Exception as e:
            st.error(f"处理过程中出错：{safe_unicode(e)}")