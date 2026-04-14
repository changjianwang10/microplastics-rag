# app.py
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# ================= 请确认这里的路径正确 =================
FAISS_INDEX_DIR = r"D:\AOP_microplastics\faiss_index"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# LLM 模型配置 (OpenRouter免费模型)
LLM_MODEL = "qwen/qwen3.6-plus-preview:free" # 100万上下文
# Embedding 模型配置 (OpenRouter免费模型)
EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
# ===================================================

def load_vectorstore():
    """加载 FAISS 向量数据库 (使用 OpenRouter 免费 Embedding 模型)"""
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "Microplastics RAG QA"
        }
    )
    vectorstore = FAISS.load_local(
        FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain(vectorstore):
    # 初始化 OpenRouter 免费 LLM
    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "Microplastics RAG QA"
        },
        temperature=0.1,
        max_tokens=1500
    )

    template = """你是一个专业的微塑料降解机理研究助手。请严格根据以下提供的文献上下文，回答关于降解机理的问题。

**重要规则**：
1. 只回答与"降解机理"直接相关的内容。
2. 如果上下文中没有明确的机理信息，请回答"根据已知文献，未提及该条件下的具体降解机理。"
3. 回答格式：
【机理描述】
...
【原文依据】
- 文献：[文件名]
  关键句："[引用原文句子]"

上下文信息：
{context}

用户问题：{question}

请回答："""

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

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

# ================= Streamlit 界面 =================
st.set_page_config(page_title="微塑料降解机理问答", page_icon="🔬")
st.title("🔬 微塑料降解机理文献问答 (基于OpenRouter)")
st.markdown("本助手基于 OpenRouter API大模型构建。请提问关于 **降解机理** 的问题。")

@st.cache_resource
def init_rag_system():
    vectorstore = load_vectorstore()
    rag_chain, retriever = get_rag_chain(vectorstore)
    return rag_chain, retriever

rag_chain, retriever = init_rag_system()

user_question = st.text_input("请输入你的问题：", placeholder="例如：光催化条件下 PE 的降解机理是什么？")

if user_question:
    with st.spinner("正在检索文献，生成机理回答..."):
        try:
            if "机理" not in user_question and "机制" not in user_question:
                enhanced_question = f"{user_question} 降解机理"
            else:
                enhanced_question = user_question

            answer = rag_chain.invoke(enhanced_question)
            retrieved_docs = retriever.invoke(enhanced_question)

            st.subheader("📝 机理回答")
            st.write(answer)

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