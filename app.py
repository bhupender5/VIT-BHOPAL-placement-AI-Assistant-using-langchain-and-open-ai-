import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()

st.set_page_config(page_title="VIT BHOPAL Placement AI", layout="wide")
st.title("🎓 VIT BHOPAL Placement AI Assistant")

# -------------------------
# Session Memory
# -------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------
# Load Documents
# -------------------------
def load_documents():
    documents = []
    data_folder = "data"

    if not os.path.exists(data_folder):
        return documents

    for file in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file)

        try:
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file.endswith(".txt"):
                loader = TextLoader(file_path)
            else:
                continue

            documents.extend(loader.load())

        except Exception as e:
            st.warning(f"Error loading {file}: {e}")

    return documents

documents = load_documents()

# -------------------------
# FAISS (Persistent)
# -------------------------
embedding = OpenAIEmbeddings()

def create_or_load_vectorstore(docs):
    if os.path.exists("faiss_index"):
        return FAISS.load_local(
            "faiss_index",
            embedding,
            allow_dangerous_deserialization=True
        )

    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embedding)
    vectorstore.save_local("faiss_index")

    return vectorstore

vectorstore = create_or_load_vectorstore(documents)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) if vectorstore else None

# -------------------------
# LLM
# -------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    streaming=True
)

# -------------------------
# Prompt with Chat Memory
# -------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a VIT placement assistant.

First use the provided context to answer.

<context>
{context}
</context>

Rules:
- If answer exists in context, answer from it.
- If not found in context, answer using your general knowledge.
- Do NOT mention context in your answer.
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# -------------------------
# RAG Chain
# -------------------------
if retriever:
    chain = (
        {
            "context": lambda x: format_docs(
                retriever.invoke(x["question"])
            ),
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )
else:
    chain = None
# -------------------------
# UI
# -------------------------
# UI
# -------------------------
from datetime import datetime

user_input = st.chat_input("Ask about placements...")

if user_input and chain:

    # Add user message to memory
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # 🔥 Inject today's date
    today = datetime.today().strftime("%Y-%m-%d")
    modified_question = f"{user_input}. Today's date is {today}."

    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""

        chain_input = {
            "question": modified_question,
            "chat_history": st.session_state.chat_history
        }

        for chunk in chain.stream(chain_input):
            full_response += chunk
            response_container.markdown(full_response)

    # Save assistant response
    st.session_state.chat_history.append(AIMessage(content=full_response))