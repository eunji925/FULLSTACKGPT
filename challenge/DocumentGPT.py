# 1.사용자의 OpenAI API 키 이용하기
#     st.text_input을 사용하여 사용자의 OpenAI API 키를 입력받습니다.
#     입력받은 API 키를 ChatOpenAI와 OpenAIEmbeddings 클래스를 사용할 때 openai_api_key 매개변수로 넘깁니다.
# 2.파일 업로드
#     st.file_uploader를 사용하여 사용자가 파일을 업로드할 수 있도록 합니다.
#     업로드할 수 있는 파일의 확장자는 pdf, txt, docx로 지정합니다.
#     업로드된 파일을 임베딩하고 vectorstore에 저장한 후, 이를 retriever로 변환하여 체인에서 사용합니다.
#     이전과 같은 파일을 선택했을 때 임베딩 과정을 다시 하지 않도록 하기 위해 embed_file 함수에 st.cache_data 데코레이터를 추가하였습니다. (st.cache_data 공식 문서)
# 3.채팅 기록
#     채팅 기록을 저장하기 위해 Session State를 사용합니다.
#     솔루션에서는 st.session_state["messages"]를 리스트로 초기화하고, 메시지를 추가하는 방법으로 구현했습니다. (save_message 함수 참고)
#     저장된 채팅 기록을 페이지에 출력하기 위해 st.session_state["messages"] 리스트에 있는 메시지들을 하나씩 출력합니다. (paint_history 함수 참고)
# 4.결론
#     이전 과제에서 구현한 RAG 파이프라인을 Streamlit을 활용하여 재구현하면서, Streamlit 사용에 익숙해지고 지난 과제 내용을 효과적으로 복습할 수 있었습니다.

import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler
from pathlib import Path

st.set_page_config(
    page_title="Assignment #15",
    page_icon="📜",
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


if "messages" not in st.session_state:
    st.session_state["messages"] = []


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    Path("./.cache/files").mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb+") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(f"{file_path}")
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(
        openai_api_key=openai_api_key,
    )
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON't make anything up

            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)


def main():
    if not openai_api_key:
        return

    llm = ChatOpenAI(
        temperature=0.1,
        streaming=True,
        openai_api_key=openai_api_key,
        callbacks=[
            ChatCallbackHandler(),
        ],
    )

    if file:
        retriever = embed_file(file)

        send_message("I'm ready! Ask away!", "ai", save=False)
        paint_history()
        message = st.chat_input("Ask anything about your file.....")
        if message:
            send_message(message, "human")
            chain = (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                }
                | prompt
                | llm
            )
            with st.chat_message("ai"):
                chain.invoke(message)

    else:
        st.session_state["messages"] = []
        return


st.title("Document GPT")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!

1. Input your OpenAI API Key on the sidebar
2. Upload your file on the sidebar.
3. Ask questions related to the document.
"""
)

with st.sidebar:
    # API Key 입력
    openai_api_key = st.text_input("Input your OpenAI API Key")

    # 파일 선택
    file = st.file_uploader(
        "Upload a. txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

try:
    main()
except Exception as e:
    st.error("Check your OpenAI API Key or File")
    st.write(e)