import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler

st.set_page_config(
    page_title = "DocumentGPT",
    page_icon = "📜"
)

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    # *args와 **kwargs로 무수히 많은 argument및 keyword argument를 받는다.
    def on_llm_start(self, *args, **kwargs):
        # 우리가 추후로 텍스트로 천천히 채워나갈 메시지 박스
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    # llm이 생성해 내는 모든 새로운 token에 linsten
    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)
        # 새로운 token을 받을때 마다 메세지에 token을 추가한다.
        # self.message = f"{self.message}{token}" 이렇게 써도 됨.       


llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ]
    )
# decorator : fucntion 상단에 넣을 수 있다.
@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator = "\n",
        chunk_size = 600,
        chunk_overlap = 100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store( embeddings,cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def save_message(message, role):
    st.session_state["messages"].append({"message":message,"role":role})

# message Funtion
def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)
        # session_state의 형태 기억하기, message & role

# 히스토리를 그리는 함수 / 이미 저장되어 있는것이기 때문에 저장은 하지 않는다.
def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"],message["role"],save=False)

# document format 지정
def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
            
            Context: {context}
            """),
    ("human","{question}")
])

st.title("DocumentGPT")

st.markdown("""
Welcome!

Use this chatbot to ask questions to an AI about your files!
            
Upload your files on the sidebar.
""")

with st.sidebar:
    # Unstructured file loader를 사용하고 있어서 더 많은 확장자를 넣을 수 있다.
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type = ["txt", "pdf", "docx"],
        )

if file:
    retriever = embed_file(file)
    send_message("I'm Ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anyting about your file ...")
    if message:
        send_message(message, "human")
        chain = {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        } | prompt | llm
        # with 블록 내부에서 chain을 invoke 시키면 
        # CallbackHandler가 st.empty method를 호춣할 때 ai가 하는것처럼 보인다.
        with st.chat_message("ai"):
            response = chain.invoke(message)
else:
    st.session_state["messages"] = []
    # 파일이 없으면 (삭제되면), message 초기화