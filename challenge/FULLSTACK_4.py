# 1. Implement a complete RAG pipeline with a Stuff Documents chain.
# 2. You must implement the chain manually.
# 3. Give a ConversationBufferMemory to the chain.
# 4. Use this document to perform RAG: "./rag_files/document.txt"
# 5. Ask the following questions to the chain: Is Aaronson guilty? / What message did he write in the table? / Who is Julia?
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

llm = ChatOpenAI(
    temperature=0.1,
)

# 문서 로드와 쪼개기 / 임베딩 생성 및 캐시
cache_dir = LocalFileStore("./.cache/")

splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n",
    chunk_size=600,
    chunk_overlap=100,
)

loader = TextLoader("./rag_files/document.txt")

docs = loader.load_and_split(text_splitter=splitter)

embeddings = OpenAIEmbeddings()

cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    embeddings,
    cache_dir,
)

vectorstore = FAISS.from_documents(docs, cached_embeddings)

# 메모리와 문서를 이용한 프롬프트
memory = ConversationBufferMemory(
    llm=llm,
    max_token_limit=20,
    return_messages=True,
)


def load_memory(_):
    return memory.load_memory_variables({})["history"]


retriever = vectorstore.as_retriever()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer questions using only the following context. If you don't know the answer just say you don't know, don't make it up:\n\n{context}",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

# 체인 연결 / 체인 호출 함수 정의
chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
        "history": load_memory,
    }
    | prompt
    | llm
)


def invoke_chain(question):
    result = chain.invoke(question)
    print(result)
    memory.save_context({"input": question}, {"output": result.content})

# 체인에 질문하여 테스트
invoke_chain("Is Aaronson guilty?")
invoke_chain("What message did he write in the table?")
invoke_chain("Who is Julia?")

# 메모리를 출력하여 메모리가 체인에 적용되었는지 확인
load_memory({})