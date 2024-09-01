# %%
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import Chroma
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough # 입력값이 통과되게 해줌

llm = ChatOpenAI(temperature=0.1)

cache_dir = LocalFileStore("./.cache/")

splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n",
    chunk_size = 600,
    chunk_overlap = 100,
) 
loader = UnstructuredFileLoader("./rag_files/chapter_one.docx")
docs = loader.load_and_split(text_splitter=splitter)
embdeeings = OpenAIEmbeddings()

cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    embdeeings, cache_dir
)
# chroma 초기화
vectorstore = Chroma.from_documents(docs, cached_embeddings)

retirver = vectorstore.as_retriever()

prompt = ChatPromptTemplate.from_messages([
    ("system","you are a helapful assistant. Answer questions using only the following context. If you don't know the answer just say you don't know, don't make it up:\n\n{context}"), # Hallucination 줄이기 위해 제한
    ("human", "{question}"),
])

chain = {"context": retirver, "question": RunnablePassthrough() } | prompt | llm

chain.invoke("Describe Victory Mansions.")



