# (EN)
# 1.Build a SiteGPT version for Cloudflare's documentation.
# 2.The chat bot should be able to answers questions about the documentation of each one of these products:
#     AI Gateway - https://developers.cloudflare.com/ai-gateway/
#     Cloudflare Vectorize - https://developers.cloudflare.com/vectorize/
#     Workers AI - https://developers.cloudflare.com/workers-ai/
# 3.Use the sitemap to find all the documentation pages for each product.
# 4.Your submission will be tested with the following questions:
#     "What is the price per 1M input tokens of the llama-2-7b-chat-fp16 model?"
#     "What can I do with Cloudflare’s AI Gateway?"
#     "How many indexes can a single account have in Vectorize?"
# 5.Allow the user to use its own OpenAI API Key, load it from an st.input inside of st.sidebar
# 6.Using st.sidebar put a link to the Github repo with the code of your Streamlit app.
# (KR)
# 1.Cloudflare 공식문서를 위한 SiteGPT 버전을 만드세요.
# 2.챗봇은 아래 프로덕트의 문서에 대한 질문에 답변할 수 있어야 합니다:
#     AI Gateway
#     Cloudflare Vectorize
#     Workers AI
# 3.사이트맵을 사용하여 각 제품에 대한 공식문서를 찾아보세요.
# 4.여러분이 제출한 내용은 다음 질문으로 테스트됩니다:
#     "llama-2-7b-chat-fp16 모델의 1M 입력 토큰당 가격은 얼마인가요?"
#     "Cloudflare의 AI 게이트웨이로 무엇을 할 수 있나요?"
#     "벡터라이즈에서 단일 계정은 몇 개의 인덱스를 가질 수 있나요?"
# 5.유저가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 이를 로드합니다.
# 6.st.sidebar를 사용하여 Streamlit app과 함께 깃허브 리포지토리에 링크를 넣습니다.



import streamlit as st
from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler


answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]

    answers_chain = answers_prompt | llm_for_get_answer
    return {
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {
                        "question": question,
                        "context": doc.page_content,
                    }
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
        "question": question,
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.
            Choose the most informed answer among the answers with the same score.

            You should always respond to the source.

            Answers: {answers}
            ---
            Examples:
                                                  
            The moon is 384,400 km away.

            Source: https://example.com
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]

    choose_chain = choose_prompt | llm_for_choose_answer
    condensed = "\n\n".join(
        f"{answer['answer']} \nSource:{answer['source']} \nDate:{answer['date']} \n\n"
        for answer in answers
    )

    return choose_chain.invoke({"answers": condensed, "question": question})


def parse_page(soup: BeautifulSoup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return str(soup.get_text()).replace("\n", "")


@st.cache_data(show_spinner="Loading Website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )

    loader = SitemapLoader(
        url,
        # filter_urls=[],
        filter_urls=(
            [
                r"https:\/\/developers.cloudflare.com/ai-gateway.*",
                r"https:\/\/developers.cloudflare.com/vectorize.*",
                r"https:\/\/developers.cloudflare.com/workers-ai.*",
            ]
        ),
        parsing_function=parse_page,
    )
    # loader.requests_per_second = 1
    ua = UserAgent()
    loader.headers = {"User-Agent": ua.random}
    docs = loader.load_and_split(splitter)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    url_copy = url[:]
    cache_filename = url_copy.replace("/", "_")
    cache_filename.strip()
    cache_dir = LocalFileStore(f"./.cache/{cache_filename}/")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        cache_dir,
    )
    vector_store = FAISS.from_documents(docs, cached_embeddings)
    return vector_store.as_retriever()


# Chat & Streaming
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


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)

st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)


with st.sidebar:
    openai_api_key = st.text_input("Input your OpenAI API Key")
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
        value="https://developers.cloudflare.com/sitemap.xml",
        disabled=True,
    )
    st.markdown("---")
    st.write("Github: https://github.com/fullstack-gpt-python/assignment-17")

if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")
    if not openai_api_key:
        st.error("Please input your OpenAI API Key on the sidebar")
    else:
        paint_history()
        llm_for_get_answer = ChatOpenAI(
            temperature=0.1,
            openai_api_key=openai_api_key,
        )
        llm_for_choose_answer = ChatOpenAI(
            temperature=0.1,
            openai_api_key=openai_api_key,
            streaming=True,
            callbacks=[ChatCallbackHandler()],
        )

        retriever = load_website(url)
        query = st.chat_input("Ask a question to the website.")
        if query:
            send_message(query, "human")
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )

            with st.chat_message("ai"):
                chain.invoke(query)
