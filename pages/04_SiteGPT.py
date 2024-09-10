from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 800,
    chunk_overlap = 200,
)

def parse_page(soup): # soup : document의 전체 HTML을 가진 beautiful soup object 값
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return str(soup.get_text()).replace("\n"," ").replace("\t"," ").replace("\xa0", " ") # 공백등을 제거하기 위한 replace
    

@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size = 800,
        chunk_overlap = 200,
    )
    loader = SitemapLoader(
        url,
        filter_urls = [r"^(?!.*survey).*",], # 사용하는 url 에 따라 달라질 수 있다.
        parsing_function = parse_page
    )
    loader.requests_per_second = 1 # 요청 속도 조정 ( 1초에 1번 )
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

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
    url = st.text_input("Write down a URL",placeholder="https://example.com",)

if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please wrrite down a Sitemap URL")
    else:
        docs = load_website(url)
        st.write(docs)