# 졸업과제
# (EN)
# Refactor the agent you made in the previous assignment into an OpenAI Assistant.
# Give it a user interface with Streamlit that displays the conversation history.
# Allow the user to use its own OpenAI API Key, load it from an st.input inside of st.sidebar
# Using st.sidebar put a link to the Github repo with the code of your Streamlit app.
# (KR)
# 이전 과제에서 만든 에이전트를 OpenAI 어시스턴트로 리팩터링합니다.
# 대화 기록을 표시하는 Streamlit 을 사용하여 유저 인터페이스를 제공하세요.
# 유저가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 이를 로드합니다.
# st.sidebar를 사용하여 Streamlit app 의 코드과 함께 깃허브 리포지토리에 링크를 넣습니다.

# 과제 해설

# 1. 어시스턴트가 사용할 함수 정의
#     Function Calling에서 사용할 함수를 정의합니다.
#     함수의 동작은 지난 과제에서 구현한 것과 동일합니다.
#     솔루션에서는 덕덕고에서 검색을 수행하는 함수인 get_ddg_results, 위키 백과에서 검색을 수행하는 함수인 get_wiki_results, 웹의 내용을 가져오는 get_web_content로 정의하였습니다.
#     Function Calling에 대한 공식 문서를 참고하세요.
# 2. Assistant -> Thread -> Run
#     함수들을 정의한 내용과 함께 Assistant를 생성합니다.
#     대화 세션인 Thread를 생성합니다.
#     사용자가 보낸 Message를 Thread에 추가하고 Run을 생성합니다.
#     이 과정은 강의에서도 자세히 다루고 Assistants API Quickstart에도 전체적인 흐름을 파악할 수 있습니다.
#     단, 리렌더링 시 다시 생성되지 않게 하기 위하여 Session State를 사용하여 저장해두었습니다.
# 3. Run의 상태
#     Run은 여러 상태를 가지지만 (공식 문서 참고) 솔루션에서는 requires_action, completed, failed, expired 에 따라 다르게 처리해 주었습니다.
#     requires_action인 경우에는 응답에서 실행할 함수와 매개변수를 추출하여 함수를 실행시킨 후, 응답을 전달합니다. 이때, requires_action은 한 번에 반복적으로 발생할 수 있으므로 반복문으로 처리해주었습니다.
#     failed인 경우에는 사용자에게 에러가 발생했음을 표시했습니다.
#     completed인 경우에는 에이전트가 성공적으로 실행을 마쳤다는 의미이므로 응답 결과를 추출하여 화면에 표시합니다.
#     expired인 경우에는 말 그대로 Run이 만료되었다는 의미이므로 Run을 다시 생성해 줍니다.
# 4. 채팅 기록 표시
#     화면 리렌더링 시 이전에 했던 채팅들의 기록을 표시해야 합니다.
#     채팅 기록을 불러올 Thread의 ID를 이용하여 메시지 리스트를 얻고 이를 화면에 표시합니다.
#     응답 결과 txt 다운로드
#     솔루션에서는 에이전트의 응답 밑에 다운로드 버튼을 표시하고 이 버튼을 누르면 해당 응답의 내용을 담은 TXT 파일이 다운로드되도록 구현하였습니다.
# 5. 결론
#     이번 챌린지는 졸업 작품인 만큼, 앞서 배운 모든 개념을 활용하여 구현해야 했습니다.
#     특히 Streamlit을 이용하여 간단하지만 인터페이스를 적용해야 했기에 쉽지는 않았을 것이라 생각합니다.
#     그러나 개발자가 아닌 사람도 쉽게 사용할 수 있도록 기능을 인터페이스에 반영하는 것은 매우 중요하다고 생각하기 때문에, 이번 졸업 작품 챌린지는 큰 의미가 있었다고 느낍니다.
#     졸업 작품 만드시느라 고생하셨습니다 :)

import streamlit as st
from langchain.utilities.wikipedia import WikipediaAPIWrapper
import json
from langchain.document_loaders import WebBaseLoader
from langchain.tools import DuckDuckGoSearchResults
from langchain.tools import WikipediaQueryRun
from openai import OpenAI
from datetime import datetime, timedelta, timezone
import requests
from bs4 import BeautifulSoup


def get_assistant():
    if "assistant" in st.session_state:
        return st.session_state["assistant"]

    functions = [
        {
            "type": "function",
            "function": {
                "name": "get_ddg_results",
                "description": "Use this tool to perform web searches using the DuckDuckGo search engine. It takes a query as an argument.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query you will search for",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_wiki_results",
                "description": "Use this tool to perform searches on Wikipedia.It takes a query as an argument.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query you will search for",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_web_content",
                "description": "If you found the website link in DuckDuckGo, Use this to get the content of the link for my research.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL of the website you want to scrape",
                        },
                    },
                    "required": ["url"],
                },
            },
        },
    ]

    assistant = client.beta.assistants.create(
        name="Research Assistant",
        instructions="""
        You are a research expert.

        Your task is to use Wikipedia or DuckDuckGo to gather comprehensive and accurate information about the query provided. 

        When you find a relevant website through DuckDuckGo, you must scrape the content from that website. Use this scraped content to thoroughly research and formulate a detailed answer to the question. 

        Combine information from Wikipedia, DuckDuckGo searches, and any relevant websites you find. Ensure that the final answer is well-organized and detailed, and include citations with links (URLs) for all sources used.

        Your research should be saved to a .txt file, and the content should match the detailed findings provided. Make sure to include all sources and relevant information.

        The information from Wikipedia must be included.
        """,
        model="gpt-4o-mini",
        tools=functions,
    )

    st.session_state["assistant"] = assistant

    return st.session_state["assistant"]


def get_ddg_results(inputs):
    query = inputs["query"]
    search = DuckDuckGoSearchResults()
    return search.run(query)


def get_wiki_results(inputs):
    query = inputs["query"]
    wrapper = WikipediaAPIWrapper(top_k_results=3)
    wiki = WikipediaQueryRun(api_wrapper=wrapper)
    return wiki.run(query)


def get_web_content(inputs):
    url = inputs["url"]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for header in soup.find_all(["header", "footer", "nav"]):
            header.decompose()
        content = soup.get_text(separator="\n", strip=True)

        return content

    except requests.RequestException as e:
        print(f"ERROR on get_web_content: {e}")
        return f"Error getting content from {url}. Use another url."


functions_map = {
    "get_ddg_results": get_ddg_results,
    "get_web_content": get_web_content,
    "get_wiki_results": get_wiki_results,
}


def get_thread_id():
    if "thread_id" not in st.session_state:

        thread = client.beta.threads.create(
            messages=[
                {
                    "role": "assistant",
                    "content": "Hi, How can I help you?",
                }
            ]
        )
        st.session_state["thread_id"] = thread.id
    return st.session_state["thread_id"]


def get_run(run_id, thread_id):

    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )


def start_run(thread_id, assistant_id, content):
    if "run" not in st.session_state or get_run(
        st.session_state["run"].id, get_thread_id()
    ).status in (
        "expired",
        "completed",
    ):

        client.beta.threads.messages.create(
            thread_id=thread_id, role="user", content=content
        )

        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread_id,
            assistant_id=assistant_id,
        )

        st.session_state["run"] = run
    else:
        print("already running")

    run = st.session_state["run"]

    with st.status("Processing..."):
        while get_run(run.id, get_thread_id()).status == "requires_action":
            submit_tool_outputs(run.id, get_thread_id())

    print(f"done, {get_run(run.id, get_thread_id()).status}")
    final_message = get_messages(get_thread_id())[-1]
    if get_run(run.id, get_thread_id()).status == "completed":
        with st.chat_message(final_message.role):
            st.markdown(final_message.content[0].text.value)

        paint_download_btn(
            final_message.content[0].text.value, createdAt=final_message.created_at
        )
        print(final_message)
    elif get_run(run.id, get_thread_id()).status == "failed":
        with st.chat_message("assistant"):
            st.markdown("Sorry. I failed researching. Try Again later :()")


def get_messages(thread_id):
    messages = list(
        client.beta.threads.messages.list(
            thread_id=thread_id,
        )
    )
    return list(reversed(messages))


def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread_id)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        st.write(f"Calling function: {function.name} with arg {function.arguments}")
        print(f"Calling function: {function.name} with arg {function.arguments}")
        output = functions_map[function.name](json.loads(function.arguments))
        outputs.append(
            {
                "output": output,
                "tool_call_id": action_id,
            }
        )
    return outputs


def submit_tool_outputs(run_id, thread_id):

    outputs = get_tool_outputs(run_id, thread_id)
    return client.beta.threads.runs.submit_tool_outputs_and_poll(
        run_id=run_id, thread_id=thread_id, tool_outputs=outputs
    )


def send_message(_content: str):
    thread_id = get_thread_id()

    start_run(thread_id, assistant_id, _content)


def paint_download_btn(content, createdAt):
    file_bytes = content.encode("utf-8")

    created_at_utc = datetime.fromtimestamp(createdAt, tz=timezone.utc)
    kst_timezone = timezone(timedelta(hours=9))
    created_at_kst = created_at_utc.astimezone(kst_timezone)
    formatted_date = created_at_kst.strftime("%y_%m_%d_%H%M_Answer")

    st.download_button(
        label="Download this answer.",
        data=file_bytes,
        file_name=f"{formatted_date}_{createdAt}.txt",
        mime="text/plain",
        key=createdAt,
    )


st.set_page_config(
    page_title="OpenAI Agent",
    page_icon="🤖",
)

st.markdown(
    """
    # OpenAI Agent

    ### Agent to help you search
    """
)

# ? Sidebar
with st.sidebar:
    openai_api_key = st.text_input(
        "Input your OpenAI API Key",
    )
    if not openai_api_key:
        st.error("OpenAI API Key is required.")

    st.markdown("---")
    st.write("https://github.com/fullstack-gpt-python/assignment-19")

# ? Main Screen
if not openai_api_key:
    st.error("Please input your OpenAI API Key on the sidebar")
else:
    query = st.chat_input("Ask a question to the website.")
    client = OpenAI(api_key=openai_api_key)
    assistant_id = get_assistant().id

    # 메시지 기록 출력
    for idx, message in enumerate(get_messages(get_thread_id())):
        with st.chat_message(message.role):
            st.markdown(message.content[0].text.value)
        if message.role == "assistant" and idx > 0:
            paint_download_btn(
                message.content[0].text.value, createdAt=message.created_at
            )

    if query:
        with st.chat_message("user"):
            st.markdown(query)
        send_message(query)