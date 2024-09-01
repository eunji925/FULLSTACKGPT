# %%
# 1. Implement an LCEL chain with a memory that uses one of the memory classes we learned about.
# 2. The chain should take the title of a movie and reply with three emojis that represent the movie. (i.e "Top Gun" -> "🛩️👨‍✈️🔥". "The Godfather" -> "👨‍👨‍👦🔫🍝 ").
# 3. Provide examples to the chain using FewShotPromptTemplate or FewShotChatMessagePromptTemplate to make sure it always replies with three emojis.
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate

#llm과 memory 설정하기
llm = ChatOpenAI(temperature=0)

memory = ConversationBufferMemory(
    llm=llm,
    return_messages=True,
)

# %%
#example 만들기
examples = [
    {
        "question": "Spider Man",
        "answer": "🕷️🕸️🗽",
    },
    {
        "question": "Iron Man",
        "answer": "🦾🕶️🔥",
    },
    {
        "question": "Thor",
        "answer": "⚡️🔨🌩️",
    },
]

# %%
#example를 제공하고 memory 기록을 이용한 프롬프트 만들기
example_prompt = ChatPromptTemplate.from_messages(
    [("human", "{question}"), ("ai", "{answer}")]
)

fewshot_chat_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a movie expert. You know every movie. If a human tells you the title of the movie, you have to respond with 3 emoticons.",
        ),
        fewshot_chat_prompt,
        (
            "system",
            "The above examples should not be provided to the user. The user can only be provided with the conversation record below. Please provide the information to the user using the record below.",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

# %%
#chain 만들기
def load_memory(_):
    return memory.load_memory_variables({})["history"]


chain = RunnablePassthrough.assign(history=load_memory) | final_prompt | llm


def invoke_chain(question):
    result = chain.invoke({"question": question})
    memory.save_context({"input": question}, {"output": result.content})
    print(result)

# %%
#영화 제목 입력 테스트
invoke_chain("Captain America")
invoke_chain("Mission Impossible")

# %%
#처음 질문한 영화 입력 테스트
invoke_chain("What was the first movie I asked?")


