# %%
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler

chat = ChatOpenAI(temperature=0.1, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])

poem_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a poetic machine. Generate a creative and metaphorical poem."),
    ("human", "Write a poetic verse about the following {programming_language}")
])

poem_chain = poem_prompt | chat

# %%
# 시를 설명하는 체인
explanation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an explanation machine. You need to Explain a poem written about a programming language."),
    ("human", "Explain the following {poem} which is written about a programming_language.")
])

explanation_chain = explanation_prompt | chat

final_chain = {"poem" : poem_chain} | explanation_chain

final_chain.invoke({
    "programming_language" : "python"
})


