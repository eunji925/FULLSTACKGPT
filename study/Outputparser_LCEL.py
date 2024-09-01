# %%
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate

chat = ChatOpenAI(temperature=0.1)

# %%
from langchain.schema import BaseOutputParser

class CommaOutputParser(BaseOutputParser):

    def parse(self, text):
        items = text.strip().split(",") 
        return list(map(str.strip,items))

# %%
template = ChatPromptTemplate.from_messages([
    ("system", "ou are a list generating machine. Everything you are asked will be answered with a comma separated list of max {max_items} in lowercase.Do NOT reply with anything else."),
    ("human","{question}")
])

chain = template | chat | CommaOutputParser()

chain.invoke({"max_items":5 , "question":"What are the poketmons?"})

# %%
