# %%
# 1. 영화 이름을 가지고 감독, 주요 출연진, 예산, 흥행 수익, 영화의 장르, 간단한 시놉시스 등 영화에 대한 정보로 답장하는 체인을 만드세요.
# 2.LLM은 항상 동일한 형식을 사용하여 응답해야 하며, 이를 위해서는 원하는 출력의 예시를 LLM에 제공해야 합니다.
# 3.예제를 제공하려면 FewShotPromptTemplate 또는 FewShotChatMessagePromptTemplate을 사용하세요.
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate

# OpenAI 모델 설정
chat = ChatOpenAI(temperature=0.1)

# 예제 출력 템플릿 정의
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "Can you tell me about the movie {movie_name}?"),
        ("ai", """
        Title: {title}
        Director: {director}
        Main Cast: {main_cast}
        Budget: {budget}
        Box Office Revenue: {box_office}
        Genre: {genre}
        Synopsis: {synopsis}
        """),
    ]
)

# 예제들 정의
examples = [
    {
        "movie_name": "Inception",
        "title": "Inception",
        "director": "Christopher Nolan",
        "main_cast": "Leonardo DiCaprio, Joseph Gordon-Levitt, Ellen Page",
        "budget": "$160 million",
        "box_office": "$829.9 million",
        "genre": "Science Fiction, Action",
        "synopsis": "A skilled thief, the absolute best in the dangerous art of extraction, stealing valuable secrets from deep within the subconscious during the dream state, is given a chance to have his criminal history erased as payment for the implantation of another person's idea into a target's subconscious.",
    },
    {
        "movie_name": "The Matrix",
        "title": "The Matrix",
        "director": "Lana Wachowski, Lilly Wachowski",
        "main_cast": "Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss",
        "budget": "$63 million",
        "box_office": "$467.2 million",
        "genre": "Science Fiction, Action",
        "synopsis": "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.",
    },
    {
        "movie_name": "Titanic",
        "title": "Titanic",
        "director": "James Cameron",
        "main_cast": "Leonardo DiCaprio, Kate Winslet",
        "budget": "$200 million",
        "box_office": "$2.202 billion",
        "genre": "Romance, Drama",
        "synopsis": "A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious, ill-fated R.M.S. Titanic.",
    },
]

# FewShot 프롬프트 템플릿 생성
example_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# 최종 프롬프트 정의
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a movie expert, provide detailed information about movies."),
        example_prompt,
        ("human", "Can you tell me about the movie {movie_name}?"),
    ]
)

# 체인 생성
chain = final_prompt | chat

# 예제 영화 정보 요청
response = chain.invoke({"movie_name": "Jurassic Park"})
print(response)



