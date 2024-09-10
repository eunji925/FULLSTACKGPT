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