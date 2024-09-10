from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_upstage import UpstageEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import ChatUpstage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from dotenv import load_dotenv

load_dotenv()

### Statefully manage chat history ###
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Initialize components
embedding = UpstageEmbeddings(model="solar-embedding-1-large")
index_name = 'markdown-index'
database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
retriever = database.as_retriever(search_kwargs={'k': 4})
llm = ChatUpstage()

# Define prompts
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are an assistant for question-answering tasks. \
        Use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, just say that you don't know. \

        {context}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

dictionary = ["""
    금융 뉴스레터 : 
    유럽시장 :
    . Condé Nast는 AI 기술을 활용한 뉴스 제공을 위해 OpenAI와 파트너십을 맺었습니다. 이를 통해 Condé Nast는 AI를 활용한 뉴스 검색과 전달을 강화하며, 정확성과 신뢰성을 유지하고자 합니다.
    . Regeneron Pharmaceuticals, Inc.는 재발성/불응성 다발성 골수종 치료제인 Linzagolix에 대한 생물학적 제제 허가신청(BLA)에 대해 미국 식품의약국(FDA)으로부터 완전 응답 서한을 받았습니다.
    . 기초 소재 분야에서 Carpenter Technology Corporation, IAMGOLD Corporation, Eldorado Gold Corporation이 높은 순위를 차지하고 있습니다. 이 세 회사는 최근 수익 예상치와 어닝 서프라이즈에서 긍정적인 성과를 보이고 있습니다.
    . 유럽 증시는 유로존 STOXX 600 지수가 0.45%, 독일 DAX 지수가 0.35%, 프랑스 CAC 40 지수가 0.22% 하락하며 하락세로 마감했습니다.
    
    아시아 태평양 시장 :
    . 아시아 태평양 시장에서는 일본 Nikkei 225 지수가 1.80% 상승하였고, 홍콩 Hang Seng 지수는 0.33% 하락하였습니다. 중국 상하이 종합 지수는 0.93% 하락하였고, 인도 BSE Sensex 지수는 0.47% 상승하였습니다.
    . Innofactor Plc는 핀란드 증권거래소에서 주식 소유권 공시를 발표하였으며, 주요 주주인 Onni Bidco Oy의 지분율이 25% 기준을 초과하였습니다.
    . Rosen Law Firm은 CAE Inc.에 대한 투자자를 대상으로 2022년 2월 11일부터 2024년 5월 21일까지의 기간 동안 구매한 투자자들에게 2024년 9월 16일까지 법원 승인 집단 소송의 주요 원고로 참여할 것을 안내하고 있습니다.
    이 뉴스레터가 다양한 금융 관련 소식을 전달해 드렸기를 바랍니다.

    경제학:
    """]

# Define dictionary prompt
prompt = ChatPromptTemplate.from_template(f"""
    사용자의 질문을 보고, 우리의 사전을 참고해서 비슷한 양식으로
    답변을 생성해주세요. 답변에 포함되어야할 내용은 다음과 같습니다. 
    정보 하나당 앞에 말머리기호 점을 붙여 표현합니다. 최대한 다양한 정보를 전해야합니다.
    사전에 있는 내용과 같은 내용은 답변으로 제공하지 마세요.
    사전: {dictionary}
    
    질문: {{question}}
""")

def get_ai_message(user_message):
    dictionary_chain = prompt | llm | StrOutputParser()
    tax_chain = {"query": dictionary_chain} | conversational_rag_chain
    ai_message = tax_chain.invoke({'question': user_message})
    return ai_message['query'] + ai_message['result']
