from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_upstage import UpstageEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import ChatUpstage
from langchain import hub
from langchain.chains import RetrievalQA

def get_ai_message(user_message):
    # Upstage에서 제공하는 Embedding Model을 활용해서 `chunk`를 vector화
    embedding = UpstageEmbeddings(model="solar-embedding-1-large")
    index_name = 'markdown-index'
    database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)




    llm = ChatUpstage()
    prompt = hub.pull("rlm/rag-prompt")

    retriever = database.as_retriever(search_kwargs={'k': 4})

    qa_chain = RetrievalQA.from_chain_type(
        llm, 
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
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


# 5. 유럽에서는 유로존의 7월 연간 인플레이션율이 2.6%로 상승하였고, 경상수지 흑자가 6월에 524억 유로를 기록하였습니다. 이탈리아와 독일의 경상수지 흑자도 증가하였습니다.

    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 비슷한 양식으로
        답변을 생성해주세요. 답변에 포함되어야할 내용은 다음과 같습니다. 
        정보 하나당 앞에 말머리기호 점을 붙여 표현합니다. 최대한 다양한 정보를 전해야합니다.
        사전에 있는 내용과 같은 내용은 답변으로 제공하지 마세요.
        사전: {dictionary}
        
        질문: {{question}}
    """)

    dictionary_chain = prompt | llm | StrOutputParser()
    tax_chain = {"query": dictionary_chain} | qa_chain
    ai_message = tax_chain.invoke({'question': user_message})

    # ai_message = qa_chain.invoke({"query": user_message})
    return ai_message['query']+ai_message['result']