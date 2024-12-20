import asyncio
import streamlit as st
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 세션 상태 초기화
if "embeddings" not in st.session_state:
    st.session_state.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = Chroma(persist_directory="./DB", embedding_function=st.session_state.embeddings)

if "conversation" not in st.session_state:
    st.session_state.conversation = []  # 대화 상태 초기화

# 스타일 설정
st.image("./image_logo.png", width=300)
st.image("robot.gif")
st.markdown(
    """
    <div style="font-family: 'Arial', sans-serif; color:#000000; font-size:18px; font-weight:bold; text-align:center;">
        누구나 쉽게 노무/인사 관련하여 챗봇에게 물어보세요🤖
    </div>
    """,
    unsafe_allow_html=True,
)
# 줄 띄우기 위해 빈 줄 추가 또는 <br> 사용
st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="font-family: 'Arial', sans-serif; color:#000000; font-size:25px; font-weight:bold; text-align:center;">
        무엇을 도와드릴까요?
    </div>
    """,
    unsafe_allow_html=True
)

# Retriever 생성
retriever = st.session_state.vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 5})

# 사용자 입력
query = st.chat_input("상담 내용을 작성하세요.예. 근로계약서는 어떻게 써야하나요?")
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("Lawbot의 답변이 모든 상황에 완벽하지 않을 수 있습니다. \n보다 정확한 상담이 필요하시면 전문가에게 문의하시길 권장드립니다. 😊")

if query:

    # 사용자의 입력 메시지를 추가
    st.session_state.conversation.append({"role": "user", "content": query})

    # Spinner를 표시하며 응답 생성
    with st.spinner("Lawbot이 응답을 생성하는 중입니다...🤖"):
        try:
            # 문서 검색
            retrievals = retriever.batch([query])
            MAX_CONTEXT_LENGTH = 1600
            if retrievals and len(retrievals[0]) > 0:
                retrieved_context = "\n".join([doc.page_content[:MAX_CONTEXT_LENGTH] for doc in retrievals[0]])
            else:
                retrieved_context = "관련 정보를 찾을 수 없습니다."

            # 대화 요약
            def summarize_conversation(conversation, model="gpt-4"):
                summary_prompt = ChatPromptTemplate.from_messages([
                    ("system", "다음 대화를 요약하세요. 중요한 정보만 유지해주세요. 맨 처음에 '기존 대화 요약'을 써주세요."),
                    ("user", conversation)
                ])
                chat_model = ChatOpenAI(model=model)
                output_parser = StrOutputParser()
                chain = summary_prompt | chat_model | output_parser
                summary = chain.invoke({"conversation": conversation})
                return summary

            # 기존 대화 요약 추가
            if st.session_state.conversation:
                summarized_conversation = summarize_conversation("\n".join([msg["content"] for msg in st.session_state.conversation]))
            else:
                summarized_conversation = ""

            # 최종 컨텍스트 생성
            final_context = f"기존 대화 요약:\n{summarized_conversation}\n\n검색된 컨텍스트:\n{retrieved_context}"

            # Prompt 생성 및 응답 처리
            prompt = ChatPromptTemplate.from_messages([
                ("system", '''
                1. 페르소나 (Persona)
당신은 한국 노동법에 정통한 노동법 전문 변호사이다.
사용자에게 법적 조언을 제공하며, 노동법 및 관련 판례에 대한 전문 지식을 바탕으로 상세하고 신뢰할 수 있는 정보를 제공한다.
친절하고 신뢰감을 주는 말투를 유지하며, 법적 조언과 함께 실질적인 실행 방안도 제시한다.
2. 역할 (Role)
사용자로부터 노동법 관련 질문이나 사례를 받으면, 그에 대한 법률적 검토와 조언을 제공한다.
한국 노동법의 조항 및 주요 판례를 참고하여, 사용자가 이해하기 쉽게 설명한다.
필요한 경우, 문서 작성 가이드, 소송 절차, 또는 노사 분쟁 해결을 위한 전략을 제안한다.
구어체를 사용하여 친절하게 답변한다. 

4. 출력 형식 (Output Format)
구조화된 답변: 사용자의 질문을 명확히 이해하고, 관련 법 조항 및 판례를 바탕으로 체계적이고 간결한 답변을 제공한다.
1단계: 요약: 질문의 핵심 단어나 이슈에 대해 이해하기 쉽게 설명하고, 질문에 대한 간략한 핵심 요약 제공.
2단계: 법적 근거: 관련 노동법 조항 및 판례를 인용하며, 사용자에게 근거를 설명.
3단계: 실행 방안: 사용자 상황에 적합한 실행 가능한 법적/행정적 방안을 제시.
4단계: 추가 정보 제공: 참고할 만한 외부 자료나 필요한 추가 조언을 안내.
'''),
                ("user", f"질문: {query}\n컨텍스트: {final_context}")
            ])

            chat_model = ChatOpenAI(model="gpt-4o")
            output_parser = StrOutputParser()
            chain = prompt | chat_model | output_parser

            response = chain.invoke({"query": query, "context": final_context})

            # 응답 추가
            st.session_state.conversation.append({"role": "assistant", "content": response})

        except Exception as e:
            error_message = f"오류가 발생했습니다: {str(e)}"
            st.session_state.conversation.append({"role": "assistant", "content": error_message})

# 이전 대화 렌더링
for msg in st.session_state.conversation:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
