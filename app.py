import streamlit as st
import os
import openai

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage

from pdfminer.high_level import extract_text
from dotenv import load_dotenv

# 🔐 Load API Key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 확인 (개발 중에만 사용)
assert openai.api_key is not None, "❌ OPENAI_API_KEY가 환경변수에서 불러와지지 않았습니다."

# 📄 PDF 텍스트 추출
def get_pdf_text(filename):
    return extract_text(filename)

# 📚 문서 처리 및 임베딩
def process_uploaded_file(FILE_PATH):
    if FILE_PATH is not None:
        raw_text = get_pdf_text(FILE_PATH)

        # 텍스트 청킹
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=10000,
            chunk_overlap=2000,
        )
        all_splits = text_splitter.create_documents([raw_text])
        print("총", len(all_splits), "개의 청크 생성됨")

        # 벡터스토어 생성
        vectorstore = FAISS.from_documents(all_splits, OpenAIEmbeddings())

        return vectorstore, raw_text
    return None, None

# 📡 스트리밍 출력 핸들러
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# 🤖 질문에 답변 생성 (RAG)
def generate_response(query_text, vectorstore, callback):
    docs_list = vectorstore.similarity_search(query_text, k=3)
    docs = ""
    for i, doc in enumerate(docs_list):
        docs += f"'문서{i+1}':{doc.page_content}\n"

    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0,
        streaming=True,
        callbacks=[callback],
    )

    rag_prompt = [
        SystemMessage(
            content="너는 문서에 대해 질의응답을 하는 '앵무새'야. 주어진 문서를 참고해서 질문에 친절하게 답변해줘. 문서에 없으면 모르겠다고 해도 괜찮아! 이모티콘도 살짝 써줘. 항상 답변 끝에 ' 짹! 🦜'이라고 붙여줘."
        ),
        HumanMessage(
            content=f"질문:{query_text}\n\n{docs}"
        ),
    ]

    response = llm(rag_prompt)
    return response.content

# 📄 요약 기능
def generate_summarize(raw_text, callback):
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0,
        streaming=True,
        callbacks=[callback],
    )

    rag_prompt = [
        SystemMessage(content="다음 문서를 Notion 스타일로 요약해줘. 핵심만, 짧고 명확하게."),
        HumanMessage(content=raw_text),
    ]

    response = llm(rag_prompt)
    return response.content

# 🧾 Streamlit 인터페이스
st.set_page_config(page_title='🦜어떤 문서든 물어봐!')
st.title('🦜 어떤 문서든 물어봐!')

st.markdown("""
<style>
.main { background-color: #f5f5f5; }
.sidebar .sidebar-content { background-color: #f0f0f0; }
.stButton>button { background-color: #4CAF50; color: white; }
</style>
""", unsafe_allow_html=True)

# 📁 파일 업로드
st.sidebar.header('📄 문서 업로드')
uploaded_file = st.sidebar.file_uploader("PDF 문서를 업로드하세요", type=["pdf"])

# 🔄 벡터스토어 처리
if uploaded_file:
    vectorstore, raw_text = process_uploaded_file(uploaded_file)
    if vectorstore:
        st.session_state['vectorstore'] = vectorstore
        st.session_state['raw_text'] = raw_text

# 💬 채팅 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        ChatMessage(
            role="assistant",
            content="안녕하세요! 저는 업로드한 문서를 이해하고 요약하거나 질문에 답변할 수 있는 앵무새예요. 🦜 무엇이 궁금한가요?"
        )
    ]

# 💬 채팅 출력
for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

# 💬 입력 처리
if prompt := st.chat_input("예: '요약' 또는 '이 문서의 목적은?'"):
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        if prompt.strip() == "요약":
            response = generate_summarize(st.session_state['raw_text'], stream_handler)
        else:
            response = generate_response(prompt, st.session_state['vectorstore'], stream_handler)

    st.session_state.messages.append(ChatMessage(role="assistant", content=response))
