%%writefile app.py

import streamlit as st
import os

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage

# Function to extract text from an PDF file
from pdfminer.high_level import extract_text

from dotenv import load_dotenv
load_dotenv()

# loader
def get_pdf_text(filename):
    raw_text = extract_text(filename)
    return raw_text

# 문서 로드 및 청킹 (Loader + Splitter)
def process_uploaded_file(FILE_PATH):
    # Load document if file is uploaded
    if FILE_PATH is not None:

        # loader
        raw_text = get_pdf_text(FILE_PATH)

        # splitter
        text_splitter = CharacterTextSplitter(
            separator = "\n\n",     # TODO: 어떤 문구를 기준으로? (스페이스바, 엔터, ',' 등)
            chunk_size = 1000,      # TODO: 문서의 잘림 크기는 몇으로?
            chunk_overlap = 200,    # TODO: 겹치는 길이는 몇으로?
        )
        all_splits = text_splitter.create_documents([raw_text])
        print("총 " + str(len(all_splits)) + "개의 passage")

        # storage
        vectorstore = FAISS.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

        return vectorstore, raw_text
    return None

# handle streaming conversation
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# RAG 기반으로 답변 생성 (검색(Retriever), 생성(Generator), 연결(Chaining))
def generate_response(query_text, vectorstore, callback):

    # retriever
    docs_list = vectorstore.similarity_search(query_text, k=3) # TODO: 연관성 있는 문서는 몇 개?!
    docs = ""
    for i, doc in enumerate(docs_list):
        docs += f"'문서{i+1}':{doc.page_content}\n"

    # generator
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0,
        streaming=True,
        callbacks=[callback]
    )

    # chaining
    # prompt formatting
#     system_prompt = """
#     너는 문서에 대해 질의응답을 하는 조교야. \n
# 주어진 문서를 참고하여 사용자의 질문에 답변을 해줘. \n
# 문서에 내용이 정확하게 나와있지 않으면 대답하지 마.

# 너는 괴팍한 성격을 가지고 있고 완전 츤데래야."""

    rag_prompt = [
        SystemMessage(
            # content=system_prompt,
            content="너는 문서에 대해 질의응답을 하는 '조교'야. 주어진 문서를 참고하여 사용자의 질문에 답변을 해줘. 문서에 내용이 정확하게 나와있지 않으면 대답하지 마. 이모티콘을 사용해서 친근하게 답변해줘!"
        ),
        HumanMessage(
            content=f"질문:{query_text}\n\n{docs}"
        ),
    ]

    response = llm(rag_prompt)
    return response.content

def generate_summarize(raw_text, callback):

    # generator
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True, callbacks=[callback])

    # prompt formatting
    rag_prompt = [
        SystemMessage(
            content="다음 나올 문서를 'Notion style'로 요약해줘. 중요한 내용만."
        ),
        HumanMessage(
            content=raw_text
        ),
    ]

    response = llm(rag_prompt)
    return response.content

# page title
st.set_page_config(page_title='🦜어떤 문서든 물어봐!🦜')
st.title('🦜어떤 문서든 물어봐!🦜')

# file upload --> PDF만 받을 수 있도록 한다.
# uploaded_file = st.file_uploader('Upload an document', type=['hwp','pdf'])
st.markdown("""
	<style>
	.main { background-color: #f5f5f5; }
	.sidebar .sidebar-content { background-color: #f0f0f0; }
	.stButton>button { background-color: #4CAF50; color: white; }
	</style>
	""", unsafe_allow_html=True)

# 파일 업로드
st.sidebar.header('📄 파일 업로드')
uploaded_file = st.sidebar.file_uploader('문서를 업로드하세요!', type=['pdf'])

# file upload logic
if uploaded_file:
    vectorstore, raw_text = process_uploaded_file(uploaded_file)

    # 초기에 한 번 만들고 session_state에 보관 --> session_state에 접근해서 반복하여 활용
    if vectorstore:
        st.session_state['vectorstore'] = vectorstore
        st.session_state['raw_text'] = raw_text

# chatbot greatings
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        ChatMessage(
            role="assistant", content="안녕하세요! 저는 문서에 대한 이해를 도와주는 앵무새입니다! 어떤게 궁금하신가요?!"
        )
    ]

# conversation history print --> 주석 처리하고 비교해보기
for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

# message interaction
if prompt := st.chat_input("'문서 요약 해줘'라고 입력해보세요!"):
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())

        if prompt == "요약":
            response = generate_summarize(st.session_state['raw_text'],stream_handler)

        else:
            response = generate_response(prompt, st.session_state['vectorstore'], stream_handler)

    st.session_state.messages.append(
        ChatMessage(role="assistant", content=response)
    )
