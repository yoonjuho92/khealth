import streamlit as st
from api import ask_rag_chatbot

st.set_page_config(page_title="벤야민 챗봇", page_icon="📚", layout="wide")

st.title("📖 발터 벤야민 챗봇")
st.caption("예술가로서 고민이 있다면 발터 벤야민에게 상담해보세요!")

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_user_input" not in st.session_state:
    st.session_state.pending_user_input = None
if "pending_assistant_response" not in st.session_state:
    st.session_state.pending_assistant_response = None

# 유저 입력 받기
user_input = st.chat_input("질문을 입력하세요...")

# 새 입력 처리
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.pending_user_input = user_input

# 이전까지의 대화 출력
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# assistant 응답 처리
if st.session_state.pending_user_input:
    with st.spinner("생각 중..."):
        assistant_msg = ask_rag_chatbot(
            st.session_state.pending_user_input,
            st.session_state.chat_history,
        )
    st.session_state.chat_history.append(
        {"role": "assistant", "content": assistant_msg}
    )
    st.session_state.pending_user_input = None

    # 👉 응답은 여기서 새로 렌더링
    with st.chat_message("assistant"):
        st.markdown(assistant_msg)
