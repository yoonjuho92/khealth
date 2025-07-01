import streamlit as st
from api import ask_rag_chatbot, load_prompt, insert_prompt

st.set_page_config(page_title="코끼리 챗봇", page_icon="📚", layout="centered")
st.title("📖 아동상담봇")
st.caption("아동상담 전문 교과서의 text를 기반으로 답변을 드립니다.")

# === 프롬프트 선택/수정 UI ===
st.subheader("⚙️ 시스템 프롬프트 설정")

prompt_names = ["instruction", "few_shots"]

if "prompt_texts" not in st.session_state:
    st.session_state.prompt_texts = {name: load_prompt(name) for name in prompt_names}

for name in prompt_names:
    with st.expander(f"✏️ {name} 프롬프트 수정"):
        new_text = st.text_area(
            f"{name} 프롬프트", st.session_state.prompt_texts[name], height=200
        )
        if st.button(f"{name} 저장"):
            insert_prompt(name, new_text)
            st.session_state.prompt_texts[name] = new_text
            st.success(f"{name} 프롬프트가 저장되었습니다.")

# === 세션 상태 초기화 ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_user_input" not in st.session_state:
    st.session_state.pending_user_input = None
if "pending_assistant_response" not in st.session_state:
    st.session_state.pending_assistant_response = None

# === 유저 입력 받기 ===
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

    with st.chat_message("assistant"):
        st.markdown(assistant_msg)
