import faiss
import pickle
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# === FAISS & chunks 불러오기 ===
index = faiss.read_index("faiss_index.idx")

with open("faiss_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)


# === 임베딩 함수 ===
def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(input=[text], model="text-embedding-3-small")
    return response.data[0].embedding


# === 검색 함수 ===
def retrieve_relevant_chunks(query: str, top_k: int = 3) -> list[str]:
    query_vector = np.array(get_embedding(query)).astype("float32").reshape(1, -1)
    D, I = index.search(query_vector, top_k)
    return [chunks[i] for i in I[0]]


# === Chat API 호출 함수 (대화 이력 지원) ===
def ask_rag_chatbot(query: str, chat_history: list[dict]) -> str:
    relevant_chunks = retrieve_relevant_chunks(query)
    context = "\n---\n".join(relevant_chunks)

    system_prompt = """당신이 발터 벤야민이라고 생각하고 사용자의 질의에 답변해 주세요. 
    참고하기 위해 주어진 정보는 발터 벤야민에 대한 간략한 설명 중에서 사용자의 질문과 의미상 유사한 부분을 발췌한 정보입니다. 
    사용자의 질문에 대해 간략하고 친절하게 답변해 주세요.철학 비 전문가인 사람에게 고민을 상담해주듯이 답변해주세요.특히 예술에 대해 고민하는 어린 작가를 대하듯이 말해주세요.그렇지만 대답할 때 벤야민의 이론과 생각은 드러나야 합니다. 말투는 자연스럽고 현대적으로 해주세요.
    사용자가 묻지 않은 정보에 대해 너무 빨리 답변하진 말아 주세요."""

    messages = [{"role": "system", "content": system_prompt}] + chat_history
    messages.append(
        {
            "role": "user",
            "content": f"다음 정보를 참고해서 대답해주세요:\n\n{context}\n\n질문: {query}",
        }
    )

    response = client.chat.completions.create(
        model="gpt-4.1-mini", messages=messages, temperature=0.2
    )
    return response.choices[0].message.content.strip()


# === 실행 예시 ===
if __name__ == "__main__":
    # 과거 대화 내역 (예시)
    previous_chat = [
        {"role": "user", "content": "벤야민은 기술과 예술의 관계를 어떻게 봤나요?"},
        {
            "role": "assistant",
            "content": "그는 기계 복제가 예술의 아우라를 파괴한다고 말했어요.",
        },
    ]

    query = "더 이상 글을 쓸 수 없게 된 작가는 어떻게 해야 할까요?"
    response = ask_rag_chatbot(query, previous_chat)
    print("💬 답변:", response)
