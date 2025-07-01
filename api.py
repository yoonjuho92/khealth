import faiss
import pickle
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from db import supabase

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
def retrieve_relevant_chunks(query: str, top_k: int = 5) -> list[str]:
    query_vector = np.array(get_embedding(query)).astype("float32").reshape(1, -1)
    D, I = index.search(query_vector, top_k)
    return [chunks[i] for i in I[0]]


# === 프롬프트 가져오기 ===
def load_prompt(prompt_name: str) -> str:
    prompt = (
        supabase.table("khealth_prompt")
        .select("prompt")
        .eq("prompt_nm", prompt_name)
        .execute()
    )
    print("-" * 60)
    print(
        f"프롬프트 '{prompt_name}' 로드: {prompt.data[0]['prompt'] if prompt.data else '없음'}"
    )
    print("-" * 60)  # 디버깅용 출력
    return prompt.data[0]["prompt"]


def insert_prompt(prompt_name: str, prompt_text: str) -> None:
    """프롬프트를 데이터베이스에 삽입합니다."""
    supabase.table("khealth_prompt").update({"prompt": prompt_text}).eq(
        "prompt_nm", prompt_name
    ).execute()


# === Chat API 호출 함수 (대화 이력 지원) ===
def ask_rag_chatbot(query: str, chat_history: list[dict]) -> str:
    docs = retrieve_relevant_chunks(query, top_k=3)
    rag_text = ""
    for doc in docs:
        rag_text += f"""----------------------
{doc.get("chapter", "")}{doc.get("section", "")}
{doc.get("doc", "")}
----------------------
"""
    FEW_SHOTS = load_prompt("few_shots")
    INSTRUCTION = load_prompt("instruction")

    system_prompt_format = """당신은 아동 상담 전문가입니다.
참고를 위해 주어진 정보는 아동심리 전문 서적에서 사용자의 발화와 관련된 내용을 찾은 것입니다.
해당 내용을 참고해서 쉽고 최대한 구체적으로 답변을 주세요.

### 참고할 이론
{rag_text}

### 답변 작성 지침
{instruction}

### 예시
{few_shots}
    """

    system_prompt = system_prompt_format.format(
        rag_text=rag_text, instruction=INSTRUCTION, few_shots=FEW_SHOTS
    )

    messages = [{"role": "system", "content": system_prompt}] + chat_history
    messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    response = client.chat.completions.create(
        model="gpt-4.1-mini", messages=messages, temperature=0.2
    )
    return response.choices[0].message.content.strip()
