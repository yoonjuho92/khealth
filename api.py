import faiss
import pickle
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

FEW_SHOTS = """
"""

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

    system_prompt_format = """당신은 아동 상담 전문가입니다.
참고를 위해 주어진 정보는 아동심리 전문 서적에서 사용자의 발화와 관련된 내용을 찾은 것입니다.
해당 내용을 참고해서 쉽고 최대한 구체적으로 답변을 주세요.

### 참고할 이론
{rag_text}

### 답변 작성 지침
대안을 제시하기보다는 찾은 내용을 활용해서 이론적인 분석을 하는 데 중점을 두세요.
답변을 할 때 다음의 예시를 참고해서 비슷한 형식으로 답변해 주세요.
형식만 참고하고, 내용은 위의 내용을 바탕으로 작성해 주세요.

### 예시
{few_shots}
    """

    system_prompt = system_prompt_format.format(rag_text=rag_text, few_shots=FEW_SHOTS)

    print("System Prompt:", system_prompt)  # 디버깅용 출력

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
