import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("🔄 Sistem yükleniyor...")
model = SentenceTransformer('intfloat/multilingual-e5-small')
index = faiss.read_index('medical_index.faiss')

with open("metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# -------------------
# RAG FUNCTION
# -------------------

def get_answer(soru):

    # QUERY EMBEDDING
    query_vector = model.encode([f"query: {soru}"])

    distances, indices = index.search(
        np.array(query_vector).astype('float32'),
        k=10
    )

    baglam = ""

    for i in indices[0]:
        baglam += metadata[i]['text'] + "\n"

    baglam = baglam[:3000]

    print("\n--- BULUNAN BAĞLAM ---\n")
    print(baglam)
    print("\n----------------------\n")

    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": """Sen yalnızca Türkçe konuşan profesyonel bir tıp asistanısın.

Kurallar:
- SADECE Türkçe cevap ver.
- İngilizce, Japonca veya başka dil kullanma.
- Tüm cevap tamamen Türkçe olmalı.
- Yabancı kelime kullanma.
- Tıbbi bilgileri açık ve anlaşılır şekilde ver.
- Kesin teşhis koyma.
- Gerekirse doktora yönlendir.
"""
        },
        {
            "role": "user",
            "content": f"Bağlam:\n{baglam}\n\nSoru: {soru}"
        }
    ],
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    top_p=0.8
)

    cevap = chat_completion.choices[0].message.content

    if cevap is None:
        return "⚠️ Model cevap üretemedi."

    return cevap
# -------------------
# TEST
# -------------------

if __name__ == "__main__":

    soru = "Başım ağrıyor napmam lazım?"

    cevap = get_answer(soru)

    print("\n🧾 CEVAP:\n")
    print(cevap)