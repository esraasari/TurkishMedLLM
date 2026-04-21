import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv() # .env dosyasını oku

# 1. Ayarlar
# ÖNEMLİ: Yeni oluşturduğun anahtarı buraya yapıştır!
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("Modeller ve indeks yükleniyor...")
model = SentenceTransformer('intfloat/multilingual-e5-small')
index = faiss.read_index('medical_index.faiss')
df = pd.read_csv('rag_hazir_veri.csv')

def get_answer(soru):
    # A) RETRIEVAL: Veritabanından bilgiyi çek
    sorgu_vektoru = model.encode([soru])
    _, indices = index.search(np.array(sorgu_vektoru).astype('float32'), k=6)
    
    baglam = ""
    for i in indices[0]:
        baglam += df.iloc[i]['content'] + "\n"
    
    print(f"\n--- SİSTEMİN OKUDUĞU METİNLER ---:\n{baglam}\n----------------------------------\n")
    # B) GENERATION: LLM (Llama 3.3) ile cevabı üret
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": """Sen profesyonel bir tıp asistanısın. 
            1. Önce sana verilen 'Bilimsel Bağlam' bölümündeki makalelere bak. 
            2. Cevabı makalelerde bulursan mutlaka makalelerden alıntı yaparak ve kaynak belirterek cevapla. 
            3. Eğer cevabı metinlerde bulamazsan, genel tıbbi bilgilerini kullanarak, dürüstçe 'Verilen metinlerde doğrudan bu bilgi yok ancak genel tıbbi bilgilere göre...' diyerek yardımcı ol."""},
            {"role": "user", "content": f"Verilen tıbbi bilgiler:\n{baglam}\n\nSoru: {soru}"}
        ],
        model="llama-3.3-70b-versatile",
    )
    
    return chat_completion.choices[0].message.content

# 2. Test Et
soru = "Son 3 gündür şiddetli migren ağrım var, ne yapmalıyım?"
cevap = get_answer(soru)
print(f"Cevap: {cevap}")