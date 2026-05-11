import pandas as pd
import glob
import re
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

print("🚀 Pipeline başlıyor...")

# -------------------
# 1. DATA LOAD + CLEAN
# -------------------

def clean_text(text):
    text = re.sub(r'<.*?>', '', str(text))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

dosyalar = glob.glob('veriler/*.csv')
df_listesi = []

for dosya in dosyalar:
    df_temp = pd.read_csv(dosya)

    if 'title' in df_temp.columns and 'abstract' in df_temp.columns:
        df_temp = df_temp[['title', 'abstract']]
    elif 'question' in df_temp.columns and 'answer' in df_temp.columns:
        df_temp['title'] = df_temp['question']
        df_temp['abstract'] = df_temp['answer']
        df_temp = df_temp[['title', 'abstract']]
    else:
        df_temp = df_temp.iloc[:, [0, 1]]
        df_temp.columns = ['title', 'abstract']

    df_listesi.append(df_temp)

df = pd.concat(df_listesi, ignore_index=True)

# temizleme
df['content'] = (df['title'].astype(str) + " " + df['abstract'].astype(str)).apply(clean_text)

# kısa metinleri at
df = df[df['content'].str.len() > 100]

print(f"✅ Temiz veri: {len(df)} kayıt")
#
# -------------------
# MEDICAL FILTER
# -------------------

tibbi_terimler = [
    'tıp', 'sağlık', 'hastalık', 'tedavi', 'cerrahi', 
    'klinik', 'hasta', 'ilaç', 'semptom', 'genetik',
    'diyabet', 'kanser', 'enfeksiyon', 'psikoloji',
    'migren', 'ağrı', 'tanı'
]

yasakli_terimler = [
    'iktisat', 'ekonomi', 'siyaset', 
    'tarih', 'sosyoloji', 'sanat'
]

def tibbi_mi(text):
    text = text.lower()
    
    if any(yasak in text for yasak in yasakli_terimler):
        return False
    
    return any(terim in text for terim in tibbi_terimler)


# uygula
df = df[df['content'].apply(tibbi_mi)]

print(f"✅ Medikal filtre sonrası veri: {len(df)}")


# -------------------
# 2. CHUNKING
# -------------------

def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))

    return chunks

chunks = []
metadata = []

for idx, text in enumerate(df['content']):
    text_chunks = chunk_text(text)

    for chunk in text_chunks:
        metadata.append({
            "id": len(chunks),
            "text": chunk,
        })
        chunks.append(chunk)

print(f"✅ Chunk sayısı: {len(chunks)}")

# -------------------
# 3. EMBEDDING
# -------------------

print("🧠 Embedding modeli yükleniyor...")
model = SentenceTransformer('intfloat/multilingual-e5-small')

print("⚡ Embedding oluşturuluyor...")
embeddings = model.encode(
    [f"passage: {c}" for c in chunks],
    show_progress_bar=True
)

# -------------------
# 4. FAISS
# -------------------

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype('float32'))

faiss.write_index(index, 'medical_index.faiss')

# -------------------
# 5. METADATA SAVE
# -------------------

with open("metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("🎉 HER ŞEY TAMAM!")
print("✔ medical_index.faiss oluşturuldu")
print("✔ metadata.json oluşturuldu")