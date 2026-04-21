import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd

# 1. Temiz veriyi yükle
print("Veri yükleniyor...")
df = pd.read_csv('rag_hazir_veri.csv')

# 2. Modeli hazırla
print("Model yükleniyor...")
model = SentenceTransformer('intfloat/multilingual-e5-small')

# 3. Metinleri vektörlere dönüştür
print("Vektörler oluşturuluyor (biraz sürebilir)...")
content_list = df['content'].fillna("").tolist()
embeddings = model.encode(content_list, show_progress_bar=True)

# 4. FAISS İndeksi oluştur
dimension = embeddings.shape[1] 
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype('float32'))

# 5. İndeksi kaydet
faiss.write_index(index, 'medical_index.faiss')
print("--- BAŞARILI: 'medical_index.faiss' dosyası oluşturuldu! ---")