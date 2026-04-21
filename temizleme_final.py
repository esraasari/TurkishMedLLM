import pandas as pd
import glob
import os

# 1. Tüm CSV dosyalarını 'veriler/' klasöründen topla
dosyalar = glob.glob('veriler/*.csv')
df_listesi = []

for dosya in dosyalar:
    df_temp = pd.read_csv(dosya)
    # Her dosyadaki title ve abstract sütunlarını standartlaştır
    # Sadece title ve abstract varsa onları al
    df_temp = df_temp[['title', 'abstract']] 
    df_listesi.append(df_temp)

# Tümünü birleştir
df = pd.concat(df_listesi, ignore_index=True)

# 2. Filtreleme (Aynı kalsın)
tibbi_terimler = ['tıp', 'sağlık', 'hastalık', 'tedavi', 'cerrahi', 'klinik', 'hasta', 'ilaç', 'semptom', 'genetik', 'diyabet', 'psikoloji']
yasakli_terimler = ['iktisat', 'ekonomi', 'siyaset', 'sanayileşme', 'tarih', 'sosyoloji']

def tibbi_mi(row):
    text = f"{str(row['abstract'])} {str(row['title'])}".lower()
    if any(yasak in text for yasak in yasakli_terimler): return False
    return any(terim in text for terim in tibbi_terimler)

df_filtered = df[df.apply(tibbi_mi, axis=1)].copy()
df_filtered['content'] = df_filtered['title'].astype(str) + " " + df_filtered['abstract'].astype(str)

# 3. Kaydet
df_filtered.to_csv('rag_hazir_veri.csv', index=False, encoding='utf-8-sig')
print("--- BAŞARILI: Veri havuzun güncellendi! ---")