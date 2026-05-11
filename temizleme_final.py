import pandas as pd
import glob

# 1. Tüm CSV dosyalarını topla
dosyalar = glob.glob('veriler/*.csv')
df_listesi = []

for dosya in dosyalar:
    df_temp = pd.read_csv(dosya)
    
    # Sütunları standartlaştır (Her dosyaya göre güncelleme yap)
    if 'title' in df_temp.columns and 'abstract' in df_temp.columns:
        # Standart makale formatı
        df_temp = df_temp[['title', 'abstract']].rename(columns={'title': 't', 'abstract': 'a'})
    elif 'question' in df_temp.columns and 'answer' in df_temp.columns:
        # Soru-Cevap formatı
        df_temp = df_temp[['question', 'answer']].rename(columns={'question': 't', 'answer': 'a'})
    else:
        # Bilinmeyen format varsa sütunları otomatik al (ilk 2 sütunu al)
        df_temp = df_temp.iloc[:, [0, 1]]
        df_temp.columns = ['t', 'a']
        
    df_listesi.append(df_temp)

# 2. Tümünü birleştir
df = pd.concat(df_listesi, ignore_index=True)
df = df.rename(columns={'t': 'title', 'a': 'abstract'})

# 2. Filtreleme
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
print("--- BAŞARILI: Veri havuzun 7 dosya ile güncellendi! ---")



