import pandas as pd

# ============================================================
# DOSYA YOLLARI
# ============================================================

input_file = r"C:\Users\ASUS\Desktop\datasets\HUMIRSentimentDatasets.csv"
output_file = r"C:\Users\ASUS\Desktop\datasets\Modified\HUMIRSentimentDatasets.csv"

# ============================================================
# CSV DOSYASINI OKU
# ============================================================

# sep=None sayesinde , ; veya tab gibi ayırıcıyı otomatik bulmaya çalışır
df = pd.read_csv(
    input_file,
    sep=None,
    engine="python",
    encoding="utf-8-sig"
)

print("Bulunan sütunlar:")
print(df.columns.tolist())

# ============================================================
# SADECE C VE D SÜTUNLARINI AL
# C = index 2
# D = index 3
# ============================================================

df_clean = df.iloc[:, [2, 3]].copy()

# Yeni sütun isimleri
df_clean.columns = ["review", "score"]

# ============================================================
# TEMİZLİK
# ============================================================

# Boş review veya score olanları sil
df_clean = df_clean.dropna(subset=["review", "score"])

# Review metnindeki gereksiz boşlukları temizle
df_clean["review"] = (
    df_clean["review"]
    .astype(str)
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

# Score değerlerini temizle
df_clean["score"] = (
    df_clean["score"]
    .astype(str)
    .str.strip()
)

# ============================================================
# CSV OLARAK KAYDET
# ============================================================

df_clean.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig",
    sep=";"
)

# ============================================================
# KONTROL
# ============================================================

print("\nDosya başarıyla oluşturuldu:")
print(output_file)

print("\nToplam kayıt:", len(df_clean))

print("\nEtiket dağılımı:")
print(df_clean["score"].value_counts())

print("\nİlk 5 kayıt:")
print(df_clean.head())