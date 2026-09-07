import pandas as pd
from ftfy import fix_text

# ============================================================
# DOSYA YOLLARI
# ============================================================

input_file = r"C:\Users\ASUS\Desktop\datasets\fsmtsa.csv"
output_file = r"C:\Users\ASUS\Desktop\datasets\Modified\fsmtsa.csv"


# ============================================================
# CSV DOSYASINI OKU
# ============================================================

# Orijinal dosyanın gerçek ayırıcısı virgül.
# Review içerisinde virgül bulunan cümleler tırnak içinde olduğu için
# pandas bunları doğru şekilde okuyacaktır.
df = pd.read_csv(
    input_file,
    sep=",",
    quotechar='"',
    encoding="utf-8-sig",
    engine="python"
)

print("Orijinal sütunlar:")
print(df.columns.tolist())


# ============================================================
# SADECE SENTENCE VE LABEL AL
# ============================================================

# Olası boşlukları sütun isimlerinden temizle
df.columns = df.columns.str.strip()

df_clean = df[["Sentence", "Label"]].copy()

# Standart isimlerimiz
df_clean.columns = ["review", "score"]


# ============================================================
# TÜRKÇE KARAKTERLERİ DÜZELT
# ============================================================

df_clean["review"] = (
    df_clean["review"]
    .astype(str)
    .apply(fix_text)
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)


# ============================================================
# SCORE TEMİZLE
# ============================================================

df_clean["score"] = pd.to_numeric(
    df_clean["score"],
    errors="coerce"
)

# Hatalı / boş kayıtları çıkar
df_clean = df_clean.dropna(
    subset=["review", "score"]
)

# 0, 1, 2 değerlerini integer olarak tut
df_clean["score"] = df_clean["score"].astype(int)


# ============================================================
# EXCEL UYUMLU CSV KAYDET
# ============================================================

# Türkçe Excel'de sütunların A ve B olarak ayrılması için ;
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

print(f"\nToplam kayıt: {len(df_clean)}")

print("\nEtiket dağılımı:")
print(df_clean["score"].value_counts().sort_index())

print("\nİlk 10 kayıt:")
print(df_clean.head(10))