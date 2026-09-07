import pandas as pd
import csv

# ============================================================
# DOSYA YOLU
# ============================================================

input_file = r"C:\Users\ASUS\Desktop\datasets\Modified\sentence_sentiment\tr_polarity_combined.csv"


# ============================================================
# AYIRICIYI BUL
# ============================================================

with open(input_file, "r", encoding="utf-8-sig") as f:
    sample = f.read(10000)

try:
    delimiter = csv.Sniffer().sniff(
        sample,
        delimiters=";,\t"
    ).delimiter
except csv.Error:
    delimiter = ";"

print("Bulunan ayırıcı:", repr(delimiter))


# ============================================================
# CSV'Yİ OKU
# ============================================================

df = pd.read_csv(
    input_file,
    sep=delimiter,
    encoding="utf-8-sig",
    engine="python"
)

print("Sütunlar:", df.columns.tolist())

# İkinci sütun
second_column = df.columns[1]


# ============================================================
# ETİKETLERİ DÜZELT
# ============================================================

df[second_column] = (
    df[second_column]
    .astype(str)
    .str.strip()
    .str.lower()
    .replace({
        "neg": "Negative",
        "pos": "Positive"
    })
)


# ============================================================
# AYNI DOSYANIN ÜZERİNE KAYDET
# ============================================================

df.to_csv(
    input_file,
    index=False,
    sep=delimiter,
    encoding="utf-8-sig"
)


# ============================================================
# KONTROL
# ============================================================

print("\nİşlem tamamlandı.")

print("\nYeni etiket dağılımı:")
print(df[second_column].value_counts())