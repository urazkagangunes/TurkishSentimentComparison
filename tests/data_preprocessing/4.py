import csv

# ============================================================
# DOSYA YOLLARI
# ============================================================

input_file = r"C:\Users\ASUS\Desktop\datasets\beyazperde_sentiment.csv"
output_file = r"C:\Users\ASUS\Desktop\datasets\Modified\beyazperde_sentiment.csv"

rows = []

# Etiketleri diğer datasetlerle aynı hale getir
label_mapping = {
    "pozitif": "Positive",
    "negatif": "Negative",
    "notr": "Neutral",
    "nötr": "Neutral"
}

# ============================================================
# DOSYAYI OKU
# ============================================================

with open(input_file, "r", encoding="utf-8-sig", errors="replace") as f:

    for line_number, line in enumerate(f):

        line = line.strip()

        if not line:
            continue

        # Header'ı atla
        if line.lower().replace(" ", "") == "cumle,etiket":
            continue

        # ÖNEMLİ:
        # Cümlede virgül bulunabileceği için SADECE SON virgülden böl
        if "," not in line:
            print(f"Atlandı (virgül bulunamadı): {line_number + 1}")
            continue

        review, label = line.rsplit(",", 1)

        review = review.strip()
        label = label.strip().lower()

        # Gereksiz boşlukları temizle
        review = " ".join(review.split())

        # Etiketi dönüştür
        score = label_mapping.get(label)

        # Bilinmeyen etiket varsa uyar ve atla
        if score is None:
            print(
                f"Bilinmeyen etiket - satır {line_number + 1}: "
                f"{label}"
            )
            continue

        if review:
            rows.append({
                "review": review,
                "score": score
            })


# ============================================================
# EXCEL UYUMLU CSV OLUŞTUR
# ============================================================

with open(
    output_file,
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=["review", "score"],
        delimiter=";"          # Excel'de A ve B sütununa ayrı düşmesi için
    )

    writer.writeheader()
    writer.writerows(rows)


# ============================================================
# KONTROL
# ============================================================

print("\nDosya başarıyla oluşturuldu:")
print(output_file)

print(f"\nToplam kayıt: {len(rows)}")

positive = sum(1 for x in rows if x["score"] == "Positive")
negative = sum(1 for x in rows if x["score"] == "Negative")
neutral = sum(1 for x in rows if x["score"] == "Neutral")

print("\nEtiket dağılımı:")
print(f"Positive : {positive}")
print(f"Negative : {negative}")
print(f"Neutral  : {neutral}")

print("\nİlk 5 kayıt:")
for row in rows[:5]:
    print(row)