import csv
import os

# ============================================================
# KLASÖR
# ============================================================

folder = r"C:\Users\ASUS\Desktop\datasets"

output_file = os.path.join(
    folder,
    "combined_sentiment.csv"
)

# Dosya -> etiket
files = {
    "negative.txt": "negative",
    "notr.txt": "notr",
    "positive.txt": "positive"
}

rows = []


# ============================================================
# TXT DOSYALARINI OKU
# ============================================================

for filename, label in files.items():

    file_path = os.path.join(folder, filename)

    print(f"Okunuyor: {filename}")

    with open(
        file_path,
        "r",
        encoding="utf-8-sig",
        errors="replace"
    ) as f:

        for line in f:

            review = line.strip()

            # Boş satırları alma
            if not review:
                continue

            # Fazla boşlukları teke indir
            review = " ".join(review.split())

            rows.append({
                "review": review,
                "score": label
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
        delimiter=";"
    )

    writer.writeheader()
    writer.writerows(rows)


# ============================================================
# KONTROL
# ============================================================

print("\nDosya başarıyla oluşturuldu:")
print(output_file)

print(f"\nToplam kayıt: {len(rows)}")

negative_count = sum(
    1 for row in rows if row["score"] == "negative"
)

neutral_count = sum(
    1 for row in rows if row["score"] == "notr"
)

positive_count = sum(
    1 for row in rows if row["score"] == "positive"
)

print("\nEtiket dağılımı:")
print(f"negative : {negative_count}")
print(f"notr     : {neutral_count}")
print(f"positive : {positive_count}")