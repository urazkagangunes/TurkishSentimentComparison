import csv
import os

# ============================================================
# KLASÖR
# ============================================================

folder = r"C:\Users\ASUS\Desktop\datasets"

output_file = os.path.join(
    folder,
    "reviews_combined.csv"
)

# Dosya -> score
files = {
    "reviews.neg": "neg",
    "reviews.pos": "pos"
}

rows = []


# ============================================================
# DOSYALARI OKU
# ============================================================

for filename, score in files.items():

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

            # Gereksiz fazla boşlukları teke indir
            review = " ".join(review.split())

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
        delimiter=";"   # Excel'de A ve B sütununa ayrı düşmesi için
    )

    writer.writeheader()
    writer.writerows(rows)


# ============================================================
# KONTROL
# ============================================================

negative_count = sum(
    1 for row in rows
    if row["score"] == "neg"
)

positive_count = sum(
    1 for row in rows
    if row["score"] == "pos"
)

print("\nDosya başarıyla oluşturuldu:")
print(output_file)

print(f"\nToplam kayıt : {len(rows)}")
print(f"neg          : {negative_count}")
print(f"pos          : {positive_count}")

print("\nİlk 10 kayıt:")
for row in rows[:10]:
    print(row)