import json
import csv

# ============================================================
# DOSYA YOLLARI
# ============================================================

input_file = r"C:\Users\ASUS\Desktop\datasets\best_movies_reviews.json"
output_file = r"C:\Users\ASUS\Desktop\datasets\Modified\best_movies_reviews.csv"

rows = []

# ============================================================
# JSON DOSYASINI OKU
# ============================================================

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# ============================================================
# TÜM FİLMLERDEKİ YORUMLARI AL
# ============================================================

for movie in data:

    reviews = movie.get("reviews", [])

    for item in reviews:

        review = item.get("review")
        rating = item.get("rating")

        # Eksik kayıtları atla
        if review is None or rating is None:
            continue

        review = str(review).strip()
        rating = str(rating).strip()

        # Fazla boşlukları temizle
        review = " ".join(review.split())

        if not review:
            continue

        rows.append({
            "review": review,
            "score": rating
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

print("Dosya başarıyla oluşturuldu:")
print(output_file)

print(f"\nToplam yorum sayısı: {len(rows)}")

print("\nİlk 5 kayıt:")
for row in rows[:5]:
    print(row)