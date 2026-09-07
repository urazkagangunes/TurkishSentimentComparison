import json
import csv

# ============================================================
# DOSYA YOLLARI
# ============================================================

input_file = r"C:\Users\ASUS\Desktop\datasets\all_movies_reviews.json"
output_file = r"C:\Users\ASUS\Desktop\datasets\Modified\all_movies_reviews.csv"

rows = []


# ============================================================
# JSON DOSYASINI OKU
# ============================================================

with open(input_file, "r", encoding="utf-8-sig") as f:
    data = json.load(f)


# ============================================================
# TÜM FİLMLERDEKİ REVIEW + RATING AL
# ============================================================

for movie in data:

    reviews = movie.get("reviews", [])

    for item in reviews:

        review = item.get("review")
        rating = item.get("rating")

        # Eksik kayıt varsa atla
        if review is None or rating is None:
            continue

        # Metinleri string yap ve temizle
        review = str(review).strip()
        rating = str(rating).strip()

        # Fazla boşlukları teke indir
        review = " ".join(review.split())

        # Boş yorum varsa alma
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

print("\nİlk 10 kayıt:")
for row in rows[:10]:
    print(row)