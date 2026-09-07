import json
import csv

# ============================================================
# DOSYA YOLLARI
# ============================================================

input_file = r"C:\Users\ASUS\Desktop\datasets\urunler-yorumlar.json"
output_file = r"C:\Users\ASUS\Desktop\datasets\Modified\urunler-yorumlar.csv"

# ============================================================
# JSON DOSYASINI OKU
# ============================================================

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

# ============================================================
# TÜM ÜRÜNLERDEKİ YORUMLARI TOPLA
# ============================================================

for product in data:

    reviews = product.get("reviews", [])

    for item in reviews:

        review = item.get("review")
        star = item.get("star")

        # Eksik kayıtları alma
        if review is None or star is None:
            continue

        # Baştaki/sondaki boşlukları temizle
        review = review.strip()

        # Birden fazla boşluğu teke indir
        review = " ".join(review.split())

        # Boş yorumları alma
        if not review:
            continue

        rows.append({
            "review": review,
            "star": star
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
        fieldnames=["review", "star"],
        delimiter=";",              # Excel sütun ayırıcı
        quoting=csv.QUOTE_MINIMAL
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