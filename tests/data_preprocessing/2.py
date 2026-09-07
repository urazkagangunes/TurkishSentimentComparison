import csv
from ftfy import fix_text

# ============================================================
# DOSYA YOLLARI
# ============================================================

input_file = r"C:\Users\ASUS\Desktop\datasets\TRSAv1.csv"
output_file = r"C:\Users\ASUS\Desktop\datasets\Modified\TRSAv1.csv"

rows = []

# ============================================================
# DOSYAYI OKU
# ============================================================

with open(input_file, "r", encoding="utf-8-sig", errors="replace") as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        # Markdown tablo karakterlerini temizle
        if line.startswith("|"):
            line = line[1:]

        if line.endswith("|"):
            line = line[:-1]

        line = line.strip()

        # Markdown ayırıcı satırı
        if line.startswith("---"):
            continue

        # Orijinal veri virgülle ayrılmış:
        # id,score,review
        parsed = next(csv.reader([line], delimiter=","))

        if len(parsed) < 3:
            continue

        # Header'ı atla
        if (
            parsed[0].strip().lower() == "id"
            and parsed[1].strip().lower() == "score"
        ):
            continue

        score = parsed[1].strip()

        # Score başındaki/sonundaki gereksiz virgülleri temizle
        score = score.strip(",")

        # Review içinde virgül olabilir, hepsini tekrar birleştir
        review = ",".join(parsed[2:]).strip()

        # Encoding problemini düzelt
        review = fix_text(review)

        # Fazla boşlukları temizle
        review = " ".join(review.split())

        if review and score:
            rows.append({
                "review": review,
                "score": score
            })


# ============================================================
# EXCEL UYUMLU CSV YAZ
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
        delimiter=";",          # ÖNEMLİ: Excel için ;
        quoting=csv.QUOTE_MINIMAL
    )

    writer.writeheader()
    writer.writerows(rows)


# ============================================================
# KONTROL
# ============================================================

print("Dosya başarıyla oluşturuldu:")
print(output_file)

print(f"\nToplam yorum: {len(rows)}")

positive = sum(
    1 for row in rows
    if row["score"].lower() == "positive"
)

negative = sum(
    1 for row in rows
    if row["score"].lower() == "negative"
)

neutral = sum(
    1 for row in rows
    if row["score"].lower() == "neutral"
)

print("\nEtiket dağılımı:")
print(f"Positive : {positive}")
print(f"Negative : {negative}")
print(f"Neutral  : {neutral}")

# İlk 5 kaydı kontrol amaçlı göster
print("\nİlk 5 kayıt:")
for row in rows[:5]:
    print(row)