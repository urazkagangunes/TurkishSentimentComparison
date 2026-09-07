import os
import csv
from ftfy import fix_text

# ============================================================
# DOSYA YOLLARI
# ============================================================

folder = r"C:\Users\ASUS\Desktop\datasets"

files = [
    "train.csv",
    "test.csv"
]

output_file = os.path.join(
    folder,
    "combined_dataset.csv"
)

rows = []
skipped_rows = []


# ============================================================
# ETİKETLERİ STANDARTLAŞTIR
# ============================================================

label_mapping = {
    "positive": "Positive",
    "negative": "Negative",
    "notr": "Notr",
    "neutral": "Notr"
}


# ============================================================
# DOSYAYI OKU
# ============================================================

def read_file(file_path):

    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp1254",
        "iso-8859-9"
    ]

    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                text = f.read()

            print(
                f"{os.path.basename(file_path)} "
                f"-> {encoding} ile okundu"
            )

            return text

        except UnicodeDecodeError:
            continue

    raise RuntimeError(
        f"Dosya okunamadı: {file_path}"
    )


# ============================================================
# TRAIN + TEST BİRLEŞTİR
# ============================================================

for filename in files:

    file_path = os.path.join(folder, filename)

    text = read_file(file_path)

    for line_number, line in enumerate(
        text.splitlines(),
        start=1
    ):

        line = line.strip()

        if not line:
            continue

        # Header
        if line.lower().replace(" ", "") == "text,label,dataset":
            continue

        # ----------------------------------------------------
        # Sağdan iki kere ayır:
        #
        # review , label , dataset
        #
        # Review içerisinde virgül olması problem çıkarmaz.
        # ----------------------------------------------------

        parts = line.rsplit(",", 2)

        if len(parts) != 3:
            skipped_rows.append(
                (filename, line_number, line)
            )
            continue

        review, label, dataset_name = parts

        review = review.strip()
        label = label.strip().lower()

        # Dataset alanını artık kullanmıyoruz
        # dataset_name = dataset_name.strip()

        # ====================================================
        # ENCODING DÜZELT
        # ====================================================

        review = fix_text(review)

        # Baştaki / sondaki gereksiz tırnakları kaldır
        if (
            len(review) >= 2
            and review.startswith('"')
            and review.endswith('"')
        ):
            review = review[1:-1]

        # Fazla boşluk, tab vs. temizle
        review = " ".join(review.split())

        # ====================================================
        # LABEL DÖNÜŞÜMÜ
        # ====================================================

        score = label_mapping.get(label)

        # Geçerli etiket yoksa alma
        if score is None:
            skipped_rows.append(
                (filename, line_number, line)
            )
            continue

        if not review:
            continue

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
        delimiter=";"
    )

    writer.writeheader()
    writer.writerows(rows)


# ============================================================
# KONTROL
# ============================================================

positive_count = sum(
    1 for row in rows
    if row["score"] == "Positive"
)

negative_count = sum(
    1 for row in rows
    if row["score"] == "Negative"
)

notr_count = sum(
    1 for row in rows
    if row["score"] == "Notr"
)


print("\n====================================")
print("İŞLEM TAMAMLANDI")
print("====================================")

print("\nDosya:")
print(output_file)

print(f"\nToplam kayıt : {len(rows)}")
print(f"Positive     : {positive_count}")
print(f"Negative     : {negative_count}")
print(f"Notr         : {notr_count}")

print(
    f"\nAtlanan bozuk/etiketsiz satır: "
    f"{len(skipped_rows)}"
)

if skipped_rows:
    print("\nİlk 10 atlanan satır:")

    for filename, line_number, line in skipped_rows[:10]:
        print(
            f"{filename} - satır {line_number}: "
            f"{line[:150]}"
        )