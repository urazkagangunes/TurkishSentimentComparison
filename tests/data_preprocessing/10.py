import csv
import os

# ============================================================
# KLASÖR
# ============================================================

folder = r"C:\Users\ASUS\Desktop\datasets"

output_file = os.path.join(
    folder,
    "tr_polarity_combined_clean.csv"
)

files = {
    "tr_polarity.neg": "neg",
    "tr_polarity.pos": "pos"
}

rows = []


# ============================================================
# DOSYAYI DOĞRU ENCODING İLE OKU
# ============================================================

def read_turkish_file(file_path):

    # Bu dataset eski olduğu için önce Türkçe Windows encodingini deniyoruz.
    encodings = [
        "cp1254",
        "iso-8859-9",
        "utf-8-sig",
        "utf-8"
    ]

    with open(file_path, "rb") as f:
        raw_data = f.read()

    for encoding in encodings:
        try:
            text = raw_data.decode(encoding)

            print(
                f"{os.path.basename(file_path)} "
                f"-> {encoding} ile okundu"
            )

            return text

        except UnicodeDecodeError:
            continue

    raise RuntimeError(
        f"{file_path} için uygun encoding bulunamadı."
    )


# ============================================================
# NEG + POS DOSYALARINI BİRLEŞTİR
# ============================================================

for filename, score in files.items():

    file_path = os.path.join(folder, filename)

    text = read_turkish_file(file_path)

    for line in text.splitlines():

        review = line.strip()

        if not review:
            continue

        # Sadece fazla whitespace temizliği
        # Cümlenin içeriğine dokunmuyoruz.
        review = " ".join(review.split())

        rows.append({
            "review": review,
            "score": score
        })


# ============================================================
# EXCEL UYUMLU CSV
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

print("\nDosya oluşturuldu:")
print(output_file)

print(f"\nToplam kayıt: {len(rows)}")

print(
    "neg:",
    sum(1 for row in rows if row["score"] == "neg")
)

print(
    "pos:",
    sum(1 for row in rows if row["score"] == "pos")
)

# � karakteri var mı kontrol et
problem_count = sum(
    1 for row in rows
    if "�" in row["review"]
)

print(f"\nBozuk '�' karakteri bulunan kayıt: {problem_count}")