import os
import csv
import unicodedata
from ftfy import fix_text

folder = r"C:\Users\ASUS\Desktop\datasets"

files = [
    "Turkish_twitter_train.csv",
    "Turkish_twitter_test.csv"
]

output_file = os.path.join(
    folder,
    "Turkish_twitter_combined_fixed.csv"
)

label_mapping = {
    "P": "Positive",
    "N": "Negative"
}

rows = []


def fix_broken_emoji(text):
    result = []

    for char in text:
        code = ord(char)

        # Örn:
        # U+F602 -> U+1F602 😂
        # U+F431 -> U+1F431 🐱
        if 0xF000 <= code <= 0xF8FF:
            candidate_code = code + 0x10000

            try:
                candidate = chr(candidate_code)
                unicodedata.name(candidate)

                result.append(candidate)
                continue

            except (ValueError, OverflowError):
                pass

        result.append(char)

    return "".join(result)


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
        if line.lower().replace(" ", "") == "text,sentiment":
            continue

        if "," not in line:
            continue

        # Sadece son virgülden ayır
        review, old_score = line.rsplit(",", 1)

        review = review.strip()
        old_score = old_score.strip().upper()

        # Çevreleyen CSV tırnaklarını kaldır
        if (
            len(review) >= 2
            and review.startswith('"')
            and review.endswith('"')
        ):
            review = review[1:-1]

        # ==============================================
        # ENCODING DÜZELT
        # ==============================================

        review = fix_text(review)

        # ==============================================
        # BOZULMUŞ EMOJİLERİ DÜZELT
        # ==============================================

        review = fix_broken_emoji(review)

        # Fazla whitespace
        review = " ".join(review.split())

        # ==============================================
        # ETİKET
        # ==============================================

        score = label_mapping.get(old_score)

        if score is None:
            continue

        if not review:
            continue

        rows.append({
            "review": review,
            "score": score
        })


# ============================================================
# CSV KAYDET
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


print("\nDosya oluşturuldu:")
print(output_file)

print(f"\nToplam kayıt: {len(rows)}")

print(
    "Positive:",
    sum(x["score"] == "Positive" for x in rows)
)

print(
    "Negative:",
    sum(x["score"] == "Negative" for x in rows)
)


# ============================================================
# HALA BOZUK KARAKTER KALMIŞ MI?
# ============================================================

problem_chars = ["�", "Ã", "Ä", "Å"]

problematic = [
    x for x in rows
    if any(c in x["review"] for c in problem_chars)
]

print(
    "\nŞüpheli karakter kalan kayıt:",
    len(problematic)
)

if problematic:
    print("\nİlk 10 problemli satır:")
    for x in problematic[:10]:
        print(x["review"])