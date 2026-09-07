import csv
import re

input_file = r"C:\Users\ASUS\Desktop\datasets\Modified\TRSAv1.csv"
output_file = r"C:\Users\ASUS\Desktop\datasets\Modified\TRSAv1.csv"

rows = []

with open(input_file, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f, delimiter=";")

    for row in reader:
        review = row["review"].strip()
        score = row["score"].strip()

        # Sadece cümlenin SONUNDA bulunan 3 noktalı virgülü sil
        review = re.sub(r";{3}\s*$", "", review).strip()

        rows.append({
            "review": review,
            "score": score
        })

with open(output_file, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["review", "score"],
        delimiter=";"
    )

    writer.writeheader()
    writer.writerows(rows)

print("İşlem tamamlandı.")
print(f"Yeni dosya: {output_file}")
print(f"Toplam kayıt: {len(rows)}")