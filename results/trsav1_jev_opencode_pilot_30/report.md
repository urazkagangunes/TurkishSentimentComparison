# OpenCode Jev 1.13 ile TRSAv1 fixed pilotu

OpenCode'un `jev-1.13-free` uç noktası API anahtarı olmadan 30 yorumun
tamamına yanıt verdi. Her istekte tek bir yorum ve sabit üç sınıflı Türkçe
Choice sorusu gönderildi. Sağlayıcı 30 isteğin tamamında `billing_source=free`
ve `cost=0` bildirdi. Toplam 12.940 girdi ve 1.170 çıktı tokenı kullanıldı.
[Model ve soru](jev_identity.json) · [OpenCode Jev belgeleri](https://opencode.ai/docs/console/models/#jev)

| Sistem | Doğru | Doğruluk | Macro-F1 |
|---|---:|---:|---:|
| HisNet | 13/30 | %43,33 | %38,66 |
| DP → HisNet | 17/30 | %56,67 | %52,77 |
| HisNet → DP | 17/30 | %56,67 | %49,84 |
| Weighted HisNet → DP | 17/30 | %56,67 | %49,84 |
| LLM | 21/30 | %70,00 | %68,33 |
| DP → LLM | 20/30 | %66,67 | %63,68 |
| Jev 1.13 (OpenCode) | 22/30 | %73,33 | %71,35 |

Altı yerel sistemin ölçümleri önceki koşudan alındı. İki SQLite
checkpoint'indeki yorumlar, doğru etiketler ve satır sırası birebir eşleşti.
LLM kolları `qwen3:4b-instruct` kullandı. Jev, LLM'nin yanlış sınıflandırdığı
iki yorumu doğru bildi. LLM ise Jev'in yanlış sınıflandırdığı bir yorumu
doğru bildi. Bu küçük pilotta aradaki net fark bir yorumdu.
[Yerel sistemlerin pilotu](../trsav1_seven_pilot_30/report.md)

| Gerçek etiket | Negatif tahmin | Nötr tahmin | Pozitif tahmin | F1 |
|---|---:|---:|---:|---:|
| Negatif | 9 | 0 | 1 | %75,00 |
| Nötr | 5 | 4 | 1 | %53,33 |
| Pozitif | 0 | 1 | 9 | %85,71 |

[Jev metrikleri](summary.json) · [Satır bazında tahminler](predictions.csv)
