# 30 cümlelik canlı altı-model kontrolü

Bu set elle etiketlenmiş geliştirme/teşhis setidir; paper benchmark sonucu değildir.

LLM modeli: `qwen3:32b`; digest: `030ee887880fc378860c2dd35101da424377520441ae4bfe7be6deff8ade7840`.
Ayarlar: `{"batch_size": 3, "max_retries": 1, "num_ctx": 8192, "seed": 42, "temperature": 0.1, "think": false, "top_p": 1.0}`.

| Sistem | Tamamlanan | Doğru | Negatif doğru | Nötr doğru | Pozitif doğru |
|---|---:|---:|---:|---:|---:|
| HisNet | 30/30 | 20/30 | 6/10 | 8/10 | 6/10 |
| DP → HisNet | 30/30 | 25/30 | 8/10 | 9/10 | 8/10 |
| HisNet → DP | 30/30 | 24/30 | 8/10 | 8/10 | 8/10 |
| LW HisNet → DP | 30/30 | 24/30 | 8/10 | 8/10 | 8/10 |
| LLM | 30/30 | 28/30 | 9/10 | 10/10 | 9/10 |
| DP → LLM | 30/30 | 29/30 | 9/10 | 10/10 | 10/10 |

## Cümle bazında karşılaştırma

| ID | Metin | Beklenen | HisNet | DP → HisNet | HisNet → DP | LW HisNet → DP | LLM | DP → LLM |
|---|---|---|---|---|---|---|---|---|
| N01 | Film çok kötü. | negative | negative | negative | negative | negative | negative | negative |
| N02 | Bu ürünü hiç beğenmedim. | negative | negative | negative | negative | negative | negative | negative |
| N03 | Bu filmi sevmedim. | negative | neutral | neutral | neutral | neutral | negative | negative |
| N04 | Hizmet berbattı. | negative | neutral | neutral | neutral | neutral | negative | negative |
| N05 | Otel güzel değildi. | negative | positive | negative | negative | negative | negative | negative |
| N06 | Bu ürünü tavsiye etmem. | negative | positive | negative | negative | negative | negative | negative |
| N07 | Bu alışverişten pişman oldum. | negative | negative | negative | negative | negative | negative | negative |
| N08 | Otel güzel ama odalar çok kötü. | negative | negative | negative | negative | negative | neutral | neutral |
| N09 | Ürün kullanışsız. | negative | negative | negative | negative | negative | negative | negative |
| N10 | Kötü. Merkezi değil. | negative | negative | negative | negative | negative | negative | negative |
| P01 | Film çok güzel. | positive | positive | positive | positive | positive | positive | positive |
| P02 | Bu ürünü çok beğendim. | positive | neutral | neutral | neutral | neutral | positive | positive |
| P03 | Bu filmi sevdim. | positive | neutral | neutral | neutral | neutral | positive | positive |
| P04 | Hizmet mükemmeldi. | positive | positive | positive | positive | positive | positive | positive |
| P05 | Otel kötü değildi. | positive | negative | positive | positive | positive | neutral | positive |
| P06 | Bu ürünü tavsiye ederim. | positive | positive | positive | positive | positive | positive | positive |
| P07 | Bu alışverişten pişman olmadım. | positive | negative | positive | positive | positive | positive | positive |
| P08 | Oda çok ama çok güzel. | positive | positive | positive | positive | positive | positive | positive |
| P09 | İYİ BİR ÜRÜN, ÇOK MEMNUNUM. | positive | positive | positive | positive | positive | positive | positive |
| P10 | Bu ürün harika🙂 | positive | positive | positive | positive | positive | positive | positive |
| U01 | Ürün dün geldi. | neutral | neutral | neutral | neutral | neutral | neutral | neutral |
| U02 | Kutuda üç kalem var. | neutral | neutral | neutral | neutral | neutral | neutral | neutral |
| U03 | Film iki saat sürüyor. | neutral | neutral | neutral | neutral | neutral | neutral | neutral |
| U04 | Otelin beş katı var. | neutral | negative | neutral | negative | negative | neutral | neutral |
| U05 | Kargo salı günü teslim edildi. | neutral | neutral | neutral | neutral | neutral | neutral | neutral |
| U06 | Bu kitabı bir yazar yazar. | neutral | neutral | neutral | neutral | neutral | neutral | neutral |
| U07 | İSTANBUL'DA BİR OTEL VAR. | neutral | neutral | neutral | neutral | neutral | neutral | neutral |
| U08 | Oda ikinci katta. | neutral | neutral | neutral | neutral | neutral | neutral | neutral |
| U09 | Ürünün modeli X200. | neutral | positive | positive | positive | positive | neutral | neutral |
| U10 | Paket masanın üzerinde. | neutral | neutral | neutral | neutral | neutral | neutral | neutral |

## Çalışma ve süre kayıtları

Süreler tamamlanan/kaydedilen istek denemelerinden toplanır. Kesilen ve yanıtı kaydedilmeyen istekler bu toplama dahil değildir. Klasik kollarda HisNet sonuçları yeniden kullanıldığından bu koşu adil bir uçtan uca hız benchmark'ı değildir.

- LLM: 10 istek, 827.5 saniye; girdi 1909, çıktı 502 token; istek başına girdi 185–199 token; 0 kayıtlı istek hatası.
- DP → LLM: 10 istek, 657.1 saniye; girdi 10616, çıktı 510 token; istek başına girdi 918–1214 token; 0 kayıtlı istek hatası.

Token kullanımı sunucunun döndürdüğü sayılardır. Bu küçük setin context gözlemi, uzun gerçek yorumlarda truncation olmayacağını kanıtlamaz.
