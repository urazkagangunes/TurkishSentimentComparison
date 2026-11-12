# 3000 yorum: 0,05 / 0,07 / 0,10 eşik karşılaştırması

Aynı 3000 yorum ve dört klasik yöntem. Önce 0,05 tabanı birebir doğrulandı; sonra 0,07 ve 0,10 sırayla çalıştırıldı.
Morfoloji, parse, sözlük puanları ve bütün bileşim/kapsam alanları sabit tutuldu. Üretim sınıflandırıcıları kayıtlı girdilerle yeniden hesaplandı; uçtan uca yeni ön işleme yapılmadı. LLM kullanılmadı.

Eşik t için P−N > t pozitif, P−N < −t negatif, sınırlar dahil aradaki bölge nötrdür. Sınırdaki kayan nokta toleransı mevcut üretim kuralıyla aynıdır.

## Yakalama oranları ve genel performans

Her sınıfta 1000 yorum var. Recall, gerçek sınıfın doğru yakalanan oranıdır. Bütün yüzde değerleri yüzde ölçeğindedir.

| Yöntem | Eşik | Pozitif recall % | Negatif recall % | Nötr recall % | Accuracy % | Macro-F1 % |
| --- | --- | --- | --- | --- | --- | --- |
| HisNet | 0.05 | 68.00 | 60.60 | 19.10 | 49.23 | 46.57 |
| HisNet | 0.07 | 67.60 | 59.10 | 19.30 | 48.67 | 46.16 |
| HisNet | 0.10 | 66.20 | 58.00 | 20.80 | 48.33 | 46.23 |
| DP → HisNet | 0.05 | 69.80 | 59.10 | 21.80 | 50.23 | 48.11 |
| DP → HisNet | 0.07 | 69.30 | 57.60 | 22.40 | 49.77 | 47.82 |
| DP → HisNet | 0.10 | 67.70 | 57.10 | 23.70 | 49.50 | 47.88 |
| HisNet → DP | 0.05 | 69.30 | 64.30 | 19.40 | 51.00 | 48.20 |
| HisNet → DP | 0.07 | 68.90 | 63.10 | 19.80 | 50.60 | 47.97 |
| HisNet → DP | 0.10 | 67.50 | 61.90 | 21.20 | 50.20 | 47.98 |
| LW HisNet → DP | 0.05 | 69.80 | 63.30 | 18.30 | 50.47 | 47.52 |
| LW HisNet → DP | 0.07 | 69.30 | 62.70 | 18.80 | 50.27 | 47.48 |
| LW HisNet → DP | 0.10 | 68.30 | 61.50 | 20.40 | 50.07 | 47.68 |

## 0,05'e göre kazanç ve kayıp

Sayılar yorum adedidir. Kaybedilen pozitif/negatif, önceden doğru olan fakat yeni eşikte nötrleşen yorumdur.

| Yöntem | Eşik | Nötr kazanılan | Pozitif kayıp | Negatif kayıp | Net doğru | Etiketi değişen |
| --- | --- | --- | --- | --- | --- | --- |
| HisNet | 0.07 | 2 | 4 | 15 | -17 | 27 |
| DP → HisNet | 0.07 | 6 | 5 | 15 | -14 | 28 |
| HisNet → DP | 0.07 | 4 | 4 | 12 | -12 | 25 |
| LW HisNet → DP | 0.07 | 5 | 5 | 6 | -6 | 21 |
| HisNet | 0.10 | 17 | 18 | 26 | -27 | 85 |
| DP → HisNet | 0.10 | 19 | 21 | 20 | -22 | 79 |
| HisNet → DP | 0.10 | 18 | 18 | 24 | -24 | 85 |
| LW HisNet → DP | 0.10 | 21 | 15 | 18 | -12 | 72 |

## Sınıf bazında precision / recall / F1

| Yöntem | Eşik | Sınıf | Doğru / gerçek | Precision % | Recall % | F1 % | ΔF1 puan (0,05'e göre) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HisNet | 0.05 | negative | 606/1000 | 53.20 | 60.60 | 56.66 | +0.00 |
| HisNet | 0.05 | neutral | 191/1000 | 38.28 | 19.10 | 25.48 | +0.00 |
| HisNet | 0.05 | positive | 680/1000 | 49.93 | 68.00 | 57.58 | +0.00 |
| DP → HisNet | 0.05 | negative | 591/1000 | 56.83 | 59.10 | 57.94 | +0.00 |
| DP → HisNet | 0.05 | neutral | 218/1000 | 36.45 | 21.80 | 27.28 | +0.00 |
| DP → HisNet | 0.05 | positive | 698/1000 | 51.25 | 69.80 | 59.10 | +0.00 |
| HisNet → DP | 0.05 | negative | 643/1000 | 55.57 | 64.30 | 59.62 | +0.00 |
| HisNet → DP | 0.05 | neutral | 194/1000 | 39.27 | 19.40 | 25.97 | +0.00 |
| HisNet → DP | 0.05 | positive | 693/1000 | 51.37 | 69.30 | 59.00 | +0.00 |
| LW HisNet → DP | 0.05 | negative | 633/1000 | 54.76 | 63.30 | 58.72 | +0.00 |
| LW HisNet → DP | 0.05 | neutral | 183/1000 | 37.65 | 18.30 | 24.63 | +0.00 |
| LW HisNet → DP | 0.05 | positive | 698/1000 | 51.40 | 69.80 | 59.20 | +0.00 |
| HisNet | 0.07 | negative | 591/1000 | 52.82 | 59.10 | 55.78 | -0.88 |
| HisNet | 0.07 | neutral | 193/1000 | 36.69 | 19.30 | 25.29 | -0.19 |
| HisNet | 0.07 | positive | 676/1000 | 49.89 | 67.60 | 57.41 | -0.17 |
| DP → HisNet | 0.07 | negative | 576/1000 | 56.30 | 57.60 | 56.95 | -1.00 |
| DP → HisNet | 0.07 | neutral | 224/1000 | 35.78 | 22.40 | 27.55 | +0.27 |
| DP → HisNet | 0.07 | positive | 693/1000 | 51.30 | 69.30 | 58.95 | -0.15 |
| HisNet → DP | 0.07 | negative | 631/1000 | 55.30 | 63.10 | 58.94 | -0.68 |
| HisNet → DP | 0.07 | neutral | 198/1000 | 38.15 | 19.80 | 26.07 | +0.10 |
| HisNet → DP | 0.07 | positive | 689/1000 | 51.42 | 68.90 | 58.89 | -0.11 |
| LW HisNet → DP | 0.07 | negative | 627/1000 | 54.71 | 62.70 | 58.43 | -0.29 |
| LW HisNet → DP | 0.07 | neutral | 188/1000 | 37.08 | 18.80 | 24.95 | +0.32 |
| LW HisNet → DP | 0.07 | positive | 693/1000 | 51.45 | 69.30 | 59.05 | -0.15 |
| HisNet | 0.10 | negative | 580/1000 | 53.02 | 58.00 | 55.40 | -1.27 |
| HisNet | 0.10 | neutral | 208/1000 | 35.62 | 20.80 | 26.26 | +0.78 |
| HisNet | 0.10 | positive | 662/1000 | 50.08 | 66.20 | 57.02 | -0.56 |
| DP → HisNet | 0.10 | negative | 571/1000 | 56.70 | 57.10 | 56.90 | -1.04 |
| DP → HisNet | 0.10 | neutral | 237/1000 | 35.01 | 23.70 | 28.26 | +0.98 |
| DP → HisNet | 0.10 | positive | 677/1000 | 51.44 | 67.70 | 58.46 | -0.64 |
| HisNet → DP | 0.10 | negative | 619/1000 | 55.52 | 61.90 | 58.53 | -1.09 |
| HisNet → DP | 0.10 | neutral | 212/1000 | 36.61 | 21.20 | 26.85 | +0.88 |
| HisNet → DP | 0.10 | positive | 675/1000 | 51.68 | 67.50 | 58.54 | -0.46 |
| LW HisNet → DP | 0.10 | negative | 615/1000 | 54.91 | 61.50 | 58.02 | -0.70 |
| LW HisNet → DP | 0.10 | neutral | 204/1000 | 36.56 | 20.40 | 26.19 | +1.56 |
| LW HisNet → DP | 0.10 | positive | 683/1000 | 51.66 | 68.30 | 58.83 | -0.37 |

## Deney sınırı ve kayıtlar

Bu bir geliştirme/eşik seçimi deneyidir; bağımsız final test değildir. Kalıcı eşik seçilmedi; kaynak koddaki varsayılan 0,05 korunuyor. 0,07 ve 0,10 yalnız ayrı deney süreçlerinde uygulandı.

LW katsayıları değiştirilmedi. LW çıktısındaki sabit 0,05 açıklama alanı deney kaydında gerçek eşikle düzeltildi; literature_weighting.dataset_tuned alanı sabit ağırlıklara ilişkindir. Eşik denemeleri açıkça geliştirme verisi kullanır.

- [Genel metrikler](metrics_comparison.csv)
- [Sınıf metrikleri ve farklar](per_class_comparison.csv)
- [Kazanç/kayıp sayıları](tradeoffs.csv)
- [Etiketi değişen yorumlar](changed_predictions.csv)
- [Karışıklık matrisleri ve tam sayısal özet](comparison.json)
- [Doğrulama](verification.json)
- Her eşik dizinindeki progress.sqlite3 tam ara çıktıları, predictions.csv bütün tahminleri içerir.
