# Eşik deneyi: nötr kazanımı ve pozitif/negatif kaybı

**11 Kasım 2026 — iki koşu tamamlandı.** Son 3000 yorumluk örneklemde önce **0,07**, ardından **0,10** uygulandı. Her koşuda dört yöntem için 12.000, toplamda **24.000 yeni tahmin** kaydedildi. Öncesinde 0,05 tabanındaki 12.000 tam semantik çıktı yeniden üretildi ve kayıtlarla birebir eşleşti.

**Sonuç:** İki eşikte de daha fazla nötr yakalandı; bütün yöntemlerde bunun karşılığında daha fazla doğru pozitif/negatif yorum kaybedildi. 0,07 hiçbir yöntemde macro-F1 kazandırmadı. 0,10'da yalnız LW'nin macro-F1'i küçük bir artış gösterdi; onun da doğruluğu ve pozitif/negatif recall değerleri düştü. **Kalıcı eşik seçilmedi; üretim kodunun varsayılanı 0,05.**

## Yakalama oranları

Her sınıfta 1000 yorum var. Yüzdeler recall'dır: gerçek sınıftaki doğru tahmin sayısı / 1000. Bu tabloda yüzdeler iki basamakla verilmiştir.

| Yöntem | Eşik | Pozitif recall % | Negatif recall % | Nötr recall % | Macro-F1 % |
|---|---:|---:|---:|---:|---:|
| HisNet | 0,05 | 68,00 | 60,60 | 19,10 | 46,57 |
| HisNet | 0,07 | 67,60 | 59,10 | 19,30 | 46,16 |
| HisNet | 0,10 | 66,20 | 58,00 | 20,80 | 46,23 |
| DP→HisNet | 0,05 | 69,80 | 59,10 | 21,80 | 48,11 |
| DP→HisNet | 0,07 | 69,30 | 57,60 | 22,40 | 47,82 |
| DP→HisNet | 0,10 | 67,70 | 57,10 | 23,70 | 47,88 |
| HisNet→DP | 0,05 | 69,30 | 64,30 | 19,40 | 48,20 |
| HisNet→DP | 0,07 | 68,90 | 63,10 | 19,80 | 47,97 |
| HisNet→DP | 0,10 | 67,50 | 61,90 | 21,20 | 47,98 |
| LW HisNet→DP | 0,05 | 69,80 | 63,30 | 18,30 | 47,52 |
| LW HisNet→DP | 0,07 | 69,30 | 62,70 | 18,80 | 47,48 |
| LW HisNet→DP | 0,10 | 68,30 | 61,50 | 20,40 | 47,68 |

## 0,05'e göre kaç yorum kazanıldı veya kaybedildi?

Pozitif/negatif kayıp, önce doğru sınıflandırılırken yeni eşikte nötrleşen yorum sayısıdır. Bu nedenle bir sınıftaki 10 yorum kaybı, o sınıfın recall değerinde 1 yüzde puanı kayıptır. Net doğru farkı üç sınıfın toplamına ilişkindir.

| Yöntem | Eşik | Nötr kazancı | Pozitif kayıp | Negatif kayıp | Net doğru farkı |
|---|---:|---:|---:|---:|---:|
| HisNet | 0,07 | +2 | 4 | 15 | −17 |
| DP→HisNet | 0,07 | +6 | 5 | 15 | −14 |
| HisNet→DP | 0,07 | +4 | 4 | 12 | −12 |
| LW HisNet→DP | 0,07 | +5 | 5 | 6 | −6 |
| HisNet | 0,10 | +17 | 18 | 26 | −27 |
| DP→HisNet | 0,10 | +19 | 21 | 20 | −22 |
| HisNet→DP | 0,10 | +18 | 18 | 24 | −24 |
| LW HisNet→DP | 0,10 | +21 | 15 | 18 | −12 |

Örneğin **HisNet→DP, 0,10**: 18 ek nötr doğru bulunuyor; 18 pozitif ve 24 negatif doğru karar kayboluyor. Toplam doğru 1530'dan 1506'ya, doğruluk %51,00'dan %50,20'ye düşüyor.

## Precision ve F1 ne söylüyor?

- **Nötr precision, iki eşikte de dört yöntemde düşüyor.** Daha çok nötr demek, nötr tahminlerinin daha güvenilir olduğu anlamına gelmiyor.
- **0,07:** Nötr F1 üç yöntemde az yükseliyor, HisNet'te düşüyor; macro-F1 dört yöntemde de düşüyor.
- **0,10:** Nötr F1 dört yöntemde yükseliyor, fakat pozitif ve negatif F1 dört yöntemde de düşüyor.
- **LW istisnası:** 0,10'da nötr F1 24,63'ten 26,19'a çıkıyor; pozitif F1 59,20'den 58,83'e ve negatif F1 58,72'den 58,02'ye düşüyor. Sınıf F1'lerinin ortalaması olan macro-F1, 47,52'den 47,68'e yükselirken toplam doğruluk %50,47'den %50,07'ye düşüyor.

Son maddede çelişki yok: macro-F1 sınıf başına precision ve recall dengesini, accuracy toplam doğru sayısını ölçüyor. LW'deki +0,16 puan, bu geliştirme örnekleminde gözlenen küçük bir farktır; istatistiksel üstünlük iddiası değildir. Diğer üç yöntemde denenen eşikler arasında en yüksek macro-F1 ve doğruluk 0,05'te kaldı.

## Nasıl çalıştırıldı ve doğrulandı?

Özgün morfoloji, parse ve sözlük sorguları kayıtlı 3000 koşusundan alındı. Üretimdeki dört sınıflandırıcının skor toplama/kapsam/bileşim ve karar adımları yeniden çalıştırıldı. Bu **kayıtlı girdilerle yeniden hesaplama** yöntemidir; morfoloji ve Stanza modellerinin baştan çalıştırıldığı yeni uçtan uca koşu değildir. Eşik yalnız karar aşamasını etkilediğinden aynı girdiler kontrollü karşılaştırma sağlar.

- Kaynak: [son 3000 koşusu](../trsav1_classical_3000_20261101_parallel/report.md).
- Ön kontrol: 0,05'te **12.000/12.000** tam semantik çıktı eşleşti.
- Yeni koşular: 0,07'de **12.000/12.000**, 0,10'da **12.000/12.000** başarılı çıktı.
- Etiket ve eşik açıklama alanları dışındaki bütün semantik alanlar aynı kaldı; pozitif/negatif puan, kapsam ve ağırlık değişmedi.
- Üretim kararları, aynı donmuş puanlar üzerindeki bağımsız eşik formülüyle eşleşti. Sonrasında ayrı PowerShell hesabı da 24.000 etiketi, metin/etiket eşliğini ve karışıklık matrislerini doğruladı.
- **8 hedefli test geçti:** eşik sınırları, kayan nokta toleransı, bağlamdan çıkınca varsayılanın korunması, eksik sözlük kaydında hata verilmesi, skor değişiminin reddi, kazanç/kayıp hesabı ve mevcut önbellek kontrolleri.
- Kaynak SQLite SHA-256 önce/sonra aynı: `823a3ff2facf2ed32ae0222ea740aade2cf0a7bad74498adf8e20b1a3754fe57`.
- Üretim kaynak dosyaları değişmedi; yalnız deney scripti, testleri ve yeni sonuçlar eklendi. LLM çağrısı yapılmadı. Yeni bir yöntem veya yeni sözlük maddesi eklenmedi.

Bu 3000 örnek geliştirme verisidir. Sonuçlar eşik seçimine yardımcı olur; bağımsız final test başarısı olarak yorumlanmamalıdır. Bu çalışmada otomatik eşik seçimi veya başka bir eşik taraması yapılmadı.

## Dosyalar ve tekrar çalıştırma

- [Bütün precision/recall/F1 ve doğruluk tabloları](report.md)
- [Sınıf metrikleri ve 0,05'e göre değişimler](per_class_comparison.csv)
- [Genel metrikler](metrics_comparison.csv)
- [Değişen tahminler ve yorum metinleri](changed_predictions.csv)
- [Kazanılan/kaybedilen örnek sayıları](tradeoffs.csv)
- [Tam sayısal özet ve karışıklık matrisleri](comparison.json)
- [Koşu doğrulaması](verification.json), [bağımsız hesap doğrulaması](independent_audit.json)
- [0,07 tahminleri](threshold_070/predictions.csv), [0,10 tahminleri](threshold_100/predictions.csv)

İki alt dizinin `progress.sqlite3` dosyaları tam ara çıktıları içerir. Varsayılan eşik kodda 0,05 kalır; script 0,07 ve 0,10'u sırayla yalnız çalışma bağlamında uygular ve sonra eski değeri geri getirir. Tekrar aynı komut verilirse kayıtlı başarılı sonuçlar doğrulanır; mükerrer kayıt oluşturulmaz.

```powershell
.venv\Scripts\python.exe -u -m scripts.run_classical_threshold_experiment --source results/trsav1_classical_3000_20261101_parallel --output results/trsav1_classical_3000_thresholds_20261111
```
