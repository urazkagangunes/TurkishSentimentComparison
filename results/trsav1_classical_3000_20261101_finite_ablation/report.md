# Çekimli sıfat yüklemi düzeltmesi — aynı 3000 yorumda önce/sonra

Yeni örneklerde DP→HisNet 2, HisNet→DP ve LW 1'er ek doğru sonuç kazandı.
Doğruyken yanlış hale gelen örnek olmadı. Kazanç küçüktür; genel performans
üstünlüğü iddiası değildir.

| Yöntem | Doğru önce → sonra | Doğruluk önce → sonra | Macro-F1 önce → sonra |
|---|---:|---:|---:|
| HisNet | 1477 → 1477 | %49,23 → %49,23 | %46,57 → %46,57 |
| DP→HisNet | 1505 → 1507 | %50,17 → %50,23 | %48,05 → %48,11 |
| HisNet→DP | 1529 → 1530 | %50,97 → %51,00 | %48,17 → %48,20 |
| LW | 1513 → 1514 | %50,43 → %50,47 | %47,49 → %47,52 |

DP→HisNet'te ara çıktısı değişen 56 yorum ve etiketi değişen 3 yorum var.
Diğer iki bağımlılık yönteminde 2'şer yorumun ara çıktısı, 1'er yorumun etiketi değişti.

- ID 1847: `üzgünüm` adayı korundu; DP→HisNet nötr → doğru negatif.
- ID 2467: “Üzgünüm pek beğenemedim. Küçücük bir şey”; üç bağımlılık kolu
  pozitif → doğru negatif. `üzgün`ün 1,0 negatif puanı artık hedef bastırmasına uğramıyor.
- ID 1052: `kalitesizmiş` eklendi; DP→HisNet pozitif → negatif, gold nötr.
  İki karar da yanlış olduğundan bu değişim iyileşme sayılmadı.

Kontrol edilen tek üretim farkı resolver'dır. Parse, morfoloji, HisNet sorguları,
eşik ve LW ağırlıkları sabit. Önce koşunun güncel resolver'ı ile 9000 bağımlılık
çıktısı birebir yeniden üretildi; ardından aynı girdiler eski resolver'a verildi.
Bu çalışma yeni uçtan uca çıkarım değildir. İlk 1200 örnekteki kaynak veriyle
metin örtüşmesi yoktur; buna rağmen TRSAv1 daha önce kullanılan bir geliştirme
korpusu olduğu için bağımsız final test iddiası yapılmıyor.

Güncel resolver: `dc37555f087cb5124f72c2a3897588a69e8cf7b1310fb82a5b725a0e3535a667`.
Önceki resolver: `bc78f43b41f9b3ceecdb2ddfb96258ec2f750057644285ff2155d84e6a931111`.
Kaynak SQLite SHA-256: `823a3ff2facf2ed32ae0222ea740aade2cf0a7bad74498adf8e20b1a3754fe57`;
salt okunur karşılaştırma sonunda değişmedi.

Güncel hedefli testler: 38/38 geçti. Eski sürüm yeni koruma API'sini içermediğinden
eski test günlüğündeki AttributeError kayıtları beklenir; veri üzerindeki yanlış
tahmin sayısıyla karıştırılmamalıdır.

Dosyalar: [metrikler ve ID listeleri](comparison.json), [tüm önce/sonra kararları](before_after.csv),
[değişen ara çıktılar](changed_details.json), [sınıf metrikleri](per_class_before_after.csv),
[test kaydı](candidate_tests.log), [3000 örneğin kapsamlı yorumu](../trsav1_classical_3000_20261101_parallel/findings.md).
