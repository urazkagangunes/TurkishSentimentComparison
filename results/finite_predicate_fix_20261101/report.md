# Çekimli sıfat yüklemlerini koruma — 1 Kasım 2026

“Pişmanım” gibi çekimli sıfat yüklemleri, hatalı bağımlılık çözümlemesinde bir
başka sözcüğün hedefi sayılınca duygu puanını kaybediyordu. DP→HisNet aynı sözcüğü
sözlüğe bakmadan önce aday listesinden çıkarabiliyordu. Morfolojide normal analizle
doğrulanan ADJ→VERB çekimi artık hedef bastırmasını önlüyor ve mevcut kurallardan
geçemeyen uygun bağımlılık adayını koruyor. Kural belirli bir kök, etiket veya
HisNet puanı aramıyor.

Koruma, yüzey ve kök eşleşmesi gerektirir; çelişkili tekrarlanan analizler,
robust analiz, isim çekimleri ve fiilimsiler koruma sağlamaz. Sıradan fiillerin
DP aday seçimi değişmedi. Sözlük, eşik (0,05), model kaynakları ve LW ağırlıkları
değiştirilmedi.

## 1200 örnekte önce / sonra

“Önce”, önceki olumsuzluk ve tekrar eşleştirme düzeltmeleri uygulanmış sürümdür.
Özgün 1200 koşusunun 3600 bağımlılık çıktısı önce özgün kodla birebir yeniden
üretildi. Ardından önceki ve yeni resolver aynı kayıtlı parse, morfoloji ve sözlük
sonuçlarıyla karşılaştırıldı. Bu ölçüm yeni uçtan uca çıkarım değildir.

| Yöntem | Doğru önce → sonra | Accuracy önce → sonra | Macro-F1 önce → sonra | Düzeltilen / bozulan |
|---|---:|---:|---:|---:|
| HisNet | 572 → 572 | %47,67 → %47,67 | %45,17 → %45,17 | 0 / 0 |
| DP→HisNet | 602 → 604 | %50,17 → %50,33 | %48,32 → %48,47 | 2 / 0 |
| HisNet→DP | 595 → 596 | %49,58 → %49,67 | %47,13 → %47,20 | 1 / 0 |
| LW HisNet→DP | 593 → 594 | %49,42 → %49,50 | %46,75 → %46,83 | 1 / 0 |

- ID 333: “kullandıktan beş dakika sonra akıyor aldığıma pişmanım”. Üç bağımlılık
  kolunda da pozitif → negatif; veri etiketi negatif. “Pişmanım”ın 1,0 negatif puanı korunuyor.
- ID 637: “Çok pişmanım” içeren uzun yorum DP→HisNet'te pozitif → negatif oldu.
- ID 247: “yokmuş gibi” içeren pozitif etiketli yorum DP→HisNet'te nötr → negatif
  oldu. İki sürüm de yanlış; koşullu/benzetmeli yokluk ifadesinin anlamı çözülmüş değil.

Ara çıktısı değişen yorum sayısı DP→HisNet'te 17, diğer iki bağımlılık kolunda
3'er. Sayısal kazanç küçüktür; bu sete bakılarak geliştirilen kural için bağımsız
genelleme kanıtı sayılmaz. “Bozulan yok” yalnız bu 1200 kayıt için geçerlidir.

## Kural neden daraltıldı?

İlk deneme bütün çekimli fiilleri koruyordu. DP→HisNet'te 5 hata düzelirken
4 doğru karar bozuldu; sıradan fiil adaylarının seçim davranışı genişledi.
Morfoloji ayrıca “sivilcelerim”i NOUN→VERB olarak yorumlamıştı; bu sözcüğü
yüklem saymak dilbilgisel olarak uygun değildi. Bu nedenle yalnız çekimli sıfat
yüklemlerini kapsayan kural tutuldu. İlk denemenin kodu ve bütün sonuçları
[broad_trial](broad_trial/) altında korundu; sonuçlara göre etiket veya sözlük
değişikliği yapılmadı.

## Doğrulama ve izlenebilirlik

- 76 çevrimdışı test geçti: yeni koruma, olumsuzluk kapsamı, önceki düzeltmeler,
  örneklem, paylaşılan ön işleme, devam kayıtları ve analiz testleri.
- Hedefli yeni sürüm testleri: 38 test, 0 hata. Önceki sürüm yeni koruma API'sini
  içermediğinden eski sürüm test kaydında beklenen AttributeError'lar da vardır;
  bu günlükteki hata sayısı veri üzerindeki yanlış karar sayısı değildir.
- Kaynak SQLite salt okunur açıldı ve SHA-256 kontrolüyle değişmediği doğrulandı.
- Son resolver SHA-256: `dc37555f087cb5124f72c2a3897588a69e8cf7b1310fb82a5b725a0e3535a667`.
- Önceki resolver SHA-256: `bc78f43b41f9b3ceecdb2ddfb96258ec2f750057644285ff2155d84e6a931111`.

Dosyalar: [sayısal karşılaştırma](comparison.json), [satır bazında önce/sonra](before_after.csv),
[tam değişen kapsamlar](changed_details.json), [sınıf metrikleri](per_class_before_after.csv),
[test kayıtları](candidate_tests.log), [sabitlenen resolver](corrected_sentiment_scope_resolver.py).

Sonraki kontrol tamamlandı: önceki 1200 metni dışlayan 1000 pozitif, 1000 negatif
ve 1000 nötr örnekte dört yöntem başarıyla çalıştı. Yeni örneklerde düzeltme
DP→HisNet'e 2, diğer iki bağımlılık koluna 1'er doğru sonuç ekledi; doğru → yanlış
geçiş yok. [3000 örneğin ayrıntılı sonuçları](../trsav1_classical_3000_20261101_parallel/findings.md).
Bu da geliştirme deneyidir; daha önce kullanılmış TRSAv1 kaynağından geldiği için
bağımsız son paper testi olarak sunulmamalıdır.
