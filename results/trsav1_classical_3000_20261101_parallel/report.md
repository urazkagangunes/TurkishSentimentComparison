# TRSAv1 — dört klasik sistem, sabit 3000 yorum

Örnek bazında hata incelemesi, istatistiksel yorum, yeni düzeltmenin ayrı etkisi ve sonraki adımlar: **[findings.md](findings.md)**.

Koşu tamamlandı: **True**. Eksik koşudaki metrikler yalnız başarılı altküme içindir.

Bu koşu geliştirme/teşhis amaçlıdır; bağımsız final test seti değildir. HisNet maddeleri, fiil kökü eşleştirmesi, morfoloji kuralları, eşik ve LW katsayıları bu koşu için değiştirilmedi.

## Örneklem ve çalışma protokolü

Kaynak: 125561 temizlenmiş TRSAv1 satırı. Örneklem: {'negative': 1000, 'neutral': 1000, 'positive': 1000}. Seed 42 stratified örneklemiyle eşleşme: True.

Protokol: dört yöntem aynı 3000 yorumda ortak, kopyalanmış morfoloji ve parse girdilerini kullanır. HisNet→DP ve LW aynı HisNet başlangıç sonucunu alır; DP→HisNet kendi dependency adaylarını seçip yalnız bu adayların köklerini sorgular. Eşikler ve karar kuralları değişmedi. Bağımsız doğrulama örnekleri independent_verification.json içindedir. Her satır SQLite'a kaydedildi; tam ara çıktılar attempts tablosunda duruyor. LLM kullanılmadı.

Önceki örneklerden dışlanan benzersiz metin: 1200; örtüşme: 0. Örnekleme referansı: this disjoint sampling protocol, not the original 1200-row draw.
Yorumlar 3 yerel işçiye dağıtıldı; her yorumdaki dört karar sıralı hesaplandı. Tablodaki süreler işçilerin ölçtüğü geçen sürelerin toplamıdır; CPU süresi veya koşunun toplam duvar saati değildir. 

Kaynak veride farklı etiketle de bulunan örnekler: [211, 224, 305, 458, 470, 505, 565, 790, 847, 906, 963, 965, 1106, 1121, 1186, 1277, 1289, 1340, 1422, 1456, 1534, 1724, 1732, 1849, 1854, 1885, 1919, 1923, 2011, 2017, 2043, 2067, 2116, 2119, 2301, 2307, 2556, 2586, 2802, 2805, 2878, 2944, 2960]. Bu satırlar ana ölçümde korunmuştur; aşağıda çıkarıldıklarındaki sonuç ayrıca verilir.

## Genel performans

| Sistem | Doğru/n | Accuracy % | Macro P % | Macro R % | Macro F1 % | Weighted F1 % | Accuracy %95 aralık |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HisNet | 1477/3000 | 49.23 | 47.14 | 49.23 | 46.57 | 46.57 | 47.45–51.02 |
| DP → HisNet | 1507/3000 | 50.23 | 48.18 | 50.23 | 48.11 | 48.11 | 48.45–52.02 |
| HisNet → DP | 1530/3000 | 51.00 | 48.74 | 51.00 | 48.20 | 48.20 | 49.21–52.79 |
| LW HisNet → DP | 1514/3000 | 50.47 | 47.94 | 50.47 | 47.52 | 47.52 | 48.68–52.25 |

Aralıklar Wilson yaklaşımıyla hesaplanan yaklaşık örneklem belirsizliğidir; etiket hatalarını veya yeniden kullanılan geliştirme setindeki iyimserliği kapsamaz. Çoğunluk sınıfı tabanı 1000/3000 = %33.33.

## Sınıf bazında precision / recall / F1

| Sistem | Sınıf | n | Precision % | Recall % | F1 % |
| --- | --- | --- | --- | --- | --- |
| HisNet | negative | 1000 | 53.20 | 60.60 | 56.66 |
| HisNet | neutral | 1000 | 38.28 | 19.10 | 25.48 |
| HisNet | positive | 1000 | 49.93 | 68.00 | 57.58 |
| DP → HisNet | negative | 1000 | 56.83 | 59.10 | 57.94 |
| DP → HisNet | neutral | 1000 | 36.45 | 21.80 | 27.28 |
| DP → HisNet | positive | 1000 | 51.25 | 69.80 | 59.10 |
| HisNet → DP | negative | 1000 | 55.57 | 64.30 | 59.62 |
| HisNet → DP | neutral | 1000 | 39.27 | 19.40 | 25.97 |
| HisNet → DP | positive | 1000 | 51.37 | 69.30 | 59.00 |
| LW HisNet → DP | negative | 1000 | 54.76 | 63.30 | 58.72 |
| LW HisNet → DP | neutral | 1000 | 37.65 | 18.30 | 24.63 |
| LW HisNet → DP | positive | 1000 | 51.40 | 69.80 | 59.20 |

## Karışıklık matrisleri

Satırlar gerçek etiket, sütunlar tahmin. Her sistemin tüm örnekleri başarıyla tamamlanmadıysa matris yalnız başarılı altkümeyi içerir.

### HisNet

| Gerçek / tahmin | negative | neutral | positive |
| --- | --- | --- | --- |
| negative | 606 | 176 | 218 |
| neutral | 345 | 191 | 464 |
| positive | 188 | 132 | 680 |

### DP → HisNet

| Gerçek / tahmin | negative | neutral | positive |
| --- | --- | --- | --- |
| negative | 591 | 231 | 178 |
| neutral | 296 | 218 | 486 |
| positive | 153 | 149 | 698 |

### HisNet → DP

| Gerçek / tahmin | negative | neutral | positive |
| --- | --- | --- | --- |
| negative | 643 | 170 | 187 |
| neutral | 337 | 194 | 469 |
| positive | 177 | 130 | 693 |

### LW HisNet → DP

| Gerçek / tahmin | negative | neutral | positive |
| --- | --- | --- | --- |
| negative | 633 | 177 | 190 |
| neutral | 347 | 183 | 470 |
| positive | 176 | 126 | 698 |

## İkili farklar

İyileşen: ilk sistem yanlışken ikinci doğru. Bozulan: ilk sistem doğruyken ikinci yanlış. Test, yalnız doğru/yanlış değişen eşleşmelerde iki yönlü exact binom testidir (exact McNemar); Holm düzeltmesi altı karşılaştırmaya uygulanır. Bunlar keşifsel ölçümlerdir. Hesaplama: [SciPy binomtest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).

| İlk sistem | İkinci sistem | Tahmini değişen | İyileşen | Bozulan | Net doğru | p | Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HisNet | DP → HisNet | 403 | 156 | 126 | 30 | 0.0840 | 0.3220 |
| HisNet | HisNet → DP | 176 | 84 | 31 | 53 | 0.0000 | 0.0000 |
| HisNet | LW HisNet → DP | 250 | 105 | 68 | 37 | 0.0060 | 0.0302 |
| DP → HisNet | HisNet → DP | 283 | 117 | 94 | 23 | 0.1297 | 0.3220 |
| DP → HisNet | LW HisNet → DP | 322 | 120 | 113 | 7 | 0.6944 | 0.6944 |
| HisNet → DP | LW HisNet → DP | 96 | 29 | 45 | -16 | 0.0805 | 0.3220 |

Dördünün birlikte doğru olduğu yorum: **1322**; birlikte yanlış: **1330**; sistemlerin farklı etiket verdiği yorum: **481**.

## Süreler

**Bu koşunun süreleri hız değerlendirmesinde kullanılamaz.** Kayıtlardaki uzun kesintiler perf_counter ölçümlerine yansımıştır. Ham değerler korunmuştur; bunlar kesintisiz çıkarım veya CPU süresi değildir. Ayrıntı: [timing_caveats.json](timing_caveats.json).

| Sistem | Başlatma s | Sınıflandırma s | Ortalama ms/yorum | Medyan ms | P95 ms |
| --- | --- | --- | --- | --- | --- |
| HisNet | 35.11 | 426418.80 | 142139.60 | 47.21 | 30176.83 |
| DP → HisNet | 0.00 | 10.51 | 3.50 | 1.00 | 10.18 |
| HisNet → DP | 0.00 | 5.48 | 1.83 | 0.46 | 6.16 |
| LW HisNet → DP | 0.00 | 9.04 | 3.01 | 0.89 | 7.76 |

**Paylaşılan ön işleme:** HisNet süresi morfolojiyi içerir; diğer üç yöntem için burada yalnız bileşim süresi vardır. Ortak parse süresi shared_preprocessing.json dosyasındadır. Bu sütunlar bağımsız uçtan uca yöntem hızlarını karşılaştırmaz. Süreler yerel CPU, önbellek ve o anki sistem yüküne bağlıdır. Başlatma ayrıca gösterilir; kayıt/dışa aktarma beklemeleri sınıflandırma süresine dahil değildir. Toplam süre tüm kaydedilmiş girişimleri kapsar; yorum başına dağılımlar son girişimden alınır. Kesilen/cevabı kaydedilmeyen girişimlerin süresi bilinmez. Yorumlar 3 yerel işçiye dağıtıldı; her yorumdaki dört karar sıralı hesaplandı. Tablodaki süreler işçilerin ölçtüğü geçen sürelerin toplamıdır; CPU süresi veya koşunun toplam duvar saati değildir. Kontrollü donanım benchmark'ı değildir.

### Morfolojik çözümleme türüne göre süre

| Sistem | Grup | Yorum | Toplam s | Ortalama ms | Medyan ms |
| --- | --- | --- | --- | --- | --- |
| HisNet | robust_or_fallback_present | 1160 | 426337.50 | 367532.33 | 12176.97 |
| HisNet | normal_only | 1840 | 81.31 | 44.19 | 17.06 |
| DP → HisNet | robust_or_fallback_present | 1160 | 5.75 | 4.96 | 1.44 |
| DP → HisNet | normal_only | 1840 | 4.76 | 2.59 | 0.81 |
| HisNet → DP | robust_or_fallback_present | 1160 | 3.08 | 2.65 | 0.63 |
| HisNet → DP | normal_only | 1840 | 2.40 | 1.31 | 0.41 |
| LW HisNet → DP | robust_or_fallback_present | 1160 | 4.34 | 3.74 | 1.16 |
| LW HisNet → DP | normal_only | 1840 | 4.70 | 2.55 | 0.78 |

Gruplar kaydedilmiş analysis_source alanına göre belirlenir. Robust/fallback içeren yorumlar daha uzun veya yazım hatalı da olabilir; bu gruplama tek başına nedensel hız ölçümü değildir. Paylaşılan akışta dependency satırları morfoloji çalıştırma süresini içermez. Mevcut morfoloji kodu her yeni OOV için izole çözümleyici kurar ve adayları önbelleğe alır; bu davranış deney sırasında değiştirilmedi.

## Kapsam ve işlem tanıları

| Sistem | Skoru 0/0 yorum | Bunlardan polar etiketli | Duygu sözcüğü olmayan |
| --- | --- | --- | --- |
| HisNet | 350 | 218 | 350 |
| DP → HisNet | 484 | 308 | 484 |
| HisNet → DP | 350 | 218 | 350 |
| LW HisNet → DP | 357 | 223 | 350 |

0/0; sözlük kapsamı, morfoloji veya dependency aday seçimi gibi farklı nedenlerden doğabilir. Tek başına kod hatası kanıtı değildir.

| Sistem | Olumsuzluk çevirme | Hedef bastırma | Eşleşmeyen kapsam | LW modifier | LW contrast |
| --- | --- | --- | --- | --- | --- |
| HisNet | 0 | 0 | 0 | 0 | 0 |
| DP → HisNet | 242 | 25 | 0 | 0 | 0 |
| HisNet → DP | 294 | 237 | 0 | 0 | 0 |
| LW HisNet → DP | 294 | 237 | 0 | 769 | 1866 |

Sayılar yorum sayısı değil kapsam/işlem sayısıdır. LW ayrıca modifier bileşeni ve contrast marker bastırması yapabilir; fark yalnız katsayıların etkisi olarak yorumlanmamalıdır.

## Otomatik kod tutarlılığı kontrolleri

İhlal sayısı: **0**. Kontroller: sonlu/negatif olmayan skorlar, kapsam toplamları, eşik–etiket uyumu, H→DP/LW başlangıç HisNet skorlarının korunması, LW temel DP skorları, HisNet morfoloji çıktılarının tutarlılığı, tekrarlanan token/olumsuzluk ID'leri ve cümleler arası olumsuzluk taşması. Sıfır ihlal bütün dilsel kapsam kararlarının doğru olduğunu kanıtlamaz.

## Etiket çelişkisine duyarlılık ve uzunluk dilimleri

| Sistem | Alt küme | n | Accuracy % | Macro F1 % |
| --- | --- | --- | --- | --- |
| HisNet | source_label_unambiguous | 2957 | 49.48 | 46.68 |
| HisNet | source_label_conflicting | 43 | 32.56 | 30.73 |
| HisNet | 1_to_5_words | 773 | 48.51 | 48.58 |
| HisNet | 6_to_15_words | 1549 | 49.58 | 45.81 |
| HisNet | 16_plus_words | 678 | 49.26 | 42.71 |
| HisNet | zero_sentiment_score | 350 | 37.71 | 18.26 |
| DP → HisNet | source_label_unambiguous | 2957 | 50.49 | 48.25 |
| DP → HisNet | source_label_conflicting | 43 | 32.56 | 30.73 |
| DP → HisNet | 1_to_5_words | 773 | 49.16 | 49.48 |
| DP → HisNet | 6_to_15_words | 1549 | 50.29 | 47.07 |
| DP → HisNet | 16_plus_words | 678 | 51.33 | 45.41 |
| DP → HisNet | zero_sentiment_score | 484 | 36.36 | 17.78 |
| HisNet → DP | source_label_unambiguous | 2957 | 51.27 | 48.33 |
| HisNet → DP | source_label_conflicting | 43 | 32.56 | 30.73 |
| HisNet → DP | 1_to_5_words | 773 | 50.19 | 50.20 |
| HisNet → DP | 6_to_15_words | 1549 | 51.58 | 47.60 |
| HisNet → DP | 16_plus_words | 678 | 50.59 | 44.12 |
| HisNet → DP | zero_sentiment_score | 350 | 37.71 | 18.26 |
| LW HisNet → DP | source_label_unambiguous | 2957 | 50.73 | 47.63 |
| LW HisNet → DP | source_label_conflicting | 43 | 32.56 | 30.73 |
| LW HisNet → DP | 1_to_5_words | 773 | 50.32 | 50.34 |
| LW HisNet → DP | 6_to_15_words | 1549 | 50.87 | 46.56 |
| LW HisNet → DP | 16_plus_words | 678 | 49.71 | 43.00 |
| LW HisNet → DP | zero_sentiment_score | 357 | 37.54 | 18.19 |

Alt kümeler sonuç sonrası teşhis içindir; ana 3000 örneklik ölçümü değiştirmez. Küçük alt kümelerde macro-F1 tüm üç etiketi içerir.

## Ayrışan örnekler

Aşağıdaki örnekler sabit satır sırasındaki ilk 20 ayrışmadır; tamamı disagreements_3000.csv içindedir.

| ID | Metin | Gold | HisNet | DP → HisNet | HisNet → DP | LW HisNet → DP |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | Yüzde ferahlatıcı bir etkisi var. Karma ciltliyim yüzümde sivilce yapmadı | neutral | neutral | positive | neutral | neutral |
| 22 | Beklediğim kalitede değil. Taş değilde plastik gibi 👎🏻 | negative | positive | negative | negative | negative |
| 25 | Saçlarım ipek gibi yumusacik Kuru ve yipranmis Saclarim ipek gibi oldu .simdiye kadar aldigim ürünlerin hespinden kat kat daha iyi işe yaradi. Saçlarım yumusacik oldu. Almak isteyenlere hiç düşünmeden almadini tavsiye ediyorum.. | positive | negative | neutral | negative | negative |
| 26 | bayıldım ❣️ biraz kokusu erkek parfümü benziyor ama yüze sürdükten sonra farkedilmiyor. | positive | positive | neutral | positive | positive |
| 28 | 30 sayfa cikti aldim kartuş yarinin altina indi | negative | positive | neutral | positive | positive |
| 29 | Fiyat performans açısından fena değil. Ama lekeli ciltler için ne kadar sık kullansanızda yeterli olmuyor. | neutral | negative | positive | positive | positive |
| 32 | Asla almayın çok soluk sanki bitmş gibi | negative | negative | neutral | negative | negative |
| 35 | Tavsiye etmiyorum 🙄 | negative | positive | negative | negative | negative |
| 36 | İlk doğduğu günden beri kızıma kullanıyorum aşırı memnunum burdada uygun fiyata görünce hemen aldım kuru ciltlilere çok iyi geliyor | positive | negative | positive | negative | negative |
| 37 | pek işe yaramadı bende ama yinede kendimi iyi hissettim.. | negative | neutral | neutral | neutral | positive |
| 41 | fiyatı uygundu ama geldigi gün iade ettim kalitesi kötü bir ürün | negative | neutral | negative | neutral | negative |
| 61 | yorumlara guvenip aldim fakat o kadar basit ve agir geldi ki iade ettim | negative | neutral | positive | positive | positive |
| 63 | Kot kaliteli ama bacakları aşırı büyük 1.61 boyum 67 kilon 40 beden tam oldu ama alt bacak kısmı çok abartı geniş ama mavi rengi de öyle yine de mavi biraz daha maalesef iade edeceğim | negative | neutral | neutral | neutral | negative |
| 70 | fena değil sürülmesi biraz zor | neutral | negative | negative | negative | neutral |
| 79 | Atopiğe eğilimli kızlarım için banyodan sonra kullanıyorum. Çok faydası oldu nemlendirmesi güzel ve kızımın artık eklem yerlerinde sırtın da filan alerjisi olmuyor | positive | negative | positive | positive | positive |
| 84 | Saçımda neredeyse hiç köpürmüyor saçımın farklı bölgeleri şampuanlandı mı anlamıyorum ama saçları ma maskeyle birlikte bir canlılık geldi kesinlikle. şampuanın kokusu muhteşem bu arada. | positive | negative | neutral | negative | negative |
| 85 | Kesinlikle hiçbir etkisi yok bayağı basit ! tavsiye etmem | negative | positive | negative | negative | negative |
| 95 | hiç güzel değildi | negative | neutral | negative | negative | negative |
| 98 | kaliteli değil malesef...derisi dökülüyor | negative | positive | negative | negative | negative |
| 104 | ürün çok büyük değil kalıcılığı yok gibi | neutral | negative | negative | positive | positive |

## Dört sistemin de yanlış olduğu örnekler

| ID | Metin | Gold | HisNet | DP → HisNet | HisNet → DP | LW HisNet → DP |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | gelir gelmez denedim. istediğim kadar güzel taramadı kaşlarımı ama 3-4 saat kadar kaldı. kalıcılık konusunda iyi fakat taramasını sevmedim. | neutral | positive | positive | positive | positive |
| 1 | Pek kullanamadım renkler birbirinin neredeyse aynısı | neutral | positive | positive | positive | positive |
| 2 | temizleyici özelliği güzel ama kokusu kötü | neutral | negative | negative | negative | negative |
| 5 | En iyi kaş boyalarınsan biri ayda 2 kez kullanıyorum fakat kaş seyreltiyor kesinlikle | neutral | positive | positive | positive | positive |
| 9 | 2. Kutum yapısı ve formunu begeniyorum ama lekelerimde ya da cildimde fark gözlemlemedim kolejeni tetiklemesi açısından kullanıyorum hala bir fark olmazsa başka markaya geçicem | positive | neutral | neutral | neutral | neutral |
| 11 | Beğenmedik çünkü kafada iyi durmuyor delikli gözüküyor | negative | positive | positive | positive | positive |
| 12 | Aşırı bol uygun fiyat | negative | positive | positive | positive | positive |
| 17 | Ben iade ettim bende kalibi olmadi omuzlari genis ve uzun boylu olanlara süper gider kumasina denecek söz yok | negative | positive | positive | positive | positive |
| 18 | güneş kremi formunda surumu yoğun olduğu için pratik değil inşallah güneş lekelerini açmada etkili olur | positive | negative | negative | negative | negative |
| 19 | beğenerek kullandigim bir koku | positive | neutral | neutral | neutral | neutral |
| 24 | sim gibi duruyor aydınlatma ozelligi yok. beğenmedim | neutral | negative | negative | negative | negative |
| 25 | Saçlarım ipek gibi yumusacik Kuru ve yipranmis Saclarim ipek gibi oldu .simdiye kadar aldigim ürünlerin hespinden kat kat daha iyi işe yaradi. Saçlarım yumusacik oldu. Almak isteyenlere hiç düşünmeden almadini tavsiye ediyorum.. | positive | negative | neutral | negative | negative |
| 27 | fırçası tahriş ediyor | neutral | negative | negative | negative | negative |
| 28 | 30 sayfa cikti aldim kartuş yarinin altina indi | negative | positive | neutral | positive | positive |
| 29 | Fiyat performans açısından fena değil. Ama lekeli ciltler için ne kadar sık kullansanızda yeterli olmuyor. | neutral | negative | positive | positive | positive |
| 30 | Yeni aldım bakalım neler olacak. | neutral | negative | negative | negative | negative |
| 31 | Ürün yeni elime ulaştı ve kullanmaya başladım gayet başarılı bir ürüne benziyor. Kargolanması 1 haftayı bulduğu için de 3 Yıldız. | neutral | positive | positive | positive | positive |
| 37 | pek işe yaramadı bende ama yinede kendimi iyi hissettim.. | negative | neutral | neutral | neutral | positive |
| 38 | memnun kalmadım. su gibi dokusu ve akışkanlığı. cildi pek koruduğunu düşünmüyorum | negative | positive | positive | positive | positive |
| 39 | süper ürün beğendim | neutral | positive | positive | positive | positive |

## Sözlükte eşleşmeyen sık kökler

| Kök | Görülme |
| --- | --- |
| kullan | 566 |
| gel | 332 |
| yap | 247 |
| beğen | 224 |
| ver | 143 |
| sev | 99 |
| sür | 90 |
| çık | 84 |
| bekle | 78 |
| temizle | 68 |
| görün | 64 |
| dene | 60 |
| düşün | 58 |
| başla | 57 |
| iste | 52 |
| bırak | 52 |
| bil | 47 |
| git | 45 |
| nemlen | 43 |
| dökül | 41 |
| kapat | 40 |
| paketle | 40 |
| göster | 38 |
| uygula | 36 |
| um | 35 |
| bayıl | 33 |
| kullanıl | 32 |
| aldim | 31 |
| giy | 31 |
| yıka | 31 |

Bu liste fiil kökü/lemma uyuşmazlığı, yazım hatası, özel ad ve kapsam dışı sözcükleri birlikte içerir; otomatik olarak sözlüğe eklenecek madde listesi değildir.

## Dosyalar

- `analysis.json`: tüm metrikler, işlem sayıları, ID listeleri ve tanılar.
- `comparison_3000.csv`: her yorumda dört tahmin ve pozitif/negatif skorlar.
- `errors_3000.csv`, `disagreements_3000.csv`: hata ve ayrışma örnekleri.
- `scope_details_3000.csv`: tüm dependency kapsamlarının skor, işlem, hedef ve olumsuzluk ayrıntıları.
- `diagnostic_cases.json`: ayrışan ve seçilmiş ortak hatalı örneklerin tam ara çıktıları.
- `per_class_3000.csv`, `pairwise_3000.csv`, `slice_metrics_3000.csv`: ayrıntılı metrikler.
- `progress.sqlite3`: kaldığı yerden devam kaydı ve tam model ara çıktıları.
