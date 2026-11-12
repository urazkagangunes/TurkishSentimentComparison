# TRSAv1 — dört klasik sistem, sabit 1200 yorum

Koşu tamamlandı: **True**. Eksik koşudaki metrikler yalnız başarılı altküme içindir.

Bu koşu geliştirme/teşhis amaçlıdır; bağımsız final test seti değildir. HisNet maddeleri, fiil kökü eşleştirmesi, morfoloji kuralları, eşik ve LW katsayıları bu koşu için değiştirilmedi.

## Örneklem ve çalışma protokolü

Kaynak: 125561 temizlenmiş TRSAv1 satırı. Örneklem: {'negative': 400, 'neutral': 400, 'positive': 400}. Seed 42 stratified örneklemiyle eşleşme: True.

Protokol: dört yöntem aynı 1200 yorumda ortak, kopyalanmış morfoloji ve parse girdilerini kullanır. HisNet→DP ve LW aynı HisNet başlangıç sonucunu alır; DP→HisNet kendi dependency adaylarını seçip yalnız bu adayların köklerini sorgular. Eşikler ve karar kuralları değişmedi. Bağımsız doğrulama örnekleri independent_verification.json içindedir. Her satır SQLite'a kaydedildi; tam ara çıktılar attempts tablosunda duruyor. LLM kullanılmadı.

Kaynak veride farklı etiketle de bulunan örnekler: [28, 51, 64, 209, 275, 431, 524, 539, 655, 672, 689, 699, 709, 765, 815, 849, 896, 905, 976, 1121]. Bu satırlar ana ölçümde korunmuştur; aşağıda çıkarıldıklarındaki sonuç ayrıca verilir.

## Genel performans

| Sistem | Doğru/n | Accuracy % | Macro P % | Macro R % | Macro F1 % | Weighted F1 % | Accuracy %95 aralık |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HisNet | 572/1200 | 47.67 | 45.53 | 47.67 | 45.17 | 45.17 | 44.85–50.50 |
| DP → HisNet | 602/1200 | 50.17 | 48.81 | 50.17 | 48.32 | 48.32 | 47.34–52.99 |
| HisNet → DP | 594/1200 | 49.50 | 47.35 | 49.50 | 47.06 | 47.06 | 46.68–52.33 |
| LW HisNet → DP | 591/1200 | 49.25 | 46.84 | 49.25 | 46.61 | 46.61 | 46.43–52.08 |

Aralıklar Wilson yaklaşımıyla hesaplanan yaklaşık örneklem belirsizliğidir; etiket hatalarını veya yeniden kullanılan geliştirme setindeki iyimserliği kapsamaz. Çoğunluk sınıfı tabanı 400/1200 = %33.33.

## Sınıf bazında precision / recall / F1

| Sistem | Sınıf | n | Precision % | Recall % | F1 % |
| --- | --- | --- | --- | --- | --- |
| HisNet | negative | 400 | 54.11 | 56.00 | 55.04 |
| HisNet | neutral | 400 | 34.58 | 18.50 | 24.10 |
| HisNet | positive | 400 | 47.90 | 68.50 | 56.38 |
| DP → HisNet | negative | 400 | 60.59 | 56.50 | 58.47 |
| DP → HisNet | neutral | 400 | 36.36 | 23.00 | 28.18 |
| DP → HisNet | positive | 400 | 49.48 | 71.00 | 58.32 |
| HisNet → DP | negative | 400 | 57.31 | 59.75 | 58.51 |
| HisNet → DP | neutral | 400 | 35.62 | 19.50 | 25.20 |
| HisNet → DP | positive | 400 | 49.11 | 69.25 | 57.47 |
| LW HisNet → DP | negative | 400 | 57.14 | 60.00 | 58.54 |
| LW HisNet → DP | neutral | 400 | 34.43 | 18.25 | 23.86 |
| LW HisNet → DP | positive | 400 | 48.94 | 69.50 | 57.44 |

## Karışıklık matrisleri

Satırlar gerçek etiket, sütunlar tahmin. Her sistemin tüm örnekleri başarıyla tamamlanmadıysa matris yalnız başarılı altkümeyi içerir.

### HisNet

| Gerçek / tahmin | negative | neutral | positive |
| --- | --- | --- | --- |
| negative | 224 | 77 | 99 |
| neutral | 127 | 74 | 199 |
| positive | 63 | 63 | 274 |

### DP → HisNet

| Gerçek / tahmin | negative | neutral | positive |
| --- | --- | --- | --- |
| negative | 226 | 89 | 85 |
| neutral | 103 | 92 | 205 |
| positive | 44 | 72 | 284 |

### HisNet → DP

| Gerçek / tahmin | negative | neutral | positive |
| --- | --- | --- | --- |
| negative | 239 | 75 | 86 |
| neutral | 121 | 78 | 201 |
| positive | 57 | 66 | 277 |

### LW HisNet → DP

| Gerçek / tahmin | negative | neutral | positive |
| --- | --- | --- | --- |
| negative | 240 | 75 | 85 |
| neutral | 122 | 73 | 205 |
| positive | 58 | 64 | 278 |

## İkili farklar

İyileşen: ilk sistem yanlışken ikinci doğru. Bozulan: ilk sistem doğruyken ikinci yanlış. Test, yalnız doğru/yanlış değişen eşleşmelerde iki yönlü exact binom testidir (exact McNemar); Holm düzeltmesi altı karşılaştırmaya uygulanır. Bunlar keşifsel ölçümlerdir. Hesaplama: [SciPy binomtest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).

| İlk sistem | İkinci sistem | Tahmini değişen | İyileşen | Bozulan | Net doğru | p | Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HisNet | DP → HisNet | 157 | 73 | 43 | 30 | 0.0068 | 0.0342 |
| HisNet | HisNet → DP | 71 | 39 | 17 | 22 | 0.0046 | 0.0274 |
| HisNet | LW HisNet → DP | 91 | 46 | 27 | 19 | 0.0344 | 0.1377 |
| DP → HisNet | HisNet → DP | 118 | 41 | 49 | -8 | 0.4608 | 0.9446 |
| DP → HisNet | LW HisNet → DP | 131 | 44 | 55 | -11 | 0.3149 | 0.9446 |
| HisNet → DP | LW HisNet → DP | 28 | 11 | 14 | -3 | 0.6900 | 0.9446 |

Dördünün birlikte doğru olduğu yorum: **515**; birlikte yanlış: **539**; sistemlerin farklı etiket verdiği yorum: **186**.

## Süreler

| Sistem | Başlatma s | Sınıflandırma s | Ortalama ms/yorum | Medyan ms | P95 ms |
| --- | --- | --- | --- | --- | --- |
| HisNet | 7.92 | 3157.64 | 2631.36 | 19.20 | 10872.40 |
| DP → HisNet | 0.00 | 1.07 | 0.90 | 0.49 | 2.33 |
| HisNet → DP | 0.00 | 0.47 | 0.39 | 0.31 | 0.89 |
| LW HisNet → DP | 0.00 | 0.86 | 0.72 | 0.58 | 1.63 |

**Paylaşılan ön işleme:** HisNet süresi morfolojiyi içerir; diğer üç yöntem için burada yalnız bileşim süresi vardır. Ortak parse süresi shared_preprocessing.json dosyasındadır. Bu sütunlar bağımsız uçtan uca yöntem hızlarını karşılaştırmaz. Süreler yerel CPU, önbellek ve o anki sistem yüküne bağlıdır. Başlatma ayrıca gösterilir; kayıt/dışa aktarma beklemeleri sınıflandırma süresine dahil değildir. Toplam süre tüm kaydedilmiş girişimleri kapsar; yorum başına dağılımlar son girişimden alınır. Kesilen/cevabı kaydedilmeyen girişimlerin süresi bilinmez. Sistemler sıralı çalışır; kontrollü donanım benchmark'ı değildir.

### Morfolojik çözümleme türüne göre süre

| Sistem | Grup | Yorum | Toplam s | Ortalama ms | Medyan ms |
| --- | --- | --- | --- | --- | --- |
| HisNet | robust_or_fallback_present | 465 | 3148.19 | 6770.30 | 3755.97 |
| HisNet | normal_only | 735 | 9.45 | 12.86 | 7.62 |
| DP → HisNet | robust_or_fallback_present | 465 | 0.62 | 1.32 | 0.65 |
| DP → HisNet | normal_only | 735 | 0.46 | 0.62 | 0.42 |
| HisNet → DP | robust_or_fallback_present | 465 | 0.22 | 0.47 | 0.36 |
| HisNet → DP | normal_only | 735 | 0.25 | 0.34 | 0.28 |
| LW HisNet → DP | robust_or_fallback_present | 465 | 0.39 | 0.85 | 0.68 |
| LW HisNet → DP | normal_only | 735 | 0.47 | 0.64 | 0.53 |

Gruplar kaydedilmiş analysis_source alanına göre belirlenir. Robust/fallback içeren yorumlar daha uzun veya yazım hatalı da olabilir; bu gruplama tek başına nedensel hız ölçümü değildir. Paylaşılan akışta dependency satırları morfoloji çalıştırma süresini içermez. Mevcut morfoloji kodu her yeni OOV için izole çözümleyici kurar ve adayları önbelleğe alır; bu davranış deney sırasında değiştirilmedi.

## Kapsam ve işlem tanıları

| Sistem | Skoru 0/0 yorum | Bunlardan polar etiketli | Duygu sözcüğü olmayan |
| --- | --- | --- | --- |
| HisNet | 151 | 95 | 151 |
| DP → HisNet | 208 | 130 | 208 |
| HisNet → DP | 151 | 95 | 151 |
| LW HisNet → DP | 152 | 95 | 151 |

0/0; sözlük kapsamı, morfoloji veya dependency aday seçimi gibi farklı nedenlerden doğabilir. Tek başına kod hatası kanıtı değildir.

| Sistem | Olumsuzluk çevirme | Hedef bastırma | Eşleşmeyen kapsam | LW modifier | LW contrast |
| --- | --- | --- | --- | --- | --- |
| HisNet | 0 | 0 | 0 | 0 | 0 |
| DP → HisNet | 88 | 11 | 0 | 0 | 0 |
| HisNet → DP | 121 | 85 | 0 | 0 | 0 |
| LW HisNet → DP | 121 | 85 | 0 | 279 | 649 |

Sayılar yorum sayısı değil kapsam/işlem sayısıdır. LW ayrıca modifier bileşeni ve contrast marker bastırması yapabilir; fark yalnız katsayıların etkisi olarak yorumlanmamalıdır.

## Otomatik kod tutarlılığı kontrolleri

İhlal sayısı: **0**. Kontroller: sonlu/negatif olmayan skorlar, kapsam toplamları, eşik–etiket uyumu, H→DP/LW başlangıç HisNet skorlarının korunması, LW temel DP skorları, HisNet morfoloji çıktılarının tutarlılığı, tekrarlanan token/olumsuzluk ID'leri ve cümleler arası olumsuzluk taşması. Sıfır ihlal bütün dilsel kapsam kararlarının doğru olduğunu kanıtlamaz.

## Etiket çelişkisine duyarlılık ve uzunluk dilimleri

| Sistem | Alt küme | n | Accuracy % | Macro F1 % |
| --- | --- | --- | --- | --- |
| HisNet | source_label_unambiguous | 1180 | 47.63 | 45.17 |
| HisNet | source_label_conflicting | 20 | 50.00 | 44.77 |
| HisNet | 1_to_5_words | 328 | 44.51 | 44.73 |
| HisNet | 6_to_15_words | 628 | 51.43 | 46.97 |
| HisNet | 16_plus_words | 244 | 42.21 | 37.46 |
| HisNet | zero_sentiment_score | 151 | 37.09 | 18.04 |
| DP → HisNet | source_label_unambiguous | 1180 | 50.17 | 48.37 |
| DP → HisNet | source_label_conflicting | 20 | 50.00 | 44.77 |
| DP → HisNet | 1_to_5_words | 328 | 46.34 | 47.02 |
| DP → HisNet | 6_to_15_words | 628 | 52.39 | 49.39 |
| DP → HisNet | 16_plus_words | 244 | 49.59 | 44.85 |
| DP → HisNet | zero_sentiment_score | 208 | 37.50 | 18.18 |
| HisNet → DP | source_label_unambiguous | 1180 | 49.49 | 47.09 |
| HisNet → DP | source_label_conflicting | 20 | 50.00 | 44.77 |
| HisNet → DP | 1_to_5_words | 328 | 46.34 | 46.77 |
| HisNet → DP | 6_to_15_words | 628 | 53.03 | 48.79 |
| HisNet → DP | 16_plus_words | 244 | 44.67 | 40.61 |
| HisNet → DP | zero_sentiment_score | 151 | 37.09 | 18.04 |
| LW HisNet → DP | source_label_unambiguous | 1180 | 49.24 | 46.63 |
| LW HisNet → DP | source_label_conflicting | 20 | 50.00 | 44.77 |
| LW HisNet → DP | 1_to_5_words | 328 | 46.04 | 46.45 |
| LW HisNet → DP | 6_to_15_words | 628 | 53.50 | 49.29 |
| LW HisNet → DP | 16_plus_words | 244 | 42.62 | 37.03 |
| LW HisNet → DP | zero_sentiment_score | 152 | 37.50 | 18.18 |

Alt kümeler sonuç sonrası teşhis içindir; ana 1200 örneklik ölçümü değiştirmez. Küçük alt kümelerde macro-F1 tüm üç etiketi içerir.

## Ayrışan örnekler

Aşağıdaki örnekler sabit satır sırasındaki ilk 20 ayrışmadır; tamamı disagreements_1200.csv içindedir.

| ID | Metin | Gold | HisNet | DP → HisNet | HisNet → DP | LW HisNet → DP |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | Saçı yıpratıyor ama aynı rengi diğer markalarda hiç denemedim. Renkten dolayı olabilir. İçinden 2 paket çıkması oldukça uygun fiyata denk geliyor | neutral | neutral | neutral | neutral | negative |
| 13 | Yorumlarda söylendiği gibi küçük evet ama oldukça kullanışlı kullanalı bir kaç gün oldu ama cildimde fark hissettim hemen kesinlikle öneririm | positive | neutral | positive | neutral | neutral |
| 17 | Sacima dolgunluk verir diye almistim. Fakat bir fark goremedim☹️ | neutral | negative | neutral | negative | negative |
| 26 | yüzümü biraz yakıyor sivilcelerimi daha kurutmadi ama kullanmaya iki hafta önce başladım.. | neutral | negative | neutral | negative | negative |
| 30 | Rengi gözüktüğü gibi yeşil yerine siyah geldi. Doğal taş olduğu söyleniyor ama kalitesinden çok emin olamadım. Sanırım daha pahalı bir versiyonunu almak daha iyi olabilirdi. | negative | positive | neutral | positive | positive |
| 53 | Kuru ciltler için uygun değilmiş beğenmedim oksitlenme yaptı | negative | positive | negative | negative | negative |
| 59 | Çok göz yakıyor almanızı tavsiye etmem | negative | neutral | negative | negative | negative |
| 62 | kumasi hemen tüylenme yapti dikiş kalitesiz...memnun kalmadim.fiyatina değer bir ürün değil. | negative | negative | negative | positive | positive |
| 73 | urün ön tarafı agarmış olarak geldi iade edecegim | negative | negative | neutral | negative | negative |
| 76 | ilk defa eczaneden almıştım daha yoğun kıvamı surdukce kafadan aşağıya doğru su gibi akmayan bir yapısı vardı. burdan aldigim ürün kesinlikle daha ince yapıda su gibi. beğenmedim eczaneden alın bence | negative | positive | negative | positive | positive |
| 80 | Yağlı his bırakmadan ağırlık yapmadan serum gibi yapıda mis gibi kokan bakım ürünü ❤️ | positive | positive | neutral | positive | positive |
| 94 | Bayildiiim. Çok fazla ürün araştırdım yorumlari okuyarak bu cihazi almaya karar verdim ve hiç pişman olmadim. 2 kere kullandim ve çok memnun kaldim burnumun ustunde puturlenmeler vardi şuan burnum gelecegimden daha parlak. Boyu kücük ama yüz icin yeterli eli yormuyor. Tavsiye ederim kesinlikle alinmali | positive | negative | positive | positive | positive |
| 96 | Annemin ayak ağrılarını rahatlatıyor ferah bir kokusu var | neutral | negative | positive | negative | negative |
| 109 | alırken bu kadar memnun kalacağımı bilmiyordum. Kesinlikle tavsiye. | positive | neutral | positive | neutral | neutral |
| 118 | Nedense bende bir etkisi olmadı bir ay kullanmama rağmen | neutral | negative | neutral | negative | neutral |
| 121 | renk düzensizliğini gerçekten eşitliyor beğendim | positive | negative | neutral | negative | negative |
| 125 | Pahali olabilir ama cook uzun sure gidiyor hepsini denedim la roche da takili kaldim temislendigini hissediyorsunuz kesinlikle tavsiye ederrim favorilere ekledim indikce aliyorum | positive | negative | neutral | negative | negative |
| 127 | Arkadaşıma aldım saçını dökmüş bilginiz olsun... | negative | positive | negative | positive | positive |
| 129 | Kokusu fena değil ama kesinlikle sıktığım an dışında kokusu gidiyor. Para vermeye değmez. | neutral | negative | positive | positive | positive |
| 131 | Uzun bir banyodan sonra gozenekler yeterince acildiysa yuzeydeki kirleri cekmede oldukca basarili. Sonrasinda hem gozenekleri toparlamak hem de morarma olmasini engellemek icin bir buz gezdirin yuzunuzde mutlaka ve toniklemeyi de ihmal etmeyin. | positive | negative | positive | negative | negative |

## Dört sistemin de yanlış olduğu örnekler

| ID | Metin | Gold | HisNet | DP → HisNet | HisNet → DP | LW HisNet → DP |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | Yakıo kızlar kullanmayın | negative | neutral | neutral | neutral | neutral |
| 5 | Ürün bayağı yakıyor 5 gündür düzenli kullanıyorum. Umarım faydasını görürüm | positive | negative | negative | negative | negative |
| 8 | Tonu bana göre çok koyu kaldı ama güzel bir ürün dokusu falan hoş | neutral | positive | positive | positive | positive |
| 11 | Çok gevşek bir kumaş kavramıyor toplanıyor yürürken defacto lcw koton külotları bile bundan bin kat iyidir | negative | positive | positive | positive | positive |
| 15 | İlk giymede diz çıkarttı | negative | neutral | neutral | neutral | neutral |
| 16 | güzel. beyandım. | neutral | positive | positive | positive | positive |
| 19 | Kokusu pek güzel dgl idare eder | neutral | positive | positive | positive | positive |
| 20 | İpi giydigim dakika koptu | negative | neutral | neutral | neutral | neutral |
| 22 | fena değil fiyatina gore iyi | neutral | positive | positive | positive | positive |
| 29 | Ürün çok güzel. Hep kullandığım ürün. Ulaşım çok uzun sürdü tabi. | neutral | positive | positive | positive | positive |
| 30 | Rengi gözüktüğü gibi yeşil yerine siyah geldi. Doğal taş olduğu söyleniyor ama kalitesinden çok emin olamadım. Sanırım daha pahalı bir versiyonunu almak daha iyi olabilirdi. | negative | positive | neutral | positive | positive |
| 32 | Krem işe yarıyor başarılı | neutral | positive | positive | positive | positive |
| 33 | Tatlı koku sevenler alabilirler | neutral | positive | positive | positive | positive |
| 34 | herkese yakışmıyor bence uzunlar giymeli | negative | neutral | neutral | neutral | neutral |
| 35 | daha önce de kullandigim bur tonik ferah ve güzel kokuku | neutral | positive | positive | positive | positive |
| 36 | Kokusu hafif ama güzel | neutral | positive | positive | positive | positive |
| 39 | ürünu kremiyle birlikte aldim iki kutuda da sadece inceltir azaltir yaziyor gec cikarma ibaresi yok daha yeni kullanmaya basladim sonra yine yorumlayacagim 👍🏻 | neutral | negative | negative | negative | negative |
| 40 | denemeli 2 hafta oldu. hayatım kolaylaştı. saçlar artık düzlestirilmiyor. :) fakat saçları ağırlaştırıyor bilginiz olsun... | positive | neutral | neutral | neutral | neutral |
| 44 | Çok tavsiye ediliyordu aldım denicez | positive | neutral | neutral | neutral | neutral |
| 45 | İlk defa aldım ama kokusu muhteşem güzel beğendim..😍😍 | negative | positive | positive | positive | positive |

## Sözlükte eşleşmeyen sık kökler

| Kök | Görülme |
| --- | --- |
| kullan | 220 |
| gel | 132 |
| beğen | 104 |
| yap | 86 |
| ver | 60 |
| bekle | 38 |
| sev | 37 |
| sür | 34 |
| düşün | 31 |
| bırak | 26 |
| icin | 24 |
| çık | 23 |
| başla | 22 |
| dene | 21 |
| görün | 21 |
| bil | 19 |
| dökül | 18 |
| paketle | 16 |
| kapat | 16 |
| git | 16 |
| yapış | 16 |
| giy | 15 |
| göster | 15 |
| temizle | 15 |
| nemlen | 15 |
| aldim | 13 |
| um | 12 |
| iste | 11 |
| vazgeç | 11 |
| yıka | 10 |

Bu liste fiil kökü/lemma uyuşmazlığı, yazım hatası, özel ad ve kapsam dışı sözcükleri birlikte içerir; otomatik olarak sözlüğe eklenecek madde listesi değildir.

## Dosyalar

- `analysis.json`: tüm metrikler, işlem sayıları, ID listeleri ve tanılar.
- `comparison_1200.csv`: her yorumda dört tahmin ve pozitif/negatif skorlar.
- `errors_1200.csv`, `disagreements_1200.csv`: hata ve ayrışma örnekleri.
- `scope_details_1200.csv`: tüm dependency kapsamlarının skor, işlem, hedef ve olumsuzluk ayrıntıları.
- `diagnostic_cases.json`: ayrışan ve seçilmiş ortak hatalı örneklerin tam ara çıktıları.
- `per_class_1200.csv`, `pairwise_1200.csv`, `slice_metrics_1200.csv`: ayrıntılı metrikler.
- `progress.sqlite3`: kaldığı yerden devam kaydı ve tam model ara çıktıları.
