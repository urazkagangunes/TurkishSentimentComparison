# 3000 yorumluk koşu: sonuçların yorumu ve sonraki adımlar

**Tamamlandı: 3000 yorum × 4 yöntem = 12.000 başarılı çıktı.** Her sınıftan
1000 yorum var; LLM ve DP→LLM çalıştırılmadı. Önceki 1200 metinle örtüşme sıfır.
Koşu 1 Kasım'da başlatıldı, kesintiler sonrasında 3 Kasım 2026'da tamamlandı.

HisNet→DP bu örneklemde en yüksek doğruluk ve macro-F1 değerini verdi. Ancak
DP→HisNet ile macro-F1 farkı yalnız **0,09 yüzde puanı**; bu iki yöntem arasında
kesin bir üstünlük sonucu çıkarmak doğru olmaz. Büyük ortak sorun nötr sınıf ve
duygu taşıyan fiillerin sözlükteki biçimlerle eşleşmesidir.

## 1. Genel sonuç

| Yöntem | Doğru / 3000 | Doğruluk | Macro-F1 | Nötr recall | Nötr F1 |
|---|---:|---:|---:|---:|---:|
| HisNet | 1477 | %49,23 | %46,57 | %19,10 | %25,48 |
| DP→HisNet | 1507 | %50,23 | %48,11 | **%21,80** | **%27,28** |
| HisNet→DP | **1530** | **%51,00** | **%48,20** | %19,40 | %25,97 |
| LW HisNet→DP | 1514 | %50,47 | %47,52 | %18,30 | %24,63 |

Dengeli örneklemde çoğunluk sınıfı tabanı %33,33. Sonuçlar bu tabanın üzerinde,
ancak doğruluk yaklaşık %50 ve nötr sınıf zayıf. Bu ölçüm başarılı bir teknik
koşudur; yüksek sınıflandırma başarısı veya bağımsız final paper sonucu anlamına gelmez.

Sınıf bazında tüm precision/recall/F1 değerleri, karışıklık matrisleri ve yaklaşık
Wilson güven aralıkları [ana sayısal raporda](report.md) bulunuyor.

## 2. Hangi farklar destekleniyor?

Aynı 3000 yorum üzerindeki doğru/yanlış değişimleri için iki yönlü exact McNemar
testi kullanıldı; altı karşılaştırmaya Holm düzeltmesi uygulandı. Bunlar keşifsel
karşılaştırmalardır. Test doğruluk farkına ilişkindir; macro-F1 farkının testi değildir.

| Karşılaştırma | İkinci yöntemle düzeltilen | Bozulan | Net doğru farkı | Holm p |
|---|---:|---:|---:|---:|
| HisNet → DP→HisNet | 156 | 126 | +30 | 0,3220 |
| HisNet → HisNet→DP | 84 | 31 | **+53** | **4,82 × 10⁻⁶** |
| HisNet → LW | 105 | 68 | **+37** | **0,0302** |
| DP→HisNet → HisNet→DP | 117 | 94 | +23 | 0,3220 |
| DP→HisNet → LW | 120 | 113 | +7 | 0,6944 |
| HisNet→DP → LW | 29 | 45 | −16 | 0,3220 |

HisNet→DP, HisNet'e göre **1,77 yüzde puanı** doğruluk kazanıyor; bu fark mevcut
örneklemde destekleniyor. DP→HisNet'in HisNet'e göre +1,00 puanlık farkı Holm
düzeltmesi sonrasında anlamlı değil. LW, HisNet'i geçse de düz HisNet→DP'yi
geçemedi. LW'nin daha kötü olduğunun genellenebilir biçimde kanıtlandığı da söylenemez.

Örneğin ID 35, “Tavsiye etmiyorum 🙄”: HisNet `tavsiye` için +0,125 üretip pozitif
derken bağımlılık yöntemleri olumsuzluğu bağlayarak −0,125 ve doğru negatif
kararını üretiyor. Bu, bağımlılık bilgisinin fayda sağladığı somut bir örnek.

## 3. Yeni çekimli yüklem düzeltmesi yeni örneklerde işe yaradı mı?

Evet, fakat etkisi küçük. Yeni koşunun parse, morfoloji ve sözlük çıktıları sabit
tutularak düzeltme öncesi resolver ile yeniden bileşim yapıldı. Önce güncel resolver'ın
**9000 bağımlılık çıktısını birebir yeniden ürettiği** doğrulandı. Sonra yalnız
resolver sürümü değiştirildi. Bu, yeni bir uçtan uca model koşusu değildir.

| Yöntem | Önce doğru | Sonra doğru | Düzeltilen / bozulan | Etiketi değişen |
|---|---:|---:|---:|---:|
| HisNet | 1477 | 1477 | 0 / 0 | 0 |
| DP→HisNet | 1505 | 1507 | **2 / 0** | 3 |
| HisNet→DP | 1529 | 1530 | **1 / 0** | 1 |
| LW | 1513 | 1514 | **1 / 0** | 1 |

- **ID 1847:** “...bende üzgünüm...” ifadesi DP→HisNet adaylarına geri girdi.
  `üzgün` için 1,0 negatif puan korununca nötr → negatif oldu; gold negatif.
- **ID 2467:** “Üzgünüm pek beğenemedim. Küçücük bir şey”. Önce DP→HisNet
  `üzgünüm`ü aday seçmiyor, diğer iki bağımlılık yöntemi hedef sayıp puanını siliyordu.
  Üçü de pozitif → doğru negatif oldu.
- **ID 1052:** `kalitesizmiş` adaylara eklenince DP→HisNet pozitif → negatif
  oldu; gold nötr olduğundan iki sürüm de yanlış. Bu satır kazanç olarak sayılmadı.

1200 örnekte “pişmanım” üzerinden bulunan hata, yeni sette **“üzgünüm”** için de
düzeldi; kural belirli bir sözcüğe bağlı değil. Bu veride doğru → yanlış geçiş
olmaması olumlu, fakat 1–2 örneklik kazanç geniş bir performans üstünlüğü kanıtı değildir.
İki düzeltme ve sıfır bozulma için exact McNemar p=0,5; tek düzeltme için p=1,0.

Kural korunuyor. Eşik, sözlük, morfoloji modeli ve LW ağırlıkları değiştirilmedi.
[Ayrı karşılaştırma raporu](../trsav1_classical_3000_20261101_finite_ablation/report.md)
ve [tam değişen kapsamlar](../trsav1_classical_3000_20261101_finite_ablation/changed_details.json)
incelenebilir.

## 4. Ana darboğaz: nötr yorumlar

Gerçekte 1000 nötr yorum var. Yöntemler yalnız 486–598 yoruma nötr diyor; bunların
önemli bir bölümü de gerçek pozitif/negatif yorumlar. En yüksek nötr recall
DP→HisNet'te **218/1000**. HisNet→DP, 1000 nötr yorumun 194'ünü yakalıyor;
337'sini negatif, 469'unu pozitif sınıflıyor. Toplam 1470 hatasının **806'sı
(%54,83)** nötr sınıftan geliyor.

Bu yalnız bir eşik meselesi değil:

- **ID 2:** “temizleyici özelliği güzel ama kokusu kötü”, gold nötr. `güzel`
  +0,4545, `kötü` −0,9; ham puan farkı negatif sonuç veriyor. LW ikinci bölümü
  1,85 ile ağırlıklandırınca negatif puan 1,665 oluyor. Bu örnek, nötr etiketinin
  karma değerlendirmeyi de kapsayabildiğini gösteriyor.
- **ID 0:** “...kalıcılık konusunda iyi fakat taramasını sevmedim”, gold nötr.
  `güzel/iyi` puanlanıyor; `sevmedim`in kökü `sev` sözlükte eşleşmiyor. Dolayısıyla
  olumlu ve olumsuz işaretler karşılaştırılmadan önce bir taraf kayboluyor.
- **ID 11:** “Beğenmedik çünkü kafada iyi durmuyor delikli gözüküyor”, gold negatif.
  `beğen` eşleşmiyor ve kayıtlı kapsamda `iyi` pozitif kalıyor. Dört yöntem de
  pozitif diyor. Mevcut kapsam kuralları her olumsuz yüklem–niteleyici ilişkisini çözmüyor.

Bu örnekler hata mekanizmalarını açıklıyor; tüm hataların bu kategorilere
ayrıldığı veya oranlarının elle etiketlendiği iddia edilmiyor. Gold etiketler değiştirilmedi.

## 5. Sözlük kapsamı: bulunan sözcük ile bulunan duygu aynı şey değil

HisNet'in yorum başına ortalama sözlük eşleşmesi %80,54; ortalama duygu taşıyan
eşleşme oranı %23,09. DP→HisNet'in adaylar üzerindeki eşleşmesi %74,68; aday
kümesi farklı olduğundan bu oran HisNet'in bütün sözcükler üzerindeki oranıyla
doğrudan eşdeğer değildir.

| Yöntem | Pozitif ve negatif puanı birlikte sıfır olan yorum | Bunların pozitif/negatif gold olanları |
|---|---:|---:|
| HisNet | 350 | 218 |
| DP→HisNet | 484 | 308 |
| HisNet→DP | 350 | 218 |
| LW | 357 | 223 |

DP aday seçimi bazı gereksiz sözcükleri elerken bazı duygu ipuçlarını da
kaybediyor. Sıfır puanlı yorumların hepsi gerçekten nötr değil. Yalnız eşiği
değiştirmek, sıfır puan yüzünden kaçırılan bu yorumlara duygu bilgisi kazandırmaz.

En sık eşleşmeyen köklerden bazıları `kullan` (566), `gel` (332), `yap` (247),
**`beğen` (224), `sev` (99), `bayıl` (33)**. Sayılar sözcük görülme sayısıdır,
benzersiz yorum sayısı değildir. Fiil kökü ile sözlük biçimini yerel XML kaynağında
salt okunur kontrol ettim:

| Biçim | Sözlükte var mı? | Pozitif | Negatif |
|---|---|---:|---:|
| beğen | hayır | — | — |
| beğenmek | evet | 0,75 | 0 |
| sev | hayır | — | — |
| sevmek | evet | 0,60 | 0 |
| bayıl | hayır | — | — |
| bayılmak | evet | 0 | 0,25 |
| al | evet | 0 | 0,1667 |
| almak | evet | 0 | 0,0143 |

Bu, fiil kökü–sözlük biçimi uyumsuzluğunu doğruluyor. Ancak körlemesine `-mak/-mek`
eklemek de güvenli bir çözüm değil: `bayılmak` sözlükte negatif; yorumdaki
“ürüne bayıldım” kullanımını otomatik olarak pozitif yapmıyor. POS ve anlam
ayrımıyla birlikte incelenmeli. [Kaynak hash'i ve doğrudan sorgu kaydı](lexicon_form_probe.json)
saklandı; bu koşunun sorgu davranışı değiştirilmedi.

## 6. LW ağırlıkları neden her zaman yardımcı olmadı?

LW, HisNet→DP'ye göre 96 yorumun etiketini değiştirdi: 29 hata düzeltti, 45 doğru
kararı bozdu, net 16 doğru kaybetti. Bu örneklemde ağırlıkların ek faydası gösterilemedi.

Somut örnek **ID 250**: “Ürün idare eder ama yüzde yağlılık hissi vermesini
sevemedim”, gold nötr. `sev` eşleşmiyor; `yağlı` kökünün +0,035714 puanı düz
HisNet→DP'de 0,05 nötr eşiğinin altında kalıyor. LW'nin 1,85 karşıtlık ağırlığı
bunu **+0,066071** yapıp kararı yanlış pozitife çeviriyor. Burada hesaplama hatası
yok; düşük bir sözlük puanı, eksik olumsuz ipucu varken eşik üstüne taşınıyor.

İşlem sayıları yorum sayıları değildir: LW'de 1866 kapsam karşıtlık, 769 kapsam
niteleyici ağırlığı aldı. HisNet→DP'de 294 olumsuzluk çevirme ve 237 hedef bastırma
işlemi var. 7207 kapsamın “matched” olması eşleştirme yordamının eşleşme bulduğunu
gösterir; her parse, kök veya anlamsal kapsamın doğru olduğunu kanıtlamaz.

## 7. Veri kalitesi ve ortak hatalar

1322 yorumda dört yöntem de doğru, **1330 yorumda dördü de yanlış**. 481 yorumda
yöntemlerin tahminleri farklı. Bu sayılar bir bölümleme değildir: farklı tahmin
veren yöntemlerin hepsi aynı yorumda yanlış olabilir.

Kaynakta başka etiketle de bulunan 43 metin (%1,43) var; seçilen 3000 içinde
tekrarlanan metin yok. Bu 43 satır ana ölçümden çıkarılmadı. Duyarlılık kontrolünde
hariç tutulduklarında doğruluklar sırasıyla **%49,48 / %50,49 / %51,27 / %50,73**
oluyor; sıralama ve ana sorun değişmiyor. Doğrulanmış bu etiket çelişkileri tek
başına yaklaşık %50 doğruluğu açıklamıyor.

## 8. Önceki 1200 ile nasıl karşılaştırılmalı?

| Yöntem | 1200: güncel kuralla yeniden bileşim doğruluğu | Yeni 3000 doğruluğu |
|---|---:|---:|
| HisNet | %47,67 | %49,23 |
| DP→HisNet | %50,33 | %50,23 |
| HisNet→DP | %49,67 | %51,00 |
| LW | %49,50 | %50,47 |

**Bunlar farklı metin kümeleri.** Bu tablodaki fark düzeltmenin etkisi veya
eşleştirilmiş bir iyileşme değildir. Düzeltmenin etkisi 3. bölümde aynı 3000
yorum üzerinde ölçüldü. Önceki sette DP→HisNet'in, burada HisNet→DP'nin önde
olması da tek küçük örneklemden kesin sıralama çıkarılmaması gerektiğini gösteriyor.

## 9. Yapılanlar ve bundan sonraki sıra

Tamamlanan işler: çekimli sıfat yüklemi koruması, 1200 kayıt üzerinde eski/yeni
kontrolü, önceki metinleri dışlayan 3000 örneklik koşu, ortak ön işleme ve üç
işçiyle hızlandırma, bağımsız doğrulama, yeni örneklerde ayrı düzeltme karşılaştırması.

Bir sonraki adım yalnız örneklemi tekrar büyütmek olmamalı:

1. **Fiil kökü–sözlük biçimi eşleştirmesini ayrı bir yöntem varyantı olarak ele al.**
   POS, mastar biçimi ve anlam belirsizliği için olumlu/olumsuz karşı örnekler ekle;
   dört yöntemde adil biçimde aynı yeni sözlük arayüzünü kullan. Mevcut sonuçlar taban olarak korunmalı.
2. **Nötr kararını ayrı geliştirme/doğrulama ayrımında kalibre et.** Karma görüş,
   düşük duygu ve bilgi yetersizliğini aynı ham puan farkıyla ele almak incelenmeli.
   Yeni lexical eşleştirme puan dağılımını değiştireceği için eşik ayarı bundan sonra yapılmalı.
3. **LW'nin karşıtlık ve niteleyici bileşenlerini ayrı karşılaştır.** Bu setteki
   96 etiket değişiminden hareketle bütün ağırlıkların faydalı olduğu varsayılmamalı.
4. Kurallar ve ayarlar sabitlenince, geliştirmede kullanılmamış bir ayrım/korpus
   üzerinde ana paper değerlendirmesini yap. Bu 3000 örnek artık incelenmiş geliştirme verisidir.

Bu yeni varyantlar mevcut koşuya geriye dönük uygulanmadı; raporlanan dört yöntem,
eşik ve sözlük puanları sabit kaldı. Örneğe özel sözcük puanı veya gold etiketi değişikliği yapılmadı.

## 10. Doğrulama ve süre sınırı

- 78 farklı çevrimdışı test geçti. Son karşılaştırmada hedefli 38 test de geçti.
- Paralel geçiş öncesi 12 tam semantik karşılaştırma geçti.
- Koşu sonunda 7 örnek × 4 yöntem = **28 bağımsız tam semantik karşılaştırma** geçti.
  İlk sınıf örnekleri, yeni işlenen sınıf örnekleri ve her işçinin ilk örneği kapsandı.
  Bu örnek kontrolleri 3000 yorumun bağımsız ikinci kez çalıştırılması değildir.
- 12.000 başarılı çıktı, 3000 ortak parse, sıfır LLM girişimi, sıfır sayısal/kapsam
  tutarsızlığı; kaynak, kod, model ve örneklem hash kontrolleri geçti.
- Son düzeltme karşılaştırması SQLite'ı salt okunur kullandı; hash değişmedi.
- **Süreler hız karşılaştırmasına uygun değil.** 1–3 Kasım arasındaki uzun
  kesintiler perf_counter sürelerine yansıdı. Paralel süre toplamları CPU süresi
  değildir. Ham kayıtlar korundu; kesinti süresi tahmin edilip çıkarılmadı.

Belgeler: [sayısal rapor](report.md), [son denetim](final_audit.json),
[test notları](validation_notes.json), [bağımsız doğrulama](independent_verification.json),
[protokol](protocol.md), [süre sınırlaması](timing_caveats.json),
[satır bazında dört tahmin](comparison_3000.csv), [tam kapsamlar](scope_details_3000.csv).
