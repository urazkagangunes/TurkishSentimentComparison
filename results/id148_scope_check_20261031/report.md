# ID 148 olumsuzluk kapsam kontrolü — 31 Ekim 2026

**Karar: kapsam kuralındaki hata düzeltildi.** ID 148'deki yanlış skor çevirme
giderildi. Aynı kural, incelemede bulunan ID 447'deki çift olumsuzluk hatasını da
giderdi. HisNet → DP ve LW HisNet → DP'de iki yanlış tahmin düzeldi; bu 500'lük
örneklemde doğru olup yanlış hâle gelen tahmin yok.

## ID 148: ne yanlış gidiyordu?

Metin: `İi değil iade edeceğim` — kayıtlı etiket: `negative`.

Kayıtlı parser çıktısında `değil` kök düğüm, `iade` onun `obj` çocuğu;
`edeceği` ise `iade`nin `compound:lvc` çocuğu. Son `m` ayrı cümleye bölünmüş.
Bu kontrol sırasında parser çıktısı aynen korundu.

HisNet'te `iade` için pozitif puan 0, negatif puan 0,125. Eski kapsam kuralı,
üst düğüm `değil` olduğunda başka koşul aramadan onu olumsuzluk listesine
ekliyordu. Böylece sonraki `iade edeceğim` yüklemi, önceki olumsuzluğun kapsamına
giriyor ve negatif puan pozitife çevriliyordu. `İi` duygu adayı olmadığı için
konumsal düzeltme aşamasında `değil`den önce duygu kapsamı bulunmuyor; bu aşama
erken dönerek hatalı ilişkiyi koruyordu.

| Sistem | Önce | Sonra | Son pozitif / negatif puan |
| --- | --- | --- | --- |
| HisNet | negative | negative | 0 / 0,125 |
| DP → HisNet | neutral | neutral | 0 / 0 |
| HisNet → DP | positive | negative | 0 / 0,125 |
| LW HisNet → DP | positive | negative | 0 / 0,125 |

DP → HisNet bu örnekte duygu adayı üretmiyor. Kapsam düzeltmesi aday seçimini
değiştirmediği için onun nötr sonucu aynı kaldı.

## Uygulanan kural

`SentimentScopeResolver._find_negations` içinde, açık `değil` üst düğümünden
olumsuzluk devralmadan önce şu yapı kontrol ediliyor:

- Duygu sözcüğü, `değil`den sonra ve aynı cümlede mi?
- Bu sözcüğe `compound:lvc` bağıyla bağlı bir VERB/AUX var mı?
- Bu yardımcı fiil de aynı cümlede ve `değil`den sonra mı?

Bu koşullarda sonraki yardımcı fiilli yüklem, önceki `değil`i devralmıyor.
Kendi yardımcı fiili olumsuzsa o olumsuzluk korunuyor. Kural örnek ID'sine,
`iade` sözcüğüne, altın etikete veya olumlu/olumsuz sözlük puanına bağlı değil.

Yalnız sözcük sırasına dayanan bir yasak eklenmedi. `Güzel değil`, `kötü değil`,
`tavsiye edecek değilim`, sentetik devrik `değil kötü` yapıları ve
`iyi ve güzel değil` ortak kapsamı korunuyor. Erken dönüş tümüyle kaldırılmadı;
devrik yapılarda başlangıçtaki geçerli bağı koruması gerekebilir.

## Kapsam taramasında bulunan ID 447

`Daha çnce kullandım böyle değildi sanki içine limon kolanyası basmışlar gibi bir ürün kesinlikle almanızı tavsiye etmem`
— kayıtlı etiket: `negative`.

HisNet → DP ve LW'de `tavsiye` hem önceki `değildi`yi hem kendi `etmem`
yardımcı fiilini alıyordu. İki olumsuzluk nedeniyle pozitif sözlük puanı
çevrilmiyordu. Düzeltmeden sonra yalnız `etmem` kaldı:

- Önce: pozitif 0,125; negatif 0,166667 → `neutral`.
- Sonra: pozitif 0; negatif 0,291667 → `negative`.

DP → HisNet'te sonraki kapsam aşaması bu bağı zaten temizliyordu. Yeni kod
başı itibarıyla hatalı bağı eklemiyor; yalnız tanısal
`negation_scope_strategy` alanı değişiyor. Skorlar ve tahmin aynı.

## 500 kayıt üzerinde doğrulama

Kaynak: `results/trsav1_classical_500_20261031_run01/progress.sqlite3`.
Dosya salt okunur açıldı. Eski resolver'ın değişiklik öncesi kopyasıyla her
dependency kolundaki 500 sonuç yeniden hesaplandı. Toplam **1.500 sonuçta**
tahmin, skor, kapsam ve ilgili ara çıktılar kayıtlı sonuçlarla birebir eşleşti.
Ardından aynı girdiler yeni resolver ile işlendi.

Bu, kayıtlı parse, morfoloji ve sözlük sonuçlarından **kapsam/bileşim yeniden
hesabıdır**. Yeni uçtan uca model koşusu veya bağımsız test seti değildir.
Stanza, morfoloji ve HisNet başlatılmadı; LLM çağrısı yapılmadı. Üretim
sınıflandırıcılarının bileşim kodu kullanıldı; gerçek model kurucuları hata
verecek şekilde engellendi. Yeni bir sözlük sorgusu gerekirse kontrol de hata
verecek şekilde tasarlandı.

| Sistem | Doğru önce → sonra | Accuracy önce → sonra | Macro-F1 önce → sonra |
| --- | --- | --- | --- |
| HisNet | 263 → 263 | %52,60 → %52,60 | %50,20 → %50,20 |
| DP → HisNet | 262 → 262 | %52,40 → %52,40 | %50,74 → %50,74 |
| HisNet → DP | 270 → 272 | %54,00 → %54,40 | %51,23 → %51,56 |
| LW HisNet → DP | 269 → 271 | %53,80 → %54,20 | %51,09 → %51,42 |

HisNet → DP kapsamlarında açık `değil` üst düğümlü sekiz bağ bulundu:
ID 148, 215, 295, 398 (iki bağ), 447, 469 (iki bağ). Etiket değiştiren
örnekler yalnız 148 ve 447. Her iki H→DP kolundaki diğer 498 örneğin
karşılaştırılan ara çıktıları da değişmedi. DP→H'de 447'nin tanısal alanı
dışında fark yok. ID 98'in kayıtlı çıktısı aynı kaldı.

Kaynak SQLite dosyasının SHA-256 değeri işlem öncesinde ve sonrasında aynı:
`20f49471d0c75e4a39b9a88cb341c017ddf545f473fa66ecb14208a70f5ef9d3`.
Eski 500'lük koşunun tahminleri, raporları ve manifesti güncellenmedi.

## Testler ve yeniden üretme

12 yeni çevrimdışı kapsam testi ve 19 mevcut regresyon testi geçti.
Yeni testler, düzeltme öncesinde hatayı yeniden üretti; düzeltmeden sonra geçti.
Sözcük çeşitleri, olumlu/olumsuz yardımcı fiiller, eksik polarity bilgisi,
devrik yapı, ortak olumsuzluk, karşıtlık ve cümle sınırları kontrol edildi.
Mevcut testlerdeki LLM hata senaryoları mock kullanır; gerçek LLM çalıştırılmaz.

```powershell
.venv\Scripts\python.exe -m unittest tests.test_explicit_negation_scope -v
.venv\Scripts\python.exe -m unittest discover -s tests -p test_audit_regressions.py -v
.venv\Scripts\python.exe -m scripts.check_saved_negation_scope --baseline-resolver results/id148_scope_check_20261031/baseline_sentiment_scope_resolver.py
```

`summary.json` metrikleri ve kaynak/kod özetlerini, `changes.json` beş değişen
sistem–örnek çiftinin önce/sonra ara çıktılarını,
`explicit_degil_parent_edges.json` sekiz bağı içerir. Eski resolver kopyası
yalnız karşılaştırmanın yeniden üretimi için bu klasörde saklanır.

Kod değiştiği için eski checkpoint güncel kodla sürdürülmemelidir; mevcut
manifest kontrolü bu karışmayı reddeder. İleride uçtan uca koşu istenirse
`--output` ile yeni bir sonuç klasörü kullanılmalı. Bu inceleme için böyle
bir koşu başlatılmadı.

## Paper için yorum

Bu örnekler, parser bağlantı hatasının aşırı geniş olumsuzluk devralma kuralıyla
büyütülebildiğini gösteriyor. Paper'da hata türü, dar yapısal düzeltme ve
değişiklik öncesi/sonrası sonuçlar birlikte sunulabilir. Sözlük kapsamı,
morfolojik sınırlamalar ve ID 98'deki token bölünmesi ayrı bulgular olarak kalır.

Kural yalnız saptanan yardımcı fiilli yapıya yöneliktir; tüm Türkçe olumsuzluk
ve parser hatalarını çözmez. İncelenmiş bu 500'lük örneklem geliştirme verisidir.
İki ek doğru tahmin, tek başına genel başarı veya istatistiksel üstünlük
iddiasını desteklemez; son değerlendirme bağımsız veriyle yapılmalıdır.
