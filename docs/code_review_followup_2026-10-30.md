# Altı sistem — ikinci ayrıntılı inceleme, 30 Ekim 2026

**Sonuç: Repo henüz “bug kalmadı / nihai deney için hazır” durumunda değil.**
Önceki 30 cümlelik koşunun tamamlanması, o koşudaki teknik kontrollerin geçtiğini
gösteriyordu. Aşağıdaki yeni girdiler, kalan kapsam ve hizalama kusurlarını ortaya
çıkarıyor. Bu tur sınıflandırıcı kuralları, katsayılar, dataset veya önceki deney
sonuçları değiştirilmedi; tekrar üretme scripti ve inceleme kayıtları eklendi.

## Kapsam ve doğrulama

- Altı sınıflandırıcının kaynakları; ortak morfoloji, Stanza wrapper ve kapsam
  çözümleyicisi; dataset yükleyici, evaluator/checkpoint ve run identity kodu.
- 30 cümlelik runner/analiz scripti; 500 örneklik runner'ın özellikle cache,
  metrik ve süre akışı; klasik 500 ve altı-model 16 scriptlerinin ilgili bölümleri.
- Mevcut offline regresyon paketi: **19/19 geçti**.
- `src/`, `scripts/`, `tests/`: **73 Python dosyası AST/sözdizimi kontrolünden geçti**.
- Gerçek Stanza/HisNet ile **16 metin × 4 klasik kol = 64 sınıflandırma** tamamlandı.
  Bunlar teşhis girdileridir; yeni bir accuracy benchmark'ı değildir.
- Hizalama, kesinti sonrası devam ve eski runner süre hesabı kontrollü girdilerle
  ayrıca çalıştırıldı. Bu kontrollerde gerçek LLM isteği gönderilmedi.
- LLM kollarının canlı 32B sonuçları için önceki tamamlanmış `run02` kanıtı mevcut:
  30'ar satır, teknik hata yok. Bu incelemede 32B koşusu tekrarlanmadı.

Tekrar üretme:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p test_audit_regressions.py -v
.venv\Scripts\python.exe -u -m scripts.audit_six_model_followup --live-classical --datasets --output results/audit_followup_new/evidence.json
```

Mevcut kayıtlar:

- [Gerçek klasik çıktılar ve ilk kontrollü testler](../results/audit_followup_20261030/evidence.json)
- [Güncel offline, süre ve dataset kontrolleri](../results/audit_followup_20261030/offline_evidence.json)
- [Tekrar üretme scripti](../scripts/audit_six_model_followup.py)

Kontrollü hizalama testleri doğru morfoloji kayıtlarını ve gerektiğinde yapay
lexicon skorlarını dışarıdan sağlar. Bunlar kodun verilen doğru girdiyi yanlış
eşlediğini gösterir; `yazar` sözcüğünün gerçek HisNet'te pozitif olduğunu veya
hataların gerçek veri üzerindeki sıklığını göstermez.

## Doğrulanmış hatalar

### 1. Yüksek: `değil`, ilgisiz duygu sözcüğüne taşınabiliyor

Etkilenen: **H → DP, DP → H, LW H → DP**.

Konum: [sentiment_scope_resolver.py](../src/dependency/sentiment_scope_resolver.py),
`_disambiguate_shared_explicit_negations`, özellikle 1471 ve 1565. Ek olarak
`_find_negations` içindeki 1256. satırdaki açık `değil` üst-düğüm kuralı geniş.

Resolver aynı cümledeki önceki duygu adaylarından en yakınını seçiyor. Seçilen
adayın `değil`in gerçek yüklemi olması şart değil; duygu sözlüğünde bulunmayan
ara yüklem ve virgülle ayrılan yapı yeterince korunmuyor. Önceki düzeltme farklı
cümleler arasındaki taşmayı engelliyordu; aynı cümlenin ayrı bölümlerini çözmüyor.

Gerçek çalıştırma:

| Metin | HisNet | H → DP | DP → H | LW H → DP |
|---|---|---|---|---|
| Otel kötü, merkezi değil. | negative | **positive** | **positive** | **positive** |
| Otel kötü. Merkezi değil. | negative | negative | negative | negative |
| Otel güzel, merkezi değil. | positive | negative | negative | negative |
| Bu bir aşk filmi değil. | positive | negative | neutral | negative |
| Bu bir korku filmi değil. | negative | positive | neutral | positive |

İlk örnekte `kötü`nün 0,9 negatif puanı 0,9 pozitife çevriliyor. Gerçek Stanza
çıktısı da sorunlu: `değil` CCONJ/root, `merkezi` NOUN/obj oluyor. Bu nedenle
gerçek örneği yalnızca bir post-processing satırına bağlamak doğru olmaz.
**Bununla birlikte**, kontrollü doğru ağaçta `değil` doğrudan `merkezi`ye
bağlandığında da resolver ilgisiz `kötü` adayına olumsuzluğu taşıyor. Uygulama
kusuru, parser hatasından bağımsız olarak da doğrulandı.

Tür belirten “aşk/korku filmi değil” örnekleri ayrıca bağlamsız lexicon puanının
gerçek bir olumlu/olumsuz görüşle aynı şey olmadığını gösteriyor. Bu iki örnekteki
hataların tamamını negation fonksiyonuna yüklemek doğru olmaz.

Düzeltme yönü: mevcut doğru dependency kapsamını önceliklendirmek; yeniden
atamayı gerçek koordinasyon/yüklem bağlantısıyla sınırlamak; duygu taşımayan
yüklemi atlayıp uzaktaki duygu adayına gitmemek. “İyi ve güzel değildi” gibi
gerçek paylaşılan kapsamlar korunmalı.

### 2. Yüksek: Fiil kökü ile lexicon madde biçimi hâlâ uyuşmuyor

Etkilenen: **dört klasik kol**. Önceden bulunan, henüz giderilmemiş sorun.

Konum: [hisnet_classifier.py](../src/hisnet/hisnet_classifier.py), `_lookup_root`
83 ve `classify` 232–241; DP → H de aynı lookup'ı kullanıyor.

Gerçek morfoloji `sev` ve `beğen` üretiyor; sözlükte bu anahtarlar yok.
`sevmek` için 0,60, `beğenmek` için 0,75 pozitif puan mevcut.

| Metin | Dört klasik kol |
|---|---|
| Bu filmi sevdim. | neutral |
| Bu filmi sevmedim. | neutral |
| Bu ürünü beğendim. | neutral |
| Bu ürünü beğenmedim. | neutral |

Bu yalnızca sözlüğün kelimeyi bilmemesi değil, bileşenlerin temsil uyuşmazlığı.
POS/türetim bilgisine dayalı ortak lexicon maddeleştirmesi gerekir.

**Bağlı yöntem sorunu:** Resolver 1141–1172'de duygu token'ının kendi
`Polarity=Neg` özelliğini bilerek skor çevirme dışında bırakıyor. Yalnızca lookup
düzeltilirse `sevmedim` pozitif kökün puanını alabilir. Ekli olumsuzluk politikasını
ayrıca kararlaştırıp pozitif/negatif fiil çiftleriyle doğrulamak gerekiyor.
HisNet-only baseline'ın bağımlılık kuralı kullanmaması ise tek başına bug değildir.

### 3. Orta: H → DP/LW, tekrar eden yüzeyin yanlış kullanımını seçebiliyor

Konum: [sentiment_scope_resolver.py](../src/dependency/sentiment_scope_resolver.py),
`resolve` 3156–3164 ve `_match_sentiment_token`.

Exact-surface önceliği, yalnızca `sentiment_words` altkümesi üzerinden eşleme
yapıyor. Önceki aynı-yazımlı kullanım duygu adayı değilse token'ı ayıran bir kayıt
yok. İkinci kullanımın duygu adayı ilk token'a atanabiliyor.

Kontrollü örnek: morfoloji sırasıyla `yazar/NOUN` ve `yaz/VERB` köklerini verdiğinde,
yalnızca ikinci kullanımın aday olduğu senaryoda beklenen token ID **2**, çıkan
ID **1**. İki kullanım ayrı cümlelerdeyse duygu adayı yanlış cümleye de taşınabiliyor.

Önceki “fallback exact eşleşmeyi çalmasın” düzeltmesi bu altküme sorununu çözmüyor.
Morfolojik kayıtların token sırası/konumu adaylara taşınıp tüm token'lar üzerinden
tek ve tutarlı eşleme yapılmalı.

### 4. Orta: DP → H'de gürültülü/onarılmış tekrarlar kök sırasını bozuyor

Konum: aynı resolver, `_find_morphological_root` 2055–2063,
`_find_morphology_record_for_surface` 2137–2175, `_build_dependency_repair_units`.

Normal occurrence sayacı yalnızca ham yüzeyin tam eşleşmesini sayıyor. İlk
`yazar🙂` kullanımının ardından `yazar` gelirse ikinci token için önceki kullanım
sayılmıyor. Onarım lookup'ı da eşleşen ilk morfolojik kaydı döndürüyor.

Kontrollü testte doğru kök sırası `[yazar, yaz]` iken sonuç `[yazar, yazar]`.
Önceki tekrar düzeltmesi normal tam-yüzey token'larda çalışıyor; bu onarım yolunu
kapsamıyor. Ortak occurrence/offset eşlemesi onarılmış token'lara da uygulanmalı.

### 5. Orta: Eski 500 örneklik LLM runner'ında resume süreleri yanlış

Konum: [test_four_systems_500.py](../tests/test_four_systems_500.py),
`run_llm_batch_condition`: 1015–1019, 1487–1582.

Tahminler diskten yükleniyor, ancak süre/token/istek sayaçları yeni process'te
sıfırdan başlıyor. Son hız hesabının payında bütün sample uzunluğu kullanılıyor.

Fonksiyonun kendisi, üst seviyedeki 500 örneklik deneyi başlatmadan, AST üzerinden
izole edilerek çalıştırıldı: önceden tamamlanmış tek satır, **0 yeni model isteği**
ve **0 token** için yaklaşık **0,00057 saniye**, **1740 review/s** raporlandı.
Bu modelin inference hızı değil, cache yüklenmiş koşunun boş işlem süresi.
Değer makineye göre değişir; hata sürelerin kapsamı ile paydanın uyuşmaması.

`run02` için kullandığımız yeni analiz scripti kayıtlı çağrıları toplayarak bu
sorundan kaçınıyor. Ancak eski runner hâlâ düzeltilmeli; onun süre tablosu resume
sonrasında doğrudan performans karşılaştırmasında kullanılmamalı.

## Operasyonel ve deney tasarımı açıkları

### Kesintiden sonra kalan recovery unutulabiliyor

[run_six_model_smoke.py](../scripts/run_six_model_smoke.py): 189 ve 212.
`pending`, ID'si hiç kaydedilmemiş satırlardan oluşuyor. Başarısız batch kaydedilip
bireysel recovery'den önce süreç kesilirse, yeni süreç o başarısız satırı tekrar
ele almıyor. Kontrollü testte başarısız `A` kaydı kaldı; yalnızca yeni `B` çağrıldı.

Bu sınırlama önceki çalışma notlarında zaten belirtilmişti. “Aynı komutla devam”
beklentisi açısından eksik akış; ayrıca açık recovery koşusu veya kalıcı recovery
kuyruğu gerekir. Başarısız satır nötre çevrilmiyor; sonuç eksik olarak kalıyor.
Tamamlanan `run02`de başarısız satır olmadığından mevcut sonuçlar etkilenmedi.

### Varsayılan LLM istemcisinde timeout yok

[ollama_zero_shot_classifier.py](../src/llm/ollama_zero_shot_classifier.py): 108.
Kurulu istemcinin çalışma anındaki değeri `Timeout(timeout=None)`.
Sunucu hiç dönmeyen bir isteğe takılırsa retry/failed akışı başlayamayabilir.
Bu, ortak tabanı kullanan iki LLM kolunu etkiler. Smoke runner'ın kendi
`RecordingClient`ı 600 saniye timeout koyduğu için aynı eksik orada yok.

### Context bütçesi uzun girdiler için doğrulanmıyor

İki LLM kolunda aynı `num_ctx` var, fakat prompt'un tamamı ve çıktı bütçesi için
tokenizer tabanlı ön kontrol bulunmuyor. Özellikle eski 100/50/20/10 batch
koşullarında sadece JSON/ID doğrulamasının geçmesi tam girdinin işlendiğini
kanıtlamaz. Bu incelemede yeni bir gerçek truncation vakası ölçülmedi.
30 kısa örnekli `run02`de girdi sayıları 185–199 ve 918–1214 token idi.

### LW karşılaştırması yalnızca ağırlıkların etkisini ölçmüyor

[literature_weighted_hisnet_dependency_classifier.py](../src/hisnet/literature_weighted_hisnet_dependency_classifier.py):
31–39'daki “adds only” / “clean ablation” açıklaması eksik. Kod ayrıca modifier
bileşenlerinin ve contrast marker'larının lexicon katkısını bastırıyor
(1850 ve 2003). Dolayısıyla sonuç farkı ağırlıklandırma **ve** bu ek kuralları
birlikte ölçüyor. Makale açıklaması buna uymalı veya bunlar ayrı ablation'larla
ayrılmalı. Katsayıları belirli bir yayına bağlayan bibliyografik kaynak da
incelenen ilgili kod/dokümanlarda bulunmadı; “literature-derived” iddiası bu
incelemede doğrulanmış değil.

### DP → LLM karşılaştırması parse ile ek talimatları birlikte değiştiriyor

[dependency_ollama_zero_shot_classifier.py](../src/llm/dependency_ollama_zero_shot_classifier.py):
474–490. DP kolu parse'ın yanında özne/nesne, modifier ve olumsuzluğu nasıl
yorumlayacağına dair ek talimatlar alıyor. İki yaklaşımın karşılaştırılması
geçerli olabilir; ancak farkı yalnızca parse verisine nedensel olarak atfetmek
için talimat farkını kontrol eden ek koşul gerekir.

### Dataset'te aynı metne çelişen etiketler kalıyor

Gerçek dosyalar mevcut `load_sentiment_dataset` ile okundu:

| Dataset | Yüklenen satır | Birden fazla etiketi olan farklı metin | Bu metinlere ait satır |
|---|---:|---:|---:|
| TRSAv1_fixed.csv | 125561 | **898** | **1884** |
| beyazperde_sentiment.csv | 2654 | **5** | **10** |

Yükleyici yalnızca aynı `(text, gold)` çiftlerini birleştiriyor
([dataset_loader.py](../src/evaluation/dataset_loader.py), 158); aynı metnin
farklı gold etiketleri korunuyor. Bu sayılar kaynak dosyaların hash'leriyle
birlikte kaydedildi. Hangi etiketin doğru olduğunu bu kontrol belirlemiyor.

Bu bir veri kalitesi ve deney protokolü sorunu. Karar test sonuçlarına bakılarak
verilmemeli; çelişki inceleme/çıkarma politikası ve varsa metin bazında ayrım
baştan tanımlanmalı. Bu inceleme train/test sızıntısı ölçümü değildir.

### Test altyapısı ve nihai runner

- Birçok `tests/test_*.py` dosyası import anında gerçek deney çalıştırıyor;
  genel unittest/pytest keşfi güvenli birim test keşfiyle aynı şey değil.
- Mevcut AST regresyon testi yalnızca `src` ve `tests` dizinlerini kapsıyor.
  Önceki analiz scripti parantez hatasının bu testte yakalanmamasının nedeni bu;
  yeni audit scripti `scripts` dizinini de kontrol ediyor.
- Genel evaluator hâlâ yalnızca H/H → DP çalıştırıyor. Yeni smoke runner altı
  kolu ortak manifestle çalıştırıyor fakat sabit geliştirme fixture'ına bağlı.
  Tam dataset/bağımsız final değerlendirme protokolü ayrıca hazırlanmalı.
- Eski 500 runner manifestinde model adı var; yeni smoke manifestindeki model
  digest'i, Stanza/lexicon dosya hash'leri gibi tam kaynak kimliği yok.

## Altı sistemin durumu

| Sistem | Doğrulanmış uygulama sorunu / kalan önemli konu |
|---|---|
| HisNet | Fiil kökü–lexicon uyuşmazlığı; bağlamsız sözcük puanları yöntemin sınırı. |
| H → DP | Ortak lookup + yanlış `değil` kapsamı + tekrar eden yüzeyin yanlış kullanımına hizalama. |
| DP → H | Ortak lookup + yanlış `değil` kapsamı + onarılmış/gürültülü tekrarlarda kök karışması. |
| LW H → DP | H → DP sorunlarını devralıyor; ayrıca ablation açıklaması gerçek ek bastırma kurallarını kapsamıyor. |
| LLM | Önceki 30 satırda teknik hata yok; ortak istemcide timeout/context ve runner süre/recovery açıkları var. |
| DP → LLM | Aynı LLM operasyonel açıkları; uzun parse girdilerinde context bütçesi ve prompt kontrol koşulu gerekli. |

Önerilen sıra: önce olumsuzluk kapsamı ve occurrence eşleme; ardından ortak
lexicon maddeleştirmesi ile ekli olumsuzluk politikasının birlikte sınanması;
sonra resume/süre/timeout ve deney protokolü. Kaynak değişiklikleri sonrasında
yeni manifestle yeni koşu alınmalı. Bu bulguların varlığı, altı sistemin temel
mimarisinin tamamen yanlış olduğu anlamına gelmez; mevcut uygulamadan henüz
hatasızlık veya bağımsız nihai performans iddiası çıkarılamaz.
