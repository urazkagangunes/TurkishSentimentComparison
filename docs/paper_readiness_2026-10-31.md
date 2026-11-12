# Paper için son koşuya hazırlık incelemesi — 31 Ekim 2026

**Karar: Uzun nihai koşuyu henüz başlatmayalım.** Kayıt/devam altyapısı testleri
geçiyor; iki kod başlığında tekrarlanabilir kusur var. Ayrıca bağlam sınırı ve
bağımsız değerlendirme verisi için deney protokolü kesinleştirilmeli.

Bu tur sınıflandırıcılar, sözlük, morfoloji, eşikler, ağırlıklar ve eski sonuçlar
değiştirilmedi. İnceleme scripti ve kanıt dosyaları eklendi. Gerçek Stanza,
HisNet/morfoloji veya LLM çıkarımı yapılmadı; uzun deney başlatılmadı.

## İncelenen kapsam ve geçen kontroller

- Altı sistemin kritik sınıflandırma akışları, ortak kapsam/hizalama kuralları,
  morfolojik kayıtların taşınması, LLM prompt/yanıt/hata akışı incelendi.
- Dataset yükleme, ortak örneklem, etiketler, skor/etiket eşikleri, metrikler,
  SQLite kayıtları, kurtarma, manifest ve model kimliği kontrolleri incelendi.
- `src/`, `scripts/`, `tests/` altında **83 Python dosyası** sözdizimi kontrolünden
  geçti. Bu, 83 dosyanın bütün davranışlarının doğrulandığı anlamına gelmez.
- Dört seçili çevrimdışı test modülündeki **51 test geçti**: mevcut regresyonlar
  19, kayıt/devam 17, açık olumsuzluk 12, kayıtlı sonuç analizi 3.
- Altı sistemde kesinti/devam, tamamlanmış satırı atlama, önceki oturumdan kalan
  başarısızı kurtarma, işlem rollback'i, süreç ölümü, tek yazıcı kilidi,
  veri/kod/model kimliği uyuşmazlığını reddetme ve metrik kontrolleri geçti.
- ID 148/447 düzeltmesi korundu. Resolver SHA-256 değeri düzeltme raporundakiyle
  aynı: `1d48e531a135410c446d6af2d049082696e8e46543a795edc4d478038d014453`.
- LLM'nin ID/etiket eşleme, başarısızlığı nötre çevirmeme, timeout ve yanıt
  doğrulamasında bu incelemede yeni bir somut kusur saptanmadı. Bu bölümdeki
  çalışma zamanı kontrolleri sahte istemci kullanır; canlı model testi değildir.

51 testin geçmesi aşağıdaki kusurları dışlamıyor: mevcut test paketi bu
karşı örneklerin tamamını kapsamıyor. Ayrı kontrollü tanı scripti bunları
güncel kaynak koduyla yeniden üretti.

## Önce kapatılmasını önerdiğim kod başlıkları

### 1. Yüksek: aynı cümlede ilgisiz yükleme olumsuzluk taşınıyor

Konum: [sentiment_scope_resolver.py](../src/dependency/sentiment_scope_resolver.py),
`_disambiguate_shared_explicit_negations`, özellikle 1498 ve 1592.

Kontrollü örnek: `Otel kötü, merkezi değil.`

Doğru ağaçta `değil`, `merkezi`ye `cop` bağıyla bağlı. Yalnız `kötü` duygu
adayı olduğunda kapsam düzeltmesi önceki en yakın duygu adayını seçiyor ve
olumsuzluğu `kötü`ye taşıyor. Verilen negatif 1,0 puan pozitif 1,0'a dönüşüyor.
Beklenen olumsuzluk listesi `[]`; gözlenen `[4]`.

Burada doğru ağaç dışarıdan verildiği için kusur yalnız parser sınırlaması
değil. ID 148 düzeltmesi sonraki yardımcı fiilli yükleme taşınmayı önlüyor;
bu örnekteki önceki ilgisiz yükleme yeniden atama farklı bir yol.

**Düzeltme yönü:** doğrudan dependency bağını önceliklendirmek; asıl yüklem
duygu sözlüğünde yok diye başka bir yükleme olumsuzluk vermemek; paylaşımı
gerçek koordinasyon/yüklem bağlantısıyla sınırlamak. `İyi ve güzel değil`,
karşıtlık, devrik yapı ve ID 148/447 testleri korunmalı.

### 2. Orta: tekrar eden sözcüklerde kullanım/kök hizalaması kayboluyor

Bu başlığın iki ayrı yolu var; ikisi de doğru morfoloji kayıtları sağlanarak
yeniden üretildi.

| Yol | Beklenen | Gözlenen |
| --- | --- | --- |
| H→DP/LW: yalnız ikinci `yazar` kullanımı duygu adayı | token ID 2 | token ID 1 |
| DP→H: `yazar🙂`, ardından `yazar`; kökler `yazar`, `yaz` | `[yazar, yaz]` | `[yazar, yazar]` |

İlkinde `resolve` içindeki exact-surface eşleme yalnız `sentiment_words`
altkümesini dolaşıyor. Önceki aynı-yazımlı duygu taşımayan kullanımın konumu
korunmadığı için sonraki aday ilk token'a atanıyor. Ayrı cümlelerdeyse duygu
kapsamı yanlış cümleye de yerleşebiliyor. Konum: aynı dosya, 3183.

İkincisinde ham yüzey occurrence sayacı `yazar🙂` ile `yazar`ı farklı görüyor;
onarım yolu ise normalize yüzey için ilk morfoloji kaydını seçiyor.
Konumlar: `_find_morphological_root` 2069/2082,
`_find_morphology_record_for_surface` 2164 ve onarım yolu 2206.

**Düzeltme yönü:** duygu adaylarını filtrelemeden önce tüm morfoloji/token
kullanımlarını sıralı ve tekil eşlemek; adaylara kullanım kimliğini taşımak;
onarılmış token'ların da aynı eşlemeyi tüketmesini sağlamak.

Bu kontrollü testler gerçek HisNet'te `yaz` kökünün pozitif olduğunu söylemez;
doğru verilen bir kaydın yanlış kullanıma bağlanabildiğini gösterir. Gerçek
veride bu kusurların sıklığı bu incelemede ölçülmedi. Bunlar, kullanıcının
korumayı seçtiği morfolojik çözümleyici sınırlamasından ayrıdır.

## Nihai koşudan önce kesinleşmesi gereken deney kararları

### Bağlam sınırı: DP→LLM'nin 8192 ayarı 500 örneğin tamamını kapsamıyor

Yeni model çalıştırmadan, aynı 500 yorumun kayıtlı parse'ları üretim prompt
oluşturucusundan geçirildi. Mevcut ön kontrolün tam hesabı kullanıldı:
UTF-8 bayt uzunluğu + 1024 şablon payı + `128 + 64 × batch_size` çıktı payı.
**Aşağıdaki sayılar gerçek tokenizer token sayıları değildir.**

| Kontrol | LLM | DP→LLM |
| --- | --- | --- |
| 8192'de reddedilecek üçlü batch | 0 / 167 | 51 / 167 |
| Bu batch'lerdeki yorum | 0 | 153 |
| Tekli kurtarmada bile reddedilecek yorum | 0 / 500 | 8 / 500 |
| En büyük tekli ön kontrol bütçesi | 3097 | 28288 |

DP→LLM'de tekli olarak da reddedilen ID'ler:
`121, 130, 143, 181, 223, 343, 398, 402`.
16384 sınırında ID 398 hâlâ reddedilir; 32768 sınırı bu **örneklemdeki** tekli
promptların ön kontrolünü geçirir. Bu, tüm dataset için yeterlilik veya
donanımda sorunsuz çalışma doğrulaması değildir.

Kontrol girdiyi sessizce kırpmıyor ve hata yerine nötr üretmiyor; bu davranış
doğru. Ancak aynı ayarla tekrar başlatmak bu sekiz kalıcı bütçe reddini çözmez.
Yeni bir koşuda iki LLM kolunun ortak context ayarı ve uzun-girdi politikası
önceden kararlaştırılmalı. Alternatif bir ortak uygunluk altkümesi kullanılacaksa
kural sonuçlara bakılmadan sabitlenmeli ve altı sisteme aynı şekilde uygulanmalı.

Mevcut `summary.json`, metrikleri `accuracy_on_successful` ve
`macro_f1_on_successful` adlarıyla doğru biçimde başarılı satırlarda hesaplıyor.
İki sistemin farklı başarılı altkümelerini doğrudan eşleştirilmiş karşılaştırma
gibi sunmamak gerekir. Ana karşılaştırma evreni, coverage ve başarısızlık
raporlaması testten önce belirlenmeli.

### Veri geçmişi: yalnız 500 örneği dışarıda bırakmak bağımsızlığı kanıtlamaz

Geçmiş tahmin dosyalarının metinleri güncel, temizlenmiş dataset ile karşılaştırıldı.
Eşleme yalnız baştaki/sondaki boşlukları temizlenen tam metne dayanır; yalnız
dosya adına veya satır sayısına dayanmaz.

| Veri | Güncel satır | Geçmiş tahminlerde metni bulunan satır |
| --- | --- | --- |
| TRSAv1 | 125561 | 125561 |
| Beyazperde | 2654 | 2654 |

Kanıt dosyaları arasında TRSAv1'in 500, 1000, 6000 ve tam koşuları ile
Beyazperde'nin 300 ve tam koşuları bulunuyor. Tam dosyalar sırasıyla
`results/trsav1_full_conservative_negation/trsav1_full_threshold005_conservative_negation_predictions.csv`
ve `results/beyazperde/beyazperde_full_predictions.csv`.

Bu, her satırın bir insan tarafından incelendiğini veya LLM eğitim verisinde
bulunduğunu göstermez. Ancak mevcut 500'ü çıkarmak, kalanın hiç kullanılmamış
bağımsız final seti olduğunu göstermeye yetmez. Bağımsız doğrulama iddiası için
deney geçmişi netleştirilmeli ve gerçekten ayrılmış veri kullanılmalı. Mevcut
veride tekrar koşu yapılırsa geliştirme sonrası aynı benchmark üzerinde
değerlendirme olduğu açıkça raporlanabilir.

Yükleyici aynı metin+etiket tekrarlarını kaldırıyor, çelişen etiketleri koruyor.
TRSAv1'de 898 metin/1884 satır, Beyazperde'de 5 metin/10 satır bu grupta.
Etiketler sonuçlara bakılarak değiştirilmedi. Yeni ayrım yapılırsa aynı metnin
farklı etiketli kopyaları da aynı grupta tutulmalı.

## Korunabilecek bulgular ve raporlama sınırları

- HisNet fiil kökü/madde biçimi uyuşmazlığı, sözlük kapsamı ve puanları,
  morfolojik çözümleyici tercihleri, ID 98'deki parser token bölünmesi önceki
  karar doğrultusunda yöntem sınırlaması olarak kalabilir.
- Token'ın kendi ekli olumsuzluğunu çevirmeyen politika açıkça yazılmalı;
  genel Türkçe olumsuzluk çözümü iddia edilmemeli.
- LW, katsayıların yanında modifier/contrast bileşenlerinin duygu katkısını
  bastıran kuralları da içeriyor. Yalnız katsayı ablation'ı olarak sunulmamalı.
- DP→LLM ek parse ile birlikte ek yorumlama talimatları da alıyor. Karşılaştırma
  bu birleşik istemin etkisidir; yalnız parse bilgisinin etkisi değildir.
- Morfoloji çalışma süresi, cache, yükleme ve donanım farkları kaydedilmeli;
  mevcut sürelerden kontrollü hız üstünlüğü çıkarılmamalı.
- Nihai koşuda yeni `scripts.run_six_systems` kullanılmalı. Eski smoke recovery
  ve eski 500'lük runner'ın süre sınırlamaları bu yeni akışa taşınmamalı.

## Önerilen çıkış ölçütü

1. İlgisiz yükleme olumsuzluk taşıma ve iki kullanım-hizalama yolunu düzeltip
   karşı örneklerini regresyon paketine eklemek.
2. Kayıtlı 500 parse üzerinde önce/sonra etkiyi kontrol etmek; ID 148/447 ve
   mevcut çalışan kapsamları korumak. Bunun için LLM çalıştırmak gerekmez.
3. Test verisi, ortak kapsama/uzun-girdi politikası, model/context/batch/seed,
   metrikler ve başarısızlık raporlamasını sabitlemek.
4. Kaynak kodu, paket sürümleri, model/kaynak hash'leri ve örnek listesini
   sabitleyip yeni çıktı dizininde nihai koşuya geçmek.

Bu inceleme, bir önceki ID 148 düzeltmesinin başarılı olduğunu değiştirmiyor;
repo genelindeki nihai koşu değerlendirmesinin kapsamı daha geniş.

## Kanıtlar ve yeniden üretme

- [Kontrollü hata tekrarları ve dataset sayıları](../results/paper_readiness_20261031/followup_evidence.json)
- [51 testin çıktısı](../results/paper_readiness_20261031/offline_tests.log)
- [Bağlam ve veri geçmişi kanıtları](../results/paper_readiness_20261031/readiness_evidence.json)
- [Kurulu paket sürümleri ve kaynak kimliği](../results/paper_readiness_20261031/environment.json)
- [ID 148 düzeltmesinin önceki ayrıntılı raporu](../results/id148_scope_check_20261031/report.md)

```powershell
.venv\Scripts\python.exe -m scripts.audit_six_model_followup --datasets --output results/paper_readiness_20261031/followup_evidence.json
.venv\Scripts\python.exe -m scripts.audit_paper_readiness
```

Genel `unittest discover` veya tüm `pytest` dizini bu incelemede çalıştırılmadı;
bazı eski deney dosyaları import anında uzun model koşularını başlatıyor.
