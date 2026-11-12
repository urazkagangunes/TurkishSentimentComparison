# Altı model için kod incelemesi — 30 Ekim 2026

İncelenen ana kapsam: `src/` altındaki altı sınıflandırıcı, ortak morfoloji ve
dependency kapsam çözümlemesi, dataset yükleyici, değerlendirme/checkpoint akışı,
16 cümlelik altı-model runner'ı ve 500 örneklik deney scriptlerinin ilgili bölümleri.
Bu rapor, gerçek veri üzerinde altı modelin yeniden değerlendirilmesi değildir.

## Doğrulanmış ve düzeltilmiş hatalar

| Öncelik | Hata | Etkilenen bölüm | Düzeltme |
|---|---|---|---|
| Yüksek | `değil` düzeltmesi global token listesindeki önceki cümlenin duygu adayını seçebiliyordu. | H→DP, DP→H, dolaylı olarak LW | Konumsal kapsam düzeltmesi aynı `sentence_index` ile sınırlandı. |
| Yüksek | Kök/lemma eşleştirmesi daha sonraki adayın tam yüzey eşleşmesini tüketebiliyordu. | H→DP, LW | Bütün adayların tam yüzey eşleşmeleri önce ayrılıyor; fallback kalan token'larda çalışıyor. |
| Yüksek | Aynı yazılan sözcüğün her kullanımına ilk morfolojik kaydın kökü atanıyordu. | DP→H; hedef kökü tanıları | Normal token'larda yüzeyin belge içindeki kaçıncı kullanımı olduğu hesaba katılıyor. |
| Orta | Morfolojik çözümleyicinin döndürdüğü önbellek nesnesi disambiguator tarafından yerinde daraltılıyordu. | Dört klasik model | Normal analiz sonuçları da tüketiciye verilmeden önce derin kopyalanıyor. Robust analizde bu koruma zaten vardı. |
| Orta | POS, çözümlemenin tüm etiketlerinde ilk bulunan kategoriye göre seçiliyordu. | DP→H fragment onarımı | Son `^DB+` türetim grubunun POS'u kullanılıyor. Örneğin `sev+VERB+POS^DB+NOUN+INF2+A3SG` artık NOUN. |
| Orta | `0.20 - 0.15` gibi işlemler matematiksel olarak 0,05 sınırında olsa da float hatasıyla positive olabiliyordu. | Dört klasik model | ±0,05 sınırında yalnızca 1e-12 sayısal tolerans eklendi. Deneysel eşik hâlâ 0,05. |
| Yüksek | LLM yanıtı geçerli JSON olsa bile liste/null/sayı biçiminde gelince deney çöküyordu. ID olarak liste/dict de TypeError üretiyor, `true` ve `1.0` ise ID 1 sayılabiliyordu. | LLM, DP→LLM | JSON nesne/tür kontrolü, gerçek integer ve 1..N ID doğrulaması eklendi. Hatalar retry/failed akışına giriyor. |
| Orta | Başarısız denemeden sonra başarı gelince önceki denemenin süresi ve bozuk yanıtın token maliyeti kayboluyordu. | LLM, DP→LLM | Batch ve toplam istatistikler tüm denemelerin süresini ve dönen token sayılarını kapsıyor. |
| Yüksek | Stanza parse hatası LLM'nin güvenli hata/recovery sözleşmesinin dışında kalıyordu. | DP→LLM | Beklenen parser çalışma hataları `prediction=None`, `status=failed` olarak dönüyor; bireysel recovery tekrar deneyebiliyor. |
| Yüksek | Kod/bağımlılık değiştikten sonra eski checkpoint veya farklı model ayarlarından kalmış CSV'ler yeniden kullanılabiliyordu. | Evaluator, dört-sistem 500 runner'ı | Checkpoint v2 kaynak+paket kimliği doğruluyor. CSV kullanan runner ayrıca sample, runner ve LLM ayarlarını manifest ile kontrol ediyor. |

Ek olarak sıfır/negatif batch size ve negatif retry sayısı başlangıçta reddediliyor.
Eski `test_negation_scope_rules.py` ilk vakası mevcut muhafazakâr politika ile
çelişiyordu; beklenti mevcut politikaya göre düzeltildi. Bu değişiklik, o
politikanın dilbilimsel olarak doğru olduğu anlamına gelmez.

Morfoloji önbelleği testi, nesnenin bozulabildiğini doğrular; bütün eski model
tahminlerinin değiştiğini veya mutlaka örnek sırasına bağımlı olduğunu kanıtlamaz.
Token hizalama düzeltmeleri de bütün tokenizer hatalarını çözmez. Özellikle
parçalanmış/emoji ile birleşmiş tekrarlar ayrıca gerçek parser çıktısıyla incelenmeli.

## Yöntem kararı veya gerçek model kontrolü gerektiren noktalar

### 1. Ekli fiil olumsuzluğu özellikle kapatılmış

`_find_negations`, duygu token'ının kendi `Polarity=Neg` özelliğini bilerek
skor çevirmek için kullanmıyor. HisNet kök üzerinden puanladığından bu seçim
`sevmedim`, `beğenmedim` gibi olumlu köklü olumsuz fiilleri kaçırabilir.
`değil` ve desteklenen light-verb ilişkileri ise korunuyor.

Bu davranış açıkça kodlanmış deneysel politika olduğu için otomatik olarak
tersine çevrilmedi. Küçük testte olumlu/olumsuz kök, ekli olumsuzluk, light verb,
çift olumsuzluk ve olumsuzluğun cümle içindeki kapsamı ayrı incelenmeli. Politika
seçimi nihai test kümesindeki accuracy'ye bakarak yapılmamalı.

### 2. LW kolunun farkı yalnızca katsayı çarpımı değil

LW kolu modifier parçalarının ve karşıtlık bağlaçlarının lexicon skorlarını da
bastırıyor. Dolayısıyla paper'da H→DP ile farkını yalnızca “aynı adaylara farklı
katsayılar uygulamak” diye tanımlamak eksik olur. Mevcut uygulama ağırlıklandırma
ve ek bileşen bastırma kurallarını birlikte içeriyor.

Katsayılar kodda mevcut, ancak incelenen dosyalarda bu tabloyu belirli bir
yayına bağlayan DOI/bibliyografik kaynak yok. Kaynak doğrulanmadı; katsayılar
değiştirilmedi. Ayrıca docstring/test açıklamalarında geçen `çok daha` ve
`daha da` gibi ifadeler katsayı tablosunda bağımsız anahtar olarak bulunmuyor.
Bunlara kaynak olmadan yeni ağırlık eklenmemeli.

### 3. Altı-model 16 cümlelik set benchmark değil

`test_six_systems_16_detailed.py` pozitif ve negatif etiketler içeriyor, neutral
örnek içermiyor. Dosyanın kendi açıklaması da bunun elle hazırlanmış sanity seti
olduğunu söylüyor. Üç sınıflı performans için yeterli değil. Bazı karma yorumların
tek doğru genel duygu etiketi ayrıca tartışmalı olabilir.

### 4. LLM context sınırı henüz doğrulanmış değil

DP→LLM prompt'u her token için parse bilgisi içerdiğinden normal LLM prompt'undan
çok daha uzun. Aynı batch sayısı ve `num_ctx`, iki kola aynı tam içeriğin ulaştığını
tek başına kanıtlamaz. Dört-sistem scriptinde 100/50/20/10 batch denemeleri de var.
Kodda gerçek tokenizer ile prompt+çıktı bütçesi doğrulaması bulunmuyor.

Bu incelemede truncation gerçekleştiği ölçülmedi. Küçük canlı koşuda modelin gerçek
tokenizer/context davranışı incelenmeli; başarılı ve doğru biçimli JSON tek başına
“girdinin tamamı işlendi” kanıtı sayılmamalı.

### 5. Paper için ortak altı-model runner ve tam manifest gerekiyor

Genel `SentimentEvaluator` yalnızca H ve H→DP çalıştırıyor. Diğer deneyler ayrı
scriptlerde; 16 cümlelik script altısını birlikte çağırıyor. Nihai koşu öncesinde
altı kolu aynı sabit örnek ID'leriyle çalıştıran ortak runner hazırlanmalı.

Eklenen kod/paket kimliği önemli bir kontrol, fakat Stanza ağırlık dosyalarının,
HisNet kaynaklarının ve Ollama model digest'inin yerine geçmez. Aynı model etiketi
altındaki ağırlık değişimi henüz otomatik yakalanmıyor. Nihai manifest bunları,
dataset hash'ini, seed/batch/context/prompt ayarlarını ve çalışma ortamını kaydetmeli.

Eski checkpoint v1 otomatik taşınmaz. Eski CSV'ler manifest olmadan kullanılmaz.
Dosyalar silinmedi. Dört-sistem scripti için yeni dizin `SENTIMENT_OUTPUT_DIR` ile
seçilebilir. Diğer runner'ların eski sonuçlarını yeni kodun sonuçlarıyla birleştirmemek gerekir.

### 6. Test scriptleri import sırasında deney başlatabiliyor

Birçok `tests/test_*.py` dosyası gerçek modelleri yükleyen ve sonuç yazan üst seviye
kod içeriyor. Genel test keşfi bu dosyaları import ederek büyük iş başlatabilir.
Bu nedenle bu incelemede yalnızca yeni offline test paketi seçilerek çalıştırıldı.
Sonraki düzenlemede deney runner'ları ile birim testler ayrılmalı veya runner'lara
`if __name__ == '__main__'` girişi eklenmeli.

### 7. Geliştirme örnekleri ve nihai değerlendirme ayrılmalı

Mevcut 500 örneklik setler hata teşhisi için de kullanılıyor. Bunlar üzerinde kural
değişikliği yaptıktan sonra aynı seti bağımsız nihai test gibi sunmak uygun olmaz.
Bu rapor veri sızıntısı ölçümü yapmadı. Paper protokolünde geliştirme/teşhis örnekleri,
bağımsız değerlendirme örnekleri ve yinelenen/çelişkili etiketli metin politikası
açıkça kaydedilmeli.

## Yapılan doğrulama

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p test_audit_regressions.py -v
```

- İlk düzeltme öncesi paket: 12 test metodu; alt vakalar dahil 12 failure ve 9 error.
  Bu sayılar ayrı bug sayısı değildir; aynı testte birden çok bozuk yanıt deneniyor.
- Son paket: 19 test metodu, tamamı geçti.
- Paket, mevcut eşik ve olumsuzluk kurallarının küçük scriptlerini de çalıştırıyor.
- `src/` ve `tests/` Python dosyalarının AST/sözdizimi kontrolü geçti.
- `git diff --check` geçti.
- Stanza ağırlıkları yüklenmedi, Ollama sunucusuna istek atılmadı, dataset benchmark'ı
  veya paper koşusu başlatılmadı. Önceki sonuç dosyaları değiştirilmedi.
- Mevcut Python ve bağımlılık importları izinli çalıştırmada başarılı. İlk sandbox
  erişim hatası ortamın bozuk olduğu anlamına gelmiyordu.

Bu testlerin geçmesi altı modelin tüm gerçek veride hatasız olduğunu veya accuracy'nin
arttığını göstermez. Ölçülen şey, belirtilen uygulama hatalarının küçük kontrollü
girdilerde giderildiğidir.

## Sonraki küçük test aşaması

1. Sabit 20–30 cümlelik davranış seti: üç sınıf, ekli olumsuzluk, `değil`, light verb,
   karşıtlık, çoklu cümle, yinelenen yüzey, Türkçe I/İ ve tokenizer parçalanması.
2. Önce dört klasik kolun token/kök/aday/kapsam/skor çıktıları incelenmeli; H→DP'nin
   başlangıç skorlarının H ile aynı kaldığı ve LW'nin aldığı temel sonuç nesnesini
   yerinde değiştirmediği doğrulanmalı.
3. Aynı set iki LLM kolunda aynı model/seed/ayarlarla çalıştırılmalı. Parse hataları,
   bozuk yanıtlar, batch recovery, context kullanımı ve süre/token kayıtları kontrol edilmeli.
4. Üç sınıftan örnekler içeren küçük gerçek veri örneklemiyle ortak değerlendirme
   akışı doğrulanmalı. Bu örneklem geliştirme kaydına alınmalı.
5. Yöntem kararları ve deney manifesti sabitlendikten sonra bağımsız paper koşusu alınmalı.
