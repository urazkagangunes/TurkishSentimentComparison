# Altı sistem için tam makale koşusu

Giriş noktası `scripts/run_paper_suite.py`. HisNet, HisNet→DP, DP→HisNet,
LW, LLM ve DP→LLM aynı veri setinin tüm uygun yorumlarında ayrı ayrı çalışır.
Veri setleri ve sistemler sırayla yürütülür; eğitim veya eşik seçimi yapılmaz.
Ana ölçüt Macro-F1, üç sınıfın yakalama oranı ise sınıf bazlı recall'dır.

Onaylanan kapsam: genel/sınıf metrikleri, karışıklık matrisleri, eşleştirilmiş
kazanç–kayıp analizi, otomatik dilsel dilimler, teknik başarısızlıklar, süreler,
güven aralıkları, veri/kod/model kimlikleri ve kesintiden devam.
**4.B: insan etiketli olumsuzluk işaretleyicisi/kapsam doğruluğu kapsam dışıdır.**

## Başlangıç

Komutları depo kökünde PowerShell ile çalıştırın. Envanter model yüklemez,
sunucuya bağlanmaz ve çıktı dizini oluşturmaz:

```powershell
.venv\Scripts\python.exe -B -m scripts.run_paper_suite --inventory
```

Varsayılan etkin kaynaklar `datasets/Modified/sentence_sentiment/` içindeki
`trsav1`, `beyazperde`, `combined_dataset`, `combined_sentiment`, `humir`,
`reviews_combined`, `tr_polarity` ve `turkish_twitter` kimlikleridir.
Diğer bütün fiziksel dosyalar, neden etkin olmadıklarıyla envantere girer.
Sayısal etiket/yıldız eşlemesi, treebank dönüşümü ve etiketsiz dosya şeması
açıkça yapılandırılmalıdır. Aspect etiketleri doğrudan yorum gold'u sayılamaz.

16 Kasım 2026'da model çağırmadan yapılan tam dosya okumasında 95 kaynak,
8 etkin veri seti ve toplam 930.237 uygun kayıt doğrulandı. Bu toplam veri
setleri arasındaki örtüşmeleri içerir; bağımsız benzersiz yorum sayısı değildir.

| Veri seti | Ham kayıt | Uygun kayıt |
|---|---:|---:|
| beyazperde | 2.684 | 2.654 |
| combined_dataset | 489.644 | 487.240 |
| combined_sentiment | 6.001 | 5.948 |
| humir | 64.999 | 64.443 |
| reviews_combined | 235.144 | 232.591 |
| tr_polarity | 10.660 | 10.639 |
| trsav1 | 133.614 | 125.561 |
| turkish_twitter | 1.161 | 1.161 |

`requirements.txt` ana Python bağımlılıklarını içerir. Klasik sistemler için
`NlpToolkit-MorphologicalAnalysis` ve `NlpToolkit-MorphologicalDisambiguation`
ile bunların Dictionary/Corpus bağımlılıkları da gereklidir. Stanza Türkçe
model dosyaları ve Ollama sunucusunda seçilen model ayrıca hazır olmalıdır.
Runner model indirmez. İsteğe bağlı XLSX okuyucusu `openpyxl` gerektirir.
Paket sürümleri tam kilitli değildir; tarihsel ortamı yeniden üretmek için
ilgili koşunun `environment.json` kaydını esas alın.

Yeni koşuda daha önce kullanılmamış bir çıktı dizini seçin. Aşağıdaki adlar
örnektir; kullanmadan önce var olmadıklarını kontrol edin:

```powershell
# Veriyi ve manifestleri hazırla; model çıkarımı yapma.
.venv\Scripts\python.exe -B -u -m scripts.run_paper_suite --output results/paper_full_20261116_run01 --prepare-only

# Bütün etkin veri setleri × altı sistem: uzun süren canlı çıkarım.
.venv\Scripts\python.exe -B -u -m scripts.run_paper_suite --output results/paper_full_20261116_run01

# Yalnız belirli veri setleri, ayrı çıktı dizini.
.venv\Scripts\python.exe -B -u -m scripts.run_paper_suite --datasets trsav1 beyazperde --output results/paper_two_datasets_20261116_run01
```

`--prepare-only` isteğe bağlıdır. Kullanılırsa çıkarım komutunda diğer bütün
seçenekler aynı kalmalıdır. Bu mod bütün uygun veriyi okuyup SQLite'a yazar;
salt okunur durum sorgusundan daha fazla iş yapar.

Varsayılanlar: `qwen3:32b`, batch 3, context 8192, seed 42, temperature 0,1,
top_p 1, timeout 600 saniye, max_retries 1, keep_alive `30m` ve `think=False`.
Sınıflandırıcıların kendi kurucu varsayılanları yerine bu etkili ayarlar
kullanılır. `--systems` ile alt küme seçilebilir; altı sistemin makale
karşılaştırması için varsayılan altı sistemin tamamını kullanın.

## İlerleme ve kesintiden devam

Ekran çıktıları veri seti/model başlangıcını, kaçıncı örnekte bulunulduğunu,
başarı/hata/bekleyen sayılarını ve tamamlanmayı gösterir. Örnek:

```text
[DATASET BAŞLADI] 1/8 | beyazperde | ...
[MODEL BAŞLADI] beyazperde | hisnet | kayıtlı başarı=0/2654
[İLERLEME] beyazperde | hisnet | son örnek 25/2654 | başarılı=25 hata=0 bekleyen=2629 | %0.9
[KAYIT] beyazperde | hisnet %10 | SQLite kalıcı kayıt hazır
[MODEL BİTTİ] beyazperde | hisnet | başarı=2654/2654 hata=0
```

`--progress-every 1` her tamamlanan yorum/batch'te ilerleme gösterir; varsayılan
25'tir. Uzun model yükleme/istek/analiz sırasında 20 saniyede bir bekleme
mesajı gelir; `--heartbeat-seconds 0` bunu kapatır.

Her sistem için %10, %20, …, %100 dönüm noktaları ve veri setindeki toplam
tamamlanmış kararlar için aynı dönüm noktaları `checkpoints/` altında yazılır.
Veri seti yüzdesinin paydası `yorum sayısı × seçili sistem sayısı`dır.
Asıl kurtarma kaynağı `progress.sqlite3`: klasiklerde her yorum, LLM'de her
tamamlanan batch tek işlemde kalıcı kaydedilir. Yalnız %10 sınırlarında
kaydetmekten daha sık kayıt yapılır. JSON dönüm noktaları veritabanının
bağımsız yedeği değildir; bütün çıktı dizinini koruyun.

Kesinti sonrası **aynı komutu, aynı ayarları ve aynı çıktı dizinini** kullanın.
Başarılı satırlar atlanır; tamamlanmamış işler işlenir; başarısızlar her
çalıştırmada tek tek bir kurtarma turuna girer. Teknik hata `None/failed`
kalır, nötre dönüşmez. Kesilen aktif örnek/batch yeniden çalışabilir.
Tamamlanan sistem yeniden yüklenmez. İşletim sistemi süreç ölünce kilidi
serbest bırakır; `suite.lock` dosyasını elle silmek gerekmez.

Kod, ortam, veri veya etkili ayar değişirse eski manifest reddedilir. Yeni
çıktı dizini kullanın; eski checkpoint'i zorla açan seçenek yoktur.
Model yükleme hataları `runtime.csv` içinde görünür, satırlar bekleyen kalır.
Çıkış kodu 0 tamamlanma/hazırlık, 2 eksik koşu/hata, 130 kullanıcı kesintisidir.

```powershell
# Canlı koşu sürerken de salt okunur durum sorgusu.
.venv\Scripts\python.exe -B -m scripts.run_paper_suite --output results/paper_full_20261116_run01 --status

# Mevcut tahminlerden raporları yeniden üret; model çağırmaz.
.venv\Scripts\python.exe -B -u -m scripts.run_paper_suite --output results/paper_full_20261116_run01 --analyze-only --bootstrap 2000 --seed 42
```

Analiz türetilmiş raporları yeniden yazar, tahminleri değiştirmez. Analiz için
ayrıca tek yazıcı kilidi gerekir. Analizin kod kimliği, bootstrap tekrar sayısı
ve seed'i `paper_summary.json/analysis_identity` altında bulunur. Bu seçenekleri
yeniden analizde açıkça belirtin; CLI varsayılanları 2000/42'dir.

## Veri protokolü ve özel kaynaklar

Doğal sınıf dağılımı korunur; örnek azaltma veya dengeleme yapılmaz. Boş/eksik
kayıtlar ve aynı kırpılmış metin+etiket tekrarları sayılarak çıkarılır.
Çelişen etiketler korunur. Kaynak satır kimlikleri kaybolmaz. Yükleyici başlık
büyük/küçük harf farklarını, BOM ve tırnaklı çok satırlı CSV metnini destekler.
Bilinmeyen etiket koşuyu durdurur; sayısal etikete kendiliğinden anlam atanmaz.

`--config config.json` içindeki `datasets` listesi envanter kayıtlarını
dosya yolu üzerinden değiştirir. Yollar `--datasets-root` dizinine göredir:

```json
{
  "datasets": [
    {
      "id": "custom_test",
      "path": "custom/test.csv",
      "enabled": true,
      "adapter": "csv",
      "task": "sentiment",
      "text_column": "text",
      "label_column": "label",
      "separator": ";",
      "split": "test",
      "include_in_aggregate": false
    }
  ]
}
```

Bu örnek İngilizce negative/neutral/positive etiketleri içindir. Sayısal veya
başka bir etiket kümesi için doğrulanmış `label_map` ve onu açıklayan
`mapping_source` birlikte eklenmelidir. Örnek bir sayısal eşleme varsayılmamıştır.
`group_column` ürün/kullanıcı gibi bilinen kaynak gruplarını korur. Resmî
train/dev/test dosyalarını ayrı kayıtlar ve doğru `split` değerleriyle tanımlayın.

Desteklenen adaptörler: `csv`, başlıklı `xlsx` (`sheet` isteğe bağlı), satır
başına yorum için `lines`, kök etiketi ve yaprakları okunan parantezli ağaçlar
için `tree`. `task: "inference"` gold gerektirmez; yalnız tahmin dağılımı ve
çalışma istatistiği üretir. Her fiziksel dosyanın ayrı tanımlanması gerekir.
Yalnız yapılandırılan kaynağı çalıştırmak için `--datasets custom_test` ekleyin.

## Üretilen raporlar

Ana dizinde:

| Dosya | İçerik |
|---|---|
| `suite_metrics.csv` | Her veri seti × model için ortak başarılı evrende genel/sınıf metrikleri ve nötr tahmin oranı |
| `suite_aggregate.csv` | Aynı gold sınıflarına sahip, tamamlanmış ve dahil edilmesi seçilmiş veri setlerine eşit ağırlıklı ortalama |
| `suite_pairwise.csv` | Beş ana karşılaştırma; veri seti içi ve bütün koşu için ayrı Holm düzeltmeleri |
| `dataset_inventory.csv/json` | Etkin ve etkin olmayan bütün dosyalar ve gerekçeleri |
| `dataset_overlap.csv` | Tam/normalize metin örtüşmeleri, etiket kümesi ve train/dev–test örtüşmeleri |
| `suite_manifest.json`, `resources_*.json` | Veri, kod, paket, ayar ve model/kaynak kimlikleri |
| `environment.json`, `source_snapshot/` | Ortam, temel CPU bilgisi, işçi/işleme politikası ve kaynak kod kopyaları |

Her veri setinin alt dizininde:

| Dosya | İçerik |
|---|---|
| `metrics.csv`, `per_class.csv` | Accuracy, hata oranı, macro P/R/F1, weighted-F1, MCC; sınıf P/R/F1, support, doğru, TP/FP/FN/TN; metrics.csv içinde nötr tahmin oranı |
| `confusion.csv` | Gold satır, tahmin sütun; sayılar ve satır yüzdeleri |
| `pairwise.csv`, `paired_changes.csv` | Düzeltilen/bozulan/yanlış→yanlış kararlar, farklar, örnek kimlikleri |
| `slices.csv`, `pairwise_slices.csv` | Dilsel ve mekanizma dilimleri, sınıf dağılımları, ortak örneklerde DP kazanç/kayıpları |
| `failures.csv`, `failure_slices.csv` | Teknik hatalar; uzunluk ve gold sınıfa göre başarısızlık/bekleyen dağılımları |
| `failure_summary.csv` | Model ve hata türü bazında güncel başarısız yorumlar, geçmiş başarısız denemeler ve yükleme hataları |
| `retry_summary.csv` | Checkpoint tekrarları, kurtarma sonuçları ve bilinen LLM iç retry sayıları |
| `runtime.csv` | Yükleme, morfoloji, parse, sözlük sorgusu, kalan pipeline ve LLM istek süreleri; token/retry sayıları |
| `mechanisms.csv`, `diagnostics.csv` | Kapsam/işlem sayıları, eşleşmeyen kökler, sözlük/duygu kapsamı ve satır tanıları |
| `predictions_wide.csv`, `provenance.csv` | Metin/gold/altı tahmin, kaynak satırları, split/grup kimlikleri |
| `paper_report.md`, `paper_summary.json` | Okunabilir özet, bütün metrikler, güven aralıkları, nötr hata oranları, sabit sınıf referansları |
| `dataset_validation.json` | Ham/temiz/tekrar/eksik/çelişen kayıt sayımları |

Özet tanılar varsayılan olarak saklanır. `--keep-details` tam sınıflandırıcı
çıktısını da saklar; disk kullanımı büyür. Satır/batch tahminleri ile tanıları
aynı SQLite işleminde yazılır.

### Nötr tahmin oranı

`neutral_prediction_rate = neutral_prediction_count / prediction_count`.
Payda seçili rapor evrenindeki geçerli tahmin sayısıdır; başarısız/bekleyen
çıktılar dahil edilmez. Geçerli tahmin yoksa oran JSON'da `null`, CSV'de boş,
Markdown'da `N/A` olur. Gold gerektirmediği için iki sınıflı ve etiketsiz
verilerde de hesaplanır. Nötr recall'ın paydası ise gerçek nötr sayısıdır;
bu iki ölçüm farklıdır.

Üç alan `metrics.csv`, `slices.csv`, `suite_metrics.csv` ve JSON metriklerinde
bulunur. `successful` görünümü modelin geçerli çıktılarıyla, `common_success`
görünümü tüm seçili modellerin başarılı olduğu ortak kümede hesaplanır.
Markdown özetinde oran yüzde olarak gösterilir; CSV/JSON ölçeği 0–1'dir.

### Hata türü ve retry sayaçları

`failure_summary.csv` her sistemde timeout, parse_error, invalid_response,
context_budget, morphology_error, lexicon_error ve other için sıfır dahil
kayıt üretir; varsa diğer hata kodları da korunur. Alanların birimleri:

| Alan | Sayılan olay |
|---|---|
| `current_failed_rows` | Son durumu hâlâ başarısız olan yorum sayısı |
| `failed_attempts` | Geçmişte başarısız tamamlanan yorum/batch çağrısı sayısı |
| `failed_row_occurrences` | Bu başarısız çağrıların içerdiği satırların toplamı; tekrarları içerir |
| `initialization_failures` | Model yükleme hataları; yorum hatalarına eklenmez |

Örneğin üç yorumluk timeout batch'i bir başarısız deneme ve üç başarısız
satır işleme demektir. Yorumlar daha sonra kurtarılırsa güncel hata sayısı
azalır, geçmiş deneme sayıları korunur. Sınıflandırıcı yalnız son hata türünü
kaydettiğinden LLM isteğinin iç tekrarlarındaki ara hata türleri bilinmez.

`retry_summary.csv` içindeki `repeated_row_attempts` bir satırın ilk kayıtlı
denemesi dışındaki bütün işleme girişimlerini, `retried_unique_rows` bunlara
konu olan farklı yorumları sayar. `successful_retried_rows` bunlardan son
durumu başarılı olanlardır. Kesintiden sonra yeniden başlatılan işler de
bu sayımlara dahildir; kayıtlı deneme fiilen tamamlanmış çağrı demek değildir.

`recovery_*` alanları runner'ın başarısız yorumları tek tek yeniden denediği
kurtarma aşamasını başarılı/başarısız/kesilmiş/devam eden olarak ayırır.
`recovery_row_occurrences` kurtarma girişimlerindeki satır toplamıdır;
`recovered_rows` başarılı kurtarma çıktılarının toplamıdır.
`known_llm_request_attempts` kayıtlı payload'lardaki LLM istek sayısını,
`known_llm_internal_retries` her çağrı için `max(attempt - 1, 0)` toplamını verir.
Context/parse ön kontrolünde `attempt=0` ise LLM isteği sayılmaz.
`llm_request_count_unknown_attempts`, kesilmiş veya sayaç içermeyen LLM
çağrılarını gösterir; bu çağrılar için istek sayısı tahmin edilmez.
Klasik sistemlerin LLM sayaçları sıfırdır. Checkpoint tekrarlarıyla LLM iç
retry sayıları farklı düzeyleri ölçer; birbirine eklenmemelidir.

Bu iki özet `paper_summary.json` ve Markdown raporuna da eklenir. Bütün
sayaçlar mevcut SQLite kayıtlarından okunur; tahmin ve deneme kayıtları
değiştirilmez. Eski paper-suite checkpoint'leri `--analyze-only` ile yeni
alanları üretebilir; çıkarıma devam için mevcut manifest uyum kuralları geçerlidir.

## Ölçümün yorumu

- Ana karşılaştırma, **seçilen sistemlerin tamamının başarılı olduğu aynı
  örnekler** üzerindedir. Her sistemin kendi başarılı evreni ayrıca verilir.
  `correct_over_all`, eksik çıktıları doğru saymadan tüm gold örnekleri kullanır.
- İki sınıflı kaynakta nötr tahmini hatadır. Olmayan gold sınıfın P/R/F1'i
  N/A gösterilir; macro hesabının sınıfları sabittir ve rapora yazılır.
- Olumsuzluk, karşıtlık, karma sözcük ipuçları, emoji, uzunluk/cümle ve yüzey
  bozukluğu dilimleri otomatik vekillerdir. Yüzey bozukluğu yalnız harf/işaret
  tekrarları ve bozuk karakterleri işaretler; yazım hatalarının tamamını ölçmez.
  Bunlar gerçek olumsuzluk/kapsam tespit recall'ı değildir. Dilimler örtüşebilir.
- Modelin gerçekleştirdiği işlemlere göre dilimler sonradan oluşur. Kuralın
  nedensel katkısını kanıtlamak için ayrıca ablation gerekir. LW bastırma da
  yapar; DP→LLM parse yanında ek talimat kullanır.
- Güven aralıkları varsayılan 2000 tekrar, seed 42 ile %95 cluster percentile
  bootstrap'tır. Eşleştirilmiş farkta aynı gruplar örneklenir. Gruplar normalize
  metin eşitliği ve varsa kaynak gruplarının geçişli birleşimidir. Anlamsal
  yakın kopya garantisi yoktur. `--bootstrap 0` hızlı kontrolde aralıkları kapatır.
- Exact McNemar doğruluk değişimini sınar; F1 testi değildir. Örnek bağımsızlığı
  varsayımı nedeniyle gruplu veride ikincil yorumlanmalıdır. Dilim p-değerleri
  keşif amaçlıdır; ana karşılaştırmaların Holm düzeltmesine dahil edilmez.
- Sistemler bağımsız pipeline çalıştırır. Kalan pipeline süresi salt bileşim
  ölçümü değildir. Süreler tamamlanan denemeleri ve retry'ları içerir; kesilmiş
  denemelerin süreleri bilinmez, bilgisayarın uykusu süreleri etkileyebilir.
  Ollama donanımı sunucu tarafından belirlenir; yerel CPU kaydı GPU ölçümü değildir.
- Örtüşen/önceden incelenmiş kaynaklar bağımsız nihai test veya bağımsız tekrar
  sayılmaz. Veri seti ortalaması bu sınırlamayı ortadan kaldırmaz.

## Çevrimdışı doğrulama

```powershell
.venv\Scripts\python.exe -B -m unittest tests.test_paper_evaluation tests.test_resumable_evaluation -v
```

Bu testler sahte sınıflandırıcılarla altı sistemin kesinti/devam davranışını,
%10 dönüm noktalarını, süreç ölümü sonrası kayıtları, manifest uyumunu,
tek yazıcı kilidini, hata kurtarmayı, etiketsiz/iki sınıflı metrikleri,
sklearn ile hesap tutarlılığını ve rapor çıktısını kontrol eder. Gerçek model
çıkarımının veya tam korpus koşusunun başarıyla tamamlandığı anlamına gelmez.
