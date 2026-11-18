# TurkishSentimentComparison — proje ve çalışma rehberi

Bu belge depo kökünden itibaren geçerlidir. Mimari, veri sayımları ve sonuç
referansları 13 Kasım 2026 tarihli salt okunur incelemede doğrulandı. Ortam ve
dosya içerikleri değişebilir; yeni çalışmalarda ilgili kaynak kodunu ve deney
manifestini yeniden kontrol et. Bütün yollar depo köküne göredir.

## Amaç ve çalışma ilkeleri

Proje, Türkçe yorum düzeyinde üç sınıflı duygu analizinde sözlük, bağımlılık
çözümlemesi ve LLM yaklaşımlarını karşılaştırır. Etiketler `negative`, `neutral`
ve `positive` değerleridir. Mevcut ana akış eğitim değil çıkarım ve değerlendirmedir.

- Değişiklikten önce ilgili modülü, çağrı akışını, testleri ve kayıtlı sonuçları oku.
- Kodla veya kayıtlı veriyle doğrulanan bulguları hipotezlerden ayır. Tarihsel
  raporlardaki hataların güncel kodda hâlâ bulunduğunu varsayma.
- Kullanıcının yetkilendirdiği kapsamda çalış. Dokümantasyon veya inceleme görevi
  sırasında kendiliğinden uzun değerlendirme, eğitim ya da canlı LLM çağrısı başlatma.
- Gold etiketleri veya örnekleri model sonuçlarını iyileştirmek için değiştirme.
  Etiket incelemesini ayrı, kayıtlı bir veri çalışması olarak yürüt.
- Eski tahminleri ve dondurulmuş deneyleri koru. Kod, kaynak, veri veya ayar değişirse
  yeni bir çıktı dizini kullan; manifest kontrolünü atlayarak eski checkpoint'i açma.
- Aynı örnekler üzerinde tek bileşenli karşılaştırmaları tercih et. Farklı örneklem
  veya kod sürümleri arasındaki farkı doğrudan yöntem iyileşmesi diye sunma.
- Bir rapordaki test başarısı geçmiş koşuya aittir; yeni çalışmada test çalıştırılmadıysa
  yeniden geçmiş gibi yazma. Yapılan kontrolleri ve sınırlarını açıkça belirt.
- Açıklamaları ve çalışma raporlarını, kullanıcı başka dil istemedikçe Türkçe yaz.

## Depo haritası

| Yol | Görev |
|---|---|
| `main.py` | `main()`: örnek bir otel yorumunu Stanza ile ayrıştıran demo; ana deney giriş noktası değil. |
| `src/common/morphology.py` | `MorphologicalRootExtractor`: normalizasyon, morfoloji, kök seçimi, robust analiz önbelleği. |
| `src/hisnet/` | Dört klasik sınıflandırıcı; ayrıntıları aşağıda. |
| `src/dependency/stanza_parser.py` | `TurkishDependencyParser`: Stanza kurulumu ve `parse()`. |
| `src/dependency/sentiment_scope_resolver.py` | `SentimentScopeResolver`: hizalama, token onarımı, hedef/olumsuzluk/niteleyici/karşıtlık kapsamları. |
| `src/llm/` | Ham metin ve bağımlılık çıktısı ile Ollama sınıflandırması; promptlar bu modüllerin içinde. |
| `src/evaluation/dataset_loader.py` | `load_sentiment_dataset()`: CSV, etiket doğrulama, temizlik, tekrar kaldırma, örnekleme. |
| `src/evaluation/sentiment_evaluator.py` | `SentimentEvaluator`: eski iki sistemli H–H→DP değerlendirmesi, JSON checkpoint ve metrikler. |
| `src/evaluation/resumable_evaluator.py` | `make_classifier()`, `evaluate()`: altı sistem için ortak yürütme ve kurtarma. |
| `src/evaluation/checkpoint_store.py` | `CheckpointStore`, `summarize_connection()`, `read_status()`: SQLite, kilit, durum ve sonuçlar. |
| `src/evaluation/run_identity.py` | `implementation_fingerprint()`, `ensure_run_manifest()`: kaynak/ortam kimliği ve uyum kontrolü. |
| `scripts/` | Deney çalıştırıcıları, sonuç analizi, eşik karşılaştırması ve denetimler. |
| `tests/` | Birim/regresyon testleri, fixture'lar ve gerçek model çalıştıran eski deney betikleri. |
| `tests/data_preprocessing/` | Eski veri dönüştürme betikleri; çoğunda kullanıcıya özgü mutlak yollar var. |
| `datasets/` | Cümle/yorum, puanlı yorum, aspect, treebank ve diğer metin kaynakları. |
| `results/` | Tahminler, metrikler, SQLite checkpoint'leri, manifestler, kaynak anlık görüntüleri ve raporlar. |
| `docs/` | İncelemeler, deney kullanımı ve araştırma önerileri. |
| `.audit_cache/` | Aday resolver kopyası; üretim kodunun yerine kullanma. |
| `.venv/`, `.idea/`, `.codex/` | Yerel Python ve geliştirme aracı yapılandırmaları. |

`src/` altındaki `__init__.py` dosyaları boş paket işaretleyicileridir.
README'nin yapı bölümündeki `data/` ve `prompts/` adları güncel dizin yapısıyla
uyuşmuyor: gerçek veri yolu `datasets/`, promptların yeri `src/llm/`.

## Altı sistem ve mimarileri

Sistem kimlikleri `src/evaluation/checkpoint_store.py::SYSTEMS` ve
`src/evaluation/resumable_evaluator.py::make_classifier()` içinde tanımlıdır.

| Kimlik | Sınıf / dosya | Akış |
|---|---|---|
| `hisnet` | `HisNetClassifier`, `src/hisnet/hisnet_classifier.py` | Metin → morfolojik kökler → HisNet sorgusu → puan toplamı → eşik. |
| `hisnet_dp` | `HisNetDependencyClassifier`, `src/hisnet/hisnet_dependency_classifier.py` | Aynı HisNet adayları → dependency hizalama/kapsam → hedef bastırma ve olumsuzluk → eşik. |
| `dp_hisnet` | `DependencyHisNetClassifier`, `src/hisnet/dependency_hisnet_classifier.py` | Parse → sözdizimsel aday seçimi → aday köklerinde HisNet → kapsam bileşimi → eşik. |
| `lw_hisnet_dp` | `LiteratureWeightedHisNetDependencyClassifier`, `src/hisnet/literature_weighted_hisnet_dependency_classifier.py` | H→DP → bileşen bastırma → niteleyici ve karşıtlık ağırlıkları → eşik. |
| `llm` | `OllamaZeroShotSentimentClassifier`, `src/llm/ollama_zero_shot_classifier.py` | Ham metin → zero-shot prompt → Ollama → doğrulanan JSON etiketi. |
| `dp_llm` | `DependencyOllamaZeroShotSentimentClassifier`, `src/llm/dependency_ollama_zero_shot_classifier.py` | Ham metin + Stanza çıktısı + sözdizimini yorumlama talimatı → aynı LLM. |

### Morfoloji ve sözlük

- `MorphologicalRootExtractor._prepare_sentence()` Türkçeye uygun küçük harf
  dönüşümü yapar; Unicode harf/rakamları ve iç apostrofları korur. Noktalama ve
  emoji işaretleri bu morfoloji girdisine alınmaz. Stanza ve LLM ham metni alır.
- `analyze_sentence()` önce `FsmMorphologicalAnalyzer`, gerektiğinde izole robust
  çözümleyici ve ardından `LongestRootFirstDisambiguation` kullanır.
- Robust analiz adayları kopyalanarak önbelleğe alınır; bağlama bağlı nihai kökler
  önbelleğe alınmaz. Çözümleme bulunamazsa yüzey biçimi ve `UNKNOWN` kaydı kullanılır.
- `HisNetClassifier._lookup_root()` kökün tam yazımını `SentiLiteralNet` içinde
  arar. Sözlük sabittir; bulunan ve bulunamayan sorgular önbelleğe alınır.
- Bağlamsal sözcük anlamı ayrımı, öğrenilmiş sentiment katmanı ve genel çok sözcüklü
  sözlük eşleme hattı mevcut değildir.

### Bağımlılık çözümlemesi ve kapsam

- `TurkishDependencyParser.__init__()` işlemcileri:
  `tokenize,mwt,pos,lemma,depparse`; `lang="tr"`, varsayılan `use_gpu=False`,
  `download_method=None`. Gerekli model kaynakları önceden hazır olmalıdır.
- `parse()` cümleler içinde `id`, `text`, `lemma`, `upos`, `head`, `deprel`, `feats`
  alanlarını döndürür. Kayıtlı deney manifestlerinde Türkçe `imst_charlm` parser var;
  sarmalayıcı kodu paket adını açıkça sabitlemez.
- `HisNetDependencyClassifier._flatten_dependency_sentences()` cümleler arasında
  token/head kimliklerini benzersizleştirir; kök head değeri 0 kalır.
- `_validate_initial_scores()` bağımlılık işlemlerinden önce HisNet adaylarının
  pozitif/negatif toplamlarının korunduğunu doğrular.
- `_apply_dependency_scope()` hedef sayılan katkıyı sıfırlar; tek sayıda kapsam
  olumsuzluğu varsa pozitif/negatif puanları değiştirir. Temel H→DP'de niteleyici
  ve karşıtlık yalnız tanısal kayıttır.
- `SentimentScopeResolver.resolve()` HisNet adaylarını hizalar;
  `resolve_dependency_candidates()` DP→H için sözlük sorgusundan önce aday seçer.
  `_resolve_sentiment_roles()` hedef/duygu rolünü ayırır.
- `_find_negations()` token'ın kendi `Polarity=Neg` özelliğini tek başına puan
  çevirme nedeni saymaz. Açık `değil` ve desteklenen bağımlılık ilişkilerindeki
  yardımcı fiil olumsuzlukları ayrı yollardan ele alınır. Bu sınırlı politika,
  bütün Türkçe olumsuzluğu çözen genel bir yöntem değildir.
- LW niteleyici katsayıları 0,2–1,8; karşıtlık öncesi/sonrası katsayılar 0,85/1,85.
  `classify()` ayrıca modifier/contrast bileşenlerini bastırır. Bu nedenle LW'yi
  yalnız katsayı eklenmiş saf bir ablation gibi tanımlama.
- Parser eğitimi veya gold dependency ağaçlarıyla UAS/LAS değerlendirmesi yoktur.

### Nötr karar kuralı

`src/hisnet/hisnet_classifier.py::NEUTRAL_THRESHOLD = 0.05`.
Dört klasik sistem `HisNetClassifier._prediction_from_scores()` kararını kullanır:

```text
net = toplam_pozitif - toplam_negatif
net >  0.05  → positive
net < -0.05  → negative
diğer       → neutral
```

Sınırlar nötrdür; sınırdaki kayan nokta farkları için `abs_tol=1e-12` kullanılır.
Toplamlar metin uzunluğuna veya duygu sözcüğü sayısına göre normalize edilmez.
`P=N=0` da nötrdür; bu yüzden duygu kanıtı yokluğu ile gerçek nötrlük aynı karara
düşebilir. Mevcut LLM promptları üç etiket adını verir, nötrün anlamsal tanımını
ayrıntılandırmaz.

### LLM ayarları ve hata davranışı

- Ana model `qwen3:32b`; `OLLAMA_MODEL` veya `--model` ile değiştirilebilir.
  `qwen3:4b-instruct` geliştirme manifesti de vardır. Altı koşul, altı ayrı
  eğitilmiş sinir ağı anlamına gelmez.
- 30 örneklik kayıtlı 32B koşusu: 32,8B parametre, GGUF, Q4_K_M.
  Kimlik: `results/smoke_30_20261030_run02/manifest.json`.
- Ana `scripts.run_six_systems` varsayılanları: batch 3, context 8192,
  temperature 0,1, top_p 1, seed 42, timeout 600 saniye, max_retries 1,
  keep_alive `30m`; düşünme `think=False` ile kapalıdır.
- Sınıf kurucusunun varsayılanları farklıdır: batch 8, context 4096,
  max_retries 2. Deneyde kullanılan giriş noktasını ve etkili ayarları kaydet.
- `_build_output_schema()` ve `_validate_predictions()` sonuç sayısı,
  benzersiz 1..N kimlikleri ve etiketleri kontrol eder; sıralamayı geri kurar.
- Teknik hatalar nötre dönüştürülmez: `None` tahmin ve `status="failed"` tutulur.
- Bağlam ön kontrolü gerçek token sayısı değil, UTF-8 bayt uzunluğu + 1024 şablon
  payı + `128 + 64 * batch_size` çıktı payıdır. Girdi sessizce kırpılmaz.
- DP→LLM hem parse hem ek yorumlama talimatı alır; mevcut karşılaştırma bu ikisinin
  ortak etkisidir. Yalnız parse etkisi iddiası için talimatlar eşitlenmelidir.

Üretim kodunda BERT/BERTurk, CNN, LSTM, SVM, lojistik regresyon veya fine-tuning
hattı bulunmuyor. Araştırma notlarındaki önerileri uygulanmış model diye sunma.

## Veri kümeleri ve veri protokolü

Ana preset'ler `scripts/run_six_systems.py::PRESETS` içinde tanımlıdır.
Dosyalar `datasets/Modified/sentence_sentiment/` altındadır.

| Preset / dosya | Metin / etiket sütunu | Ham satır | Temiz satır | Negatif | Nötr | Pozitif |
|---|---|---:|---:|---:|---:|---:|
| `trsav1` / `TRSAv1_fixed.csv` | `review` / `score` | 133614 | 125561 | 41963 | 41571 | 42027 |
| `beyazperde` / `beyazperde_sentiment.csv` | `Review` / `Score` | 2684 | 2654 | 335 | 937 | 1382 |

Son üç sütun, yükleyicinin metin+etiket tekrarlarını kaldırmasından sonraki sayılardır.
TRSAv1 yaklaşık dengelidir; Beyazperde'de pozitif çoğunluk yaklaşık %52,1'dir.

`load_sentiment_dataset()` UTF-8 BOM destekli CSV okur; varsayılan ayırıcı `;`.
Eksik alanları, boş metinleri ve baş/son boşluklarını temizler. `LABEL_MAP`,
İngilizce üç etiketin küçük, baş harfi büyük ve tamamı büyük yazımlarını kabul eder.
Bilinmeyen etiket hata üretir. Sayısal puan, yıldız, `pos/neg`, `notr/Notr`
etiketlerini otomatik eşlemez. Aynı metin+etiket çifti kaldırılır; aynı metnin
çelişen etiketleri korunur. İsteğe bağlı sınıf başına örnekleme seed 42 kullanır.

Diğer mevcut kaynaklar:

| Yol (`sentence_sentiment/` aksi belirtilmedikçe) | Ham içerik / durum |
|---|---|
| `TRSAv1.csv` | 133617 satır; üç geçersiz etiketli kayıt var. Ana preset düzeltilmiş dosyayı kullanır. |
| `combined_dataset.csv` | 489644 satır: 262166 pozitif, 170917 nötr, 56561 negatif. |
| `combined_sentiment.csv` | 6001 satır: 1552 pozitif, 1448 nötr, 3001 negatif. |
| `HUMIRSentimentDatasets.csv` | 64999 satır, iki sınıf: 32500 pozitif, 32499 negatif. |
| `reviews_combined.csv` | 235144 satır: 220265 pozitif, 14879 negatif. |
| `tr_polarity_combined.csv` | 10660 satır, iki sınıf eşit. |
| `Turkish_twitter_combined.csv` | 1161 satır: 523 pozitif, 638 negatif. |
| `datasets/Modified/reviews/` | Ürün yıldızları, film puanları, `fsmtsa.csv` içinde 0/1/2 etiketleri; doğrulanmış eşleme olmadan ana yükleyiciye verme. |
| `datasets/Modified/aspect_sentiment/` | Restoran train/test/trial dosyaları; hedef/aspect/polarite içerir, ana yorum pipeline'ına bağlı değildir. |
| `datasets/Turkish Sentiment Treebank/` | Alanlara ve temsil türlerine göre train/dev/test ağaçları; mevcut sınıflandırıcılar bunlarla eğitilmiyor. |
| `datasets/Unlabeled/` | Metin/Excel kaynakları; bazı alt dizinlerde positive/negative dosyaları da var. Dizin adından etiket durumunu varsayma. |

### Veri kalitesi ve bağımsızlık

- TRSAv1'de 898 metin, 1884 metin+etiket satırında birden fazla etikete sahiptir;
  Beyazperde'de 5 metin/10 satır. Bunları çoğunluk oyu veya model çıktısıyla
  kendiliğinden yeniden etiketleme.
- Özgün nötr anotasyon yönergesi depoda doğrulanamadı. Nötrün kesin olarak bir
  yıldız aralığına karşılık geldiğini varsayma. Kayıtlarda karma görüşlü nötr
  örnekler bulunması, bütün nötr sınıfın tanımı değildir.
- `tests/data_preprocessing/fix_trsav1.py` fiziksel satırları son `;` üzerinden
  ayırır ve geçersiz etiketleri eler; genel çok satırlı CSV dönüştürücüsü değildir.
- `tests/data_preprocessing/11.py` ve `12.py` kaynak train/test dosyalarını
  birleştirir. `12.py` nötrü `Notr` yaparken mevcut CSV `Neutral` içerir; dönüşüm
  geçmişi tam açıklanmıyor. Özgün bölünmelerin korunduğunu varsayma.
- 500 örneklik seçim `train_test_split(..., test_size=500, stratify=gold,
  random_state=42)` kullanır; kalan bölümde model eğitilmez.
- 1200 örneklik seçim sınıf başına 400; 3000 örneklik seçim sınıf başına 1000'dir.
  `scripts/balanced_classical_sample.py::load_disjoint_sample()` eski örneklemdeki
  kırpılmış tam metinleri dışlar; seçilen metinleri tekilleştirir. Sınıf seçim sırası
  negative, neutral, positive olarak kaydedilir.
- Kayıtlı 1200 ve 3000 kümeleri arasında tam metin örtüşmesi sıfırdır; 3000 kümesinde
  kaynakta başka etiketi de olan 43 metin vardır. Bu, yakın kopya denetimi değildir.
- Güncel TRSAv1'in 125561 ve Beyazperde'nin 2654 satırının tamamının metni eski tam
  değerlendirme çıktılarında vardır. Mevcut korpustan yeni örnek çekmek tek başına
  bağımsız nihai test oluşturmaz. Bu, LLM ön eğitimine sızıntı kanıtı da değildir.
- Yeni öğrenilmiş yöntemlerde tam/yakın kopyaları aynı grupta tut; varsa ürün,
  kullanıcı ve kaynak gruplarını dikkate al. Eşik/özellik seçimini testte yapma.

## Ortam ve yeniden üretilebilirlik

`requirements.txt`: stanza, pandas, numpy, scikit-learn, NlpToolkit-SentiNet,
setuptools==80.9.0, ollama, httpx. Setuptools dışında sürümler sabitlenmemiştir;
bu dosya tek başına tarihsel ortamın eksiksiz kilidi değildir.

13 Kasım 2026 incelemesinde mevcut `.venv` içinde `MorphologicalAnalysis`,
`MorphologicalDisambiguation`, `Dictionary`, `Corpus`, `ftfy`, `matplotlib`
bulunamadı. Bunların bir kısmı klasik pipeline, diğerleri eski ön işleme veya
çizim yolları için gereklidir. NlpToolkit-SentiNet kurulu olması morfoloji
bağımlılıklarının hazır olduğunu göstermiyor. Çalıştırmadan önce durumu kontrol et.

| Bileşen | 30 Ekim kayıtlı ortam | 13 Kasım yerel ortam |
|---|---|---|
| Python | 3.11.9 | 3.14.10 |
| Stanza | 1.14.0 | 1.15.0 |
| PyTorch | 2.13.0 | 2.15.0 |
| pandas | 3.0.5 | 3.1.1 |
| scikit-learn | 1.9.0 | 1.9.1 |

Referans: `results/smoke_30_20261030_run02/environment.json` ve `.venv/pyvenv.cfg`.
Kurulumdan önce gerekli paket/sürüm listesini kaynak importları ve hedef deney
manifestiyle karşılaştır; eksik ortamı sonuçların yeniden üretildiği şeklinde sunma.

- `implementation_fingerprint()` `src/` Python dosyaları, Python sürümü ve kurulu
  paket sürümlerini kapsar. Runner'lar ayrıca kendi kod ve veri hash'lerini kaydeder.
- `resource_manifest()` Stanza ve sözlük/morfoloji kaynaklarını hash'ler;
  `ollama_identity()` model digest'i ve sunucu/model yapılandırmasını kaydeder.
- Kaynak manifestlerinde mutlak yollar ve bazı `__pycache__` girdileri bulunabilir;
  farklı makineye taşıma birebir manifest eşitliğini etkileyebilir.
- Seed 42 kullanımı bütün donanım ve kütüphane sürümlerinde bit düzeyinde
  determinizm garantisi değildir. Genel bir deterministik PyTorch ayarı yoktur.
- SQLite tek yazıcı kilidi ve işlemsel kayıt kullanır. Klasik çıktılar yorum başına,
  LLM çıktıları tamamlanan batch başına kaydedilir. Başarılı satırlar atlanır,
  başarısızlar tek tek yeniden denenir. Yarım kalan batch tekrar çalışabilir.
- Paylaşılan runner morfoloji ve parse'ı bir kez üretir. Paralel işçilere gold
  gönderilmez; ana süreç checkpoint'i yazar. Bağımsız semantik karşılaştırmalar
  doğruluğu kontrol eder, bağımsız hız ölçümü oluşturmaz.

## Çalıştırma komutları

Komutları depo kökünde PowerShell ile çalıştır. Aşağıdaki komutlar kullanım
örnekleridir; belgeyi okumak bunları çalıştırma talimatı değildir. Ortam/model
kaynakları hazır olmalıdır. Yeni deney için var olmayan, benzersiz bir `--output`
dizini seç; aşağıdaki `_new` adlarını da kullanmadan önce kontrol et.

### Model çağırmadan mevcut durum okuma

```powershell
.venv\Scripts\python.exe -B -m scripts.run_six_systems --output results/trsav1_classical_3000_20261101_parallel --status
.venv\Scripts\python.exe -B -m scripts.audit_classical_run results/trsav1_classical_3000_20261101_parallel --progress
```

`--status` mevcut checkpoint'i okur. `--prepare-only` farklıdır: model çıkarımı
yapmaz ama örneklem/manifest/checkpoint dosyaları oluşturur ve kaynak kontrolü
gerektirebilir. `audit_classical_run --close-interrupted` kayıt değiştirir.

### Dört klasik sistem: yeni deneyler

```powershell
# Mevcut 500 örneklik örneklemi yeni çıktı dizininde çalıştır.
.venv\Scripts\python.exe -u -m tests.run_trsav1_classical_500 --output results/trsav1_classical_500_new

# Sınıf başına 400, ortak morfoloji/parse: toplam 1200 yorum.
.venv\Scripts\python.exe -u -m scripts.run_balanced_classical_cached --output results/trsav1_classical_1200_new

# Önceki 1200 metni dışla, sınıf başına 1000: toplam 3000 yorum.
.venv\Scripts\python.exe -u -m scripts.run_balanced_classical_cached --sample-per-class 1000 --exclude-sample results/trsav1_classical_1200_20261101_shared_baseline/sample_1200.csv --output results/trsav1_classical_3000_new

# Aynı dışlama protokolünün üç işçili alternatifi; ayrı bir deneydir.
.venv\Scripts\python.exe -u -m scripts.run_parallel_classical_cached --workers 3 --torch-threads 1 --sample-per-class 1000 --exclude-sample results/trsav1_classical_1200_20261101_shared_baseline/sample_1200.csv --output results/trsav1_classical_3000_parallel_new
```

Bu runner'lar LLM çağırmaz; yine de morfoloji ve parser nedeniyle maliyetli olabilir.
`--source` ile eski çıktı içe aktarımı yalnız kimliği doğrulanan uyumlu kayıtlar
içindir; varsayılan yeni deneyin gereği değildir.

### Altı sistem: canlı çıkarım

```powershell
# Tam korpus; uzun sürer ve iki LLM kolunu da çalıştırır.
.venv\Scripts\python.exe -u -m scripts.run_six_systems --dataset trsav1 --output results/trsav1_six_new --keep-details
.venv\Scripts\python.exe -u -m scripts.run_six_systems --dataset beyazperde --output results/beyazperde_six_new --keep-details

# Yalnız klasik sistemlerle küçük örneklem.
.venv\Scripts\python.exe -u -m scripts.run_six_systems --dataset trsav1 --sample-per-class 10 --systems hisnet dp_hisnet hisnet_dp lw_hisnet_dp --output results/trsav1_classical_small_new --keep-details

# Elle hazırlanmış 30 örnek üzerinde altı sistemin canlı smoke koşusu.
.venv\Scripts\python.exe -u -m scripts.run_six_model_smoke --model qwen3:32b --batch-size 3 --num-ctx 8192 --output results/smoke_30_new
```

`run_six_systems` özel CSV için `--csv`, `--text-column`, `--label-column` ve
gerekirse `--separator` alır. `--dataset` ile `--csv` birlikte verilmez.
`--keep-details` kapsam/parse incelemesi için önemlidir; varsayılan çıktı daha
sınırlıdır. Kesinti sonrası değişmemiş komutu ve çıktı dizinini tekrar kullan.

### Kayıtlı sonuçlardan rapor ve eşik deneyi

```powershell
# Yeni koşu tamamlandıktan sonra rapor üret; hedefteki türetilmiş dosyalar yeniden yazılır.
.venv\Scripts\python.exe -m scripts.analyze_classical_500 results/trsav1_classical_3000_new

# Sabit girdilerle 0,05 tabanını doğrular, 0,07 ve 0,10 kararlarını ayrı çıktıda üretir.
.venv\Scripts\python.exe -m scripts.run_classical_threshold_experiment --source results/trsav1_classical_3000_20261101_parallel --output results/trsav1_thresholds_new
```

Eşik betiği bu 3000 örneklik protokolü doğrular; genel amaçlı herhangi bir veri
kümesinde eşik araması değildir. Kaynak checkpoint salt okunur kullanılır,
yeni sonuçlar yazılır. Varsayılan üretim eşiği kalıcı olarak değiştirilmez.

### Hedefli çevrimdışı testler

Bağımlılıklar hazır olduğunda, değişen davranışa uygun modülleri seç:

```powershell
.venv\Scripts\python.exe -B -m unittest tests.test_classical_scope_corrections tests.test_explicit_negation_scope tests.test_finite_predicate_protection -v
.venv\Scripts\python.exe -B -m unittest tests.test_audit_regressions tests.test_resumable_evaluation -v
.venv\Scripts\python.exe -B -m unittest tests.test_classical_threshold_experiment -v
```

Bu seçili modüller çevrimdışı regresyon içindir; bazıları geçici dosyalar oluşturur.
`tests/test_sentiment_threshold.py` ayrıca doğrudan assert kullanan bir betiktir.
Genel `pytest` veya `unittest discover` çağırmadan dosyaları incele: bazı eski
`test_*.py` ve `run_*.py` dosyaları import anında model/deney çalıştırır.

## Evaluation yöntemleri ve raporlama

- Eski akış: `SentimentEvaluator._calculate_metrics()` sklearn ile accuracy,
  macro/weighted precision–recall–F1, sınıf raporu ve confusion matrix hesaplar.
- Yeni akış: `checkpoint_store.summarize_connection()` başarı, hata, bekleyen
  satır ve coverage yanında `accuracy_on_successful`, `macro_f1_on_successful`
  üretir. Matris sırası `negative`, `neutral`, `positive`; satır gold, sütun tahmin.
- `scripts/analyze_classical_500.py::analyze()` örneklem boyutunu kayıtlardan alır;
  sınıf metrikleri, hata/uzunluk/sıfır-puan dilimleri, sözlük kapsamı ve işlem
  tanıları üretir. 500 adlı olması 1200/3000 analizini engellemez.
- Eşleştirilmiş doğru/yanlış değişimleri için iki yönlü exact binom testiyle
  McNemar, çoklu karşılaştırmalar için Holm düzeltmesi ve doğruluk için Wilson
  aralıkları kullanılır. McNemar nötr F1 farkının anlamlılık testi değildir.
- Başarısız tahminleri nötr veya başarı sayma. Farklı sistemlerin başarılı
  altkümeleri farklıysa metrikler doğrudan aynı evrene ait değildir; ortak
  kapsam ve başarısızlık politikasını önceden tanımla.
- Sözlük eşleşme oranı ile duygu taşıyan eşleşme oranı farklıdır. DP→H adayları
  üzerindeki kapsamı, H'nin bütün token'lar üzerindeki kapsamıyla eşitleme.
- Paylaşılan ön işleme sürelerini bağımsız tam pipeline süresi gibi sunma.
  3000 paralel koşusundaki 1–3 Kasım kesintileri süreleri bozmuştur:
  `results/trsav1_classical_3000_20261101_parallel/timing_caveats.json`.

## Doğrulanmış sonuçlar ve sınırlar

### Dengeli 3000 örneklik klasik koşu

Kaynak: `results/trsav1_classical_3000_20261101_parallel/` içindeki
`comparison_3000.csv`, `metrics_3000.csv`, `per_class_3000.csv`, `report.md`,
`final_audit.json`. Her sınıfta 1000 örnek; 12000 başarılı tahmin vardır.
Aşağıdaki değerler yüzde ölçeğindedir.

| Sistem | Accuracy | Macro-F1 | Negatif F1 | Nötr precision | Nötr recall | Nötr F1 | Pozitif F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| H | 49,23 | 46,57 | 56,66 | 38,28 | 19,10 | 25,48 | 57,58 |
| DP→H | 50,23 | 48,11 | 57,94 | 36,45 | 21,80 | 27,28 | 59,10 |
| H→DP | 51,00 | 48,20 | 59,62 | 39,27 | 19,40 | 25,97 | 59,00 |
| LW | 50,47 | 47,52 | 58,72 | 37,65 | 18,30 | 24,63 | 59,20 |

13 Kasım incelemesinde tahminlerden hesaplanan sınıf metrikleri raporlarla uyuştu;
12000 çıktıda skor–etiket tutarsızlığı bulunmadı. Saklanan 24 kaynak anlık
görüntüsü dosyası güncel karşılıklarıyla aynıydı. Bu, farklı mevcut ortamda
uçtan uca çıkarımın yeniden üretildiği anlamına gelmez.

### Diğer sonuçların yeri

| Yol | Kapsam / yorum sınırı |
|---|---|
| `results/trsav1_full_conservative_negation/` | Eski H/H→DP, 125561 örnek; nötr F1 %24,53/%24,51. |
| `results/beyazperde/` | Tam 2654 ve dengeli 300 örneklik eski koşular; tam koşuda nötr F1 %38,04/%38,24. |
| `results/trsav1_fixed/` | Eski 6000 örnek, sınıf başına 2000; nötr F1 %25,57/%25,23. |
| `results/trsav1_classical_500_20261031_run01/` | Dondurulmuş 500 örnek, dört klasik yöntem ve ayrıntılı tanılar. |
| `results/trsav1_classical_1200_20261101_shared_baseline/` | Dengeli 1200, ortak ön işleme ve bağımsız semantik kontroller. |
| `results/trsav1_classical_1200_20261101_corrections/` | Sabit girdilerde kapsam/hizalama düzeltmelerinin önce/sonra karşılaştırması. |
| `results/finite_predicate_fix_20261101/` | Çekimli sıfat yüklemi koruması ve regresyonlar. |
| `results/trsav1_classical_3000_20261101_finite_ablation/` | Aynı 3000 örnekte resolver düzeltmesinin etkisi. |
| `results/trsav1_classical_3000_thresholds_20261111/` | Sabit girdilerle 0,05/0,07/0,10 eşik deneyi. |
| `results/smoke_30_20261030_run02/` | Elle hazırlanmış, sınıf başına 10 örnekli altı sistem smoke koşusu. |
| `results/six_systems_16_detailed/` | 10 pozitif, 6 negatif; nötr yok. `macro_f1_present` ile `macro_f1_3class` farklıdır. |

30 örneklik koşuda Qwen3-32B accuracy %93,33 ve nötr F1 %90,91;
DP→LLM accuracy %96,67 ve nötr F1 %95,24. Bu küçük geliştirme kümesini doğal
3000 yorumla doğrudan kıyaslama. İncelemede tamamlanmış tam-korpus altı sistem
sonucu bulunmadı; bir runner'ın bulunması deneyin tamamlandığını göstermez.

### Nötr başarısındaki doğrulanmış darboğazlar

- Dengeli 3000 kümesinde düşük nötr başarı sınıf azlığıyla açıklanamıyor.
  H→DP gerçek nötrlerin 337'sini negatif, 469'unu pozitif, 194'ünü nötr sınıflıyor.
- H'de `P=N=0` olan 350 yorumun 218'i gerçekte polar; DP→H'de 484 yorumun 308'i
  polar. Kanıt yokluğunu nötr saymak precision'ı düşürüyor.
- Yerel sözlükte `beğen`, `sev`, `bayıl` yok; `beğenmek` +0,75,
  `sevmek` +0,60, `bayılmak` negatif 0,25. Kök–madde biçimi uyuşmazlığı gerçek;
  kör mastar ekleme anlam belirsizliğini çözmez.
  Kanıt: `results/trsav1_classical_3000_20261101_parallel/lexicon_form_probe.json`.
- Karma görüşler eşit puanlarla temsil edilmiyor. Örnek ID 2,
  “temizleyici özelliği güzel ama kokusu kötü”, gold nötr: `güzel` yaklaşık
  +0,4545, `kötü` -0,9; net negatif. LW karşıtlık ağırlığı bunu güçlendirebilir.
- 0,05→0,10 eşik deneyi nötr F1'i yönteme göre yalnız 0,78–1,56 yüzde puanı
  artırırken bütün yöntemlerde pozitif/negatif F1 ve accuracy'yi düşürüyor.
  Varsayılan eşik 0,05 olarak korunmuştur.
- 31 Ekim raporundaki ilgisiz yükleme olumsuzluk taşıma ve tekrar eden yüzey
  hizalama sorunları için 1 Kasım düzeltmeleri mevcuttur; bunları hâlen açık
  hata sayma. Finite predicate koruması da güncel resolver'dadır.
- İroni, emoji kaybı, metin uzunluğu ve kalan parser hatalarının toplam hata
  payları sistematik biçimde ölçülmemiştir; bu açıklamaları hipotez olarak tut.

## Sonraki deneylerin bilimsel önceliği

1. Bağımlılıkları, kaynak kimliklerini ve veri protokolünü sabitle. Mevcut
   kümeleri geliştirme olarak işaretle; bağımsız test iddiasını veri geçmişiyle doğrula.
2. Nötrü görüşsüz, karma ve bağlamı yetersiz örneklerden ayıran anotasyon yönergesini
   netleştir; tahminlere kör insan denetimi ve anlaşmazlık kaydı kullan.
3. POS duyarlı sözlük biçimi eşleme ile ekli olumsuzluğu ayrı ve birlikte 2×2
   deneyde değerlendir. Aynı sözlük arayüzünü dört klasik yönteme adil uygula.
4. Ardından yeni puan dağılımında nötr kararı kalibre et: ham fark yanında toplam
   kanıt, normalize fark, iki kutbun varlığı, kapsam ve cümlecik bilgisi denenebilir.
   Öğrenilmiş karar katmanı eklenirse ayrı model ve eğitim protokolü olarak raporla.
5. LW niteleyici, karşıtlık ve bileşen bastırma etkilerini ayrı ablation'larla ölç.
6. LLM'de açık sınıf tanımı ve ortak talimatlı parse var/yok deneyi yap; bağlam
   kapsamını eşitle. Etiketli örnek eklenirse bunu zero-shot diye adlandırma.
7. Yeni denetimli referans istenirse önce TF-IDF + doğrusal model, sonra BERTurk
   değerlendir. Bunlar henüz mevcut değildir; başarı artışı garanti değildir.

Seçim ölçütünü sonuçlardan önce belirle: nötr F1 artışı, macro-F1 ve üç sınıfın
P/R/F1 değerleri birlikte izlenmeli. Pozitif/negatif için kabul edilebilir kayıp
sınırını önceden tanımla; yalnız nötr recall artışını başarı sayma. F1 farklarında
eşleştirilmiş bootstrap güven aralıkları, öğrenilmiş modellerde birden çok seed
ve çoklu karşılaştırma kontrolü kullan. Nihai testi ayar seçmek için kullanma.

## Ayrıntılı başvuru belgeleri

- [README](README.md)
- [Kayıt/devam akışı](docs/resumable_evaluation.md)
- [31 Ekim paper hazırlığı — tarihsel bulgular](docs/paper_readiness_2026-10-31.md)
- [1200 örnekte kapsam düzeltmeleri](results/trsav1_classical_1200_20261101_corrections/report.md)
- [3000 örnek bulguları](results/trsav1_classical_3000_20261101_parallel/findings.md)
- [3000 örnek son denetimi](results/trsav1_classical_3000_20261101_parallel/final_audit.json)
- [Eşik karşılaştırması](results/trsav1_classical_3000_thresholds_20261111/report.md)
- [Araştırma önerileri — uygulanmış özellik listesi değildir](docs/notr_pozitif_negatif_literatur_onerileri_2026-11-11.md)
