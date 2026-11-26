# Altı sistem için kayıtlı çalışma — 31 Ekim 2026

HisNet fiil kökü–sözlük maddesi uyuşmazlığı, sözlük kapsamı/puanları ve morfolojik
çözümleyicinin sınırlamaları bu değişiklikte korunmuştur. Paper'da yöntem
sınırlaması/bulgu olarak raporlanacaktır. Önceki incelemelerde bu sınırlamaları
düzeltme önerileri deney için zorunlu bir ön koşul değildir.

## LLM kontrolü

LLM ve DP → LLM kodu ile mevcut 30 örnekli koşu kayıtları incelendi. Etiket/ID
eşleştirmesinde yeni bir somut hata bulunmadı; bu, tüm girdiler için hatasızlık
iddiası değildir. Uzun canlı LLM testi veya dataset koşusu başlatılmadı.

Ortak istemcide giderilen çalışma sorunları:

- Varsayılan sınırsız bekleme yerine 600 saniyelik, ayarlanabilir HTTP timeout.
  Bu HTTP işlem/ağ bekleme sınırıdır; bütün deneyin toplam süresi değildir.
- Uzun prompt için muhafazakâr bütçe kontrolü: UTF-8 bayt sayısı + 1024 şablon
  payı + `128 + 64 * batch_uzunluğu` çıktı payı. Çıktı aynı `num_predict` ile sınırlı.
  Bu Qwen için ihtiyatlı bir ön elemedir; gerçek tokenizer ile kesin ölçüm değildir.
  Sığabilecek bazı uzun girdiler de reddedilebilir. Metin/parse sessizce kırpılmaz.
  Başarısız batch'in satırları tek tek denenir; tek yorum hâlâ büyükse `failed`
  kalır. Daha büyük context yeni çıktı dizini/koşu gerektirir.
- `done_reason=length` ile biten ya da bildirilen token toplamı context sınırına
  ulaşan yanıt başarı olarak kabul edilmez. Yanıt mesajının yokluğu da kurtarılabilir
  hata sözleşmesine dahildir. Token/ID/etiket doğrulaması korunur.

Ollama'nın yanıt alanları için birincil kaynak:
[Chat API](https://docs.ollama.com/api/chat). Kurulu Python istemcisinde chat için
`truncate` argümanı yoktur; ön kontrol kesin bir sunucu tarafı truncation garantisi
olarak sunulmamalıdır. Nihai uzun-girdi protokolünde bu sınırlama raporlanmalıdır.

DP → LLM, parse bilgisinin yanında onu yorumlamaya ilişkin ek talimatlar da alır.
Karşılaştırma bu ikisini birlikte ölçer; yalnız parse verisinin etkisi olarak
sunulmamalıdır. Prompt içeriği bu çalışma kapsamında değiştirilmedi.

## Kullanım

Proje kökünde, TRSAv1'in tamamı ve altı sistem:

```powershell
.venv\Scripts\python.exe -u -m scripts.run_six_systems --dataset trsav1 --output results/trsav1_six_v1
```

Beyazperde'nin tamamı:

```powershell
.venv\Scripts\python.exe -u -m scripts.run_six_systems --dataset beyazperde --output results/beyazperde_six_v1
```

Kesilirse **aynı komutu, aynı `--output` ile** tekrar çalıştır. `--resume` bayrağına
gerek yok; devam etmek varsayılandır. Kaydedilmiş her başarılı sistem/satır atlanır.
Tamamlanmış LLM sistemleri için sınıflandırıcı kurulmaz ve Ollama sorgulanmaz.

Yalnız kayıtları görüntülemek (model veya ağ çağrısı yapmaz):

```powershell
.venv\Scripts\python.exe -m scripts.run_six_systems --output results/trsav1_six_v1 --status
```

Önce dört klasik sistemi, daha sonra iki LLM sistemini çalıştırmak mümkündür.
İkinci komutta aynı veri/ayarlar/çıktı dizini korunmalıdır:

```powershell
.venv\Scripts\python.exe -u -m scripts.run_six_systems --dataset trsav1 --output results/trsav1_six_v1 --systems hisnet hisnet_dp dp_hisnet lw_hisnet_dp
.venv\Scripts\python.exe -u -m scripts.run_six_systems --dataset trsav1 --output results/trsav1_six_v1 --systems llm dp_llm
```

Başka CSV için:

```powershell
.venv\Scripts\python.exe -u -m scripts.run_six_systems --csv datasets/example.csv --text-column review --label-column score --separator ";" --output results/example_six_v1
```

Mevcut yükleyici kullanılır: etiketler positive/negative/neutral olmalı; aynı
metin+etiket tekrarları kaldırılır, çelişen etiketli metinler korunur. Satır numarası,
temizlenmiş dataset içindeki sıfır tabanlı sıradır. `--sample-per-class 100` her
sınıftan sabit seed ile 100 örnek seçer; verilmezse tüm kullanılabilir veri işlenir.

Varsayılan: `qwen3:32b`, seed 42, temperature 0.1, top_p 1.0, batch 3,
context 8192, bir retry, timeout 600 saniye. İki LLM kolunda aynı ayarlar kullanılır.
`--model`, `--batch-size`, `--num-ctx`, `--max-retries`, `--timeout`, `--seed`
ilk koşuda değiştirilebilir; devam ederken aynı kalmalıdır.
`--prepare-only` veriyi yükleyip checkpoint oluşturur, hiçbir sınıflandırıcı çalıştırmaz.

## Kesinti ve kayıt davranışı

- `progress.sqlite3` asıl kayıttır. Klasik sistemlerde her yorum; LLM'de her batch
  tek transaction içinde kaydedilir. Bütün geçmiş JSON her seferinde yeniden yazılmaz.
- Ctrl+C, süreç kapanması veya elektrik kesintisinde tamamlanmış transaction'lar
  korunur. Cevabı henüz kaydedilmemiş aktif yorum/batch yeniden çalışabilir.
  Sunucu yanıtı ile disk commit'i arasında tam "yalnız bir kez" garantisi yoktur.
- Yeni satırlar özgün batch sınırları korunarak işlenir. Hem bu oturumdaki hem
  önceki oturumlardan kalan başarısızlar sonunda bir kez tek tek denenir.
  Yine başarısız olanlar `None/failed` kalır; sonraki çalıştırmada tekrar denenir.
  Başarısızlık hiçbir zaman `neutral` etiketi değildir.
- Aynı çıktı dizininde iki yazıcıya izin verilmez. İşletim sistemi kilidi süreç
  ölünce bırakır; eski kilit dosyasını silmek gerekmez.
- Veri içeriği/sırası, kod/paket kimliği, kaynak dosyaları ve ayarlar manifestle
  doğrulanır. LLM ilk gerektiğinde model digest'i, sunucu sürümü ve model ayarları
  ayrıca kaydedilir/doğrulanır. Uyuşmazlıkta sonuçlar korunur ve yeni dizin istenir.
- `manifest.json` ve `ollama_identity.json` okunabilir kimlik kopyalarıdır;
  kimliklerin asıl kopyası SQLite içindedir. Eski smoke JSON/CSV kayıtları bu yeni
  sisteme otomatik aktarılmaz; tahminleri farklı deney kimlikleriyle birleştirmez.
- Yedek alırken çalışan süreç varsa SQLite'ın WAL dosyaları da gerekebilir.
  En kolay yöntem süreci durdurduktan sonra çıktı klasörünün tamamını kopyalamaktır.

`summary.json` ve `predictions.csv` her sistem sonunda ve Ctrl+C'de üretilir.
Zorla kapatmada bu dışa aktarımlar eski kalabilir; `--status` doğrudan güncel
SQLite kaydını okur. Başarılı satırlardaki accuracy, macro-F1 ve confusion matrix;
başarılı/başarısız/bekleyen sayıları ve kapsam oranıyla birlikte raporlanır.
Eksik koşunun başarılı-altküme metriği tam dataset skoru değildir.

Süre/token toplamları tüm tamamlanmış girişimlerden alınır; süreç içi sayaçlarla
sıfırlanmaz. Kesilen, cevabı kaydedilmemiş girişimlerin süre/tokenları bilinmez;
bu girişimler ayrıca sayılır. Başlatma süresi ölçüme dahil değildir. Klasik sistemler
birbirlerinin sonuçlarını tekrar kullanmaz; yine de önbellek, donanım ve model
yükleme farkları nedeniyle süreler kontrollü hız benchmark'ı sayılmaz.

Varsayılan kayıtlar etiketler, hatalar, süre/tokenlar, LLM ham cevabı ve temel
skorlarla sınırlıdır. `--keep-details` tam parse/kapsam çıktısını da kaydeder;
büyük veride disk tüketimini artırır. Bu tercih koşu kimliğinin parçasıdır.

Çıkış kodları: seçilen sistemler tam başarıysa 0, çözülmemiş hata varsa 2,
Ctrl+C ile kesilirse 130. Genel hata veya kimlik uyuşmazlığı traceback ile sonlanır.

## Yedinci sistem olarak Jev

`--systems all` dört HisNet yöntemini, iki Ollama yöntemini ve TypeSafe Jev'i
aynı örnekler üzerinde çalıştırır. Sistem seçimi verilmediğinde yerel altı
yöntem kullanılır. Aşağıdaki komut sınıf başına 10 örnek seçer. Yalnız Jev'i
bir pilot checkpoint'ine eklemek için aynı ayarlar ve çıktı diziniyle
`--systems all` yerine `--systems jev` kullanılabilir.

```powershell
.venv\Scripts\python.exe -u -m scripts.run_six_systems --dataset trsav1 --sample-per-class 10 --seed 42 --model qwen3:4b-instruct --systems all --output results/trsav1_seven_typesafe
```

Jev, TypeSafe hesabının `TYPESAFE_API_KEY` ortam değişkeniyle çalışır.
Her API isteği, tek bir yorum metnini ve `negative`, `neutral`, `positive`
seçeneklerini tanımlayan sabit bir Türkçe Choice sorusunu gönderir. İstek
etiketli örnek, doğru etiket veya bağımlılık ağacı içermez. Modelin seçtiği
etiket, sınıf olasılıkları, güven skoru ve sağlayıcının bildirdiği token
sayıları SQLite'a kaydedilir. Jev'in yanıtında bildirilen model sürümü
istenen sürümle eşleşmelidir.
[TypeSafe Choice](https://docs.typesafe.ai/primitives/choice) · [TypeSafe API](https://docs.typesafe.ai/api/system-one)

`--jev-model` varsayılan olarak `jev-1.14.0` kullanır ve sürüm numarası ister.
`--jev-timeout` varsayılan olarak 30 saniyedir. Uç nokta, sürüm, soru ve
timeout `jev_identity.json` dosyasına ve SQLite'a kaydedilir. API anahtarı
checkpoint'e yazılmaz. Ağ veya hesap hatasında koşu kaydedilen noktada
durur. Rate limit ve overload yanıtlarında, bekleme aralığı uygunsa tek
bir API tekrarı yapılır. Eksik satırlar aynı komutla sürdürülebilir.

`summary.json` her sistemin doğruluğunu, macro-F1'ını, hata matrisini,
kapsamını, süresini ve istek/token toplamlarını içerir. Nihai kıyas için
yedi sistemde de aynı yorumların tamamlanmasını bekle. Pilotun 30 yorumu
entegrasyonu kontrol etmek içindir. Nihai deneyde örneklemi ve model
sürümlerini önceden sabitle. Jev süresi ağ ve sağlayıcı gecikmesini,
Ollama süreleri yerel çalıştırmayı içerir.

### OpenCode üzerinden ücretsiz Jev 1.13

`--jev-provider opencode`, OpenCode'un System One uç noktasında
`jev-1.13-free` modelini seçer. Anahtarsız istek çalışırsa
`OPENCODE_API_KEY` gerekmez. Ortam değişkeni ayarlıysa istemci onu yalnız
OpenCode'a gönderir. Sağlayıcı bu modeli sınırlı süreyle ücretsiz listeler.
Çalıştırıcı her başarılı tahminin `billing_source=free` ve `cost=0`
olduğunu doğrular. Bu alanlar yanıtta doğrulanamazsa koşu durur.
[OpenCode Jev belgeleri](https://opencode.ai/docs/console/models/#jev)

```powershell
.venv\Scripts\python.exe -u -m scripts.run_six_systems --dataset trsav1 --sample-per-class 10 --seed 42 --model qwen3:4b-instruct --systems jev --jev-provider opencode --output results/trsav1_jev_opencode_pilot_30
```

Bu komut OpenCode Jev 1.13 için ayrı bir checkpoint kullanır. Altı yerel
sistemin önceki pilotuyla sıra ve etiket dahil örneklem hash'i aynıdır.
Yeni bir yedi sistem koşusu için `--systems all` ve yeni bir çıktı dizini
kullanılabilir. Sağlayıcı veya sürüm değişirse yeni bir çıktı dizini gerekir.

## Önceki altı veri seti doğrulaması

Ana makale koşusunun komutları mevcut iki test girişini kullanan
[makale protokolünde](paper_evaluation.md) yer alır.
`scripts.run_six_systems --paper` önceki canlı doğrulamalarda kullanılan
SQLite profilidir.

`--paper` aşağıdaki altı dosyayı sırayla açar ve yedi sistemi aynı örnekler
üzerinde çalıştırır. Jev için OpenCode `jev-1.13-free` ve iki Ollama kolu
için `qwen3:32b` kullanılır. Seed 42, batch boyutu 3 ve Ollama bağlam sınırı
8192 varsayılanları korunur.

| Veri seti | Negatif | Nötr | Pozitif | Toplam |
|---|---:|---:|---:|---:|
| `beyazperde_sentiment` | 335 | 937 | 1.382 | 2.654 |
| `HUMIRSentimentDatasets` | 32.295 | 0 | 32.148 | 64.443 |
| `TRSAv1_fixed` | 41.963 | 41.571 | 42.027 | 125.561 |
| `combined_sentiment` | 2.960 | 1.443 | 1.545 | 5.948 |
| `tr_polarity_combined` | 5.316 | 0 | 5.323 | 10.639 |
| `Turkish_twitter_combined` | 638 | 0 | 523 | 1.161 |
| **Toplam** | **83.507** | **43.951** | **82.948** | **210.406** |

`TRSAv1.csv`, `combined_dataset.csv` ve `reviews_combined.csv` makale
profilinin dışında. Satır sayıları mevcut yükleyicinin boş değer temizliği ve aynı metin–etiket
tekilleştirmesinden sonraki sayılardır. Etiketi çelişen tekrarlar
mevcut yükleyici kuralıyla kalır. Veri setleri ayrı raporlanır.
Örneğin Beyazperde ve HUMIR arasında 82 farklı tam metin iki dosyada da bulunur.

```powershell
.venv\Scripts\python.exe -m scripts.run_six_systems --paper --output results/paper_main_20261126 --status
.venv\Scripts\python.exe -m scripts.run_six_systems --paper --output results/paper_live_smoke_20261126 --status
```

Tam kapsam 210.406 yorum–etiket çifti ve 1.472.842 sistem tahminidir.
Her veri setinin kendi manifesti ve SQLite checkpoint'i vardır.
`paper_summary.json` sistem bazında başarı, başarısızlık ve kapsamı,
ayrıca yedi sistemin ortak başarılı satırlarındaki accuracy, Macro-F1
ve hata matrisini kaydeder. İki etiketli setlerde Macro-F1 negatif ve pozitif
sınıfları ortalar. Nötr tahmin hata sayılır ve hata matrisinin nötr sütununda
görünür. Başarısız veya bekleyen satırlar kapsamda ayrıca gösterilir.
8192 varsayılanıyla düz LLM'in tek yorumluk bağlam ön kontrolü HUMIR'deki
21 örneği reddeder. DP → LLM için ayrıştırma içeriği de bütçeye dahil
olur. Çalıştırıcı bu girdileri kesmeden başarısız olarak kaydeder.

`--sample-per-class 1 --model qwen3:4b-instruct` ile ayrı bir çıktı dizininde
küçük bir canlı doğrulama yapılabilir. Bu ayar her veri setinin mevcut
etiketlerinden birer örnek seçer. Tam makale koşusunun checkpoint'inde
örneklem ve model ayarları sabittir.

## Çevrimdışı doğrulama

31 Ekim 2026 sonucu: yeni kayıt/devam paketinde **17/17**, mevcut regresyon
paketinde **19/19** test geçti; toplam **36 başarılı test**. İlk denemede bulunan
Windows SQLite durum-okuma bağlantısı sızıntısı kapatıldı ve tekrar doğrulandı.

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p test_resumable_evaluation.py
.venv\Scripts\python.exe -m unittest discover -s tests -p test_audit_regressions.py
```

İlk paket altı sistemde kesinti/devam, tamamlanmış satırı tekrar çağırmama,
başarısız satır kurtarma, batch transaction rollback, zorla süreç sonlandırma,
kimlik uyuşmazlığı, eşzamanlı yazıcı kilidi, CSV ve LLM hata sözleşmelerini sahte
sınıflandırıcı/istemcilerle sınar. Stanza modeli/HisNet yüklenmez ve LLM üretimi
yapılmaz. Genel `tests/` keşfi kullanılmamalıdır: bazı eski dosyalar import
sırasında gerçek uzun deneyleri başlatır.
