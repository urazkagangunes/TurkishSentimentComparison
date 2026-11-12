c# Canlı küçük test — 30 Ekim 2026

30 elle hazırlanmış cümle: 10 pozitif, 10 negatif, 10 nötr. Örnekler ve sıra
testten önce sabitlendi. Bu set geliştirme/teşhis amaçlıdır; paper test kümesi değildir.
Koşu sırasında model karar kuralları, sözlük puanları veya katsayılar değiştirilmedi.

Tamamlanan güncel koşu: `results/smoke_30_20261030_run02`.
Önceki yarım koşu: `results/smoke_30_20261030_run01`; kayıtları korunuyor.
LLM: `qwen3:32b`, Q4_K_M; seed 42; temperature 0.1; top_p 1;
batch 3; context 8192; thinking kapalı. İki LLM kolunda aynı ayarlar kullanılıyor.

Çalıştırma:

```powershell
.venv\Scripts\python.exe -u -m scripts.run_six_model_smoke --output results/smoke_30_20261030_run02 --model qwen3:32b
```

Kaydedilen sonuçlardan devam etmek için aynı komut kullanılabilir. Kaynak veya
ayar değişmişse manifest kontrolü devam etmeyi reddeder; yeni çıktı dizini gerekir.
Mevcut runner tamamlanmış veya failed olarak kaydedilmiş satırları yeniden çalıştırmaz;
otomatik bireysel recovery aynı sürecin sonunda yapılır. Başarısız satırları ayrıca
yeniden denemek gerekirse yeni ve açıkça kaydedilmiş bir recovery koşusu kullanılmalı.

## Kayıtlar

- `manifest.json`: kaynak/paket kimliği, fixture hash'i, model digest'i, Stanza ve
  lexicon/morfoloji kaynak hash'leri, örnek sırası, LLM ayarları.
- `hisnet.json`, `hisnet_dp.json`, `dp_hisnet.json`, `lw_hisnet_dp.json`: kökler,
  token'lar, adaylar, kapsamlar, puanlar, teknik kontroller ve tekrar sonuçları.
- `llm.json`, `dp_llm.json`: batch sonuçları, ham yanıt, istek süreleri/tokenları;
  DP kolunda parse bilgisi de bulunur.
- Önceki `run01/lexicon_probe.json`: hatalı örnekler için gerçek sözlük maddeleri
  ve puanları. İki koşunun sözlük hash'leri aynı.
- `comparison.csv`, `summary.json`: runner'ın son karşılaştırması.
- `comparison.md`, `analysis.json`: analiz scriptinin örnek bazında raporu ve
  tüm kaydedilmiş oturumları birleştiren süre/token toplamları.

Analiz komutu:

```powershell
.venv\Scripts\python.exe -m scripts.analyze_six_model_smoke results/smoke_30_20261030_run02
```

## Altı modelin tamamlanan sonuçları

| Sistem | Doğru / 30 |
|---|---:|
| HisNet | 20 |
| DP → HisNet | 25 |
| HisNet → DP | 24 |
| LW HisNet → DP | 24 |
| LLM (`qwen3:32b`) | 28 |
| DP → LLM (`qwen3:32b`) | 29 |

Altı kolun her birinde 30 sonuç mevcut; toplam 180 ana sınıflandırmada teknik
hata veya çözümlenmemiş satır yok. İki LLM kolunda toplam 20 istek, sıfır retry
ve sıfır bireysel recovery kaydedildi. Bütün isteklerin bitiş nedeni `stop`.

## Klasik modellerin kontrolleri

120 ana klasik sınıflandırmada teknik hata veya kontrol ihlali bulunmadı.
Her model için 5 örnek ters sırada tekrarlandı: 20/20 tekrar aynı tahmin, skor ve
morfolojik analizi verdi. Bu dar kontrol, bütün veride sıra bağımsızlığını kanıtlamaz.

Kontrol edilen koşullar: skorlar sonlu ve negatif değil; kapsam skorları toplamı
nihai skora eşit; H→DP ve LW'nin başlangıç lexicon skorları H ile aynı; LW'nin
yeniden hesapladığı temel DP skorları H→DP ile aynı; temel HisNet sonucu yerinde
değiştirilmemiş. Klasik süreler ortak temel sonuçların tekrar kullanımını içerdiğinden
adil bir uçtan uca hız karşılaştırması olarak kullanılmamalı.

Yeni koşudaki 120 klasik sonucun ayrıntılı `result` nesneleri önceki koşuyla
aynı. Karşılaştırma `run02/classical_rerun_check.json` dosyasında kayıtlı.

## LLM karşılaştırması

İki kolun tahminleri yalnızca P05'te farklı:

| ID | Metin | Sabit beklenen etiket | LLM | DP → LLM |
|---|---|---|---|---|
| P05 | Otel kötü değildi. | positive | neutral | positive |
| N08 | Otel güzel ama odalar çok kötü. | negative | neutral | neutral |

Normal LLM'nin iki, DP → LLM'nin bir etiket farkı var; diğer 28 örnek iki kolda
da beklenen etiketi aldı. İki kol da 10 nötr örneğin tamamını doğru sınıflandırdı.
Önceki yarım koşudaki 27 normal LLM tahmini, yeni koşuda aynı şekilde tekrarlandı.

P05'in pozitif etiketi testten önce sabitlenmişti; "kötü değildi" ifadesi
nötr/zayıf olumlu yorumlara açık. N08 de karma duygu içeriyor; testin negatif
etiketi karşıtlık sonrası bölümün baskın olduğu yorumunu yansıtıyor. Etiketler
sonuçlara bakılarak değiştirilmedi. Bu küçük elle hazırlanmış sette tek örneklik
fark, dependency bilgisinin genel başarıyı artırdığını göstermeye yetmez.

| Kayıtlı ölçüm | LLM | DP → LLM |
|---|---:|---:|
| İstek sayısı | 10 | 10 |
| Toplam istek süresi (saniye) | 827,47 | 657,07 |
| Girdi token toplamı | 1909 | 10616 |
| Çıktı token toplamı | 502 | 510 |
| İstek başına girdi token aralığı | 185–199 | 918–1214 |

DP → LLM için ayrıca 6,55 saniye parse süresi kaydedildi. Normal LLM'nin ilk
isteği 379,07 saniyeydi; sunucu bunun 231,58 saniyesini model yükleme olarak
bildirdi. DP kolu aynı model bellekte yüklüyken çalıştı. Bu nedenle tablodaki
toplamlardan DP kolunun daha hızlı olduğu sonucu çıkarılamaz.

Ollama'nın canlı model kaydı 8192 context değerini doğruladı
(`run02/ollama_loaded_model.json`). Sunucunun bildirdiği girdi/çıktı token sayıları
bu sınırın oldukça altında. Bu ölçüm gerçek tokenizer ile bağımsız bir tam-girdi
doğrulaması değildir; uzun gerçek yorumlar için truncation kontrolü hâlâ gerekli.

## Yeni önemli bulgu: fiil kökü ile sözlük maddesi uyuşmuyor

`Bu filmi sevdim` ve `Bu ürünü çok beğendim` bütün klasik kollarda nötr çıktı.
Morfolojik analiz sırasıyla `sev` ve `beğen` köklerini üretiyor; bu anahtarlar
HisNet'te bulunmuyor. Gerçek sözlükte:

| Madde | Pozitif | Negatif |
|---|---:|---:|
| sevmek | 0.60 | 0 |
| beğenmek | 0.75 | 0 |

Bu, yalnızca sözlüğün kelimeyi bilmemesi değil, iki bileşenin madde biçimlerinin
uyuşmamasıdır. POS kontrollü fiil maddeleştirme gerekir; isimlerle aynı yazılan
kökler için körlemesine `-mak/-mek` eklemek doğru olmaz. Ortak düzeltme dört klasik
kolun aynı lexicon mantığını kullanmasını korumalı.

Olumsuzlukla etkileşim ayrıca test edilmeli: mevcut muhafazakâr kapsam politikası
token'ın kendi ekli olumsuzluğunu dışlıyor. Fiil lookup'ı tek başına düzeltilirse
`sevmedim` de `sevmek` pozitif puanını alabilir. Dolayısıyla lookup düzeltmesi ile
ekli olumsuzluk politikasının etkileri ayrı kaydedilmeli.

`Bu ürünü hiç beğenmedim` şu an doğru negatif çıkıyor, ancak ara çıktılar bunun
`beğenmedim` çözümünden değil, `hiç` maddesinin 0.5 negatif puanından geldiğini
gösteriyor. Doğru etiket tek başına doğru dilsel çözümleme kanıtı değil.

## Sözlük ve morfoloji sınırlamaları

- `Hizmet berbattı`: `berbat` maddesi sözlükte var, ama puanları 0/0. Bütün klasik
  kollar nötr çıkıyor. Bu, lookup kodu düzeltilerek çözülecek bir eksik anahtar değil.
- `Ürünün modeli X200`: `model` maddesi 0.1111 pozitif puana sahip; bütün klasik
  kollar nötr olgu cümlesine pozitif diyor. Bağlamsız sözlük puanının sınırı.
- `Otelin beş katı var`: morfoloji `katı+ADJ` seçiyor; sözlük 0.142857 negatif
  puan veriyor. H, H→DP ve LW negatif. DP→H, özne konumundaki bu token'ı duygu
  adayı seçmediği için nötr.
- `Bu kitabı bir yazar yazar`: sonuç nötr, fakat hem morfoloji hem Stanza iki
  `yazar` kullanımını da isim analiz ediyor. Tekrarlanan token eşleştirme düzeltmesi,
  dış çözümleyicinin yanlış POS/kök seçimini kendiliğinden düzeltmez.

## Çalıştığı gözlenen durumlar

- `Otel güzel değildi`, `Otel kötü değildi`, `tavsiye etmem`, `pişman olmadım`
  örneklerinde dependency kolları beklenen etikete ulaşıyor.
- `Kötü. Merkezi değil.` örneğinde ikinci cümlenin olumsuzluğu ilk cümlenin `kötü`
  puanını çevirmiyor; cümle sınırı düzeltmesi gerçek parser ile de çalıştı.
- `çok ama çok güzel` için LW modifier 1.4, contrast 1.0; buradaki `ama`
  yanlış karşıtlık ağırlığına dönüşmüyor.
- `Otel güzel ama odalar çok kötü`: LW, `güzel` için 0.85; `kötü` için
  modifier 1.2 ve contrast 1.85 uyguluyor.
- Büyük harfli Türkçe I/İ ve seçilen emoji örneği beklenen şekilde işleniyor.
  Emoji örneğinde Stanza token'ları zaten doğru ayırdı; bu örnek fragment repair
  mekanizmasının bütün bozuk tokenizer çıktılarında çalıştığını kanıtlamaz.

## Çalıştırma notu

### Kesinti sonrası devam

Akşam devam edildiğinde önceki koşuda normal LLM için 27/30 sonuç kayıtlıydı;
DP → LLM henüz başlamamıştı. Aynı klasörde devam girişimi manifest kontrolünde,
yeni tahmin üretilmeden reddedildi: Ollama sunucusu `0.40.11`den `0.40.13`e
geçmişti. İki manifestin karşılaştırmasında değişen tek alan `server` oldu;
kaynak/paket kimliği, runner, fixture, örnek sırası, model digest'i, ayarlar ve
kaynak dosyası hash'leri aynı kaldı.

Bu nedenle `run02` klasöründe altı kolun tamamı yeniden başlatıldı; önceki
tahminler yeni koşuya taşınmadı. Sınıflandırıcı ve runner kodu değiştirilmedi.
Önceki koşunun `execution_notes.json` kaydı `incomplete_preserved` olarak
işaretlendi. Önceki ortam kaydı ayrıca `environment_pre_resume_20261030.json`
dosyasında korundu.

`run02` başarıyla tamamlandı ve karşılaştırma raporları üretildi. Rapor scripti
ilk çalıştırmada `wrong_ids` liste ifadesindeki eksik `]` nedeniyle sözdizimi
hatası verdi; yalnızca bu eksik parantez düzeltildi. Tekrar çalıştırma başarılı
oldu. Bu düzeltme model tahminlerine veya koşu manifestine dokunmadı.

### Önceki oturum

32B ilk batch yaklaşık 296 saniye sürdü (yaklaşık 96 saniye yükleme, 79 saniye
prompt değerlendirme, 118 saniye üretim). İlk batch 3/3 doğruydu. Donanım nedeniyle
geliştirme modeliyle ayrı bir koşuya kısa süreli geçiş başlatıldı; kullanıcının
32B ile devam tercihi gelince bu girişim klasik aşamada durduruldu. Ana koşu 32B
ile kaydedilmiş satırlardan sürdürüldü. Yarım 4B klasörü ana karşılaştırmaya dahil değil.

Süre analizinde süreç içi `get_stats()` tek başına kullanılmamalı: resume sonrası
bu sayaçlar yeniden başlar. Analiz scripti bütün kaydedilmiş batch/attempt kayıtlarını
toplar. Kesilmiş ve yanıtı kaydedilmemiş isteklerin süresi bu toplamda yoktur.

Yeni bulunan lookup uyuşmazlığı ve ekli olumsuzluk politikası ele alınmadan nihai
paper koşusuna geçilmemeli. Bu geliştirme setindeki sonuçlar bağımsız benchmark
performansı olarak sunulmamalı.
