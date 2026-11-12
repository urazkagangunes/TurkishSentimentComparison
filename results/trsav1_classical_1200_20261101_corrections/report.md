# TRSAv1: 1200 örnek, dört yöntem ve düzeltme sonrası analiz

1 Kasım 2026. `TRSAv1_fixed.csv` içindeki 125.561 temizlenmiş satırdan seed 42 ile
400 pozitif, 400 negatif ve 400 nötr örnek seçildi. Dört yöntemde **4800/4800 başarılı
sonuç**, sıfır teknik başarısızlık elde edildi. LLM ve DP→LLM çalıştırılmadı.
Örneklerin metinleri ve etiketleri sonuçlara göre değiştirilmedi.

**En yüksek sonuç DP→HisNet'te; asıl zayıflık nötr sınıf.** Uygulanan düzeltmeler
HisNet→DP'ye bir, ağırlıklı sürüme iki ek doğru tahmin getirdi. Bu örneklemde
doğruyken yanlışa dönen tahmin kalmadı. Kazanç sınırlıdır; yöntemlerin temel
kapsam ve bağlam sorunlarını çözdüğü anlamına gelmez.

| Yöntem | Doğru önce → sonra | Accuracy önce → sonra | Macro-F1 önce → sonra |
| --- | --- | --- | --- |
| HisNet | 572 → 572 | %47,67 → %47,67 | %45,17 → %45,17 |
| DP→HisNet | 602 → 602 | %50,17 → %50,17 | %48,32 → %48,32 |
| HisNet→DP | 594 → 595 | %49,50 → %49,58 | %47,06 → %47,13 |
| Ağırlıklı HisNet→DP | 591 → 593 | %49,25 → %49,42 | %46,61 → %46,75 |

“Önce” gerçek model hesaplarının saklandığı ilk koşudur. “Sonra”, aynı kayıtlı
morfoloji, parse ve sözlük sorguları üzerinde güncel üretim koduyla yapılan
bileşim hesabıdır; 4800 yeni uçtan uca model çalıştırması değildir. Önce eski
resolver ile üç dependency kolundaki **3600 sonucun tamamı** skor, tahmin ve
karşılaştırılan ara alanlar bakımından birebir yeniden üretildi. Sonra yeni
resolver kullanıldı. Kaynak SQLite dosyasının SHA-256 değeri değişmedi:
`d5da29edddb9e311ce4aec8f904117d4cc6e580faf94b2ee5589e9e6536a31ca`.

İlk koşudaki karşılaştırmalarda DP→HisNet, HisNet'in 73 yanlışını düzeltirken
43 doğrusunu bozdu: net +30, yani +2,50 yüzde puan. HisNet→DP'de 39 iyileşme ve
17 bozulma var: net +22. Altı eşleştirilmiş karşılaştırmaya uygulanan Holm
düzeltmesinden sonra exact McNemar p değerleri sırasıyla **0,0342** ve **0,0274**.
Bu geliştirme örnekleminde iki fark da 0,05'in altında; bağımsız bir final veri
setindeki üstünlüğün kanıtı olarak sunulmamalı.

DP→HisNet ile HisNet→DP arasında yalnız sekiz doğru tahmin farkı vardı ve Holm
p=0,9446. Dolayısıyla en yüksek puanı almak, bu iki dependency düzeninden birinin
genel olarak üstün olduğunun kanıtı değil. Ağırlıklı sürüm, düz HisNet→DP'ye göre
28 etiketi değiştirdi; 11 hatayı düzeltti, 14 doğruyu bozdu. İlk koşuda net −3,
Holm p=0,9446. Bu istatistikler **düzeltme öncesi dondurulmuş koşuya** aittir.

İlk koşunun yaklaşık %95 Wilson doğruluk aralıkları HisNet için %44,85–50,50,
DP→HisNet için %47,34–52,99, HisNet→DP için %46,68–52,33 ve ağırlıklı sürüm
için %46,43–52,08. Dengeli örneklemde çoğunluk sınıfı tabanı %33,33'tür.
Aralıklar etiket sorunlarını veya geliştirme verisinin tekrar kullanımını kapsamaz.

| Yöntem | Negatif recall | Nötr recall | Pozitif recall | Nötr F1 |
| --- | --- | --- | --- | --- |
| HisNet | %56,00 | %18,50 | %68,50 | %24,10 |
| DP→HisNet | %56,50 | %23,00 | %71,00 | %28,18 |
| HisNet→DP | %59,75 | %19,50 | %69,25 | %25,20 |
| Ağırlıklı HisNet→DP | %60,00 | %18,25 | %69,50 | %23,86 |

Bu sınıf tablosu da ilk koşuyu gösterir. En iyi nötr recall bile 92/400:
DP→HisNet, 400 nötr yorumun 205'ini pozitif, 103'ünü negatif tahmin etti.
Dört yöntem yalnız 212–253 yorumu nötr, 564–574 yorumu pozitif olarak işaretledi.
Bu dağılım, yalnız genel doğruluğa bakıldığında görünmeyen bir sorunu gösteriyor.

Örneğin ID 8, “Tonu bana göre çok koyu kaldı ama güzel bir ürün dokusu falan hoş”,
nötr etiketli karma bir yorum; dört yöntem de pozitif. ID 16, “güzel. beyandım.”,
de nötr etiketli ve dört yöntem pozitif. İlk örnek bağlam/karma duygu sorununa,
ikincisi etiket yönergesinin incelenmesi ihtiyacına işaret ediyor. Yalnız bu
metinlerden kesin etiket hatası sonucu çıkarılmadı. Hiçbir gold etiket düzeltilmedi.

Kaynak veride başka etiketle de bulunan 20 metin var (%1,67); seçilen örneklem
içinde farklı etiketli aynı metin çifti yok. Bu 20 kayıt çıkarıldığında ilk koşu
doğrulukları %47,63, %50,17, %49,49 ve %49,24. Sıralama değişmiyor; bu çelişkiler
yaklaşık yarı yarıya kalan başarıyı tek başına açıklamıyor.

| Yöntem | Skoru 0/0 olan yorum | Bunlardan pozitif/negatif etiketli |
| --- | --- | --- |
| HisNet | 151 | 95 |
| DP→HisNet | 208 | 130 |
| HisNet→DP | 151 | 95 |
| Ağırlıklı HisNet→DP | 152 | 95 |

DP→HisNet'in aday seçimi, puanlanabilir sözcük kalmayan yorum sayısını artırıyor.
ID 4, “Yakıo kızlar kullanmayın”, dört yöntemde 0/0 ve nötr: `yakıo` ve `kullan`
kökleri sorguda eşleşmemiş. Genel eşleşmeyen kök listesinde `beğen` 104, `sev`
37 kez görülüyor; bunlar yorum sayısı değil kök görülme sayılarıdır. `kullan`
220 ve `gel` 132 kez eşleşmemiş olsa da her fiil kullanımı duygu taşımak zorunda
değildir. Liste otomatik sözlük genişletme listesi olarak kullanılmadı.

539 yorumda dört yöntem de yanlış, 515 yorumda dördü de doğru. 186 yorumda
yöntemler farklı etiket veriyor. Ortak yanlışların büyüklüğü, yalnız pipeline
sırasını veya LW katsayılarını değiştirmenin sınırlı kalacağını gösteriyor.
LW'de 279 kapsamda modifier, 649 kapsamda contrast ağırlığı uygulandı; bu işlem
sayıları doğrudan iyileşen yorum sayısı değildir.

Dependency'nin yararlı olduğu açık örnekler de var. ID 53'te “uygun değilmiş”
ile +0,5 puan ters çevriliyor; HisNet pozitifken üç dependency kolu negatif
etikete ulaşıyor. ID 59'da “tavsiye etmem” olumsuzluğu, HisNet'in nötr tahminini
negatife çeviriyor. Buna karşılık ID 11'de “başka markalar ... bundan bin kat
iyidir” karşılaştırmasındaki `iyi` puanı, incelenen ürün lehine toplanıyor;
dört yöntem de negatif yoruma pozitif diyor. Bu, basit olumsuzluk çözümünden
farklı bir karşılaştırma/hedef bağlamı sınırlaması.

Uygulanan düzeltmeler `src/dependency/sentiment_scope_resolver.py` içindedir:

1. Açık `değil`, asıl yüklem sözlükte bulunmuyor diye ilgisiz önceki duygu
   sözcüğüne taşınmıyor. Gerçek koordinasyon ile “güzel bir ürün değil” ve
   “tavsiye ettiğim ürün değil” gibi bağlı yapılar korunuyor.
2. Parser'ın `değil`i sonraki yükleme bağladığı durumlarda, `değil`e bağlı
   önceki yüklem varsa kapsam o yüklemde tutuluyor. Bu ek kontrol ID 1049'un
   incelenmesiyle eklendi; ilk düzeltme denemesindeki bozulma giderildi.
3. Tekrarlanan yüzey biçimleri tam morfoloji dizisine göre eşleniyor. Önceki
   duygu taşımayan kullanım, sonraki adayın yerini alamıyor. Süslenmiş veya
   parçalanmış token'lar da kullanım sırasını tüketiyor. `yazar🙂 / yazar`
   gibi kontrollü karşı örnekler ve başka cümleden olumsuzluk ödünç alma
   senaryosu regresyon testlerine eklendi.

ID 406'da “diş macunu kadar bile değil” içindeki `değil`, önceki `alacağımı`
sözcüğünü çeviriyordu. Bu bağlantı kaldırılınca ağırlıklı sürümün skoru
0,836364/0,462500'den 0,528030/0,770833'e, etiketi pozitiften negatife döndü.
ID 1049'da “göründüğü gibi değil ... amblemi bile yok” için parser `değil`i
sonraki `yok`a bağlamıştı. Düzeltme H→DP ve LW'de nötr sonucu negatife çevirdi;
DP→HisNet'in zaten doğru negatif sonucu korundu. ID 451'de iki kol pozitiften
negatife geçti ancak gold nötr olduğu için bu bir kazanım olarak sayılmadı.

Son karşılaştırmada DP→HisNet'te 21, H→DP ve LW'de 15'er sonuçta karşılaştırılan
ara alanlar değişti. Etiket değişimleri sırasıyla 0, 2 ve 3; doğruya dönenler
0, 1 ve 2. Doğrudan başarı kazanımları yukarıdaki olumsuzluk örneklerinden
geliyor; kontrollü hizalama testlerinin geçmesini geniş bir başarı artışı gibi
yorumlamamak gerekir. İlk deneme de `round1/` altında korunmuştur.

Sınırlar sürüyor. ID 62'de parser `kalitesiz`i `kalite / siz` diye bölüyor ve
sonraki `değil`i `conj` olarak bağlayarak yanlış kapsam için zemin hazırlıyor;
H→DP/LW'nin yanlış pozitif sonucu bu dar düzeltmeyle çözülmedi. ID 333'te
`pişmanım` token'ı parse'ta hedef/özne gibi göründüğünden negatif katkısı hedef
bastırma kuralıyla siliniyor. Bu tür parse ve hedef seçimi sorunları raporda
ayrı tutuldu; bütün Türkçe kapsam problemlerinin çözüldüğü iddia edilmiyor.
Fiil maddeleştirmesi, sözlük puanları, token'ın kendi ekli olumsuzluğu politikası,
0,05 eşiği ve LW katsayıları değiştirilmedi.

İlk 616 HisNet sonucu aynı veri/kod/kaynak kimliği kontrolüyle aktarıldı.
Morfoloji ve parse her yorum için bir kez hesaplandı; dört yöntemin kararları
ayrı üretildi. Her sınıfın ilk yorumu dört bağımsız pipeline ile tekrar
çalıştırıldı: **12/12 tam anlamsal çıktı eşleşmesi**. JSON'da sayı anahtarlarının
metne dönüşmesinden doğan doğrulayıcı farkı düzeltildi; tanı alanları karşılaştırmadan
çıkarılmadı. Bu kontrol 4800 bağımsız tekrar veya hız benchmark'ı değildir.

HisNet/morfoloji sınıflandırma toplamı 3157,64 saniye (52,63 dakika), ortak parse
277,03 saniye (4,62 dakika). Üç dependency kolunun kayıtlı süreleri ortak ön
işleme sonrasındaki bileşim süreleridir; bunlardan uçtan uca hız üstünlüğü
çıkarılamaz. HisNet'te robust/fallback içeren 465 yorum 3148,19 saniye, kalan
735 yorum 9,45 saniye aldı. Bu gruplama profil bulgusudur; uzunluk ve yazım
farklarını kontrol eden nedensel bir karşılaştırma değildir.

**Son doğrulama: 67 çevrimdışı test geçti.** Bunlar örneklem, JSON karşılaştırması,
paylaşılan girdilerin değişmemesi, kapsam/hizalama ve kayıt/devam kontrollerini
içeriyor. Logdaki LLM hata senaryoları mock kullanır; canlı LLM çağrısı yoktur.
İlk koşudaki 4800 sonuçta otomatik sayısal/yapısal kontrol ihlali sıfırdı;
bu, bütün dilsel kararların doğru olduğu anlamına gelmez.

Ana paper koşusundan önce bu örneklem geliştirme verisi olarak ayrılmalı;
bağımsız değerlendirme iddiası için veri geçmişi dikkate alınmalı. Nötr etiket
yönergesi, fiil kökü/sözlük kapsamı, karşılaştırmalı ifadeler ve parser kaynaklı
hedef bastırması ayrı çalışma başlıklarıdır. Burada eşik veya katsayılar bu
1200 örnekte puan artırmak için ayarlanmadı.

Dosyalar:

- [İlk koşunun tüm metrikleri, sınıf ölçümleri ve karışıklık matrisleri](../trsav1_classical_1200_20261101_shared_baseline/report.md)
- [Dengeli örneklem](../trsav1_classical_1200_20261101_shared_baseline/sample_1200.csv)
- [Dört yöntemin ilk tahminleri](../trsav1_classical_1200_20261101_shared_baseline/comparison_1200.csv)
- [Önce/sonra bütün tahminler](before_after.csv) ve [sınıf metrikleri](per_class_before_after.csv)
- [Değişen ara çıktılar](changed_details.json) ve [karşılaştırma özeti](comparison.json)
- [Bağımsız doğrulama](../trsav1_classical_1200_20261101_shared_baseline/independent_verification.json)
- [67 testin çıktısı](offline_tests_final.log) ve [hedefli eski/yeni test karşılaştırması](regressions.json)

Eski checkpoint güncel kodla devam ettirilmemeli; kaynak kimliği kontrolü bunu
reddeder. Yeni uçtan uca koşu için yeni bir çıktı dizini kullanılmalı.

```powershell
.venv\Scripts\python.exe -u -m scripts.validate_classical_corrections --baseline-resolver results/trsav1_classical_1200_20261101_shared_baseline/source_snapshot/src/dependency/sentiment_scope_resolver.py --source results/trsav1_classical_1200_20261101_shared_baseline --output results/trsav1_classical_1200_20261101_corrections
.venv\Scripts\python.exe -u -m scripts.run_balanced_classical_cached --output results/trsav1_classical_1200_new
```
