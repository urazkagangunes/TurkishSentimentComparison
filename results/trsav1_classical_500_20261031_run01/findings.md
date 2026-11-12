# 500 yorumluk koşunun yorumlanması

31 Ekim 2026. Dört sistemde 500'er başarılı sonuç, toplam 2.000 sınıflandırma.
Örneklem önceki seed 42 örneklemiyle aynı: 167 negatif, 166 nötr, 167 pozitif.
Eşik, sözlük, morfoloji ve ağırlıklandırma kuralları bu deney sırasında değiştirilmedi.
LLM çalıştırılmadı. Koşu, sohbet limiti dolduktan sonra arka planda tamamlandı;
buradaki inceleme kayıtlı sonuçlara dayanır.

## Başarı farkı küçük; en yüksek skor HisNet → DP'de

| Sistem | Doğru | Accuracy | Macro-F1 |
| --- | --- | --- | --- |
| HisNet | 263 | %52,60 | %50,20 |
| DP → HisNet | 262 | %52,40 | %50,74 |
| HisNet → DP | 270 | %54,00 | %51,23 |
| LW HisNet → DP | 269 | %53,80 | %51,09 |

HisNet → DP, HisNet'in 12 yanlışını düzeltti, 5 doğrusunu bozdu: net +7 yorum,
yani +1,40 yüzde puan. Eşleştirilmiş exact McNemar p=0,1435; altı karşılaştırma
için Holm düzeltmesi sonrası p=0,8608. Bu küçük geliştirme örnekleminde
istatistiksel üstünlük gösterilmiş değildir. Bütün sistem çiftlerinde düzeltilmiş
p değeri 0,05'in üzerindedir; bu da yöntemlerin eşdeğer olduğunu kanıtlamaz.

LW, HisNet → DP'ye göre 17 tahmini değiştirdi; 4 hatayı düzeltti, 5 doğruyu bozdu.
124 kapsamda modifier, 272 kapsamda contrast ağırlığı uygulanmasına rağmen
etiketlerin çoğu değişmedi. Ağırlık uygulamak, tek başına başarı artışı getirmedi.

## En belirgin zayıflık nötr sınıf

| Sistem | Negatif recall | Nötr recall | Pozitif recall |
| --- | --- | --- | --- |
| HisNet | %68,86 | %22,29 | %66,47 |
| DP → HisNet | %61,68 | %25,90 | %69,46 |
| HisNet → DP | %71,26 | %21,08 | %69,46 |
| LW HisNet → DP | %70,66 | %21,08 | %69,46 |

166 gerçek nötr yoruma karşılık sistemler yalnız 89–111 yorumu nötr olarak tahmin
ediyor. HisNet → DP'de 131/166 nötr yorum polar sınıflara gidiyor. Bu durum,
bağlamsız sözcük puanları, karma yorumlar ve etiket tanımının birlikte incelenmesini
gerektiriyor; doğrudan bir sayısal eşik bug'ı olarak sınıflandırılmadı.

Nötr etiketli ID 30 (`Beğendim fiyata göre çok iyi`) ve ID 32 (`Ürünler çok iyi
özellikle tonik çok güzel kokuyor`) dört modelde de pozitif. Bu örnekler etiket
yönergesini incelemek için adaydır; yalnız metne bakarak etiketlerin yanlış olduğu
kesinleştirilmedi ve hiçbir etiket değiştirilmedi.

Kaynak veride aynı metni farklı etiketle de bulunan 8 örnek mevcut. Bu sekiz
satır çıkarıldığında doğruluklar sırasıyla %53,05, %52,85, %54,47 ve %54,27 oluyor.
Sıralama değişmiyor; bu çelişkiler toplam hata oranını tek başına açıklamıyor.
Ana sonuçlar yine 500 yorumun tamamına aittir.

## Örneklerde görülen davranışlar

### ID 9 — olumsuzluk düzeltmesinin işe yaradığı durum

`Aşırı derecede cildimi kuruttu tavsiye etmem` — gold: negative.

HisNet skoru +0,125 / −0,05 ve tahmini positive. Üç dependency kolunda `etmem`,
`tavsiye` kapsamını çeviriyor; son skor +0 / −0,175 ve tahmin negative oluyor.
Burada yalnız doğru etiket değil, beklenen ara işlem de kayıtlıdır.

### ID 77 — aday seçimi ve eksik kökün birlikte etkisi

`indiirimde takrar stoklayacagim kalitesini çok beğendim` — gold: positive.

HisNet, `kalite` kökünden +0,5 alıyor; H→DP ve LW bunu koruyor. DP→H, `kalitesini`
nesne/hedef olarak görüyor; duygu adayı `beğendim` için morfolojik kök `beğen`
HisNet'te bulunamıyor. Sonuç 0/0 ve neutral. Bu, önceden kabul edilen
sözlük–morfoloji sınırlamasının aday seçimiyle etkileşimini gösteriyor.
Sözlük veya fiil maddeleştirme kuralı değiştirilmedi.

Genelde HisNet'in 64 yorumunda skor 0/0 iken DP→H'de bu sayı 97. Bu 97 yorumun
64'ünün gold etiketi pozitif veya negatif. Aday seçimi sonrasında puanlanabilir
duygu kalmaması, DP→H için belirgin bir hata kaynağı.

### ID 48 — ağırlıklandırma doğru tahmini bozabiliyor

`Kapatıcılığı çok iyi ama çok koyu halkalarınız varsa tek başına yeterli değil.`
— gold: positive.

H→DP: +0,611111 / −0,3125, net +0,298611 → positive.
LW: +0,623333 / −0,60125, net +0,022083 → neutral.

`iyi` için 1,2 modifier ve 0,85 contrast; `koyu` için 1,2 ve 1,85; olumsuzlanan
`yeterli` için 1,85 contrast uygulanıyor. Sayısal işlem tanımlanan kurallarla
uyumlu. Koşullu olumsuz bölümün ağırlıklandırılması genel etiketi değiştirmiş;
bu örnek bir aritmetik bug değil, ağırlık politikasının davranış örneğidir.

### ID 98 — parser parçalanması olumsuzluğu kaybettiriyor

`Rengi çok koyu 1 numara olmasına rağmen. Tavsiye edebileceğim gibi değil`
— gold: negative.

Kaydedilmiş Stanza çıktısında ikinci cümle `Tavsiye edebileceğim gibi deği`,
üçüncü cümle yalnız `l`. `deği` NOUN/nsubj, `l` PUNCT olarak analiz edilmiş.
Bu yüzden `değil` olumsuzluğu hiçbir dependency kolunun `tavsiye` kapsamına
girmemiş. DP→H positive, LW neutral çıkmış.

HisNet ve H→DP'nin negative demesi olumsuzluğu doğru çözmelerinden kaynaklanmıyor:
`rağmen` maddesi −0,5 ve `numara` maddesi −0,125 katkı veriyor. Doğru son etiket,
tek başına doğru dilsel çözümleme kanıtı değildir.

### ID 148 — yanlış kapsam aktarımı için hedefli inceleme adayı

`İi değil iade edeceğim` — gold: negative.

HisNet `iade` kökünden −0,125 alıp negative diyor. H→DP, `değil`i `iade`ye
uygulayıp skoru +0,125'e çeviriyor ve positive diyor. Kaydedilen Stanza çıktısı
da bozuk: `değil` CCONJ/root, `iade` onun obj bağımlısı; `edeceğim` ise `edeceği`
ve ayrı cümlede `m` olarak bölünmüş.

Bu, cümle sınırı kontrolüyle yakalanmayan, yanlış parse üzerinde kapsam aktarımı
örneğidir. Parser hatası ile resolver'ın bozuk parse'a karşı davranışının payı
ayrı bir küçük testte incelenmeli. Yalnız bu çıktıdan saf uygulama bug'ı sonucu
çıkarılmadı ve 500'lük sonuçlar sonradan değiştirilmedi.

## Bug kontrollerinin anlamı

2.000 sınıflandırmada teknik başarısızlık veya retry/recovery yok. Otomatik
kontrollerde 0 ihlal: skorların sonlu ve sıfırdan küçük olmaması, kapsam toplamları,
eşik–etiket uyumu, başlangıç HisNet skorları, LW temel DP skorları, bağımsız
H/H→DP/LW morfoloji tutarlılığı, global token ID'leri, yinelenen olumsuzluk ID'leri
ve cümleler arası olumsuzluk taşması. Dependency kollarında toplam 3.160 kapsam
kaydı var; eşleşmeyen kapsam kaydedilmedi.

Bu, tüm dilsel kararların doğru olduğu anlamına gelmez. Yukarıdaki ID 98 ve
148 gibi örnekler teknik olarak başarılı çalışıp anlam bakımından hatalı sonuç
verebilir. Önce bunları ayrı teşhis testlerine dönüştürmek, sonra doğrulanan
uygulama sorunlarını ele almak daha açıklanabilir bir sonraki adımdır.

## Süre neden uzun?

Bağımsız sınıflandırma süreleri HisNet 21,51; DP→H 24,20; H→DP 24,13;
LW H→DP 23,71 dakika. Toplam yaklaşık 93,54 dakika; başlatma ve disk yazımı hariç.
Kayıtlar yaklaşık 13:38–15:13 Türkiye saati arasında üretildi.

500 yorumun 186'sında robust/fallback morfoloji var. HisNet'te bu grubun toplam
süresi 1.285,49 saniye; kalan 314 yorumun toplamı yalnız 4,89 saniye.
Yani sürenin %99,62'si 186 yorumda toplanıyor. Bu bir grup karşılaştırmasıdır;
uzunluk ve yazım hataları gibi farkları kontrol eden nedensel profil değildir.

Kod her yeni OOV sözcük için izole bir morfolojik çözümleyici kuruyor ve adayları
önbelleğe alıyor. Bu mekanizma mevcut davranışın parçası olarak korunmuştur.
Mevcut sürelerden LW'nin kesin daha hızlı olduğu sonucu çıkarılmamalı; koşu sırası,
önbellek ve sistem yükü kontrol edilmedi.

## Kayıtlar

- [Ayrıntılı sayısal rapor](report.md)
- [Accuracy / macro-F1 grafiği](performance_500.png)
- [Süre grafiği](runtime_500.png)
- [500 yorumun dört modelde karşılaştırması](comparison_500.csv)
- [86 ayrışan yorum](disagreements_500.csv)
- [Kapsam kayıtları](scope_details_500.csv)
- [İncelenen örneklerin tam ara çıktıları](diagnostic_cases.json)

Bu örneklem önceden kullanılan bir geliştirme örneklemidir. Burada saptanan
sorunlara göre değişiklik yapılırsa aynı set regresyon için kullanılabilir;
bağımsız nihai performans iddiası için ayrı bir test seti gerekir.

Aynı çalıştırma komutu tamamlanmış kayıtta yeniden çalıştırılarak devam davranışı
da doğrulandı: dört sistem de `complete, skipped` yazdı. Sınıflandırma girişimleri
500'er olarak kaldı; yeni model kurulmadı ve tahmin üretilmedi. Yalnız raporlar
kayıtlı veriden yeniden oluşturuldu.
