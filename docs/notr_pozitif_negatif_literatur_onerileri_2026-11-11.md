# Nötr, pozitif ve negatif sınıflar için araştırma ve yöntem önerileri

**Tarih:** 11 Kasım 2026  
**Proje:** TurkishSentimentComparison  
**Durum:** Karar vermek için hazırlanmış araştırma notu. Hiçbir öneri uygulanmadı; model çalıştırılmadı, eşik veya sözlük değiştirilmedi.

## 1. Karar özeti

**Evet, denenmeye değer yöntemler var.** Bizim durumda ilk iki aday, nötr kararını tek bir puan farkından daha fazla bilgiyle vermek ve pozitif/negatif ipuçlarının sözlükte kaybolmasını azaltmak. Yeni bir eğitilmiş model eklemek istenirse BERTurk, doğrudan Türkçe üç sınıflı deneyle desteklenen güçlü bir aday.

Bu rapor 2024–2026 yayınlarına odaklanan hedefli bir taramadır; sistematik derleme veya literatürün tamamını kapsama iddiası değildir. Yayıncı sayfaları, ACL Anthology, yazarların makaleleri ve veri depoları incelendi. Yayın sonucu ile **bizim projeye yönelik uyarlama önerisi** ayrı tutuldu. Başka veri kümelerindeki başarılar bizim projenin beklenen skoru değildir.

| Kod | Denenebilecek seçenek | Uygun yöntemler | Ana hedef | Kaynak ve önerinin niteliği | Öncelik / maliyet |
|---|---|---|---|---|---|
| **A** | Nötr kararını kalibre etmek; ardından küçük bir karar modeli | Dört klasik yöntem | Nötr precision, recall ve F1 | K2–K3'ten esinlenen proje uyarlaması | **İlk sıra / düşük–orta** |
| **B** | POS bilgisiyle sözlük biçimi eşleme ve olumsuzluk etkileşimi | Dört klasik yöntem | Pozitif/negatif kanıtı geri kazanmak | Yerel hata kanıtı; K4 ve K6 ile ilişkili | **İlk sıra / orta** |
| **C** | Açık sınıf tanımı ve geri bildirimli örnek seçimi | LLM, DP→LLM | Nötr sınırı ve ince duygu ayrımları | K5; iki ayrı prompt varyantı | **LLM tarafında ilk sıra / düşük–orta** |
| **D** | Sözdizimiyle çözümlenmiş kısa duygu ipuçları | Yeni HisNet+DP→LLM kolu | Pozitif/negatif, olumsuzluk | K6'ya dayalı uyarlama | İkinci sıra / orta |
| **E** | Cümlecik/hedef düzeyinde birleştirme; LW bileşenlerini ayırma | Özellikle HisNet→DP ve LW | Karma görüş, yanlış ağırlıklandırma | Yerel kanıt + K2/K4'ten uyarlama | İkinci sıra / orta |
| **F** | Üç sınıflı BERTurk eğitimi | Yeni, denetimli karşılaştırma yöntemi | Üç sınıf birlikte, özellikle nötr | **K1: doğrudan Türkçe deney** | Yeni model istenirse yüksek / orta–yüksek |
| **G** | SentiTurkNet ile kontrollü sözlük karşılaştırması | Dört klasik yöntemin yeni varyantları | Pozitif/negatif kapsamı ve anlamı | K4: Türkçe, fakat ikili hedef sınıflandırma | İkinci sıra / orta |
| **H** | Karşıt örneklerle veri artırma ve contrastive learning | Eğitilecek BERTurk veya benzeri model | Zor pozitif/negatif ayrımı ve genelleme | K7; nötre aktarımı henüz hipotez | Sonraki aşama / yüksek |

**Benim tercih sıram:** Klasik yöntemlerde **B → A**, LLM tarafında önce **C'nin yalnız sınıf tanımı içeren sürümü**. B puan dağılımını değiştireceği için son eşik ayarı B'den sonra yapılmalı. A'nın mevcut puanlarla tek başına denemesi de ayrı bir karşılaştırma olabilir. F, mevcut altı yöntemin yerine sessizce geçirilmemeli; yeni bir denetimli yöntem olarak eklenmeli.

Maliyetler ölçülmüş çalışma süresi değil, geliştirme ve veri ihtiyacına ilişkin göreli değerlendirmedir. Bu rapor hiçbir seçenek için yüzde kazanç vaat etmez.

## 2. Projede şu anda nerede kaybediyoruz?

Son tamamlanmış büyük klasik koşu, her sınıftan 1000 örnek içeren 3000 yorumluk geliştirme deneyidir. Aşağıdaki sayılar yeni bir çalıştırmadan değil, kayıtlı [sayısal rapordan](../results/trsav1_classical_3000_20261101_parallel/report.md) alınmıştır.

| Yöntem | Macro-F1 % | Nötr precision % | Nötr recall % | Nötr F1 % | Pozitif F1 % | Negatif F1 % |
|---|---:|---:|---:|---:|---:|---:|
| HisNet | 46,57 | 38,28 | 19,10 | 25,48 | 57,58 | 56,66 |
| DP→HisNet | 48,11 | 36,45 | 21,80 | 27,28 | 59,10 | 57,94 |
| HisNet→DP | 48,20 | 39,27 | 19,40 | 25,97 | 59,00 | 59,62 |
| LW HisNet→DP | 47,52 | 37,65 | 18,30 | 24,63 | 59,20 | 58,72 |

**Üç sınıfın yakalama oranı (recall):** O sınıftaki doğru tahmin sayısının, gerçekte o sınıfa ait yorum sayısına bölünmesidir. Her sınıfta 1000 yorum vardır. Aşağıdaki değerler 11 Kasım 2026'da [satır bazındaki kayıtlı tahminlerden](../results/trsav1_classical_3000_20261101_parallel/comparison_3000.csv) yeniden hesaplandı ve ana raporla eşleşti; modeller yeniden çalıştırılmadı.

| Yöntem | Pozitif yakalama | Negatif yakalama | Nötr yakalama |
|---|---:|---:|---:|
| HisNet | 680/1000 = **%68,00** | 606/1000 = **%60,60** | 191/1000 = **%19,10** |
| DP→HisNet | 698/1000 = **%69,80** | 591/1000 = **%59,10** | 218/1000 = **%21,80** |
| HisNet→DP | 693/1000 = **%69,30** | 643/1000 = **%64,30** | 194/1000 = **%19,40** |
| LW HisNet→DP | 698/1000 = **%69,80** | 633/1000 = **%63,30** | 183/1000 = **%18,30** |

Bu örneklemde pozitif yakalamada DP→HisNet ve LW HisNet→DP eşit; negatif yakalamada HisNet→DP en yüksek değeri veriyor. Bunlar recall sıralamalarıdır, genel başarı veya istatistiksel üstünlük iddiası değildir. Örneğin HisNet→DP gerçek pozitiflerin %69,30'unu yakalarken, pozitif dediği yorumların %51,37'si gerçekten pozitiftir; ikinci oran precision'dır.

HisNet→DP, gerçek 1000 nötr yorumun 337'sine negatif, 469'una pozitif diyor. Yani nötr örneklerin **806'sı kaçıyor**. Nötr precision da düşük olduğundan yalnız daha fazla nötr üretmek yeterli değil. Bu koşuda LLM ve DP→LLM yok; onların güncel büyük örneklem performansını bu tablodan çıkaramayız.

Kod ve kayıtlı [hata incelemesi](../results/trsav1_classical_3000_20261101_parallel/findings.md) dört somut mekanizma gösteriyor:

1. **Karar tek fark üzerinden veriliyor.** [HisNetClassifier](../src/hisnet/hisnet_classifier.py) pozitif toplamdan negatif toplamı çıkarıyor; ±0,05 aralığını nötr kabul ediyor. Düşük kanıt, iki güçlü karşıt görüş ve eksik sözlük kapsamı aynı küçük farka düşebilir.
2. **Fiil kökü ile sözlük maddesi uyuşmuyor.** Kayıtlı incelemede `beğen` ve `sev` bulunamazken `beğenmek` ve `sevmek` bulunuyor. Buna karşılık `bayılmak` sözlükte negatif; “ürüne bayıldım” kullanımının anlamı farklı. Salt mastar ekleme yeterli değil.
3. **Kanıt yokluğu nötrlükle karışıyor.** HisNet'te P=N=0 olan 350 yorumun 218'i gerçekte polar; DP→HisNet'te 484 yorumun 308'i polar. Bu örneklere yalnız eşik ayarıyla yeni bilgi eklenmez.
4. **LW bazen eksik bilgiyi büyütüyor.** Düz HisNet→DP'ye göre 29 hata düzelirken 45 doğru karar bozulmuş. Karşıtlık ve niteleyici işlemleri ayrı incelenmeli; her katsayının yararlı olduğu varsayılmamalı.

Buradaki “pozitif/negatif skoru iyileştirme” iki şeyi kapsıyor: P ve N'nin daha doğru kanıtlardan oluşması ve sonuçta ilgili sınıfların precision/recall/F1 değerlerinin yükselmesi. **P/N büyüklüğünün artması tek başına başarı değildir; bunlar sınıf olasılığı değildir.**

## 3. Nötr sınıfı için gerekli ayrım

Bizim etiket uzayımız yine üç sınıf kalabilir; fakat hata incelemesinde nötrün farklı nedenleri ayrılmalı:

| Tanı grubu | Açıklama | Temsili örnek; yeni gold etiketi değildir |
|---|---|---|
| Nesnel / görüş içermeyen | Özellik veya olay bildirimi | “Kutuda iki parça var.” |
| Zayıf değerlendirme | Belirgin olumlu/olumsuz yargıya ulaşmayan görüş | “İdare eder.” |
| Karma görüş | Farklı yönlerde olumlu ve olumsuz değerlendirme | “Temizliği güzel ama kokusu kötü.” |
| Yetersiz kanıt / kapsam | Sistem duygu ipucunu yakalayamamış | Sözlükte bulunmayan bir övgü fiili |

**Son satır bir nötr anlam türü değildir; modelin bilgi eksikliğidir.** Benzer şekilde karma görüşlerin tamamı otomatik nötr değildir: baskın yargı veya veri kümesinin etiketleme kuralı belirleyici olabilir. Yerel ID 2 karma görüşün nötr etiketlendiği bir örnektir, bütün karma yorumlar için kural değildir.

2026 tarihli subjectivity çalışması, öznel görüşün de nötr olabileceğini açıkça ayırıyor. Bu nedenle “öznel değilse nötr, öznel ise mutlaka pozitif/negatif” biçiminde sert bir kapı önermiyorum. [K2 — Chong ve arkadaşları](https://link.springer.com/article/10.1007/s44196-026-01153-x).

## 4. Seçeneklerin ayrıntısı

### A — Nötr kararını kalibre etmek ve ayrı karar katmanı denemek

**Literatür dayanağı:** K2, subjectivity tespitini yardımcı görev olarak duygu sınıflandırmasıyla birlikte öğreniyor. K3 ise genel kalibrasyon iyi görünürken belirli sınıflarda sistematik, yüksek güvenli hatalar kalabildiğini gösteriyor. İkisi de bizim HisNet çıktılarımıza hazır bir eşik veya kanıtlanmış kazanç vermiyor. [K2](https://link.springer.com/article/10.1007/s44196-026-01153-x), [K3](https://aclanthology.org/2026.uncertainlp-main.41/).

**Projeye önerdiğim uyarlama, iki ayrı deneydir:**

- **A1 — Karar eşikleri:** Mevcut P−N için negatif ve pozitif sınırları ayrı seçmek; simetrik ±0,05 tabanıyla karşılaştırmak. Ek olarak duygu eşleşmesi sayısına göre normalize edilmiş farkı ayrı varyant olarak değerlendirmek. Eşikler yalnız geliştirme/doğrulama verisinde seçilir.
- **A2 — Küçük karar modeli:** P, N, toplam kanıt, puan farkı, duygu eşleşmesi sayısı, eşleşmeyen aday oranı, metin uzunluğu ve varsa cümleciklerde karşıt görüş göstergeleriyle düzenlileştirilmiş üç sınıflı lojistik regresyon denemek. P=N=0 için açık bir gösterge kullanmak. Başlangıçta daha karmaşık bir model gerekmiyor.

Nötr-vs-polar ve ardından pozitif-vs-negatif ayrımı da ayrıca karşılaştırılabilir; ancak ilk aşamadaki hatanın ikinci aşamada düzeltilemeyeceği hesaba katılmalı. A2'nin üç sınıflı sürümü iyi bir ilk kontroldür.

**Neden işe yarayabilir?** Aynı net farka sahip “az kanıt” ve “güçlü karşıt kanıt” örnekleri ayrılabilir. **Sınırı:** Özelliklerde metnin anlamı yoksa örtük duygu ve sözlük dışı övgü yine çözülemeyebilir. P ve N'nin ikisinin de pozitif olması, tek başına karma görüş kanıtı değildir; aynı sözlük maddesi iki skora da katkı verebilir.

**Deney kimliği:** A1 etiketli veriyle ayarlanmış bir karar kuralıdır. A2 öğrenilmiş bir katman eklediği için artık salt sözlük yöntemi değildir. Her dört sisteme aynı özellik tanımı ve aynı ayarlama bütçesi verilmeli; parametreler yöntem bazında seçilecekse bu açıkça belirtilmeli.

**Başarı koşulu:** Nötr recall ile birlikte nötr precision/F1 ve macro-F1 yükselmeli; pozitif/negatif F1 kaybı ayrıca raporlanmalı. Yalnız nötr recall artışı yeterli değildir.

### B — Sözlük biçimini ve olumsuzluk etkileşimini düzeltmek

**En güçlü gerekçe yerel kanıt:** `beğen`/`sev` biçim uyuşmazlığı kayıtlı olarak doğrulanmış. K4'ün Türkçe hedef sözcükleriyle çalışması ve K6'nın sözcük türü ile olumsuzluk içeren ifade ipuçlarını kullanması bu yönü destekliyor; fakat bizim için belirli bir kök eşleme algoritmasının başarısını kanıtlamıyor. [K4](https://www.mdpi.com/2076-3417/16/13/6612), [K6](https://aclanthology.org/2024.findings-acl.286/).

**Öneri:** Morfolojik çözümleyiciyi başlangıçta değiştirmeden, seçilmiş kök ve POS bilgisinden kontrollü sözlük arama adayları üretmek. Fiil olduğu doğrulanan biçimlerde kök/mastar uyuşmasını çözmek; isim–fiil eşseslilerini ve çok anlamlı kullanımları ayrı incelemek. İlk aşamada yeni sözcük puanları icat etmek yerine mevcut sözlük maddesini doğru kullanmayı hedeflemek.

**Kritik etkileşim:** [Mevcut resolver](../src/dependency/sentiment_scope_resolver.py), token'ın kendi morfolojik olumsuzluğunu otomatik skor takasına çevirmeyen muhafazakâr politikayı içeriyor. `beğenmedim → beğenmek` eşleşmesini açıp olumlu puanı doğrudan toplamak hatalı pozitifleri artırabilir. Bu nedenle üç ayrı varyant gerekir:

1. Yalnız biçim eşleme.
2. Yalnız kapsamı doğrulanmış ekli olumsuzluk bileşimi.
3. İkisinin birleşimi; aynı olumsuzluğun iki kere uygulanmaması kontrolü.

“Beğendim / beğenmedim”, “bayıldım / bayılacak gibi oldum”, “kötü değil / iyi değil” gibi karşı örnekler tanı setine alınmalı. Bunlar genel dönüşüm kurallarını denemek içindir; sonuçlara göre sözcüğe özel gold ezberleyen puanlar eklenmemeli.

**Uygunluk:** Ortak sözlük arayüzü dört klasik varyantta aynı olmalı. DP→HisNet'in aday seçimi yine kendi yöntemine ait kalmalı. Temel HisNet'e de olumsuzluk modülü eklenecekse bu ayrıca adlandırılmış yeni bir varyanttır.

**Beklenti ve sınır:** Pozitif/negatif kapsamı artırmak için en somut yerel aday. Ancak yeni eşleşmelerin ne kadarının doğru anlam ve kapsam taşıdığı ölçülmeden F1 kazancı söylenemez. Yeni P/N dağılımında A yeniden ayarlanmalıdır.

### C — LLM'de sınıf tanımı ve geri bildirimli örnekler

**Literatür:** Feedback-ICL, aday örnek havuzundaki önceki tahminleri ve doğruluk geri bildirimini kullanarak örnekli prompt oluşturuyor. Dokuz duygu analizi veri kümesinde geleneksel ICL'ye göre iyileşme raporluyor; bizim Türkçe veri/model bileşimimizde sınanmamış. [K5 — Xu ve arkadaşları, ACL Findings 2024](https://aclanthology.org/2024.findings-acl.232/).

**Projeye uyarlama:**

- **C1 — Örneksiz tanım:** [Mevcut LLM prompt'u](../src/llm/ollama_zero_shot_classifier.py) üç etiketi listeliyor; nötrün görev özelindeki kapsamını açıklamıyor. Etiketleme politikasına göre nesnel ifade, zayıf değerlendirme, karma görüş ve baskın yargı ayrımını kısa metinle tanımlamak. Aynı çekirdek tanım LLM ve DP→LLM'ye verilmelidir. Bu, K5'in tam yöntemi değil, önce denenmesini önerdiğim daha ucuz bir kontroldür.
- **C2 — Örnek ve geri bildirim:** Yalnız eğitim/geliştirme havuzundan, nötr ile zayıf pozitif/negatif sınırını gösteren ve önceki model hatalarını açıklayan örnekler seçmek. Önce eşit sayıda sıradan few-shot örnekle karşılaştırmak; sonra geri bildirimin ek etkisini ölçmek.

**Deney kimliği:** C1 etiketli gösterim içermezse zero-shot çıkarım olarak kalır; geliştirme etiketleriyle prompt seçilmişse bu seçim yine raporlanır. **C2 few-shot yöntemdir**, zero-shot adıyla raporlanamaz. Test örneğinin gerçek etiketi veya test hatası prompt seçimine girmez.

**Maliyet ve sınır:** C1 düşük, C2 daha yüksek token maliyetlidir. Mevcut DP→LLM bağlam bütçesi özellikle uzun metinlerde zaten sınırlı; örnek eklemek başarısız girdileri artırabilir. Ortak örnekler, bağlam sınırı ve kapsama oranı kontrol edilmelidir. “Emin değilsen nötr de” kuralı önerilmez; belirsizlik nötrlük değildir.

### D — LLM'ye kısa, çözümlenmiş sözdizimsel duygu ipuçları

**Literatür:** Kanayama ve arkadaşları, sözdizimi ve sözlükten elde edilen olumlu/olumsuz sözcük veya ifade ipuçlarını LLM prompt'una ekliyor. Çalışma 15 dilde **ikili** sınıflandırmayı değerlendiriyor; üç sınıflı Türkçe nötr başarısının kanıtı değil. [K6 — ACL Findings 2024](https://aclanthology.org/2024.findings-acl.286/).

**Öneri:** Tam parse'a alternatif bir bilgi sunumu olarak “ifade → hedef → kapsamı çözülmüş polarite” kayıtlarını denemek. Örneğin çözümlenmiş “tavsiye etmiyorum” ifadesini, çıplak `tavsiye` sözcüğünün olumlu sözlük puanından ayrı sunmak. LLM ham yorumu da görmeli; ipuçları kesin cevap olarak verilmemeli.

**Mimari etkisi:** Şu anki DP→LLM, HisNet puanı kullanmıyor. HisNet tabanlı ipuçları eklenirse bu kol **HisNet+DP→LLM** olarak ayrıca raporlanmalıdır. Sadece parse kısaltma deneyiyle aynı değişiklik değildir.

**Risk:** Hatalı sözlük anlamı veya olumsuzluk kapsamı LLM'yi yanlış yöne çekebilir. B'nin güvenilirliği bu seçenek için önemlidir. Mevcut DP→LLM, kısa sözdizimi özeti ve kısa duygu ipucu varyantları ayrı karşılaştırılmalı. Kısa çıktının gerçekten daha az token kullandığı da ölçülmelidir.

### E — Karma görüşleri cümlecik/hedef düzeyinde birleştirmek

**Öneri:** Bir yorumun tüm puanlarını hemen toplamak yerine önce cümlecik ve hedef düzeyinde katkıları tutmak; son kararda güçlü karşıt görüşlerin farklı hedeflerden gelip gelmediğini kullanmak. Bu, ABSA çalışmalarından esinlenen **bizim uyarlama hipotezimizdir**; K2/K4 sonuçları doğrudan yorum düzeyindeki nötr kazancını göstermez. [K2](https://link.springer.com/article/10.1007/s44196-026-01153-x), [K4](https://www.mdpi.com/2076-3417/16/13/6612).

Yerel örnekte “temizleyici özelliği güzel ama kokusu kötü” yorumu nötr etiketliyken `ama` sonrası negatif katkı LW ile büyüyor. Farklı özelliklere yönelmiş iki görüşün korunması, son karar katmanına mevcut net farktan daha fazla bilgi verebilir.

**İlk deneme:** Düz HisNet→DP; yalnız modifier işlemi; yalnız contrast işlemi; ikisi birlikte. [LW uygulaması](../src/hisnet/literature_weighted_hisnet_dependency_classifier.py) katsayıların yanında katkı bastırma da yaptığı için bunlar yalnız “katsayı değişti” diye sunulmamalı. Ardından cümlecik bilgisi A2'ye eklenerek ek faydası ölçülebilir.

**Sınırı:** Her `ama` karma/nötr değildir; güçlü son yargı gerçekten baskın olabilir. Parse hataları hedefleri yanlış birleştirebilir. Yeni ağırlıklar veriden öğrenilirse “literatürde sabit ağırlık” varyantından ayrı tutulmalıdır.

### F — Üç sınıflı BERTurk: doğrudan Türkçe kanıtı en yakın yeni model

Karakaş ve Şimşek'in 28 Haziran 2026 arXiv sürümü, 6.367 işlenmiş Türkçe e-ticaret yorumunda ortak 1.274 örneklik test kullanıyor. BERTurk Cased 128k için doğruluk **%83,7**, weighted-F1 **%83,4**, nötr recall **%60,8**; Qwen3:32B için doğruluk **%77,3**, nötr recall **%31,2** raporlanıyor. ArXiv kaydı SIU 2026 kabulünü belirtiyor; burada incelenen belge arXiv sürümüdür. [K1 — makale ve deney ayrıntıları](https://arxiv.org/html/2606.29614v1).

**Bize anlamı:** Nötrü görev etiketleriyle öğrenen Türkçe bir model eklemek güçlü bir aday. Bu rakamlar bizim TRSAv1 koşumuzla doğrudan karşılaştırılamaz; veri, dağılım, ön işleme ve eğitim farklı. Çalışma tek bir prompt düzenini değerlendiriyor; bütün LLM yöntemlerine üstünlük kanıtlamıyor. Aynı test üzerinde kullanılan yaklaşık oran testleri de eşleştirilmiş örnek analizinin yerini tutmuyor.

**Öneri:** Önce üç sınıflı TF-IDF+lojistik regresyon gibi ucuz bir denetimli kontrol, sonra aynı ayrımla BERTurk. 32k ve 128k sözcük dağarcığı sürümlerinden hangisinin kullanılacağı geliştirme bütçesine göre sabitlenmeli. Eğitimsiz HisNet/LLM kollarıyla birlikte raporlanırken eğitim verisi avantajı açıkça yazılmalı.

**Sonraki geliştirme:** Gerekirse BERTurk'a K2'deki gibi yardımcı subjectivity görevi eklenebilir. Bu ek görev için uygun etiket gerekir; mevcut üç duygu etiketinden “nötr=nesnel” varsayımıyla sahte subjectivity gold üretilmemeli. Önce sade modelin kazancı görülmelidir.

### G — SentiTurkNet ile kapsam/puan karşılaştırması

2 Temmuz 2026 tarihli Aksaya–Gülseçen çalışması, Türkçe okul yorumlarında hedef düzeyinde SentiTurkNet ile iki SentiWordNet düzenini karşılaştırıyor. SentiTurkNet weighted-F1=0,887 ile önde; fakat dört nötr örnek değerlendirme dışında, HisNet karşılaştırması yok. Uygulama olumsuzluk ve anlam ayrıştırmasını da tam çözmüyor. [K4 — makale](https://www.mdpi.com/2076-3417/16/13/6612).

Bu yüzden çalışma **HisNet yerine geçince kazanırız** veya **nötr sorununu çözer** sonucunu desteklemiyor. Ayrıca tablodaki toplam 1000 yorum ile tek/çok hedefli yorum sayılarının toplamı arasında raporlama tutarsızlığı bulunuyor; bildirilen yüksek değeri temkinli okumak gerekir.

**Öneri:** Aynı morfoloji ve aynı aday seçiminde yalnız sözlük kaynağını değiştiren bir deney; ikinci adımda yalnız HisNet'te bulunamayan adaylar için ikincil sorgu. Aynı token'ın iki kere sayılmaması ve sözlük puan ölçeklerinin uyumu kontrol edilmeli. Her varyantın eşiği aynı doğrulama bütçesiyle seçilmeli.

**Sınırı:** İkincil sözlük de yanlış anlamı seçebilir. Yalnız “daha çok eşleşme” başarı değildir; kazandırılan eşleşmelerin duygu doğruluğu ve sınıf F1 etkisi ölçülmelidir.

### H — Zor karşıt örneklerle eğitim ve contrastive learning

SALAD, yapıyı gözeten artırılmış örnekleri ve LLM ile üretilen karşıt örnekleri contrastive learning içinde kullanıyor. Duygu deneyindeki kaynak sınıflar pozitif/negatif; Türkçe üç sınıflı nötr başarısı sınanmıyor. [K7 — Bae ve arkadaşları, NAACL 2025](https://aclanthology.org/2025.naacl-long.634/).

**Projeye öneri:** F seçilirse, yalnız eğitim verisinden anlamı koruyan ve anlamı kontrollü değiştiren örnekler üretmek. Olumsuzluk, “idare eder” gibi zayıf değerlendirme ve bir yönü olumlu diğer yönü olumsuz yorumlar hedeflenebilir. Türkçe ekleri veya bağlaçları rastgele silmek yerine örneklerin etiket tutarlılığı denetlenmeli.

**Beklenti:** Modelin belirli kelimeleri ezberlemek yerine bağlam farkını öğrenmesine yardımcı olabilir. **Sınır:** Üç sınıfa genişletme bizim hipotezimizdir; sentetik nötrlerin yanlış etiketlenmesi sorunu büyütebilir. Aynı özgün örneğin bütün türevleri aynı veri bölümünde kalmalı. Eğitim gerektirdiğinden mevcut kural tabanlı yöntemlere doğrudan eklenebilecek bir puan düzeltmesi değildir.

## 5. İlk aşamada önermediğim yollar

- **Yalnız nötr eşiğini genişletip başarı ilan etmek:** Pozitif/negatif örnekleri nötre çekerek recall artırılabilir; precision ve diğer sınıflar zarar görebilir. A1 karşılaştırma olarak değerli, tek başına çözüm garantisi değildir.
- **Düşük güveni nötr etiketi saymak:** K3'ün kalibrasyon bulgusu bir ölçüm uyarısıdır; nötr sınıfı belirsizlik sınıfına dönüştürmez. [K3](https://aclanthology.org/2026.uncertainlp-main.41/).
- **Salt temperature scaling ile F1 artışı beklemek:** Tek pozitif sıcaklıkla bütün logitleri ölçeklemek argmax sınıfını değiştirmez. Olasılık kalitesi ile karar başarısı farklıdır; karar kuralı ayrıca değişmedikçe etiketler aynı kalır.
- **SMOTE, focal loss veya class weighting'i otomatik ilk çözüm saymak:** İncelenen 3000 örnek zaten sınıf bakımından dengeli; mevcut klasik sistemler eğitim kaybı kullanmıyor. Eğitimli bir modele geçilirse eğitim dağılımı ayrıca incelenebilir.
- **“Prompt kalibrasyonu” başlıklı her makaleyi prompt'a bir cümle eklemek sanmak:** He ve arkadaşlarının 2024 yöntemi, null-input dağılımlarıyla modelin bias parametrelerini güncelliyor. Mevcut etiket döndüren Ollama akışına doğrudan takılan bir metin düzeltmesi değil. Bu nedenle ilk listeye alınmadı. [K8](https://aclanthology.org/2024.findings-emnlp.741/).
- **Dört klasik sistemin basit çoğunluk oylamasını ilk çözüm seçmek:** Yerel koşuda 1330 yorumda dördü de yanlış. Ortak sözlük ve kapsam hatalarını yalnız oylama gideremez.
- **Nötrleri çıkarıp yeni skoru üç sınıflı iyileşme diye sunmak:** Bu farklı bir görevdir. Pozitif/negatif altküme analizi ek tanı olabilir; ana üç sınıflı ölçüm korunmalı.

## 6. Seçimden sonra nasıl ölçmeliyiz?

Bu bölüm **gelecekte yapılabilecek deney planıdır; hiçbir adım bu araştırmada çalıştırılmadı.**

### Veri ve etiket protokolü

1. **Nötr tanımını sabitle:** Model hatalarına bakarak gold değiştirmeden, geliştirme verisinde nötrleri nesnel/zayıf/karma ve belirsiz etiket grupları açısından incele. Mümkünse iki değerlendiriciyle küçük bir etiket denetimi yap; anlaşmazlığı kaydet.
2. **Geliştirme, seçim ve son değerlendirmeyi ayır:** 1200 ve 3000 örnekler incelenmiş geliştirme verisidir. [Önceki veri geçmişi denetimi](paper_readiness_2026-10-31.md), TRSAv1 ve Beyazperde'nin daha önce tam koşularda kullanıldığını gösteriyor. Yalnız bu 4200 metni dışlamak, kalanları kendiliğinden hiç görülmemiş test yapmaz.
3. **Bağımsızlık iddiası gerekiyorsa yeni ve ayrılmış veri kullan:** Yeni toplanmış/etiketlenmiş bir test veya geçmiş kullanımı denetlenmiş başka kaynak. Aynı/çok benzer metinler, çelişik etiketli kopyalar ve sentetik türevler aynı bölümde tutulmalı. Mevcut benchmark yeniden kullanılacaksa bu açıkça geliştirme sonrası yeniden değerlendirme diye yazılmalı.
4. **Alternatif veri adayı:** 2025 FSMTSAD, yazar deposuna göre 15.853 örnek ve üç sınıf içeriyor; doğal metin yanında LLM üretimi ve veri artırımı da var. Nötr için hem görüşsüz hem dengeli karşıt görüşlü örnekler tanımlanmış. Örtüşme ve türev denetimi yapılmadan bağımsız final test diye kabul edilmemeli; doğal ve sentetik kısımlar ayrı raporlanabilir. Bu raporda veri indirilmedi veya örtüşme ölçülmedi. [K9 — makale](https://www.mdpi.com/2076-3417/15/4/2062), [yazarların veri açıklaması](https://github.com/kevserbusrayildirim/FSMTSAD).

### Karşılaştırmalar

| Deney | Sabit tutulacaklar | Değişecek tek unsur |
|---|---|---|
| B1 | Morfoloji, parse, eski karar kuralı | Sözlük biçimi eşleme |
| B2 | Eski biçim eşleme, diğer kapsam kuralları | Ekli olumsuzluk bileşimi |
| B1+B2 | Aynı örnekler ve kaynaklar | İki bileşenin ortak etkisi |
| A1 / A2 | Seçilmiş sözlük/kapsam sürümü | Karar kuralı / öğrenilmiş karar katmanı |
| C1 | Model, örneksiz yapı, çıktı şeması | Sınıf tanımı |
| C2 | Model, örnek sayısı ve token bütçesi | Sıradan örnek seçimi / geri bildirimli seçim |
| D | Model, ham metin, çekirdek talimat | Parse veya duygu ipucunun sunumu |
| E | Sözlük ve temel kapsam sürümü | Modifier, contrast ve cümlecik bileşenleri |
| F | Eğitim/doğrulama/test ayrımları | Denetimli model ailesi |
| G | Morfoloji ve aday seçimi | Sözlük kaynağı / ikincil sorgu |
| H | Temel model ve veri ayrımları | Artırma ve contrastive loss; ayrı ve birlikte |

Önce tek bileşenli etkiler ölçülmeli, sonra birlikte kazanç kontrol edilmeli. Örneğin B ve A birlikte iyi sonuç verirse katkının hangisinden geldiğini ayrı satırlar göstermelidir. Her sürüm yeni sonuç dizinine kaydedilmeli; mevcut taban koşuları korunmalıdır.

### Başarı ölçütleri

- Ana ölçüm **macro-F1**; yanında nötr/pozitif/negatif için ayrı precision, recall ve F1.
- Nötrü polar sınıflara, polar sınıfları nötre ve pozitif ile negatifi birbirine karıştırma sayıları.
- Sıfır kanıt, eşleşmeyen fiil, ekli olumsuzluk, açık olumsuzluk, karma görüş ve uzun metin dilimleri.
- Aynı örneklerde düzeltilen/bozulan kararlar; macro-F1 ve sınıf-F1 farkları için eşleştirilmiş bootstrap güven aralıkları. Birden çok karşılaştırmada çoklu deneme etkisi dikkate alınmalı. McNemar kullanılacaksa bunun F1 değil doğru/yanlış farkını sınadığı belirtilmeli.
- Eğitilen modellerde birden fazla sabit tohum; LLM tarafında prompt/model sürümü, çıktı başarısızlıkları, token maliyeti ve kapsama oranı.
- Olasılık üretilirse sınıf bazlı kalibrasyon; mevcut HisNet toplamlarını veya LLM'nin kendisinin söylediği güveni doğrudan kalibre olasılık saymamak.

**Örnek seçim kuralı:** Geliştirmede macro-F1 ve nötr F1 birlikte yükselen, pozitif/negatif F1 kaybı önceden kabul edilen sınırı aşmayan varyant seçilir. Tolerans sonuçlara bakıldıktan sonra icat edilmemeli. Tek bir nötr recall artışı veya başka veri üzerindeki yüksek doğruluk yeterli kabul edilmemeli.

## 7. Kaynakça ve kanıtın sınırı

**K1. Karakaş, S.; Şimşek, Y. (2026).** *Do We Still Need Fine Tuning? Turkish Sentiment Analysis in the Era of Large Language Models.* 28 Haziran 2026, arXiv:2606.29614; kayıt SIU 2026 kabulünü bildiriyor. [Kayıt](https://arxiv.org/abs/2606.29614), [tam metin](https://arxiv.org/html/2606.29614v1). **Türkçe, üç sınıflı; F için en doğrudan kanıt.**

**K2. Chong, W. K. ve arkadaşları (2026).** *Multitask Learning and BERT Embedding: A Comprehensive Approach to Subjectivity Detection and Aspect-Based Sentiment Analysis.* International Journal of Computational Intelligence Systems, 19, 51; 23 Ocak 2026. [Yayıncı / DOI](https://link.springer.com/article/10.1007/s44196-026-01153-x). **MEMD üzerinde aspect düzeyinde çalışma; bizim yorum düzeyine aktarım dolaylı.**

**K3. Andrade, C. M. V. ve arkadaşları (2026).** *Beyond Global Calibration: Analyzing Class-Specific Miscalibration and High-Confidence Errors in Text Classification.* UncertaiNLP 2026, Ekim, ss. 683–692. [ACL Anthology](https://aclanthology.org/2026.uncertainlp-main.41/). **Yedi duygu/konu veri kümesinde kalibrasyon analizi; doğrudan nötr iyileştirme algoritması değil.**

**K4. Aksaya, H.; Gülseçen, S. (2026).** *A Novel Lexicon-Based Approach for Sentiment Analysis in Turkish.* Applied Sciences, 16(13), 6612; 2 Temmuz 2026. [Yayıncı / DOI](https://www.mdpi.com/2076-3417/16/13/6612). **Türkçe sözlük ve hedef düzeyi için ilgili; nötr dışlanmış, HisNet kıyası yok.**

**K5. Xu, H. ve arkadaşları (2024).** *Improving In-Context Learning with Prediction Feedback for Sentiment Analysis.* Findings of ACL 2024, Ağustos, ss. 3879–3890. [ACL Anthology](https://aclanthology.org/2024.findings-acl.232/), [yazar kodu](https://github.com/HITSZ-HLT/Feedback-ICL). **C2'nin yöntem dayanağı; örnekli yaklaşım.**

**K6. Kanayama, H.; Zhao, Y.; Iwamoto, R.; Ohko, T. (2024).** *Incorporating Syntax and Lexical Knowledge to Multilingual Sentiment Classification on Large Language Models.* Findings of ACL 2024, Ağustos, ss. 4810–4817. [ACL Anthology](https://aclanthology.org/2024.findings-acl.286/). **D için ilgili; ikili görev, nötr başarısı göstermez.**

**K7. Bae, S. ve arkadaşları (2025).** *SALAD: Improving Robustness and Generalization through Contrastive Learning with Structure-Aware and LLM-Driven Augmented Data.* NAACL 2025, ss. 12724–12738. [ACL Anthology](https://aclanthology.org/2025.naacl-long.634/), [tam metin](https://aclanthology.org/2025.naacl-long.634.pdf). **H için ilgili; üç sınıflı Türkçe uyarlaması yeni deney gerektirir.**

**K8. He, K.; Long, Y.; Roy, K. (2024).** *Prompt-Based Bias Calibration for Better Zero/Few-Shot Learning of Language Models.* Findings of EMNLP 2024, Kasım, ss. 12673–12691. [ACL Anthology](https://aclanthology.org/2024.findings-emnlp.741/). **Parametre günceller; salt prompt düzenleme önerisi olarak alınmadı.**

**K9. Zümberoğlu, K. B.; Dik, S. Z.; Karadeniz, B. S.; Sahmoud, S. (2025).** *Towards Better Sentiment Analysis in the Turkish Language: Dataset Improvements and Model Innovations.* Applied Sciences, 15(4), 2062; 16 Şubat 2025. [Yayıncı / DOI](https://www.mdpi.com/2076-3417/15/4/2062), [yazarların veri deposu](https://github.com/kevserbusrayildirim/FSMTSAD). **Ek veri adayı; bağımsızlık ve doğal/sentetik ayrımı ayrıca denetlenmeli.**

## 8. Uygulama seçimi için kısa liste

| Tercih | Seçilecek kodlar | Ne elde etmeyi amaçlıyor? |
|---|---|---|
| Mevcut klasik yöntemleri iyileştir | **B ve A** | Eksik duygu kanıtını azalt, nötr sınırını daha iyi kur |
| LLM tarafında küçük bir değişiklik dene | **C1** | Etiketlerin anlamını modele açık tanımla |
| LLM'ye örnek ve dış bilgi ekle | **C2 ve/veya D** | Ayrı few-shot veya sözlük+sözdizimi hibrit yöntemi |
| LW'nin katkısını netleştir | **E** | Yararlı ve zararlı bileşenleri ayır |
| Yeni, eğitimli bir karşılaştırma ekle | **F** | Türkçe üç sınıflı güçlü bir referans oluştur |
| Daha kapsamlı araştırma yap | **G ve/veya H** | Sözlük alternatifini veya veri artırmayı sınayarak ilerle |

**Seçim bekleniyor. Bu dosya yalnız araştırma ve öneri içerir; A–H seçeneklerinden hiçbiri uygulanmadı.**
