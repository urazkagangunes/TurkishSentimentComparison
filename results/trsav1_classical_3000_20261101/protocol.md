# 3000 örneklik geliştirme koşusunun sabit protokolü

1 Kasım 2026, koşu başladıktan sonra ve tam sonuçlar görülmeden kaydedildi.
Bu dosya resmi ön kayıt veya bağımsız test seti iddiası değildir.

- Kaynak: TRSAv1_fixed.csv; 125561 temizlenmiş metin/etiket çifti.
- Örneklem: seed 42 ile 1000 negatif, 1000 nötr, 1000 pozitif; toplam 3000 benzersiz metin.
- Önceki 1200 örnekteki metinler etiketinden bağımsız dışlandı. Böylece kaynakta
  1225 satır dışlandı; yeni örneklemle önceki 1200 arasında metin örtüşmesi sıfır.
- Sınıf sırası negatif, nötr, pozitif. Önceden seçilen aynı metin rastgele sınıf
  sıralamasında yeniden görülürse atlanır; bu çekilişte böyle bir atlama gerekmedi.
- Kaynakta başka etiketle de bulunan 43 seçili metin ana sonuçta tutulacak;
  ayrıca etiket çelişkisi duyarlılık analizi verilecek.
- Yöntemler HisNet, DP→HisNet, HisNet→DP ve LW HisNet→DP. LLM ve DP→LLM yok.
- Ortak morfoloji ve parse bir kere hesaplanır; her yöntemin kararı ayrı üretilir.
  Paylaşılan girdiler kopyalanarak korunur. Her sınıfın ilk örneğinde dört bağımsız
  tam pipeline ile toplam 12 semantik çıktı karşılaştırması yapılır.
- Resolver SHA-256: `dc37555f087cb5124f72c2a3897588a69e8cf7b1310fb82a5b725a0e3535a667`.
  Diğer kod, paket ve model kaynaklarının kimlikleri manifest ve source_snapshot içindedir.
- Eşik 0,05; mevcut sözlük, morfoloji ve LW ağırlıkları sabit. Koşu sırasında
  yöntem kodu veya kaynak modeller değiştirilmez.
- Rapor: accuracy, macro-F1, sınıf bazında precision/recall/F1, karışıklık
  matrisleri, altı eşleştirilmiş exact McNemar testi ve Holm düzeltmesi,
  sözlük kapsamı, sıfır skorlar, ortak hatalar, işlem/kapsam tutarlılığı.
- Süreler paylaşılan ön işleme ve yöntem bileşimi olarak ayrılır; bağımsız
  yöntem hız karşılaştırması olarak yorumlanmaz.
- Son çekimli sıfat yüklemi düzeltmesinin etkisi, yeni örneklemin kaydedilmiş
  parse/morfoloji/sözlük sonuçlarında önceki resolver ile ayrıca karşılaştırılacak.
  Bu karşılaştırma aynı 3000 metinde eşleştirilmiştir. Önceki 1200 ve yeni 3000
  farklı metinler içerdiğinden iki koşu arasındaki skor farkı düzeltmenin etkisi sayılamaz.
- Bu koşunun sonuçlarına göre eşik veya sözlük ayarlanırsa aynı koşu artık
  geliştirme verisi olur; yeni bir doğrulama ayrımı gerekir. TRSAv1 zaten daha önce
  kullanılan bir kaynak olduğundan mevcut koşu bağımsız final paper testi değildir.
