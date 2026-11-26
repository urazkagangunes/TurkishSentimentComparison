# TRSAv1 fixed pilotu

TRSAv1 fixed'ten seed 42 ile her duygu sınıfından 10 yorum seçildi. Altı yerel
sistem aynı 30 farklı yorumu tamamladı. Her iki Ollama kolu
`qwen3:4b-instruct` modelini, 8192 bağlam sınırını ve üçlü batch'i kullandı.
TypeSafe API anahtarı bulunmadığı için Jev'in 30 tahmini bekliyor.
[Koşu ayarları](manifest.json) · [Ollama model kimliği](ollama_identity.json)

| Sistem | Doğru / toplam | Accuracy | Macro-F1 | Negatif F1 | Nötr F1 | Pozitif F1 |
|---|---:|---:|---:|---:|---:|---:|
| HisNet | 13/30 | %43,33 | %38,66 | %45,45 | %16,67 | %53,85 |
| DP → HisNet | 17/30 | %56,67 | %52,77 | %66,67 | %30,77 | %60,87 |
| HisNet → DP | 17/30 | %56,67 | %49,84 | %69,23 | %16,67 | %63,64 |
| Weighted HisNet → DP | 17/30 | %56,67 | %49,84 | %69,23 | %16,67 | %63,64 |
| LLM | 21/30 | %70,00 | %68,33 | %75,00 | %50,00 | %80,00 |
| DP → LLM | 20/30 | %66,67 | %63,68 | %72,00 | %42,86 | %76,19 |
| Jev 1.14.0 | 0/30 tahmin | — | — | — | — | — |

Her iki bağımlılık destekli HisNet kolu yalın HisNet'e göre dört ek yorumu
doğru sınıflandırdı. DP → HisNet 10 nötr yorumun ikisini, diğer HisNet
kolları birini doğru buldu. Ağırlıklandırma bu pilotta HisNet → DP'nin
30 etiket kararını korudu.

LLM, DP → LLM'den bir yorum daha fazla doğru sınıflandırdı. DP → LLM,
bir üçlü batch'i 8192 bağlam sınırı için yapılan ön kontrolde reddetti ve
üç yorumu tekli isteklerle tamamladı. Altı yerel sistemin nihai kapsamı
%100. Nötr sınıfta en yüksek doğru sayısı 10 üzerinden 4'tü.
[Metrikler ve hata matrisleri](summary.json) · [Satır bazında tahminler](predictions.csv)

Bir yorum doğruluğu 3,33 yüzde puan değiştiriyor. Bu küçük örneklem,
entegrasyon ve ilk hata analizi için kullanıldı. Jev, `TYPESAFE_API_KEY`
ayarlanıp aynı komuta `--systems jev` eklendiğinde aynı 30 yorumu alacak.
Jev yapılandırmasının denetimi pilot sırasında güncellendi. Yerel tahminler,
girişim kayıtları ve metrikler korunarak checkpoint kimliği doğrulandı.
[Jev kullanım notları](../../docs/resumable_evaluation.md#yedinci-sistem-olarak-jev) · [Checkpoint geçiş kaydı](pilot_runner_transition.json)
