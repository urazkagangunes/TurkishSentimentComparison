from src.common.morphology import MorphologicalRootExtractor


extractor = MorphologicalRootExtractor()

sentences = [
    "Filmi beğenmedim.",
    "Otel güzeldi ama odalar kötüydü.",
    "Bu ürünü kesinlikle tavsiye etmiyorum.",
    "Film güzel değildi.",
    "Film kötü değildi."
]


for sentence in sentences:

    print("\n" + "=" * 80)
    print("SENTENCE:", sentence)

    analyses = extractor.analyze_sentence(sentence)

    for item in analyses:
        print(
            f"{item['word']:<15}"
            f"ROOT={item['root']:<15}"
            f"ANALYSIS={item['analysis']}"
        )