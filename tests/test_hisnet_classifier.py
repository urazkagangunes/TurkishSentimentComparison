from src.hisnet.hisnet_classifier import HisNetClassifier


classifier = HisNetClassifier()

sentences = [
    "Film güzel.",
    "Film kötü.",
    "Otel mükemmel.",
    "Bu ürün berbat.",
    "Film güzel değildi.",
    "Film kötü değildi.",
    "Otel güzeldi ama odalar kötüydü.",
    "Filmi beğenmedim.",
    "Bu ürünü kesinlikle tavsiye etmiyorum."
]


for sentence in sentences:

    result = classifier.classify(sentence)

    print("\n" + "=" * 100)

    print("Sentence:", result["text"])
    print("Prediction:", result["prediction"])

    print("Positive Score:", result["positive_score"])
    print("Negative Score:", result["negative_score"])

    print("Tokens:", result["token_count"])
    print("Matched Words:", result["matched_word_count"])
    print("Sentiment Words:", result["sentiment_word_count"])

    print(
        "Lexicon Coverage:",
        round(result["lexicon_coverage"], 4)
    )

    print(
        "Sentiment Coverage:",
        round(result["sentiment_coverage"], 4)
    )

    print("\nMorphological analyses:")

    for item in result["morphological_analyses"]:
        print(
            f"  {item['word']:<15}"
            f" -> {item['root']:<15}"
            f" {item['analysis']}"
        )

    print("\nSentiment-bearing words:")

    for word in result["sentiment_words"]:
        print(
            f"  {word['word']:<15}"
            f"ROOT={word['root']:<15}"
            f"POS={word['positive']:<8}"
            f"NEG={word['negative']}"
        )