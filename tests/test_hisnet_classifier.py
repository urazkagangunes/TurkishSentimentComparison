from src.hisnet.hisnet_classifier import HisNetClassifier


classifier = HisNetClassifier()

sentences = [
    "Film güzel.",
    "Film kötü.",
    "Otel mükemmel.",
    "Bu ürün berbat.",
    "Film güzel değildi.",
    "Otel güzeldi ama odalar kötüydü."
]


for sentence in sentences:

    result = classifier.classify(sentence)

    print("\n" + "=" * 80)
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

    print("Matched:")
    for word in result["matched_words"]:
        print("  ", word)

    print("Sentiment-bearing:")
    for word in result["sentiment_words"]:
        print("  ", word)