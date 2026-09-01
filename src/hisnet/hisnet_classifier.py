import re

from SentiNet.SentiLiteralNet import SentiLiteralNet


class HisNetClassifier:

    def __init__(self):
        self.sentinet = SentiLiteralNet()

    @staticmethod
    def _turkish_lower(text):
        """
        Turkish-aware lowercase conversion.
        """
        text = text.replace("I", "ı").replace("İ", "i")
        return text.lower()

    def tokenize(self, text):
        """
        Simple tokenizer for the initial HisNet-only baseline.

        Dependency parsing is NOT used here.
        """

        text = self._turkish_lower(text)

        return re.findall(
            r"[abcçdefgğhıijklmnoöprsştuüvyz]+",
            text
        )

    def classify(self, text):
        """
        Classifies a Turkish text using only HisNet lexical scores.

        Returns:
            - final prediction
            - total positive and negative scores
            - all words found in HisNet
            - sentiment-bearing words
            - coverage statistics
        """

        words = self.tokenize(text)

        positive_score = 0.0
        negative_score = 0.0

        matched_words = []
        sentiment_words = []

        for word in words:

            try:
                senti_literal = self.sentinet.getSentiLiteral(word)

                positive = float(
                    senti_literal.getPositiveScore()
                )

                negative = float(
                    senti_literal.getNegativeScore()
                )

                positive_score += positive
                negative_score += negative

                # Word exists in HisNet, even if its scores are 0 / 0.
                matched_words.append({
                    "word": word,
                    "positive": positive,
                    "negative": negative
                })

                # Only words carrying actual sentiment information.
                if positive > 0.0 or negative > 0.0:
                    sentiment_words.append({
                        "word": word,
                        "positive": positive,
                        "negative": negative
                    })

            except KeyError:
                # Word does not exist in HisNet.
                continue

        if positive_score > negative_score:
            prediction = "positive"

        elif negative_score > positive_score:
            prediction = "negative"

        else:
            prediction = "neutral"

        token_count = len(words)
        matched_word_count = len(matched_words)
        sentiment_word_count = len(sentiment_words)

        lexicon_coverage = (
            matched_word_count / token_count
            if token_count > 0
            else 0.0
        )

        sentiment_coverage = (
            sentiment_word_count / token_count
            if token_count > 0
            else 0.0
        )

        return {
            "text": text,
            "prediction": prediction,

            "positive_score": positive_score,
            "negative_score": negative_score,

            "matched_words": matched_words,
            "sentiment_words": sentiment_words,

            "token_count": token_count,
            "matched_word_count": matched_word_count,
            "sentiment_word_count": sentiment_word_count,

            "lexicon_coverage": lexicon_coverage,
            "sentiment_coverage": sentiment_coverage
        }