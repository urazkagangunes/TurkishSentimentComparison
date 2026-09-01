from SentiNet.SentiLiteralNet import SentiLiteralNet

from src.common.morphology import MorphologicalRootExtractor


class HisNetClassifier:

    def __init__(self):
        self.sentinet = SentiLiteralNet()
        self.root_extractor = MorphologicalRootExtractor()

    def _lookup_root(self, root):
        """
        Looks up a disambiguated morphological root in HisNet.

        Returns None if the root does not exist in HisNet.
        """

        try:
            senti_literal = self.sentinet.getSentiLiteral(root)

            positive = float(
                senti_literal.getPositiveScore()
            )

            negative = float(
                senti_literal.getNegativeScore()
            )

            return {
                "root": root,
                "positive": positive,
                "negative": negative
            }

        except KeyError:
            return None

    def classify(self, text):
        """
        HisNet-only sentiment classification.

        Pipeline:
            Text
              -> Morphological analysis
              -> LongestRootFirstDisambiguation
              -> Selected morphological roots
              -> HisNet lookup
              -> Score aggregation
              -> Positive / Negative / Neutral

        Dependency information is NOT used here.
        """

        morphological_analyses = (
            self.root_extractor.analyze_sentence(text)
        )

        positive_score = 0.0
        negative_score = 0.0

        matched_words = []
        sentiment_words = []
        unmatched_words = []

        for item in morphological_analyses:

            word = item["word"]
            root = item["root"]
            analysis = item["analysis"]

            senti_result = self._lookup_root(root)

            if senti_result is None:

                unmatched_words.append({
                    "word": word,
                    "root": root,
                    "analysis": analysis
                })

                continue

            positive = senti_result["positive"]
            negative = senti_result["negative"]

            positive_score += positive
            negative_score += negative

            matched_entry = {
                "word": word,
                "root": root,
                "analysis": analysis,
                "positive": positive,
                "negative": negative
            }

            matched_words.append(matched_entry)

            if positive > 0.0 or negative > 0.0:
                sentiment_words.append(matched_entry)

        if positive_score > negative_score:
            prediction = "positive"

        elif negative_score > positive_score:
            prediction = "negative"

        else:
            prediction = "neutral"

        token_count = len(morphological_analyses)
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

            "morphological_analyses": morphological_analyses,

            "matched_words": matched_words,
            "sentiment_words": sentiment_words,
            "unmatched_words": unmatched_words,

            "token_count": token_count,
            "matched_word_count": matched_word_count,
            "sentiment_word_count": sentiment_word_count,

            "lexicon_coverage": lexicon_coverage,
            "sentiment_coverage": sentiment_coverage
        }