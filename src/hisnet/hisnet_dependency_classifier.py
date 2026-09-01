from src.hisnet.hisnet_classifier import HisNetClassifier
from src.dependency.stanza_parser import TurkishDependencyParser


class HisNetDependencyClassifier:
    """
    HisNet + Dependency Parsing sentiment classifier.

    Pipeline:
        Text
          -> HisNet + morphological preprocessing
          -> lexical sentiment scores

        Text
          -> Stanford Stanza
          -> dependency parse

        Lexical sentiment + dependency-based negation scope
          -> final sentiment
    """

    def __init__(self):
        self.hisnet = HisNetClassifier()
        self.dependency_parser = TurkishDependencyParser()

    @staticmethod
    def _turkish_lower(text):
        return text.replace("I", "ı").replace("İ", "i").lower()

    def _is_negation_token(self, token):
        """
        Detect grammatical negation.

        IMPORTANT:
        Stanza's Polarity=Neg is morphological negation,
        NOT sentiment polarity.
        """

        feats = token.get("feats") or ""
        lemma = self._turkish_lower(token.get("lemma") or "")
        deprel = token.get("deprel") or ""

        return (
            "Polarity=Neg" in feats
            or lemma == "değil"
            or deprel == "neg"
        )

    def _match_sentiment_word_to_token(
        self,
        sentiment_word,
        dependency_tokens,
        used_token_ids
    ):
        """
        Align a HisNet sentiment-bearing word with
        the corresponding Stanza dependency token.
        """

        word = self._turkish_lower(sentiment_word["word"])
        root = self._turkish_lower(sentiment_word["root"])

        candidates = []

        for token in dependency_tokens:

            token_id = token["id"]

            if token_id in used_token_ids:
                continue

            token_text = self._turkish_lower(
                token.get("text") or ""
            )

            token_lemma = self._turkish_lower(
                token.get("lemma") or ""
            )

            score = 0

            # Strongest match: same surface form
            if token_text == word:
                score = 100

            # Root equals dependency token
            elif token_text == root:
                score = 90

            # Root equals Stanza lemma
            elif token_lemma == root:
                score = 80

            # Useful for cases where Stanza splits
            # copular forms such as kötüydü -> kötü + ydü
            elif word.startswith(root) and token_text == root:
                score = 70

            if score > 0:
                candidates.append((score, token))

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: item[0],
            reverse=True
        )

        best_token = candidates[0][1]

        used_token_ids.add(best_token["id"])

        return best_token

    def _find_negation_target(
        self,
        negation_token,
        sentiment_by_token,
        dependency_tokens
    ):
        """
        Finds which sentiment-bearing token is directly
        affected by a negation token.

        Initial conservative strategy:
            1. Negated token itself is sentiment-bearing
            2. Negation token's HEAD is sentiment-bearing
            3. A direct child of the negation token is sentiment-bearing

        No long-distance guessing is performed yet.
        """

        negation_id = negation_token["id"]
        head_id = negation_token["head"]

        # Case 1:
        # sevmedim / beğenmedim etc.
        if negation_id in sentiment_by_token:
            return negation_id

        # Case 2:
        # güzel <- değildi
        # tavsiye <- etmiyorum
        if head_id in sentiment_by_token:
            return head_id

        # Case 3:
        # Parser may make the negated auxiliary the parent.
        for token in dependency_tokens:

            if token["head"] == negation_id:

                child_id = token["id"]

                if child_id in sentiment_by_token:
                    return child_id

        return None

    def _classify_sentence(self, sentence_text, dependency_tokens):
        """
        Classifies one sentence using HisNet + dependency-based
        negation correction.
        """

        hisnet_result = self.hisnet.classify(sentence_text)

        sentiment_states = []
        sentiment_by_token = {}

        used_token_ids = set()

        # -----------------------------------------------------
        # ALIGN HISNET SENTIMENT WORDS WITH STANZA TOKENS
        # -----------------------------------------------------

        for sentiment_word in hisnet_result["sentiment_words"]:

            dependency_token = (
                self._match_sentiment_word_to_token(
                    sentiment_word,
                    dependency_tokens,
                    used_token_ids
                )
            )

            state = {
                "word": sentiment_word["word"],
                "root": sentiment_word["root"],

                "original_positive":
                    sentiment_word["positive"],

                "original_negative":
                    sentiment_word["negative"],

                "adjusted_positive":
                    sentiment_word["positive"],

                "adjusted_negative":
                    sentiment_word["negative"],

                "dependency_token_id": None,
                "negated": False,
                "negated_by": []
            }

            if dependency_token is not None:

                token_id = dependency_token["id"]

                state["dependency_token_id"] = token_id

                sentiment_by_token[token_id] = state

            sentiment_states.append(state)

        # -----------------------------------------------------
        # FIND NEGATION TOKENS
        # -----------------------------------------------------

        negation_tokens = [
            token
            for token in dependency_tokens
            if self._is_negation_token(token)
        ]

        adjustments = []
        unresolved_negations = []

        # -----------------------------------------------------
        # APPLY NEGATION TO LOCAL SENTIMENT TARGET
        # -----------------------------------------------------

        for negation_token in negation_tokens:

            target_token_id = self._find_negation_target(
                negation_token,
                sentiment_by_token,
                dependency_tokens
            )

            if target_token_id is None:

                unresolved_negations.append({
                    "word": negation_token["text"],
                    "lemma": negation_token["lemma"],
                    "id": negation_token["id"],
                    "head": negation_token["head"],
                    "deprel": negation_token["deprel"]
                })

                continue

            target = sentiment_by_token[target_token_id]

            # Negation reverses lexical polarity.
            #
            # Instead of simply multiplying by -1,
            # swap positive and negative scores.
            old_positive = target["adjusted_positive"]
            old_negative = target["adjusted_negative"]

            target["adjusted_positive"] = old_negative
            target["adjusted_negative"] = old_positive

            target["negated"] = not target["negated"]

            target["negated_by"].append(
                negation_token["text"]
            )

            adjustments.append({
                "negation_word": negation_token["text"],
                "negation_id": negation_token["id"],

                "target_word": target["word"],
                "target_root": target["root"],
                "target_token_id": target_token_id,

                "before_positive": old_positive,
                "before_negative": old_negative,

                "after_positive":
                    target["adjusted_positive"],

                "after_negative":
                    target["adjusted_negative"]
            })

        # -----------------------------------------------------
        # FINAL SCORE
        # -----------------------------------------------------

        positive_score = sum(
            item["adjusted_positive"]
            for item in sentiment_states
        )

        negative_score = sum(
            item["adjusted_negative"]
            for item in sentiment_states
        )

        if positive_score > negative_score:
            prediction = "positive"

        elif negative_score > positive_score:
            prediction = "negative"

        else:
            prediction = "neutral"

        return {
            "text": sentence_text,

            "hisnet_prediction":
                hisnet_result["prediction"],

            "prediction":
                prediction,

            "positive_score":
                positive_score,

            "negative_score":
                negative_score,

            "sentiment_words":
                sentiment_states,

            "negation_tokens":
                negation_tokens,

            "adjustments":
                adjustments,

            "unresolved_negations":
                unresolved_negations,

            "dependency_tokens":
                dependency_tokens
        }

    def classify(self, text):
        """
        Supports one or multiple sentences.
        """

        parsed_sentences = self.dependency_parser.parse(text)

        sentence_results = []

        total_positive = 0.0
        total_negative = 0.0

        for parsed_sentence in parsed_sentences:

            result = self._classify_sentence(
                parsed_sentence["text"],
                parsed_sentence["tokens"]
            )

            sentence_results.append(result)

            total_positive += result["positive_score"]
            total_negative += result["negative_score"]

        if total_positive > total_negative:
            prediction = "positive"

        elif total_negative > total_positive:
            prediction = "negative"

        else:
            prediction = "neutral"

        return {
            "text": text,
            "prediction": prediction,
            "positive_score": total_positive,
            "negative_score": total_negative,
            "sentences": sentence_results
        }