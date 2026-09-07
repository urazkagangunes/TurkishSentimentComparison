from src.hisnet.hisnet_classifier import HisNetClassifier
from src.dependency.stanza_parser import TurkishDependencyParser

from src.dependency.sentiment_scope_resolver import (
    SentimentScopeResolver
)


hisnet = HisNetClassifier()
parser = TurkishDependencyParser()
resolver = SentimentScopeResolver()


sentences = [
    "Otel çok güzeldi.",
    "Personel biraz kötüydü.",
    "Film güzel değildi.",
    "Film kötü değildi.",
    "Bu ürünü kesinlikle tavsiye etmiyorum.",
    "Otel güzeldi ama odalar kötüydü.",
    "Personel iyiydi fakat yemekler kötüydü."
]


for sentence in sentences:

    print("\n" + "=" * 100)
    print("SENTENCE:", sentence)

    hisnet_result = hisnet.classify(
        sentence
    )

    parsed_sentences = parser.parse(
        sentence
    )

    for parsed_sentence in parsed_sentences:

        scopes = resolver.resolve(
            hisnet_result["sentiment_words"],
            parsed_sentence["tokens"]
        )

        for scope in scopes:

            print("\nSENTIMENT:")
            print(
                "  word   :",
                scope["word"]
            )
            print(
                "  root   :",
                scope["root"]
            )

            print(
                "  token  :",
                scope["sentiment_token"]
            )

            print(
                "  target :",
                scope["target"]
            )

            print(
                "  target strategy:",
                scope["target_strategy"]
            )

            print(
                "  role   :",
                scope["contextual_role"]
            )

            print(
                "  active :",
                scope["active_sentiment"]
            )

            print(
                "  suppression:",
                scope["suppression_reason"]
            )

            print(
                "  modifiers:",
                scope["modifiers"]
            )

            print(
                "  negations:",
                scope["negations"]
            )

            print(
                "  contrast:",
                scope["contrast"]
            )