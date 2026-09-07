from src.dependency.stanza_parser import TurkishDependencyParser


parser = TurkishDependencyParser()

sentences = [
    "Otel aşırı güzel.",
    "Oda hiç güzel değildi.",
    "Otel güzel ama odalar kötü.",
    "Personel iyi fakat yemekler kötü.",
    "Oda kötü ancak manzara güzel."
]


for sentence in sentences:

    print("\n" + "=" * 120)
    print("SENTENCE:", sentence)

    parsed_sentences = parser.parse(sentence)

    for parsed_sentence in parsed_sentences:

        for token in parsed_sentence["tokens"]:

            print(
                f"{token['id']:<3}"
                f"{token['text']:<15}"
                f"lemma={str(token.get('lemma')):<15}"
                f"upos={str(token.get('upos')):<8}"
                f"head={str(token.get('head')):<4}"
                f"deprel={str(token.get('deprel')):<16}"
                f"feats={token.get('feats')}"
            )