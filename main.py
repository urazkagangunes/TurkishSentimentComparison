from src.dependency.stanza_parser import TurkishDependencyParser


def main():

    parser = TurkishDependencyParser()

    text = "Otel güzeldi ama odalar temiz değildi."

    parsed_sentences = parser.parse(text)

    for sentence in parsed_sentences:

        print("\nSentence:")
        print(sentence["text"])

        print("\nDependency Parse:")

        for token in sentence["tokens"]:
            print(token)


if __name__ == "__main__":
    main()