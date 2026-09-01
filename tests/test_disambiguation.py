from Corpus.Sentence import Sentence

from MorphologicalAnalysis.FsmMorphologicalAnalyzer import (
    FsmMorphologicalAnalyzer
)

from MorphologicalDisambiguation.LongestRootFirstDisambiguation import (
    LongestRootFirstDisambiguation
)


fsm = FsmMorphologicalAnalyzer()
disambiguator = LongestRootFirstDisambiguation()

sentences = [
    "Filmi beğenmedim.",
    "Otel güzeldi ama odalar kötüydü.",
    "Bu ürünü kesinlikle tavsiye etmiyorum."
]

for text in sentences:

    print("\n" + "=" * 80)
    print("SENTENCE:", text)

    sentence = Sentence(text)

    all_parses = fsm.robustMorphologicalAnalysis(sentence)

    selected_parses = disambiguator.disambiguate(all_parses)

    for parse in selected_parses:
        print(parse.transitionList())