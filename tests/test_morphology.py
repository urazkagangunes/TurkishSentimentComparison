from MorphologicalAnalysis.FsmMorphologicalAnalyzer import FsmMorphologicalAnalyzer


fsm = FsmMorphologicalAnalyzer()

test_words = [
    "güzel",
    "güzeldi",
    "kötü",
    "kötüydü",
    "odalar",
    "beğenmedim",
    "etmiyorum",
    "mükemmel",
    "berbat"
]


for word in test_words:

    print("\n" + "=" * 80)
    print("WORD:", word)

    parses = fsm.morphologicalAnalysis(word)

    print("Parse count:", parses.size())

    for i in range(parses.size()):
        parse = parses.getFsmParse(i)

        print(
            f"{i}: {parse.transitionList()}"
        )