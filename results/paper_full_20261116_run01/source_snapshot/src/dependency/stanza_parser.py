import stanza


class TurkishDependencyParser:
    """
    Wrapper for Stanford Stanza Turkish dependency parser.
    """

    def __init__(self, use_gpu=False):
        self.nlp = stanza.Pipeline(
            lang="tr",
            processors="tokenize,mwt,pos,lemma,depparse",
            use_gpu=use_gpu,
            download_method=None
        )

    def parse(self, text):
        """
        Parses a Turkish text and returns sentence/token level
        dependency information.
        """

        doc = self.nlp(text)

        sentences = []

        for sentence in doc.sentences:

            tokens = []

            for word in sentence.words:
                tokens.append({
                    "id": word.id,
                    "text": word.text,
                    "lemma": word.lemma,
                    "upos": word.upos,
                    "head": word.head,
                    "deprel": word.deprel,
                    "feats": word.feats
                })

            sentences.append({
                "text": sentence.text,
                "tokens": tokens
            })

        return sentences