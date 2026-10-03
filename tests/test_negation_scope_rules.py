from src.dependency.sentiment_scope_resolver import (
    SentimentScopeResolver
)


resolver = SentimentScopeResolver()


def run_case(
    name,
    sentiment_token,
    tokens,
    expected_count
):

    token_by_id, children = (
        resolver._build_indexes(
            tokens
        )
    )

    result = (
        resolver._find_negations(
            sentiment_token,
            token_by_id,
            children
        )
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        name
    )

    print(
        "Negations:",
        result
    )

    assert (
        len(result)
        == expected_count
    ), (
        f"{name}: "
        f"expected {expected_count}, "
        f"got {len(result)}"
    )


# =============================================================
# 1. SENTIMENT TOKEN ITSELF NEGATED
#
# sevmedim
# =============================================================

tokens = [
    {
        "id": 1,
        "text": "sevmedim",
        "lemma": "sev",
        "upos": "VERB",
        "head": 0,
        "deprel": "root",
        "feats": "Polarity=Neg"
    }
]

run_case(
    "1. Morphologically negated sentiment verb",
    tokens[0],
    tokens,
    1
)


# =============================================================
# 2. EXPLICIT DEGIL CHILD
#
# güzel <- değildi
# =============================================================

tokens = [
    {
        "id": 1,
        "text": "güzel",
        "lemma": "güzel",
        "upos": "ADJ",
        "head": 0,
        "deprel": "root",
        "feats": None
    },
    {
        "id": 2,
        "text": "değildi",
        "lemma": "değil",
        "upos": "AUX",
        "head": 1,
        "deprel": "cop",
        "feats": "Polarity=Neg"
    }
]

run_case(
    "2. Explicit değil child",
    tokens[0],
    tokens,
    1
)


# =============================================================
# 3. NEGATED LIGHT VERB CHILD
#
# tavsiye <- etmiyorum
# =============================================================

tokens = [
    {
        "id": 1,
        "text": "tavsiye",
        "lemma": "tavsiye",
        "upos": "NOUN",
        "head": 0,
        "deprel": "root",
        "feats": None
    },
    {
        "id": 2,
        "text": "etmiyorum",
        "lemma": "et",
        "upos": "VERB",
        "head": 1,
        "deprel": "compound:lvc",
        "feats": "Polarity=Neg"
    }
]

run_case(
    "3. Negated light verb child",
    tokens[0],
    tokens,
    1
)


# =============================================================
# 4. EXPLICIT DEGIL PARENT
#
# kötü -> değildi
#
# Known alternative Stanza structure.
# =============================================================

tokens = [
    {
        "id": 1,
        "text": "kötü",
        "lemma": "kötü",
        "upos": "ADJ",
        "head": 2,
        "deprel": "amod",
        "feats": None
    },
    {
        "id": 2,
        "text": "değildi",
        "lemma": "değil",
        "upos": "AUX",
        "head": 0,
        "deprel": "root",
        "feats": "Polarity=Neg"
    }
]

run_case(
    "4. Explicit değil parent",
    tokens[0],
    tokens,
    1
)


# =============================================================
# 5. ARBITRARY NEGATED PARENT
#
# This is the important regression case.
#
# güzel -> beklemiyordum
#
# Parent is morphologically NEG, but there is no sufficiently
# strong dependency evidence that it directly negates güzel.
# =============================================================

tokens = [
    {
        "id": 1,
        "text": "güzel",
        "lemma": "güzel",
        "upos": "ADJ",
        "head": 2,
        "deprel": "amod",
        "feats": None
    },
    {
        "id": 2,
        "text": "beklemiyordum",
        "lemma": "bekle",
        "upos": "VERB",
        "head": 0,
        "deprel": "root",
        "feats": "Polarity=Neg"
    }
]

run_case(
    "5. Arbitrary negative parent must NOT flip",
    tokens[0],
    tokens,
    0
)


# =============================================================
# 6. ARBITRARY NEGATIVE CHILD
#
# NEG morphology alone is insufficient when the dependency
# relation is not a supported negation relation.
# =============================================================

tokens = [
    {
        "id": 1,
        "text": "güzel",
        "lemma": "güzel",
        "upos": "ADJ",
        "head": 0,
        "deprel": "root",
        "feats": None
    },
    {
        "id": 2,
        "text": "beklemiyorum",
        "lemma": "bekle",
        "upos": "VERB",
        "head": 1,
        "deprel": "conj",
        "feats": "Polarity=Neg"
    }
]

run_case(
    "6. Arbitrary negative child must NOT flip",
    tokens[0],
    tokens,
    0
)


print(
    "\n"
    + "=" * 80
)

print(
    "ALL CONSERVATIVE NEGATION TESTS PASSED"
)