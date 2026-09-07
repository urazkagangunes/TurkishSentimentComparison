from src.hisnet.hisnet_classifier import HisNetClassifier
from src.dependency.stanza_parser import TurkishDependencyParser
from src.dependency.sentiment_scope_resolver import SentimentScopeResolver


hisnet = HisNetClassifier()
parser = TurkishDependencyParser()
resolver = SentimentScopeResolver()


TEST_CASES = [

    # ============================================================
    # 1. SIMPLE PREDICATE
    # ============================================================

    {
        "category": "simple",
        "text": "Film güzel.",
        "expected": [
            {
                "root": "güzel",
                "target": "film"
            }
        ]
    },

    {
        "category": "simple",
        "text": "Film kötü.",
        "expected": [
            {
                "root": "kötü",
                "target": "film"
            }
        ]
    },

    {
        "category": "simple",
        "text": "Otel mükemmel.",
        "expected": [
            {
                "root": "mükemmel",
                "target": "otel"
            }
        ]
    },

    {
        "category": "simple",
        "text": "Personel iyi.",
        "expected": [
            {
                "root": "iyi",
                "target": "personel"
            }
        ]
    },

    {
        "category": "simple",
        "text": "Oda güzel.",
        "expected": [
            {
                "root": "güzel",
                "target": "oda"
            }
        ]
    },

    {
        "category": "simple",
        "text": "Hizmet kötü.",
        "expected": [
            {
                "root": "kötü",
                "target": "hizmet"
            }
        ]
    },

    {
        "category": "simple",
        "text": "Yemek güzel.",
        "expected": [
            {
                "root": "güzel",
                "target": "yemek"
            }
        ],
        "expected_suppressed": ["yemek"]
    },

    {
        "category": "simple",
        "text": "Ürün kötü.",
        "expected": [
            {
                "root": "kötü",
                "target": "ürün"
            }
        ]
    },

    # ============================================================
    # 2. INTENSIFIER / DOWNTONER
    # ============================================================

    {
        "category": "modifier",
        "text": "Otel çok güzel.",
        "expected": [
            {
                "root": "güzel",
                "target": "otel",
                "modifiers": ["çok"]
            }
        ]
    },

    {
        "category": "modifier",
        "text": "Film çok kötü.",
        "expected": [
            {
                "root": "kötü",
                "target": "film",
                "modifiers": ["çok"]
            }
        ]
    },

    {
        "category": "modifier",
        "text": "Personel biraz iyi.",
        "expected": [
            {
                "root": "iyi",
                "target": "personel",
                "modifiers": ["biraz"]
            }
        ]
    },

    {
        "category": "modifier",
        "text": "Oda biraz kötü.",
        "expected": [
            {
                "root": "kötü",
                "target": "oda",
                "modifiers": ["biraz"]
            }
        ]
    },

    {
        "category": "modifier",
        "text": "Hizmet oldukça güzel.",
        "expected": [
            {
                "root": "güzel",
                "target": "hizmet",
                "modifiers": ["oldukça"]
            }
        ]
    },

    {
        "category": "modifier",
        "text": "Ürün son derece kötü.",
        "expected": [
            {
                "root": "kötü",
                "target": "ürün"
            }
        ]
    },

    {
        "category": "modifier",
        "text": "Otel aşırı güzel.",
        "expected": [
            {
                "root": "güzel",
                "target": "otel",
                "modifiers": ["aşırı"]
            }
        ]
    },

    {
        "category": "modifier",
        "text": "Yemek pek kötü.",
        "expected": [
            {
                "root": "kötü",
                "target": "yemek",
                "modifiers": ["pek"]
            }
        ],
        "expected_suppressed": ["yemek"]
    },

    # ============================================================
    # 3. NEGATION
    # ============================================================

    {
        "category": "negation",
        "text": "Film güzel değil.",
        "expected": [
            {
                "root": "güzel",
                "target": "film",
                "negated": True
            }
        ]
    },

    {
        "category": "negation",
        "text": "Film kötü değil.",
        "expected": [
            {
                "root": "kötü",
                "target": "film",
                "negated": True
            }
        ]
    },

    {
        "category": "negation",
        "text": "Otel güzel değildi.",
        "expected": [
            {
                "root": "güzel",
                "target": "otel",
                "negated": True
            }
        ]
    },

    {
        "category": "negation",
        "text": "Personel kötü değildi.",
        "expected": [
            {
                "root": "kötü",
                "target": "personel",
                "negated": True
            }
        ]
    },

    {
        "category": "negation",
        "text": "Oda hiç güzel değildi.",
        "expected": [
            {
                "root": "güzel",
                "target": "oda",
                "negated": True
            }
        ]
    },

    {
        "category": "negation",
        "text": "Ürün hiç kötü değildi.",
        "expected": [
            {
                "root": "kötü",
                "target": "ürün",
                "negated": True
            }
        ]
    },

    {
        "category": "negation",
        "text": "Bu ürünü tavsiye etmiyorum.",
        "expected": [
            {
                "root": "tavsiye",
                "target": "ürün",
                "negated": True
            }
        ]
    },

    {
        "category": "negation",
        "text": "Bu filmi tavsiye etmiyorum.",
        "expected": [
            {
                "root": "tavsiye",
                "target": "film",
                "negated": True
            }
        ]
    },

    # ============================================================
    # 4. CONTRAST / MULTIPLE TARGETS
    # ============================================================

    {
        "category": "contrast",
        "text": "Otel güzel ama odalar kötü.",
        "expected": [
            {
                "root": "güzel",
                "target": "otel"
            },
            {
                "root": "kötü",
                "target": "oda",
                "contrast": "ama"
            }
        ]
    },

    {
        "category": "contrast",
        "text": "Personel iyi fakat yemekler kötü.",
        "expected": [
            {
                "root": "iyi",
                "target": "personel"
            },
            {
                "root": "kötü",
                "target": "yemek",
                "contrast": "fakat"
            }
        ],
        "expected_suppressed": ["yemek"]
    },

    {
        "category": "contrast",
        "text": "Oda kötü ancak manzara güzel.",
        "expected": [
            {
                "root": "kötü",
                "target": "oda"
            },
            {
                "root": "güzel",
                "target": "manzara",
                "contrast": "ancak"
            }
        ]
    },

    {
        "category": "contrast",
        "text": "Ürün güzel ama fiyatı kötü.",
        "expected": [
            {
                "root": "güzel",
                "target": "ürün"
            },
            {
                "root": "kötü",
                "target": "fiyat",
                "contrast": "ama"
            }
        ]
    },

    {
        "category": "contrast",
        "text": "Otel kötü fakat personel mükemmel.",
        "expected": [
            {
                "root": "kötü",
                "target": "otel"
            },
            {
                "root": "mükemmel",
                "target": "personel",
                "contrast": "fakat"
            }
        ]
    },

    {
        "category": "contrast",
        "text": "Film güzel ancak oyunculuk kötü.",
        "expected": [
            {
                "root": "güzel",
                "target": "film"
            },
            {
                "root": "kötü",
                "target": "oyunculuk",
                "contrast": "ancak"
            }
        ]
    },

    {
        "category": "contrast",
        "text": "Yemek kötü ama servis iyi.",
        "expected": [
            {
                "root": "kötü",
                "target": "yemek"
            },
            {
                "root": "iyi",
                "target": "servis",
                "contrast": "ama"
            }
        ],
        "expected_suppressed": ["yemek"]
    },

    {
        "category": "contrast",
        "text": "Personel kötü fakat otel güzel.",
        "expected": [
            {
                "root": "kötü",
                "target": "personel"
            },
            {
                "root": "güzel",
                "target": "otel",
                "contrast": "fakat"
            }
        ]
    },

    # ============================================================
    # 5. ATTRIBUTIVE / STRUCTURAL
    # ============================================================

    {
        "category": "structural",
        "text": "Güzel otel merkeze yakın.",
        "expected": [
            {
                "root": "güzel",
                "target": "otel"
            }
        ]
    },

    {
        "category": "structural",
        "text": "Kötü ürün hemen bozuldu.",
        "expected": [
            {
                "root": "kötü",
                "target": "ürün"
            }
        ]
    },

    {
        "category": "structural",
        "text": "Mükemmel personel bize yardımcı oldu.",
        "expected": [
            {
                "root": "mükemmel",
                "target": "personel"
            }
        ]
    },

    {
        "category": "structural",
        "text": "İyi hizmet bekliyordum.",
        "expected": [
            {
                "root": "iyi",
                "target": "hizmet"
            }
        ]
    },

    # ============================================================
    # 6. HARD / COMBINED CASES
    # ============================================================

    {
        "category": "combined",
        "text": "Personel gerçekten çok iyi.",
        "expected": [
            {
                "root": "iyi",
                "target": "personel",
                "modifiers": ["çok"]
            }
        ]
    },

    {
        "category": "combined",
        "text": "Otel hiç de kötü değil.",
        "expected": [
            {
                "root": "kötü",
                "target": "otel",
                "negated": True
            }
        ]
    },

    {
        "category": "combined",
        "text": "Otel çok güzel ama odalar biraz kötü.",
        "expected": [
            {
                "root": "güzel",
                "target": "otel",
                "modifiers": ["çok"]
            },
            {
                "root": "kötü",
                "target": "oda",
                "modifiers": ["biraz"],
                "contrast": "ama"
            }
        ]
    },

    {
        "category": "combined",
        "text": "Personel çok iyi fakat yemekler hiç güzel değildi.",
        "expected": [
            {
                "root": "iyi",
                "target": "personel",
                "modifiers": ["çok"]
            },
            {
                "root": "güzel",
                "target": "yemek",
                "negated": True,
                "contrast": "fakat"
            }
        ],
        "expected_suppressed": ["yemek"]
    }
]


def normalize(text):
    if text is None:
        return None

    return (
        text
        .replace("I", "ı")
        .replace("İ", "i")
        .lower()
    )


def get_root_candidates(scopes, root):
    return [
        scope
        for scope in scopes
        if normalize(scope.get("root")) == normalize(root)
    ]


def check_expected_scope(expected, scopes):

    root = expected["root"]

    candidates = get_root_candidates(
        scopes,
        root
    )

    if not candidates:
        return False, f"Opinion root not resolved: {root}"

    for scope in candidates:

        if not scope.get(
            "active_sentiment",
            True
        ):
            continue

        # --------------------------------------------
        # TARGET
        # --------------------------------------------

        expected_target = expected.get(
            "target"
        )

        if expected_target is not None:

            target = scope.get("target")

            if target is None:
                continue

            actual_target = normalize(
                target.get("root")
                or target.get("lemma")
                or target.get("word")
            )

            if actual_target != normalize(
                expected_target
            ):
                continue

        # --------------------------------------------
        # MODIFIERS
        # --------------------------------------------

        expected_modifiers = {
            normalize(x)
            for x in expected.get(
                "modifiers",
                []
            )
        }

        actual_modifiers = {
            normalize(
                modifier.get("lemma")
                or modifier.get("word")
            )
            for modifier
            in scope.get(
                "modifiers",
                []
            )
        }

        if not expected_modifiers.issubset(
            actual_modifiers
        ):
            continue

        # --------------------------------------------
        # NEGATION
        # --------------------------------------------

        if expected.get(
            "negated",
            False
        ):

            if not scope.get(
                "negations"
            ):
                continue

        # --------------------------------------------
        # CONTRAST
        # --------------------------------------------

        expected_contrast = expected.get(
            "contrast"
        )

        if expected_contrast is not None:

            contrast = scope.get(
                "contrast"
            )

            if contrast is None:
                continue

            actual_contrast = normalize(
                contrast.get("lemma")
                or contrast.get("word")
            )

            if actual_contrast != normalize(
                expected_contrast
            ):
                continue

        return True, "OK"

    return False, (
        f"Root found but expected scope did not match: {root}"
    )


def check_suppression(root, scopes):

    candidates = get_root_candidates(
        scopes,
        root
    )

    if not candidates:
        # If HisNet did not consider this root sentiment-bearing,
        # suppression is not needed.
        return True, "Not a HisNet sentiment candidate"

    for scope in candidates:

        if (
            scope.get("active_sentiment") is False
            and scope.get("contextual_role") == "target"
        ):
            return True, "Correctly suppressed"

    return False, (
        f"{root} was expected to be suppressed as target"
    )


passed = 0
failed = 0
skipped = 0

category_stats = {}


for index, case in enumerate(
    TEST_CASES,
    start=1
):

    category = case["category"]
    text = case["text"]

    category_stats.setdefault(
        category,
        {
            "pass": 0,
            "fail": 0,
            "skip": 0
        }
    )

    hisnet_result = hisnet.classify(
        text
    )

    parsed = parser.parse(
        text
    )

    if not parsed:
        print(
            f"[{index:02}] FAIL_PARSER "
            f"[{category}] {text}"
        )

        failed += 1
        category_stats[category]["fail"] += 1

        continue

    # Every stress-test item currently contains
    # a single sentence.
    dependency_tokens = parsed[0][
        "tokens"
    ]

    scopes = resolver.resolve(
        hisnet_result["sentiment_words"],
        dependency_tokens,
        hisnet_result["morphological_analyses"]
    )

    hisnet_roots = {
        normalize(word["root"])
        for word
        in hisnet_result[
            "sentiment_words"
        ]
    }

    expected_roots = {
        normalize(
            expected["root"]
        )
        for expected
        in case["expected"]
    }

    missing_from_lexicon = (
        expected_roots
        - hisnet_roots
    )

    if missing_from_lexicon:

        print(
            f"[{index:02}] SKIP_LEXICON "
            f"[{category}] {text}"
        )

        print(
            "     Missing sentiment roots:",
            sorted(
                missing_from_lexicon
            )
        )

        skipped += 1
        category_stats[category]["skip"] += 1

        continue

    errors = []

    for expected in case["expected"]:

        success, message = (
            check_expected_scope(
                expected,
                scopes
            )
        )

        if not success:
            errors.append(message)

    for root in case.get(
        "expected_suppressed",
        []
    ):

        success, message = (
            check_suppression(
                root,
                scopes
            )
        )

        if not success:
            errors.append(message)

    if not errors:

        print(
            f"[{index:02}] PASS "
            f"[{category}] {text}"
        )

        passed += 1
        category_stats[category]["pass"] += 1

    else:

        print(
            f"[{index:02}] FAIL_RESOLVER "
            f"[{category}] {text}"
        )

        for error in errors:
            print(
                "     -",
                error
            )

        print(
            "     HisNet sentiment roots:",
            sorted(
                hisnet_roots
            )
        )

        print(
            "     Resolved scopes:"
        )

        for scope in scopes:

            target = (
                (
                        scope["target"].get("root")
                        or scope["target"].get("lemma")
                        or scope["target"].get("word")
                )
                if scope.get("target")
                else None
            )

            modifiers = [
                x.get("lemma")
                or x.get("word")
                for x
                in scope.get(
                    "modifiers",
                    []
                )
            ]

            negations = [
                x.get("lemma")
                or x.get("word")
                for x
                in scope.get(
                    "negations",
                    []
                )
            ]

            contrast = (
                scope["contrast"].get(
                    "lemma"
                )
                if scope.get("contrast")
                else None
            )

            print(
                "       ",
                {
                    "root":
                        scope.get("root"),

                    "target":
                        target,

                    "role":
                        scope.get(
                            "contextual_role"
                        ),

                    "active":
                        scope.get(
                            "active_sentiment"
                        ),

                    "modifiers":
                        modifiers,

                    "negations":
                        negations,

                    "contrast":
                        contrast
                }
            )

        failed += 1
        category_stats[category]["fail"] += 1


print(
    "\n"
    + "=" * 100
)

print("STRESS TEST SUMMARY")
print("=" * 100)

print(
    f"Total   : {len(TEST_CASES)}"
)

print(
    f"PASS    : {passed}"
)

print(
    f"FAIL    : {failed}"
)

print(
    f"SKIP    : {skipped}"
)

evaluated = passed + failed

if evaluated > 0:

    accuracy = (
        passed / evaluated
    )

    print(
        f"Resolver success rate "
        f"(excluding lexicon skips): "
        f"{accuracy:.4f}"
    )


print(
    "\nCATEGORY BREAKDOWN"
)

for category, stats in (
    category_stats.items()
):

    evaluated_category = (
        stats["pass"]
        + stats["fail"]
    )

    if evaluated_category > 0:

        rate = (
            stats["pass"]
            / evaluated_category
        )

    else:
        rate = 0.0

    print(
        f"{category:<12}"
        f"PASS={stats['pass']:<3}"
        f"FAIL={stats['fail']:<3}"
        f"SKIP={stats['skip']:<3}"
        f"RATE={rate:.4f}"
    )