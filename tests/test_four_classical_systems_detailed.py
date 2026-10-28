import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from src.hisnet.hisnet_classifier import HisNetClassifier
from src.hisnet.hisnet_dependency_classifier import HisNetDependencyClassifier
from src.hisnet.dependency_hisnet_classifier import DependencyHisNetClassifier
from src.hisnet.literature_weighted_hisnet_dependency_classifier import (
    LiteratureWeightedHisNetDependencyClassifier,
)

SYSTEMS = ["H", "H->D", "D->H", "H->D-LW"]

CASES = [
    {"id": 1, "text": "Film güzel.", "gold": "positive", "focus": "basic positive predicate"},
    {"id": 2, "text": "Film kötü.", "gold": "negative", "focus": "basic negative predicate"},
    {"id": 3, "text": "Harika!", "gold": "positive", "focus": "targetless sentiment"},
    {"id": 4, "text": "Bugün pazartesi.", "gold": "neutral", "focus": "neutral control"},
    {"id": 5, "text": "Masa odada.", "gold": "neutral", "focus": "neutral control"},
    {"id": 6, "text": "Yemekler kötüydü.", "gold": "negative", "focus": "target filtering"},
    {"id": 7, "text": "Yemekler güzel.", "gold": "positive", "focus": "target filtering with potentially sentiment-bearing noun"},
    {"id": 8, "text": "Bu ürünü tavsiye ediyorum.", "gold": "positive", "focus": "light-verb positive recommendation"},
    {"id": 9, "text": "Bu ürünü kesinlikle tavsiye etmiyorum.", "gold": "negative", "focus": "light-verb negation"},
    {"id": 10, "text": "Film güzel değildi.", "gold": "negative", "focus": "explicit copular negation"},
    {"id": 11, "text": "Odalar kötü değildi.", "gold": "positive", "focus": "negated negative adjective"},
    {"id": 12, "text": "Bu ürün iyi değil.", "gold": "negative", "focus": "explicit değil"},
    {"id": 13, "text": "Bu ürün kötü değil.", "gold": "positive", "focus": "explicit değil on negative adjective"},
    {"id": 14, "text": "Bu ürün çok kötü değil.", "gold": "positive", "focus": "modifier + explicit negation", "expect_lw_modifier": True},
    {"id": 15, "text": "Otel çok güzel.", "gold": "positive", "focus": "single-word intensifier", "expect_lw_modifier": True},
    {"id": 16, "text": "Otel biraz kötü.", "gold": "negative", "focus": "mitigator", "expect_lw_modifier": True},
    {"id": 17, "text": "Bu ürün son derece güzel.", "gold": "positive", "focus": "multiword intensifier + component suppression", "expect_lw_modifier": True},
    {"id": 18, "text": "Bu ürün çok ama çok güzel.", "gold": "positive", "focus": "modifier phrase containing contrast word", "expect_lw_modifier": True},
    {"id": 19, "text": "Bu ürün aşırı kötü.", "gold": "negative", "focus": "single-word intensifier + ADV candidate leakage probe", "expect_lw_modifier": True},
    {"id": 20, "text": "Büyük ekran güzel.", "gold": "positive", "focus": "modifier false-positive guard"},
    {"id": 21, "text": "Küçük oda çok güzel.", "gold": "positive", "focus": "modifier attachment guard", "expect_lw_modifier": True},
    {"id": 22, "text": "Otel güzel ama odalar kötü.", "gold": "negative", "focus": "ama contrast / negative second clause", "expect_lw_contrast": True},
    {"id": 23, "text": "Otel kötü ama odalar güzel.", "gold": "positive", "focus": "ama contrast / positive second clause", "expect_lw_contrast": True},
    {"id": 24, "text": "Otel güzel fakat odalar kötü.", "gold": "negative", "focus": "fakat contrast", "expect_lw_contrast": True},
    {"id": 25, "text": "Otel kötü ancak odalar güzel.", "gold": "positive", "focus": "ancak contrast + lexical marker suppression", "expect_lw_contrast": True},
    {"id": 26, "text": "Otel güzel ama odalar kötü değildi.", "gold": "positive", "focus": "contrast + negation; negation should stay in second clause", "expect_lw_contrast": True},
    {"id": 27, "text": "Otel kötü ama odalar güzel değildi.", "gold": "negative", "focus": "contrast + negation; negation should stay in second clause", "expect_lw_contrast": True},
    {"id": 28, "text": "Ürün güzel ama pahalı fakat kaliteli.", "gold": "positive", "focus": "multiple contrast markers", "expect_lw_contrast": True},
    {"id": 29, "text": "Film güzel. Oda kötü değil.", "gold": "positive", "focus": "multi-sentence positive + negated negative"},
    {"id": 30, "text": "Film kötü. Oda güzel değil.", "gold": "negative", "focus": "multi-sentence negative + negated positive"},
    {"id": 31, "text": "Otel İstanbul'da.", "gold": "neutral", "focus": "neutral entity/location control"},
    {"id": 32, "text": "Bu bir ürün.", "gold": "neutral", "focus": "neutral product statement"},
    {
    "id": 33,
    "text": "Film iyi ve güzel değildi.",
    "gold": "negative",
    "focus": "shared negation over coordinated positive predicates",
    },
    {
    "id": 34,
    "text": "Film güzel ama iyi değildi.",
    "gold": "negative",
    "focus": "negation must not cross adversative contrast boundary",
    },
    {
    "id": 35,
    "text": "Otel güzel ve harika değildi.",
    "gold": "negative",
    "focus": "shared explicit negation over coordination",
    },
    {
    "id": 36,
    "text": "Otel güzel fakat harika değildi.",
    "gold": "negative",
    "focus": "explicit negation restricted to post-contrast predicate",
    },
]

OUT_DIR = Path("results/four_classical_diagnostic")
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = OUT_DIR / "console_report.txt"
CSV_PATH = OUT_DIR / "predictions.csv"
WARNINGS_PATH = OUT_DIR / "warnings.csv"
JSON_PATH = OUT_DIR / "details.json"


class Reporter:
    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")

    def log(self, *values):
        text = " ".join(str(v) for v in values)
        print(text)
        self.file.write(text + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


reporter = Reporter(REPORT_PATH)


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, set):
        return sorted(json_safe(v) for v in value)
    try:
        return value.item()
    except Exception:
        return str(value)


def score_triplet(result):
    pos = safe_float(result.get("positive_score"))
    neg = safe_float(result.get("negative_score"))
    return pos, neg, pos - neg


def scope_short(scope):
    token = scope.get("sentiment_token") or {}
    target = scope.get("target") or {}
    return {
        "word": scope.get("word"),
        "root": scope.get("root"),
        "token_id": token.get("id"),
        "upos": token.get("upos"),
        "deprel": token.get("deprel"),
        "target": target.get("word"),
        "target_strategy": scope.get("target_strategy"),
        "active": scope.get("active_sentiment"),
        "negations": [
            {"id": n.get("id"), "word": n.get("word") or n.get("text"), "lemma": n.get("lemma"), "deprel": n.get("deprel")}
            for n in (scope.get("negations") or [])
        ],
        "modifiers": [
            {"id": m.get("id"), "word": m.get("word") or m.get("text"), "lemma": m.get("lemma"), "deprel": m.get("deprel")}
            for m in (scope.get("modifiers") or [])
        ],
        "contrast": scope.get("contrast"),
        "operations": scope.get("operations") or [],
        "original_positive": scope.get("original_positive"),
        "original_negative": scope.get("original_negative"),
        "adjusted_positive": scope.get("adjusted_positive"),
        "adjusted_negative": scope.get("adjusted_negative"),
        "modifier_phrase": scope.get("modifier_phrase"),
        "modifier_multiplier": scope.get("modifier_multiplier"),
        "contrast_marker": scope.get("contrast_marker"),
        "contrast_role": scope.get("contrast_role"),
        "contrast_multiplier": scope.get("contrast_multiplier"),
        "literature_multiplier": scope.get("literature_multiplier"),
    }


def candidate_short(candidate):
    token = candidate.get("sentiment_token") or {}
    target = candidate.get("target") or {}
    return {
        "word": candidate.get("word"),
        "root": candidate.get("root"),
        "token_id": token.get("id"),
        "upos": token.get("upos"),
        "deprel": token.get("deprel"),
        "strategy": candidate.get("candidate_strategy"),
        "target": target.get("word"),
        "target_strategy": candidate.get("target_strategy"),
        "negations": [n.get("word") or n.get("text") for n in (candidate.get("negations") or [])],
        "modifiers": [m.get("word") or m.get("text") for m in (candidate.get("modifiers") or [])],
        "contrast": candidate.get("contrast"),
        "lexicon_status": candidate.get("lexicon_status"),
        "positive": candidate.get("positive"),
        "negative": candidate.get("negative"),
    }


def find_shared_negation(scopes):
    """
    Returns only SUSPICIOUS shared negation assignments.

    A shared explicit negation is legitimate when the affected
    sentiment expressions form a non-adversative coordination,
    e.g.:

        iyi ve güzel değildi

    In that case the same explicit negation may correctly scope
    over multiple coordinated sentiment expressions.
    """

    owners = defaultdict(list)

    for scope in scopes or []:

        if not scope.get(
            "active_sentiment",
            True
        ):
            continue

        for negation in (
            scope.get("negations")
            or []
        ):

            negation_id = negation.get(
                "id"
            )

            if negation_id is None:
                continue

            owners[
                int(negation_id)
            ].append(scope)

    suspicious = {}

    for negation_id, linked_scopes in owners.items():

        if len(linked_scopes) <= 1:
            continue

        sentiment_ids = set()

        for scope in linked_scopes:

            token = scope.get(
                "sentiment_token"
            )

            if (
                token is not None
                and token.get("id")
                is not None
            ):
                sentiment_ids.add(
                    int(token["id"])
                )

        # --------------------------------------------
        # Legitimate shared coordination:
        #
        #   iyi ve güzel değildi
        #
        # One sentiment token is a conj dependent of
        # another sentiment token in the same group.
        # --------------------------------------------

        coordinated = False

        for scope in linked_scopes:

            token = scope.get(
                "sentiment_token"
            ) or {}

            if (
                token.get("deprel")
                == "conj"
            ):

                head = token.get(
                    "head"
                )

                if (
                    head is not None
                    and int(head)
                    in sentiment_ids
                ):
                    coordinated = True
                    break

        # --------------------------------------------
        # If the scopes are a normal coordination,
        # shared negation is expected and should not
        # produce a warning.
        # --------------------------------------------

        if coordinated:
            continue

        suspicious[
            negation_id
        ] = [
            scope.get("word")
            for scope in linked_scopes
        ]

    return suspicious


def duplicate_contrast_pairs(lw_result):
    duplicates = []
    relations_by_sentence = lw_result.get("contrast_relations") or {}
    for sentence_index, relations in relations_by_sentence.items():
        pair_to_markers = defaultdict(list)
        for relation in relations or []:
            pair = (relation.get("pre_head_id"), relation.get("post_head_id"))
            pair_to_markers[pair].append(relation.get("marker"))
        for pair, markers in pair_to_markers.items():
            if len(markers) > 1:
                duplicates.append({"sentence_index": sentence_index, "pair": pair, "markers": markers})
    return duplicates


def add_warning(warnings, case_id, code, message, system=None):
    warnings.append({"case_id": case_id, "system": system or "", "code": code, "message": message})


reporter.log("=" * 120)
reporter.log("FOUR CLASSICAL SYSTEMS — STRUCTURAL DIAGNOSTIC TEST")
reporter.log("=" * 120)
reporter.log("Systems:", ", ".join(SYSTEMS))
reporter.log("Cases:", len(CASES))
reporter.log("IMPORTANT: This is a structural sanity set, not a benchmark or tuning set.")

hisnet = HisNetClassifier()
h_to_d = HisNetDependencyClassifier(hisnet_classifier=hisnet)
d_to_h = DependencyHisNetClassifier()
lw = LiteratureWeightedHisNetDependencyClassifier(hisnet_classifier=hisnet)

all_rows = []
all_details = []
all_warnings = []
correct_counts = Counter()

for case in CASES:
    case_id = case["id"]
    text = case["text"]
    gold = case["gold"]
    focus = case["focus"]

    reporter.log()
    reporter.log("=" * 120)
    reporter.log(f"CASE {case_id}: {text}")
    reporter.log("Gold:", gold)
    reporter.log("Focus:", focus)
    reporter.log("=" * 120)

    h = hisnet.classify(text)
    hd = h_to_d.classify(text, hisnet_result=h)
    dh = d_to_h.classify(text)
    lw_result = lw.classify(text, hisnet_result=h)

    results = {"H": h, "H->D": hd, "D->H": dh, "H->D-LW": lw_result}

    for system in SYSTEMS:
        result = results[system]
        prediction = result.get("prediction")
        pos, neg, net = score_triplet(result)
        is_correct = prediction == gold
        if is_correct:
            correct_counts[system] += 1
        reporter.log(
            f"{system:8s}", f"pred={prediction:8s}", f"POS={pos:.6f}", f"NEG={neg:.6f}", f"NET={net:.6f}", "✓" if is_correct else "✗"
        )
        all_rows.append({
            "id": case_id, "text": text, "gold": gold, "focus": focus,
            "system": system, "prediction": prediction,
            "positive_score": pos, "negative_score": neg, "net_score": net,
            "correct": is_correct,
        })
        if prediction != gold:
            add_warning(all_warnings, case_id, "EXPECTED_LABEL_MISMATCH", f"{system} predicted {prediction}; sanity label is {gold}.", system)
        if not all(math.isfinite(v) for v in (pos, neg, net)):
            add_warning(all_warnings, case_id, "NONFINITE_SCORE", f"{system} produced a non-finite score.", system)

    # H -> D fairness invariant
    if (
        hd.get("hisnet_prediction") != h.get("prediction")
        or not math.isclose(safe_float(hd.get("hisnet_positive_score")), safe_float(h.get("positive_score")), rel_tol=1e-9, abs_tol=1e-9)
        or not math.isclose(safe_float(hd.get("hisnet_negative_score")), safe_float(h.get("negative_score")), rel_tol=1e-9, abs_tol=1e-9)
    ):
        add_warning(all_warnings, case_id, "H_TO_D_BASELINE_MISMATCH", "H->D does not preserve the exact supplied HisNet baseline.", "H->D")

    if hd.get("baseline_reused") is not True:
        add_warning(all_warnings, case_id, "H_TO_D_BASELINE_NOT_REUSED", "H->D reports that the precomputed HisNet result was not reused.", "H->D")

    # LW must begin from exact frozen H->D
    if (
        lw_result.get("dependency_prediction") != hd.get("prediction")
        or not math.isclose(safe_float(lw_result.get("dependency_positive_score")), safe_float(hd.get("positive_score")), rel_tol=1e-9, abs_tol=1e-9)
        or not math.isclose(safe_float(lw_result.get("dependency_negative_score")), safe_float(hd.get("negative_score")), rel_tol=1e-9, abs_tol=1e-9)
    ):
        add_warning(all_warnings, case_id, "LW_BASELINE_MISMATCH", "H->D-LW does not start from the exact frozen H->D result.", "H->D-LW")

    reporter.log()
    reporter.log("H sentiment words:")
    if not h.get("sentiment_words"):
        reporter.log("  []")
    for item in h.get("sentiment_words", []):
        reporter.log(" ", item.get("word"), "| root=", item.get("root"), "| POS=", item.get("positive"), "| NEG=", item.get("negative"))

    reporter.log()
    reporter.log("H->D scopes:")
    for scope in hd.get("scopes", []):
        s = scope_short(scope)
        reporter.log(
            " ", s["word"], "| target=", s["target"],
            "| neg=", [n["word"] for n in s["negations"]],
            "| mod=", [m["word"] for m in s["modifiers"]],
            "| contrast=", s["contrast"], "| active=", s["active"],
            "| ops=", s["operations"], "| adj=", (s["adjusted_positive"], s["adjusted_negative"]),
        )

    if hd.get("unmatched_scopes"):
        add_warning(all_warnings, case_id, "H_TO_D_UNMATCHED_SCOPE", f"{len(hd['unmatched_scopes'])} HisNet sentiment scope(s) could not be aligned to Stanza.", "H->D")

    hd_shared_neg = find_shared_negation(hd.get("scopes", []))
    if hd_shared_neg:
        add_warning(all_warnings, case_id, "H_TO_D_SHARED_NEGATION_SCOPE", f"Same negation token attached to multiple active sentiment scopes: {hd_shared_neg}", "H->D")

    reporter.log()
    reporter.log("D->H dependency candidates:")
    for candidate in dh.get("candidate_diagnostics", []):
        c = candidate_short(candidate)
        reporter.log(
            " ", c["word"], "| root=", c["root"], "| POS=", c["upos"],
            "| deprel=", c["deprel"], "| strategy=", c["strategy"],
            "| target=", c["target"], "| lexicon=", c["lexicon_status"],
            "| scores=", (c["positive"], c["negative"]),
            "| neg=", c["negations"], "| mod=", c["modifiers"],
        )
        if c["deprel"] == "advmod" and c["lexicon_status"] == "sentiment":
            add_warning(
                all_warnings, case_id, "D_TO_H_SENTIMENT_ADV_MOD_CANDIDATE",
                f"ADV modifier '{c['word']}' is also an independent sentiment-bearing D->H candidate. Inspect for score leakage.", "D->H"
            )
        if c["deprel"] in {"aux", "aux:pass", "cop", "compound:lvc", "nmod", "det", "case", "cc", "mark", "punct"}:
            add_warning(
                all_warnings, case_id, "D_TO_H_HARD_EXCLUDED_RELATION_SELECTED",
                f"Candidate '{c['word']}' has structurally excluded dependency relation '{c['deprel']}'.", "D->H"
            )

    dh_shared_neg = find_shared_negation(dh.get("scopes", []))
    if dh_shared_neg:
        add_warning(all_warnings, case_id, "D_TO_H_SHARED_NEGATION_SCOPE", f"Same negation token attached to multiple active sentiment scopes: {dh_shared_neg}", "D->H")

    reporter.log()
    reporter.log("H->D-LW weighted scopes:")
    for scope in lw_result.get("scopes", []):
        s = scope_short(scope)
        reporter.log(
            " ", s["word"], "| active=", s["active"],
            "| dep=", (scope.get("dependency_adjusted_positive"), scope.get("dependency_adjusted_negative")),
            "| modifier=", (s["modifier_phrase"], s["modifier_multiplier"]),
            "| contrast=", (s["contrast_marker"], s["contrast_role"], s["contrast_multiplier"]),
            "| final_mult=", s["literature_multiplier"],
            "| final=", (s["adjusted_positive"], s["adjusted_negative"]),
            "| ops=", s["operations"],
        )

    reporter.log(
        "LW counts:", "modifier_weighted=", lw_result.get("modifier_weighted_scope_count"),
        "| modifier_components_suppressed=", lw_result.get("modifier_component_suppressed_scope_count"),
        "| contrast_markers_suppressed=", lw_result.get("contrast_marker_suppressed_scope_count"),
        "| contrast_weighted=", lw_result.get("contrast_weighted_scope_count"),
    )
    reporter.log("LW contrast relations:", lw_result.get("contrast_relations"))

    if case.get("expect_lw_modifier") and safe_float(lw_result.get("modifier_weighted_scope_count")) <= 0:
        add_warning(all_warnings, case_id, "LW_EXPECTED_MODIFIER_NOT_WEIGHTED", "This stress case expected a literature modifier, but no scope was weighted.", "H->D-LW")

    if case.get("expect_lw_contrast") and safe_float(lw_result.get("contrast_weighted_scope_count")) <= 0:
        add_warning(all_warnings, case_id, "LW_EXPECTED_CONTRAST_NOT_WEIGHTED", "This stress case expected a licensed contrast, but no scope was contrast-weighted.", "H->D-LW")

    duplicate_pairs = duplicate_contrast_pairs(lw_result)
    if duplicate_pairs:
        add_warning(
            all_warnings, case_id, "LW_DUPLICATE_CONTRAST_PAIR",
            f"Multiple contrast markers were reconstructed as the same pre/post dependency pair: {duplicate_pairs}", "H->D-LW"
        )

    lw_shared_neg = find_shared_negation(lw_result.get("scopes", []))
    if lw_shared_neg:
        add_warning(
            all_warnings, case_id, "LW_INHERITED_SHARED_NEGATION_SCOPE",
            f"LW inherits a negation token shared across multiple sentiment scopes from H->D: {lw_shared_neg}", "H->D-LW"
        )

    all_details.append({
        "case": case,
        "results": {
            "H": json_safe(h),
            "H->D": json_safe(hd),
            "D->H": json_safe(dh),
            "H->D-LW": json_safe(lw_result),
        },
    })

reporter.log()
reporter.log("=" * 120)
reporter.log("SANITY-LABEL SUMMARY")
reporter.log("=" * 120)
for system in SYSTEMS:
    reporter.log(f"{system:8s}", f"{correct_counts[system]}/{len(CASES)}", f"({correct_counts[system] / len(CASES):.3f})")

reporter.log()
reporter.log("=" * 120)
reporter.log("WARNING SUMMARY")
reporter.log("=" * 120)
warning_counts = Counter(w["code"] for w in all_warnings)
if not all_warnings:
    reporter.log("No warnings.")
else:
    for code, count in warning_counts.most_common():
        reporter.log(f"{code}: {count}")
    reporter.log()
    reporter.log("DETAILED WARNINGS")
    reporter.log("-" * 120)
    for warning in all_warnings:
        reporter.log(
            f"Case {warning['case_id']:02d}", f"[{warning['system'] or 'GLOBAL'}]",
            warning["code"], "->", warning["message"]
        )

with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "id", "text", "gold", "focus", "system", "prediction",
            "positive_score", "negative_score", "net_score", "correct",
        ],
    )
    writer.writeheader()
    writer.writerows(all_rows)

with open(WARNINGS_PATH, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["case_id", "system", "code", "message"])
    writer.writeheader()
    writer.writerows(all_warnings)

with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {"systems": SYSTEMS, "cases": CASES, "details": all_details, "warnings": all_warnings},
        f,
        ensure_ascii=False,
        indent=2,
    )

reporter.log()
reporter.log("=" * 120)
reporter.log("OUTPUT FILES")
reporter.log("=" * 120)
reporter.log("Console report:", REPORT_PATH)
reporter.log("Predictions:", CSV_PATH)
reporter.log("Warnings:", WARNINGS_PATH)
reporter.log("Full details:", JSON_PATH)
reporter.close()
