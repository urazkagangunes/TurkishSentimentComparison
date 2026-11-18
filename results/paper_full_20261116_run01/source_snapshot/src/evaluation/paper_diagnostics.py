"""Automatic proxies, not human gold negation/scope detection metrics."""
import re
from collections import Counter

TEXT_SLICES = ("auto_explicit_negation", "auto_suffix_negation", "auto_light_verb_negation",
               "auto_multiple_negation", "auto_any_negation", "auto_no_negation", "auto_contrast",
               "auto_negation_and_contrast", "auto_modifier", "auto_mixed_cues",
               "emoji_present", "short_1_5", "medium_6_15", "long_16_plus",
               "single_sentence_proxy", "multi_sentence_proxy", "surface_noise_proxy", "source_label_conflict")
MODEL_SLICES = ("zero_score", "robust_or_fallback", "unmatched_opinion_root",
                "unaligned_scope", "negation_flip", "suppressed_as_target",
                "modifier_weighted", "contrast_weighted", "mixed_lexicon_evidence")


def lower(text):
    return text.replace("I", "ı").replace("İ", "i").lower()


def text_flags(text, conflict=False):
    """Model-independent surface heuristics fixed before scoring; imperfect by design."""
    words = re.findall(r"[^\W_]+", lower(text))
    explicit = [w for w in words if re.fullmatch(r"(?:değil|yok)(?:di|dı|dir|tur|tu|muş|miş|um|im|uz|iz|ler|lar|du|tü|tür)*", w)]
    suffix = [w for w in words if re.search(r"(?:ma|me)(?:dı|di|du|dü|mış|miş|muş|müş|yacak|yecek|yın|yin|z|zsınız|zsiniz)|m[ıiuü]yor", w)]
    negative_words = set(explicit + suffix)
    negative_count = sum(w in negative_words for w in words)
    contrast = bool(set(words)&{"ama", "fakat", "ancak", "lakin", "oysa", "rağmen"})
    hasneg = bool(explicit or suffix)
    positive_cues = {"iyi", "güzel", "harika", "mükemmel", "başarılı", "memnun"}
    negative_cues = {"kötü", "berbat", "rezalet", "başarısız", "pişman"}
    multi_sentence = len(re.findall(r"[.!?]+(?:\s|$)", text)) >= 2
    return dict(auto_explicit_negation=bool(explicit), auto_suffix_negation=bool(suffix),
        auto_light_verb_negation=bool(re.search(r"\b(?:tavsiye|memnun|pişman|rahatsız)\s+(?:etm|olm)", lower(text))),
        auto_multiple_negation=negative_count >= 2, auto_any_negation=hasneg,
        auto_no_negation=not hasneg, auto_contrast=contrast, auto_negation_and_contrast=hasneg and contrast,
        auto_modifier=bool(set(words)&{"çok", "biraz", "pek", "aşırı", "az", "oldukça"}),
        auto_mixed_cues=bool(set(words)&positive_cues) and bool(set(words)&negative_cues),
        emoji_present=bool(re.search("[\U0001f000-\U0001faff\u2600-\u27bf]", text)),
        short_1_5=len(words)<=5, medium_6_15=6<=len(words)<=15, long_16_plus=len(words)>=16,
        single_sentence_proxy=not multi_sentence, multi_sentence_proxy=multi_sentence,
        surface_noise_proxy=bool(re.search(r"([^\W\d_])\1{2,}|[!?]{3,}|\ufffd", lower(text))),
        source_label_conflict=bool(conflict))


def diagnostics(result, lookup_misses=None):
    scopes = result.get("scopes", [])
    operations = Counter(op for scope in scopes for op in scope.get("operations", []))
    unmatched = Counter(w.get("root", "") for w in result.get("unmatched_words", []) if isinstance(w, dict))
    if lookup_misses is not None and "unmatched_words" not in result:
        unmatched = Counter(lookup_misses)
    sources = Counter(w.get("analysis_source", "unspecified") for w in result.get("morphological_analyses", []))
    p, n = result.get("positive_score"), result.get("negative_score")
    flags = dict(zero_score=p == n == 0, robust_or_fallback=bool(sources["robust"]+sources["fallback"]),
        unmatched_opinion_root=bool(set(unmatched)&{"sev", "beğen", "bayıl", "nefret"}),
        unaligned_scope=any(s.get("alignment_status")=="unmatched" for s in scopes),
        negation_flip=operations["negation_flip"]>0, suppressed_as_target=operations["suppressed_as_target"]>0,
        modifier_weighted=any(k.startswith("literature_modifier_weight:") for k in operations),
        contrast_weighted=any(k.startswith("literature_contrast_weight:") for k in operations),
        mixed_lexicon_evidence=p is not None and n is not None and p>0 and n>0)
    return dict(flags=flags, operations=dict(operations), unmatched_roots=dict(unmatched), morphology_sources=dict(sources),
                positive_score=p, negative_score=n, lexicon_coverage=result.get("lexicon_coverage"),
                sentiment_coverage=result.get("sentiment_coverage"),
                candidate_lexicon_coverage=result.get("candidate_lexicon_coverage"),
                unaligned_scopes=sum(s.get("alignment_status")=="unmatched" for s in scopes),
                scope_count=len(scopes),
                scope_operations=[dict(word=s.get("word"), root=s.get("root"),
                    target=s.get("target"), negations=s.get("negations", []),
                    alignment_status=s.get("alignment_status"), operations=s.get("operations", []))
                    for s in scopes if s.get("operations") or s.get("alignment_status")=="unmatched"])
