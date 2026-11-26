# Turkish Sentiment Comparison

Comparative evaluation of lexicon-based, dependency-aware, LLM-based,
and decision-model approaches to Turkish sentiment analysis.

The [31 October 2026 paper-readiness review](docs/paper_readiness_2026-10-31.md)
records the remaining scope/alignment issues and the context/data protocol
decisions to resolve before a final experiment.

## Experimental Settings

The project evaluates seven configurations.

1. HisNet
2. HisNet → Dependency Parsing
3. Dependency Parsing → HisNet
4. Literature-weighted HisNet → Dependency Parsing
5. LLM
6. Dependency Parsing → LLM
7. Jev (TypeSafe System One, zero-shot)

Stanford Stanza is used to obtain Turkish dependency parses.

## Project Structure

- `data/`: Raw and processed sentiment datasets
- `src/hisnet/`: HisNet-based sentiment methods
- `src/dependency/`: Stanford Stanza dependency parsing
- `src/llm/`: LLM-based sentiment classification
- `src/evaluation/`: Evaluation metrics and comparison
- `prompts/`: LLM prompts
- `results/`: Experimental results
- `tests/`: Tests

## Dependency Parser

Turkish dependency parsing is performed using Stanford Stanza.

Processors:

- Tokenization
- Multi-word token expansion
- POS tagging
- Lemmatization
- Dependency parsing

## Resumable runs

Ana makale koşusu aynı altı veri setini iki ayrı test grubuyla çalıştırır.
`test_paper_evaluation_classical` dört HisNet yöntemini ve OpenCode
`jev-1.13-free` modelini çalıştırır. `test_paper_evaluation_llm` ise
`qwen3:32b` ile LLM ve DP→LLM koşullarını çalıştırır.
[Makale protokolü](docs/paper_evaluation.md)

```powershell
.venv\Scripts\python.exe -B -u -m tests.test_paper_evaluation_classical --live --output results/paper_classical_jev_v1
.venv\Scripts\python.exe -B -u -m tests.test_paper_evaluation_llm --live --output results/paper_llm_v1
```

Her komuta `--prepare-only` ekleyerek model çağırmadan örnekleri ve ayarları
kaydedebilirsin. Kesintiden sonra aynı komut ve çıktı diziniyle devam edilir.

### Single-dataset runs

The six local systems use a shared SQLite checkpoint by default.

```powershell
.venv\Scripts\python.exe -u -m scripts.run_six_systems --dataset trsav1 --output results/trsav1_six_v1
.venv\Scripts\python.exe -u -m scripts.run_six_systems --dataset beyazperde --output results/beyazperde_six_v1
```

Repeat the **same command and output directory** after an interruption. Successful
predictions are skipped; missing rows run, and failed rows are retried individually.
Classical systems commit every review; LLM systems commit every completed batch.
Only an uncommitted/in-flight batch may need repeating. The default LLM is
`qwen3:32b`, batch size 3, context 8192, request timeout 600 seconds.

Inspect progress without loading or calling a model:

```powershell
.venv\Scripts\python.exe -m scripts.run_six_systems --output results/trsav1_six_v1 --status
```

Add hosted Jev explicitly with `--systems all`. For a balanced TRSAv1 pilot
with 10 reviews from each class, run the following after setting
`TYPESAFE_API_KEY` in the environment.

```powershell
.venv\Scripts\python.exe -u -m scripts.run_six_systems --dataset trsav1 --sample-per-class 10 --seed 42 --model qwen3:4b-instruct --systems all --output results/trsav1_seven_typesafe
```

Use `--systems jev` with the same data, Ollama settings and output directory to
add Jev to an existing compatible checkpoint. Only the selected review text
and a fixed three-class rubric are sent to the TypeSafe API. The default
`jev-1.14.0` is pinned for reproducibility. See [the Turkish usage and validation notes](docs/resumable_evaluation.md)
for model settings, output files and comparison rules.

Use `--jev-provider opencode` for OpenCode's `jev-1.13-free` endpoint. The
runner can send anonymous requests and records the billing source and cost
from successful responses. Use an output directory for this provider and
version. [OpenCode instructions](docs/resumable_evaluation.md#opencode-üzerinden-ücretsiz-jev-113)

## Frozen 500-review classical diagnostic

Run the four classical systems on the existing, verified TRSAv1 sample:

```powershell
.venv\Scripts\python.exe -u -m tests.run_trsav1_classical_500
```

This uses exactly 500 reviews (167 negative, 166 neutral, 167 positive), saves
every review with full diagnostics, and resumes automatically. It does not run
either LLM system. Each classical system executes its own full pipeline.

The default output is `results/trsav1_classical_500_20261031_run01`. Its `report.md`
contains per-class metrics, confusion matrices, paired comparisons, timing,
coverage and numeric/scope checks. `comparison_500.csv`, `errors_500.csv`,
`scope_details_500.csv` and `diagnostic_cases.json` support individual error review.
This is a reused development sample, not an independent final benchmark.

Regenerate reports from saved results without rerunning models:

```powershell
.venv\Scripts\python.exe -m scripts.analyze_classical_500 results/trsav1_classical_500_20261031_run01
```

The ID 148 scope correction was checked using frozen parses and morphology,
without rerunning models. See the [scope check and before/after results](results/id148_scope_check_20261031/report.md).
The original run remains frozen. Its implementation fingerprint predates this
fix, so a future full run must use a new `--output` directory; resuming the old
checkpoint with changed code is intentionally rejected.

```powershell
.venv\Scripts\python.exe -m unittest tests.test_explicit_negation_scope -v
.venv\Scripts\python.exe -m scripts.check_saved_negation_scope --baseline-resolver results/id148_scope_check_20261031/baseline_sentiment_scope_resolver.py
```

## Balanced 1200-review classical development experiment

The balanced runner selects exactly 400 positive, 400 negative and 400 neutral
reviews from `TRSAv1_fixed.csv`, using seed 42. Both LLM conditions are excluded.
For a new run with the current implementation:

```powershell
.venv\Scripts\python.exe -u -m scripts.run_balanced_classical_cached --output results/trsav1_classical_1200_new
```

Morphology and parsing are computed once per review. HisNet→DP and its weighted
variant reuse the same immutable HisNet input; DP→HisNet still selects its own
syntactic candidates before their lexical lookup. Full outputs and shared parses
are saved in SQLite. A successful run also compares full semantic outputs against
12 independent pipeline evaluations (the first review from each class, four methods).
Timing in this mode is shared preprocessing plus composition, not an independent
pipeline speed benchmark. Repeat the same command to resume unchanged code.

The 1 November baseline is frozen in
`results/trsav1_classical_1200_20261101_shared_baseline`; 616 completed HisNet
rows were imported from the matching partial independent run. Future code changes
require a new output directory. Analysis can be regenerated without inference:

```powershell
.venv\Scripts\python.exe -m scripts.analyze_classical_500 results/trsav1_classical_1200_20261101_shared_baseline
```

Despite its historical filename, the analyzer derives the sample size, output
filenames and majority baseline from the saved cases. This is a development
experiment on a previously used benchmark, not an independent final paper test.

See the [1200-review analysis and verified corrections](results/trsav1_classical_1200_20261101_corrections/report.md)
for the frozen baseline, before/after metrics, remaining limitations and validation.

## Disjoint 3000-review classical development experiment

After the [finite predicate correction](results/finite_predicate_fix_20261101/report.md),
run 1000 examples per class while excluding every text from the original 1200:

```powershell
.venv\Scripts\python.exe -u -m scripts.run_balanced_classical_cached --sample-per-class 1000 --exclude-sample results/trsav1_classical_1200_20261101_shared_baseline/sample_1200.csv --output results/trsav1_classical_3000_20261101
```

Repeat the command to resume unchanged code; append `--status` for read-only
progress or `--prepare-only` to freeze the sample without inference. The four
classical decisions share immutable morphology/parsing. Neither LLM system runs.
Sampling uses seed 42 after text-level exclusion. Duplicate selected texts are
skipped in fixed class order; conflicting labels elsewhere in the source are
reported, without relabelling. Source IDs, exclusions, hashes and code snapshots
are saved alongside the checkpoint. This remains a development experiment on a
previously used corpus, not an independent final paper test.

The 1 November run moved to three local worker processes after 373 completed
reviews. Their 1492 outputs and parse records were preserved and imported into
`results/trsav1_classical_3000_20261101_parallel`. Resume that run with:

```powershell
.venv\Scripts\python.exe -u -m scripts.run_parallel_classical_cached --workers 3 --torch-threads 1 --source results/trsav1_classical_3000_20261101 --sample-per-class 1000 --exclude-sample results/trsav1_classical_1200_20261101_shared_baseline/sample_1200.csv --output results/trsav1_classical_3000_20261101_parallel
```

Workers receive text and optional immutable preprocessing, never gold labels.
Only the parent process writes the checkpoint. A fresh parallel preflight matched
all 12 semantic outputs from three original cases. Final independent checks also
cover new cases and each worker. Times are summed per-worker elapsed durations,
not CPU times. Long interruptions on 1–3 November contaminated this run's timings;
raw values are preserved and must not be used for speed comparisons. See its
`timing_caveats.json`. Use `scripts.audit_classical_run OUTPUT --progress` for
counts without model loading.

Completed on 3 November: all 12,000 predictions succeeded, no LLM calls, 28 final
independent semantic checks passed, and no numeric/scope invariant violations.
See the [detailed findings and next steps](results/trsav1_classical_3000_20261101_parallel/findings.md),
[complete metric report](results/trsav1_classical_3000_20261101_parallel/report.md),
[same-sample correction comparison](results/trsav1_classical_3000_20261101_finite_ablation/report.md),
and [final audit](results/trsav1_classical_3000_20261101_parallel/final_audit.json).
