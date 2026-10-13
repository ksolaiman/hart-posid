# POSID: Target-Task-Unsupervised Person-Attribute Extraction from Heterogeneous Public-Safety Text

POSID recovers explicitly stated person attributes from public-safety text without
supervised training or fine-tuning on the target task. It extracts gender, race,
height, clothing items, and clothing descriptors from heterogeneous narratives.

The system produces structured representations of claims made in the source
text. It does not infer unstated attributes or establish that a source statement
is factually correct or complete.

## Method at a glance

```text
document -> evidence localization -> provenance-controlled scope
         -> structured recovery and traversal -> person association
```

POSID evaluates six candidate-localization configurations: regular expressions (RE),
Word2Vec, WordNet, zero-shot natural-language inference (NLI), RE + WordNet, and RE + NLI.

Every selector returns the retrieval output, score, triggering query, and full or
expanded context. Combined configurations concatenate ordered RE and semantic
candidates while preserving each candidate's downstream scope.

The clothing parser supports two traversal policies. **Continuous** traversal keeps
scanning after unsupported tokens; **early stop** terminates clothing recovery once
parsing has begun and local syntactic evidence is no longer supported.

## Data

InciText is released separately on [Zenodo](https://zenodo.org/records/23334976).
Its metadata, annotations, and document paths match the repository loaders.

The evaluated cohorts are Base28, Challenge14, Generated30, Compact500, and
Facebook21, spanning different structures, sources, and description conditions.

A typical local layout is:

```text
project-root/
├── InciText/
│   ├── Annotation/
│   └── DataSet_Attribute_Extraction/
└── src/                    # this repository
```

The dataset may be stored elsewhere by passing its location through
`--incitext-root`.

## Environment

Create the declared Conda environment:

```bash
conda env create -f environment.yml
conda activate hart39
```

Runs require local NLTK tokenizer, WordNet, POS-tagger, and Word2Vec sample
resources. NLI methods require `facebook/bart-large-mnli`. The runner prints a
resource report and records unavailable methods in `blocked_methods.json`.

## Quick start

Run all six configurations on Base28 with continuous traversal:

```bash
python -u scripts/reproduce_legacy_posid.py \
  --incitext-root ../InciText \
  --subset non_generated_28 \
  --output-dir outputs/base28/continuous \
  --methods re_posid wordnet_posid re_wordnet_posid \
            word2vec_posid sbert_posid re_sbert_posid
```

For early stopping, use a distinct output directory and append
`--stop-on-unexpected-pos` to the command.

The runner also supports selecting individual methods and cohorts, alternate
metadata and annotation filenames, and the optional
`--candidate-selector-non-RE-first-match-only` semantic-selection mode. Run
`python scripts/reproduce_legacy_posid.py --help` for the complete interface.

## Additional entry points

Challenge14 uses the prepared extension metadata:

```bash
python -u scripts/run_new14_posid.py continuous
python -u scripts/run_new14_posid.py early_stop
```

Prepare and run Facebook21:

```bash
python -u scripts/prepare_facebook21_metadata.py
python -u scripts/run_facebook21_posid.py continuous
python -u scripts/run_facebook21_posid.py early_stop
```

The optional stateless generative baseline is `scripts/run_llm_baseline.py`; it
supports `--dry-run`, `--resume`, and `--score`. Preparation and diagnostic
utilities are also kept under `scripts/`.

## Outputs and results

`results.json` is the main evaluation report. For every completed method, it contains
primary document-level metrics for gender, race, height, clothing Attr-Only, and
clothing Attr-Value. Attr-Only evaluates item recovery; Attr-Value also evaluates
descriptors. It also records person-aware ownership and fragmentation diagnostics,
processing and gold-unit counts, and per-document exceptions. Provenance-aware
evaluation handles explicitly tagged collective evidence while preserving ordinary
person-specific evidence.

Other artifacts are intentionally direct:

- `*_predictions.json`: person-level structured predictions for one method
- `resource_status.json` and `blocked_methods.json`: runtime availability
- `run_manifest.json`, where provided: command and run provenance

## Repository map

- `hart/candidates/` - RE, Word2Vec, WordNet, NLI, and combined selectors
- `hart/extractors/` - POSID extraction, traversal, and person association
- `hart/data/` - metadata, document, and legacy-annotation loaders
- `hart/evaluation/` - primary, provenance-aware, and person-aware evaluation
- `hart/schema/` - canonical annotation and prediction models
- `scripts/` - reproduction, cohort, preparation, and baseline drivers
- `tests/` - focused unit and regression coverage

## Tests

```bash
python -m pytest -q
```

## Paper and dataset

- Paper: [arXiv:2507.03949](https://arxiv.org/abs/2507.03949)
- Dataset: [InciText on Zenodo](https://zenodo.org/records/23334976)
