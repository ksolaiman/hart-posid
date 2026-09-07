# HART / POSID Development Instructions

This repository implements and extends the POSID/HART research codebase.

## Source of truth

The repository structure and algorithm behavior should follow the verified design decisions captured in this repository.

Do not silently redesign, modernize, or substitute algorithms.

When legacy behavior is uncertain, preserve the uncertainty in comments or documentation rather than inventing an implementation detail.

## Current development scope

At the current stage, implement only candidate sentence selection.

Current intended files:

- `src/hart/candidates/__init__.py`
- `src/hart/candidates/base.py`
- `src/hart/candidates/regex.py`
- `src/hart/candidates/word2vec.py`
- `src/hart/candidates/wordnet.py`
- `src/hart/candidates/zero_shot.py`
- `tests/test_candidates_sample.py`

Do not implement POSID property extraction yet unless explicitly instructed.

## Repository design rule

Keep algorithm implementation separate from experiment configuration.

Reusable candidate classes should not hardcode experiment-specific query phrases, thresholds, or model checkpoints.

Experiment-specific values will later be supplied through configs, tests, or pipeline calls.

## Tokenization

Use NLTK tokenization:

- `nltk.sent_tokenize`
- `nltk.word_tokenize`

Do not use regex tokenization, spaCy tokenization, or another tokenizer inside candidate modules unless explicitly requested later.

Provide shared helpers in `base.py`, including:

- `split_sentences()`
- `tokenize_words()`

## Parameterization

Do not hardcode:

- query phrases such as `wear`, `wearing`, or `clothes`
- thresholds such as `0.5`, `0.85`, or `0.9`

Queries should be supplied to `select(...)`.

Thresholds should be explicit constructor/configuration parameters.

## Candidate interface

Candidate methods should return a common structured result, such as:

- sentence id
- sentence/span text
- score
- matched query

The current shared abstraction is expected to include:

- `CandidateSentence`
- `CandidateSelector`

## Regex candidate selection

Regex is a lightweight exact/pattern candidate-selection method.

The historical implementation used a pattern conceptually equivalent to:

`wearing[^.]*`

The reusable selector should therefore return the matched span beginning at the supplied query and continuing until the next period, rather than returning the entire containing sentence.

Generic behavior should be equivalent to:

`query[^.]*`

Requirements:

- safely escape caller-supplied query text
- do not hardcode `wearing`
- preserve caller-supplied queries as parameters

Example for query `wearing`:

`wearing a red shirt and blue pants`

## Word2Vec candidate selection

Historical resource:

- NLTK `word2vec_sample`
- `models/word2vec_sample/pruned.word2vec.txt`
- loaded with `gensim.models.KeyedVectors.load_word2vec_format(..., binary=False)`

Historical behavior:

- split text into sentences with NLTK
- split each sentence into tokens with NLTK
- compare tokens against the query using `vectors.similarity(token, query)`
- sentence score corresponds to the best valid token-query similarity
- preserve legacy threshold comparison: `score >= threshold`
- return only the first qualifying sentence

OOV handling:

- skip OOV token/query comparisons
- do not invent a score for OOV tokens
- if no token in a sentence yields a valid similarity score, that sentence has no valid score and should not be returned

Keep the vectors injected into `Word2VecSelector`.

Do not download or load vectors inside the selector implementation.

Historical experiment values such as query `clothes` and threshold `0.5` belong outside the reusable class.

## WordNet candidate selection

Use:

- `nltk.corpus.wordnet`
- noun synsets
- Wu-Palmer similarity

Support a configurable `synset_strategy` with these options:

- `"first"`: compare only the first noun synset for each term
- `"max_all"`: compare all valid noun-synset pairs and take the maximum Wu-Palmer score
- `"average"`: average all valid Wu-Palmer scores

For legacy reproduction calls/tests, use:

`synset_strategy="first"`

Important unresolved-status note:

Historical WordNet candidate synset strategy remains unresolved; first/max-all/average are supported, with first used for legacy reproduction until original code confirms the behavior.

Do not present `"first"` as historically confirmed.

Known POSID extraction behavior includes:

`wn.synsets("color", "n")[0]`

but this belongs to later color/property extraction and should not be inserted into WordNet candidate selection as proof of the candidate-stage strategy.

Historical experiment value `0.9` is configuration, not a reusable class default.

## Zero-shot candidate selection

The implementation is exposed as `ZeroShotSelector` / `zero_shot.py`.

The paper describes Sentence-BERT sentence-level encoding as part of the zero-shot natural language inference (NLI) process, but the code/module name should remain zero-shot for now.

Preserve the paper/legacy process:

1. Split the document into sentences.
2. Process one sentence at a time.
3. Treat the current sentence as the NLI premise.
4. Construct a hypothesis from the caller-supplied query using the template:
   `This text is about {}`
   Example for query `clothes`:
   `This text is about clothes`
5. The paper describes both the sentence and the hypothesis as being encoded with the sentence-level encoder Sentence-BERT.
6. Use the NLI classifier to obtain the zero-shot class probability for the query/hypothesis.
7. Use strict `score > threshold`.
8. Return only the first qualifying sentence; otherwise continue to the next sentence.

Historical/reproduction framing:

- implementation name in HART: `ZeroShotSelector`
- task formulation: Multi-NLI-style zero-shot NLI
- implementation path: Hugging Face `pipeline("zero-shot-classification")`
- sentence/hypothesis formulation must be preserved
- the paper describes Sentence-BERT sentence-level encoding as part of this process
- do not replace this method with plain embedding cosine similarity

Keep the classifier injected into `ZeroShotSelector`.

Do not hardcode query phrases or thresholds inside the reusable selector.

Historical experiment threshold `0.85` belongs in configuration/test setup.

## Stacked behavior

The legacy system used fallback behavior where an exact/pattern method could run first and a semantic method was used only when the primary method returned no candidate.

Do not add a stacked module until explicitly instructed.

When added later, keep primary and fallback candidate selectors composable rather than duplicating algorithm logic.

## Data and annotation loading

Current stored gold remains the existing **Schema 1.0** file:

`InciText/Annotation/annotations.json`

Do not rewrite the legacy annotations into Schema 2.0 merely to run the current experiments.

The repository layout has `src/` and `InciText/` as sibling directories under the same project root.

Use a loader/adapter boundary so on-disk annotation schemas are isolated from the rest of the system:

```text
src/hart/
├── data/
│   ├── document.py
│   ├── legacy_adapter.py
│   └── loaders/
│       ├── base.py
│       ├── legacy_annotations.py
│       ├── facebook.py
│       └── schema2.py
│
└── schema/
    ├── models.py
    └── normalize.py
```

`facebook.py` and `schema2.py` may remain empty placeholders until those formats are actually used.

Do not require `manifest.py` or `validators.py` at the current stage. Add them only if a concrete need appears.

### Loader contract

All annotation loaders must convert their source format into the same canonical in-memory representation.

The downstream extractors, pipelines, evaluation, and experiment code must depend only on that canonical representation, not on Schema 1.0 field layout.

Canonical rule:

`document_id -> list[PersonAnnotation]`

Every document maps to a list even when there is only one annotated person.

The single known legacy document containing two people must produce two `PersonAnnotation` records for that document.

If Schema 1.0 does not provide an explicit stable person identifier, runtime-only identifiers may be generated deterministically as:

- `<document_id>:0`
- `<document_id>:1`
- etc.

Do not rewrite those generated runtime IDs back into the original `annotations.json`.

### Schema isolation

A future annotation schema should require only a new loader/adapter implementation.

Do not require changes to extractors, experiment runners, normalization/evaluation logic, or benchmark pipelines simply because the on-disk annotation schema changes.

Schema 1.0 storage and the canonical runtime model are intentionally separate concepts.

### Paths

Do not hardcode user-specific absolute filesystem paths.

Resolve dataset locations from explicit arguments/configuration. A repository-relative default may point from the project root to:

`InciText/Annotation/annotations.json`

when appropriate.

## Data and annotation loading architecture

The repository source tree and the legacy InciText dataset directory are siblings in the same parent directory.

Expected local layout:

```text
<project-parent>/
├── src/
│   └── hart/
│       ├── candidates/
│       ├── data/
│       └── schema/
└── InciText/
    ├── Annotation/
    │   ├── annotations.json
    │   └── annotation_load_example.py
    ├── consolidated_report_metadata.json
    ├── report_metadata.json
    ├── DataSet_Attribute_Extraction/
    ├── Generated_data_Annotations/
    ├── Generated_flash/
    ├── Generated_flash_2/
    └── ...
```

Do not copy or rewrite the legacy annotation file into a new schema just to run the current experiments.

Current stored gold is:

`InciText/Annotation/annotations.json`

using the existing legacy Schema 1.0.

The code must isolate storage schema from runtime representation:

```text
annotations.json (Schema 1.0)
        ↓
LegacyAnnotationLoader
        ↓
canonical in-memory representation
        ↓
extractors / runs / evaluation / experiments
```

Required design:

- keep the original Schema 1.0 annotations unchanged on disk
- add a loader/adapter boundary that converts legacy annotations into the canonical in-memory model
- downstream extractors, pipelines, and evaluation code must depend only on the canonical model, not on Schema 1.0 field layout
- future annotation schemas should require only a new loader/adapter, not changes to experiment or evaluation logic

Current intended data/schema files:

```text
src/hart/
├── data/
│   ├── document.py
│   ├── loaders.py
│   └── legacy_adapter.py
└── schema/
    ├── models.py
    └── normalize.py
```

Do not create a separate `data/loaders/` directory at this stage.

For the current project size, keep loading/adaptation code together inside the single `data/` package:
- `document.py` for document/data objects
- `loaders.py` for file-reading/loading logic
- `legacy_adapter.py` for Schema 1.0 → canonical in-memory adaptation

A dedicated `data/loaders/` subpackage should only be introduced later if multiple distinct loaders make `loaders.py` unwieldy.

Do not add `manifest.py` or `validators.py` unless a concrete need emerges.

### Authoritative InciText document index

Use only:

`InciText/consolidated_report_metadata.json`

as the authoritative source for discovering which InciText documents belong to the dataset and which file paths should be loaded.

Rules:

- do not use `InciText/report_metadata.json` for document discovery
- do not recursively scan `InciText/` for source documents
- do not infer dataset membership from folder contents
- do not substitute alternate files if a listed file is missing
- only load documents whose paths are explicitly present in `consolidated_report_metadata.json`
- resolve listed paths relative to the `InciText/` root unless the metadata explicitly establishes otherwise
- validate that each resolved path stays inside the `InciText/` root
- if a listed file is missing, report the missing path instead of searching elsewhere
- `annotations.json` remains the annotation source; `consolidated_report_metadata.json` determines the authoritative document/path inventory

The intended ingestion flow is:

```text
InciText/consolidated_report_metadata.json
                ↓
      authoritative document paths
                ↓
             loaders.py
                ↓
             Document
```

and independently:

```text
InciText/Annotation/annotations.json
                ↓
        legacy_adapter.py
                ↓
      PersonAnnotation(s)
```

The two streams should be joined by `document_id`.

### Canonical person representation

Every document must normalize to a list of person annotations:

```python
dict[str, list[PersonAnnotation]]
```

This rule applies even when a document contains only one annotated person.

The one known legacy document containing two people should produce two `PersonAnnotation` objects for that same `document_id`.

If the legacy record does not contain a stable person identifier, runtime-only IDs may be generated deterministically as:

```text
<document_id>:0
<document_id>:1
```

These runtime IDs are internal identifiers only and must not require rewriting `annotations.json`.

The loader must handle the legacy top-level list structure directly and preserve both people in the two-person document.

### Path handling

Do not hardcode absolute user-machine paths.

Because `src/` and `InciText/` are siblings, code/tests may resolve the dataset path relative to the repository/project parent or accept the dataset root as a caller-supplied/configuration path.

Prefer explicit configurable paths over embedding local filesystem paths in reusable modules.

## Testing

The current smoke-test text is:

> The suspect was described as a short Asian male, about 145 lbs. wearing a green jacket and blue pants.

The smoke test belongs at:

`tests/test_candidates_sample.py`

not inside `src/hart/candidates/`.

The smoke test should exercise:

- RegexSelector
- Word2VecSelector
- WordNetSelector
- ZeroShotSelector

Historical experiment settings may be used in the test setup:

- regex query: `wearing`
- Word2Vec query: `clothes`
- Word2Vec threshold: `0.5`
- WordNet query: `clothes`
- WordNet threshold: `0.9`
- WordNet strategy: `first`
- zero-shot label/query: `clothes`
- zero-shot threshold: `0.85`

Do not claim expected numeric Word2Vec, WordNet, or zero-shot scores until they are actually run with the corresponding resources.

## Change discipline

When asked to implement a file:

1. Read this file and `docs/HART_DEV_CONTEXT.md`.
2. Modify only the requested file(s).
3. Do not introduce new abstractions without need.
4. Do not silently change legacy semantics.
5. Run only the relevant test(s).
6. Report any ambiguity before resolving it by assumption.
7. Preserve small, reviewable diffs.

## Git discipline

Do not run `git add`, `git commit`, or any other staging/commit command automatically.

After each completed file change:

1. Run the relevant tests.
2. Show the exact diff for the changed file(s).
3. Summarize what changed.
4. Wait for explicit user review and approval.
5. Only after the user explicitly approves staging, run:
   `git add <approved-file>`
6. Show the staged diff/status.
7. Wait for explicit user approval before committing.
8. Only after explicit approval, run `git commit`.

If a staged file is modified again, do not re-stage it automatically. Show the new diff and wait for approval again.

At the end of a task, report:

- `git status --short`
- unstaged modified files
- staged files
- `git diff --stat`
- `git diff --cached --stat`
