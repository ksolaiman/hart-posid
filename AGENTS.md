# HART / POSID Development Instructions

This repository implements and extends the POSID/HART research codebase.

## Source of truth

The repository structure and algorithm behavior should follow the verified design decisions captured in this repository.

Do not silently redesign, modernize, or substitute algorithms.

When legacy behavior is uncertain, preserve the uncertainty in comments or documentation rather than inventing an implementation detail.

## Current development scope

Candidate retrieval and the current data-ingestion/annotation-adaptation layer are completed checkpoints.

The extraction layer has now been implemented and is in **regression-stabilization**.

Current work should be limited to:
- fixing concrete extraction regressions exposed by approved deterministic tests
- preserving verified/reconstructed semantics already documented here
- avoiding broad redesign of completed candidate/data/extraction architecture

Do not reopen or redesign the completed data layer.

Do not broadly redesign completed candidate modules. Candidate changes should be limited to concrete, approved fixes required by the current regression-stabilization work.

Do not implement evaluation, LLM adapters, benchmark runners, experiment configs, bootstrap analysis, cost analysis, or error analysis during this stabilization checkpoint unless explicitly instructed later.

After extraction regressions are stable, the next major milestone is the legacy reproduction/evaluation harness.


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

Candidate methods return a common structured result through `CandidateSentence`.

Confirmed semantics:

- `sentence_id`: deterministic candidate/context identifier
- `text`: the candidate returned by the retrieval algorithm
- `context_text`: optional surrounding candidate sentence/context used by downstream extraction
- `score`: candidate score
- `query`: matched/query phrase

Important distinction:

```text
CandidateSentence.text
    = retrieval output

CandidateSentence.context_text
    = surrounding candidate sentence/context for downstream extraction
```

For the historical Regex path, preserve the recovered span behavior:

```text
text = "wearing a red shirt and blue pants"
```

while `context_text` should retain the full descriptive candidate context needed by POSID.

For Word2Vec, WordNet, and ZeroShot, which already return sentence-level candidates, `context_text` may equal `text` or be omitted when `text` itself is sufficient.

The current shared abstraction includes:

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

Do not redefine the historical Regex retrieval output merely to satisfy downstream POSID. Preserve the matched span in `CandidateSentence.text`; use `CandidateSentence.context_text` for the surrounding candidate context.

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

Recovered legacy artifact confirms the historical candidate behavior:

1. obtain the **first noun synset of the query** (historically `clothes`)
2. tokenize each sentence with NLTK
3. for each token, obtain **all noun synsets of that token**
4. compare the query synset against each token synset using Wu-Palmer similarity
5. if any valid score is `>= threshold`, return that sentence immediately
6. otherwise continue sentence-by-sentence

Conceptually:

```python
query_synset = wn.synsets(query, "n")[0]

for sentence in sent_tokenize(text):
    for token in word_tokenize(sentence):
        for token_synset in wn.synsets(token, "n"):
            score = query_synset.wup_similarity(token_synset)
            if score is not None and score >= threshold:
                return [sentence]
```

This asymmetric historical behavior is now verified from the recovered 2021/2023 artifact and should be used for legacy reproduction.

The existing reusable selector may continue to support exploratory/configurable strategies such as:

- `"first"`: first noun synset for both query and token
- `"max_all"`: maximum valid score across all noun-synset pairs
- `"average"`: average valid score across all noun-synset pairs

However, none of those should be mislabeled as the verified legacy behavior. The implementation should add/preserve a clearly identified legacy mode matching the asymmetric behavior above.

Known separate POSID extraction behavior includes:

`wn.synsets("color", "n")[0]`

for later clothing/color interpretation.

Historical experiment values such as query `clothes` and threshold `0.9` remain configuration, not reusable class defaults.

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

`StackedSelector` is implemented as fallback composition.

Confirmed behavior:

```text
primary selector
    -> if candidates are found, return them
    -> otherwise run fallback selector
```

Requirements:

- do not concatenate fallback results when the primary succeeds
- keep primary and fallback selectors composable
- do not duplicate candidate algorithm logic inside `StackedSelector`
- support separately configurable query lists:
  - `primary_queries`
  - `fallback_queries`
- do not hardcode historical query phrases in the reusable stacked implementation

Separate query lists are required because historical stacked experiments can use different queries for Regex and semantic selectors.


## Data and annotation loading architecture

The repository source tree and the legacy InciText dataset directory are siblings under the same project root.

Expected local relationship:

```text
<project-root>/
├── src/
│   └── hart/
│       ├── candidates/
│       ├── data/
│       └── schema/
└── InciText/
    └── Annotation/
        ├── annotations.json
        ├── consolidated_annotation_file.json
        ├── consolidated_report_metadata.json
        └── report_metadata.json
```

Do not hardcode user-specific absolute filesystem paths.

### Current data/schema package structure

Use the single `data/` package:

```text
src/hart/
├── data/
│   ├── document.py
│   ├── loaders.py
│   ├── legacy_adapter.py
│   └── consolidate_annotations.py
└── schema/
    ├── models.py
    └── normalize.py
```

Do not create a separate `data/loaders/` subpackage at this stage.

Do not add `manifest.py` or `validators.py` unless a concrete need emerges.

Responsibilities:

- `document.py` = lightweight document/data models
- `loaders.py` = authoritative metadata loading and lazy/on-demand document text access
- `legacy_adapter.py` = legacy Schema 1.0 annotation → canonical in-memory adaptation
- `consolidate_annotations.py` = one-purpose preparation utility that builds the consolidated legacy annotation file
- `schema/models.py` = canonical runtime models
- `schema/normalize.py` = conservative normalization at the adapter/evaluation boundary

### Authoritative document inventory

Use only:

`InciText/Annotation/consolidated_report_metadata.json`

as the authoritative source for dataset membership and source-document paths.

Rules:

- do not use `InciText/Annotation/report_metadata.json` for document discovery
- do not recursively scan `InciText/` for source documents
- do not infer dataset membership from folder contents
- do not substitute alternate files if a listed file is missing
- only load documents whose paths are explicitly present in `consolidated_report_metadata.json`
- resolve source paths against the `InciText/` dataset root
- validate that resolved paths remain inside the `InciText/` root
- if a listed file is unavailable or missing, report it rather than searching elsewhere

The document flow is:

```text
InciText/Annotation/consolidated_report_metadata.json
                ↓
      lightweight DocumentRecord index
                ↓
             loaders.py
                ↓
        load_document(...) on demand
```

### Lazy document text loading

Do not eagerly read all InciText document contents during initialization.

Load only the lightweight metadata/index first.

Read source text only when a specific document is requested, or when an explicit benchmark/test intentionally iterates the corpus.

Production loading must separate document indexing from document text access.

A generator/iterator may be provided for explicit full-corpus runs, but index construction itself must remain metadata-only.

### Annotation consolidation

The legacy root file:

`InciText/Annotation/annotations.json`

is incomplete for the current corpus and is not the runtime annotation source.

Relevant per-folder annotation files are consolidated into:

`InciText/Annotation/consolidated_annotation_file.json`

using:

`src/hart/data/consolidate_annotations.py`

The consolidation utility must be metadata-driven.

Use:

`InciText/Annotation/consolidated_report_metadata.json`

to determine authoritative document IDs and the relevant annotation paths/source folders.

Required behavior:

- do not recursively scan `InciText/` for annotation files
- derive annotation inputs only from authoritative metadata
- skip folders with no standalone annotation file, including `Generated_flash_2`
- merge annotations by `document_id`
- retain only authoritative document IDs
- preserve each document's list-of-persons structure exactly during consolidation
- accept identical duplicate annotations without duplication
- fail loudly on conflicting duplicate annotations
- never modify original annotation files
- write only `consolidated_annotation_file.json`

Verified consolidated annotation counts:

```text
Documents:       558
People:          553
Empty lists:       6
Single-person:   551
Two-person:        1
```

### Universal extractor prediction contract

All extraction methods must emit the same canonical **person-level prediction** representation.

The prediction-side model belongs in `src/hart/schema/models.py` and should mirror the person-level structure of canonical gold annotations.

Confirmed conceptual model:

```python
@dataclass
class PersonPrediction:
    document_id: str
    person_id: str | None
    role: str | None
    attributes: dict
    metadata: dict | None = None
```

Canonical prediction rule:

```python
dict[str, list[PersonPrediction]]
```

This is the primary output contract for Regex, Dependency, POSID, and later LLM extractors.

Do **not** make per-attribute records such as `AttributePrediction` the primary extractor interface. Person identity and structured attributes such as clothing item/color associations should remain intact until evaluation.

Gold and prediction sides should therefore be conceptually parallel:

```text
legacy annotation source
        ↓
legacy_adapter.py
        ↓
PersonAnnotation

raw document
        ↓
extractor
        ↓
PersonPrediction
```

Per-attribute flattening belongs later in the evaluation layer for metrics, stratification, bootstrap analysis, and error analysis.

`sentence_id` and `evidence_text` are not required fields in the core prediction contract. Method-specific provenance may be stored optionally under `metadata` when useful.

### Runtime annotation source and canonical representation

At runtime, `legacy_adapter.py` must consume:

`InciText/Annotation/consolidated_annotation_file.json`

not the incomplete root `annotations.json`.

The storage schema and runtime representation are intentionally separate.

Canonical rule:

```python
dict[str, list[PersonAnnotation]]
```

Every `document_id` maps to a list, even when there is only one person.

The one known two-person document must yield two separate `PersonAnnotation` objects under the same `document_id`.

If the legacy source lacks a stable person identifier, deterministic runtime-only IDs may be generated as:

```text
<document_id>:0
<document_id>:1
```

Do not write generated runtime IDs back to legacy annotation files.

Legacy string `"null"` values should be normalized at the adapter/normalization boundary so downstream canonical objects do not need to special-case them.

Do not rewrite all old annotations into Schema 2.0 merely to run the current experiments.

Future annotation schemas should require only a new loader/adapter that emits the same canonical model. Extractors, pipelines, evaluation, and experiment code should remain unchanged.

### Data-layer checkpoint

The current data layer has been implemented and tested.

Implemented pieces include:

- metadata-only `DocumentRecord`
- lazy `load_document_index()`
- on-demand `load_document()`
- generator-based `load_documents()`
- path containment checks
- legacy annotation consolidation
- canonical `PersonAnnotation` adaptation
- data-loading tests

Verified test state:

```text
Document index:          558 unique IDs
Annotation documents:    558
People:                  553
Empty lists:               6
Single-person documents: 551
Two-person documents:      1
Metadata/annotation IDs: exact match
```

Do not modify the completed data layer unless a concrete bug or new requirement appears.


## POSID fixed-property reconstruction rule

Recovered legacy artifacts provide only partial historical regex logic for fixed properties, especially gender/race.

The surviving gender/race notebook is an intermediate prototype and contains male-specific logic. It is not sufficient by itself to reproduce the broader gender/race behavior described and evaluated in the paper.

Confirmed implementation decision:

- preserve any directly verified legacy regex behavior where available
- conservatively reconstruct the missing **symmetric gender/race rules** needed to match the paper/task
- do not pretend reconstructed rules are directly recovered historical code
- mark reconstructed behavior clearly in code comments/tests/documentation
- keep the reconstruction narrow and task-driven rather than expanding into a new rule system
- do not infer or add unrelated attributes merely because they exist in the annotation schema

This reconstruction policy applies to `POSIDExtractor` fixed-property handling. It does not redefine the separate modern Regex baseline.

## POSID candidate scope and person resolution

### Candidate-scoped POSID extraction

Confirmed POSID control flow follows Algorithm 1 at the candidate-context level.

For each selected candidate context `s`:

```text
candidate context s
    -> extract GENDER / RACE / HEIGHT from s
    -> derive the CLOTHES partial span from s
    -> run clothing extraction on that partial span
```

Successful candidate retrieval is **not** permission to scan the entire document for fixed properties.

If a retrieval method such as historical Regex returns only a span in `CandidateSentence.text`, POSID should use `CandidateSentence.context_text` as the full candidate context `s` when available.

### Shared deterministic person resolution

`src/hart/extractors/person_resolution.py` is the shared person-assignment policy for Regex, Dependency, and POSID.

Confirmed rules:

- clearly distinct person mentions produce separate `PersonPrediction` objects
- explicit names or person roles create/select person anchors
- same-sentence attributes attach to the nearest compatible local anchor
- pronouns continue the most recent compatible person when locally unambiguous
- for an unanchored pronoun/continuation candidate, locally preceding person-anchor context may be used for resolution without extracting fixed properties from that noncandidate sentence
- when multiple people are plausible, prefer the nearest preceding compatible anchor
- coordinated/shared descriptions may attach to multiple people
- if no plausible anchor exists, create an anonymous person prediction
- deterministic runtime IDs may use `<document_id>:0`, `<document_id>:1`, etc., in textual discovery order
- do not introduce a heavy coreference/entity-resolution system unless explicitly requested

### Temporary POSID clothing implementation status

During the current regression-stabilization pass, preserve the existing simplified POSID clothing implementation except for narrow fixes required by already-approved regression behavior.

This simplified clothing implementation is **temporary** and is **not** considered the final historical-reproduction implementation of Algorithm 1.

Do not perform a full POSID clothing rewrite during the current regression-fix pass. Revisit the clothing procedure against Algorithm 1 and the recovered notebooks after extraction stabilization and before attempting legacy table reproduction.


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
- WordNet strategy: verified legacy asymmetric behavior (first noun synset of query vs all noun synsets of each token)
- zero-shot label/query: `clothes`
- zero-shot threshold: `0.85`

Do not claim expected numeric Word2Vec, WordNet, or zero-shot scores until they are actually run with the corresponding resources.

### Unit/regression tests versus experimental accuracy

Unit/regression tests validate explicitly chosen algorithm semantics on controlled cases and are expected to pass completely.

They are **not** corpus-level retrieval or accuracy targets.

Do not:
- add assertions requiring perfect corpus recall/accuracy
- modify an algorithm merely to increase benchmark recall
- treat a green unit-test suite as evidence that a method reproduces published precision/recall/F1

Historical-paper reproduction and the new benchmark will later measure empirical precision, recall, and F1 on the relevant corpus/evaluation setup.

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
