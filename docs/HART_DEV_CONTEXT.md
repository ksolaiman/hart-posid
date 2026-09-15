# HART Development Context

This file captures verified technical and research decisions transferred from the HART/POSID design discussion.

It is intended to give local coding agents enough context to implement the repository consistently with the research decisions.

## 1. Project identity

HART is the cleaned/reproducible continuation of the earlier POSID work.

The earlier paper is **A Modular Unsupervised Framework for Attribute Recognition from Unstructured Text**.

POSID is a modular, lightweight, unsupervised framework for extracting structured attributes from unstructured text without task-specific fine-tuning.

The current project goal is not to discard the old system, but to reproduce it faithfully enough to compare it against modern baselines and LLMs.

## 2. High-level POSID task

Given an input document/text:

1. identify candidate sentence(s) likely to contain human/object attributes
2. expose or detect target properties
3. extract corresponding property values

The original use case focused on human attributes such as gender, race, build, height, clothing, and clothing descriptions/colors.

## 3. Legacy candidate-selection methods

The original paper/system used:

1. Regex / pattern matching
2. Word2Vec semantic similarity
3. WordNet similarity
4. Zero-shot classification
5. Stacked fallback combinations

The current implementation should preserve those as modular candidate selectors.


### Confirmed stacked-selector behavior

`StackedSelector` is implemented as fallback composition:

```text
primary selector
    -> return primary results when non-empty
    -> otherwise invoke fallback selector
```

The primary and fallback selectors use separately configurable query lists (`primary_queries` and `fallback_queries`) so historical Regex and semantic stages can use different query phrases without hardcoding them in reusable classes.

## 4. Regex candidate method

The legacy code included behavior conceptually equivalent to:

```python
regex = r"wearing[^.]*"
w_text_all = re.compile(regex).findall(text)
```

This means the historical regex path returned the matched span beginning with `wearing` and continuing until the next period.

Current reusable behavior should preserve this span-return semantics while keeping the query parameterized.

Generic behavior should be equivalent to:

```python
pattern = rf"{re.escape(query)}[^.]*"
```

For example, with query `wearing`:

```text
wearing a red shirt and blue pants
```

should be returned rather than the entire containing sentence.

Important distinction: legacy RE + POSID is not the same as a modern direct-regex extraction baseline. Legacy RE + POSID means candidate retrieval by regex, followed by POSID property extraction.

## 5. Word2Vec candidate method

Verified from legacy code:

```python
nltk.download("word2vec_sample")
word2vec_sample = str(find("models/word2vec_sample/pruned.word2vec.txt"))

model = gensim.models.KeyedVectors.load_word2vec_format(
    word2vec_sample,
    binary=False
)
```

So the historical embedding resource was the NLTK pruned Word2Vec sample.

Verified historical candidate logic:

```python
sentences = nltk.sent_tokenize(text)

for sentence in sentences:
    words = nltk.word_tokenize(sentence)

    for word in words:
        try:
            if model.similarity(word, "clothes") >= 0.5:
                return [sentence]
        except:
            pass
```

Verified historical experiment values:

- query: `clothes`
- threshold: `0.5`

Important details:

- NLTK sentence tokenization was used
- NLTK word tokenization was used
- individual tokens were compared to the query with `model.similarity(token, query)`
- threshold comparison was `>=`
- the first qualifying sentence was returned immediately
- OOV failures were ignored in the legacy code

Current reproducible OOV handling:

- skip OOV token/query comparisons
- do not assign a fabricated score
- retain the best valid token-query score for the sentence
- if no valid comparison exists for a sentence, treat it as unscorable and do not return it

For the reusable implementation:

- vectors should be injected
- query should be passed to `select(...)`
- threshold should be supplied as a parameter
- NLTK resource loading should stay outside the selector class

## 6. Zero-shot NLI candidate method

The implementation is exposed in HART as `ZeroShotSelector` / `zero_shot.py`.

The paper describes Sentence-BERT sentence-level encoding as part of the method, but the implementation name can remain zero-shot for now. The method is not plain embedding cosine similarity.

The paper describes the process as follows:

1. split the document into sentences
2. take one sentence at a time as the input sequence / NLI premise
3. use the caller-supplied key phrase as the candidate label
4. construct a hypothesis from the key phrase

Example:

```text
query: clothes
hypothesis: This text is about clothes
```

5. encode both the current sentence and the hypothesis using the sentence-level encoder Sentence-BERT
6. use a pre-trained NLI model in a zero-shot sequence-classification setup
7. obtain entailment / contradiction probabilities and convert these to the class probability for the candidate label
8. if the class probability exceeds the threshold, select that sentence
9. otherwise continue sentence-by-sentence

The paper frames this as a Multi-NLI-style zero-shot NLI method.

Verified legacy implementation path:

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification")
candidate_labels = ["clothes"]
```

Verified candidate behavior:

```python
sentences = nltk.sent_tokenize(text)

for sentence in sentences:
    res = classifier(sentence, candidate_labels)

    if res["scores"][0] > 0.85:
        return [sentence]
```

Verified historical experiment values:

- candidate label/query: `clothes`
- threshold: `0.85`

Also verified:

- regex was attempted first in the legacy fallback function
- the SBERT/zero-shot path executed only when the regex path did not find a result
- comparison used strict `>`
- the first qualifying sentence was returned immediately

Authoritative implementation interpretation:

- implementation name in HART: `ZeroShotSelector`
- task formulation: zero-shot NLI
- NLI framing: Multi-NLI-style premise/hypothesis classification
- implementation path: Hugging Face `pipeline("zero-shot-classification")`
- the paper describes Sentence-BERT sentence-level encoding as part of the process
- preserve the premise/hypothesis construction
- do **not** replace the method with plain embedding cosine similarity

For the reusable implementation, inject the classifier and keep query/threshold parameterized.

## 7. WordNet candidate method

Recovered 2021/2023 artifact now confirms the historical WordNet candidate-selection behavior.

Historical method:

1. obtain the **first noun synset of the query** (historically `clothes`)
2. tokenize each sentence with NLTK
3. obtain **all noun synsets** for each token
4. compare the query synset with each token synset using Wu-Palmer similarity
5. if any valid score is `>= threshold`, return that sentence immediately
6. otherwise continue to the next sentence

Conceptually:

```python
query_synset = wn.synsets(query, "n")[0]

for sentence in nltk.sent_tokenize(text):
    for token in nltk.word_tokenize(sentence):
        for token_synset in wn.synsets(token, "n"):
            score = query_synset.wup_similarity(token_synset)
            if score is not None and score >= threshold:
                return [sentence]
```

Verified historical experiment values remain:

- query: `clothes`
- threshold: `0.9`

This is an asymmetric strategy: **first query noun synset vs all token noun synsets**.

The reusable selector may retain `first`, `max_all`, and `average` strategies for experimentation, but legacy reproduction must use a clearly identified mode matching the verified asymmetric historical behavior. Do not describe `first`-vs-`first` as the historical strategy.

Known separate POSID extraction behavior:

```python
color = wn.synsets("color", "n")[0]
```

This belongs to later clothing/color interpretation.

## 8. Tokenization decision

The candidate package should use NLTK throughout:

```python
from nltk.tokenize import sent_tokenize, word_tokenize
```

Shared helper behavior should live in `base.py`.

Do not use the earlier provisional regex tokenizer.

## 9. Candidate package structure

Current intended structure:

```text
src/hart/candidates/
├── __init__.py
├── base.py
├── regex.py
├── word2vec.py
├── wordnet.py
└── zero_shot.py
```

The current smoke test lives outside the package:

```text
tests/test_candidates_sample.py
```

`StackedSelector` is implemented as the confirmed fallback-composition selector described above.

## 10. Shared candidate abstraction

The candidate abstraction now distinguishes retrieval output from downstream extraction context.

Conceptual form:

```python
@dataclass(frozen=True)
class CandidateSentence:
    sentence_id: int
    text: str
    score: float
    query: str | None = None
    context_text: str | None = None
```

Confirmed semantics:

```text
text
    = the candidate returned by the retrieval algorithm

context_text
    = optional surrounding candidate sentence/context used by downstream extraction
```

For the recovered historical Regex path:

```text
text = matched "wearing..." span
context_text = full descriptive candidate context
```

This preserves the historical Regex output while giving POSID the candidate context required by Algorithm 1.

For Word2Vec, WordNet, and ZeroShot, `text` is already sentence-level; `context_text` may equal `text` or remain unnecessary.

The abstract selector interface remains minimal:

```python
class CandidateSelector(ABC):
    @abstractmethod
    def select(self, text, queries):
        ...
```


## 11. Current smoke-test sample

Use:

```text
The suspect was described as a short Asian male, about 145 lbs. wearing a green jacket and blue pants.
```

Historical expected qualitative behavior:

- regex: should return a matched span beginning with `wearing`
- Word2Vec: measure with NLTK pruned Word2Vec vectors
- WordNet: use the verified legacy asymmetric strategy: first noun synset of the query vs all noun synsets of each token
- zero-shot: measure with the actual classifier environment

Do not fabricate numeric semantic scores.

## 12. Current data and annotation architecture

The data layer is implemented and should now be treated as a completed checkpoint unless a concrete bug or new requirement appears.

### Local layout

The source tree and InciText dataset are siblings under the same project root:

```text
project-root/
├── src/
│   └── hart/
└── InciText/
    └── Annotation/
        ├── annotations.json
        ├── consolidated_annotation_file.json
        ├── consolidated_report_metadata.json
        └── report_metadata.json
```

Do not hardcode machine-specific absolute paths.

### Current data/schema package

Use:

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

Do not introduce a separate `data/loaders/` package at this stage.

`manifest.py` and `validators.py` are deferred unless a concrete need emerges.

### Authoritative source-document inventory

The only authoritative source for document membership and source paths is:

```text
InciText/Annotation/consolidated_report_metadata.json
```

Do not use:

```text
InciText/Annotation/report_metadata.json
```

for document discovery.

Do not recursively scan `InciText/`, infer membership from folder contents, or substitute alternate files.

Source paths are resolved against the `InciText/` root and must remain inside that root.

The implemented loader first creates a lightweight metadata-only `DocumentRecord` index, then reads text lazily/on demand.

Conceptually:

```text
consolidated_report_metadata.json
        ↓
metadata-only DocumentRecord index
        ↓
load_document(record) on demand
```

Full-corpus text reads are reserved for explicit benchmark/test passes.

### Annotation consolidation

The root legacy file:

```text
InciText/Annotation/annotations.json
```

is incomplete for the current corpus.

The preparation utility:

```text
src/hart/data/consolidate_annotations.py
```

uses the authoritative consolidated metadata to derive the relevant annotation inputs and builds:

```text
InciText/Annotation/consolidated_annotation_file.json
```

The utility is metadata-driven and does not blindly scan folders.

It:

1. reads authoritative document IDs and annotation/source paths from `consolidated_report_metadata.json`
2. uses only annotation inputs associated with those authoritative records
3. skips folders with no standalone annotation file, including `Generated_flash_2`
4. merges by `document_id`
5. preserves the raw list-of-persons structure
6. tolerates identical duplicates without duplicating them
7. fails on conflicting duplicates
8. never modifies original annotation files

Verified consolidated output:

```text
Documents:       558
People:          553
Empty lists:       6
Single-person:   551
Two-person:        1
```

### Runtime annotation source

The runtime legacy annotation source is:

```text
InciText/Annotation/consolidated_annotation_file.json
```

not the incomplete root `annotations.json`.

The runtime flow is:

```text
consolidated_annotation_file.json
        ↓
legacy_adapter.py
        ↓
canonical in-memory PersonAnnotation objects
```

### Canonical representation and schema isolation

Canonical annotation contract:

```python
dict[str, list[PersonAnnotation]]
```

Every document maps to a list of person annotations.

The one known two-person document remains two separate `PersonAnnotation` records for the same `document_id`.

If a stable source person ID is unavailable, deterministic runtime-only IDs may be created as:

```text
<document_id>:0
<document_id>:1
```

These IDs are internal only and are never written back to the source files.

Legacy string `"null"` values are normalized at the adapter/normalization boundary.

The current experiments do not require rewriting stored annotations into Schema 2.0.

The design goal is schema isolation: a future annotation format should need only a new loader/adapter that emits the same canonical model. Downstream extractors, experiment runners, and evaluation should not depend on the on-disk annotation schema.

### Implemented data-layer checkpoint

The following are implemented:

- metadata-only `DocumentRecord`
- lazy `load_document_index()`
- on-demand `load_document()`
- generator-based `load_documents()`
- path-containment checks
- media-type handling
- legacy consolidation utility
- canonical legacy adapter
- recursive normalization checks
- data-loading tests

Verified data test state:

```text
Document index:          558 unique IDs
Annotation documents:    558
People:                  553
Empty lists:               6
Single-person documents: 551
Two-person documents:      1
Metadata/annotation IDs: exact match
```

The candidate package remains a separate completed checkpoint and should not be modified while working on later layers.



## 12A. Confirmed extraction-layer decisions

### POSID fixed-property reconstruction

The surviving fixed-property code is incomplete for some categories, so the current implementation preserves verified behavior and uses narrow explicit reconstructions where necessary.

Current rules:
- retain the approved symmetric explicit gender terms
- explicit race extraction includes `asian`, `black`, `white`, `hispanic`, `latino`, `latina`, and `indian`
- never infer race/gender from names
- candidate retrieval still gates POSID
- when a selected candidate begins with a clothing relation, fixed attributes may also be recovered from the **immediately preceding role-based descriptive sentence**
- unrelated document sentences must not be scanned/attached merely because retrieval succeeded
- explicit numeric Height takes precedence over a co-occurring relative `tall` in the same local description
- standalone `short` and `tall` remain supported; `average` is not currently implemented

### Height extraction and conservative normalization

A verified Height failure mode was duplicate emission from strings such as `5'10" tall`: the numeric Height and `tall` were both emitted, then scalar last-write-wins behavior retained `tall`. The extraction fix makes the numeric pattern consume adjacent terminal `tall`, so only the explicit numeric Height is emitted.

Height matching now uses only conservative surface normalization:
- curly quotation marks are normalized to straight quote equivalents
- whitespace is normalized
- a terminal `tall` is removed only when it follows an otherwise identical explicit numeric/feet Height expression

Explicitly forbidden equivalences include:
- `5'11" == 6 feet`
- approximate/tolerance matching between nearby numeric heights
- `tall` or `short` == numeric Height

### POSID local candidate scope

POSID remains candidate-gated. Clothing remains extracted from the selected clothing candidate/context, while fixed properties may use the selected context plus a narrowly adjacent role-based descriptive sentence when the candidate begins with a clothing relation.

Conceptually:

```text
selected candidate/context
    -> fixed properties from local context
       + immediately preceding role-based description when needed
    -> clothing extraction from the clothing candidate context
```

This specifically supports local sentence-boundary constructions without turning successful retrieval into permission for document-wide fixed-property scanning.

### Shared person-resolution policy

`extractors/person_resolution.py` remains the shared deterministic assignment layer for Regex, Dependency, and POSID and was not changed by the final Height correction.

Confirmed behavior:
- distinct named/role person mentions remain separate predictions
- explicit names and roles create/select anchors
- same-sentence attributes attach to the nearest compatible local anchor
- pronouns continue the most recent compatible person when locally unambiguous
- narrow immediately preceding role-based context may supply fixed properties/anchors for a clothing-relation candidate
- unrelated named-person attributes elsewhere in a document must not leak to the selected target
- when multiple people are plausible, use the nearest preceding compatible anchor
- coordinated/shared descriptions may attach to multiple people
- descriptions with no plausible anchor produce an anonymous person
- runtime prediction IDs may be generated deterministically as `<document_id>:0`, `<document_id>:1`, etc.
- no heavy coreference/entity-resolution system is part of the current baseline architecture

### Current POSID clothing implementation status

The earlier temporary whitelist-based clothing implementation has been replaced with a stable open-vocabulary POS/noun reconstruction.

Current confirmed behavior:
- preserve noun surface forms such as `jeans`, `pants`, `boots`, and `shorts`; do not singularize inside POSID
- appropriate `JJ` tokens are retained as clothing descriptors regardless of whether they are colors
- noun-tagged color interpretation uses `wn.synsets("color", "n")[0]` and strict Wu-Palmer similarity `> 0.7`
- descriptor state may continue across structural connector tokens when they remain part of the same clothing phrase
- multiword clothing items remain supported
- do not reintroduce a closed clothing vocabulary
- VBD/RP/RB phrase behavior such as `buttoned up shirt` remains deliberately untouched and is not a blocker for the current reproduction checkpoint

The current clothing implementation should now be treated as frozen for the legacy arXiv revision unless a separate reproducible bug is identified.

### Universal prediction representation

All extractors should emit a canonical person-level prediction object rather than method-specific raw dictionaries or per-attribute records.

Confirmed conceptual model in `schema/models.py`:

```python
@dataclass
class PersonPrediction:
    document_id: str
    person_id: str | None
    role: str | None
    attributes: dict
    metadata: dict | None = None
```

Canonical prediction contract:

```python
dict[str, list[PersonPrediction]]
```

This should be used by Regex, Dependency, POSID, and later LLM extractors.

The gold and prediction sides should mirror one another conceptually:

```text
legacy annotation source -> legacy_adapter.py -> PersonAnnotation
raw document             -> extractor          -> PersonPrediction
```

Do not use a flattened `AttributePrediction` object as the primary extractor interface.

Per-attribute flattening should occur only in the evaluation layer, where it can support per-attribute metrics, source/domain splits, bootstrap confidence intervals, and error analysis.

Core `PersonPrediction` does not require `sentence_id` or `evidence_text`. Optional method-specific provenance can be placed in `metadata`.

## 12B. Regression tests versus empirical evaluation

Deterministic unit/regression tests and corpus-level evaluation serve different purposes.

Unit/regression tests:
- verify explicitly chosen algorithm semantics on controlled examples
- are expected to pass completely once implementation is stable
- should not encode a requirement for perfect corpus retrieval

Empirical evaluation:
- measures actual precision, recall, and F1
- is used for historical-paper reproduction and the new 2026 benchmark
- may reveal genuine false positives/false negatives even when all unit tests pass

Do not modify a method merely to force perfect corpus accuracy, and do not interpret a green unit-test suite as reproduction of the paper's reported tables.


## 12C. Confirmed primary evaluation policy

The canonical extractor output remains person-structured:

```python
dict[str, list[PersonPrediction]]
```

Person structure is preserved during extraction. Primary extraction evaluation and person attribution are intentionally separate questions.

### Primary document-level extraction view

For each document, pool extractable facts across all gold people and across all predicted `PersonPrediction` records while preserving multiplicity. Do not use `dict.update()` or set-based flattening that can overwrite or collapse duplicate instances.

This policy is especially important because the Full-558 corpus is almost entirely single-person: 551 single-person documents, 1 two-person document, and 6 zero-person documents.

For single-gold-person documents, person fragmentation by itself is not an extraction error: if the correct fact appears in any predicted person record, it can be matched. Extra/conflicting predicted facts still contribute false positives.

For multi-person gold documents, the same document-level extraction metric remains primary. Person-level attribution is reported separately as an extension/diagnostic rather than silently folded into the extraction score.

### Clothing Attr-Only

Attr-Only measures clothing-item extraction.

Confirmed semantics:
- normalize item names conservatively
- preserve duplicate item instances
- perform one-to-one item-name matching
- descriptors do not affect Attr-Only matching
- matched item -> TP
- unmatched predicted item -> FP
- unmatched gold item -> FN

Historical Attr-Only is treated as item-name membership/matching rather than whole-dictionary equality.

### Clothing Attr-Value

Attr-Value is decomposable rather than all-or-nothing dictionary equality.

For a matched clothing item:
- the item itself is one evaluation unit
- each descriptor/value is an additional evaluation unit
- correctly matched descriptor/value -> TP
- unmatched predicted descriptor/value -> FP
- unmatched gold descriptor/value -> FN

This provides partial credit when an item is correct but only some descriptors are recovered.

For duplicate same-name clothing instances, use deterministic one-to-one matching that maximizes descriptor/value overlap. Preserve multiplicity; do not collapse duplicates into a set.

The old `dict.update()` flattening shortcut is not adopted as the clean evaluator because it can overwrite duplicate clothing keys.

### Gender, Race, and Height

Gender, Race, and Height are evaluated as finite-valued attributes, independently, with micro P/R/F1.

For each attribute:
- pool non-null gold values across gold people in the document
- pool non-null predicted values across predicted people
- preserve multiplicity
- perform deterministic one-to-one exact/current-normalized matching
- matched value -> TP
- unmatched predicted value -> FP
- unmatched gold value -> FN

The current evaluator does not introduce fuzzy semantic matching.

For Height, the final frozen normalization is intentionally conservative: curly/straight quote variants are canonicalized, whitespace is normalized, and terminal `tall` is ignored only when it follows an otherwise identical explicit numeric/feet measurement. Thus `6 feet` and `6 feet tall` may match, while values such as `5'11"` and `6 feet` remain distinct. No numeric tolerance, rounding, or categorical-to-numeric equivalence is allowed.

### Person-aware diagnostic / attribution view

The corrected person-aware evaluator remains available and should not be weakened or replaced. Its purpose is to measure whether extracted facts were attached to the correct predicted person structure.

Primary extraction score and person-aware attribution score must remain separately labeled. The person-aware view may be reported over the full corpus for diagnostics, but its conceptual importance is greatest for genuinely multi-person documents.

### Historical-evaluator provenance

Recovered evidence supports the clothing Attr-Only item-name interpretation and the decomposable Attr-Value conceptual behavior. The historical `dict.update()` implementation shortcut is not treated as theoretically necessary.

The exact surviving historical TP/FP/FN implementation for Gender/Race/Height has not been recovered. Their current finite-value metric is therefore a defensible reconstruction consistent with the task and paper, not a claim that the historical evaluator code itself has been recovered.


## 13. Benchmark plan

The intended modern benchmark compares lightweight methods against LLMs while preserving the legacy POSID line.

Core methods currently planned:

- direct Regex baseline
- dependency-based baseline
- POSID-Light / RE + POSID
- POSID-Full / best original stacked POSID configuration
- two LLM extractors

All methods should emit the same canonical runtime prediction representation used by evaluation. This runtime representation is independent of the current on-disk Schema 1.0 gold format.

Evaluation should eventually include:

- overall F1
- per-attribute performance
- source/domain breakdown
- real vs synthetic breakdown
- runtime
- API/token cost where applicable
- document-level bootstrap confidence intervals
- reproducible error analysis

The full modern benchmark remains a later stage. Current work is limited to stabilizing/reproducing the legacy POSID evaluation before expanding to the complete 2026 benchmark.

## 14. Main benchmark data split

Current planned main real benchmark:

- approximately 18 real legacy documents
  - 13 press releases
  - 5 incident reports
- 100 Facebook-style missing-person cases

Total main real benchmark target: approximately 118 documents.

Older generated/synthetic documents should be treated separately as auxiliary synthetic data.

## 15. Annotation workflow

Minimum-effort annotation strategy:

1. Use the existing Schema 1.0 annotations as the current stored gold and correct them only where the benchmark review identifies errors.
2. Undergraduate reviewer verifies/corrects the real benchmark cases as needed.
3. Project lead audits/adjudicates a stratified subset plus disputes/odd cases.
4. Run automated consistency checks.
5. Report correction/agreement statistics on the audited subset.

Synthetic data remains auxiliary/silver unless separately validated.

## 16. Research framing

The intended publication framing is a modern revisiting of lightweight human-attribute extraction in the LLM era.

The project is expected to emphasize:

1. cleaned human-attribute extraction resource
2. systematic comparison of rules, linguistic methods, POSID, and LLMs
3. source/domain and synthetic-real shift
4. accuracy-efficiency/deployment tradeoffs

The goal is to preserve the old POSID contribution while testing where it remains useful in 2026.

## 17. Implementation principles

For local Codex or any coding agent:

- preserve legacy behavior where verified
- keep uncertainty explicit where unresolved
- do not silently substitute a newer algorithm
- keep experiment parameters outside reusable algorithm classes
- keep source code modular
- prefer small, reviewable changes
- run targeted tests after each module
- do not implement future modules early
- do not infer missing historical details without approval

## 18. Current reproduction checkpoint

The Full-558 legacy reproduction harness and reconciled primary evaluator are operational, and the legacy reconstruction is now a **frozen arXiv-revision checkpoint**.

Verified corpus:
- 558 indexed documents
- 553 gold person annotations
- 6 zero-person documents
- 551 single-person documents
- 1 multi-person document

Verified primary gold denominators:
- Gender values: 553
- Race values: 552
- Height values: 13
- Clothing Attr-Only item units: 1503
- Clothing Attr-Value decomposable units: 3011

These denominators remained unchanged through the final extraction/Height correction passes.

### Final available-method primary F1 results

| Method | Gender F1 | Race F1 | Height F1 | Clothes AO F1 | Clothes AV F1 |
|---|---:|---:|---:|---:|---:|
| RE + POSID | 0.987 | 0.988 | 0.762 | 0.991 | 0.832 |
| WordNet + POSID | 0.462 | 0.462 | 0.000 | 0.480 | 0.368 |
| RE + WordNet + POSID | 0.987 | 0.987 | 0.762 | 0.992 | 0.832 |

### Final RE + POSID exact counts

| Attribute/view | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Gender | 539 | 0 | 14 | 1.000 | 0.975 | 0.987 |
| Race | 539 | 0 | 13 | 1.000 | 0.976 | 0.988 |
| Height | 8 | 0 | 5 | 1.000 | 0.615 | 0.762 |
| Clothes Attr-Only | 1490 | 14 | 13 | 0.991 | 0.991 | 0.991 |
| Clothes Attr-Value | 2155 | 16 | 856 | 0.993 | 0.716 | 0.832 |

### What changed during the final reconstruction pass

The final improvements were narrow and independently motivated:
- fixed attributes can use the immediately preceding role-based descriptive sentence when a selected candidate begins with a clothing relation
- `indian` was added as an explicit race value
- appropriate `JJ` tokens are retained as clothing descriptors rather than requiring color classification
- descriptor state can survive structural connector tokens inside a clothing phrase
- clothing noun surface forms such as `jeans`, `pants`, `boots`, and `shorts` are preserved rather than singularized
- explicit numeric Height takes precedence over co-occurring relative `tall`
- Height evaluation canonicalizes curly quote variants, whitespace, and terminal `tall` on otherwise identical explicit measurements

These changes moved RE + POSID from the earlier checkpoint of AO 0.878 / AV 0.732 / Height 0.235 to the final AO 0.991 / AV 0.832 / Height 0.762.

### Final Height diagnostic

RE + POSID now matches 8 of the 13 gold Height values with no false positives:

| Gold Height | Final outcome |
|---|---|
| `short` | matched |
| `5'11"` | matched after quote normalization |
| `tall` | missed |
| `5'10"` | matched; numeric Height no longer overwritten by `tall` |
| `5'7"` | matched; numeric Height no longer overwritten by `tall` |
| `average` | missed; not currently supported |
| `5'8"` | matched; numeric Height no longer overwritten by `tall` |
| `6'1"` | missed in one document |
| `6 feet` | matched against `6 feet tall` by conservative terminal-`tall` normalization |
| `6'1"` | matched in one document |
| `6 feet tall` | matched |
| `5'6"` | missed |
| `tall` | missed |

Remaining Height misses are accepted for the frozen legacy checkpoint; do not add broader equivalence or corpus-specific retrieval rules to remove them.

### Source-backed Synthetic_2 annotation correction

The malformed legacy Height value for:

```text
document_id: 3fcd11db-4875-4dcb-8ec6-93dd5a43804b
title: Synthetic_2
source: Synthetic/DescriptionsforPU2.txt
```

was corrected from `5'15"` to the source-supported `5'11"`. This was handled as a data correction rather than a generic normalization rule; no rule such as “inches > 11” was introduced. The Height denominator remained 13.

### Person-aware diagnostic interpretation

An earlier pre-final-extraction diagnostic showed very low person-aware clothing scores and demonstrated that prediction fragmentation could dominate Attr-Only errors. That conceptual conclusion remains useful, but its old numeric F1 values are **not final-current metrics** because POSID clothing predictions changed materially afterward.

Primary extraction remains document-pooled and multiplicity-preserving. Person-aware attribution remains a separate extension/diagnostic and should be rerun only if current attribution numbers are needed for the paper.

### Remaining known clothing limitation

Clothing Attr-Only is effectively saturated for RE + POSID on Full-558 (F1 0.991). Attr-Value remains lower (F1 0.832) because descriptor/value recall is still incomplete. VBD/RP/RB phrase handling such as `buttoned up shirt` remains intentionally untouched. Do not continue rewriting the legacy parser solely to close the residual gap.

### Published-result comparison caution

Published Table I Gender/Race/Height values (`.94/.73/.82`, `.94/.73/.82`, `.72/.57/.63`) belong to the historical best RE + SBERT/zero-shot + POSID configuration. They are context, not a direct RE + POSID target.

The revised Full-558 numbers use the reconciled current evaluator and broader Full-558 corpus. Do not force them to equal the old paper table.

### Resource status

Current local resource/runtime state:
- WordNet: available
- POS tagger: available
- historical NLTK Word2Vec sample: available
- `gensim`: available
- Word2Vec + POSID has completed a full 558-document local run successfully
- zero-shot/NLI dependencies and local BART checkpoint are present, but local macOS PyTorch/BART inference currently fails with a native bus error inside `torch.nn.Linear`
- run `sbert_posid` and `re_sbert_posid` on a stable server/Colab runtime rather than treating them as algorithmically blocked

Do not substitute a different model merely to avoid the local runtime failure.

### Validation state

Latest complete suite:

```text
75 passed, 0 failed
```

Focused final Height validation included:
- POSID Height tests: 5 passed, 18 deselected
- Height evaluation tests: 4 passed, 17 deselected
- `tests/test_posid.py`: 23 passed
- `tests/test_extraction_metrics.py`: 21 passed
- `tests/test_legacy_metrics.py`: 7 passed

### Next steps

1. Freeze the legacy implementation/evaluator at this checkpoint.
2. Update the arXiv paper with the verified Full-558 dataset description, corrected method details, current results, limitations, and reproducibility/code-release information.
3. Do not spend additional time tuning the legacy system solely to reproduce the old table.
4. Treat the modern Regex + Dependency + POSID + LLM comparison on contemporary/real data as a separate conference-paper extension.

Any future change to the legacy extraction/evaluation path should require a separately identified reproducible bug and explicit approval, not score chasing.


## 18A. Non-generated 28-document legacy slice

A dedicated non-generated legacy slice is now defined for evaluating whether Full-558 results are inflated by the 530 generated documents.

Composition:

| Source group | Documents |
|---|---:|
| Synthetic | 10 |
| Press Release | 13 |
| Incident Reports | 5 |
| **Total** | **28** |

For this analysis, the 10 documents stored under `Synthetic` are treated as police-provided/real descriptions and are included in the 28-document legacy slice. They are distinct from the explicitly generated augmentation sets.

Excluded from this slice:

| Source group | Documents |
|---|---:|
| Generated | 30 |
| Generated_flash | 500 |
| **Excluded generated total** | **530** |

Thus the Full-558 corpus is approximately 95% generated augmentation (`530/558`), while the `non_generated_28` slice isolates the 28 legacy documents that are not from `Generated` or `Generated_flash`.

This slice is intended as a diagnostic, not a replacement for Full-558. The purpose is to compare the same frozen extraction/evaluation pipeline across source composition and determine how much the generated data contributes to the very high Full-558 RE-based scores.

The expected runner invocation is:

```bash
python scripts/reproduce_legacy_posid.py \
  --incitext-root ../InciText \
  --output-dir outputs/legacy_reproduction_non_generated_28 \
  --subset non_generated_28 \
  --methods re_posid wordnet_posid re_wordnet_posid word2vec_posid
```

Use a separate output directory so this diagnostic does not overwrite the Full-558 results.

The same runner should support the following source-composition diagnostics without changing extractors or metrics:

```text
full                              = 558 authoritative documents
non_generated_28                  = 10 Synthetic + 13 Press Release + 5 Incident Reports
non_generated_plus_generated_58   = non_generated_28 + 30 Generated
generated_only                    = 30 Generated
generated_flash_only              = 500 Generated_flash
```

This 28-document slice has now been executed with the frozen reproduction pipeline. Verified corpus/gold counts are:

```text
Indexed documents:       28
Gold person annotations: 23
Zero-person documents:    6
Single-person documents: 21
Multi-person documents:   1
Gold Gender values:      23
Gold Race values:        22
Gold Height values:      13
```

The exact 28-document metric table is not reproduced in this context file yet. Do not infer or copy Full-558 or 58-document metric values into this slice; use the verified run output when adding the table.

### Runner method selection

The legacy reproduction runner supports/should support method selection independently of subset selection. Valid method keys are:

```text
re_posid
wordnet_posid
re_wordnet_posid
word2vec_posid
sbert_posid
re_sbert_posid
```

The two controls are orthogonal:

```text
--methods  = which legacy algorithms run
--subset   = which authoritative documents are evaluated
```

Examples:

```bash
# Full-558 lightweight/local methods
python scripts/reproduce_legacy_posid.py \
  --incitext-root ../InciText \
  --output-dir outputs/legacy_reproduction \
  --subset full \
  --methods re_posid wordnet_posid re_wordnet_posid word2vec_posid

# Full-558 zero-shot methods on server/Colab
python scripts/reproduce_legacy_posid.py \
  --incitext-root ../InciText \
  --output-dir outputs/legacy_reproduction_sbert \
  --subset full \
  --methods sbert_posid re_sbert_posid

# 28-document non-generated legacy slice
python scripts/reproduce_legacy_posid.py \
  --incitext-root ../InciText \
  --output-dir outputs/legacy_reproduction_non_generated_28 \
  --subset non_generated_28 \
  --methods re_posid wordnet_posid re_wordnet_posid word2vec_posid

# 58-document legacy + Generated slice, excluding Generated_flash
python scripts/reproduce_legacy_posid.py \
  --incitext-root ../InciText \
  --output-dir outputs/legacy_reproduction_58 \
  --subset non_generated_plus_generated_58 \
  --methods re_posid wordnet_posid re_wordnet_posid word2vec_posid
```

Unselected heavy resources should not be initialized. This is especially important because the local macOS environment has successfully run Word2Vec but BART/PyTorch zero-shot inference has produced a native bus error; the zero-shot methods should therefore be run separately on a stable server/Colab runtime rather than forcing them into local non-SBERT runs.

## 18B. 58-document legacy + Generated reproduction slice

A second source-composition slice has now been executed using the frozen extraction and evaluation pipeline. It retains the 28-document legacy slice and the 30 documents in `Generated`, while excluding all 500 `Generated_flash` documents.

Composition:

| Source group | Documents |
|---|---:|
| Synthetic | 10 |
| Press Release | 13 |
| Incident Reports | 5 |
| Generated | 30 |
| **Total** | **58** |

Observed corpus/gold counts:

```text
Indexed documents:       58
Gold person annotations: 53
Zero-person documents:    6
Single-person documents: 51
Multi-person documents:   1
Gold Gender values:      53
Gold Race values:        52
Gold Height values:      13
```

The run used:

```bash
python scripts/reproduce_legacy_posid.py \
  --incitext-root ../InciText \
  --output-dir outputs/legacy_reproduction_58 \
  --subset non_generated_plus_generated_58 \
  --methods re_posid wordnet_posid re_wordnet_posid word2vec_posid
```

All four selected methods processed all 58 documents with zero per-document exceptions.

Run-level prediction counts:

| Method | Docs with predictions | Predicted people | Predicted clothing items |
|---|---:|---:|---:|
| Word2Vec + POSID | 22 | 36 | 62 |
| RE + POSID | 44 | 68 | 117 |
| WordNet + POSID | 11 | 25 | 32 |
| RE + WordNet + POSID | 45 | 69 | 119 |

### 58-document primary extraction results

| Method | Gender P/R/F1 | Race P/R/F1 | Height P/R/F1 | Clothes AO P/R/F1 | Clothes AV P/R/F1 |
|---|---|---|---|---|---|
| Word2Vec + POSID | 1.000 / 0.302 / 0.464 | 1.000 / 0.308 / 0.471 | 1.000 / 0.154 / 0.267 | 0.855 / 0.457 / 0.596 | 0.907 / 0.451 / 0.603 |
| RE + POSID | 1.000 / 0.736 / 0.848 | 1.000 / 0.750 / 0.857 | 1.000 / 0.615 / 0.762 | 0.880 / 0.888 / 0.884 | 0.928 / 0.869 / 0.898 |
| WordNet + POSID | 1.000 / 0.094 / 0.172 | 0.833 / 0.096 / 0.172 | 0.000 / 0.000 / 0.000 | 0.812 / 0.224 / 0.351 | 0.864 / 0.215 / 0.345 |
| RE + WordNet + POSID | 1.000 / 0.736 / 0.848 | 0.975 / 0.750 / 0.848 | 1.000 / 0.615 / 0.762 | 0.882 / 0.905 / 0.894 | 0.929 / 0.882 / 0.905 |

### Clothing comparison with published paper values

| Method | Published AO F1 | 58-doc AO F1 | Δ AO | Published AV F1 | 58-doc AV F1 | Δ AV |
|---|---:|---:|---:|---:|---:|---:|
| Word2Vec + POSID | 0.52 | 0.596 | +0.076 | 0.49 | 0.603 | +0.113 |
| RE + POSID | 0.84 | 0.884 | +0.044 | 0.87 | 0.898 | +0.028 |
| WordNet + POSID | 0.49 | 0.351 | -0.139 | 0.45 | 0.345 | -0.105 |
| RE + WordNet + POSID | 0.77 | 0.894 | +0.124 | 0.90 | 0.905 | +0.005 |

Interpretation:

- Removing the 500 `Generated_flash` documents materially changes the RE-based results from the near-ceiling Full-558 regime.
- RE + POSID on the 58-document slice is close to the published clothing results: AO F1 0.884 versus 0.84 and AV F1 0.898 versus 0.87.
- RE + WordNet + POSID reproduces the published Attr-Value result especially closely: 0.905 versus 0.90, while its Attr-Only score remains higher than the paper.
- Word2Vec + POSID is moderately above the published paper on both clothing metrics.
- WordNet + POSID remains substantially below its published clothing values and is still an unresolved reproduction difference.
- This source-slice result supports dataset-composition effects as an important contributor to the Full-558 discrepancy, especially for RE-based methods.
- Do **not** claim that these 58 documents are the exact historical paper evaluation subset unless separate historical evidence establishes that fact. This is a controlled legacy-oriented/source-composition reproduction slice.

