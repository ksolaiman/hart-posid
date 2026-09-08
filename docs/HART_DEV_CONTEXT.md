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

A stacked selector may be added later, but should not be introduced until explicitly requested.

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

### POSID gender/race regex reconstruction

Recovered legacy artifacts contain only partial fixed-property regex code. The surviving gender/race notebook is an intermediate, male-specific prototype and does not fully specify the broader gender/race behavior described/evaluated in the paper.

Confirmed decision:

- preserve directly verified legacy regex behavior where available
- conservatively reconstruct missing **symmetric gender/race rules**
- clearly distinguish reconstructed rules from directly recovered historical code
- keep reconstruction narrow and faithful to the paper/task
- do not expand the reconstruction into unrelated attributes or a newly designed rule system

This decision applies to POSID fixed-property extraction, not to the separate 2026 direct Regex baseline.


### POSID candidate scope

Confirmed interpretation of Algorithm 1 for the current implementation:

For each selected candidate context `s`:

1. extract fixed properties `GENDER`, `RACE`, and `HEIGHT` from `s`
2. derive the `CLOTHES` partial span from `s`
3. apply clothing extraction to that partial span

Do not treat successful candidate retrieval as permission to scan the entire document for fixed properties.

When a retrieval method returns a partial span (historical Regex), use `CandidateSentence.context_text` as the full candidate context when available while preserving the original retrieval span in `CandidateSentence.text`.

### Shared person-resolution policy

`extractors/person_resolution.py` is the shared deterministic assignment layer for Regex, Dependency, and POSID.

Confirmed behavior:

- distinct named/role person mentions remain separate predictions
- explicit names and roles create/select anchors
- same-sentence attributes attach to the nearest compatible local anchor
- pronouns continue the most recent compatible person when locally unambiguous
- locally preceding person-anchor context may be used to resolve an unanchored pronoun/continuation candidate without extracting that noncandidate sentence's fixed attributes
- when multiple people are plausible, use the nearest preceding compatible anchor
- coordinated/shared descriptions may attach to multiple people
- descriptions with no plausible anchor produce an anonymous person
- runtime prediction IDs may be generated deterministically as `<document_id>:0`, `<document_id>:1`, etc.
- no heavy coreference/entity-resolution system is part of the current baseline architecture

### Temporary POSID clothing implementation status

The current `POSIDExtractor` uses a simplified clothing-extraction implementation as an interim checkpoint.

This is **not** considered the final historical-reproduction implementation of Algorithm 1.

During the current regression-stabilization pass:
- preserve the existing simplified clothing approach
- allow only narrow fixes required by already-approved regression behavior
- do not perform a full historical clothing rewrite

After extraction stabilization, revisit POSID clothing against Algorithm 1 and the recovered notebooks **before attempting legacy table reproduction**.

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

Do not implement this full benchmark yet during the candidate-module stage.

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

The Full-558 legacy reproduction harness is operational.

Verified corpus:
- 558 indexed documents
- 553 gold person annotations
- 6 zero-person documents
- 551 single-person documents
- 1 multi-person document

The temporary whitelist-based POSID clothing implementation has been
replaced with a closer Algorithm-1 reconstruction using POS-tagged,
open-vocabulary noun/property construction and candidate-scoped fallback
when no explicit wear/wearing/wore span exists.

The legacy evaluator has also been corrected:
- unmatched gold people now contribute false negatives
- unmatched predicted people now contribute false positives
- the Full-558 clothing gold denominator is fixed at 1503 items for all methods

Current corrected untuned results:
- RE + POSID: AO F1 0.247, AV F1 0.217
- WordNet + POSID: AO F1 0.093, AV F1 0.080
- RE + WordNet + POSID: AO F1 0.249, AV F1 0.217

Important current observation:
RE + POSID produced 1265 predicted people, of which 721 were unmatched,
despite the corpus containing only 553 gold people and being almost
entirely single-person documents.

This strongly suggests person-fragmentation may be a major source of
the low person-level score.

Next step:
perform diagnostic-only comparison of person-level versus document-level
clothing evaluation, plus distribution of predicted people per document,
before modifying person resolution or POSID again.

Do not tune thresholds, queries, candidate methods, evaluator semantics,
or POSID extraction based on these scores until this diagnosis is complete.