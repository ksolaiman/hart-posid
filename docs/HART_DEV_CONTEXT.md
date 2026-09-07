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

The paper and prior implementation context establish:

- NLTK WordNet
- noun synsets
- Wu-Palmer similarity
- historical experiment threshold around `0.9`
- historical query `clothes`

The exact historical candidate-selection synset strategy remains unresolved.

Current implementation should therefore support three explicit strategies:

- `first`: compare only the first noun synset for each term
- `max_all`: compute valid Wu-Palmer scores across all noun-synset pairs and use the maximum
- `average`: average all valid Wu-Palmer scores

For the parameterized legacy reproduction call/test, use:

```python
synset_strategy="first"
```

This is a reproduction choice, not a claim that historical candidate code is confirmed to have used first-synset-only behavior.

**Standing reminder:** Historical WordNet candidate synset strategy remains unresolved; first/max-all/average are supported, with `first` used for legacy reproduction until original code confirms the behavior.

Known separate POSID extraction behavior:

```python
color = wn.synsets("color", "n")[0]
```

This verifies that the first noun synset of `color` was used in later clothing/color interpretation logic. It does not prove that WordNet candidate retrieval used first-synset-only behavior.

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

The intended base abstraction includes a structured candidate result, for example:

```python
@dataclass(frozen=True)
class CandidateSentence:
    sentence_id: int
    text: str
    score: float
    query: str | None = None
```

and an abstract selector interface:

```python
class CandidateSelector(ABC):
    @abstractmethod
    def select(self, text, queries):
        ...
```

Keep the implementation minimal.

## 11. Current smoke-test sample

Use:

```text
The suspect was described as a short Asian male, about 145 lbs. wearing a green jacket and blue pants.
```

Historical expected qualitative behavior:

- regex: should return a matched span beginning with `wearing`
- Word2Vec: measure with NLTK pruned Word2Vec vectors
- WordNet: measure with actual WordNet resources using `synset_strategy="first"` for the legacy reproduction call
- zero-shot: measure with the actual classifier environment

Do not fabricate numeric semantic scores.

## 12. Current data and annotation architecture

The current experiments should use the existing **Schema 1.0** annotations directly as stored gold.

Primary current annotation source:

```text
InciText/Annotation/annotations.json
```

The source tree and dataset tree are siblings under the same project root:

```text
project-root/
├── src/
└── InciText/
    └── Annotation/
        ├── annotations.json
        └── consolidated_report_metadata.json
```

Do not rewrite all old annotations into Schema 2.0 for the current benchmark.

The design goal is instead to isolate annotation storage format behind a loader/adapter boundary.

### Intended data-layer structure

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

`facebook.py` and `schema2.py` may remain empty placeholders until those formats are actually needed.

A separate `manifest.py` and `validators.py` are deferred for now; they should be added only if a concrete requirement appears.

### Canonical in-memory representation

All source-specific loaders must normalize annotations into the same internal representation.

Canonical rule:

```text
document_id -> list[PersonAnnotation]
```

Every document maps to a list, including documents with only one annotated person.

The one known legacy document containing two people should load as two separate person records associated with the same `document_id`.

If the source annotation has no explicit person identifier, generate deterministic runtime-only IDs such as:

```text
<document_id>:0
<document_id>:1
```

These IDs are internal conveniences only and must not be written back into the original Schema 1.0 file.

The canonical runtime model should be defined under `schema/models.py`. A minimal conceptual form is:

```python
@dataclass
class PersonAnnotation:
    document_id: str
    person_id: str | None
    role: str | None
    attributes: dict
```

The exact fields may evolve, but downstream code must depend on this canonical model rather than the raw Schema 1.0 dictionary layout.

### Loader boundary

Conceptually:

```text
annotations.json (Schema 1.0)
        ↓
LegacyAnnotationLoader
        ↓
canonical in-memory representation
        ↓
all extractors / runs / evaluation / experiments
```

If another annotation format is introduced later:

```text
new annotation schema
        ↓
NewSchemaAnnotationLoader
        ↓
same canonical in-memory representation
        ↓
unchanged downstream code
```

Changing the on-disk annotation schema should therefore require only a new loader/adapter, not modifications to experiment runners, extractors, or evaluation.

### Facebook data

A future `facebook.py` loader can normalize the Facebook annotations into the same canonical model.

If the Facebook annotations are instead rewritten to match the legacy Schema 1.0 format, they may reuse the legacy loader. No decision is required yet.

### Normalization

`schema/normalize.py` is responsible for canonicalizing values needed by downstream evaluation, such as representation differences in measurements, colors, clothing items, and categorical values.

The raw stored annotations should remain untouched.

### Metadata

`InciText/Annotation/consolidated_report_metadata.json` can be used to recover document/source/subset metadata as needed.

A separate manifest abstraction is not required at the current stage unless dataset discovery or source metadata management becomes difficult.


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

## 18. Immediate next step

Before implementing anything new, inspect the local repository and compare it against:

- `AGENTS.md`
- this file

For the current candidate stage, the candidate package and candidate smoke test remain the active implementation scope. When the data layer is implemented next, preserve Schema 1.0 on disk and introduce the loader/canonical-model boundary described above.


### Authoritative document inventory

The authoritative index for InciText document discovery is:

```text
InciText/Annotation/consolidated_report_metadata.json
```

Only paths listed in this file should be used to locate source documents.

Do not use:

```text
InciText/Annotation/report_metadata.json
```

for dataset/document discovery.

Do not recursively scan the `InciText/` directory and do not infer dataset membership from whatever files or folders happen to be present.

Required behavior:

- read document/path membership from `consolidated_report_metadata.json`
- read the listed source paths from `InciText/Annotation/consolidated_report_metadata.json` and resolve those source paths under the `InciText/` dataset root
- reject any path that resolves outside the dataset root
- if a listed file is missing, report the error rather than searching for an alternate copy
- do not include additional documents that are not listed in the consolidated metadata

This makes dataset membership deterministic and prevents excluded or auxiliary folders from entering experiments merely because they exist on disk.

The document/annotation flows are separate:

```text
InciText/Annotation/consolidated_report_metadata.json
                ↓
      authoritative source paths
                ↓
             loaders.py
                ↓
             Document
```

```text
InciText/Annotation/annotations.json
                ↓
        legacy_adapter.py
                ↓
      PersonAnnotation(s)
```

Join source documents and annotations using `document_id`.


### Lazy source-text loading

Document discovery and document text loading are separate concerns.

Load the lightweight authoritative metadata index first.

Do not read the full corpus text eagerly during initialization.

Load source text lazily/on demand for the requested document, or iterate documents explicitly during benchmark/test runs that require full-corpus processing.

This avoids unnecessary I/O and keeps coding/agent workflows from pulling the entire corpus into context unnecessarily.

### Annotation consolidation

The legacy root file:

```text
InciText/Annotation/annotations.json
```

is not the complete annotation source for the current corpus.

Per-folder annotation files should be consolidated into one generated file:

```text
InciText/Annotation/consolidated_annotation_file.json
```

Use:

```text
src/hart/data/consolidate_annotations.py
```

for this preparation step.

The consolidation utility must use:

```text
InciText/Annotation/consolidated_report_metadata.json
```

as the authoritative dataset index.

It must **not** blindly scan folders.

Required logic:

1. read the authoritative document IDs and source paths from `consolidated_report_metadata.json`
2. derive only the source folders represented by those authoritative records
3. locate the annotation files associated with those folders
4. skip folders with no standalone annotation file, including `Generated_flash_2`
5. merge annotations by `document_id`
6. retain only authoritative document IDs
7. preserve the original list-of-persons structure without normalization at consolidation time
8. accept identical duplicate annotations without duplication
9. raise an error on conflicting duplicate annotations
10. never modify original annotation files
11. write only `consolidated_annotation_file.json`

The generated consolidated annotation file then becomes the runtime legacy annotation source:

```text
consolidated_annotation_file.json
        ↓
legacy_adapter.py
        ↓
canonical in-memory PersonAnnotation objects
```

Normalization of legacy values such as the string `"null"` belongs in the adapter/normalization boundary, not in the consolidation utility.

This keeps consolidation separate from runtime loading and keeps future schema support isolated behind loader/adapter boundaries.
