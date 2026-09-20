"""Live diagnostic using the reproduction runner's methods and corpus configuration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts import reproduce_legacy_posid as reproduction


def test_stacked_candidate_convergence() -> None:
    args = reproduction.parse_args(["--subset", "historical_loader_16"])
    root = Path(args.incitext_root).resolve()
    records = reproduction._filter_document_records(
        reproduction.load_document_index(root), args.subset
    )
    document_ids = [record.document_id for record in records]
    expected_ids = set(document_ids)
    gold = reproduction.LegacyAnnotationLoader().load(
        root / "Annotation" / "consolidated_annotation_file.json"
    )
    gold = {document_id: gold.get(document_id, []) for document_id in document_ids}
    method_names = (
        "re_posid", "wordnet_posid", "sbert_posid",
        "re_wordnet_posid", "re_sbert_posid",
    )
    print(f"\nDIAGNOSTIC CONFIGURATION: subset={args.subset}, documents={len(records)}",
          flush=True)
    methods, blocked = reproduction._build_methods(
        reproduction.resource_status(), method_names
    )
    if blocked:
        raise RuntimeError(f"Diagnostic requires all five methods: {blocked}")

    predictions = {name: {} for name in method_names}
    candidates = {name: {} for name in method_names}
    categories = {"A": [], "B": [], "C": []}
    with_wearing = []
    without_wearing = []
    sbert_differences = []

    for index, record in enumerate(records, start=1):
        document = reproduction.load_document(record, root)
        document_id = document.document_id
        regex = methods["re_posid"]
        regex_candidates = regex.candidate_selector.select(document.text, regex.queries)
        candidates["re_posid"][document_id] = regex_candidates
        has_wearing = any(
            candidate.query.casefold() == "wearing" for candidate in regex_candidates
        )
        (with_wearing if has_wearing else without_wearing).append(document_id)

        # Call the actual selectors and extractors; do not inject cached candidates
        # or reproduce any selector/extractor logic inside this diagnostic.
        facts = {}
        for name in method_names:
            extractor = methods[name]
            if name != "re_posid":
                candidates[name][document_id] = extractor.candidate_selector.select(
                    document.text, extractor.queries
                )
            result = extractor.extract(document.text, document_id)
            assert set(result) == {document_id}
            assert all(person.document_id == document_id for person in result[document_id])
            predictions[name].update(result)
            # Keep person IDs, descriptor contents/order, and duplicate item facts.
            # Only the detached display list is sorted; predictions stay untouched.
            facts[name] = sorted(
                [
                    {"person_id": person.person_id, "item": item["item"],
                     "descriptors": item["descriptors"]}
                    for person in result[document_id]
                    for item in person.attributes.get("wearing", [])
                ],
                key=json.dumps,
            )

        if facts["sbert_posid"] != facts["re_sbert_posid"]:
            sbert_differences.append(document_id)
            print("\nSTANDALONE SBERT != RE+SBERT", document_id,
                  record.metadata.get("title"), record.source)
            print("SBERT clothing:", json.dumps(facts["sbert_posid"], ensure_ascii=False))
            print("RE+SBERT clothing:", json.dumps(facts["re_sbert_posid"], ensure_ascii=False))

        if not has_wearing:
            identities = {
                name: [
                    (candidate.sentence_id, candidate.text,
                     candidate.context_text, candidate.query)
                    for candidate in candidates[name][document_id]
                ]
                for name in ("wordnet_posid", "sbert_posid")
            }
            # Candidate order is retained because POSID processes candidates in order.
            # Scores are deliberately excluded from cross-selector identity.
            if facts["re_wordnet_posid"] != facts["re_sbert_posid"]:
                category = "C"
            elif identities["wordnet_posid"] == identities["sbert_posid"]:
                category = "A"
            else:
                category = "B"
            categories[category].append(document_id)
            print(f"\nDOCUMENT {document_id}: category {category}")
            print("title:", record.metadata.get("title"), "source:", record.source)
            for name, label in (
                ("re_posid", "Regex fixed candidates"),
                ("wordnet_posid", "WordNet fallback candidates"),
                ("sbert_posid", "SBERT fallback candidates"),
                ("re_wordnet_posid", "RE+WordNet stacked candidates"),
                ("re_sbert_posid", "RE+SBERT stacked candidates"),
            ):
                print(label + ":", json.dumps([
                    {"sentence_id": candidate.sentence_id, "query": candidate.query,
                     "score": candidate.score, "text": candidate.text,
                     "context_text": candidate.context_text}
                    for candidate in candidates[name][document_id]
                ], ensure_ascii=False))
            for name in method_names[1:]:
                print(name + " final clothing facts:",
                      json.dumps(facts[name], ensure_ascii=False))
        print(f"Processed {index}/{len(records)}: {document_id}", flush=True)

    assert len(document_ids) == len(expected_ids)
    assert set(gold) == expected_ids
    assert set(predictions) == set(candidates) == set(method_names)
    for name in method_names:
        assert set(predictions[name]) == expected_ids
        assert set(candidates[name]) == expected_ids
    assert without_wearing, "Diagnostic requires a document without a Regex wearing candidate"
    assigned = [document_id for ids in categories.values() for document_id in ids]
    assert set(assigned) == set(without_wearing)
    assert len(assigned) == len(set(assigned))
    assert sum(len(ids) for ids in categories.values()) == len(without_wearing)
    assert set(with_wearing).isdisjoint(without_wearing)
    assert set(with_wearing) | set(without_wearing) == expected_ids

    print("\nDIAGNOSTIC SUMMARY")
    print("subset:", args.subset)
    print("total documents:", len(records))
    print('documents with Regex "wearing":', len(with_wearing))
    print('documents without Regex "wearing":', len(without_wearing))
    for category, ids in categories.items():
        print(f"category {category}: count={len(ids)}, IDs={sorted(ids)}")
    print("standalone SBERT vs RE+SBERT differing documents:",
          f"count={len(sbert_differences)}, IDs={sorted(sbert_differences)}")

    counts = {}
    print("\nEXACT CORPUS CLOTHING COUNTS (TP, FP, FN)")
    for name in ("sbert_posid", "re_wordnet_posid", "re_sbert_posid"):
        metrics = reproduction.evaluate_corpus(gold, predictions[name], document_ids)
        counts[name] = {}
        for view in ("attr_only", "attr_value"):
            metric = metrics[view]
            counts[name][view] = (
                metric.true_positive, metric.false_positive, metric.false_negative
            )
            print(name, view, counts[name][view])
    equal_counts = counts["sbert_posid"] == counts["re_wordnet_posid"] == counts["re_sbert_posid"]
    print("All three methods have exactly equal AO/AV confusion counts:", equal_counts)
    print("A = selector convergence; B = extractor/resolution convergence; "
          "C = differing stacked predictions. Empty candidate/fact lists count as equal.")
    print("Person IDs are included in prediction equality; corpus evaluation pools people.",
          flush=True)
