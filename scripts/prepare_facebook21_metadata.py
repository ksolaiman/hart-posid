"""Prepare the Facebook-21 cohort for the existing HART data loaders."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from xml.etree import ElementTree
from zipfile import ZipFile


SRC_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = SRC_ROOT.parent / "InciText" / "Facebook"
SOURCE_EXCEL = SOURCE_ROOT / "fb-posts-descriptions-and-uuid.xlsx"
SOURCE_GOLD = SOURCE_ROOT / "Annotation" / "goldfb21.json"
OUTPUT_ROOT = SRC_ROOT / "Facebook"
METADATA_FILENAME = "report_metadata_fb21.json"
ANNOTATIONS_FILENAME = "goldfb21.json"
DOCUMENT_COUNT = 21

_XML_NAMESPACE = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _worksheet_rows(path: Path) -> list[dict[str, str]]:
    """Read the first XLSX worksheet without adding a spreadsheet dependency."""
    with ZipFile(path) as archive:
        worksheet_names = sorted(
            name for name in archive.namelist()
            if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name)
        )
        if not worksheet_names:
            raise ValueError(f"Workbook has no worksheets: {path}")

        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            shared_strings = [
                "".join(node.text or "" for node in item.iterfind(".//main:t", _XML_NAMESPACE))
                for item in shared_root.findall("main:si", _XML_NAMESPACE)
            ]

        worksheet = ElementTree.fromstring(archive.read(worksheet_names[0]))

    rows: list[dict[str, str]] = []
    for row in worksheet.findall(".//main:sheetData/main:row", _XML_NAMESPACE):
        values: dict[str, str] = {}
        for cell in row.findall("main:c", _XML_NAMESPACE):
            match = re.match(r"[A-Z]+", cell.attrib.get("r", ""))
            if match is None:
                continue
            column = match.group()
            cell_type = cell.attrib.get("t")
            value_node = cell.find("main:v", _XML_NAMESPACE)
            value = "" if value_node is None else value_node.text or ""
            if cell_type == "s" and value:
                value = shared_strings[int(value)]
            elif cell_type == "inlineStr":
                value = "".join(
                    node.text or "" for node in cell.iterfind(".//main:t", _XML_NAMESPACE)
                )
            values[column] = value
        rows.append(values)
    return rows


def load_facebook21_inputs(
    excel_path: Path = SOURCE_EXCEL,
    gold_path: Path = SOURCE_GOLD,
) -> tuple[list[tuple[str, str]], dict[str, dict[str, Any]]]:
    rows = _worksheet_rows(excel_path)
    if not rows:
        raise ValueError("Facebook workbook is empty")

    headers = {column: value.strip().casefold() for column, value in rows[0].items()}
    uuid_columns = [column for column, value in headers.items() if value == "uuid"]
    description_columns = [column for column, value in headers.items() if value == "description"]
    if len(uuid_columns) != 1 or len(description_columns) != 1:
        raise ValueError("Facebook workbook must have one UUID and one description column")
    uuid_column, description_column = uuid_columns[0], description_columns[0]

    raw_gold = json.loads(gold_path.read_text(encoding="utf-8"))
    if not isinstance(raw_gold, dict) or len(raw_gold) != DOCUMENT_COUNT:
        raise ValueError(f"Facebook gold must contain exactly {DOCUMENT_COUNT} UUID records")
    if not all(isinstance(key, str) and isinstance(value, dict) for key, value in raw_gold.items()):
        raise ValueError("Facebook gold must map UUID strings to person objects")
    for document_id, person in raw_gold.items():
        if person.get("id") != document_id:
            raise ValueError(f"Facebook gold id does not match its UUID key: {document_id}")

    matched: list[tuple[str, str]] = []
    for row in rows[1:]:
        document_id = row.get(uuid_column, "").strip()
        description = row.get(description_column, "")
        if document_id in raw_gold:
            if not description.strip():
                raise ValueError(f"Facebook description is empty: {document_id}")
            matched.append((document_id, description))

    document_ids = [document_id for document_id, _ in matched]
    if len(matched) != DOCUMENT_COUNT:
        raise ValueError(f"Expected {DOCUMENT_COUNT} populated Facebook records, found {len(matched)}")
    if len(set(document_ids)) != DOCUMENT_COUNT:
        raise ValueError("Facebook workbook contains duplicate selected UUIDs")
    source_ids = set(document_ids)
    gold_ids = set(raw_gold)
    if source_ids != gold_ids:
        raise ValueError(
            "Facebook workbook/gold UUID mismatch: "
            f"missing_gold={sorted(source_ids - gold_ids)}, "
            f"missing_source={sorted(gold_ids - source_ids)}"
        )
    return matched, raw_gold


def _write_new_or_verify(path: Path, content: str) -> None:
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Refusing to overwrite differing file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def prepare(output_root: Path = OUTPUT_ROOT) -> list[str]:
    records, raw_gold = load_facebook21_inputs()
    documents = []
    for document_id, description in records:
        source_path = f"Descriptions/{document_id}.txt"
        _write_new_or_verify(output_root / source_path, description)
        documents.append({
            "id": document_id,
            "subset": "Synthetic",
            "is_synthetic": False,
            "source": {"path": source_path, "media_type": "text/plain"},
        })

    metadata = {"source_root": ".", "documents": documents}
    annotations = {document_id: [raw_gold[document_id]] for document_id, _ in records}
    _write_new_or_verify(
        output_root / "Annotation" / METADATA_FILENAME,
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
    )
    _write_new_or_verify(
        output_root / "Annotation" / ANNOTATIONS_FILENAME,
        json.dumps(annotations, indent=2, ensure_ascii=False) + "\n",
    )
    return [document_id for document_id, _ in records]


def main() -> int:
    document_ids = prepare()
    print(f"metadata: {OUTPUT_ROOT / 'Annotation' / METADATA_FILENAME}")
    print(f"annotations: {OUTPUT_ROOT / 'Annotation' / ANNOTATIONS_FILENAME}")
    print(f"documents: {len(document_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
