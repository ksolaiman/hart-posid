"""Run the existing POSID pipeline on the prepared Facebook-21 cohort."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.prepare_facebook21_metadata import (
    ANNOTATIONS_FILENAME,
    DOCUMENT_COUNT,
    METADATA_FILENAME,
    OUTPUT_ROOT as FACEBOOK_ROOT,
    SOURCE_EXCEL,
    SOURCE_GOLD,
    prepare,
)


METHODS = (
    "re_posid",
    "wordnet_posid",
    "re_wordnet_posid",
    "word2vec_posid",
    "sbert_posid",
    "re_sbert_posid",
)


def build_command(mode: str) -> list[str]:
    output_dir = SRC_ROOT / "outputs" / "facebook21" / mode
    command = [
        sys.executable,
        "-u",
        "scripts/reproduce_legacy_posid.py",
        "--incitext-root", str(FACEBOOK_ROOT),
        "--metadata-filename", METADATA_FILENAME,
        "--annotations-filename", ANNOTATIONS_FILENAME,
        "--subset", "non_generated_28",
        "--output-dir", str(output_dir),
        "--methods", *METHODS,
    ]
    if mode == "early_stop":
        command.append("--stop-on-unexpected-pos")
    return command


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_artifacts(output_dir: Path, document_ids: list[str]) -> tuple[list[str], dict[str, str]]:
    results = _load_json(output_dir / "results.json")
    blocked = _load_json(output_dir / "blocked_methods.json")
    expected_ids = set(document_ids)
    methods_run = [method for method in METHODS if method in results]
    if set(methods_run) | set(blocked) != set(METHODS):
        raise RuntimeError("Facebook run did not account for every requested method")
    for method in methods_run:
        predictions = _load_json(output_dir / f"{method}_predictions.json")
        if set(predictions) != expected_ids:
            raise RuntimeError(f"{method} predictions do not cover the Facebook-21 UUIDs")
    return methods_run, blocked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("continuous", "early_stop"))
    args = parser.parse_args()

    document_ids = prepare()
    if len(document_ids) != DOCUMENT_COUNT:
        raise RuntimeError(f"Expected {DOCUMENT_COUNT} prepared Facebook documents")

    output_dir = SRC_ROOT / "outputs" / "facebook21" / args.mode
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing experiment output: {output_dir}")

    command = build_command(args.mode)
    command_text = shlex.join(command)
    print(command_text, flush=True)
    completed = subprocess.run(command, cwd=SRC_ROOT)
    if completed.returncode:
        return completed.returncode

    methods_run, blocked = _validate_artifacts(output_dir, document_ids)
    manifest = {
        "git_branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=SRC_ROOT, text=True
        ).strip(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=SRC_ROOT, text=True
        ).strip(),
        "command": command_text,
        "mode": args.mode,
        "input_excel_path": str(SOURCE_EXCEL.resolve()),
        "gold_json_path": str(SOURCE_GOLD.resolve()),
        "metadata_path": str((FACEBOOK_ROOT / "Annotation" / METADATA_FILENAME).resolve()),
        "document_count": len(document_ids),
        "methods_requested": list(METHODS),
        "methods_run": methods_run,
        "methods_blocked": blocked,
        "stop_on_unexpected_pos": args.mode == "early_stop",
    }
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
