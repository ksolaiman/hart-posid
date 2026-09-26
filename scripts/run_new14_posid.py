"""Run the existing POSID experiment on the 14-document extension set."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("continuous", "early_stop"))
    mode = parser.parse_args().mode

    command = [
        sys.executable,
        "-u",
        "scripts/reproduce_legacy_posid.py",
        "--incitext-root", "../InciText",
        "--metadata-filename", "report_metadata_new14.json",
        "--annotations-filename", "annotation_additions.json",
        "--subset", "non_generated_28",
        "--output-dir", f"outputs/new14/{mode}",
        "--methods",
        "re_posid", "wordnet_posid", "re_wordnet_posid",
        "word2vec_posid", "sbert_posid", "re_sbert_posid",
    ]
    if mode == "early_stop":
        command.append("--stop-on-unexpected-pos")

    print(shlex.join(command), flush=True)
    return subprocess.run(command, cwd=Path(__file__).resolve().parents[1]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
