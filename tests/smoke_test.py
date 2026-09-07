from __future__ import annotations

import argparse
from pathlib import Path

from hart.candidates import RegexSelector


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test candidate extraction on a descriptions file.")
    parser.add_argument("input_file", type=Path)
    parser.add_argument("query", nargs="+", help="candidate query phrases")
    arguments = parser.parse_args()

    text = arguments.input_file.read_text(encoding="utf-8")
    candidates = RegexSelector().select(text, arguments.query)

    assert isinstance(candidates, list)
    assert all(candidate.text for candidate in candidates)
    print(f"read {len(text)} characters; selected {len(candidates)} candidate sentences")


if __name__ == "__main__":
    main()