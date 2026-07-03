#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


DEFAULT_DOCS_REL = Path("reference") / "asc-devkit-api-9.0.0" / "docs" / "api"


def configure_output() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def normalize(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def candidate_roots(start: Path) -> list[Path]:
    roots: list[Path] = []
    for item in [start, *start.parents]:
        roots.append(item)
    script = Path(__file__).resolve()
    for item in [script.parent, *script.parents]:
        roots.append(item)
    seen = set()
    unique = []
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def resolve_docs_root(raw: str) -> Path:
    candidates: list[Path] = []
    if raw:
        candidates.append(Path(raw).expanduser())
    env_root = os.environ.get("ASCENDC_API_DOCS_ROOT", "")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    for root in candidate_roots(Path.cwd().resolve()):
        candidates.append(root / DEFAULT_DOCS_REL)

    for path in candidates:
        resolved = path.resolve()
        if (resolved / "README.md").is_file() and (resolved / "context").is_dir():
            return resolved
    tried = "\n".join(f"- {path}" for path in candidates[:12])
    raise SystemExit(f"error: cannot locate Ascend C API docs root. Tried:\n{tried}")


def title_for(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                if line.startswith("#"):
                    return line.lstrip("#").strip()
    except OSError:
        pass
    return path.stem


def first_hit(path: Path, terms: list[str]) -> tuple[int, str]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for index, line in enumerate(handle, start=1):
                lowered = line.lower()
                if any(term.lower() in lowered for term in terms):
                    return index, line.strip()
    except OSError:
        pass
    return 0, ""


def score_file(path: Path, docs_root: Path, terms: list[str], content: bool) -> tuple[int, int, str]:
    rel = path.relative_to(docs_root).as_posix()
    stem = normalize(path.stem)
    rel_norm = normalize(rel)
    score = 0
    for term in terms:
        term_norm = normalize(term)
        if not term_norm:
            continue
        if stem == term_norm:
            score += 120
        elif term_norm in stem:
            score += 60
        if term_norm in rel_norm:
            score += 20
    line_no = 0
    line_text = ""
    if content:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            text = ""
        for term in terms:
            term_norm = term.lower()
            count = text.count(term_norm)
            if count:
                score += min(count, 12) * 4
        line_no, line_text = first_hit(path, terms)
    return score, line_no, line_text


def lookup(args: argparse.Namespace) -> int:
    docs_root = resolve_docs_root(args.docs_root)
    terms = [term.strip() for term in args.terms if term.strip()]
    if not terms:
        print("error: at least one query term is required", file=sys.stderr)
        return 1

    rows = []
    for path in docs_root.rglob("*.md"):
        score, line_no, line_text = score_file(path, docs_root, terms, not args.filename_only)
        if score > 0:
            rows.append((score, path, line_no, line_text))
    rows.sort(key=lambda item: (-item[0], item[1].as_posix()))

    print(f"docs_root: {docs_root}")
    print(f"query: {' '.join(terms)}")
    print(f"matches: {len(rows)}")
    for rank, (score, path, line_no, line_text) in enumerate(rows[: args.limit], start=1):
        rel = path.relative_to(docs_root).as_posix()
        print(f"{rank}. score={score} {rel}")
        print(f"   title: {title_for(path)}")
        if line_no and line_text:
            print(f"   hit: {line_no}: {line_text[:220]}")
    if not rows:
        print("No matches. Try API name, struct name, parameter name, and Chinese topic keywords.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search the offline Ascend C API docs snapshot.")
    parser.add_argument("terms", nargs="+", help="API names or keywords, e.g. DataCopyPad TSCM Nd2NzParams")
    parser.add_argument("--docs-root", default="", help="Path to docs/api. Defaults to ASCENDC_API_DOCS_ROOT or repository reference snapshot.")
    parser.add_argument("--limit", type=int, default=12, help="Maximum number of matches to print.")
    parser.add_argument("--filename-only", action="store_true", help="Only score file paths and names, without scanning file contents.")
    return parser


def main() -> int:
    configure_output()
    parser = build_parser()
    return lookup(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
