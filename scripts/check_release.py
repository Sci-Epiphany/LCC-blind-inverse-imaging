from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = (
    "revision_2026",
    "unified_v13",
    "crossdomain",
    "spatial_kernel",
    "archive_out_of_scope",
)
ARTIFACT_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".pkl",
    ".pyc",
    ".zip",
    ".npz",
}


def main() -> None:
    files = [
        path
        for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(ROOT).parts
    ]
    text_hits = []
    artifact_hits = []
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if relative in {
            "scripts/check_release.py",
            "docs/CODE_THEORY_AUDIT.md",
            "experiments/build_release_package.py",
        }:
            continue
        if path.suffix.lower() in ARTIFACT_SUFFIXES:
            artifact_hits.append(relative)
        if path.suffix.lower() in {".py", ".md", ".yaml", ".yml", ".txt"}:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            if any(token in text for token in FORBIDDEN):
                text_hits.append(relative)
    if text_hits:
        raise RuntimeError("Out-of-scope strings found: " + ", ".join(text_hits))
    if artifact_hits:
        raise RuntimeError("Binary artifacts found: " + ", ".join(artifact_hits))
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    print(f"release check passed: {len(files)} files")
    print(f"tree sha256: {digest.hexdigest()}")


if __name__ == "__main__":
    main()
