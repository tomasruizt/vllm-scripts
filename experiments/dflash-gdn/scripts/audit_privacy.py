"""Audit public artifacts without printing potential secret values."""

import gzip
import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "credential": r"\b(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]+|sk-[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16})\b",
    "private_key": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    "personal_home": r"/home/(?!USER(?:/|\b))[^/\s\"']+",
    "host": r"\bnm-(?:frk-)?[A-Za-z0-9_-]+",
    "private_ip": r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b",
    "corporate_email": r"[A-Za-z0-9_.+-]+@(?:redhat\.com|nvidia\.com)",
}


def main():
    failures = []
    checked = 0
    for path in ROOT.rglob("*"):
        if (
            not path.is_file()
            or path.name in ("audit_privacy.py", "sanitize_artifacts.py")
            or "__pycache__" in path.parts
        ):
            continue
        if path.suffix == ".nsys-rep":
            failures.append((str(path.relative_to(ROOT)), "opaque_binary"))
            continue
        if path.suffix == ".sqlite":
            conn = sqlite3.connect(path)
            for (name,) in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ):
                if (
                    "ENV" in name.upper()
                    and conn.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]
                ):
                    failures.append((str(path.relative_to(ROOT)), "environment_table"))
            content = "\n".join(conn.iterdump())
            if conn.execute(
                "SELECT count(*) FROM META_DATA_CAPTURE WHERE name LIKE '%ENVIRONMENT%' OR name LIKE '%USER_NAME%' OR name LIKE '%HOST_NAME%'"
            ).fetchone()[0]:
                failures.append(
                    (str(path.relative_to(ROOT)), "private_capture_metadata")
                )
            conn.close()
        else:
            data = (
                gzip.decompress(path.read_bytes())
                if path.suffix == ".gz"
                else path.read_bytes()
            )
            content = data.decode("utf-8")
        for kind, pattern in PATTERNS.items():
            if re.search(pattern, content):
                failures.append((str(path.relative_to(ROOT)), kind))
        checked += 1
    for path, kind in failures:
        print(f"REVIEW {kind}: {path}")
    print(f"Audited {checked} files; {len(failures)} findings.")
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
