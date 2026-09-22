"""Sanitize copied experiment artifacts; never run against the source worktrees."""

import argparse
import gzip
import hashlib
import json
import re
import sqlite3
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    manifest = args.root / "artifact_manifest.json"
    previous = (
        {r["path"]: r for r in json.loads(manifest.read_text())}
        if manifest.exists()
        else {}
    )
    records = []
    for path in sorted(args.root.rglob("*")):
        if not path.is_file() or path.name in (
            "sanitize_artifacts.py",
            "audit_privacy.py",
            "artifact_manifest.json",
        ):
            continue
        if path.suffix == ".nsys-rep" or "__pycache__" in path.parts:
            path.unlink()
            continue
        before = previous.get(str(path.relative_to(args.root)), {}).get(
            "source_sha256", hashlib.sha256(path.read_bytes()).hexdigest()
        )
        if path.suffix == ".sqlite":
            sanitize_sqlite(path)
        elif path.suffix == ".gz":
            text = gzip.decompress(path.read_bytes()).decode("utf-8")
            path.write_bytes(gzip.compress(clean(text).encode(), mtime=0))
        else:
            data = path.read_bytes()
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                raise RuntimeError(f"Unreviewed binary: {path.relative_to(args.root)}")
            text = clean(text)
            if path.suffix == ".json":
                text = re.sub(r"(:\s*)REDACTED_SECRET\b", r'\1"REDACTED_SECRET"', text)
                json.loads(text)
            path.write_text(text)
        name = clean(path.name)
        if name != path.name:
            path = path.rename(path.with_name(name))
        records.append(
            {
                "path": str(path.relative_to(args.root)),
                "source_sha256": before,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (args.root / "artifact_manifest.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )
    print(
        f"Sanitized and hashed {len(records)} files; opaque Nsight binaries excluded."
    )


def clean(text):
    text = re.sub(r"/home/[^/\s\"'<>]+", "/home/USER", text)
    text = re.sub(r"(?:nm-frk|nm-)[A-Za-z0-9_-]+", "HOST", text)
    text = re.sub(
        r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b",
        "PRIVATE_IP",
        text,
    )
    text = re.sub(
        r"[A-Za-z0-9_.+-]+@(?:redhat\.com|nvidia\.com)", "REDACTED_EMAIL", text
    )
    text = re.sub(
        r"\b(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]+|sk-[A-Za-z0-9_-]{20,})\b",
        "REDACTED_SECRET",
        text,
    )
    text = re.sub(
        r"(?i)(authorization[\"'\s:=]+bearer\s+)[A-Za-z0-9._-]+",
        r"\1REDACTED_SECRET",
        text,
    )
    text = re.sub(
        r"https?://[^/\s\"']*\.(?:corp|internal)\b[^\s\"']*",
        "REDACTED_INTERNAL_URL",
        text,
    )
    return text


def sanitize_sqlite(path):
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA secure_delete=ON")
        tables = [
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        ]
        for table in tables:
            if table.startswith("sqlite_"):
                continue
            quoted = '"' + table.replace('"', '""') + '"'
            if "ENV" in table.upper():
                conn.execute(f"DELETE FROM {quoted}")
                continue
            if table == "META_DATA_CAPTURE":
                conn.execute(
                    f"DELETE FROM {quoted} WHERE name LIKE '%ENVIRONMENT%' OR name LIKE '%USER_NAME%' OR name LIKE '%HOST_NAME%'"
                )
            columns = [r[1] for r in conn.execute(f"PRAGMA table_info({quoted})")]
            for row in conn.execute(f"SELECT rowid, * FROM {quoted}").fetchall():
                for column, value in zip(columns, row[1:]):
                    if column.lower() in ("uuid", "luid") and value is not None:
                        col = '"' + column.replace('"', '""') + '"'
                        replacement = (
                            bytes(len(value))
                            if isinstance(value, bytes)
                            else "REDACTED_DEVICE_ID"
                        )
                        conn.execute(
                            f"UPDATE {quoted} SET {col}=? WHERE rowid=?",
                            (replacement, row[0]),
                        )
                        continue
                    if isinstance(value, str) and (new := clean(value)) != value:
                        col = '"' + column.replace('"', '""') + '"'
                        conn.execute(
                            f"UPDATE {quoted} SET {col}=? WHERE rowid=?", (new, row[0])
                        )
        conn.commit()
        conn.execute("VACUUM")


if __name__ == "__main__":
    main()
