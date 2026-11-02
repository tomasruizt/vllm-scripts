"""Portable readers and exports for locally collected benchmark artifacts."""

import csv
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_csv(rows: list[dict], path: Path) -> None:
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_parquet(rows: list[dict], output: Path) -> None:
    """Export through the DuckDB CLI without overwriting a caller's JSONL artifact."""
    output = Path(output)
    with TemporaryDirectory(prefix="parquet-", dir=output.parent) as directory:
        staging = Path(directory) / "rows.jsonl"
        staging.write_text("".join(json.dumps(row) + "\n" for row in rows))
        source_sql = str(staging).replace("'", "''")
        output_sql = str(output).replace("'", "''")
        query = f"COPY (SELECT * FROM read_json_auto('{source_sql}', maximum_object_size=268435456)) TO '{output_sql}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        subprocess.run(["duckdb", "-c", query], check=True)
