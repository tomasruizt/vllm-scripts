"""Verify nested exports and preservation of input artifacts."""

import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from benchmarking.artifacts import write_parquet


class ArtifactTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("duckdb"), "DuckDB CLI required")
    def test_nested_export_preserves_existing_jsonl(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "run's metrics.parquet"
            original = output.with_suffix(".jsonl")
            original.write_text("original artifact\n")
            write_parquet([dict(concurrency=8, nested={"tokens": [1, 2]}, optional=None)], output)
            sql_path = str(output).replace("'", "''")
            query = f"SELECT concurrency, nested.tokens[2] token, optional FROM '{sql_path}'"
            rows = json.loads(subprocess.check_output(["duckdb", "-json", "-c", query], text=True))
            self.assertEqual(rows, [dict(concurrency=8, token=2, optional=None)])
            self.assertEqual(original.read_text(), "original artifact\n")
            self.assertEqual({p.name for p in output.parent.iterdir()}, {output.name, original.name})

    def test_failed_export_cleans_only_its_staging_directory(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "metrics.parquet"
            original = output.with_suffix(".jsonl")
            original.write_text("keep\n")
            with patch("benchmarking.artifacts.subprocess.run", side_effect=subprocess.CalledProcessError(1, "duckdb")):
                with self.assertRaises(subprocess.CalledProcessError):
                    write_parquet([{"x": 1}], output)
            self.assertEqual(list(output.parent.iterdir()), [original])
            self.assertEqual(original.read_text(), "keep\n")
