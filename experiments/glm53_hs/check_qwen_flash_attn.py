"""Check whether the B300 Qwen failures depend on automatic backend selection."""

from functools import partial

import pytest

from tests.v1.kv_connector.extract_hidden_states_integration import test_extraction
from vllm import LLM


def main():
    test_extraction.LLM = partial(LLM, attention_config={"backend": "FLASH_ATTN"})
    return pytest.main(
        [
            "tests/v1/kv_connector/extract_hidden_states_integration/test_extraction.py",
            "-k", "qwen35", "-q", "-rs",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
