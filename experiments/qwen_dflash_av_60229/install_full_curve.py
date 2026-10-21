"""Install experiment-only hooks into an isolated copy of frozen vLLM."""

from pathlib import Path
import sys


def main() -> None:
    root = Path(sys.argv[1]) / "vllm/v1/worker/gpu"
    update(root / "spec_decode/adaptive_verification.py", [
        ("        self.num_speculative_steps = req_states.num_speculative_steps\n",
         "        self.num_speculative_steps = req_states.num_speculative_steps\n"
         "        from av_full_curve import FullCurveProbe\n"
         "        self._full_curve = FullCurveProbe(self.num_speculative_steps)\n"),
        ("        draft_budget = int(np.argmax(num_tokens_to_estimated_accepted_tokens / costs))\n",
         "        costs = self._full_curve.get_costs(locals())\n"
         "        draft_budget = int(np.argmax(num_tokens_to_estimated_accepted_tokens / costs))\n"
         "        draft_budget = self._full_curve.select(locals())\n"),
    ])
    update(root / "model_runner.py", [
        ("        self.step_timing.forward_start()\n",
         "        if self.adaptive_verification is not None:\n"
         "            self.adaptive_verification._full_curve.before_forward(input_batch, batch_desc, dummy_run)\n"
         "        self.step_timing.forward_start()\n"),
        ("        # Last rank: sample tokens\n",
         "        if self.adaptive_verification is not None:\n"
         "            self.adaptive_verification._full_curve.forward_end()\n"
         "        # Last rank: sample tokens\n"),
        ("            with use_workspace_lane(self._draft_workspace_lane):\n"
         "                draft_tokens = self.speculator.propose(\n",
         "            if self.adaptive_verification is not None:\n"
         "                self.adaptive_verification._full_curve.draft_start()\n"
         "            with use_workspace_lane(self._draft_workspace_lane):\n"
         "                draft_tokens = self.speculator.propose(\n"),
        ("            if num_spec_tokens < self.num_speculative_steps:\n",
         "            if self.adaptive_verification is not None:\n"
         "                self.adaptive_verification._full_curve.draft_end()\n"
         "            if num_spec_tokens < self.num_speculative_steps:\n"),
    ])


def update(path: Path, replacements: list[tuple[str, str]]) -> None:
    source = path.read_text()
    for before, after in replacements:
        if source.count(before) != 1:
            raise RuntimeError(f"Expected exactly one hook location in {path}: {before!r}")
        source = source.replace(before, after)
    path.write_text(source)


if __name__ == "__main__":
    main()
