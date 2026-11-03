"""Semantic work stays independent of kernel decomposition and hardware profiles."""

import unittest

from benchmarking.roofline import Hardware, Work, estimate, price, utilization_reference


class RooflineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hardware = Hardware(1e12, {"bf16": 1e15, "fp32": 1e13})

    def test_projection_counts_activation_io_and_logical_work(self) -> None:
        value = estimate("linear", dict(weight_bytes=8192, weight_elements=4096, tokens=8, compute_dtype="bf16",
                                        input_width=128, output_width=32, activation_bytes=2), self.hardware)
        self.assertEqual(value.work.read_bytes, 8192 + 8 * 128 * 2)
        self.assertEqual(value.work.write_bytes, 8 * 32 * 2)
        self.assertEqual(value.work.flops, {"bf16": 2 * 4096 * 8})
        self.assertEqual(value.quality, "specified")

    def test_same_work_reprices_for_another_gpu(self) -> None:
        work = Work(1e9, flops={"bf16": 1e9})
        fast = price("test", work, self.hardware)
        slow = price("test", work, Hardware(5e11, {"bf16": 5e14}))
        self.assertEqual(slow.bound_ns, 2 * fast.bound_ns)

    def test_counter_reference_does_not_invent_semantic_work(self) -> None:
        value = utilization_reference(1000, 5, 10)
        self.assertEqual(value.bound_ns, 100)
        self.assertIsNone(value.work)
        self.assertEqual(value.quality, "counter-reference")
        self.assertIsNone(utilization_reference(1000, float("nan"), float("nan")).bound_ns)

    def test_missing_fp32_ceiling_is_never_replaced_by_bf16_peak(self) -> None:
        value = price("test", Work(1024, flops={"fp32": 1e9}), Hardware(1e12, {"bf16": 1e15}))
        self.assertIsNone(value.compute_ns)
        self.assertEqual(value.bound_ns, value.memory_ns)
        self.assertEqual(value.quality, "partial")
        self.assertIn("fp32", value.reason)

    def test_unknown_operations_and_invalid_metadata_are_unsupported(self) -> None:
        for operation, meta in (("new_model_magic", {}), ("matmul", {}), ("traffic", {"read_bytes": -1}),
                                ("traffic", {"read_bytes": float("nan")})):
            value = estimate(operation, meta, self.hardware)
            self.assertEqual(value.quality, "unsupported")
            self.assertIsNone(value.bound_ns)

    def test_unknown_query_shape_retains_weight_read_only(self) -> None:
        value = estimate("linear", dict(weight_bytes=8192, weight_elements=4096, tokens=None), self.hardware)
        self.assertEqual(value.work.read_bytes, 8192)
        self.assertEqual(value.work.flops, {})
        self.assertEqual(value.quality, "partial")

    def test_d2d_counts_both_directions_and_host_links_are_unsupported(self) -> None:
        copied = estimate("copy", dict(kind="device_to_device", bytes=4096), self.hardware)
        zeroed = estimate("copy", dict(kind="memset", bytes=4096), self.hardware)
        host = estimate("copy", dict(kind="host_to_device", bytes=4096), self.hardware)
        self.assertEqual(copied.memory_ns, 2 * zeroed.memory_ns)
        self.assertIsNone(host.bound_ns)

    def test_checkpointing_is_an_explicit_amortized_budget(self) -> None:
        meta = dict(state_bytes=2**21, requests=1, input_bytes=64, output_bytes=64, flops={"fp32": 1024})
        baseline = estimate("stateful", meta, self.hardware)
        replay = estimate("stateful", meta | {"checkpoint_write_bytes": 2**21 / 16}, self.hardware)
        self.assertEqual(baseline.work.write_bytes, 64)
        self.assertEqual(replay.work.write_bytes, 64 + 2**21 / 16)
        self.assertEqual(baseline.work.flops, replay.work.flops)

    def test_ragged_attention_uses_query_pairs_and_gqa_kv_heads(self) -> None:
        value = estimate("attention", dict(query_lengths=[2, 1], context_lengths=[10, 20], query_heads=8,
                                           kv_heads=2, head_dim=16, activation_bytes=2, kv_bytes=2,
                                           compute_dtype="bf16"), self.hardware)
        self.assertEqual(value.work.flops["bf16"], 4 * 8 * 16 * (2 * 10 + 3 + 20 + 1))
        self.assertEqual(value.work.read_bytes, 3 * 8 * 16 * 2 + 33 * 2 * 2 * 16 * 2)

    def test_depthwise_convolution_and_reduction(self) -> None:
        conv = estimate("convolution", dict(tokens=8, input_channels=128, output_channels=128, groups=128,
                                            kernel_width=4, activation_bytes=2, weight_bytes=1024, compute_dtype="bf16"), self.hardware)
        self.assertEqual(conv.work.flops["bf16"], 2 * 8 * 128 * 4)
        reduced = estimate("reduction", dict(elements=128, output_elements=4, input_bytes=4, output_bytes=4,
                                             compute_dtype="fp32"), self.hardware)
        self.assertEqual(reduced.work.flops["fp32"], 124)


if __name__ == "__main__":
    unittest.main()
