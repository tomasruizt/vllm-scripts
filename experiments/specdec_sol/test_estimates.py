"""Cross-model estimate policy, legacy artifacts, execution shapes and honest partial coverage."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kernel_models as km
import sol
import report
import run as runner


class EstimateTests(unittest.TestCase):
    def test_shape_capture_is_opt_in_and_keeps_graph_node_tracing(self) -> None:
        normal = runner.nsys_command({}, Path("trace"))
        shapes = runner.nsys_command({"nsys_shapes": True}, Path("trace"))
        self.assertNotIn("--pytorch", normal)
        self.assertIn("autograd-shapes-nvtx", shapes)
        self.assertEqual(shapes[shapes.index("--cuda-graph-trace") + 1], "node")
        self.assertEqual(shapes[-2:], ["--output", "trace"])

    def test_shape_measurement_excludes_startup_without_changing_timing_columns(self) -> None:
        with duckdb.connect() as db:
            db.execute("CREATE TABLE steps(globalTid BIGINT)")
            db.execute("INSERT INTO steps VALUES (16777227)")
            db.execute("CREATE TABLE nvtx(start BIGINT,globalTid BIGINT,text VARCHAR)")
            db.execute("INSERT INTO nvtx VALUES (100,16777227,'sol/measurement_start'), (300,16777227,'sol/measurement_end')")
            steps = pd.DataFrame(dict(cpu_start=[0,100,200,300],cpu_end=[100,200,300,400],steady=[True]*4,cost_ns=[4,5,6,7]))
            result, rule = sol.measured_steps(db, steps, "decode")
        self.assertEqual(result.steady.tolist(), [False,True,True,False])
        pd.testing.assert_series_equal(result.cost_ns, steps.cost_ns)
        self.assertIn("measurement window", rule)

    def test_shape_models_require_declared_precision_and_single_kernel_operators(self) -> None:
        ops = pd.DataFrame([dict(operator_name="aten::mm",operator_input_shapes="[[8, 16], [16, 32]]",
                                 operator_kernel_count=count,model_family="elementwise",model_sol_ns=np.nan) for count in (1,2)])
        missing = km.shape_models(ops.copy(), {"activation_dtype": "bf16"}, {"bw": 1e12})
        self.assertFalse(missing.operator_shape_estimate.any())
        declared = km.shape_models(ops.copy(), {}, {"bw": 1e12}, "bf16")
        self.assertEqual(declared.operator_shape_estimate.tolist(), [True,False])
        self.assertEqual(declared.model_read_bytes.iloc[0], (8*16+16*32)*2)

    def test_shape_models_can_use_matching_layer_precision_declarations(self) -> None:
        ops = pd.DataFrame([dict(operator_name="aten::mm",operator_input_shapes="[[8, 16], [16, 32]]",
                                 operator_kernel_count=1,model_family="gemm",model_sol_ns=1.0)])
        layer = dict(input_size=16,output_size=32,compute_dtype="bf16")
        result = km.shape_models(ops, {"layers": {"target": [layer]}}, {"bw": 1e12})
        self.assertTrue(result.operator_shape_estimate.iloc[0])
        self.assertEqual(result.model_write_bytes.iloc[0], 8*32*2)

    def test_unknown_forward_kernel_does_not_inherit_hidden_activation_shape(self) -> None:
        info = dict(hidden_size=2048, cache=[], layers={"target": [], "draft": []})
        model = km.Model(info, {"bw": 1e12}, SimpleNamespace(tokens=8, reqs=1, context=100))
        ops = pd.DataFrame([dict(model_family="elementwise")])
        self.assertEqual(model.forward(ops, "target", {}, {}), {})

    def test_unknown_kernel_uses_labeled_counters_without_inventing_semantic_work(self) -> None:
        ops = pd.DataFrame([dict(name="unknown_kernel", kind="kernel", model_sol_ns=np.nan, model_family="elementwise",
                                 sol_pct=99, sm_pct=1, mem_pct=99, ncu_ns=1000, dur_ns=800, kernel_sol_ns=999,
                                 busy_ns=800, ncu_match="exact", bytes=None)])
        sol.sol_sources(ops, {"bw": 1e12})
        self.assertTrue(pd.isna(ops.model_sol_ns.iloc[0]))
        self.assertEqual(ops.sol_ns.iloc[0], 990)
        self.assertEqual(ops.sol_quality.iloc[0], "counter-reference")

    def test_transferred_counter_reference_is_labeled_and_uses_current_duration(self) -> None:
        ops = pd.DataFrame([dict(kind="kernel", model_sol_ns=np.nan, model_family="elementwise", sm_pct=1,
                                 mem_pct=10, dur_ns=2000, ncu_ns=np.nan, busy_ns=0, ncu_match="other grid", bytes=None)])
        sol.sol_sources(ops, {"bw": 1e12})
        self.assertEqual(ops.sol_ns.iloc[0], 200)
        self.assertEqual(ops.sol_quality.iloc[0], "inferred-counter-reference")

    def test_counter_bank_prefers_primary_profile_and_keeps_additional_signatures(self) -> None:
        primary = pd.DataFrame([dict(signature="shared", ncu_ns=100, ncu_source="primary")])
        extra = pd.DataFrame([dict(signature="shared", ncu_ns=200, ncu_source="extra"),
                              dict(signature="new", ncu_ns=300, ncu_source="extra")])
        with patch.object(sol, "ncu_sol", side_effect=[primary, extra]):
            bank = sol.ncu_profiles([Path("primary"), Path("extra")], {"bw": 1e12}).set_index("signature")
        self.assertEqual(bank.loc["shared", "ncu_ns"], 100)
        self.assertEqual(bank.loc["new", "ncu_source"], "extra")

    def test_counter_bank_rejects_different_gpu_type(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "kernels.csv"
            path.write_text('ID,device__attribute_display_name\n,\n0,Other GPU\n')
            with self.assertRaisesRegex(ValueError, "GPU mismatch"):
                sol.ncu_sol(path, {"bw": 1e12, "gpu": "Expected GPU"})

    def test_other_grid_uses_preferred_profile_instead_of_mixing_workloads(self) -> None:
        ncu = pd.DataFrame([dict(signature="kernel|grid1|block", ncu_priority=0, ncu_source="primary", sol_pct=10, sm_pct=1, mem_pct=10),
                            dict(signature="kernel|grid2|block", ncu_priority=1, ncu_source="extra", sol_pct=90, sm_pct=1, mem_pct=90)])
        ops = pd.DataFrame([dict(signature="kernel|grid3|block", kind="kernel")])
        result = sol.match_ncu(ops, ncu)
        self.assertEqual(result.mem_pct.iloc[0], 10)
        self.assertEqual(result.ncu_source.iloc[0], "primary")

    def test_bounds_above_measurement_remain_visible(self) -> None:
        ops = pd.DataFrame([dict(kind="kernel", model_sol_ns=2000, model_family="gemm", busy_ns=1000, dur_ns=1000, bytes=None)])
        sol.sol_sources(ops, {"bw": 1e12})
        self.assertEqual(ops.sol_ns.iloc[0], 2000)

    def test_partial_table_groups_do_not_imply_unknown_kernels_run_at_sol(self) -> None:
        ops = pd.DataFrame([dict(name="same", start=1, busy_ns=1000, gap_ns=0, sol_ns=100),
                            dict(name="same", start=2, busy_ns=2000, gap_ns=0, sol_ns=np.nan)])
        table = sol.breakdown_table(ops, "name", 1)
        self.assertEqual(table.loc["same", "busy"], 3)
        self.assertTrue(pd.isna(table.loc["same", "sol"]))
        self.assertTrue(pd.isna(table.loc["same", "x_sol"]))
        self.assertIn("n/a", sol.breakdown(ops.convert_dtypes(), "name", 1))

    def test_actual_and_padded_shapes_override_scheduler_independently(self) -> None:
        step = SimpleNamespace(tokens=9, actual_tokens=5, padded_tokens=8, av_enabled=True)
        self.assertEqual(km.query_tokens(step), 5)
        self.assertEqual(km.query_tokens(step, padded=True), 8)
        self.assertIsNone(km.query_tokens(step, "draft"))
        self.assertIsNone(km.query_tokens(SimpleNamespace(tokens=9, av_enabled=True)))
        self.assertIsNone(km.query_tokens(SimpleNamespace(tokens=pd.NA)))

    def test_execution_mark_is_joined_without_changing_measured_tokens(self) -> None:
        with duckdb.connect() as db:
            db.execute("CREATE TABLE steps(step INT,start BIGINT,next_start BIGINT,globalTid BIGINT)")
            db.execute("INSERT INTO steps VALUES (1,0,100,7)")
            db.execute("CREATE TABLE nvtx(start BIGINT,globalTid BIGINT,text VARCHAR)")
            mark = 'sol/execution ' + json.dumps(dict(actual_tokens=5, padded_tokens=8, query_lengths=[5]))
            db.execute("INSERT INTO nvtx VALUES (50,7,?)", [mark])
            result = sol.execution_shapes(db, pd.DataFrame([dict(step=1, tokens=9)]), Path("missing.yaml"))
        self.assertEqual(result.tokens.iloc[0], 9)
        self.assertEqual(result.actual_tokens.iloc[0], 5)
        self.assertEqual(result.padded_tokens.iloc[0], 8)

    def test_legacy_llama_metadata_keeps_a_partial_forward_reference(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model-info.json"
            raw = dict(target_weight_bytes=1000, target_weight_elems=500, draft_weight_bytes=100,
                       draft_weight_elems=50, target_kv_bytes_per_token=16, draft_kv_bytes_per_token=4)
            path.write_text(json.dumps({"results": [raw]}))
            info = sol.model_info(path)
        step = SimpleNamespace(tokens=4, context=10, reqs=1)
        value = sol.forward_estimate(info["target"], info, "target", step, {"bw": 1e12})
        self.assertIsNotNone(value.bound_ns)
        self.assertEqual(value.quality, "partial")
        self.assertIn("bf16", value.reason)
        unknown = sol.forward_estimate(info["target"], info, "target", SimpleNamespace(tokens=pd.NA, context=pd.NA, reqs=pd.NA), {"bw": 1e12})
        self.assertEqual(unknown.memory_ns, 1)

    def test_prepared_shape_never_treats_cpu_av_distribution_as_exact(self) -> None:
        torch = SimpleNamespace(nn=SimpleNamespace(Module=object))
        with patch.dict(sys.modules, {"torch": torch}):
            import importlib.util
            spec = importlib.util.spec_from_file_location("probe_shape_test", Path(__file__).with_name("sol_probe.py"))
            probe = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(probe)
        batch = SimpleNamespace(num_reqs=2, num_tokens=5, num_tokens_after_padding=8,
                                query_start_loc_np=np.array([0, 3, 5]), num_computed_tokens_np=np.array([10, 20]))
        self.assertIsNone(probe.prepared_shape(batch, adaptive=True)["query_lengths"])
        self.assertEqual(probe.prepared_shape(batch, adaptive=False)["query_lengths"], [3, 2])

    def test_overlapping_kernel_uses_duration_and_keeps_its_bound(self) -> None:
        op = SimpleNamespace(Index=1, step=1, start=0, end=1000, dur_ns=1000, busy_ns=0, sol_ns=2000,
                             phase="execute", name="gemm", category="gemm", inferred_launch=False,
                             sol_source="semantic: gemm", sol_pct=np.nan, gap_ns=0, kind="kernel", deviceId=0, streamId=7)
        rendered = report.kernel_box(op, lambda time: time / 1000, 0)
        self.assertIn("duration is 0.5x SOL", rendered)
        self.assertIn("SOL: 2.0 µs, from semantic: gemm", rendered)
        self.assertNotIn('width="inf"', rendered)

    def test_stall_summary_separates_pipeline_stalls_from_kernel_cost(self) -> None:
        steps = pd.DataFrame({"gap_ns": [100e3, 200e3, 50e6], "cost_ns": [1e6, 2e6, 60e6]})
        stall = sol.stall_summary(steps)
        self.assertEqual(stall["count"], 1)
        self.assertEqual(stall["threshold_us"], 1000)
        self.assertAlmostEqual(stall["time_share"], 60 / 63)
        self.assertAlmostEqual(stall["mean_cost_ns"], 1.5e6)
        self.assertEqual(sol.stall_check(steps), "1 of 3 (95.2% of the time, host gap above 1 ms)")

    def test_no_stall_keeps_the_mean_unqualified(self) -> None:
        steps = pd.DataFrame({"gap_ns": [100e3, 200e3], "cost_ns": [1e6, 2e6]})
        self.assertIsNone(sol.stall_summary(steps))
        self.assertEqual(sol.stall_check(steps), "none")
        self.assertEqual(sol.stall_lines(steps, ("analytic SOL", 500.0)), [])

    def test_missing_ncu_csv_leaves_sources_na_instead_of_crashing(self) -> None:
        ops = pd.DataFrame({"signature": ["k|1|2"], "kind": "kernel"})
        matched = sol.match_ncu(ops, sol.ncu_profiles([Path("/nonexistent/kernels.csv")], {"bw": 1e12}))
        self.assertEqual(int(matched.ncu_source.isna().sum()), 1)
        self.assertIsNone(matched.ncu_match.iloc[0])

    def test_group_na_reports_busy_coverage_not_op_coverage(self) -> None:
        ops = pd.DataFrame({
            "category": ["gemm"] * 3, "busy_ns": [10e3, 10e3, 980e3], "gap_ns": [0, 0, 0], "start": [0, 1, 2],
            "sol_ns": [1e3, 1e3, np.nan],
        })
        table = sol.breakdown_table(ops, "category", 1)
        self.assertTrue(pd.isna(table.sol.iloc[0]))
        self.assertAlmostEqual(table.mapped.iloc[0], 20e3 / 1000e3)


if __name__ == "__main__":
    unittest.main()
