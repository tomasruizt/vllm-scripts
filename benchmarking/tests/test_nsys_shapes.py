"""Native annotation parsing and graph identity, including repeated names, clones and multi-kernel operators."""

import json
import unittest

import duckdb
import pandas as pd

from benchmarking.nsys_shapes import annotate_shapes, estimate_shapes, parse_annotation, resolve_node
from benchmarking.roofline import Hardware


class ShapeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db = duckdb.connect()
        self.db.execute("ATTACH ':memory:' AS t")
        self.db.execute('CREATE TABLE t.NVTX_EVENTS(start BIGINT,"end" BIGINT,globalTid BIGINT,text VARCHAR,textId INT)')
        self.db.execute('CREATE TABLE t.StringIds(id INT,value VARCHAR)')
        self.process = 16777216
        self.thread = self.process + 11

    def tearDown(self) -> None:
        self.db.close()

    def test_parse_native_label_with_trailing_tensor_ids(self) -> None:
        parsed = parse_annotation('aten::mm, seq = 0, sizes = [[8, 256], [256, 512]], input_op_ids = [(1,0), (4,0)]')
        self.assertEqual(parsed['operator_name'], 'aten::mm')
        self.assertEqual(json.loads(parsed['operator_input_shapes']), [[8, 256], [256, 512]])
        self.assertIsNone(parse_annotation('aten::mm, sizes = [[-1, 256]]'))
        self.assertIsNone(parse_annotation('sol/execute'))

    def test_eager_launch_selects_innermost_operator(self) -> None:
        self.annotation(100, 150, 'aten::relu', [[8, 16]])
        self.annotation(110, 140, 'aten::clamp_min', [[8, 16], []])
        ops = self.ops([dict(launch_start=120)])
        result, annotations = annotate_shapes(self.db, ops)
        self.assertEqual(result.operator_name.iloc[0], 'aten::clamp_min')
        self.assertEqual(result.operator_shape_source.iloc[0], 'nsys launch range')
        self.assertEqual(len(annotations), 2)
        pd.testing.assert_series_equal(result.launch_start, ops.launch_start)

    def test_clone_chain_and_process_identity_preserve_different_shapes(self) -> None:
        self.graph_table()
        self.annotation(100, 150, 'aten::relu', [[8, 16]])
        self.annotation(100, 150, 'aten::relu', [[4, 32]], self.thread + self.process)
        nodes = [(120, self.thread, 10, None), (160, self.thread, 22, 10), (170, self.thread, 33, 22),
                 (120, self.thread + self.process, 10, None), (170, self.thread + self.process, 33, 10)]
        self.nodes(nodes)
        ops = self.ops([dict(graphNodeId=33, launch_start=1000), dict(graphNodeId=33, launch_start=2000),
                        dict(graphNodeId=33, globalPid=2*self.process, globalTid=self.thread+self.process, launch_start=1000)])
        result, _ = annotate_shapes(self.db, ops)
        self.assertEqual(result.operator_input_shapes.tolist(), ['[[8, 16]]', '[[8, 16]]', '[[4, 32]]'])
        self.assertEqual(result.operator_kernel_count.tolist(), [1, 1, 1])
        self.assertEqual(set(result.operator_shape_source), {'nsys graph node'})

    def test_multi_kernel_operator_is_counted_per_graph_launch(self) -> None:
        self.graph_table()
        self.annotation(100, 150, 'aten::mm', [[8, 16], [16, 32]])
        self.nodes([(120, self.thread, 10, None), (125, self.thread, 11, None)])
        ops = self.ops([dict(graphNodeId=10, launch_start=1000), dict(graphNodeId=11, launch_start=1000),
                        dict(graphNodeId=10, launch_start=2000), dict(graphNodeId=11, launch_start=2000)])
        result, _ = annotate_shapes(self.db, ops)
        self.assertEqual(result.operator_kernel_count.tolist(), [2, 2, 2, 2])

    def test_multi_kernel_eager_operator_does_not_duplicate_work(self) -> None:
        self.annotation(100, 150, 'aten::mm', [[8, 16], [16, 32]])
        result, _ = annotate_shapes(self.db, self.ops([dict(launch_start=120), dict(launch_start=130)]))
        self.assertEqual(result.operator_kernel_count.tolist(), [2, 2])

    def test_reused_node_with_conflicting_shapes_is_unmapped(self) -> None:
        self.graph_table()
        self.annotation(100, 150, 'aten::relu', [[8, 16]])
        self.annotation(200, 250, 'aten::relu', [[4, 32]])
        self.nodes([(120, self.thread, 10, None), (220, self.thread, 10, None)])
        result, _ = annotate_shapes(self.db, self.ops([dict(graphNodeId=10)]))
        self.assertTrue(result.operator_name.isna().all())

    def test_missing_capture_metadata_does_not_match_by_name(self) -> None:
        self.annotation(100, 150, 'aten::relu', [[8, 16]])
        result, _ = annotate_shapes(self.db, self.ops([dict(name='relu', graphNodeId=10, launch_start=1000)]))
        self.assertTrue(result.operator_name.isna().all())

    def test_ambiguous_or_cyclic_clone_edges_are_rejected(self) -> None:
        self.assertIsNone(resolve_node((1, 2), {}, {(1, 2): {(1, 2)}}))
        self.assertIsNone(resolve_node((1, 2), {}, {(1, 2): {(1, 3), (1, 4)}}))

    def test_matrix_work_uses_recorded_shapes_and_declared_precision(self) -> None:
        value = estimate_shapes('aten::mm', [[8, 256], [256, 512]], 'bf16', Hardware(1e12, {'bf16': 1e12}))
        self.assertEqual(value.work.read_bytes, (8*256 + 256*512)*2)
        self.assertEqual(value.work.write_bytes, 8*512*2)
        self.assertEqual(value.work.flops, {'bf16': 2*8*256*512})
        self.assertEqual(value.quality, 'assumed')
        self.assertIsNone(estimate_shapes('aten::mm', [[8, 256], [256, 512]], None, Hardware(1e12)).bound_ns)

    def test_unknown_or_incompatible_operations_remain_unsupported(self) -> None:
        hardware = Hardware(1e12)
        self.assertIsNone(estimate_shapes('aten::view', [[8, 16]], 'bf16', hardware).bound_ns)
        self.assertIsNone(estimate_shapes('aten::mm', [[8, 15], [16, 32]], 'bf16', hardware).bound_ns)

    def annotation(self, start: int, end: int, operator: str, shapes: list, thread: int | None = None) -> None:
        label = operator + ', sizes = ' + json.dumps(shapes)
        self.db.execute('INSERT INTO t.NVTX_EVENTS VALUES (?,?,?,?,NULL)', [start, end, thread or self.thread, label])

    def graph_table(self) -> None:
        self.db.execute('CREATE TABLE t.CUDA_GRAPH_NODE_EVENTS(start BIGINT,globalTid BIGINT,graphNodeId BIGINT,originalGraphNodeId BIGINT)')

    def nodes(self, nodes: list[tuple]) -> None:
        self.db.executemany('INSERT INTO t.CUDA_GRAPH_NODE_EVENTS VALUES (?,?,?,?)', nodes)

    def ops(self, rows: list[dict]) -> pd.DataFrame:
        defaults = dict(kind='kernel', name='same_kernel', globalPid=self.process, globalTid=self.thread,
                        graphNodeId=0, launch_start=1000, step=1)
        return pd.DataFrame([defaults | row for row in rows], index=range(10, 10+len(rows)))


if __name__ == '__main__':
    unittest.main()
