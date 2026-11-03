"""Associate native PyTorch NVTX input shapes with NSYS launches and captured graph nodes.

No association by kernel name or launch order. Graph clones follow NSYS originalGraphNodeId edges.
Native autograd annotations contain input sizes, not tensor dtypes or complete output metadata.
"""

from dataclasses import replace
import json
import math
import re

import duckdb
import pandas as pd

from benchmarking.roofline import Estimate, Hardware, estimate, price, unsupported


def annotate_shapes(db: duckdb.DuckDBPyConnection, ops: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = ops.copy()
    defaults = {"operator_name": None, "operator_input_shapes": None, "operator_annotation_id": None,
                "operator_shape_source": None, "operator_kernel_count": None}
    for field, value in defaults.items():
        result[field] = value
    annotations = operator_annotations(db)
    if annotations.empty:
        return result, annotations
    db.register("shape_annotations", annotations)
    launches = ops.reset_index(names="op_index")
    db.register("shape_launches", launches)
    direct = db.execute(DIRECT_SQL).df()
    direct = direct[direct.groupby("op_index").operator_annotation_id.transform("nunique") == 1]
    direct["shape_invocation"] = 0
    graph = graph_associations(db)
    graph_records = [{"globalPid": process, "graphNodeId": node,
                      **{field: value[field] for field in ("operator_annotation_id", "operator_name", "operator_input_shapes")}}
                     for (process, node), value in graph.items()]
    matches = direct
    if graph_records:
        db.register("shape_graph_map", pd.DataFrame(graph_records))
        matched = db.execute("""SELECT l.op_index,l.launch_start AS shape_invocation,
            n.operator_annotation_id,n.operator_name,n.operator_input_shapes,'nsys graph node' AS operator_shape_source
            FROM shape_launches l JOIN shape_graph_map n USING (globalPid,graphNodeId) WHERE l.kind='kernel'""").df()
        matches = pd.concat([direct, matched], ignore_index=True)
    if matches.empty:
        return result, annotations
    matches = matches.drop_duplicates("op_index", keep="last")
    # One formula must not be charged to every kernel of a multi-kernel operator.
    counts = matches.groupby(["operator_annotation_id", "shape_invocation"], dropna=False).size()
    keys = pd.MultiIndex.from_frame(matches[["operator_annotation_id", "shape_invocation"]])
    matches["operator_kernel_count"] = counts.reindex(keys).to_numpy()
    matches = matches.set_index("op_index")
    for field in defaults:
        result.loc[matches.index, field] = matches[field]
    return result, annotations


def estimate_shapes(operator: str, shapes: list[list[int]], dtype: str | None, hardware: Hardware) -> Estimate:
    """Price supported operator semantics, requiring a declared precision, never a model-wide inferred dtype."""
    sizes = {"bf16": 2, "fp16": 2, "fp32": 4, "fp64": 8, "int32": 4, "int64": 8}
    if dtype not in sizes:
        return unsupported(operator, "native NSYS input shapes do not declare tensor precision")
    size = sizes[dtype]
    if operator in ("aten::mm", "aten::bmm"):
        dimensions = 2 if operator == "aten::mm" else 3
        if len(shapes) != 2 or any(len(shape) != dimensions for shape in shapes):
            return unsupported(operator, "unsupported matrix dimensions")
        a = shapes[0]
        b = shapes[1]
        if a[-1] != b[-2] or dimensions == 3 and a[0] != b[0]:
            return unsupported(operator, "incompatible matrix dimensions")
        value = estimate("matmul", {"m": a[-2], "n": b[-1], "k": a[-1], "batches": a[0] if dimensions == 3 else 1,
                                    "a_bytes": size, "b_bytes": size, "output_bytes": size, "compute_dtype": dtype}, hardware)
    elif shapes and shapes[0] and operator in ("aten::relu", "aten::clamp_min", "aten::sum"):
        payload = math.prod(shapes[0]) * size
        writes = 0 if operator == "aten::sum" else payload
        note = "output shape and reduction arithmetic unrecorded" if operator == "aten::sum" else "nonlinear compute ceiling unmodeled"
        value = estimate("traffic", {"read_bytes": payload, "write_bytes": writes, "assumptions": (note,)}, hardware)
    else:
        return unsupported(operator, "no shape-based operator model")
    if value.work is None:
        return value
    assumptions = value.work.assumptions + ("input dimensions recorded by NSYS, output dimensions inferred only for supported operators",
                                          f"uniform operand precision declared as {dtype}, not recorded by native NSYS annotations")
    return price(operator, replace(value.work, assumptions=assumptions), hardware)


def operator_annotations(db: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    raw = db.execute("""SELECT n.start,n."end",n.globalTid,coalesce(n.text,s.value) AS label
        FROM t.NVTX_EVENTS n LEFT JOIN t.StringIds s ON s.id=n.textId
        WHERE n."end" IS NOT NULL AND coalesce(n.text,s.value) LIKE '%sizes = [%'""").df()
    records = []
    for row in raw.itertuples():
        parsed = parse_annotation(row.label)
        if parsed is not None:
            records.append({"operator_annotation_id": f"{row.globalTid}:{row.start}:{row.end}",
                            "start": row.start, "end": row.end, "globalTid": row.globalTid, **parsed})
    columns = ["operator_annotation_id", "start", "end", "globalTid", "operator_name", "operator_input_shapes"]
    return pd.DataFrame(records, columns=columns)


def parse_annotation(label: str) -> dict | None:
    match = re.match(r"([\w:._]+),.*?sizes = ", label)
    if match is None:
        return None
    try:
        shapes, _ = json.JSONDecoder().raw_decode(label[match.end():])
    except (ValueError, TypeError):
        return None
    if not isinstance(shapes, list) or not all(isinstance(shape, list) and all(type(n) is int and n >= 0 for n in shape) for shape in shapes):
        return None
    return {"operator_name": match[1], "operator_input_shapes": json.dumps(shapes)}


def graph_associations(db: duckdb.DuckDBPyConnection) -> dict[tuple[int, int], dict]:
    tables = {row[0] for row in db.execute("SELECT table_name FROM information_schema.tables WHERE table_catalog='t'").fetchall()}
    if "CUDA_GRAPH_NODE_EVENTS" not in tables:
        return {}
    created = db.execute(GRAPH_SQL).df()
    edges = db.execute("""SELECT DISTINCT globalTid // 16777216 * 16777216 AS globalPid,
        graphNodeId,originalGraphNodeId FROM t.CUDA_GRAPH_NODE_EVENTS
        WHERE originalGraphNodeId IS NOT NULL AND globalTid IS NOT NULL""").fetchall()
    direct = {}
    for key, rows in created.groupby(["globalPid", "graphNodeId"]):
        unique = rows.drop_duplicates("operator_annotation_id")
        direct[tuple(map(int, key))] = unique.iloc[0].to_dict() if len(unique) == 1 else None
    parents: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for process, child, parent in edges:
        parents.setdefault((int(process), int(child)), set()).add((int(process), int(parent)))
    resolved = {}
    for node in direct.keys() | parents.keys():
        source = resolve_node(node, direct, parents)
        if source is not None:
            resolved[node] = source
    return resolved


def resolve_node(node: tuple[int, int], direct: dict, parents: dict) -> dict | None:
    seen = set()
    while node not in seen:
        seen.add(node)
        if node in direct:
            return direct[node]
        sources = parents.get(node, set())
        if len(sources) != 1:
            return None
        node = next(iter(sources))
    return None


DIRECT_SQL = """
SELECT l.op_index,l.step,a.operator_annotation_id,a.operator_name,a.operator_input_shapes,
       'nsys launch range' AS operator_shape_source
FROM shape_launches l JOIN shape_annotations a
ON l.globalTid=a.globalTid AND l.launch_start>=a.start AND l.launch_start<a."end"
WHERE l.kind='kernel' AND coalesce(l.graphNodeId,0)=0
QUALIFY rank() OVER (PARTITION BY l.op_index ORDER BY a."end"-a.start,a.start DESC)=1
"""

GRAPH_SQL = """
SELECT n.globalTid // 16777216 * 16777216 AS globalPid,n.graphNodeId,
       a.operator_annotation_id,a.operator_name,a.operator_input_shapes
FROM t.CUDA_GRAPH_NODE_EVENTS n JOIN shape_annotations a
ON n.globalTid=a.globalTid AND n.start>=a.start AND n.start<a."end"
WHERE n.originalGraphNodeId IS NULL
QUALIFY rank() OVER (PARTITION BY n.globalTid,n.graphNodeId,n.start ORDER BY a."end"-a.start,a.start DESC)=1
"""
