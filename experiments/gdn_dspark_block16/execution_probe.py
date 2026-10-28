"""External, diagnostic-only hooks. No hooks are used in timed runs."""

import dataclasses
import functools
import json
import os
import time
from pathlib import Path


def emit(event: str, **values: object) -> None:
    import torch.distributed as dist

    path = Path(os.environ["EXECUTION_EVIDENCE"])
    with path.open("a") as stream:
        stream.write(
            json.dumps(
                dict(event=event, time=time.time(), rank=dist.get_rank(), **values),
                default=str,
            )
            + "\n"
        )


class ExecutionProbe:
    def init_worker_extension(self) -> None:
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            QwenGatedDeltaNetAttention,
        )
        from vllm.utils.gpu_sync_debug import gpu_sync_allowed
        from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
        from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
            AdaptiveVerificationManager,
        )
        from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

        state = {"step": 0, "layout": None, "gdn_seen": set()}
        enabled = lambda: Path(os.environ["EXECUTION_PROBE_ENABLE"]).exists()

        def loaded(original, obj, *args, **kwargs):
            model = original(obj, *args, **kwargs)
            head = model.model.confidence_head
            emit(
                "draft_configuration",
                sample_from_anchor=obj.sample_from_anchor,
                query_tokens_per_request=obj.num_query_per_req,
                speculative_tokens=obj.num_speculative_steps,
                mask_token_id=obj.parallel_drafting_token_id,
            )
            emit(
                "confidence_head_loaded",
                enabled=obj.use_confidence_head,
                present=head is not None,
                parameters=[(n, list(p.shape)) for n, p in head.named_parameters()]
                if head is not None
                else [],
            )
            return model

        def prepared(original, obj, *args, **kwargs):
            if enabled():
                state["step"] += 1
                state["layout"] = None
            return original(obj, *args, **kwargs)

        def reallocated(original, obj, *args, **kwargs):
            result = original(obj, *args, **kwargs)
            if enabled():
                with gpu_sync_allowed():
                    state["layout"] = result[1].diff().cpu().tolist()
                emit("gpu_query_lens", step=state["step"], query_lens=state["layout"])
            return result

        def confidence(original, obj, probs, batch):
            result = original(obj, probs, batch)
            if enabled():
                emit("confidence_consumed", step=state["step"], shape=list(probs.shape))
            return result

        def replayed(original, obj, desc):
            result = original(obj, desc)
            if enabled() and obj.varlen_decode:
                emit(
                    "target_full_graph_replayed",
                    step=state["step"],
                    descriptor=dataclasses.asdict(desc),
                    query_lens=state["layout"],
                )
            return result

        def eligible(original, obj, metadata):
            result = original(obj, metadata)
            indices = metadata.spec_state_indices_tensor
            width = indices.shape[1] if indices is not None else None
            key = (width, result)
            if key not in state["gdn_seen"]:
                state["gdn_seen"].add(key)
                emit("fused_gdn_eligibility", state_width=width, eligible=result)
            return result

        def wrap(cls, name, hook):
            original = getattr(cls, name)

            @functools.wraps(original)
            def call(obj, *args, **kwargs):
                return hook(original, obj, *args, **kwargs)

            setattr(cls, name, call)

        wrap(DSparkSpeculator, "load_draft_model", loaded)
        wrap(GPUModelRunner, "prepare_inputs", prepared)
        wrap(AdaptiveVerificationManager, "reallocate_drafts", reallocated)
        wrap(AdaptiveVerificationManager, "record_confidences", confidence)
        wrap(CudaGraphManager, "run_fullgraph", replayed)
        wrap(QwenGatedDeltaNetAttention, "_can_use_fused_gdn_mtp_decode", eligible)
