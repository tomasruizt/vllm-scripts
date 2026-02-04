# Gemma-3 Multimodal Models: torch.compile Bug

This issue affects **main branch** and is unrelated to PR #33318.

## Models Tested

| Model | Architecture | Vocab Size | Status |
|-------|-------------|------------|--------|
| google/gemma-3-270m-it | Gemma3ForCausalLM (text-only) | 262144 | **WORKS** |
| google/gemma-3-1b-it | Gemma3ForCausalLM (text-only) | 262144 | **WORKS** |
| google/gemma-3-4b-it | Gemma3ForConditionalGeneration (multimodal) | 262208 | **FAILS** |
| google/gemma-3-12b-it | Gemma3ForConditionalGeneration (multimodal) | 262208 | **FAILS** |
| google/gemma-3-27b-it | Gemma3ForConditionalGeneration (multimodal) | 262208 | **FAILS** |

## Error Message

The multimodal Gemma-3 models (4b, 12b, 27b) fail with a `torch.compile` assertion error:

```
AssertionError: expected size 1048576==131072, stride 128==128 at dim=0
This error most often comes from a incorrect fake (aka meta) kernel for a custom op.
```

## Analysis

**Key observation**: The size mismatch ratio (1048576 / 131072 = 8) matches the `rope_scaling.factor` of 8.0 in the model config.

**Root cause**: Likely a bug in vLLM's torch.compile integration for multimodal Gemma-3 models with rope scaling.

## Workaround

Use the text-only Gemma-3 models (270m, 1b) which work correctly.

## Notes

- Clearing torch compile cache (`~/.cache/vllm/torch_compile_cache`, `/tmp/torchinductor_*`) did not fix the issue
- The `--enforce-eager` flag was not tested
