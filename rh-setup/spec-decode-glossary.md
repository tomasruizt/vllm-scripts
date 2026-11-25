# Spec-decode glossary

Terms that agents got wrong in past sessions. Re-verify entries older than a few weeks against the code, and update them here.

* K = `num_speculative_tokens` = draft depth. T = 1 + K = tokens the target scores per step.
* DFlash convention: K = block_size - 1. DSpark convention: K = block_size, so a block_size=8 drafter runs at T=9 (vLLM #65874). Many kernels cap T at 8. (checked 2026-11-23)
* Adaptive verification (AV) does not change K. It decides how many of the K drafted tokens the target verifies. "Drafted", "verified" and "accepted" are three different counts.
* Do not assume a proposer drafts greedily. Read the proposer code for the model at hand.
* `--use-replayssm` covers two mechanisms. ReplaySSM (#48018): checkpoint plus ring buffer of recent inputs, decode and spec-decode, on main only for Mamba2. RecoverSSM (#51855): spec-decode only, reconstructs the accepted state after sampling, on main only for KDA. (checked 2026-11-16)
* A ReplaySSM PR for a family does not imply support for every model of that family, nor for both Hopper and Blackwell.
