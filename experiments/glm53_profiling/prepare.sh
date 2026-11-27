#!/usr/bin/env bash
# Runs once in the CPU pod, after ../glm53_agentx/prepare.sh: the small clone (for nsight.sh and for smoke tests of
# forward.sh), the Nsight Perf SDK and the CUDA 13.5 user-mode driver
# (compat package) onto the model disk.
set -euo pipefail

hf download inference-optimization/GLM-5.3-Flash-2.6B-A2.3B-small > /dev/null
mkdir -p /models/nsight /models/results && cd /models/nsight
SDK_URL=https://github.com/NVIDIA/nsight-perf-sdk-for-ai/releases/download/beta-preview-update1
curl -sfL $SDK_URL/nsight-perf-sdk-for-ai-beta-preview-update1-linux-x86_64.tar.gz | tar xz
COMPAT_DEB=cuda-compat-13-5_620.70.10-1ubuntu1_amd64.deb
curl -sfLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/$COMPAT_DEB
dpkg-deb -x $COMPAT_DEB compat135

# The profiling pod runs as another user (root with only CAP_PERFMON), so it reaches the disk through the group.
chmod -R g+rwX /models/hf /models/nsight /models/results /models/home /models/agentx-venv /models/uv-python
echo "NSIGHT_PREPARE_DONE elapsed_time=${SECONDS}s"
