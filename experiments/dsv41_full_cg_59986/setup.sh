#!/usr/bin/env bash
# Create the base and PR source trees on the node. Both reuse the precompiled build of BUILD_SRC
# (main at BASE_COMMIT): #59986 is Python-only, so the compiled extensions are shared.
set -euo pipefail
BASE_COMMIT=${BASE_COMMIT:-71e9320633219b3e94a2b0db70647bbfe4e330b9}
PR_COMMIT=${PR_COMMIT:-d29af5185c295a0df21ded0573a379b0531237aa}
REPO=${REPO:-$HOME/code/vllm}
BUILD_SRC=${BUILD_SRC:-$HOME/code/vllm-dsv41}

main() {
    git -C "$REPO" fetch -q origin "$BASE_COMMIT" pull/59986/head
    make_tree "$HOME/code/vllm-59986-base"
    make_tree "$HOME/code/vllm-59986-pr"
    git -C "$HOME/code/vllm-59986-pr" cherry-pick --no-commit "$PR_COMMIT"
    git -C "$HOME/code/vllm-59986-pr" diff --cached --stat
}

make_tree() {
    [[ -d "$1" ]] && { echo "exists: $1"; return; }
    git -C "$REPO" worktree add -q --detach "$1" "$BASE_COMMIT"
    # Copy the build artifacts (extensions, vendored kernels, version) that git ignores.
    (cd "$BUILD_SRC" && git ls-files --others --ignored --exclude-standard -z vllm | grep -zv __pycache__ |
        rsync -a --from0 --files-from=- ./ "$1/")
}

main
