# find-gpus

Report available NVIDIA GPUs across SSH hosts, with preferred regions first. Requires local Python 3 and SSH, plus `canhazgpu` (with JSON status support) and `nvidia-smi` on each remote host. No Python packages are needed.

## Setup

From this directory:

```sh
mkdir -p "$HOME/.local/bin" "$HOME/.config/find-gpus"
ln -s "$PWD/find-gpus" "$HOME/.local/bin/find-gpus"
# For first-time setup only; preserve any existing private configuration.
(umask 077; set -C; cat hosts.example.json > "$HOME/.config/find-gpus/hosts.json")
```

Ensure `~/.local/bin` is on your PATH. Edit the private configuration to provide SSH aliases from your own `~/.ssh/config`, regions, and GPU descriptions. The example contains fictional aliases and cannot connect until configured. Keep usernames, addresses and identity files in your private SSH configuration.

## Usage

```sh
find-gpus
find-gpus --region eu
find-gpus --available
find-gpus --host 'example-*'
find-gpus --json
find-gpus --prefer eu,us-west,us-east,unknown
```

The default configuration is `~/.config/find-gpus/hosts.json`; use `--config` to select another private file. Regions are user-provided labels. Default ordering is EU, US East, US West, then unknown, with more free GPUs first within each priority group. `--region eu` excludes hosts labeled `unknown`.

Free means AVAILABLE/FREE in reservation status, no compute process, 0% GPU utilization, and at most 256 MiB memory used. `--memory-threshold` changes the memory limit. Reserved-but-idle and unreserved-but-busy GPUs are not free. Missing, failed or unrecognized telemetry is reported as unknown. Gaudi is reported as unsupported; it requires different telemetry. Queries are snapshots; reserve GPUs before starting work.

Checks run concurrently (`--jobs`, default 32), with a total timeout per host
(`--timeout`, default 25 seconds). SSH accepts new host keys and rejects changed keys. The command does not reserve or release GPUs or stop workloads. Exit 1 means all selected hosts failed verification; exit 2 means invalid arguments or configuration. JSON includes `checked_at` and `failed_hosts` even with `--available`.

## Privacy

Keep real host lists, SSH configuration, inventories, credentials, and command output outside this repository. Only generic code and fictional fixtures are included here. The local `.gitignore` allowlists the intended public files as a backstop; it cannot prevent force-adds or protect previously tracked files.

Runtime output is **private**: it intentionally includes configured SSH aliases, regions, GPU counts, and may include account names or addresses in SSH errors. Do not commit reports or paste them into public issues. Extra inventory fields such as addresses, VPN labels, and owners are not copied into JSON reports.

Run the synthetic tests without network access:

```sh
python3 test_find_gpus.py
```
