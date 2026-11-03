"""Profile one spec-decode workload, on the GPU host. See configs/ for the workload format.

    run.py CONFIG.yaml                    # launcher: one chg reservation per concurrency level, in parallel
    run.py CONFIG.yaml --concurrency 32   # one level, inside a reservation: nsys pass, then (TP1 only) ncu pass
"""

import argparse
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
DEFAULTS = {
    "gpus": 1, "concurrencies": [1, 32], "passes": ["nsys", "ncu"], "env": {}, "timeout": "3h",
    "python": "~/.venv/bin/python", "aiperf": "~/aiperf-venv/bin/aiperf", "request_extra": {},
    "max_completion_tokens": 1024, "capture_seconds": 3, "ncu_idle_seconds": 120, "ncu_max_seconds": 1800,
    "ncu_gpu_memory_utilization": 0.3, "nsys_shapes": False,
    # ncu >= 2026.1, for --communicator with TP>1, e.g. from NVIDIA's conda channel (nsight-compute package)
    "ncu": "~/opt/ncu-2026.2.1/nsight-compute-2026.2.1/ncu",
}  # fmt: skip
NSYS = [
    "nsys", "profile", "--trace", "cuda,nvtx", "--sample", "process-tree", "--cpuctxsw", "process-tree",
    "--cuda-graph-trace", "node", "--capture-range", "cudaProfilerApi", "--capture-range-end", "stop",
    "--export", "sqlite", "--force-overwrite", "true", "--output",
]  # fmt: skip
# Only the metrics sol.py uses, instead of the full SpeedOfLight section.
NCU_METRICS = [
    "gpu__time_duration.sum", "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", "dram__bytes_read.sum", "dram__bytes_write.sum",
]  # fmt: skip
NCU = [
    "--target-processes", "all", "--profile-from-start", "off", "--graph-profiling", "node",
    "--metrics", ",".join(NCU_METRICS), "--clock-control", "none", "--cache-control", "none", "-f",
    # the first launch of every (mangled name, grid, block, shared memory), i.e. once per signature sol.py matches on
    "--filter-mode", "per-launch-config", "--kernel-name-base", "mangled", "--kernel-id", "::regex:.*:1",
]  # fmt: skip


def nsys_command(config: dict, output: Path) -> list[str]:
    annotations = ["--pytorch", "autograd-shapes-nvtx"] if config.get("nsys_shapes") else []
    return NSYS[:-1] + annotations + ["--output", str(output)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--pass", dest="pass_name", choices=["nsys", "ncu"])
    args = parser.parse_args()
    config = {**DEFAULTS, **yaml.safe_load(args.config.read_text())}
    if args.concurrency is None:
        launch_all(config, args.config)
    else:
        Level(config, args.concurrency, args.config.resolve().parent).profile(args.pass_name)


def launch_all(config: dict, config_path: Path) -> None:
    """The config sits in the run directory. Every (level, pass) gets its own reservation, so all run in parallel,
    and writes into c<N>/<pass>/."""
    root = config_path.resolve().parent  # the run directory
    (root / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    record_source(config, root)
    passes = config["passes"]
    jobs = [launch_job(config, config_path, c, p) for c in config["concurrencies"] for p in passes]
    codes = [job.wait() for job in jobs]
    (root / "launch-exit-code.txt").write_text(str(max(codes)))


def launch_job(config: dict, config_path: Path, concurrency: int, pass_name: str) -> subprocess.Popen:
    note = f"specdec SOL c{concurrency} {pass_name}"
    reserve = ["chg", "run", "--gpus", str(config["gpus"]), "--timeout", config["timeout"], "--note", note]
    level = ["--concurrency", str(concurrency), "--pass", pass_name]
    command = reserve + ["--", sys.executable, str(HERE / "run.py"), str(config_path.resolve())] + level
    log = (config_path.resolve().parent / f"c{concurrency}-{pass_name}-job.log").open("w")
    return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)


def record_source(config: dict, root: Path) -> None:
    """The vLLM checkout the server imports, its commit and local diff."""
    python = os.path.expanduser(config["python"])
    env = {**os.environ, **config["env"]}
    code = "import os, vllm; print(os.path.dirname(os.path.dirname(vllm.__file__)))"
    source = subprocess.run([python, "-c", code], env=env, capture_output=True, text=True, check=True).stdout.strip()
    log = subprocess.run(["git", "-C", source, "log", "-1", "--format=%H %cd %s", "--date=short"], capture_output=True, text=True)
    (root / "source-commit.txt").write_text(log.stdout or f"unknown {source}\n")
    diff = subprocess.run(["git", "-C", source, "diff"], capture_output=True, text=True)
    (root / "source.patch").write_text(diff.stdout)


class Level:
    """One pass of one concurrency level: nsys for the timeline, or ncu for per-kernel SOL."""

    def __init__(self, config: dict, concurrency: int, run_dir: Path) -> None:
        self.config = config
        self.concurrency = concurrency
        self.root = run_dir / f"c{concurrency}"
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.server: subprocess.Popen | None = None

    def profile(self, pass_name: str) -> None:
        directory = self.root / pass_name
        directory.mkdir(parents=True, exist_ok=True)
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
        exit_code = 1
        try:
            getattr(self, f"{pass_name}_pass")(directory)
            exit_code = 0
        finally:
            self.stop_server()
            (directory / "exit-code.txt").write_text(str(exit_code))

    def nsys_pass(self, directory: Path) -> None:
        self.start_server(nsys_command(self.config, directory / "trace"), directory, self.config["serve_args"])
        if self.config.get("nsys_shapes"):
            status = self.rpc("sol_shape_capture_status").get("results", [])
            if not status or not all(status):
                raise RuntimeError("shape capture did not start before CUDA graph construction, check the worker extension startup hook")
        (directory / "model-info.json").write_text(json.dumps(self.rpc("sol_model_info"), indent=2))
        if not self.config.get("nsys_shapes"):
            self.aiperf(directory, "reference", duration=30).wait()
        client = self.aiperf(directory, "capture", duration=25)
        time.sleep(12)
        if self.config.get("nsys_shapes"):
            self.rpc("sol_start_shape_measurement")
        else:
            self.post("/start_profile")
        time.sleep(self.config["capture_seconds"])
        if self.config.get("nsys_shapes"):
            self.rpc("sol_stop_shape_measurement")
        else:
            self.post("/stop_profile")
        client.wait()
        self.stop_server()

    def ncu_pass(self, directory: Path) -> None:
        """A small KV pool lets ncu back up device memory on the GPU for kernel replay, not in (much slower) host memory."""
        serve_args = with_flag(self.config["serve_args"], "--gpu-memory-utilization", self.config["ncu_gpu_memory_utilization"])
        ncu = [os.path.expanduser(self.config["ncu"])] + NCU + lockstep(self.config["gpus"])
        self.start_server(ncu + ["--export", str(directory / "kernels")], directory, serve_args)
        client = self.aiperf(directory, "capture", duration=3600)
        time.sleep(20)
        self.post("/start_profile", timeout=3600)
        self.wait_for_ncu(directory / "server.log")
        client.send_signal(signal.SIGINT)
        self.stop_server()
        export = [os.path.expanduser(self.config["ncu"]), "--import", str(directory / "kernels.ncu-rep"), "--csv", "--page", "raw"]
        run(export + ["--print-kernel-base", "mangled", "--print-units", "base"], directory / "kernels.csv")

    def wait_for_ncu(self, log: Path) -> None:
        """ncu profiles each new signature once, so it stops on its own when no new ones show up. Return after
        ncu_idle_seconds without a new one (shapes that vary per step keep trickling in), or after ncu_max_seconds."""
        started = time.monotonic()
        last = -1
        last_change = started
        while time.monotonic() - last_change < self.config["ncu_idle_seconds"]:
            if time.monotonic() - started > self.config["ncu_max_seconds"]:
                return
            profiled = log.read_text(errors="replace").count("==PROF== Profiling")
            self.status(f"ncu profiled {profiled} kernel signatures")
            if profiled != last:
                last = profiled
                last_change = time.monotonic()
            time.sleep(10)

    def start_server(self, profiler: list[str], directory: Path, serve_args: list) -> None:
        python = os.path.expanduser(self.config["python"])
        own = ["--port", self.port, "--host", "127.0.0.1", "--max-num-seqs", self.concurrency,
               "--worker-extension-cls", "sol_probe.SolProbe", "--profiler-config", '{"profiler":"cuda"}']  # fmt: skip
        command = profiler + [python, "-m", "vllm.entrypoints.openai.api_server", "--model", self.config["model"]]
        command = [str(arg) for arg in command + list(serve_args) + own]
        (directory / "command.json").write_text(json.dumps(command, indent=2))
        env = self.server_env()
        if self.config.get("nsys_shapes") and profiler[0] == "nsys":
            env["SOL_NSYS_SHAPES"] = "1"  # Do not start an NCU pass during graph construction.
        log = (directory / "server.log").open("w")
        self.server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.wait_for_health(directory)
        (directory / "probe.json").write_text(json.dumps(self.rpc("init_worker_extension")))  # vLLM does not call it
        self.status(f"READY {directory.name}")

    def server_env(self) -> dict:
        env = {**os.environ, **{k: str(v) for k, v in self.config["env"].items()}, "VLLM_SERVER_DEV_MODE": "1"}
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(HERE), env.get("PYTHONPATH")]))
        venv = os.path.dirname(os.path.expanduser(self.config["python"]))  # like an activated venv, e.g. for ninja
        env["PATH"] = f"{venv}:{env['PATH']}:/usr/local/cuda/bin"  # last, /usr/local/cuda/bin holds an nsys stub
        env["TMPDIR"] = str(self.root / "tmp")  # nsys fails when /tmp/nvidia belongs to another user
        (self.root / "tmp").mkdir(exist_ok=True)
        return env

    def wait_for_health(self, directory: Path) -> None:
        deadline = time.monotonic() + 3600
        while time.monotonic() < deadline:
            if self.server.poll() is not None:
                raise RuntimeError(f"server exited {self.server.returncode}, see {directory / 'server.log'}")
            try:
                urllib.request.urlopen(self.url + "/health", timeout=2)
                return
            except OSError:
                time.sleep(5)
        raise TimeoutError("server startup exceeded 1 hour")

    def stop_server(self) -> None:
        """SIGINT the whole group so nsys/ncu forward it and write their reports."""
        if self.server is None or self.server.poll() is not None:
            return
        os.killpg(self.server.pid, signal.SIGINT)
        try:
            self.server.wait(timeout=1200)
        except subprocess.TimeoutExpired:
            os.killpg(self.server.pid, signal.SIGKILL)
            self.server.wait()
        self.server = None

    def aiperf(self, directory: Path, phase: str, duration: int) -> subprocess.Popen:
        path = directory / f"aiperf-{phase}.yaml"
        path.write_text(yaml.safe_dump(self.aiperf_config(directory / f"aiperf-{phase}", duration), sort_keys=False))
        log = (directory / f"aiperf-{phase}.log").open("w")
        command = ["env", "TMPDIR=/tmp", os.path.expanduser(self.config["aiperf"]), "profile", "--config", str(path)]
        return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)

    def aiperf_config(self, artifacts: Path, duration: int) -> dict:
        phase = {"name": "profiling", "type": "concurrency", "concurrency": self.concurrency, "duration": duration}
        phase["grace_period"] = 30
        extra = {"temperature": 0, "max_completion_tokens": self.config["max_completion_tokens"], **self.config["request_extra"]}
        endpoint = {"url": self.url, "type": "chat", "streaming": True, "use_server_token_count": True, "timeout": 7200}
        endpoint["extra"] = extra
        dataset = {"name": "main", "type": "file", "sampling": "shuffle", "random_seed": 42, **self.config["dataset"]}
        tokenizer = {"name": self.config.get("tokenizer", self.config["model"])}
        benchmark = {"model": self.served_name(), "endpoint": endpoint, "tokenizer": tokenizer, "datasets": [dataset]}
        benchmark["phases"] = [phase]
        benchmark["server_metrics"] = {"enabled": True, "urls": [self.url + "/metrics"], "formats": ["json"]}
        benchmark["artifacts"] = {"dir": str(artifacts), "summary": ["json"], "records": ["jsonl"]}
        benchmark["runtime"] = {"ui": "none"}
        return {"schemaVersion": "2.0", "randomSeed": 42, "benchmark": benchmark}

    def served_name(self) -> str:
        return flag_value(self.config["serve_args"], "--served-model-name") or self.config["model"]

    def post(self, path: str, timeout: float = 600) -> None:
        urllib.request.urlopen(urllib.request.Request(self.url + path, method="POST"), timeout=timeout).read()

    def rpc(self, method: str) -> dict:
        body = json.dumps({"method": method, "args": [], "timeout": 120}).encode()
        headers = {"Content-Type": "application/json"}
        request = urllib.request.Request(self.url + "/collective_rpc", data=body, headers=headers)
        return json.loads(urllib.request.urlopen(request, timeout=150).read())

    def status(self, message: str) -> None:
        with (self.root / "status.txt").open("a") as stream:
            stream.write(f"{time.strftime('%H:%M:%S')} {message}\n")


def lockstep(gpus: int) -> list[str]:
    """Kernel replay serializes kernels, so a rank replaying alone would wait forever in a collective. With TP>1 the
    worker processes launch and replay every kernel in lockstep instead."""
    if gpus == 1:
        return []
    return ["--communicator", "shmem", "--communicator-shmem-num-peers", str(gpus), "--lockstep-kernel-launch"]


def with_flag(args: list, flag: str, value) -> list:
    """args with `flag=value`, replacing an existing `flag value` or `flag=value`."""
    args = [str(a) for a in args]
    if flag in args:
        index = args.index(flag)
        args = args[:index] + args[index + 2 :]
    return [a for a in args if not a.startswith(f"{flag}=")] + [f"{flag}={value}"]


def flag_value(args: list, flag: str) -> str | None:
    args = [str(a) for a in args]
    if flag in args:
        return args[args.index(flag) + 1]
    return next((a.split("=", 1)[1] for a in args if a.startswith(f"{flag}=")), None)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def run(command: list[str], log: Path) -> None:
    with log.open("w") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
