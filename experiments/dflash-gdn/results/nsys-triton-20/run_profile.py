import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('benchmark_runner', ROOT / 'benchmarks/dflash_4b/run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
base = 'http://127.0.0.1:8110'

def request(path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=300) as response:
        data = response.read()
        try:
            return json.loads(data) if data else {}
        except json.JSONDecodeError:
            return {'message': data.decode()}

def main():
    target = runner.download_model(runner.TARGET)
    draft = runner.download_model(runner.DRAFT)
    server = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--model', target,
              '--served-model-name', 'qwen', '--quantization', 'fp8', '--language-model-only',
              '--max-model-len', '4096', '--max-num-seqs', '8', '--gpu-memory-utilization', '0.8',
              '--speculative-config', json.dumps({'method':'dflash','model':draft,'num_speculative_tokens':15}),
              '--profiler-config', json.dumps({'profiler':'cuda'}), '--disable-uvicorn-access-log',
              '--host','127.0.0.1','--port','8110']
    command = ['nsys','profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none',
               '--cuda-graph-trace=graph','--capture-range=cudaProfilerApi','--capture-range-end=stop-shutdown',
               '--kill=none','--output',str(OUT/'20_tokens'), *server]
    env = os.environ.copy()
    env.update(VLLM_USE_V2_MODEL_RUNNER='1', VLLM_SERVER_DEV_MODE='1')
    results = json.loads((ROOT/'issue_49730/results.json').read_text())
    inputs = json.loads(results['workload']['inputs_json'])
    body = {'model':'qwen','messages':inputs['data'][0]['payloads'][0]['messages'],'max_tokens':20,'temperature':0}
    runner.write_json(OUT/'run_config.json', {'command':command,'request':body,'versions':runner.versions('vllm'),
                                           'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                                           'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES']})
    with (OUT/'server.log').open('w') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=log, env=env, start_new_session=True)
        try:
            runner.wait_ready(proc,base)
            for phase in ['warmup','capture']:
                if phase=='capture':
                    request('/start_profile',{})
                start=time.perf_counter()
                result=request('/v1/chat/completions',body)
                elapsed=(time.perf_counter()-start)*1000
                if phase=='capture':
                    request('/stop_profile',{})
                runner.write_json(OUT/f'{phase}_response.json',{'elapsed_ms':elapsed,'response':result})
                print(json.dumps({'phase':phase,'elapsed_ms':elapsed,'usage':result.get('usage')}),flush=True)
            proc.wait(timeout=120)
        finally:
            runner.stop(proc)
    subprocess.run(['nsys','export','--type=sqlite','--output',str(OUT/'20_tokens.sqlite'),str(OUT/'20_tokens.nsys-rep')],check=True)

if __name__=='__main__':
    main()
