# infoseek/serving — fleet vLLM manager

Owner: serving helper. Python: `python` (it has vllm 0.30, aiohttp and httpx). The pathcal venv also has aiohttp.

## What runs
- **Watchdog**: `fleet.py watchdog` runs detached on <host> (CPU only). Its PID is in `watchdog.pid` and its log is `logs/watchdog.log`. Every 45 s it does the following:
  1. **Adopt** the lead's servers from `naacl_dcfr/logs/servers_launched.tsv`. The watchdog health-checks them, never kills them, and never relaunches duplicates on the same GPU. It also re-adopts its own running vLLM servers on ports 8800–8899 after a watchdog restart.
  2. **Scan** every allowed host over ssh, as in `naacl_dcfr/scripts/gpu_check.sh`:
     - free memory per GPU;
     - <other>, p4_workshop and <other> processes (that GPU is blocked);
     - a <other> docker shell on the host (the whole host is skipped; this is currently <host>);
     - processes owned by other users (a +10 GB margin instead of 6 GB).
     The forbidden list combines `~/ts_fleet/FORBIDDEN_HOSTS` with the hard-coded list 08, 10, 13, 17–26, 28, 29, 30. <host> (<other>) is always avoided.
  3. **Check health** with `GET /v1/models`. Each new server gets a smoke test: one `/v1/completions` call (1 token, `logprobs=20`) plus one `/v1/chat/completions` call with logprobs. A server enters `endpoints.json` only after the completions smoke test passes.
  4. **Restart** dead servers. A process that is gone, or 6 failed health checks in a row (about 4.5 min), marks the server dead, and the fill step relaunches a replacement wherever memory fits. If <other> or <other> appears on one of our GPUs, or a host becomes forbidden, our server there is stopped immediately.
  5. **Fill**: launches roster models into free GPU memory.
     - Order: first one replica of every model, then the second replica of every model, and so on, up to each model's `target` in `roster.json`.
     - GPU choice: best fit, meaning the smallest slot that holds the model, so large slots stay free for 14B and TP2 models.
     - Each launch runs on the target host inside the shared per-GPU flock (`~/ts_fleet/locks/{host}_g{g}.lock`). It reads free memory twice, 10 s apart, and uses the smaller value. Budget = min(pref, free − 6 GB, or free − 10 GB with a foreign process), and the launch is refused below `min_mib`. The lock is held until the new server's memory shows up, so concurrent launchers see real free memory.
     - At most one launch is in flight per host.
- **Servers**: `serve_one.sh` runs vLLM 0.30 as an OpenAI-compatible server.
  - Each server is registered under its HF id **and** a short name (for example `qwen3-14b`).
  - Settings: `--max-logprobs 20`, prefix caching on, `--max-num-batched-tokens 8192`, `--max-model-len 8192` (12288 for Qwen3-32B-AWQ and gpt-oss-20b).
  - Our ports are 8800–8899. The lead's ports are 87xx.
  - Logs: `logs/<short>_<host>_<port>.log`.
  - Servers keep running when the watchdog stops.

## Interfaces
### `endpoints.json` (live registry; rewritten every tick)
```json
{"Qwen/Qwen3-14B": ["http://<host>:8707", "http://<host>:8802"], ...}
```
- Keys are HF model ids, which are also the `model` field to send.
- Values are base URLs with no `/v1`, the same format as `dcfr_lib.endpoints()`.
- Only healthy, smoke-passed servers are listed.
- Lead servers and our servers are mixed. All replicas of one model use the same checkpoint and quantization. The one exception is Qwen/Qwen3-8B, which is served with `--quantization fp8` on all replicas to match the lead's q8.

To use it from `dcfr_lib`, replace `endpoints()` with:
```python
json.load(open("./serving/endpoints.json"))
```

### `state.json`
The full table. Per server it records: host, gpus, port, pid, alloc_mib, status (`starting|up|unhealthy|dead|failed|released`), the smoke result, `owner` (`serving|lead`) and `why`. It also holds the recent launch failures.

### `client.py` (async, load balanced, cached)
```python
import sys; sys.path.insert(0, "./serving")
from client import FleetClient, first_token_top_logprobs, option_probs, run_complete
async with FleetClient() as fc:                       # cache_dir defaults to serving/cache (or $FLEET_CACHE_DIR)
    rs = await fc.complete_batch("Qwen/Qwen3-14B", prompts, max_tokens=1, logprobs=20, temperature=0)
    ps = [option_probs(r, ("A","B","C","D","E","F","S")) for r in rs]
    cs = await fc.chat_batch("google/gemma-2-9b-it", convs, max_tokens=256, temperature=0)
```
- **Balancing**: requests go to the replica with the fewest outstanding requests (at most 48 per replica). Replicas added or dropped by the watchdog are picked up within 30 s.
- **Retries**: 6 attempts with exponential backoff on connection errors, timeouts, 5xx and 429. A failing replica is cooled down for 30 s and the next attempt goes to another replica. HTTP 400 is returned as `{"error": ...}` and not retried.
- **Cache**: `cache/<model>/<host>_<pid>_<t>.jsonl.gz`, keyed by the sha256 of the canonical request (endpoint, model, prompt or messages, all params, and an optional `cache_tag`).
  - Shards are per process, with one gzip member per flush, and they tolerate a truncated tail after a crash.
  - An identical request is served from the cache, so for fresh samples at temperature > 0 vary `seed` or `cache_tag`.
  - `use_cache=False` turns caching off.
- Each result is the OpenAI response dict plus `_url` (the replica that served it) and `_cached`.
- Self-test: `python client.py [--model M] [--n 8]` sends one completion (top-20 logprobs) and one chat per model and prints which replicas were used.

### Lead controls (the watchdog applies them within one tick)
```
python fleet.py status                       # table + models not yet served
python fleet.py scan                         # free memory / flags per GPU
python fleet.py release --model qwen2.5-7b   # stop OUR replicas of a model (target is lowered so they are not refilled)
python fleet.py release --url http://host.onprem:8801 [--reserve]   # --reserve keeps that GPU out of future fills
python fleet.py release --idle-min 30        # stop our servers with no finished request in 30 min
python fleet.py target --model Qwen/Qwen3-14B --n 4   # change the replica target
python fleet.py pause | resume               # stop / restart filling (health checks and restarts continue)
python fleet.py unreserve <host>[:1]
python fleet.py stop-watchdog | start-watchdog   # stopping is graceful: it waits for in-flight launches
```
The watchdog frees our servers **only** on these requests. It never kills lead servers or other users' processes.

## Roster (`roster.json`)
**Memory budgets (`min_mib` / `pref_mib`):**
- 14B bf16 models need ≥ 35 GB. At 32 GB they failed with too little KV cache.
- 7B bf16 models need ≥ 22 GB.
- Qwen3-8B fp8 needs ≥ 15.5 GB.
- Llama-3.1-8B-AWQ needs ≥ 9.5 GB.
- gemma-2-27b (51 GB bf16), Llama-3.3-70B-AWQ and Qwen2.5-72B-AWQ need TP2 with ≥ 26–31 GB on **both** GPUs of one host. They start only when such a pair is free; right now no allowed host has one.

**EXAONE-Finance (LLM) is not available.** Its HF cache directories hold only README/refs and no weights. The `/nfs-fin` exaone-finance-api on <host> is the EXAONE *Forecast* time-series model, not an LLM. It needs a download decision from the lead.

## Known behaviour
- Qwen/Qwen3-32B-AWQ (lead's q32) returned HTTP 400 "Out of range float values are not JSON compliant" on a **multi-token** completion with `logprobs=5` (a -inf logprob). 1-token calls with `logprobs=20` work. Keep logprob calls at `max_tokens=1`, or handle `{"error"}`.
- Startup takes about 2–6 min (load from NFS + torch.compile + CUDA graphs).

## Progress log
- 2026-10-09 00:00 KST: Read DESIGN.md, SERVER.md, the forecastbrief launch scripts and the lead's serve scripts. Checked the HF cache for the roster: all present except EXAONE-Finance. Fleet scan: <host> has a <other> docker shell, so it is skipped.
- 00:04: Watchdog started. It adopted 8 lead servers; 7 passed the smoke test at once, and q32 passed after the smoke test was changed to 1-token logprobs.
- 00:06–00:10: First fill wave (13 launches).
  - Failures from a too-small KV budget: phi-4 at 32 GB, R1-14B at 32.9 GB, Qwen2.5-14B at 32 GB, and Qwen2.5-7B / R1-7B at 24 GB with 16384 batched tokens.
  - Fix: raised the minimum budgets and set `--max-num-batched-tokens` to 8192.
  - Also fixed two bugs: a port collision between two servers that were both still loading, and launches orphaned when the watchdog restarted. The watchdog now re-adopts orphans and stops gracefully.
- 00:14: Client tests passed.
  - All 12 served models answered 16 parallel completions with top-20 logprobs, spread over their replicas.
  - Cache: the second run was all hits (5/5).
  - Failover: a registry with one dead URL still completed 30/30 requests, with 10 retries rerouted to live replicas.
- 00:16: Fixed Mistral-7B chat. Its template rejects `chat_template_kwargs`, so the smoke test now passes that only to Qwen3. option_probs() now strips the SentencePiece '▁' prefix.
- 00:17: Allowed <host> GPU by GPU. Our server stops automatically if <other> appears on its GPU.
  - Qwen2.5-14B is up on 11 g0.
  - gemma-3-12b failed at a 32 GB budget, so its minimum is now 33 GB.
  - gemma-2-9b failed at 22 GB, so its minimum is now 25 GB.
- 00:20: Smoke-verified every endpoint: 28/29 PASS, covering 13 models.
  - The 1 FAIL is the lead's q32 (<host>:8701). Its `/v1/models` returns 200, but it was saturated: 1109 waiting requests, 1.36k prompt tok/s, so the smoke request timed out at 180 s.
  - Two more Qwen3-32B-AWQ replicas (05:8800, 34:8802) are idle. **Point the q32 workload at endpoints.json.**
- 00:27: The watchdog auto-restarted Llama-8B-AWQ on <host> after its process died. The restart was not diagnosed.

### Endpoint table (snapshot 2026-10-09 00:28 KST; the live version is `python fleet.py table` / endpoints.json)
| model | replicas | base URLs (host:port, GPU, owner, quant) |
|---|---|---|
| Qwen/Qwen3-8B | 4 | <host>:8801 g0 serving fp8; <host>:8800 g0 serving fp8; <host>:8708 g? lead fp8; <host>:8800 g0 serving fp8 |
| Qwen/Qwen3-14B | 1 | <host>:8707 g? lead |
| Qwen/Qwen3-32B-AWQ | 3 | <host>:8701 g? lead; <host>:8800 g1 serving; <host>:8802 g1 serving |
| Qwen/Qwen2.5-7B-Instruct | 3 | <host>:8801 g0 serving; <host>:8800 g0 serving; <host>:8801 g1 serving |
| Qwen/Qwen2.5-14B-Instruct | 1 | <host>:8800 g0 serving |
| hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 | 4 | <host>:8801 g0 serving; <host>:8800 g1 serving; <host>:8709 g? lead; <host>:8800 g0 serving |
| google/gemma-2-9b-it | 2 | <host>:8800 g1 serving; <host>:8710 g? lead |
| google/gemma-3-12b-it | 1 | <host>:8705 g? lead |
| mistralai/Mistral-7B-Instruct-v0.3 | 3 | <host>:8801 g0 serving; <host>:8800 g0 serving; <host>:8800 g0 serving |
| casperhansen/mistral-small-24b-instruct-2501-awq | 3 | <host>:8704 g? lead; <host>:8800 g0 serving; <host>:8801 g1 serving |
| openai/gpt-oss-20b | 3 | <host>:8703 g? lead; <host>:8801 g0 serving; <host>:8801 g1 serving |
| microsoft/phi-4 | 0 | not served yet |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 2 | <host>:8800 g0 serving; <host>:8801 g0 serving |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-14B | 1 | <host>:8801 g1 serving |
| google/gemma-2-27b-it | 0 | not served yet |
| casperhansen/llama-3.3-70b-instruct-awq | 0 | not served yet |
| Qwen/Qwen2.5-72B-Instruct-AWQ | 0 | not served yet |

Not yet placed (the watchdog places them as soon as memory frees; there is currently no free 35 GB slot or 2×26 GB pair):
- phi-4 (bf16, needs 35 GB)
- gemma-2-27b (TP2)
- Llama-3.3-70B-AWQ (TP2)
- Qwen2.5-72B-AWQ (TP2)

### Notes for users of the fleet
- Different replicas of the same model can return slightly different logprobs for the same request (seen: -0.89 vs -0.85), because batching differs. Use the cache, or one replica per comparison, when exact reproducibility matters.
- R1-distill completions on raw text start with `\` or `<think>`. Render prompts with `dcfr_lib.render`, as the lead does.
