# vLLM Serving

Two serving milestones for the trained adapters:

1. Load the adapters into a pinned vLLM container and score a bounded
   validation subset (`scripts/vllm/<task>/evaluate.py`).
2. Run the FastAPI orchestrator + browser demo (`src/vllm/app.py`) as a second
   container in the same compose file.

All vLLM-related Python code lives in `src/vllm/`: `client.py` (sync stdlib
client for uv-env scripts), `aio.py` (async httpx client for the container),
`tasks.py` (task registry), `evaluation.py` (evaluation engine), and `app.py`
(FastAPI orchestrator). Shared metric functions live in `src/utils/metrics.py`
and are also used by the task evaluators.

The compose file lives at `.docker/docker-compose.yml`, not the repo root, so
every `docker compose` command below needs an explicit `-f` flag (or `cd
.docker` first). The project name is pinned to `master-thesis` in the file
itself, so container and volume names stay stable regardless of where you run
it from.

## Start vLLM

```bash
docker compose -f .docker/docker-compose.yml up vllm
```

The first start downloads
`unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit` into the named `hf-cache`
volume. This is the same pre-quantized base checkpoint recorded by the LoRA
adapters; `--dtype bfloat16` specifies their compute dtype. The current image is
`vllm/vllm-openai:v0.24.0`. The server registers these LoRA model names:

- `address` -> `adapters/address/qwen-0.5b`
- `extraction` -> `adapters/entity_name/extraction/qwen-0.5b`
- `classification` -> `adapters/entity_name/classification_external/qwen-0.5b`

Check what vLLM is serving:

```bash
curl http://localhost:8000/v1/models
```

## Score A Smoke Subset

The per-task evaluate CLIs use existing prompt templates, output parsers, and
the shared metrics from `src/utils/metrics.py`, computing the same task
metrics as the current evaluators without importing Unsloth.

```bash
uv run scripts/vllm/address/evaluate.py --limit 50
uv run scripts/vllm/extraction/evaluate.py --limit 50
uv run scripts/vllm/classification/evaluate.py --limit 50
```

By default, each CLI reads the validation split under `data/processed/...` and
uses the batch evaluator token budget (`1024`) so the metrics are comparable to
the existing evaluation scripts. To test production-style single calls, pass
`--max-tokens 64`. To score an entire split, pass `--limit 0`.

If processed data is not present, regenerate or restore it first. The default
paths are:

- `data/processed/address/validation`
- `data/processed/entity_name/extraction/validation`
- `data/processed/entity_name/classification_external/validation`

To save local raw completions for debugging:

```bash
uv run scripts/vllm/address/evaluate.py \
  --limit 50 \
  --predictions-output outputs/evaluation/vllm-address-smoke.json
```

Do not treat vLLM serving as trusted until the bounded smoke check and full
test-split metrics are close enough to the existing Unsloth reference metrics.

## Orchestrator + Demo UI

The orchestrator is a FastAPI app (`src/vllm/app.py`) running in its own
container built from `.docker/app/Dockerfile`. It is intentionally not part
of the uv project: the image installs only `fastapi`, `uvicorn`, `httpx`, and
`pydantic`, and reuses the project's prompt templates and output parsers from
the bind-mounted `src/` tree. It reaches vLLM at `http://vllm:8000/v1` over the
compose network (`VLLM_BASE_URL`).

```bash
# start vLLM + orchestrator (first run builds the app image)
docker compose -f .docker/docker-compose.yml up

# then open the demo page
# http://localhost:8080/
```

After editing code under `src/vllm/`, restart the app container — no
rebuild needed because `src/` is bind-mounted:

```bash
docker compose -f .docker/docker-compose.yml restart app
```

Rebuild only when `.docker/app/requirements.txt` changes:

```bash
docker compose -f .docker/docker-compose.yml build app
```

### Endpoints

| Endpoint | Body | Runs |
| --- | --- | --- |
| `POST /api/address` | `{"text": "..."}` | address adapter |
| `POST /api/entity-name` | `{"text": "..."}` | extraction adapter |
| `POST /api/entity-type` | `{"name": "..."}` | classification adapter |
| `POST /api/pipeline` | `{"text": "..."}` | address ∥ extraction, then classification on the parsed name |
| `GET /api/health` | – | proxies `/v1/models`, reports adapter readiness |

Each stage responds with `{ok, value, raw, latency_ms, error, skipped}`;
`/api/pipeline` nests one stage object per task plus `total_latency_ms`.
Endpoints return 503 when vLLM is unreachable (e.g. still loading).

```bash
curl -s -X POST http://localhost:8080/api/pipeline \
  -H 'Content-Type: application/json' \
  -d '{"text": "KAISER ALUM & CHEM CORP 1907 REYMET RD RICHMOND VA 23237"}'
```
