"""FastAPI orchestrator exposing the extraction pipeline over vLLM."""

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.vllm import aio
from src.vllm.client import (
    REQUEST_TIMEOUT,
    VLLMConnectionError,
    VLLMError,
    get_base_url,
)
from src.vllm.tasks import ADDRESS, ALL_TASKS, CLASSIFICATION, EXTRACTION, TaskSpec

MAX_INPUT_CHARS = 512
UI_DIR = Path(__file__).resolve().parents[2] / 'ui'
STATIC_DIR = UI_DIR / 'static'
TEMPLATES_DIR = UI_DIR / 'templates'


class TextRequest(BaseModel):
    """Request body for stages that take a raw text line."""

    text: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)


class NameRequest(BaseModel):
    """Request body for classifying an already-extracted entity name."""

    name: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)


class StageResult(BaseModel):
    """Outcome of one pipeline stage."""

    ok: bool
    value: Any = None
    raw: str | None = None
    latency_ms: int | None = None
    error: str | None = None
    skipped: bool = False


class PipelineResult(BaseModel):
    """Combined outcome of the full extraction pipeline."""

    address: StageResult
    entity_name: StageResult
    entity_type: StageResult
    total_latency_ms: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create and dispose of the shared vLLM HTTP client."""
    app.state.client = httpx.AsyncClient(
        base_url=get_base_url(), timeout=REQUEST_TIMEOUT
    )
    yield
    await app.state.client.aclose()


app = FastAPI(title='Structured Extraction Pipeline', lifespan=lifespan)


async def run_stage(spec: TaskSpec, text: str) -> StageResult:
    """Run one adapter stage and capture value, raw output, and latency.

    Args:
        spec: Task spec selecting the adapter, prompt, and parser
        text: Stage input text

    Returns:
        StageResult with the parsed value on success or the error message
        on a failed vLLM call

    Raises:
        VLLMConnectionError: If the vLLM endpoint cannot be reached
    """
    start = time.perf_counter()
    try:
        raw = await aio.chat_completion(
            app.state.client,
            spec.vllm_model,
            spec.create_messages(text),
            spec.single_max_tokens,
        )
    except VLLMConnectionError:
        raise
    except VLLMError as e:
        return StageResult(
            ok=False,
            error=str(e),
            latency_ms=int((time.perf_counter() - start) * 1000),
        )
    return StageResult(
        ok=True,
        value=spec.parse(raw),
        raw=raw,
        latency_ms=int((time.perf_counter() - start) * 1000),
    )


async def run_single(spec: TaskSpec, text: str) -> StageResult:
    """Run one stage for a single-stage endpoint.

    Args:
        spec: Task spec selecting the adapter, prompt, and parser
        text: Stage input text

    Returns:
        StageResult for the stage

    Raises:
        HTTPException: 503 when the vLLM endpoint is unreachable
    """
    try:
        return await run_stage(spec, text)
    except VLLMConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.post('/api/address')
async def extract_address(request: TextRequest) -> StageResult:
    """Extract structured address fields from a raw text line."""
    return await run_single(ADDRESS, request.text)


@app.post('/api/entity-name')
async def extract_entity_name(request: TextRequest) -> StageResult:
    """Extract and clean the entity name from a raw text line."""
    return await run_single(EXTRACTION, request.text)


@app.post('/api/entity-type')
async def classify_entity_type(request: NameRequest) -> StageResult:
    """Classify an entity name as company or person."""
    return await run_single(CLASSIFICATION, request.name)


@app.post('/api/pipeline')
async def run_pipeline(request: TextRequest) -> PipelineResult:
    """Run the full pipeline: address and name in parallel, then type.

    Address extraction and entity-name extraction run concurrently on the
    raw text; classification runs afterwards on the parsed entity name and
    is skipped when no name was extracted.
    """
    start = time.perf_counter()
    try:
        address_result, name_result = await asyncio.gather(
            run_stage(ADDRESS, request.text),
            run_stage(EXTRACTION, request.text),
        )
    except VLLMConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    name_value = name_result.value if name_result.ok else ''
    if name_value:
        try:
            type_result = await run_stage(CLASSIFICATION, name_value)
        except VLLMConnectionError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
    else:
        reason = (
            'skipped: extraction returned an empty name'
            if name_result.ok
            else 'skipped: entity name stage failed'
        )
        type_result = StageResult(ok=False, skipped=True, error=reason)

    return PipelineResult(
        address=address_result,
        entity_name=name_result,
        entity_type=type_result,
        total_latency_ms=int((time.perf_counter() - start) * 1000),
    )


@app.get('/api/health')
async def health() -> dict:
    """Report vLLM reachability and which adapters are served."""
    expected = sorted(spec.vllm_model for spec in ALL_TASKS)
    try:
        served = await aio.list_models(app.state.client)
    except VLLMError as e:
        return {
            'vllm_up': False,
            'served_models': [],
            'expected_adapters': expected,
            'adapters_ready': False,
            'error': str(e),
        }
    return {
        'vllm_up': True,
        'served_models': sorted(served),
        'expected_adapters': expected,
        'adapters_ready': all(model in served for model in expected),
        'error': None,
    }


@app.get('/')
async def index() -> FileResponse:
    """Serve the demo page."""
    return FileResponse(TEMPLATES_DIR / 'index.html')


app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')
