"""Synchronous stdlib-only client for the OpenAI-compatible vLLM endpoint.

Importable from the uv environment (no httpx dependency); the async
counterpart for the serving container lives in ``src/vllm/aio.py``.
"""

import json
import os
import urllib.error
import urllib.request

from dotenv import load_dotenv

load_dotenv()

REQUEST_TIMEOUT = 60.0


class VLLMError(Exception):
    """Raised when a vLLM request fails or returns an unexpected shape."""


class VLLMConnectionError(VLLMError):
    """Raised when the vLLM endpoint cannot be reached."""


def get_base_url() -> str:
    """Return the vLLM base URL from the environment.

    Reads ``VLLM_BASE_URL``, set directly, via ``.env`` locally (see
    ``.env.example``), or via ``docker-compose.yml`` in the container.

    Returns:
        OpenAI-compatible base URL, without a trailing slash

    Raises:
        KeyError: If VLLM_BASE_URL is not set in the environment or .env
    """
    return os.environ['VLLM_BASE_URL'].rstrip('/')


def post_chat_completion(
    base_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    timeout: float = REQUEST_TIMEOUT,
) -> str:
    """Request one greedy chat completion from vLLM.

    Args:
        base_url: OpenAI-compatible base URL, usually http://localhost:8000/v1
        model: vLLM model name, which selects the LoRA adapter
        messages: OpenAI chat messages
        max_tokens: Maximum generated tokens
        timeout: Request timeout in seconds

    Returns:
        Assistant message content, stripped

    Raises:
        VLLMConnectionError: If the endpoint cannot be reached
        VLLMError: If the request fails or the response shape is unexpected
    """
    payload = {
        'model': model,
        'messages': messages,
        'temperature': 0,
        'max_tokens': max_tokens,
    }
    request = urllib.request.Request(
        url=f'{base_url.rstrip("/")}/chat/completions',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        detail = e.read().decode('utf-8', errors='replace')
        raise VLLMError(f'vLLM request failed: HTTP {e.code}: {detail}') from e
    except urllib.error.URLError as e:
        raise VLLMConnectionError(f'vLLM endpoint unreachable: {e}') from e

    try:
        content = body['choices'][0]['message'].get('content') or ''
        return content.strip()
    except (KeyError, IndexError, AttributeError, TypeError) as e:
        raise VLLMError(f'Unexpected vLLM response shape: {body}') from e


def list_models(base_url: str, timeout: float = REQUEST_TIMEOUT) -> set[str]:
    """Fetch the served model IDs from the vLLM endpoint.

    Args:
        base_url: OpenAI-compatible base URL
        timeout: Request timeout in seconds

    Returns:
        Set of model IDs reported by /models

    Raises:
        VLLMConnectionError: If the endpoint cannot be reached
        VLLMError: If the request fails or the response shape is unexpected
    """
    request = urllib.request.Request(url=f'{base_url.rstrip("/")}/models')

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        detail = e.read().decode('utf-8', errors='replace')
        raise VLLMError(f'vLLM /models failed: HTTP {e.code}: {detail}') from e
    except urllib.error.URLError as e:
        raise VLLMConnectionError(f'vLLM endpoint unreachable: {e}') from e

    try:
        return {item['id'] for item in body['data']}
    except (KeyError, TypeError) as e:
        raise VLLMError(f'Unexpected /models response shape: {body}') from e
