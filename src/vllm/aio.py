"""Async httpx client for the OpenAI-compatible vLLM endpoint.

Used by the serving container only; the uv environment uses the stdlib
client in ``src/vllm/client.py``. Both raise the same error hierarchy.
"""

import httpx

from src.vllm.client import VLLMConnectionError, VLLMError


async def chat_completion(
    client: httpx.AsyncClient,
    model: str,
    messages: list[dict],
    max_tokens: int,
) -> str:
    """Request one greedy chat completion from vLLM.

    Args:
        client: Shared async HTTP client configured with the vLLM base URL
        model: vLLM model name, which selects the LoRA adapter
        messages: OpenAI chat messages
        max_tokens: Maximum generated tokens

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
    try:
        response = await client.post('/chat/completions', json=payload)
        response.raise_for_status()
        body = response.json()
    except httpx.HTTPStatusError as e:
        raise VLLMError(
            f'vLLM request failed: HTTP {e.response.status_code}: {e.response.text}'
        ) from e
    except httpx.TransportError as e:
        raise VLLMConnectionError(f'vLLM endpoint unreachable: {e}') from e
    except (httpx.HTTPError, ValueError) as e:
        raise VLLMError(f'vLLM request failed: {e}') from e

    try:
        content = body['choices'][0]['message'].get('content') or ''
        return content.strip()
    except (KeyError, IndexError, AttributeError, TypeError) as e:
        raise VLLMError(f'Unexpected vLLM response shape: {body}') from e


async def list_models(client: httpx.AsyncClient) -> set[str]:
    """Fetch the served model IDs from the vLLM endpoint.

    Args:
        client: Shared async HTTP client configured with the vLLM base URL

    Returns:
        Set of model IDs reported by /models

    Raises:
        VLLMConnectionError: If the endpoint cannot be reached
        VLLMError: If the request fails or the response shape is unexpected
    """
    try:
        response = await client.get('/models')
        response.raise_for_status()
        body = response.json()
    except httpx.HTTPStatusError as e:
        raise VLLMError(
            f'vLLM /models failed: HTTP {e.response.status_code}: {e.response.text}'
        ) from e
    except httpx.TransportError as e:
        raise VLLMConnectionError(f'vLLM endpoint unreachable: {e}') from e
    except (httpx.HTTPError, ValueError) as e:
        raise VLLMError(f'vLLM /models failed: {e}') from e

    try:
        return {item['id'] for item in body['data']}
    except (KeyError, TypeError) as e:
        raise VLLMError(f'Unexpected /models response shape: {body}') from e
