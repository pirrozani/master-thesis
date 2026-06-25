from pathlib import Path

from huggingface_hub import HfApi, create_repo

from src.address.config import MODEL_REGISTRY as ADDRESS_REGISTRY
from src.address.config import TASK as ADDRESS_TASK
from src.config import get_model_config
from src.entity_name.classification.config import (
    MODEL_REGISTRY as CLASSIFICATION_REGISTRY,
)
from src.entity_name.classification.config import TASK as CLASSIFICATION_TASK
from src.entity_name.extraction.config import MODEL_REGISTRY as EXTRACTION_REGISTRY
from src.entity_name.extraction.config import TASK as EXTRACTION_TASK

# Default quantization method
QUANT_METHOD = 'q4_k_m'

# Quantization methods accepted by Unsloth's GGUF export (llama.cpp)
ALLOWED_QUANTS = frozenset(
    {
        'not_quantized',
        'fast_quantized',
        'quantized',
        'f32',
        'f16',
        'q8_0',
        'q4_k_m',
        'q5_k_m',
        'q2_k',
        'q3_k_l',
        'q3_k_m',
        'q3_k_s',
        'q4_0',
        'q4_1',
        'q4_k_s',
        'q4_k',
        'q5_k',
        'q5_0',
        'q5_1',
        'q5_k_s',
        'q6_k',
        'iq2_xxs',
        'iq2_xs',
        'iq3_xxs',
        'q3_k_xs',
    }
)

# Map each task to its per-task model registry.
TASK_REGISTRIES = {
    ADDRESS_TASK: ADDRESS_REGISTRY,
    EXTRACTION_TASK: EXTRACTION_REGISTRY,
    CLASSIFICATION_TASK: CLASSIFICATION_REGISTRY,
}

# Human-readable task label used in the GGUF filename.
TASK_GGUF_LABELS = {
    ADDRESS_TASK: 'address_extraction',
    EXTRACTION_TASK: 'entity_name_extraction',
    CLASSIFICATION_TASK: 'entity_name_classification',
}

# Suffixes stripped from a base model id when building the filename model token.
_MODEL_TOKEN_SUFFIXES = (
    '-unsloth-bnb-4bit',
    '-bnb-4bit',
    '-instruct',
    '-it',
)


def parse_adapter_path(adapter_path: str | Path) -> tuple[str, str]:
    """Recover the ``(task, model_alias)`` pair from an adapter directory path.

    Args:
        adapter_path: Path to a trained adapter, e.g. ``adapters/address/qwen-0.5b``
            or ``adapters/entity_name/classification/qwen-3b``.

    Returns:
        A ``(task, model_alias)`` tuple, where ``task`` is the registry task key
        (e.g. ``'entity_name/classification'``).

    Raises:
        ValueError: If the path does not follow ``adapters/<task>/<model>`` or the
            task/alias are not registered.
    """
    parts = Path(adapter_path).parts
    if 'adapters' not in parts:
        raise ValueError(
            f"Adapter path must live under 'adapters/<task>/<model>': {adapter_path}"
        )

    tail = parts[parts.index('adapters') + 1 :]
    if len(tail) < 2:
        raise ValueError(
            f"Adapter path must be 'adapters/<task>/<model>': {adapter_path}"
        )

    model_alias = tail[-1]
    task = '/'.join(tail[:-1])

    if task not in TASK_REGISTRIES:
        available = ', '.join(TASK_REGISTRIES)
        raise ValueError(
            f"Unknown task '{task}' from adapter path. Available tasks: {available}"
        )
    if model_alias not in TASK_REGISTRIES[task]:
        available = ', '.join(TASK_REGISTRIES[task])
        raise ValueError(
            f"Unknown model alias '{model_alias}' for task '{task}'. "
            f'Available models: {available}'
        )

    return task, model_alias


def clean_model_token(base_model: str) -> str:
    """Build the filename model token from a base model id.

    Example: ``'Qwen/Qwen2.5-0.5B-Instruct'`` -> ``'qwen2.5-0.5b'``.

    Args:
        base_model: Hugging Face base model id from the registry.

    Returns:
        A lowercase, suffix-stripped token suitable for a filename.
    """
    token = base_model.split('/')[-1].lower()

    stripped = True
    while stripped:
        stripped = False
        for suffix in _MODEL_TOKEN_SUFFIXES:
            if token.endswith(suffix):
                token = token[: -len(suffix)]
                stripped = True
                break

    return token


def resolve_base_model(task: str, model_alias: str) -> str:
    """Look up the registered base model id for a ``(task, alias)`` pair.

    Args:
        task: Registry task key (e.g. ``'address'``).
        model_alias: Model alias (e.g. ``'qwen-0.5b'``).

    Returns:
        The base model id.
    """
    config = get_model_config(model_alias, task=task, registry=TASK_REGISTRIES[task])
    return config.base_model


def validate_quants(quants: list[str]) -> None:
    """Raise if any requested quantization method is not supported.

    Args:
        quants: Requested quantization methods.

    Raises:
        ValueError: If one or more methods are not in ``ALLOWED_QUANTS``.
    """
    invalid = [q for q in quants if q not in ALLOWED_QUANTS]
    if invalid:
        allowed = ', '.join(sorted(ALLOWED_QUANTS))
        raise ValueError(
            f'Unsupported quantization method(s): {", ".join(invalid)}. '
            f'Allowed: {allowed}'
        )


def build_repo_stem(base_model: str, task: str) -> str:
    """Build the quant-agnostic stem ``{model_token}_{task_label}``.

    Used as the default repo name and local working-directory name; every quant
    variant for a model+task shares this stem.

    Args:
        base_model: Base model id from the registry.
        task: Registry task key.

    Returns:
        e.g. ``'qwen2.5-0.5b_address_extraction'``.
    """
    return f'{clean_model_token(base_model)}_{TASK_GGUF_LABELS[task]}'


def build_gguf_filename(base_model: str, task: str, quant: str = QUANT_METHOD) -> str:
    """Construct the GGUF filename ``{model_token}_{task_label}.{quant}.gguf``.

    The quant is part of the name, so multiple quantizations can live in the same
    repo without overwriting each other.

    Args:
        base_model: Base model id from the registry.
        task: Registry task key.
        quant: Quantization method.

    Returns:
        e.g. ``'qwen2.5-0.5b_address_extraction.q4_k_m.gguf'``.
    """
    return f'{build_repo_stem(base_model, task)}.{quant}.gguf'


def convert_adapter_to_gguf(
    adapter_path: str | Path,
    work_dir: str | Path,
    quant_filenames: dict[str, str],
) -> dict[str, Path]:
    """Merge a LoRA adapter and export it to one GGUF file per quant.

    Loads the adapter once with Unsloth (which clones and builds llama.cpp on
    first run), quantizes to every method in ``quant_filenames``, and renames
    each produced file to its desired filename.

    Args:
        adapter_path: Path to the trained adapter directory.
        work_dir: Directory to hold the merged model and GGUFs (ephemeral).
        quant_filenames: Mapping of quantization method -> desired filename.

    Returns:
        Mapping of quantization method -> path to the produced GGUF file.

    Raises:
        RuntimeError: If a requested quant produces no GGUF file.
    """
    from unsloth import FastLanguageModel  # heavy, GPU-only import

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    quants = list(quant_filenames)
    model.save_pretrained_gguf(str(work_dir), tokenizer, quantization_method=quants)

    gguf_dir = work_dir
    unsloth_dir = work_dir.parent / (work_dir.name + '_gguf')
    if unsloth_dir.is_dir():
        gguf_dir = unsloth_dir

    produced = sorted(gguf_dir.glob('*.gguf'))
    results: dict[str, Path] = {}
    used: set[Path] = set()
    for quant, filename in quant_filenames.items():
        tag = quant.lower()
        candidates = [p for p in produced if p not in used and tag in p.name.lower()]
        if not candidates:
            raise RuntimeError(f'No GGUF produced for quant {quant!r} in {gguf_dir}')

        source = candidates[0]
        used.add(source)

        target = gguf_dir / filename
        if source != target:
            source.replace(target)
        results[quant] = target

    return results


def get_hf_username(token: str) -> str:
    """Return the Hugging Face username for ``token`` (validates the credential).

    Args:
        token: Hugging Face access token.

    Returns:
        The authenticated account's username.

    Raises:
        RuntimeError: If authentication fails.
    """
    try:
        info = HfApi().whoami(token=token)
    except Exception as exc:
        raise RuntimeError(f'Hugging Face authentication failed: {exc}') from exc
    return info['name']


def ensure_private_repo(repo_id: str, token: str) -> None:
    """Create the target repo as private if it does not already exist.

    Args:
        repo_id: Target Hub repo id (``owner/name``).
        token: Hugging Face access token with write scope.
    """
    create_repo(repo_id, repo_type='model', private=True, exist_ok=True, token=token)


def upload_gguf(
    gguf_path: str | Path,
    repo_id: str,
    filename: str,
    token: str,
    commit_message: str,
) -> str:
    """Upload a GGUF file to ``repo_id`` and return the repo URL.

    Args:
        gguf_path: Local path to the GGUF file.
        repo_id: Target Hub repo id (``owner/name``).
        filename: Destination filename within the repo.
        token: Hugging Face access token with write scope.
        commit_message: Commit message for the upload.

    Returns:
        The Hub URL of the repository.
    """
    HfApi().upload_file(
        path_or_fileobj=str(gguf_path),
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type='model',
        token=token,
        commit_message=commit_message,
    )
    return f'https://huggingface.co/{repo_id}'
