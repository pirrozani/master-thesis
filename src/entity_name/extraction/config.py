"""Model registry for the entity name extraction task."""

from src.config import ModelConfig

TASK = 'entity_name/extraction'
DEFAULT_MODEL = 'qwen-0.5b'

# Generated-output length budgets (max_new_tokens) for inference
SINGLE_MAX_NEW_TOKENS = 64  # single-example inference
BATCH_MAX_NEW_TOKENS = 1024  # batch inference

# Model registry mapping aliases to their configurations
MODEL_REGISTRY: dict[str, ModelConfig] = {
    # Qwen 2.5 variants
    'qwen-0.5b': ModelConfig(
        name='qwen-0.5b',
        base_model='Qwen/Qwen2.5-0.5B-Instruct',
    ),
    'qwen-1.5b': ModelConfig(
        name='qwen-1.5b',
        base_model='Qwen/Qwen2.5-1.5B-Instruct',
    ),
    'qwen-3b': ModelConfig(
        name='qwen-3b',
        base_model='Qwen/Qwen2.5-3B-Instruct',
    ),
}
