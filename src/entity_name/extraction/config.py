"""Model registry for the entity name extraction task."""

from src.config import ModelConfig

TASK = 'entity_name/extraction'
DEFAULT_MODEL = 'qwen-0.5b'

# Model registry mapping aliases to their configurations
MODEL_REGISTRY: dict[str, ModelConfig] = {
    # Qwen 2.5 variants
    'qwen-0.5b': ModelConfig(
        name='qwen-0.5b',
        base_model='Qwen/Qwen2.5-0.5B-Instruct',
        max_seq_length=128,
    ),
    'qwen-1.5b': ModelConfig(
        name='qwen-1.5b',
        base_model='Qwen/Qwen2.5-1.5B-Instruct',
        max_seq_length=128,
    ),
    'qwen-3b': ModelConfig(
        name='qwen-3b',
        base_model='Qwen/Qwen2.5-3B-Instruct',
        max_seq_length=128,
    ),
    # Qwen 3.5 variants
    'qwen3.5-4b': ModelConfig(
        name='qwen3.5-4b',
        base_model='Qwen/Qwen3.5-4B',
        max_seq_length=128,
    ),
    # Gemma variants
    'gemma-270m': ModelConfig(
        name='gemma-270m',
        base_model='unsloth/gemma-3-270m-it-unsloth-bnb-4bit',
        max_seq_length=128,
    ),
}
