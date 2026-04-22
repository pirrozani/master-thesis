"""Centralized model configuration."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for a specific model variant."""

    name: str
    base_model: str
    max_seq_length: int = 512
    lora_r: int = 16
    lora_alpha: int = 16
    adapter_dir: str = ''

    def __post_init__(self):
        """Set default adapter directory based on model name."""
        if not self.adapter_dir:
            self.adapter_dir = f'adapters/{self.name}'


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

# Default model alias
DEFAULT_MODEL = 'qwen-0.5b'


def get_model_config(model_alias: str, task: str = 'address') -> ModelConfig:
    """Get model configuration by alias with task-specific adapter directory.

    Args:
        model_alias: Short model alias (e.g., 'qwen-0.5b', 'gemma-270m')
        task: Task name for adapter directory scoping

    Returns:
        ModelConfig instance with task-specific adapter_dir

    Raises:
        ValueError: If model alias is not found in the registry
    """
    if model_alias not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Model alias '{model_alias}' not found in registry. "
            f'Available models: {available}'
        )

    config = MODEL_REGISTRY[model_alias]

    # Return a copy with task-specific adapter directory
    return ModelConfig(
        name=config.name,
        base_model=config.base_model,
        max_seq_length=config.max_seq_length,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        adapter_dir=f'adapters/{task}/{config.name}',
    )


def list_available_models() -> list[str]:
    """List all Available models.

    Returns:
        List of registered model aliases
    """
    return list(MODEL_REGISTRY.keys())


def get_model_choices() -> str:
    """Get a formatted string of model choices for CLI help.

    Returns:
        Formatted string listing all available models
    """
    lines = ['Available models:']
    for alias, config in MODEL_REGISTRY.items():
        lines.append(f'  {alias}: {config.base_model}')
    return '\n'.join(lines)
