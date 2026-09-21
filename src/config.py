"""Centralized model configuration helpers.

The per-task model registries live in each task's own ``config.py``
(e.g. ``src/address/config.py``); this module holds only the shared
``ModelConfig`` dataclass and the registry-agnostic helper functions.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for a specific model variant."""

    name: str
    base_model: str
    lora_r: int = 16
    lora_alpha: int = 16
    adapter_dir: str = ''

    def __post_init__(self):
        """Set the default adapter directory based on the model name."""
        if not self.adapter_dir:
            self.adapter_dir = f'adapters/{self.name}'


def get_model_config(
    model_alias: str, task: str, registry: dict[str, ModelConfig]
) -> ModelConfig:
    """Get model configuration by alias with a task-specific adapter directory.

    Args:
        model_alias: Short model alias (e.g., 'qwen-0.5b', 'qwen-3b')
        task: Task name for adapter directory scoping
        registry: Task-specific registry mapping aliases to configurations

    Returns:
        ModelConfig instance with task-specific adapter_dir

    Raises:
        ValueError: If the model alias is not found in the registry
    """
    if model_alias not in registry:
        available = ', '.join(registry.keys())
        raise ValueError(
            f"Model alias '{model_alias}' not found in registry. "
            f'Available models: {available}'
        )

    config = registry[model_alias]

    # Return a copy with task-specific adapter directory
    return ModelConfig(
        name=config.name,
        base_model=config.base_model,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        adapter_dir=f'adapters/{task}/{config.name}',
    )


def list_available_models(registry: dict[str, ModelConfig]) -> list[str]:
    """List all available models.

    Args:
        registry: Task-specific registry mapping aliases to configurations

    Returns:
        List of registered model aliases
    """
    return list(registry.keys())
