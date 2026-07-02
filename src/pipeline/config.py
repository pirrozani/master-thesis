from src.address import config as address_config
from src.config import ModelConfig, get_model_config
from src.entity_name.classification import config as classification_config
from src.entity_name.classification_external import (
    config as classification_external_config,
)
from src.entity_name.extraction import config as extraction_config

# Classification task variants selectable per run
CLASSIFICATION_TASKS: dict[str, str] = {
    'classification': classification_config.TASK,
    'classification_external': classification_external_config.TASK,
}
DEFAULT_CLASSIFICATION_TASK = 'classification_external'

# Per-stage default model aliases
DEFAULT_ADDRESS_MODEL = address_config.DEFAULT_MODEL
DEFAULT_EXTRACTION_MODEL = extraction_config.DEFAULT_MODEL
DEFAULT_CLASSIFICATION_MODEL = classification_config.DEFAULT_MODEL

# Chunk size for batch pipeline runs
DEFAULT_BATCH_SIZE = 8


def resolve_stage_configs(
    address_model: str | None = None,
    extraction_model: str | None = None,
    classification_model: str | None = None,
    classification_task: str = DEFAULT_CLASSIFICATION_TASK,
) -> dict[str, ModelConfig]:
    """Resolve the per-stage model configurations for a pipeline run.

    Args:
        address_model: Model alias for the address stage (None uses the
            stage default)
        extraction_model: Model alias for the name extraction stage
            (None uses the stage default)
        classification_model: Model alias for the classification stage
            (None uses the stage default)
        classification_task: Classification task variant whose adapters
            to use (key of CLASSIFICATION_TASKS)

    Returns:
        Mapping with keys 'address', 'extraction', and 'classification'
        to task-scoped ModelConfig instances

    Raises:
        ValueError: If classification_task or any model alias is unknown
    """
    if classification_task not in CLASSIFICATION_TASKS:
        valid = ', '.join(sorted(CLASSIFICATION_TASKS))
        raise ValueError(
            f"Unknown classification task '{classification_task}'. Valid tasks: {valid}"
        )

    return {
        'address': get_model_config(
            address_model or DEFAULT_ADDRESS_MODEL,
            task=address_config.TASK,
            registry=address_config.MODEL_REGISTRY,
        ),
        'extraction': get_model_config(
            extraction_model or DEFAULT_EXTRACTION_MODEL,
            task=extraction_config.TASK,
            registry=extraction_config.MODEL_REGISTRY,
        ),
        'classification': get_model_config(
            classification_model or DEFAULT_CLASSIFICATION_MODEL,
            task=CLASSIFICATION_TASKS[classification_task],
            registry=classification_config.MODEL_REGISTRY,
        ),
    }
