from src.entity_name.classification.config import (
    BATCH_MAX_NEW_TOKENS,
    DEFAULT_MODEL,
    MODEL_REGISTRY,
    SINGLE_MAX_NEW_TOKENS,
)

TASK = 'entity_name/classification_external'

__all__ = [
    'TASK',
    'DEFAULT_MODEL',
    'MODEL_REGISTRY',
    'SINGLE_MAX_NEW_TOKENS',
    'BATCH_MAX_NEW_TOKENS',
]
