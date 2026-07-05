"""Task registry mapping pipeline stages to vLLM adapters, prompts, parsers."""

from collections.abc import Callable
from dataclasses import dataclass

from src.address.config import (
    BATCH_MAX_NEW_TOKENS as ADDRESS_BATCH_TOKENS,
)
from src.address.config import (
    SINGLE_MAX_NEW_TOKENS as ADDRESS_SINGLE_TOKENS,
)
from src.address.models import Address
from src.address.prompt_templates import create_inference_messages as address_messages
from src.entity_name.classification.config import (
    BATCH_MAX_NEW_TOKENS as CLASSIFICATION_BATCH_TOKENS,
)
from src.entity_name.classification.config import (
    SINGLE_MAX_NEW_TOKENS as CLASSIFICATION_SINGLE_TOKENS,
)
from src.entity_name.classification.models import EntityType
from src.entity_name.classification.prompt_templates import (
    create_inference_messages as classification_messages,
)
from src.entity_name.extraction.config import (
    BATCH_MAX_NEW_TOKENS as EXTRACTION_BATCH_TOKENS,
)
from src.entity_name.extraction.config import (
    SINGLE_MAX_NEW_TOKENS as EXTRACTION_SINGLE_TOKENS,
)
from src.entity_name.extraction.models import EntityName
from src.entity_name.extraction.prompt_templates import (
    create_inference_messages as extraction_messages,
)


def parse_address(output: str) -> dict:
    """Parse a pipe-delimited completion into address fields.

    Args:
        output: Raw model completion

    Returns:
        Dict with street, city, state, zip_code, and country keys; empty
        strings when the completion is unparseable
    """
    try:
        address = Address.from_pipe(output)
    except Exception:
        address = Address()
    return address.model_dump()


def parse_entity_name(output: str) -> str:
    """Parse a completion into a cleaned entity name.

    Args:
        output: Raw model completion

    Returns:
        Cleaned entity name, or an empty string
    """
    return EntityName.from_output(output).name


def parse_entity_type(output: str) -> str:
    """Parse a completion into an entity type label.

    Args:
        output: Raw model completion

    Returns:
        'company' or 'person', or an empty string when unparseable
    """
    return EntityType.from_output(output).label


@dataclass(frozen=True)
class TaskSpec:
    """Settings for one vLLM-served adapter (serving and evaluation)."""

    key: str
    vllm_model: str
    create_messages: Callable[[str], list[dict]]
    parse: Callable[[str], object]
    single_max_tokens: int
    batch_max_tokens: int
    default_data_path: str
    input_field: str


ADDRESS = TaskSpec(
    key='address',
    vllm_model='address',
    create_messages=address_messages,
    parse=parse_address,
    single_max_tokens=ADDRESS_SINGLE_TOKENS,
    batch_max_tokens=ADDRESS_BATCH_TOKENS,
    default_data_path='data/processed/address/validation',
    input_field='name_address',
)

EXTRACTION = TaskSpec(
    key='extraction',
    vllm_model='extraction',
    create_messages=extraction_messages,
    parse=parse_entity_name,
    single_max_tokens=EXTRACTION_SINGLE_TOKENS,
    batch_max_tokens=EXTRACTION_BATCH_TOKENS,
    default_data_path='data/processed/entity_name/extraction/validation',
    input_field='name_address',
)

CLASSIFICATION = TaskSpec(
    key='classification',
    vllm_model='classification',
    create_messages=classification_messages,
    parse=parse_entity_type,
    single_max_tokens=CLASSIFICATION_SINGLE_TOKENS,
    batch_max_tokens=CLASSIFICATION_BATCH_TOKENS,
    default_data_path='data/processed/entity_name/classification_external/validation',
    input_field='entity_name',
)

ALL_TASKS = (ADDRESS, EXTRACTION, CLASSIFICATION)
