"""Prompt templates and message formatting for entity type classification."""

from transformers import PreTrainedTokenizer
from src.entity_name.classification.models import EntityType

# Standard prompt template
CLASSIFY_PROMPT = 'Classify: {name}'

# System prompt for entity type classification
SYSTEM_PROMPT = (
    'You are an expert at classifying entity names. Given an entity name, '
    'decide whether it refers to a company or a person, using contextual and '
    "semantic cues in the name. Respond with exactly one word: 'company' or "
    "'person'."
)


def create_system_message() -> dict:
    """Create a system message for entity type classification.

    Returns:
        Message dict with 'role' and 'content' keys
    """
    return {
        'role': 'system',
        'content': SYSTEM_PROMPT,
    }


def create_user_message(name: str) -> dict:
    """Create a user message for entity type classification.

    Args:
        name: Cleaned entity name to classify

    Returns:
        Message dict with 'role' and 'content' keys
    """
    return {
        'role': 'user',
        'content': CLASSIFY_PROMPT.format(name=name),
    }


def create_assistant_message(entity_type: EntityType) -> dict:
    """Create an assistant message with the entity type label.

    Args:
        entity_type: EntityType object with the target label

    Returns:
        Message dict with 'role' and 'content' keys
    """
    return {
        'role': 'assistant',
        'content': entity_type.to_output(),
    }


def create_training_messages(fields: dict) -> list[dict]:
    """Create a complete message list for training.

    Args:
        fields: Dictionary with keys:

            - entity_name: Cleaned entity name to classify
            - label: Target class label ('company' or 'person')

    Returns:
        List of message dicts for training
    """
    return [
        create_system_message(),
        create_user_message(fields.get('entity_name', '')),
        create_assistant_message(EntityType(label=fields.get('label', ''))),
    ]


def create_inference_messages(name: str) -> list[dict]:
    """Create a message list for inference (system and user messages).

    Args:
        name: Cleaned entity name to classify

    Returns:
        List with system and user message dicts
    """
    return [create_system_message(), create_user_message(name)]


def format_training_example(fields: dict, tokenizer: PreTrainedTokenizer) -> str:
    """Format a complete training example with a chat template.

    Args:
        fields: Dictionary with keys:

            - entity_name: Cleaned entity name to classify
            - label: Target class label ('company' or 'person')
        tokenizer: HuggingFace tokenizer with chat template support

    Returns:
        Formatted chat text ready for training
    """
    messages = create_training_messages(fields)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def format_inference_prompt(name: str, tokenizer: PreTrainedTokenizer) -> str:
    """Format an inference prompt with chat template.

    Args:
        name: Cleaned entity name to classify
        tokenizer: HuggingFace tokenizer with chat template support

    Returns:
        Formatted chat prompt with generation prompt appended
    """
    messages = create_inference_messages(name)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
