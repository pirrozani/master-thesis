"""Prompt templates and message formatting for entity name extraction."""

from typing import TYPE_CHECKING

from src.entity_name.extraction.models import EntityName

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

# Standard prompt template
EXTRACT_PROMPT = 'Extract: {text}'

# System prompt for entity name extraction
SYSTEM_PROMPT = (
    'You are an expert at extracting and cleaning entity names from raw text. '
    'Given a raw text entry containing an entity name mixed with address '
    'information, abbreviations, store numbers, and other noise, extract and '
    'return ONLY the clean, normalised entity name. '
    'Handle abbreviations, misspellings, and variations. '
    'Output the cleaned entity name as a single string.'
)


def create_system_message() -> dict:
    """Create a system message for entity name extraction.

    Returns:
        Message dict with 'role' and 'content' keys
    """
    return {
        'role': 'system',
        'content': SYSTEM_PROMPT,
    }


def create_user_message(text: str) -> dict:
    """Create a user message for entity name extraction.

    Args:
        text: Raw text containing entity name to extract

    Returns:
        Message dict with 'role' and 'content' keys
    """
    return {
        'role': 'user',
        'content': EXTRACT_PROMPT.format(text=text),
    }


def create_assistant_message(entity_name: EntityName) -> dict:
    """Create an assistant message with the cleaned entity name.

    Args:
        entity_name: EntityName object with the cleaned name

    Returns:
        Message dict with 'role' and 'content' keys
    """
    return {
        'role': 'assistant',
        'content': entity_name.to_output(),
    }


def create_training_messages(fields: dict) -> list[dict]:
    """Create a complete message list for training.

    Args:
        fields: Dictionary with keys:

            - name_address: Raw text containing entity name and address
            - cleaned_name: The cleaned/normalised entity name

    Returns:
        List of message dicts for training
    """
    return [
        create_system_message(),
        create_user_message(fields.get('name_address', '')),
        create_assistant_message(EntityName(name=fields.get('cleaned_name', ''))),
    ]


def create_inference_messages(text: str) -> list[dict]:
    """Create a message list for inference (system and user messages).

    Args:
        text: Raw text containing entity name

    Returns:
        List with system and user message dicts
    """
    return [create_system_message(), create_user_message(text)]


def format_training_example(fields: dict, tokenizer: 'PreTrainedTokenizer') -> str:
    """Format a complete training example with a chat template.

    Args:
        fields: Dictionary with keys:

            - name_address: Raw text containing entity name and address
            - cleaned_name: The cleaned/normalised entity name
        tokenizer: HuggingFace tokenizer with chat template support

    Returns:
        Formatted chat text ready for training
    """
    messages = create_training_messages(fields)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def format_inference_prompt(text: str, tokenizer: 'PreTrainedTokenizer') -> str:
    """Format an inference prompt with chat template.

    Args:
        text: Raw text containing entity name
        tokenizer: HuggingFace tokenizer with chat template support

    Returns:
        Formatted chat prompt with generation prompt appended
    """
    messages = create_inference_messages(text)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
