"""Prompt templates and message formatting for address extraction."""

from transformers import PreTrainedTokenizer
from src.address.models import Address

# Standard prompt template
EXTRACT_PROMPT = 'Extract: {text}'

# System prompt for address extraction
SYSTEM_PROMPT = 'You are an expert at extracting structured addresses from raw text. Output format: street|city|state|zip|country'


def create_system_message() -> dict:
    """Create a system message for address extraction.

    Returns:
        Message dict with 'role' and 'content' keys
    """
    return {
        'role': 'system',
        'content': SYSTEM_PROMPT,
    }


def create_user_message(text: str) -> dict:
    """Create a user message for address extraction.

    Args:
        text: Raw text containing address to extract

    Returns:
        Message dict with 'role' and 'content' keys
    """
    return {
        'role': 'user',
        'content': EXTRACT_PROMPT.format(text=text),
    }


def create_assistant_message(address: Address) -> dict:
    """Create an assistant message with a pipe-delimited address.
    Args:
        address: Address object with extracted fields
    Returns:
        Message dict with 'role' and 'content' keys
    """
    return {
        'role': 'assistant',
        'content': address.to_pipe(),
    }


def create_training_messages(fields: dict) -> list[dict]:
    """Create a complete message list for training.

    Args:
        fields : Dictionary with keys:

            - name_address: Raw text containing address
            - street: Street address
            - city: City name
            - state: State or province
            - zip_code: ZIP or postal code
            - country: Country name

    Returns:
        List of message dicts for training
    """
    return [
        create_system_message(),
        create_user_message(fields.get('name_address', '')),
        create_assistant_message(
            Address(
                street=fields.get('street', ''),
                city=fields.get('city', ''),
                state=fields.get('state', ''),
                zip_code=fields.get('zip_code', ''),
                country=fields.get('country', '')
            )
        ),
    ]


def create_inference_messages(text: str) -> list[dict]:
    """Create a message list for inference (system and user messages).

    Args:
        text: Raw text containing address

    Returns:
        List with system and user message dicts
    """
    return [create_system_message(), create_user_message(text)]


def format_training_example(fields: dict, tokenizer: PreTrainedTokenizer) -> str:
    """Format a complete training example with a chat template.

    Args:
        fields: Dictionary with keys:

            - name_address: Raw text containing address
            - street: Street address
            - city: City name
            - state: State or province
            - zip_code: ZIP or postal code
            - country: Country name
        tokenizer: HuggingFace tokenizer with chat template support

    Returns:
        Formatted chat text ready for training
    """
    messages = create_training_messages(fields)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def format_inference_prompt(text: str, tokenizer: PreTrainedTokenizer) -> str:
    """Format an inference prompt with chat template.

    Args:
        text: Raw text containing address
        tokenizer: HuggingFace tokenizer with chat template support

    Returns:
        Formatted chat prompt with generation prompt appended
    """
    messages = create_inference_messages(text)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
