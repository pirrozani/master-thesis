"""Inference pipeline for address extraction."""

from unsloth import FastLanguageModel
from src.config import ModelConfig, get_model_config, DEFAULT_MODEL
from src.address.models import Address
from src.address.prompt_templates import format_inference_prompt


class AddressExtractor:
    """Inference pipeline for extracting addresses."""

    def __init__(
        self,
        model: str | ModelConfig | None = None,
        adapter_path: str | None = None,
        device: str = 'cuda',
    ):
        """Initialize extractor.

        Args:
            model: Model alias (e.g., 'qwen-0.5b') or ModelConfig instance.
                   Takes precedence over individual parameters if provided.
            adapter_path: Path to adapter weights (overrides config)
            device: Device for inference (cuda/cpu)
        """
        # Resolve model configuration
        if isinstance(model, ModelConfig):
            config = model
        elif isinstance(model, str):
            config = get_model_config(model, task='address')
        else:
            # Use the default model
            config = get_model_config(DEFAULT_MODEL, task='address')

        self.base_model = config.base_model
        self.adapter_path = (
            adapter_path if adapter_path is not None else config.adapter_dir
        )
        self.device = device
        self.model = None
        self.tokenizer = None
        self.config = config

    def load_model(self) -> None:
        """Load base model and merge adapter weights.

        Loads the base model with Unsloth optimizations and merges the
        fine-tuned adapter weights for inference.

        Raises:
            RuntimeError: If model or adapter loading fails
        """
        try:
            # Load base model with adapter weights
            # FastLanguageModel.from_pretrained automatically merges adapters
            # when adapter_path is provided

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.adapter_path,  # Load from the adapter directory
                max_seq_length=self.config.max_seq_length,  # Use config value
                dtype=None,  # Auto-detect dtype
                load_in_4bit=True,  # Use 4-bit quantization for efficiency
            )

            # Enable inference mode for faster generation
            FastLanguageModel.for_inference(model)

            self.model = model
            self.tokenizer = tokenizer

            print(
                f'Model {self.config.base_model} loaded '
                f'successfully from {self.adapter_path}'
            )

        except Exception as e:
            raise RuntimeError(
                f'Failed to load model from {self.adapter_path}: {str(e)}'
            ) from e

    def extract(self, text: str) -> Address:
        """Extract address from single text input.

        Uses chat template formatting and greedy decoding for deterministic outputs.

        Args:
            text: Raw text containing address

        Returns:
            Extracted Address object with validated fields

        Raises:
            RuntimeError: If the model is not loaded
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError('Model not loaded. Call load_model() before extraction.')

        # Handle empty input
        if not text or text.strip() == '':
            return Address()

        # Format prompt using centralized prompts module
        prompt = format_inference_prompt(text, self.tokenizer)

        # Tokenize input (use model's configured max_seq_length)
        inputs = self.tokenizer(
            text=prompt,
            return_tensors='pt',
            truncation=True,
            max_length=self.config.max_seq_length,
        ).to(self.device)

        # Generate output with greedy decoding (temperature=0)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=128,  # Addresses are short
            temperature=0.0,  # Greedy decoding for deterministic output
            do_sample=False,  # Disable sampling
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Extract only the generated tokens (exclude input tokens)
        generated_ids = outputs[0][len(inputs['input_ids'][0]) :]

        # Decode output
        completion = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()

        # Parse pipe-delimited format to an Address object
        try:
            address = Address.from_pipe(completion)
        except Exception as e:
            # If parsing fails, return an empty Address
            print(f'Warning: Failed to parse model output: {e}')
            print(f'Raw output: {completion}')
            address = Address()

        return address

    def extract_batch(self, texts: list[str]) -> list[Address]:
        """Extract addresses from multiple inputs.

        Processes multiple texts in a batch for efficiency using a chat template.

        Args:
            texts: List of raw texts containing addresses

        Returns:
            List of extracted Address objects

        Raises:
            RuntimeError: If the model is not loaded
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError('Model not loaded. Call load_model() before extraction.')

        # Handle empty input
        if not texts:
            return []

        # Create chat-formatted prompts for all texts using a centralized module
        prompts = [format_inference_prompt(text, self.tokenizer) for text in texts]

        # Tokenize all inputs with padding
        inputs = self.tokenizer(
            text=prompts,
            return_tensors='pt',
            truncation=True,
            max_length=self.config.max_seq_length,
            padding=True,
        ).to(self.device)

        # Generate outputs with greedy decoding
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.0,  # Greedy decoding
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Decode all outputs and parse addresses
        addresses = []
        for i, (input_ids, output_ids) in enumerate(zip(inputs['input_ids'], outputs)):
            # Extract only generated tokens (exclude input tokens)
            generated_ids = output_ids[len(input_ids) :]
            completion = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).strip()

            # Parse pipe-delimited format to Address object
            try:
                address = Address.from_pipe(completion)
            except Exception as e:
                # If parsing fails, return empty Address
                print(f'Warning: Failed to parse output for text {i}: {e}')
                print(f'Raw output: {completion}')
                address = Address()

            addresses.append(address)

        return addresses
