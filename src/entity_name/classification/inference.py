"""Inference pipeline for entity type classification."""

import torch
from unsloth import FastLanguageModel

from src.config import ModelConfig, get_model_config
from src.entity_name.classification.config import (
    MODEL_REGISTRY,
    DEFAULT_MODEL,
    TASK,
    SINGLE_MAX_NEW_TOKENS,
    BATCH_MAX_NEW_TOKENS,
)
from src.entity_name.classification.models import EntityType
from src.entity_name.classification.prompt_templates import format_inference_prompt


class EntityTypeClassifier:
    """Inference pipeline for classifying entity names as company or person."""

    def __init__(
        self,
        model: str | ModelConfig | None = None,
        adapter_path: str | None = None,
        device: str = 'cuda',
    ):
        """Initialize classifier.

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
            config = get_model_config(model, task=TASK, registry=MODEL_REGISTRY)
        else:
            # Use the default model
            config = get_model_config(DEFAULT_MODEL, task=TASK, registry=MODEL_REGISTRY)

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
            dtype = (
                torch.bfloat16
                if self.device.startswith('cuda')
                and torch.cuda.is_available()
                and torch.cuda.is_bf16_supported(including_emulation=False)
                else None
            )
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.adapter_path,  # Load from the adapter directory
                max_seq_length=2048,  # model context window
                dtype=dtype,  # Use BF16 explicitly on supported CUDA GPUs
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

    def classify(self, text: str) -> EntityType:
        """Classify a single entity name as company or person.

        Uses chat template formatting and greedy decoding for deterministic outputs.

        Args:
            text: Cleaned entity name to classify

        Returns:
            EntityType with the predicted label (empty label if unparseable)

        Raises:
            RuntimeError: If the model is not loaded
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                'Model not loaded. Call load_model() before classification.'
            )

        # Handle empty input (returns the invalid sentinel, not a default class)
        if not text or text.strip() == '':
            return EntityType()

        # Format prompt using centralized prompts module
        prompt = format_inference_prompt(text, self.tokenizer)

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
        ).to(self.device)

        # Generate output with greedy decoding (temperature=0)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=SINGLE_MAX_NEW_TOKENS,
            max_length=None,
            temperature=0.0,  # Greedy decoding for deterministic output
            do_sample=False,  # Disable sampling
            use_cache=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Extract only the generated tokens (exclude input tokens)
        generated_ids = outputs[0][len(inputs['input_ids'][0]) :]

        # Decode output
        completion = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()

        # Parse model output into the nearest valid label
        return EntityType.from_output(completion)

    def classify_batch(self, texts: list[str]) -> list[EntityType]:
        """Classify multiple entity names.

        Processes multiple texts in a batch for efficiency using a chat template.

        Args:
            texts: List of cleaned entity names to classify

        Returns:
            List of EntityType objects with predicted labels (empty names
            yield the invalid sentinel without reaching the model)

        Raises:
            RuntimeError: If the model is not loaded
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                'Model not loaded. Call load_model() before classification.'
            )

        # Handle empty input
        if not texts:
            return []

        # Pre-fill the invalid sentinel; empty names never reach the model
        results: list[EntityType] = [EntityType() for _ in texts]
        to_classify = [(i, text) for i, text in enumerate(texts) if text.strip()]
        if not to_classify:
            return results

        indices, non_empty_texts = zip(*to_classify)

        # Create chat-formatted prompts for all texts using a centralized module
        prompts = [
            format_inference_prompt(text, self.tokenizer) for text in non_empty_texts
        ]

        # Tokenize all inputs with padding
        inputs = self.tokenizer(
            prompts,
            return_tensors='pt',
            truncation=True,
            padding=True,
        ).to(self.device)

        # Generate outputs with greedy decoding
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=BATCH_MAX_NEW_TOKENS,
            max_length=None,
            temperature=0.0,  # Greedy decoding
            do_sample=False,
            use_cache=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Decode all outputs and parse labels back into their original slots
        for index, input_ids, output_ids in zip(indices, inputs['input_ids'], outputs):
            # Extract only generated tokens (exclude input tokens)
            generated_ids = output_ids[len(input_ids) :]
            completion = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).strip()

            results[index] = EntityType.from_output(completion)

        return results
