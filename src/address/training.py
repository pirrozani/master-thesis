"""Model training for address extraction."""

from unsloth import FastLanguageModel, is_bfloat16_supported, unsloth_train
from datasets import DatasetDict
from trl import SFTTrainer, SFTConfig
from src.config import ModelConfig, get_model_config
from src.address.config import MODEL_REGISTRY, DEFAULT_MODEL, TASK
from src.address.prompt_templates import format_training_example


class AddressModelTrainer:
    """Trainer for an address extraction model."""

    def __init__(
        self,
        model: str | ModelConfig | None = None,
        output_dir: str | None = None,
    ):
        """Initialize trainer.

        Args:
            model: Model alias (e.g., 'qwen-0.5b') or ModelConfig instance.
            output_dir: Directory for saving adapters (overrides config)
        """
        # Resolve model configuration
        if isinstance(model, ModelConfig):
            config = model
        elif isinstance(model, str):
            config = get_model_config(model, task=TASK, registry=MODEL_REGISTRY)
            if config is None:
                raise ValueError(f'Unknown model: {model}')
        elif model is None:
            config = get_model_config(DEFAULT_MODEL, task=TASK, registry=MODEL_REGISTRY)
        else:
            raise TypeError(f'Expected ModelConfig or str, got {type(model)}')

        # Apply configuration
        self.base_model = config.base_model
        self.lora_r = config.lora_r
        self.lora_alpha = config.lora_alpha
        self.output_dir = output_dir if output_dir is not None else config.adapter_dir
        self.config = config

    def load_model(self):
        """Load base model with Unsloth 4-bit quantization.

        Returns:
            Tuple of (model, tokenizer)
        """

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model,
            max_seq_length=2048,  # model context window
            dtype=None,  # Auto-detect dtype
            load_in_4bit=True,  # Use 4-bit quantization
        )

        self.model = model
        self.tokenizer = tokenizer

        return model, tokenizer

    def setup_lora(self, model):
        """Configure QLoRA adapter with target modules.

        Args:
            model: Base model

        Returns:
            PEFT model with LoRA
        """

        model = FastLanguageModel.get_peft_model(
            model,
            r=self.lora_r,
            target_modules=[
                'q_proj',
                'k_proj',
                'v_proj',
                'o_proj',
                'gate_proj',
                'up_proj',
                'down_proj',
            ],
            lora_alpha=self.lora_alpha,
            lora_dropout=0,  # Optimized for inference
            bias='none',  # Standard for QLoRA
            use_gradient_checkpointing='unsloth',  # Unsloth optimization
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

        return model

    def train(
        self, dataset: DatasetDict, num_epochs: int = 1, push_to_hub: bool = False
    ):
        """Execute fine-tuning with SFTTrainer.

        Args:
            dataset: Training dataset with 'train' split containing raw fields
                     (name_address, street, city, state, zip_code, country)
            num_epochs: Number of training epochs (1-2 recommended)
            push_to_hub: Whether to push trained adapter to Hugging Face Hub
                For `push_to_hub=True`, ensure you have set up Hugging Face credentials


        Returns:
            Training statistics dictionary
        """

        if not hasattr(self, 'model') or not hasattr(self, 'tokenizer'):
            raise RuntimeError(
                'Model not loaded. Call load_model() and setup_lora() first.'
            )

        # Apply chat template formatting to create a 'text' field
        def apply_chat_template(example):
            fields = {
                'name_address': example['name_address'],
                'street': example['street'],
                'city': example['city'],
                'state': example['state'],
                'zip_code': example['zip_code'],
                'country': example['country'],
            }
            formatted_text = format_training_example(fields, self.tokenizer)
            return {'text': formatted_text}

        print('Applying chat template formatting...')
        dataset = dataset.map(apply_chat_template, desc='Formatting with chat template')

        train_dataset = dataset['train']
        eval_dataset = dataset['validation']

        # Configure training arguments using SFTConfig
        training_args = SFTConfig(
            output_dir=self.output_dir,
            per_device_train_batch_size=2,  # Actual batch size = 2
            gradient_accumulation_steps=8,  # Effective batch size = 16
            gradient_checkpointing=False,
            warmup_ratio=0.1,
            num_train_epochs=num_epochs,
            learning_rate=2e-4,
            lr_scheduler_type='cosine',
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim='adamw_8bit',
            weight_decay=0.01,
            save_strategy='epoch',
            save_total_limit=2,
            do_eval=True,
            eval_strategy='epoch',
            report_to='none',
            dataset_num_proc=1,  # Disable multiprocessing for Windows compatibility
            dataset_text_field='text',  # Specify the text field explicitly
            packing=False,
        )

        # Initialize SFTTrainer with preformatted dataset
        trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
        )

        # Execute training
        print(f'Starting training for {num_epochs} epoch(s)...')
        train_result = unsloth_train(trainer)

        # Push to huggingface hub if configured
        if push_to_hub:
            print('Pushing trained adapter to Hugging Face Hub...')
            repository_url = trainer.push_to_hub(
                commit_message='Trained address extraction adapter'
            )
            print(f'Adapter pushed to: {repository_url}')

        # Log training statistics
        metrics = train_result.metrics

        print('\nTraining completed!')
        print(f'Final training loss: {metrics.get("train_loss", "N/A"):.4f}')
        print(f'Total training runtime: {metrics.get("train_runtime", "N/A"):.2f}s')

        # Store trainer for saving
        self.trainer = trainer

        return {
            'train_loss': metrics.get('train_loss'),
            'train_runtime': metrics.get('train_runtime'),
        }

    def save_adapter(self, path: str) -> None:
        """Save LoRA adapter weights to adapters/ directory.

        Args:
            path: Path to save adapter weights

        Raises:
            RuntimeError: If the model hasn't been trained yet
        """
        if not hasattr(self, 'model'):
            raise RuntimeError(
                'Model not loaded. Call load_model() and setup_lora() first.'
            )

        # Save adapter using Unsloth's optimized save method
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        print(f'Adapter weights saved to: {path}')
