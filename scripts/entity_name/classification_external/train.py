import argparse
from pathlib import Path
from datasets import load_from_disk
from src.entity_name.classification.training import EntityTypeModelTrainer
from src.config import get_model_config, list_available_models
from src.entity_name.classification_external.config import (
    MODEL_REGISTRY,
    DEFAULT_MODEL,
    TASK,
)

DEFAULT_DATASET = 'data/processed/entity_name/classification_external'


def main():
    """Run a training pipeline."""
    parser = argparse.ArgumentParser(
        description='Train external entity type classification model with QLoRA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available models:
  {', '.join(list_available_models(MODEL_REGISTRY))}

Examples:
  uv run scripts/entity_name/classification_external/train.py {DEFAULT_DATASET}
    --model qwen-0.5b
    --num-epochs 1
        """,
    )
    parser.add_argument(
        'dataset_path',
        type=str,
        nargs='?',  # Make optional for --list-models
        default=DEFAULT_DATASET,
        help='Path to preprocessed dataset directory',
    )
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_MODEL,
        help=f'Model alias (default: {DEFAULT_MODEL}). '
        f'Use --list-models to see options.',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for adapter weights (default: task-scoped config dir)',
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=1,
        help='Number of training epochs (1-2 recommended)',
    )
    parser.add_argument(
        '--list-models', action='store_true', help='List available models and exit'
    )
    parser.add_argument(
        '--push-to-hub',
        action='store_true',
        default=False,
        help='Push trained adapter to Hugging Face Hub (default: False)',
    )

    args = parser.parse_args()

    # Handle --list-models flag
    if args.list_models:
        print('Available models:')
        for alias in list_available_models(MODEL_REGISTRY):
            config = get_model_config(alias, task=TASK, registry=MODEL_REGISTRY)
            print(f'  {alias}: {config.base_model}')
        return

    # Validate dataset path
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise ValueError(f'Dataset path does not exist: {dataset_path}')

    # Resolve model configuration
    config = get_model_config(args.model, task=TASK, registry=MODEL_REGISTRY)
    print('\nModel Configuration:')
    print(f'  Model Alias: {args.model}')
    print(f'  Base Model: {config.base_model}')
    print(f'  Adapter Dir: {args.output_dir or config.adapter_dir}')

    # Load dataset
    print(f'\nLoading dataset from {dataset_path}...')
    dataset = load_from_disk(str(dataset_path))
    print(f'Train: {len(dataset["train"])} samples')
    print(f'Validation: {len(dataset["validation"])} samples')
    print(f'Test: {len(dataset["test"])} samples')

    # Initialize trainer with config
    print(f'\nInitializing trainer with base model: {config.base_model}')
    trainer = EntityTypeModelTrainer(
        model=config,
        output_dir=args.output_dir,
    )

    # Load model with 4-bit quantization
    print('Loading model with 4-bit quantization...')
    model, tokenizer = trainer.load_model()
    print('Model loaded successfully')

    # Setup LoRA adapter
    print('Setting up QLoRA adapter...')
    trainer.model = trainer.setup_lora(model)
    print(f'LoRA configured with r={trainer.lora_r}, alpha={trainer.lora_alpha}')

    # Execute training
    print(f'\nStarting training for {args.num_epochs} epoch(s)...')
    print('This may take a while depending on dataset size and hardware.')

    try:
        stats = trainer.train(
            dataset, num_epochs=args.num_epochs, push_to_hub=args.push_to_hub
        )

        # Display training statistics
        print('\n' + '=' * 60)
        print('Training Statistics:')
        print('=' * 60)
        print(f'Final Loss: {stats["train_loss"]:.4f}')
        print(f'Runtime: {stats["train_runtime"]:.2f}s')
        print('=' * 60)

    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print('\n' + '=' * 60)
            print('ERROR: CUDA Out of Memory')
            print('=' * 60)
            print('Suggestions:')
            print('  - Reduce batch size (currently 2)')
            print('  - Reduce max_seq_length (currently 2048)')
            print('  - Reduce gradient_accumulation_steps')
            print('  - Close other GPU-intensive applications')
            print('=' * 60)
        raise

    # Save adapter weights
    output_path = Path(trainer.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f'\nSaving adapter weights to {output_path}...')
    trainer.save_adapter(str(output_path))

    print('\n' + '=' * 60)
    print('Training Complete!')
    print('=' * 60)
    print(f'Adapter weights saved to: {output_path}')
    print('Use these weights for inference with EntityTypeClassifier')
    print('=' * 60)


if __name__ == '__main__':
    main()
