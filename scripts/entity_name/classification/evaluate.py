"""Evaluation script for entity type classification models."""

import argparse
import sys
from pathlib import Path
from src.utils import save_file
from src.config import get_model_config, list_available_models
from src.entity_name.classification.config import MODEL_REGISTRY, DEFAULT_MODEL, TASK
from src.entity_name.classification.evaluation import EntityTypeEvaluator
from src.entity_name.classification.inference import EntityTypeClassifier

DEFAULT_TEST_DATA = 'data/processed/entity_name/classification/test'


def main():
    """Run the evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description='Evaluate entity type classification model on test dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available models:
  {', '.join(list_available_models(MODEL_REGISTRY))}

Examples:
  uv run scripts/entity_name/classification/evaluate.py <test_dir> --model qwen-0.5b
  uv run scripts/entity_name/classification/evaluate.py <test_dir> --save-predictions
        """,
    )
    parser.add_argument(
        'test_data_path',
        type=str,
        nargs='?',  # Make optional for --list-models
        default=DEFAULT_TEST_DATA,
        help='Path to test dataset directory',
    )
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_MODEL,
        help=f'Model alias (default: {DEFAULT_MODEL}). Use --list-models for options.',
    )
    parser.add_argument(
        '--adapter-path',
        type=str,
        default=None,
        help='Path to adapter weights directory (default: from model config)',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for inference (default: cuda)',
    )
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        help='Save predictions to JSON file',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path for predictions '
        '(default: per-task under outputs/evaluation/)',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for inference (default: 8)',
    )
    parser.add_argument(
        '--list-models', action='store_true', help='List available models and exit'
    )

    args = parser.parse_args()

    # Handle --list-models flag
    if args.list_models:
        print('Available models:')
        for alias in list_available_models(MODEL_REGISTRY):
            config = get_model_config(alias, task=TASK, registry=MODEL_REGISTRY)
            print(f'  {alias}: {config.base_model}')
        return

    # Validate paths
    test_data_path = Path(args.test_data_path)
    if not test_data_path.exists():
        print(
            f'Error: Test data path does not exist: {test_data_path}',
            file=sys.stderr,
        )
        sys.exit(1)

    # Resolve model configuration
    config = get_model_config(args.model, task=TASK, registry=MODEL_REGISTRY)

    # Resolve adapter path (CLI arg overrides config)
    adapter_path = (
        Path(args.adapter_path) if args.adapter_path else Path(config.adapter_dir)
    )
    if not adapter_path.exists():
        print(
            f'Error: Adapter path does not exist: {adapter_path}',
            file=sys.stderr,
        )
        sys.exit(1)

    print('\nModel Configuration:')
    print(f'  Model Alias: {args.model}')
    print(f'  Base Model: {config.base_model}')
    print(f'  Adapter Path: {adapter_path}')

    # Initialize classifier
    print(f'\nInitializing EntityTypeClassifier with adapter: {adapter_path}')
    classifier = EntityTypeClassifier(
        model=config,
        adapter_path=str(adapter_path),
        device=args.device,
    )

    # Load model
    print('Loading model...')
    try:
        classifier.load_model()
    except RuntimeError as e:
        print(f'Error loading model: {e}', file=sys.stderr)
        sys.exit(1)

    # Initialize evaluator
    evaluator = EntityTypeEvaluator(
        classifier=classifier,
        test_data_path=str(test_data_path),
    )

    # Resolve predictions output path
    output_path = args.output or f'outputs/evaluation/{args.model}_predictions.json'

    # Run evaluation
    try:
        metrics = evaluator.evaluate(
            save_predictions=args.save_predictions,
            output_path=output_path if args.save_predictions else None,
            batch_size=args.batch_size,
        )
    except Exception as e:
        print(f'Error during evaluation: {e}', file=sys.stderr)
        sys.exit(1)

    results_path = f'outputs/evaluation/{args.model}_metrics.txt'
    save_file(str(metrics), results_path)
    print(f'\nSaved evaluation results to: {results_path}')

    # Display results
    print('\n' + str(metrics))


if __name__ == '__main__':
    main()
