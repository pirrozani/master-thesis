"""Evaluation script for spaCy-based entity name extraction."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from src.entity_name.evaluation import EntityNameEvaluator
from src.entity_name.inference import EntityNameExtractionPipeline
from src.utils import save_file


def _run_id(spacy_model: str) -> str:
    """Build a run id of the form 'entity_name-<spacy>-<YYYYMMDD-HHMMSS>'."""
    safe = spacy_model.replace('/', '-')
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    return f'entity_name-{safe}-{ts}'


def main():
    """Run evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description='Evaluate spaCy-based entity-name extraction on a test split',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on the test split
  uv run scripts/entity_name/evaluate.py data/processed/entity_name

  # Evaluate and save per-sample raw records under outputs/evaluation/raw/
  uv run scripts/entity_name/evaluate.py data/processed/entity_name --include-raw
        """,
    )
    parser.add_argument(
        'dataset_path',
        type=str,
        help='Path to processed entity_name DatasetDict (with train/test splits)',
    )
    parser.add_argument(
        '--spacy-model',
        type=str,
        default='en_core_web_trf',
        help='spaCy pipeline name (default: en_core_web_trf)',
    )
    parser.add_argument(
        '--include-raw',
        action='store_true',
        help='Save per-sample raw predictions under outputs/evaluation/raw/ '
        '(PII-bearing; see CLAUDE.md §8).',
    )
    parser.add_argument(
        '--metrics-output',
        type=str,
        default=None,
        help='Path for metrics JSON '
        '(default: outputs/evaluation/<run_id>.metrics.json)',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for spaCy pipe (default: 32)',
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f'Error: dataset path does not exist: {dataset_path}', file=sys.stderr)
        sys.exit(1)

    test_path = dataset_path / 'test'
    if not test_path.exists():
        print(
            f'Error: test split not found at {test_path}. '
            f'Expected DatasetDict with train/validation/test subdirectories.',
            file=sys.stderr,
        )
        sys.exit(1)

    run_id = _run_id(args.spacy_model)
    metrics_output = args.metrics_output or f'outputs/evaluation/{run_id}.metrics.json'
    raw_output = f'outputs/evaluation/raw/{run_id}.raw.json'

    print('\nEvaluation Configuration:')
    print(f'  Run ID:           {run_id}')
    print(f'  Dataset:          {dataset_path}')
    print(f'  spaCy model:      {args.spacy_model}')
    print(f'  Batch size:       {args.batch_size}')
    print(f'  Metrics out:      {metrics_output}')
    if args.include_raw:
        print(f'  Raw out:          {raw_output} (PII)')

    pipeline = EntityNameExtractionPipeline(spacy_model=args.spacy_model)

    print('\nLoading spaCy model...')
    try:
        pipeline.load_model()
    except RuntimeError as e:
        print(f'Error loading model: {e}', file=sys.stderr)
        sys.exit(1)

    evaluator = EntityNameEvaluator(
        extractor=pipeline,
        test_data_path=str(test_path),
    )

    try:
        metrics = evaluator.evaluate(
            batch_size=args.batch_size,
            include_raw=args.include_raw,
            raw_output_path=raw_output if args.include_raw else None,
        )
    except Exception as e:
        print(f'Error during evaluation: {e}', file=sys.stderr)
        sys.exit(1)

    save_file(EntityNameEvaluator.metrics_to_json(metrics), metrics_output)
    print(f'\nMetrics saved to: {metrics_output}')

    print('\n' + str(metrics))


if __name__ == '__main__':
    main()
