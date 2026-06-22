"""Preprocessing script for address, entity name, and entity type data."""

import argparse
from collections import Counter
from pathlib import Path

from src.data.preprocessing import (
    AddressPreprocessor,
    EntityNamePreprocessor,
    EntityTypePreprocessor,
)

# Default output directory per task
DEFAULT_OUTPUT_DIRS = {
    'address': 'data/processed/address',
    'entity_name': 'data/processed/entity_name/extraction',
    'entity_type': 'data/processed/entity_name/classification',
}


def preprocess_address(csv_path: str, output_dir: str) -> None:
    """Run address preprocessing pipeline.

    Args:
        csv_path: Path to input CSV file
        output_dir: Output directory for processed dataset
    """
    preprocessor = AddressPreprocessor()

    print(f'Loading CSV from {csv_path}...')
    df = preprocessor.load_csv(csv_path)
    print(f'Loaded {len(df)} rows')

    print('Cleaning data...')
    df = preprocessor.clean_data(df)

    print('Creating dataset...')
    dataset = preprocessor.create_dataset(df)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_path))

    print(f'\nDataset saved to {output_path}')
    print(f'Train: {len(dataset["train"])} samples')
    print(f'Validation: {len(dataset["validation"])} samples')
    print(f'Test: {len(dataset["test"])} samples')

    print('\nExample raw fields:')
    print('-' * 60)
    example = dataset['train'][0]
    for key, value in example.items():
        print(f'{key}: {value}')
    print('-' * 60)


def preprocess_entity_name(csv_path: str, output_dir: str) -> None:
    """Run entity name preprocessing pipeline.

    Args:
        csv_path: Path to input CSV file
        output_dir: Output directory for processed dataset
    """
    preprocessor = EntityNamePreprocessor()

    print(f'Loading CSV from {csv_path}...')
    df = preprocessor.load_csv(csv_path)
    print(f'Loaded {len(df)} rows')

    print('Cleaning data...')
    df = preprocessor.clean_data(df)
    print(f'After cleaning: {len(df)} rows')

    print('Creating dataset...')
    dataset = preprocessor.create_dataset(df)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_path))

    print(f'\nDataset saved to {output_path}')
    print(f'Train: {len(dataset["train"])} samples')
    print(f'Validation: {len(dataset["validation"])} samples')
    print(f'Test: {len(dataset["test"])} samples')

    print('\nExample raw fields:')
    print('-' * 60)
    example = dataset['train'][0]
    for key, value in example.items():
        print(f'{key}: {value}')
    print('-' * 60)


def preprocess_entity_type(csv_path: str, output_dir: str) -> None:
    """Run entity type classification preprocessing pipeline.

    Args:
        csv_path: Path to input CSV file
        output_dir: Output directory for processed dataset
    """
    preprocessor = EntityTypePreprocessor()

    print(f'Loading CSV from {csv_path}...')
    df = preprocessor.load_csv(csv_path)
    print(f'Loaded {len(df)} rows')

    print('Cleaning data...')
    df = preprocessor.clean_data(df)
    print(f'After cleaning: {len(df)} rows')

    print('Creating dataset...')
    dataset = preprocessor.create_dataset(df)

    # Save dataset
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_path))

    print(f'\nDataset saved to {output_path}')
    print(f'Train: {len(dataset["train"])} samples')
    print(f'Validation: {len(dataset["validation"])} samples')
    print(f'Test: {len(dataset["test"])} samples')

    # Show class balance only (counts) — never raw rows (PII)
    print('\nClass balance (label counts per split):')
    print('-' * 60)
    for split_name in ['train', 'validation', 'test']:
        counts = Counter(dataset[split_name]['label'])
        summary = ', '.join(f'{label}={counts[label]}' for label in sorted(counts))
        print(f'{split_name}: {summary}')
    print('-' * 60)


def main():
    """Run preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description='Preprocess CSV data for address, entity name, or entity type',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run scripts/data/preprocess.py data/raw/address.csv --task address
  uv run scripts/data/preprocess.py data/raw/address.csv --task entity_name
  uv run scripts/data/preprocess.py data/raw/address.csv --task entity_type
  uv run scripts/data/preprocess.py data/raw/address.csv \\
    --task address --output-dir data/processed/address
        """,
    )
    parser.add_argument('csv_path', type=str, help='Path to input CSV file')
    parser.add_argument(
        '--task',
        type=str,
        choices=['address', 'entity_name', 'entity_type'],
        default='address',
        help='Task to preprocess data for (default: address)',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for processed dataset (default: per-task)',
    )

    args = parser.parse_args()

    # Default output directory based on task
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIRS[args.task]

    if args.task == 'address':
        preprocess_address(args.csv_path, output_dir)
    elif args.task == 'entity_name':
        preprocess_entity_name(args.csv_path, output_dir)
    elif args.task == 'entity_type':
        preprocess_entity_type(args.csv_path, output_dir)


if __name__ == '__main__':
    main()
