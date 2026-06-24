"""Preprocessing script for address, entity name, and entity type data."""

import argparse
from collections import Counter
from pathlib import Path

from src.data.preprocessing import (
    AddressPreprocessor,
    EntityNamePreprocessor,
    EntityTypeExternalPreprocessor,
    EntityTypePreprocessor,
)

# Default output directory per task
DEFAULT_OUTPUT_DIRS = {
    'address': 'data/processed/address',
    'entity_name': 'data/processed/entity_name/extraction',
    'entity_type': 'data/processed/entity_name/classification',
    'entity_type_external': 'data/processed/entity_name/classification_external',
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

    # Load some of the data because there is a class imbalance
    df = df.sample(frac=0.1, random_state=42)
    df = df.reset_index(drop=True)

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


def preprocess_entity_type_external(
    train_csv_path: str, test_csv_path: str, output_dir: str
) -> None:
    """Run the external person/company classification preprocessing pipeline.

    The source data is already cleaned and pre-split, so this only maps the
    ``text``/``label`` columns to the downstream schema and draws class-balanced,
    fixed-size splits (train/validation from train.csv, test from test.csv).

    Args:
        train_csv_path: Path to the source train.csv
        test_csv_path: Path to the source test.csv
        output_dir: Output directory for the processed dataset
    """
    preprocessor = EntityTypeExternalPreprocessor()

    print(f'Loading train CSV from {train_csv_path}...')
    train_df = preprocessor.load_csv(train_csv_path)
    print(f'Loaded {len(train_df)} train rows')

    print(f'Loading test CSV from {test_csv_path}...')
    test_df = preprocessor.load_csv(test_csv_path)
    print(f'Loaded {len(test_df)} test rows')

    print('Creating dataset (no cleaning; balanced fixed-size splits)...')
    dataset = preprocessor.create_dataset(train_df, test_df)

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
  uv run scripts/data/preprocess.py \\
data/raw/person-company-names-classification/train.csv --task entity_type_external
  uv run scripts/data/preprocess.py data/raw/address.csv \\
    --task address --output-dir data/processed/address

Prerequisite for --task entity_type_external (download the raw dataset first):
  hf download ele-sage/person-company-names-classification \\
    --type dataset --local-dir data/raw/person-company-names-classification
        """,
    )
    parser.add_argument(
        'csv_path',
        type=str,
        help='Path to input CSV file (for entity_type_external this is train.csv)',
    )
    parser.add_argument(
        '--task',
        type=str,
        choices=['address', 'entity_name', 'entity_type', 'entity_type_external'],
        default='address',
        help='Task to preprocess data for (default: address)',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for processed dataset (default: per-task)',
    )
    parser.add_argument(
        '--test-csv',
        type=str,
        default=None,
        help='Path to the test CSV (entity_type_external only; '
        'defaults to a sibling test.csv next to csv_path)',
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
    elif args.task == 'entity_type_external':
        test_csv_path = args.test_csv or str(Path(args.csv_path).with_name('test.csv'))
        preprocess_entity_type_external(args.csv_path, test_csv_path, output_dir)


if __name__ == '__main__':
    main()
