"""Preprocessing script for address and entity-name data."""

import argparse
from pathlib import Path

from src.data.preprocessing import AddressPreprocessor, EntityNamePreprocessor

TASKS = ('address', 'entity_name')


def _build_preprocessor(task: str):
    """Return the preprocessor instance for the given task."""
    if task == 'address':
        return AddressPreprocessor()
    if task == 'entity_name':
        return EntityNamePreprocessor()
    raise ValueError(f'Unknown task: {task!r}. Must be one of {TASKS}.')


def main():
    """Run preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description='Preprocess raw CSV data into HuggingFace Arrow splits',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run scripts/data/preprocess.py data/raw/address.csv
  uv run scripts/data/preprocess.py data/raw/address.csv --task entity_name
  uv run scripts/data/preprocess.py data/raw/address.csv --task entity_name \\
      --output-dir data/processed/entity_name
        """,
    )
    parser.add_argument('csv_path', type=str, help='Path to input CSV file')
    parser.add_argument(
        '--task',
        type=str,
        default='address',
        choices=TASKS,
        help='Which task to preprocess for (default: address)',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for processed dataset (default: data/processed/<task>)',
    )

    args = parser.parse_args()

    output_dir = args.output_dir or f'data/processed/{args.task}'

    preprocessor = _build_preprocessor(args.task)

    print(f'Task: {args.task}')
    print(f'Loading CSV from {args.csv_path}...')
    df = preprocessor.load_csv(args.csv_path)
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

    # Show example field keys (not values) - per CLAUDE.md §8 redaction.
    print('\nExample field keys:')
    print('-' * 60)
    example = dataset['train'][0]
    for key in example.keys():
        print(f'  {key}')
    print('-' * 60)


if __name__ == '__main__':
    main()
