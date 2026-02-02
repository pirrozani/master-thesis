"""Preprocessing script for address data."""

import argparse
from pathlib import Path
from src.data.preprocessing import AddressPreprocessor


def main():
    """Run preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description='Preprocess address CSV data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run scripts/data/preprocess.py data/address.csv
  uv run scripts/data/preprocess.py data/address.csv --output-dir data/processed
        """,
    )
    parser.add_argument('csv_path', type=str, help='Path to input CSV file')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Output directory for processed dataset',
    )

    args = parser.parse_args()

    # Initialize preprocessor
    preprocessor = AddressPreprocessor()

    # Load and process data
    print(f'Loading CSV from {args.csv_path}...')
    df = preprocessor.load_csv(args.csv_path)
    print(f'Loaded {len(df)} rows')

    print('Cleaning data...')
    df = preprocessor.clean_data(df)

    print('Creating dataset...')
    dataset = preprocessor.create_dataset(df)

    # Save dataset
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_path))

    print(f'\nDataset saved to {output_path}')
    print(f'Train: {len(dataset["train"])} samples')
    print(f'Validation: {len(dataset["validation"])} samples')
    print(f'Test: {len(dataset["test"])} samples')

    # Show example of raw fields
    print('\nExample raw fields:')
    print('-' * 60)
    example = dataset['train'][0]
    for key, value in example.items():
        print(f'{key}: {value}')
    print('-' * 60)


if __name__ == '__main__':
    main()
