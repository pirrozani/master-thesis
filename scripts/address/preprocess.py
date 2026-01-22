"""Preprocessing script for address data."""

import argparse
from pathlib import Path
from src.address.preprocessing import AddressPreprocessor


def main():
    """Run preprocessing pipeline."""
    parser = argparse.ArgumentParser(description='Preprocess address CSV data')
    parser.add_argument('csv_path', type=str, help='Path to input CSV file')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Output directory for processed dataset'
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

    print(f'Dataset saved to {output_path}')
    print(f"Train: {len(dataset['train'])} samples")
    print(f"Validation: {len(dataset['validation'])} samples")
    print(f"Test: {len(dataset['test'])} samples")


if __name__ == '__main__':
    main()
