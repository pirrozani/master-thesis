"""Data preprocessing for address and entity name extraction."""

import pandas as pd
from datasets import Dataset, DatasetDict


class AddressPreprocessor:
    """Preprocessor for address data."""

    def __init__(
        self,
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
    ):
        """Initialize preprocessor.

        Args:
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
        """
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def load_csv(self, path: str) -> pd.DataFrame:
        """Load and validate CSV with required columns.

        Args:
            path: Path to CSV file

        Returns:
            Loaded DataFrame

        Raises:
            ValueError: If required columns are missing
        """
        required_columns = [
            'name_address',
            'gm_name',
            'gm_address',
            'gm_types',
            'entity_name',
            'address',
            'city',
            'state',
            'zip_code',
            'country',
        ]

        df = pd.read_csv(path)

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f'Missing required columns: {missing_columns}')

        return df

    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize countries, remove quotes, handle nulls.

        Args:
            df: Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()

        # Define fields to clean
        fields_to_clean = ['address', 'city', 'state', 'zip_code', 'country']

        for field in fields_to_clean:
            if field in df.columns:
                # Handle nulls - convert to empty string
                df[field] = df[field].fillna('')

                # Convert to a string type
                df[field] = df[field].astype(str)

                # Replace 'nan' string with empty string
                df[field] = df[field].replace('nan', '')

                # Remove extraneous quotes
                df[field] = df[field].str.replace('"', '', regex=False)
                df[field] = df[field].str.replace("'", '', regex=False)

                # Strip leading/trailing whitespace
                df[field] = df[field].str.strip()

        return df

    @staticmethod
    def format_row(row: pd.Series) -> dict:
        """Convert row to dictionary with required fields.

        Extracts the relevant fields from a DataFrame row for dataset creation.

        Args:
            row: DataFrame row with columns: name_address, address, city,
                state, zip_code, country

        Returns:
            Dictionary with raw field values for training
        """
        return {
            'name_address': row.get('name_address', ''),
            'street': row.get('address', ''),
            'city': row.get('city', ''),
            'state': row.get('state', ''),
            'zip_code': row.get('zip_code', ''),
            'country': row.get('country', ''),
        }

    def create_dataset(self, df: pd.DataFrame) -> DatasetDict:
        """Create HuggingFace DatasetDict with splits.

        Args:
            df: Input DataFrame with cleaned and normalized data

        Returns:
            DatasetDict with train/val/test splits containing raw field values
        """
        # Shuffle the dataframe
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        # Calculate split sizes
        n = len(df)

        train_size = int(n * self.train_ratio)
        val_size = int(n * self.val_ratio)

        # Split the data
        train_df = df[:train_size]
        val_df = df[train_size : train_size + val_size]
        test_df = df[train_size + val_size : n]

        # Format each split with raw field values
        train_data = [self.format_row(row) for _, row in train_df.iterrows()]
        val_data = [self.format_row(row) for _, row in val_df.iterrows()]
        test_data = [self.format_row(row) for _, row in test_df.iterrows()]

        # Create HuggingFace datasets
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        test_dataset = Dataset.from_list(test_data)

        # Create DatasetDict
        dataset_dict = DatasetDict(
            {'train': train_dataset, 'validation': val_dataset, 'test': test_dataset}
        )

        return dataset_dict


class EntityNamePreprocessor:
    """Preprocessor for entity name data."""

    def __init__(
        self,
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
    ):
        """Initialize preprocessor.

        Args:
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
        """
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def load_csv(self, path: str) -> pd.DataFrame:
        """Load and validate CSV with required columns.

        Args:
            path: Path to CSV file

        Returns:
            Loaded DataFrame

        Raises:
            ValueError: If required columns are missing
        """
        required_columns = ['name_address', 'entity_name']

        df = pd.read_csv(path)

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f'Missing required columns: {missing_columns}')

        return df

    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean entity name data: nulls, quotes, whitespace.

        Drops rows where either field is empty after cleaning.

        Args:
            df: Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()

        fields_to_clean = ['name_address', 'entity_name']

        for field in fields_to_clean:
            if field in df.columns:
                df[field] = df[field].fillna('')
                df[field] = df[field].astype(str)
                # Remove zero-width spaces (the actual U+200B character)
                df[field] = df[field].str.replace('​', '', regex=False)
                df[field] = df[field].str.replace('"', '', regex=False)
                df[field] = df[field].str.replace("'", '', regex=False)
                df[field] = df[field].str.strip()

        mask = (df['name_address'].str.len() > 0) & (df['entity_name'].str.len() > 0)
        df = pd.DataFrame(df[mask]).reset_index(drop=True)

        return df

    @staticmethod
    def format_row(row: pd.Series) -> dict:
        """Convert row to dictionary with required fields.

        Args:
            row: DataFrame row with columns: name_address, entity_name

        Returns:
            Dictionary with raw field values
        """
        return {
            'name_address': row.get('name_address', ''),
            'cleaned_name': row.get('entity_name', ''),
        }

    def create_dataset(self, df: pd.DataFrame) -> DatasetDict:
        """Create HuggingFace DatasetDict with splits.

        Args:
            df: Input DataFrame with cleaned and normalized data

        Returns:
            DatasetDict with train/val/test splits
        """
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        n = len(df)
        train_size = int(n * self.train_ratio)
        val_size = int(n * self.val_ratio)

        train_df = df[:train_size]
        val_df = df[train_size : train_size + val_size]
        test_df = df[train_size + val_size : n]

        train_data = [self.format_row(row) for _, row in train_df.iterrows()]
        val_data = [self.format_row(row) for _, row in val_df.iterrows()]
        test_data = [self.format_row(row) for _, row in test_df.iterrows()]

        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        test_dataset = Dataset.from_list(test_data)

        dataset_dict = DatasetDict(
            {'train': train_dataset, 'validation': val_dataset, 'test': test_dataset}
        )

        return dataset_dict
