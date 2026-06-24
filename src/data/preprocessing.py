"""Data preprocessing for address, entity name, and entity type tasks."""

import re

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
        required_columns = [
            'name_address',
            'entity_name',
        ]

        df = pd.read_csv(path)

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f'Missing required columns: {missing_columns}')

        return df

    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean entity name data: handle nulls, strip whitespace, remove quotes.

        Args:
            df: Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()

        fields_to_clean = ['name_address', 'entity_name']

        for field in fields_to_clean:
            if field in df.columns:
                # Handle nulls - convert to empty string
                df[field] = df[field].fillna('')

                # Convert to string type
                df[field] = df[field].astype(str)

                # Replace 'nan' string with empty string
                df[field] = df[field].replace('nan', '')

                # Remove extraneous quotes
                df[field] = df[field].str.replace('"', '', regex=False)
                df[field] = df[field].str.replace("'", '', regex=False)

                # Strip leading/trailing whitespace
                df[field] = df[field].str.strip()

        # Drop rows where either field is empty
        mask = (df['name_address'].str.len() > 0) & (df['entity_name'].str.len() > 0)
        df = pd.DataFrame(df[mask]).reset_index(drop=True)

        return df

    @staticmethod
    def format_row(row: pd.Series) -> dict:
        """Convert row to dictionary with required fields.

        Args:
            row: DataFrame row with columns: name_address, entity_name

        Returns:
            Dictionary with raw field values for training
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


class EntityTypePreprocessor:
    """Preprocessor for entity type classification (company vs person).

    Derives a binary label from the Google Maps ``gm_types`` column: a name is
    labeled ``person`` if any of its types is a person-indicating professional
    type, otherwise ``company``. The model is trained on the cleaned entity name
    only; ``gm_types`` is kept in the processed dataset purely for auditing.
    """

    # Person-indicating Google Maps types. This curated set is the primary
    # tunable of the labeling heuristic and the accuracy ceiling of the task;
    # refine it after inspecting the type/label distribution.
    PERSON_GM_TYPES = {
        'doctor',
        'dentist',
        'lawyer',
        'physiotherapist',
    }

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
        required_columns = ['entity_name', 'gm_types']

        df = pd.read_csv(path)

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f'Missing required columns: {missing_columns}')

        return df

    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize entity names and gm_types, drop empty-name rows.

        Args:
            df: Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()

        # Handle nulls / literal 'nan' for both columns
        for field_name in ['entity_name', 'gm_types']:
            if field_name in df.columns:
                df[field_name] = df[field_name].fillna('')
                df[field_name] = df[field_name].astype(str)
                df[field_name] = df[field_name].replace('NaN', '')

        # Clean the entity name (quotes are NOT stripped from gm_types: they are
        # part of the parse format).
        df['entity_name'] = df['entity_name'].str.replace('"', '', regex=False)
        df['entity_name'] = df['entity_name'].str.replace("'", '', regex=False)
        df['entity_name'] = df['entity_name'].str.strip()

        # Drop rows with an empty entity name
        df = df[df['entity_name'] != '']

        return df.reset_index(drop=True)

    @staticmethod
    def _parse_gm_types(cell) -> list[str]:
        """Parse a numpy-array-style gm_types cell into a list of types.

        The cell looks like ``['doctor' 'point of interest' 'establishment']`` —
        single-quoted tokens separated by spaces (no commas), and tokens may
        themselves contain spaces. Quoted substrings are extracted with a regex.

        Args:
            cell: Raw gm_types value (string, may be empty/NaN)

        Returns:
            List of lowercased, stripped type tokens (empty list if none)
        """
        if pd.isna(cell):
            return []

        text = str(cell)
        tokens = re.findall(r"'([^']*)'", text)
        return [token.lower().strip() for token in tokens]

    @classmethod
    def _derive_label(cls, types: list[str]) -> str:
        """Derive the binary label from a list of gm_types.

        Args:
            types: List of lowercased type tokens

        Returns:
            'person' if any type is person-indicating, else 'company'
        """
        for token in types:
            if token in cls.PERSON_GM_TYPES:
                return 'person'
        return 'company'

    @staticmethod
    def format_row(row: pd.Series) -> dict:
        """Convert a row to a dictionary with classification fields.

        Args:
            row: DataFrame row with columns: entity_name, label, gm_types

        Returns:
            Dictionary with the entity name, derived label, and raw gm_types
        """
        return {
            'entity_name': row.get('entity_name', ''),
            'label': row.get('label', ''),
            'gm_types': row.get('gm_types', ''),
        }

    def create_dataset(self, df: pd.DataFrame) -> DatasetDict:
        """Create HuggingFace DatasetDict with label-stratified splits.

        Labels are derived from gm_types, then an 80/10/10 split is taken within
        each class (seed 42) so the minority class is present in every split.

        Args:
            df: Input DataFrame with cleaned data

        Returns:
            DatasetDict with train/val/test splits
        """
        df = df.copy()

        # Derive labels from gm_types
        df['label'] = df['gm_types'].apply(
            lambda cell: self._derive_label(self._parse_gm_types(cell))
        )

        # Stratified split: slice each class independently, then concatenate
        train_parts, val_parts, test_parts = [], [], []
        for _, group in df.groupby('label'):
            group = group.sample(frac=1.0, random_state=42).reset_index(drop=True)
            n = len(group)
            train_size = int(n * self.train_ratio)
            val_size = int(n * self.val_ratio)

            train_parts.append(group[:train_size])
            val_parts.append(group[train_size : train_size + val_size])
            test_parts.append(group[train_size + val_size : n])

        # Concatenate per-class slices and shuffle each split (seed 42)
        train_df = (
            pd.concat(train_parts)
            .sample(frac=1.0, random_state=42)
            .reset_index(drop=True)
        )
        val_df = (
            pd.concat(val_parts)
            .sample(frac=1.0, random_state=42)
            .reset_index(drop=True)
        )
        test_df = (
            pd.concat(test_parts)
            .sample(frac=1.0, random_state=42)
            .reset_index(drop=True)
        )

        # Format each split
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


class EntityTypeExternalPreprocessor:
    """Preprocessor for the external person/company classification dataset.

    The source data (``ele-sage/person-company-names-classification``) is already
    cleaned and pre-split into ``train.csv`` / ``test.csv``, each with two
    columns: ``text`` (the cleaned, title-cased name) and ``label`` (integer:
    ``0`` = person, ``1`` = company). No cleaning is applied here; the
    preprocessor only maps columns/labels and draws class-balanced, fixed-size
    splits so the result matches the schema used by the entity-name
    classification task (columns ``entity_name`` and ``label``).

    Download the raw dataset first:
    ``hf download ele-sage/person-company-names-classification --type dataset
    --local-dir data/raw/person-company-names-classification``
    """

    # Integer label -> string label mapping (matches the dataset card).
    LABEL_MAP = {0: 'person', 1: 'company'}

    def __init__(
        self,
        n_train: int = 20000,
        n_val: int = 5000,
        n_test: int = 5000,
    ):
        """Initialize preprocessor.

        Args:
            n_train: Total training rows (split evenly across both classes)
            n_val: Total validation rows (split evenly across both classes)
            n_test: Total test rows (split evenly across both classes)
        """
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test

    def load_csv(self, path: str) -> pd.DataFrame:
        """Load and validate a CSV with the required columns.

        Args:
            path: Path to CSV file

        Returns:
            Loaded DataFrame

        Raises:
            ValueError: If required columns are missing
        """
        required_columns = ['text', 'label']

        df = pd.read_csv(path)

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f'Missing required columns: {missing_columns}')

        return df

    @classmethod
    def _prepare(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Map columns and labels without cleaning the names.

        Coerces the integer label to the string labels used downstream, renames
        ``text`` to ``entity_name``, and drops rows whose label is not 0 or 1.
        The names themselves are left untouched (the source is pre-cleaned).

        Args:
            df: Input DataFrame with columns ``text`` and ``label``

        Returns:
            DataFrame with columns ``entity_name`` and ``label``
        """
        df = df.copy()

        # Coerce label to numeric and drop anything outside {0, 1}
        df['label'] = pd.to_numeric(df['label'], errors='coerce')
        df = df[df['label'].isin(cls.LABEL_MAP.keys())]

        df['entity_name'] = df['text'].astype(str)
        df['label'] = df['label'].astype(int).map(cls.LABEL_MAP)

        return df[['entity_name', 'label']].reset_index(drop=True)

    @staticmethod
    def format_row(row: pd.Series) -> dict:
        """Convert a row to a dictionary with classification fields.

        Args:
            row: DataFrame row with columns ``entity_name`` and ``label``

        Returns:
            Dictionary with the entity name and its string label
        """
        return {
            'entity_name': row.get('entity_name', ''),
            'label': row.get('label', ''),
        }

    def create_dataset(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> DatasetDict:
        """Create a HuggingFace DatasetDict from pre-split source frames.

        Draws class-balanced, fixed-size splits: ``train`` and ``validation`` are
        disjoint samples drawn per class from ``train_df`` (seed 42); ``test`` is
        sampled per class from ``test_df`` (seed 42). Each split is balanced 50/50
        across person/company.

        Args:
            train_df: Source train frame (columns ``text``, ``label``)
            test_df: Source test frame (columns ``text``, ``label``)

        Returns:
            DatasetDict with train/validation/test splits
        """
        train_df = self._prepare(train_df)
        test_df = self._prepare(test_df)

        per_class_train = self.n_train // 2
        per_class_val = self.n_val // 2
        per_class_test = self.n_test // 2

        # Train + validation come from disjoint per-class slices of train_df
        train_parts, val_parts, test_parts = [], [], []
        for _, group in train_df.groupby('label'):
            group = group.sample(frac=1.0, random_state=42).reset_index(drop=True)
            train_parts.append(group[:per_class_train])
            val_parts.append(group[per_class_train : per_class_train + per_class_val])

        # Test is sampled per class from the held-out test_df
        for _, group in test_df.groupby('label'):
            group = group.sample(frac=1.0, random_state=42).reset_index(drop=True)
            test_parts.append(group[:per_class_test])

        # Concatenate per-class slices and shuffle each split (seed 42)
        train_split = (
            pd.concat(train_parts)
            .sample(frac=1.0, random_state=42)
            .reset_index(drop=True)
        )
        val_split = (
            pd.concat(val_parts)
            .sample(frac=1.0, random_state=42)
            .reset_index(drop=True)
        )
        test_split = (
            pd.concat(test_parts)
            .sample(frac=1.0, random_state=42)
            .reset_index(drop=True)
        )

        # Format each split
        train_data = [self.format_row(row) for _, row in train_split.iterrows()]
        val_data = [self.format_row(row) for _, row in val_split.iterrows()]
        test_data = [self.format_row(row) for _, row in test_split.iterrows()]

        # Create DatasetDict
        dataset_dict = DatasetDict(
            {
                'train': Dataset.from_list(train_data),
                'validation': Dataset.from_list(val_data),
                'test': Dataset.from_list(test_data),
            }
        )

        return dataset_dict
