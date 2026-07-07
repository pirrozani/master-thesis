"""Evaluation pipeline for address extraction models."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from datasets import load_from_disk
from tqdm import tqdm

from src.address.inference import AddressExtractor
from src.address.models import Address
from src.utils import normalize_text, save_file


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    total_samples: int
    exact_match: int
    exact_match_rate: float

    # Field-level metrics
    street_correct: int
    city_correct: int
    state_correct: int
    zip_correct: int
    country_correct: int

    street_accuracy: float
    city_accuracy: float
    state_accuracy: float
    zip_accuracy: float
    country_accuracy: float

    # Overall field accuracy
    field_accuracy: float

    def __str__(self) -> str:
        """Format metrics as readable string."""
        lines = [
            '=' * 60,
            'Evaluation Results',
            '=' * 60,
            f'Total Samples: {self.total_samples}',
            '',
            'Exact Match (all fields correct):',
            f'  Count: {self.exact_match}/{self.total_samples}',
            f'  Rate: {self.exact_match_rate:.2%}',
            '',
            'Field-Level Accuracy:',
            f'  Street: {self.street_accuracy:.2%} '
            f'({self.street_correct}/{self.total_samples})',
            f'  City: {self.city_accuracy:.2%} '
            f'({self.city_correct}/{self.total_samples})',
            f'  State: {self.state_accuracy:.2%} '
            f'({self.state_correct}/{self.total_samples})',
            f'  Zip Code: {self.zip_accuracy:.2%} '
            f'({self.zip_correct}/{self.total_samples})',
            f'  Country: {self.country_accuracy:.2%} '
            f'({self.country_correct}/{self.total_samples})',
            '',
            f'Overall Field Accuracy: {self.field_accuracy:.2%}',
            '=' * 60,
        ]
        return '\n'.join(lines)


class AddressEvaluator:
    """Evaluator for address extraction models."""

    def __init__(
        self,
        extractor: AddressExtractor,
        test_data_path: str,
    ):
        """Initialize evaluator.

        Args:
            extractor: Initialized AddressExtractor with loaded model
            test_data_path: Path to test dataset directory
        """
        self.extractor = extractor
        self.test_data_path = Path(test_data_path)

    def load_test_data(self):
        """Load test dataset from the disk.

        Returns:
            HuggingFace Dataset with test samples

        Raises:
            FileNotFoundError: If a test data path doesn't exist
        """
        if not self.test_data_path.exists():
            raise FileNotFoundError(f'Test data not found at {self.test_data_path}')

        print(f'Loading test data from {self.test_data_path}...')
        dataset = load_from_disk(str(self.test_data_path))
        print(f'Loaded {len(dataset)} test samples')

        return dataset

    def _compare_addresses(
        self,
        predicted: Address,
        ground_truth: Address,
    ) -> dict[str, bool]:
        """Compare predicted and ground truth addresses.

        Args:
            predicted: Predicted Address object
            ground_truth: Ground truth Address object

        Returns:
            Dictionary with field-level comparison results
        """
        return {
            name: normalize_text(getattr(predicted, name))
            == normalize_text(getattr(ground_truth, name))
            for name in Address.model_fields
        }

    def evaluate(
        self,
        save_predictions: bool = False,
        output_path: Optional[str] = None,
        batch_size: int = 8,
    ) -> EvaluationMetrics:
        """Run evaluation on a test dataset.

        Args:
            save_predictions: Whether to save predictions to file
            output_path: Path to save predictions (required if save_predictions=True)
            batch_size: Number of samples to process in each batch

        Returns:
            EvaluationMetrics with computed metrics

        Raises:
            RuntimeError: If an extractor model is not loaded,
            ValueError: If save_predictions=True but output_path is None
        """
        if self.extractor.model is None:
            raise RuntimeError('Model not loaded. Call extractor.load_model() first.')

        if save_predictions and output_path is None:
            raise ValueError('output_path required when save_predictions=True')

        # Load test data
        test_dataset = self.load_test_data()

        # get only 50 samples for quick eval
        # test_dataset = test_dataset.select(range(min(50, len(test_dataset))))

        # Initialize counters
        total = len(test_dataset)
        exact_matches = 0
        field_correct = {
            'street': 0,
            'city': 0,
            'state': 0,
            'zip_code': 0,
            'country': 0,
        }

        # Store predictions if requested
        predictions = []

        # Process in batches
        print('\nRunning evaluation...')
        num_batches = (total + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc='Evaluating'):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total)
            batch_samples = test_dataset[start_idx:end_idx]

            # Extract input texts and ground truths directly from raw fields
            input_texts = batch_samples['name_address']
            ground_truths = [
                Address(
                    street=batch_samples['street'][i],
                    city=batch_samples['city'][i],
                    state=batch_samples['state'][i],
                    zip_code=batch_samples['zip_code'][i],
                    country=batch_samples['country'][i],
                )
                for i in range(len(batch_samples['name_address']))
            ]

            # Run batch inference
            predicted_addresses = self.extractor.extract_batch(input_texts)

            # Compare addresses
            for input_text, predicted, ground_truth in zip(
                input_texts, predicted_addresses, ground_truths
            ):
                comparison = self._compare_addresses(predicted, ground_truth)

                # Update counters
                if all(comparison.values()):
                    exact_matches += 1

                for field, is_correct in comparison.items():
                    if is_correct:
                        field_correct[field] += 1

                # Store prediction if requested
                if save_predictions:
                    predictions.append(
                        {
                            'input': input_text,
                            'ground_truth': ground_truth.to_pipe(),
                            'predicted': predicted.to_pipe(),
                            'exact_match': all(comparison.values()),
                            'field_correct': comparison,
                        }
                    )

        # Calculate metrics
        metrics = EvaluationMetrics(
            total_samples=total,
            exact_match=exact_matches,
            exact_match_rate=exact_matches / total if total > 0 else 0.0,
            street_correct=field_correct['street'],
            city_correct=field_correct['city'],
            state_correct=field_correct['state'],
            zip_correct=field_correct['zip_code'],
            country_correct=field_correct['country'],
            street_accuracy=field_correct['street'] / total if total > 0 else 0.0,
            city_accuracy=field_correct['city'] / total if total > 0 else 0.0,
            state_accuracy=field_correct['state'] / total if total > 0 else 0.0,
            zip_accuracy=field_correct['zip_code'] / total if total > 0 else 0.0,
            country_accuracy=field_correct['country'] / total if total > 0 else 0.0,
            field_accuracy=sum(field_correct.values()) / (total * 5)
            if total > 0
            else 0.0,
        )

        # Save predictions if requested
        if save_predictions and predictions:
            self._save_predictions(predictions, output_path)

        return metrics

    def _save_predictions(
        self,
        predictions: list[dict],
        output_path: str,
    ) -> None:
        """Save predictions to JSON file.

        Args:
            predictions: List of prediction dictionaries
            output_path: Path to save predictions
        """
        # Convert predictions to JSON string
        json_content = json.dumps(predictions, indent=2, ensure_ascii=False)

        # Save using utility function
        save_file(json_content, output_path)

        print(f'\nPredictions saved to {output_path}')
