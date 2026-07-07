"""Evaluation pipeline for entity name extraction models."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from datasets import load_from_disk
from tqdm import tqdm

from src.entity_name.extraction.inference import EntityNameExtractor
from src.entity_name.extraction.models import EntityName
from src.utils import (
    levenshtein_similarity,
    normalize_text,
    save_file,
    similarity_bucket,
    token_f1,
)


@dataclass
class EvaluationMetrics:
    """Container for entity name evaluation metrics."""

    total_samples: int

    # Exact match
    exact_match: int
    exact_match_rate: float

    # Levenshtein similarity (averaged)
    avg_levenshtein_similarity: float

    # Token-level F1 (averaged)
    avg_token_precision: float
    avg_token_recall: float
    avg_token_f1: float

    # Distribution of similarity scores
    similarity_buckets: dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        """Format metrics as readable string."""
        lines = [
            '=' * 60,
            'Entity Name Evaluation Results',
            '=' * 60,
            f'Total Samples: {self.total_samples}',
            '',
            'Exact Match (case-insensitive):',
            f'  Count: {self.exact_match}/{self.total_samples}',
            f'  Rate: {self.exact_match_rate:.2%}',
            '',
            'Levenshtein Similarity:',
            f'  Average: {self.avg_levenshtein_similarity:.4f}',
            '',
            'Token-Level Metrics:',
            f'  Precision: {self.avg_token_precision:.4f}',
            f'  Recall: {self.avg_token_recall:.4f}',
            f'  F1: {self.avg_token_f1:.4f}',
            '',
            'Similarity Distribution:',
        ]

        for bucket, count in sorted(self.similarity_buckets.items()):
            lines.append(f'  {bucket}: {count}')

        lines.append('=' * 60)
        return '\n'.join(lines)


class EntityNameEvaluator:
    """Evaluator for entity name extraction models."""

    def __init__(
        self,
        extractor: EntityNameExtractor,
        test_data_path: str,
    ):
        """Initialize evaluator.

        Args:
            extractor: Initialized EntityNameExtractor with loaded model
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

    def _compare_names(
        self,
        predicted: EntityName,
        ground_truth: EntityName,
    ) -> dict:
        """Compare predicted and ground truth entity names.

        Args:
            predicted: Predicted EntityName object
            ground_truth: Ground truth EntityName object

        Returns:
            Dictionary with comparison metrics
        """
        pred_norm = normalize_text(predicted.name)
        truth_norm = normalize_text(ground_truth.name)

        is_exact = pred_norm == truth_norm
        lev_sim = levenshtein_similarity(pred_norm, truth_norm)
        tok_f1 = token_f1(pred_norm, truth_norm)

        return {
            'exact_match': is_exact,
            'levenshtein_similarity': lev_sim,
            'token_precision': tok_f1['precision'],
            'token_recall': tok_f1['recall'],
            'token_f1': tok_f1['f1'],
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
            RuntimeError: If an extractor model is not loaded
            ValueError: If save_predictions=True but output_path is None
        """
        if self.extractor.model is None:
            raise RuntimeError('Model not loaded. Call extractor.load_model() first.')

        if save_predictions and output_path is None:
            raise ValueError('output_path required when save_predictions=True')

        # Load test data
        test_dataset = self.load_test_data()

        # Initialize counters
        total = len(test_dataset)
        exact_matches = 0
        total_lev_sim = 0.0
        total_token_precision = 0.0
        total_token_recall = 0.0
        total_token_f1 = 0.0
        similarity_buckets: dict[str, int] = {}

        # Store predictions if requested
        predictions = []

        # Process in batches
        print('\nRunning evaluation...')
        num_batches = (total + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc='Evaluating'):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total)
            batch_samples = test_dataset[start_idx:end_idx]

            # Extract input texts and ground truths
            input_texts = batch_samples['name_address']
            ground_truths = [
                EntityName(name=batch_samples['cleaned_name'][i])
                for i in range(len(batch_samples['name_address']))
            ]

            # Run batch inference
            predicted_names = self.extractor.extract_batch(input_texts)

            # Compare names
            for input_text, predicted, ground_truth in zip(
                input_texts, predicted_names, ground_truths
            ):
                comparison = self._compare_names(predicted, ground_truth)

                # Update counters
                if comparison['exact_match']:
                    exact_matches += 1

                total_lev_sim += comparison['levenshtein_similarity']
                total_token_precision += comparison['token_precision']
                total_token_recall += comparison['token_recall']
                total_token_f1 += comparison['token_f1']

                # Update similarity distribution
                bucket = similarity_bucket(comparison['levenshtein_similarity'])
                similarity_buckets[bucket] = similarity_buckets.get(bucket, 0) + 1

                # Store prediction if requested
                if save_predictions:
                    predictions.append(
                        {
                            'input': input_text,
                            'ground_truth': ground_truth.to_output(),
                            'predicted': predicted.to_output(),
                            'exact_match': comparison['exact_match'],
                            'levenshtein_similarity': comparison[
                                'levenshtein_similarity'
                            ],
                            'token_f1': comparison['token_f1'],
                        }
                    )

        # Calculate metrics
        metrics = EvaluationMetrics(
            total_samples=total,
            exact_match=exact_matches,
            exact_match_rate=exact_matches / total if total > 0 else 0.0,
            avg_levenshtein_similarity=total_lev_sim / total if total > 0 else 0.0,
            avg_token_precision=total_token_precision / total if total > 0 else 0.0,
            avg_token_recall=total_token_recall / total if total > 0 else 0.0,
            avg_token_f1=total_token_f1 / total if total > 0 else 0.0,
            similarity_buckets=similarity_buckets,
        )

        # Save predictions if requested
        if save_predictions and predictions and output_path is not None:
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
        json_content = json.dumps(predictions, indent=2, ensure_ascii=False)
        save_file(json_content, output_path)
        print(f'\nPredictions saved to {output_path}')
