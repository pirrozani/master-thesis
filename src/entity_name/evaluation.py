"""Evaluation pipeline for entity name extraction."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from datasets import load_from_disk
from rapidfuzz.distance import Levenshtein
from tqdm import tqdm

from src.entity_name.inference import EntityNameExtractionPipeline
from src.entity_name.models import EntityName
from src.utils import save_file


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Compute normalized Levenshtein similarity between two strings.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Similarity score in [0.0, 1.0]; 1.0 indicates identical strings.
    """
    if not s1 and not s2:
        return 1.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    distance = Levenshtein.distance(s1, s2)
    return 1.0 - (distance / max_len)


def token_f1(predicted: str, ground_truth: str) -> dict[str, float]:
    """Compute token-level precision, recall, and F1.

    Tokenizes both strings on whitespace, lowercased.

    Args:
        predicted: Predicted entity-name string.
        ground_truth: Ground-truth entity-name string.

    Returns:
        Dict with keys ``precision``, ``recall``, ``f1`` in [0.0, 1.0].
    """
    pred_tokens = set(predicted.lower().split())
    truth_tokens = set(ground_truth.lower().split())

    if not pred_tokens and not truth_tokens:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    if not pred_tokens or not truth_tokens:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    common = pred_tokens & truth_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return {'precision': precision, 'recall': recall, 'f1': f1}


@dataclass
class EvaluationMetrics:
    """Container for entity-name evaluation metrics."""

    total_samples: int

    exact_match: int
    exact_match_rate: float

    avg_levenshtein_similarity: float

    avg_token_precision: float
    avg_token_recall: float
    avg_token_f1: float

    similarity_buckets: dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        """Format metrics as a readable string."""
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
    """Evaluator for entity name extraction pipelines."""

    def __init__(
        self,
        extractor: EntityNameExtractionPipeline,
        test_data_path: str,
    ):
        """Initialize evaluator.

        Args:
            extractor: Initialized ``EntityNameExtractionPipeline`` whose
                underlying spaCy model has been loaded.
            test_data_path: Path to the test split directory on disk.
        """
        self.extractor = extractor
        self.test_data_path = Path(test_data_path)

    def load_test_data(self):
        """Load the test split.

        Returns:
            HuggingFace ``Dataset`` containing the test samples.

        Raises:
            FileNotFoundError: If the test data path does not exist.
        """
        if not self.test_data_path.exists():
            raise FileNotFoundError(f'Test data not found at {self.test_data_path}')

        print(f'Loading test data from {self.test_data_path}...')
        dataset = load_from_disk(str(self.test_data_path))
        print(f'Loaded {len(dataset)} test samples')

        return dataset

    @staticmethod
    def _normalize(value: str) -> str:
        """Lowercase + strip a value for comparison."""
        return value.lower().strip()

    def _compare_names(
        self,
        predicted: EntityName,
        ground_truth: EntityName,
    ) -> dict:
        """Compute per-sample comparison metrics."""
        pred_norm = self._normalize(predicted.name)
        truth_norm = self._normalize(ground_truth.name)

        is_exact = pred_norm == truth_norm
        lev_sim = levenshtein_similarity(pred_norm, truth_norm)
        tok = token_f1(pred_norm, truth_norm)

        return {
            'exact_match': is_exact,
            'levenshtein_similarity': lev_sim,
            'token_precision': tok['precision'],
            'token_recall': tok['recall'],
            'token_f1': tok['f1'],
        }

    @staticmethod
    def _get_similarity_bucket(similarity: float) -> str:
        """Bucket a similarity score for distribution reporting."""
        if similarity == 1.0:
            return '1.00 (exact)'
        elif similarity >= 0.9:
            return '0.90-0.99'
        elif similarity >= 0.8:
            return '0.80-0.89'
        elif similarity >= 0.7:
            return '0.70-0.79'
        elif similarity >= 0.5:
            return '0.50-0.69'
        else:
            return '< 0.50'

    def evaluate(
        self,
        batch_size: int = 32,
        include_raw: bool = False,
        raw_output_path: Optional[str] = None,
    ) -> EvaluationMetrics:
        """Run evaluation over the test split.

        Returns aggregate metrics only. Per-sample raw records are written
        only when ``include_raw=True`` and are routed to ``raw_output_path``
        (callers should place this under ``outputs/evaluation/raw/`` per
        CLAUDE.md §8).

        Args:
            batch_size: Batch size for spaCy pipe.
            include_raw: Save per-sample raw-strings JSON to ``raw_output_path``.
            raw_output_path: Path to the raw-strings JSON (required when
                ``include_raw=True``).

        Returns:
            ``EvaluationMetrics`` with aggregate metrics.

        Raises:
            RuntimeError: If the underlying spaCy model is not loaded.
            ValueError: If ``include_raw=True`` but ``raw_output_path`` is None.
        """
        if self.extractor.model is None:
            raise RuntimeError('Model not loaded. Call extractor.load_model() first.')

        if include_raw and raw_output_path is None:
            raise ValueError('raw_output_path is required when include_raw=True')

        test_dataset = self.load_test_data()

        total = len(test_dataset)
        exact_matches = 0
        total_lev_sim = 0.0
        total_token_precision = 0.0
        total_token_recall = 0.0
        total_token_f1 = 0.0
        similarity_buckets: dict[str, int] = {}

        raw_records: list[dict] = []

        print('\nRunning evaluation...')
        num_batches = (total + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc='Evaluating'):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total)
            batch_samples = test_dataset[start_idx:end_idx]

            input_texts = batch_samples['name_address']
            ground_truths = [
                EntityName(name=batch_samples['cleaned_name'][i])
                for i in range(len(batch_samples['name_address']))
            ]

            predicted_names = self.extractor.extract_batch(input_texts)

            for input_text, predicted, ground_truth in zip(
                input_texts, predicted_names, ground_truths
            ):
                comparison = self._compare_names(predicted, ground_truth)

                if comparison['exact_match']:
                    exact_matches += 1
                total_lev_sim += comparison['levenshtein_similarity']
                total_token_precision += comparison['token_precision']
                total_token_recall += comparison['token_recall']
                total_token_f1 += comparison['token_f1']

                bucket = self._get_similarity_bucket(
                    comparison['levenshtein_similarity']
                )
                similarity_buckets[bucket] = similarity_buckets.get(bucket, 0) + 1

                if include_raw:
                    raw_records.append(
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

        if include_raw and raw_records and raw_output_path is not None:
            json_content = json.dumps(raw_records, indent=2, ensure_ascii=False)
            save_file(json_content, raw_output_path)
            print(f'Saved {len(raw_records)} raw records to {raw_output_path}')

        return metrics

    @staticmethod
    def metrics_to_json(metrics: EvaluationMetrics) -> str:
        """Serialize metrics to a JSON string."""
        return json.dumps(asdict(metrics), indent=2, ensure_ascii=False)
