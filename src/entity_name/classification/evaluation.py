"""Evaluation pipeline for entity type classification models."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from datasets import load_from_disk
from tqdm import tqdm

from src.entity_name.classification.inference import EntityTypeClassifier
from src.entity_name.classification.models import COMPANY, PERSON, VALID_LABELS
from src.utils import class_metrics, normalize_text, save_file


@dataclass
class EvaluationMetrics:
    """Container for binary classification evaluation metrics.

    All rates are derived from a single 2x2 confusion matrix (over valid
    predictions) plus a separate count of invalid (unparseable) predictions.
    Invalid predictions are counted as incorrect for accuracy and as false
    negatives for their true class.
    """

    total_samples: int
    correct: int
    accuracy: float
    invalid_predictions: int

    company_precision: float
    company_recall: float
    company_f1: float

    person_precision: float
    person_recall: float
    person_f1: float

    macro_f1: float

    # 2x2 confusion matrix over valid predictions: confusion[true][predicted]
    confusion: dict = field(default_factory=dict)

    def __str__(self) -> str:
        """Format metrics as readable string."""
        cm = self.confusion
        lines = [
            '=' * 60,
            'Entity Type Classification - Evaluation Results',
            '=' * 60,
            f'Total Samples: {self.total_samples}',
            f'Invalid Predictions: {self.invalid_predictions} '
            '(output not in {company, person}; counted as incorrect)',
            '',
            f'Macro F1: {self.macro_f1:.2%}',
            f'Accuracy: {self.accuracy:.2%} ({self.correct}/{self.total_samples})',
            '',
            'Per-Class Metrics:',
            f'  company:  P={self.company_precision:.2%}  '
            f'R={self.company_recall:.2%}  F1={self.company_f1:.2%}',
            f'  person:   P={self.person_precision:.2%}  '
            f'R={self.person_recall:.2%}  F1={self.person_f1:.2%}',
            '',
            'Confusion Matrix (valid predictions, rows=true / cols=predicted):',
            '                  pred:company   pred:person',
            f'  true:company  {cm[COMPANY][COMPANY]:>12}  {cm[COMPANY][PERSON]:>12}',
            f'  true:person   {cm[PERSON][COMPANY]:>12}  {cm[PERSON][PERSON]:>12}',
            '=' * 60,
        ]
        return '\n'.join(lines)


class EntityTypeEvaluator:
    """Evaluator for entity type classification models."""

    def __init__(
        self,
        classifier: EntityTypeClassifier,
        test_data_path: str,
    ):
        """Initialize evaluator.

        Args:
            classifier: Initialized EntityTypeClassifier with loaded model
            test_data_path: Path to test dataset directory
        """
        self.classifier = classifier
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
            RuntimeError: If a classifier model is not loaded,
            ValueError: If save_predictions=True but output_path is None
        """
        if self.classifier.model is None:
            raise RuntimeError('Model not loaded. Call classifier.load_model() first.')

        if save_predictions and output_path is None:
            raise ValueError('output_path required when save_predictions=True')

        # Load test data
        test_dataset = self.load_test_data()

        # Initialize counters
        total = len(test_dataset)
        # confusion[true][predicted] over valid predictions only
        confusion = {
            COMPANY: {COMPANY: 0, PERSON: 0},
            PERSON: {COMPANY: 0, PERSON: 0},
        }
        invalid_by_true = {COMPANY: 0, PERSON: 0}

        # Store predictions if requested
        predictions = []

        # Process in batches
        print('\nRunning evaluation...')
        num_batches = (total + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc='Evaluating'):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total)
            batch_samples = test_dataset[start_idx:end_idx]

            # Extract input names and ground-truth labels directly from raw fields
            input_names = batch_samples['entity_name']
            ground_truths = [normalize_text(label) for label in batch_samples['label']]

            # Run batch inference
            predicted_types = self.classifier.classify_batch(input_names)

            # Tally predictions
            for input_name, predicted, gt in zip(
                input_names, predicted_types, ground_truths
            ):
                pred = predicted.label
                is_valid = pred in VALID_LABELS
                is_correct = is_valid and pred == gt

                if gt in VALID_LABELS:
                    if is_valid:
                        confusion[gt][pred] += 1
                    else:
                        invalid_by_true[gt] += 1

                # Store prediction if requested
                if save_predictions:
                    record = {
                        'input_name': input_name,
                        'predicted': pred,
                        'ground_truth': gt,
                        'correct': is_correct,
                    }
                    predictions.append(record)

        # Derive metrics from the confusion matrix
        company_tp = confusion[COMPANY][COMPANY]
        company_fp = confusion[PERSON][COMPANY]
        company_fn = confusion[COMPANY][PERSON] + invalid_by_true[COMPANY]

        person_tp = confusion[PERSON][PERSON]
        person_fp = confusion[COMPANY][PERSON]
        person_fn = confusion[PERSON][COMPANY] + invalid_by_true[PERSON]

        company_p, company_r, company_f1 = class_metrics(
            company_tp, company_fp, company_fn
        )
        person_p, person_r, person_f1 = class_metrics(person_tp, person_fp, person_fn)

        correct = company_tp + person_tp
        invalid = invalid_by_true[COMPANY] + invalid_by_true[PERSON]

        metrics = EvaluationMetrics(
            total_samples=total,
            correct=correct,
            accuracy=correct / total if total > 0 else 0.0,
            invalid_predictions=invalid,
            company_precision=company_p,
            company_recall=company_r,
            company_f1=company_f1,
            person_precision=person_p,
            person_recall=person_r,
            person_f1=person_f1,
            macro_f1=(company_f1 + person_f1) / 2,
            confusion=confusion,
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
