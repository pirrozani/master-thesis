"""Evaluation engine scoring dataset splits through the vLLM endpoint.

Task-specific CLIs live in ``scripts/vllm/<task>/evaluate.py``; each passes
its ``TaskSpec`` into :func:`run` / :func:`main`. Uses the stdlib client so
it works in the uv environment without the container-only dependencies.
"""

import argparse
import json
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

from datasets import load_from_disk

from src.address.models import Address
from src.entity_name.classification.models import VALID_LABELS
from src.utils import (
    class_metrics,
    levenshtein_similarity,
    normalize_text,
    save_file,
    similarity_bucket,
    token_f1,
)
from src.vllm.client import get_base_url, list_models, post_chat_completion
from src.vllm.tasks import TaskSpec, parse_address, parse_entity_name, parse_entity_type


def compare_address(sample: dict[str, Any], completion: str) -> dict[str, Any]:
    """Compare one address prediction to its ground truth."""
    predicted = Address(**parse_address(completion))
    ground_truth = Address(
        **{name: sample.get(name, '') for name in Address.model_fields}
    )
    fields = {
        name: normalize_text(getattr(predicted, name))
        == normalize_text(getattr(ground_truth, name))
        for name in Address.model_fields
    }

    return {
        'predicted': predicted.to_pipe(),
        'ground_truth': ground_truth.to_pipe(),
        'exact_match': all(fields.values()),
        'field_correct': fields,
    }


def compare_extraction(sample: dict[str, Any], completion: str) -> dict[str, Any]:
    """Compare one entity-name prediction to its ground truth."""
    pred_norm = normalize_text(parse_entity_name(completion))
    truth_norm = normalize_text(sample.get('cleaned_name', ''))
    tok_f1 = token_f1(pred_norm, truth_norm)

    return {
        'predicted': parse_entity_name(completion),
        'ground_truth': sample.get('cleaned_name', ''),
        'exact_match': pred_norm == truth_norm,
        'levenshtein_similarity': levenshtein_similarity(pred_norm, truth_norm),
        'token_precision': tok_f1['precision'],
        'token_recall': tok_f1['recall'],
        'token_f1': tok_f1['f1'],
    }


def compare_classification(sample: dict[str, Any], completion: str) -> dict[str, Any]:
    """Compare one entity-type prediction to its ground truth."""
    predicted = parse_entity_type(completion)
    ground_truth = normalize_text(sample.get('label', ''))
    is_valid = predicted in VALID_LABELS

    return {
        'predicted': predicted,
        'ground_truth': ground_truth,
        'valid': is_valid,
        'correct': is_valid and predicted == ground_truth,
    }


def summarize_address(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate address metrics."""
    total = len(comparisons)
    field_names = list(Address.model_fields)
    field_correct = {
        name: sum(1 for c in comparisons if c['field_correct'][name])
        for name in field_names
    }
    exact = sum(1 for c in comparisons if c['exact_match'])

    summary = {
        'total_samples': total,
        'exact_match': exact,
        'exact_match_rate': exact / total if total else 0.0,
    }
    for name in field_names:
        key = 'zip_accuracy' if name == 'zip_code' else f'{name}_accuracy'
        summary[key] = field_correct[name] / total if total else 0.0
    summary['field_accuracy'] = (
        sum(field_correct.values()) / (total * len(field_names)) if total else 0.0
    )
    summary['field_correct'] = field_correct
    return summary


def summarize_extraction(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate entity-name extraction metrics."""
    total = len(comparisons)
    exact = sum(1 for c in comparisons if c['exact_match'])
    buckets: dict[str, int] = {}
    for c in comparisons:
        bucket = similarity_bucket(c['levenshtein_similarity'])
        buckets[bucket] = buckets.get(bucket, 0) + 1

    def avg(key: str) -> float:
        return fmean(c[key] for c in comparisons) if comparisons else 0.0

    return {
        'total_samples': total,
        'exact_match': exact,
        'exact_match_rate': exact / total if total else 0.0,
        'avg_levenshtein_similarity': avg('levenshtein_similarity'),
        'avg_token_precision': avg('token_precision'),
        'avg_token_recall': avg('token_recall'),
        'avg_token_f1': avg('token_f1'),
        'similarity_buckets': buckets,
    }


def summarize_classification(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate entity-type classification metrics."""
    total = len(comparisons)
    confusion = {t: {p: 0 for p in VALID_LABELS} for t in VALID_LABELS}
    invalid_by_true = dict.fromkeys(VALID_LABELS, 0)

    for c in comparisons:
        gt = c['ground_truth']
        if gt not in VALID_LABELS:
            continue
        if c['valid']:
            confusion[gt][c['predicted']] += 1
        else:
            invalid_by_true[gt] += 1

    per_class = {}
    for label in VALID_LABELS:
        others = [other for other in VALID_LABELS if other != label]
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in others)
        fn = sum(confusion[label][other] for other in others) + invalid_by_true[label]
        per_class[label] = class_metrics(tp, fp, fn)

    correct = sum(confusion[label][label] for label in VALID_LABELS)
    summary = {
        'total_samples': total,
        'correct': correct,
        'accuracy': correct / total if total else 0.0,
        'invalid_predictions': sum(invalid_by_true.values()),
    }
    for label in VALID_LABELS:
        precision, recall, f1 = per_class[label]
        summary[f'{label}_precision'] = precision
        summary[f'{label}_recall'] = recall
        summary[f'{label}_f1'] = f1
    summary['macro_f1'] = fmean(per_class[label][2] for label in VALID_LABELS)
    summary['confusion'] = confusion
    return summary


COMPARE = {
    'address': compare_address,
    'extraction': compare_extraction,
    'classification': compare_classification,
}

SUMMARIZE = {
    'address': summarize_address,
    'extraction': summarize_extraction,
    'classification': summarize_classification,
}


def run(spec: TaskSpec, args: argparse.Namespace) -> dict[str, Any]:
    """Run the bounded vLLM evaluation for one task.

    Args:
        spec: Task spec selecting the adapter, prompt, and defaults
        args: Parsed CLI arguments from :func:`build_parser`

    Returns:
        Result dict with run settings and aggregated metrics
    """
    data_path = Path(args.data_path or spec.default_data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f'Dataset split not found at {data_path}. Restore or regenerate '
            'processed data before running the vLLM evaluation.'
        )

    vllm_model = args.vllm_model or spec.vllm_model
    max_tokens = args.max_tokens or spec.batch_max_tokens

    if not args.skip_model_check:
        served_models = list_models(args.base_url, args.timeout)
        if vllm_model not in served_models:
            models = ', '.join(sorted(served_models))
            raise RuntimeError(
                f"vLLM model '{vllm_model}' is not served. Available: {models}"
            )

    dataset = load_from_disk(str(data_path))
    if args.start < 0:
        raise ValueError('--start must be >= 0')
    if args.limit is not None and args.limit < 0:
        raise ValueError('--limit must be >= 0')
    if args.progress_every < 0:
        raise ValueError('--progress-every must be >= 0')

    limit = None if args.limit == 0 else args.limit
    end = len(dataset) if limit is None else min(args.start + limit, len(dataset))
    if args.start >= end:
        raise ValueError(
            f'No samples selected: start={args.start}, limit={args.limit}, '
            f'dataset_size={len(dataset)}'
        )

    selected = dataset.select(range(args.start, end))
    compare = COMPARE[spec.key]
    comparisons = []
    predictions = []

    for index, sample in enumerate(selected, start=args.start):
        text = sample.get(spec.input_field, '')
        completion = post_chat_completion(
            base_url=args.base_url,
            model=vllm_model,
            messages=spec.create_messages(text),
            max_tokens=max_tokens,
            timeout=args.timeout,
        )
        comparison = compare(sample, completion)
        comparisons.append(comparison)

        if args.predictions_output:
            predictions.append(
                {
                    'index': index,
                    'input': text,
                    'raw_completion': completion,
                    **comparison,
                }
            )

        if args.progress_every and (len(comparisons) % args.progress_every == 0):
            print(
                f'Processed {len(comparisons)}/{len(selected)} samples', file=sys.stderr
            )

    result = {
        'task': spec.key,
        'vllm_model': vllm_model,
        'data_path': str(data_path),
        'start': args.start,
        'limit': limit,
        'max_tokens': max_tokens,
        'metrics': SUMMARIZE[spec.key](comparisons),
    }

    if args.predictions_output:
        save_file(
            json.dumps(predictions, indent=2, ensure_ascii=False),
            args.predictions_output,
        )

    return result


def build_parser(spec: TaskSpec) -> argparse.ArgumentParser:
    """Build the CLI parser for one task's evaluate script.

    Args:
        spec: Task spec used for the CLI description and defaults

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description=(
            f"Score the '{spec.vllm_model}' adapter served by vLLM on a dataset split."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help=f'Dataset split path. Defaults to {spec.default_data_path}.',
    )
    parser.add_argument(
        '--base-url',
        type=str,
        default=get_base_url(),
        help='OpenAI-compatible vLLM base URL. Defaults to VLLM_BASE_URL (.env).',
    )
    parser.add_argument(
        '--vllm-model',
        type=str,
        default=None,
        help='Override the vLLM model/LoRA name used in the request',
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=None,
        help='Override max generated tokens. Defaults to batch evaluator budget.',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=50,
        help='Maximum number of samples to score. Use 0 to score all samples.',
    )
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help='Start offset within the dataset split',
    )
    parser.add_argument(
        '--timeout',
        type=float,
        default=120.0,
        help='HTTP timeout in seconds per request',
    )
    parser.add_argument(
        '--progress-every',
        type=int,
        default=10,
        help='Print progress every N samples. Use 0 to disable.',
    )
    parser.add_argument(
        '--skip-model-check',
        action='store_true',
        help='Skip querying /models before scoring',
    )
    parser.add_argument(
        '--predictions-output',
        type=str,
        default=None,
        help='Optional local JSON file for raw predictions and comparisons',
    )
    return parser


def main(spec: TaskSpec) -> None:
    """Run one task's evaluate CLI.

    Args:
        spec: Task spec for the wrapping script
    """
    args = build_parser(spec).parse_args()

    try:
        result = run(spec, args)
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)

    print(json.dumps(result, indent=2, ensure_ascii=False))
