"""Shared, dependency-free evaluation metrics.

Pure functions used by the task evaluators (``src/*/evaluation.py``) and the
vLLM evaluation engine (``src/vllm/evaluation.py``).
"""


def normalize_text(value: str) -> str:
    """Normalize text for exact-match style comparisons.

    Args:
        value: Raw string value

    Returns:
        Normalized value (lowercase, stripped)
    """
    return value.lower().strip()


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein (edit) distance between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Minimum number of single-character edits to transform s1 into s2
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Compute normalised Levenshtein similarity between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Similarity score between 0.0 and 1.0 (1.0 = identical)
    """
    if not s1 and not s2:
        return 1.0

    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0

    return 1.0 - (levenshtein_distance(s1, s2) / max_len)


def token_f1(predicted: str, ground_truth: str) -> dict[str, float]:
    """Compute token-level precision, recall, and F1 score.

    Tokenises both strings by whitespace and computes overlap metrics.

    Args:
        predicted: Predicted string
        ground_truth: Ground truth string

    Returns:
        Dictionary with 'precision', 'recall', and 'f1' scores
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
    f1 = (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )

    return {'precision': precision, 'recall': recall, 'f1': f1}


def class_metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 from per-class counts.

    Args:
        tp: True positives for the class
        fp: False positives for the class
        fn: False negatives for the class

    Returns:
        Tuple of (precision, recall, f1), each 0.0 on zero denominator
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def similarity_bucket(similarity: float) -> str:
    """Assign a similarity score to a reporting bucket.

    Args:
        similarity: Similarity score between 0.0 and 1.0

    Returns:
        Bucket label string
    """
    if similarity == 1.0:
        return '1.00 (exact)'
    if similarity >= 0.9:
        return '0.90-0.99'
    if similarity >= 0.8:
        return '0.80-0.89'
    if similarity >= 0.7:
        return '0.70-0.79'
    if similarity >= 0.5:
        return '0.50-0.69'
    return '< 0.50'
