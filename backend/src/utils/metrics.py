"""
Evaluation Metrics for Report Generation

Implements BLEU, ROUGE, and other metrics for evaluating
generated radiology reports.
"""

import numpy as np
from typing import List, Dict, Union
from collections import Counter
import math
from loguru import logger

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logger.warning("rouge_score not installed. ROUGE metrics will be unavailable.")

try:
    import nltk
    nltk.download('punkt', quiet=True)
    from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("nltk not installed. BLEU metrics will be unavailable.")


def compute_bleu(
    predictions: List[str],
    references: List[str],
    max_n: int = 4,
) -> Dict[str, float]:
    """
    Compute BLEU scores for generated reports.

    Args:
        predictions: List of generated texts
        references: List of reference texts
        max_n: Maximum n-gram order (default: 4 for BLEU-4)

    Returns:
        Dictionary with BLEU scores (BLEU-1 through BLEU-4)
    """
    if not NLTK_AVAILABLE:
        return {f"bleu_{i}": 0.0 for i in range(1, max_n + 1)}

    smoothing = SmoothingFunction()
    bleu_scores = {f"bleu_{i}": [] for i in range(1, max_n + 1)}

    for pred, ref in zip(predictions, references):
        # Tokenize
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()

        # Calculate BLEU for different n-gram orders
        for n in range(1, max_n + 1):
            weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
            try:
                score = sentence_bleu(
                    [ref_tokens],
                    pred_tokens,
                    weights=weights,
                    smoothing_function=smoothing.method1,
                )
                bleu_scores[f"bleu_{n}"].append(score)
            except Exception:
                bleu_scores[f"bleu_{n}"].append(0.0)

    # Average scores
    return {k: np.mean(v) if v else 0.0 for k, v in bleu_scores.items()}


def compute_rouge(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Compute ROUGE scores for generated reports.

    Args:
        predictions: List of generated texts
        references: List of reference texts

    Returns:
        Dictionary with ROUGE-1, ROUGE-2, and ROUGE-L F1 scores
    """
    if not ROUGE_AVAILABLE:
        return {
            "rouge_1": 0.0,
            "rouge_2": 0.0,
            "rouge_l": 0.0,
        }

    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True,
    )

    rouge_scores = {
        "rouge_1": [],
        "rouge_2": [],
        "rouge_l": [],
    }

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge_scores["rouge_1"].append(scores["rouge1"].fmeasure)
        rouge_scores["rouge_2"].append(scores["rouge2"].fmeasure)
        rouge_scores["rouge_l"].append(scores["rougeL"].fmeasure)

    # Average scores
    return {k: np.mean(v) if v else 0.0 for k, v in rouge_scores.items()}


def compute_meteor(
    predictions: List[str],
    references: List[str],
) -> float:
    """
    Compute METEOR score for generated reports.

    METEOR considers synonyms and stemming for more semantic matching.
    """
    try:
        from nltk.translate.meteor_score import meteor_score
        nltk.download('wordnet', quiet=True)

        scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()
            score = meteor_score([ref_tokens], pred_tokens)
            scores.append(score)

        return np.mean(scores) if scores else 0.0

    except Exception as e:
        logger.warning(f"METEOR computation failed: {e}")
        return 0.0


def compute_cider(
    predictions: List[str],
    references: List[str],
) -> float:
    """
    Compute CIDEr score (simplified version).

    CIDEr is specifically designed for image captioning evaluation.
    """
    # Simplified CIDEr-D implementation
    def get_ngrams(text: str, n: int) -> Counter:
        tokens = text.lower().split()
        return Counter([
            tuple(tokens[i:i+n])
            for i in range(len(tokens) - n + 1)
        ])

    def compute_tf(ngrams: Counter) -> Dict:
        total = sum(ngrams.values())
        return {ng: count / total for ng, count in ngrams.items()} if total > 0 else {}

    # Compute document frequency for IDF
    n = 4  # Use 4-grams
    doc_freq = Counter()

    for ref in references:
        ngrams = get_ngrams(ref, n)
        doc_freq.update(set(ngrams.keys()))

    num_docs = len(references)

    # Compute CIDEr for each prediction
    scores = []
    for pred, ref in zip(predictions, references):
        pred_ngrams = get_ngrams(pred, n)
        ref_ngrams = get_ngrams(ref, n)

        pred_tf = compute_tf(pred_ngrams)
        ref_tf = compute_tf(ref_ngrams)

        # Compute TF-IDF weighted cosine similarity
        common = set(pred_tf.keys()) & set(ref_tf.keys())

        if not common:
            scores.append(0.0)
            continue

        numerator = sum(
            pred_tf[ng] * ref_tf[ng] * math.log((num_docs + 1) / (doc_freq[ng] + 1)) ** 2
            for ng in common
        )

        pred_norm = sum(
            (pred_tf[ng] * math.log((num_docs + 1) / (doc_freq.get(ng, 0) + 1))) ** 2
            for ng in pred_tf
        )
        ref_norm = sum(
            (ref_tf[ng] * math.log((num_docs + 1) / (doc_freq.get(ng, 0) + 1))) ** 2
            for ng in ref_tf
        )

        denominator = math.sqrt(pred_norm * ref_norm) if pred_norm * ref_norm > 0 else 1

        scores.append(numerator / denominator)

    return np.mean(scores) * 10 if scores else 0.0  # Scale by 10 for readability


def compute_clinical_f1(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    IMPROVED: Compute Clinical Entity F1 Score with Negation Awareness.

    This metric evaluates the accuracy of clinical findings mentioned in
    generated reports, which is more clinically relevant than BLEU/ROUGE.

    Features:
    - Extracts 22 common chest X-ray findings
    - Handles negation (e.g., "no pneumonia" vs "pneumonia")
    - Computes precision, recall, and F1 for clinical entities

    Args:
        predictions: List of generated texts
        references: List of reference texts

    Returns:
        Dictionary with clinical_precision, clinical_recall, clinical_f1
    """
    # Clinical findings to track (expanded list)
    clinical_findings = [
        'cardiomegaly', 'pneumonia', 'effusion', 'edema', 'consolidation',
        'atelectasis', 'pneumothorax', 'infiltrate', 'mass', 'nodule',
        'pleural thickening', 'opacity', 'fibrosis', 'fracture',
        'pleural effusion', 'pulmonary edema', 'emphysema', 'hernia',
        'calcification', 'scoliosis', 'tortuous aorta', 'lymphadenopathy'
    ]

    # Negation patterns
    negation_patterns = [
        'no ', 'no evidence of ', 'without ', 'negative for ',
        'absence of ', 'absent ', 'not ', 'denies ', 'ruled out ',
        'no acute ', 'no significant ', 'clear', 'unremarkable', 'normal'
    ]

    def extract_findings(text: str) -> tuple:
        """Extract positive and negative findings from text."""
        text_lower = text.lower()
        positive = set()
        negative = set()

        for finding in clinical_findings:
            if finding in text_lower:
                # Check if negated
                finding_pos = text_lower.find(finding)
                context_start = max(0, finding_pos - 40)
                context = text_lower[context_start:finding_pos]

                is_negated = any(neg in context for neg in negation_patterns)
                if is_negated:
                    negative.add(finding)
                else:
                    positive.add(finding)

        return positive, negative

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_critical_errors = 0

    for pred, ref in zip(predictions, references):
        pred_pos, pred_neg = extract_findings(pred)
        ref_pos, ref_neg = extract_findings(ref)

        # True positives: correctly identified positive findings
        tp = len(pred_pos & ref_pos) + len(pred_neg & ref_neg)

        # False positives: predicted positive but not in reference
        fp = len(pred_pos - ref_pos) + len(pred_neg - ref_neg)

        # False negatives: in reference but not predicted
        fn = len(ref_pos - pred_pos) + len(ref_neg - pred_neg)

        # Critical errors: saying positive when actually negative (dangerous!)
        critical = len(pred_pos & ref_neg)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_critical_errors += critical

    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "clinical_precision": precision,
        "clinical_recall": recall,
        "clinical_f1": f1,
        "clinical_critical_errors": total_critical_errors,
    }


def compute_radgraph_f1(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    IMPROVED: Compute RadGraph F1 Score.

    RadGraph extracts entities and relations from radiology reports
    and computes F1 based on entity/relation matching.

    Note: Requires radgraph package. Falls back to entity-based F1 if unavailable.

    Args:
        predictions: List of generated texts
        references: List of reference texts

    Returns:
        Dictionary with radgraph_f1 and related metrics
    """
    try:
        # Try to use official RadGraph scorer
        from radgraph import F1RadGraph
        scorer = F1RadGraph(reward_level="all")
        score, _, _ = scorer(hyps=predictions, refs=references)
        return {
            "radgraph_f1": score,
            "radgraph_available": True,
        }
    except ImportError:
        # Fallback: Compute entity-based F1 (simplified RadGraph)
        logger.info("RadGraph not installed. Using simplified entity F1.")

        # Use clinical F1 as approximation
        clinical_metrics = compute_clinical_f1(predictions, references)
        return {
            "radgraph_f1": clinical_metrics["clinical_f1"],
            "radgraph_available": False,
        }
    except Exception as e:
        logger.warning(f"RadGraph computation failed: {e}")
        return {
            "radgraph_f1": 0.0,
            "radgraph_available": False,
        }


def compute_bertscore(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    IMPROVED: Compute BERTScore for semantic similarity.

    BERTScore uses contextualized embeddings to measure
    semantic similarity between generated and reference texts.

    Args:
        predictions: List of generated texts
        references: List of reference texts

    Returns:
        Dictionary with bertscore_precision, bertscore_recall, bertscore_f1
    """
    try:
        from bert_score import score
        P, R, F1 = score(predictions, references, lang="en", rescale_with_baseline=True)
        return {
            "bertscore_precision": P.mean().item(),
            "bertscore_recall": R.mean().item(),
            "bertscore_f1": F1.mean().item(),
        }
    except ImportError:
        logger.info("bert_score not installed. Skipping BERTScore.")
        return {
            "bertscore_precision": 0.0,
            "bertscore_recall": 0.0,
            "bertscore_f1": 0.0,
        }
    except Exception as e:
        logger.warning(f"BERTScore computation failed: {e}")
        return {
            "bertscore_precision": 0.0,
            "bertscore_recall": 0.0,
            "bertscore_f1": 0.0,
        }


def compute_metrics(
    predictions: List[str],
    references: List[str],
    include_all: bool = True,
    include_clinical: bool = True,
) -> Dict[str, float]:
    """
    Compute all metrics for report generation evaluation.

    IMPROVED: Now includes clinical metrics for medical AI evaluation.

    Args:
        predictions: List of generated texts
        references: List of reference texts
        include_all: Whether to include all NLG metrics
        include_clinical: Whether to include clinical metrics (F1, RadGraph)

    Returns:
        Dictionary with all metric scores
    """
    metrics = {}

    # BLEU scores
    bleu = compute_bleu(predictions, references)
    metrics.update(bleu)

    # ROUGE scores
    rouge = compute_rouge(predictions, references)
    metrics.update(rouge)

    if include_all:
        # METEOR
        metrics["meteor"] = compute_meteor(predictions, references)

        # CIDEr
        metrics["cider"] = compute_cider(predictions, references)

    # IMPROVED: Clinical metrics for medical AI evaluation
    if include_clinical:
        # Clinical Entity F1 (negation-aware)
        clinical = compute_clinical_f1(predictions, references)
        metrics.update(clinical)

        # RadGraph F1 (if available)
        radgraph = compute_radgraph_f1(predictions, references)
        metrics.update(radgraph)

    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Metrics"):
    """Pretty print metrics."""
    print(f"\n{'=' * 50}")
    print(f"{title:^50}")
    print(f"{'=' * 50}")

    for name, value in metrics.items():
        print(f"  {name.upper():<15}: {value:.4f}")

    print(f"{'=' * 50}\n")


class MetricsTracker:
    """Track metrics over training epochs."""

    def __init__(self):
        self.history = {}

    def update(self, metrics: Dict[str, float], epoch: int):
        """Add metrics for an epoch."""
        for name, value in metrics.items():
            if name not in self.history:
                self.history[name] = []
            self.history[name].append((epoch, value))

    def get_best(self, metric_name: str) -> tuple:
        """Get best value and epoch for a metric."""
        if metric_name not in self.history:
            return 0.0, 0

        values = self.history[metric_name]
        best = max(values, key=lambda x: x[1])
        return best[1], best[0]

    def get_history(self, metric_name: str) -> List[float]:
        """Get history of values for a metric."""
        if metric_name not in self.history:
            return []
        return [v for _, v in self.history[metric_name]]


if __name__ == "__main__":
    # Test metrics
    predictions = [
        "The lungs are clear. No acute cardiopulmonary abnormality.",
        "There is mild cardiomegaly. No pleural effusion seen.",
    ]
    references = [
        "The lungs are clear without focal consolidation. No acute cardiopulmonary process.",
        "Mild cardiomegaly is present. No pleural effusion or pneumothorax.",
    ]

    metrics = compute_metrics(predictions, references)
    print_metrics(metrics)
