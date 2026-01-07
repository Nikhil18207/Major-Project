"""
Statistical Significance Tests for Research Paper

Implements bootstrap confidence intervals, paired t-tests, and
significance testing for comparing model performance.

Authors: S. Nikhil, Dadhania Omkumar
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy import stats
from dataclasses import dataclass


@dataclass
class SignificanceResult:
    """Result of a significance test."""
    metric_name: str
    mean_a: float
    mean_b: float
    difference: float
    ci_lower: float
    ci_upper: float
    p_value: float
    is_significant: bool
    effect_size: float  # Cohen's d


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic: callable = np.mean,
    n_bootstrap: int = 10000,
    ci: float = 95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval.

    Args:
        data: Array of values
        statistic: Function to compute (default: mean)
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level (%)
        seed: Random seed

    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    np.random.seed(seed)

    point_estimate = statistic(data)
    bootstrapped = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrapped.append(statistic(sample))

    alpha = (100 - ci) / 2
    lower = np.percentile(bootstrapped, alpha)
    upper = np.percentile(bootstrapped, 100 - alpha)

    return point_estimate, lower, upper


def paired_bootstrap_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Paired bootstrap test for comparing two models.

    Args:
        scores_a: Per-sample scores from model A
        scores_b: Per-sample scores from model B
        n_bootstrap: Number of bootstrap samples
        seed: Random seed

    Returns:
        (mean_difference, p_value, ci_95)
    """
    np.random.seed(seed)

    n = len(scores_a)
    assert len(scores_b) == n, "Score arrays must have same length"

    # Observed difference
    observed_diff = np.mean(scores_a) - np.mean(scores_b)

    # Bootstrap
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        diff = np.mean(scores_a[indices]) - np.mean(scores_b[indices])
        bootstrap_diffs.append(diff)

    bootstrap_diffs = np.array(bootstrap_diffs)

    # Two-tailed p-value
    if observed_diff >= 0:
        p_value = 2 * np.mean(bootstrap_diffs <= 0)
    else:
        p_value = 2 * np.mean(bootstrap_diffs >= 0)

    # 95% CI
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)

    return observed_diff, p_value, (ci_lower, ci_upper)


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def compare_models(
    scores_ours: Dict[str, np.ndarray],
    scores_baseline: Dict[str, np.ndarray],
    model_name_ours: str = "Ours",
    model_name_baseline: str = "Baseline",
    n_bootstrap: int = 10000,
) -> List[SignificanceResult]:
    """
    Compare two models across multiple metrics with significance tests.

    Args:
        scores_ours: Dict of metric_name -> per-sample scores for our model
        scores_baseline: Dict of metric_name -> per-sample scores for baseline
        model_name_ours: Name of our model
        model_name_baseline: Name of baseline model
        n_bootstrap: Number of bootstrap samples

    Returns:
        List of SignificanceResult for each metric
    """
    results = []

    for metric_name in scores_ours.keys():
        if metric_name not in scores_baseline:
            continue

        ours = np.array(scores_ours[metric_name])
        baseline = np.array(scores_baseline[metric_name])

        # Bootstrap test
        diff, p_value, (ci_lower, ci_upper) = paired_bootstrap_test(
            ours, baseline, n_bootstrap=n_bootstrap
        )

        # Effect size
        effect = cohens_d(ours, baseline)

        result = SignificanceResult(
            metric_name=metric_name,
            mean_a=np.mean(ours),
            mean_b=np.mean(baseline),
            difference=diff,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            is_significant=p_value < 0.05,
            effect_size=effect,
        )
        results.append(result)

    return results


def format_significance_table(results: List[SignificanceResult]) -> pd.DataFrame:
    """Format significance results as a DataFrame."""
    rows = []
    for r in results:
        rows.append({
            'Metric': r.metric_name,
            'Ours': f"{r.mean_a:.4f}",
            'Baseline': f"{r.mean_b:.4f}",
            'Diff': f"{r.difference:+.4f}",
            '95% CI': f"[{r.ci_lower:.4f}, {r.ci_upper:.4f}]",
            'p-value': f"{r.p_value:.4f}" if r.p_value >= 0.0001 else "<0.0001",
            'Sig.': '***' if r.p_value < 0.001 else ('**' if r.p_value < 0.01 else ('*' if r.p_value < 0.05 else '')),
            "Cohen's d": f"{r.effect_size:.2f}",
        })
    return pd.DataFrame(rows)


def generate_latex_significance_table(results: List[SignificanceResult]) -> str:
    """Generate LaTeX table for significance results."""
    latex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Statistical significance of improvements. * p<0.05, ** p<0.01, *** p<0.001}",
        r"\label{tab:significance}",
        r"\begin{tabular}{l|cc|ccc}",
        r"\hline",
        r"\textbf{Metric} & \textbf{Ours} & \textbf{Baseline} & \textbf{Diff} & \textbf{95\% CI} & \textbf{p} \\",
        r"\hline",
    ]

    for r in results:
        sig = '***' if r.p_value < 0.001 else ('**' if r.p_value < 0.01 else ('*' if r.p_value < 0.05 else ''))
        p_str = f"{r.p_value:.3f}" if r.p_value >= 0.001 else "<0.001"
        latex.append(
            f"{r.metric_name} & {r.mean_a:.3f} & {r.mean_b:.3f} & "
            f"{r.difference:+.3f}{sig} & [{r.ci_lower:.3f}, {r.ci_upper:.3f}] & {p_str} \\\\"
        )

    latex.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return '\n'.join(latex)


# Interpretation guidelines
def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)

    # Simulate per-sample BLEU-4 scores
    ours_bleu4 = np.random.normal(0.17, 0.05, 1000)
    baseline_bleu4 = np.random.normal(0.13, 0.05, 1000)

    ours_rougel = np.random.normal(0.35, 0.08, 1000)
    baseline_rougel = np.random.normal(0.29, 0.08, 1000)

    scores_ours = {'BLEU-4': ours_bleu4, 'ROUGE-L': ours_rougel}
    scores_baseline = {'BLEU-4': baseline_bleu4, 'ROUGE-L': baseline_rougel}

    results = compare_models(scores_ours, scores_baseline)

    print("Statistical Significance Test Results")
    print("=" * 60)
    print(format_significance_table(results).to_string(index=False))

    print("\n" + generate_latex_significance_table(results))
