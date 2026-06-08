"""paired t-test 기반 통계적 신뢰도 계산.

whatif_effect.py 동일 방식:
  ttest_1samp(D_list, popmean=0.0)
  95% CI via student-t distribution
  verdict: improved / worsened / unchanged
"""

from __future__ import annotations

try:
    from scipy.stats import ttest_1samp
    from scipy.stats import t as student_t
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _verdict(mean_d: float) -> str:
    if mean_d < -1e-9:
        return "improved"
    if mean_d > 1e-9:
        return "worsened"
    return "unchanged"


def compute_paired_stats(deltas: list[float]) -> dict:
    """
    paired t-test → p-value, 95% CI, verdict.

    Args:
        deltas: D_i = whatif_i − baseline_i 목록 (N=30)

    Returns:
        {mean_delta, ci_lo, ci_hi, paired_t_p, paired_n, verdict}
    """
    n = len(deltas)
    if n == 0:
        return {
            "mean_delta": 0.0,
            "ci_lo": 0.0,
            "ci_hi": 0.0,
            "paired_t_p": None,
            "paired_n": 0,
            "verdict": "unchanged",
        }

    mean_d = sum(deltas) / n
    std_dev = (sum((x - mean_d) ** 2 for x in deltas) / max(n - 1, 1)) ** 0.5

    p_val: float | None = None
    ci_lo, ci_hi = mean_d, mean_d

    if _HAS_SCIPY and n >= 2:
        res = ttest_1samp(deltas, popmean=0.0)
        p_val = float(res.pvalue)
        if std_dev > 0:
            tcrit = float(student_t.ppf(0.975, n - 1))
            margin = tcrit * std_dev / (n ** 0.5)
            ci_lo = mean_d - margin
            ci_hi = mean_d + margin

    return {
        "mean_delta": round(mean_d, 6),
        "ci_lo": round(ci_lo, 6),
        "ci_hi": round(ci_hi, 6),
        "paired_t_p": round(p_val, 6) if p_val is not None else None,
        "paired_n": n,
        "verdict": _verdict(mean_d),
    }
