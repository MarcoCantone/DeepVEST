"""Mean and percentile-bootstrap 95% confidence interval of the per-case metrics (Table 2).

    python bootstrap_ci.py outputs/deepvest_test_metrics.csv

The input is the CSV written by ``evaluate.py`` (one row per test case, columns
``sensitivity``, ``matthews correlation coefficient`` and ``f1_score``; for binary
segmentation the F1 score equals the DSC). For each metric, the point estimate is the mean
across cases and the 95% CI is the percentile bootstrap of that mean: 10,000 case-level
resamples with replacement, taking the 2.5th and 97.5th percentiles of the resampled means.

Reproducing the paper exactly
-----------------------------
In the paper, the bootstrap of all Table 2 models was computed with a single random stream
(seed 34) and DeepVEST was preceded by three baseline models (3 models x 3 metrics = 9 draws).
``--burn-in 9`` (default) skips those draws so that the printed CIs match Table 2 exactly;
use ``--burn-in 0`` for a standalone bootstrap (CI bounds differ by about 0.001-0.002).
"""

import argparse

import numpy as np
import pandas as pd

# CSV column name -> label used in Table 2
COLUMN_LABELS = {
    "f1_score":                          "DSC",
    "matthews correlation coefficient":  "MCC",
    "sensitivity":                       "Sensitivity",
}
# order in which the metrics consume the random stream (as in the original analysis)
BOOTSTRAP_ORDER = ["sensitivity", "matthews correlation coefficient", "f1_score"]


def bootstrap_ci_mean(values, n_resamples, conf, rng):
    """Percentile bootstrap CI for the mean (point estimate = sample mean)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    idx = rng.integers(0, n, size=(n_resamples, n))     # case-level resampling
    boot_means = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means,
                           [(1 - conf) / 2 * 100, (1 + conf) / 2 * 100])
    return values.mean(), lo, hi


def main():
    parser = argparse.ArgumentParser(description="Percentile bootstrap 95% CI of per-case metrics.")
    parser.add_argument("metrics_csv", type=str, help="per-case metrics written by evaluate.py")
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=34)
    parser.add_argument("--conf", type=float, default=0.95)
    parser.add_argument("--burn-in", type=int, default=9,
                        help="bootstrap draws skipped before DeepVEST (9 reproduces the paper, 0 = standalone)")
    parser.add_argument("--output", type=str, default=None, help="optional CSV file for the results")
    args = parser.parse_args()

    df = pd.read_csv(args.metrics_csv, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]    # tolerate stray spaces
    n = len(df)

    rng = np.random.default_rng(args.seed)
    for _ in range(args.burn_in):
        rng.integers(0, n, size=(args.n_resamples, n))

    results = {}
    for col in BOOTSTRAP_ORDER:
        vals = df[col].dropna().to_numpy()
        results[col] = bootstrap_ci_mean(vals, args.n_resamples, args.conf, rng)

    print(f"DeepVEST, Duke test set (n = {n}), mean (bootstrap {int(args.conf * 100)}% CI):")
    rows = []
    for col, label in COLUMN_LABELS.items():
        mean, lo, hi = results[col]
        print(f"  {label:12s} {mean:.3f} ({lo:.3f}, {hi:.3f})")
        rows.append({"metric": label, "mean": round(mean, 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3)})

    if args.output:
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
