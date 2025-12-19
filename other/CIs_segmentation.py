import pandas as pd
import numpy as np
from scipy import stats

# Load your CSV file: 15 rows × 3 columns (sensitivity, mcc, dsc)
df = pd.read_csv(
    "/data/cantone/experiments/MRI/paper_experiments/seed34_AttentionUnet/metrics.csv")  # adjust the path if needed
confidence = 0.95
print("=== 95% Confidence Intervals (t-distribution) ===")
for column in df.columns:
    data = df[column].dropna().values
    mean = np.mean(data)
    sem = stats.sem(data)  # standard error of the mean
    t_value = stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    ci_low = mean - t_value * sem
    ci_high = mean + t_value * sem
    print(f"{column:12s} Mean = {mean:.3f}, 95% CI = ({ci_low:.3f}–{ci_high:.3f})")
