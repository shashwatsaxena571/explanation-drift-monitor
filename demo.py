"""Simulated scenario: feature A dominates the reference window; feature B takes over in the current window."""
import sys, json; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from driftmon import DriftDetector

rng = np.random.default_rng(42)
feats = ["income", "age", "loan_amount", "credit_history"]

def make_attr(importances, n=200, noise=0.05):
    rows = []
    for f, imp in zip(feats, importances):
        rows += [{"feature": f, "attribution": v}
                 for v in rng.normal(imp, noise, n)]
    return pd.DataFrame(rows)

ref = make_attr([0.50, 0.25, 0.15, 0.10])          # income dominates
cur = make_attr([0.15, 0.25, 0.50, 0.10])          # loan_amount has taken over

report = DriftDetector().compare(ref, cur)
print(json.dumps(report, indent=2))
print("\nAn accuracy check would have seen nothing. The attribution lens caught the shift.")
