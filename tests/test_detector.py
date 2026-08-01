import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from driftmon import DriftDetector

def _attr(imps, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame([{"feature": f, "attribution": v}
                         for f, i in zip("abc", imps)
                         for v in rng.normal(i, 0.02, 100)])

def test_stable_when_same():
    d = DriftDetector()
    assert d.compare(_attr([.5,.3,.2], 1), _attr([.5,.3,.2], 2))["verdict"] == "STABLE"

def test_detects_swap():
    d = DriftDetector()
    r = d.compare(_attr([.5,.3,.2], 1), _attr([.2,.3,.5], 2))
    assert r["verdict"] == "EXPLANATION DRIFT DETECTED"
