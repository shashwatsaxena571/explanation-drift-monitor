"""Attribution drift detection through three lenses:
1. PSI (Population Stability Index) — industry standard for distribution shift
2. KS test — non-parametric two-sample significance test
3. Spearman rank correlation — did the importance ranking change?"""
import numpy as np
import pandas as pd
from scipy import stats


class DriftDetector:
    def __init__(self, psi_threshold: float = 0.2, ks_pvalue: float = 0.05,
                 rank_corr_threshold: float = 0.7):
        self.psi_threshold = psi_threshold
        self.ks_pvalue = ks_pvalue
        self.rank_corr_threshold = rank_corr_threshold

    @staticmethod
    def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
        """PSI < 0.1 stable | 0.1-0.2 moderate | > 0.2 significant shift.

        Two nuances that break naive PSI on small samples:
        1. Build bins from the reference *quantiles* (not equal-width) — otherwise
           narrow distributions leave empty bins in the tails.
        2. Laplace smoothing (+1 per bin) — an empty bin's log-ratio must not explode.
        """
        edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        r, _ = np.histogram(ref, bins=edges); c, _ = np.histogram(cur, bins=edges)
        r = (r + 1) / (r.sum() + bins); c = (c + 1) / (c.sum() + bins)
        return float(np.sum((c - r) * np.log(c / r)))

    def compare(self, ref_attr: pd.DataFrame, cur_attr: pd.DataFrame) -> dict:
        """ref/cur: DataFrames with columns [feature, attribution] — per-sample or per-run values."""
        report = {"features": {}, "drifted": []}
        for feat in ref_attr["feature"].unique():
            r = ref_attr.loc[ref_attr.feature == feat, "attribution"].values
            c = cur_attr.loc[cur_attr.feature == feat, "attribution"].values
            if len(r) < 2 or len(c) < 2:
                continue
            psi_val = self.psi(r, c)
            ks_stat, ks_p = stats.ks_2samp(r, c)
            # Design decision:
            # - KS alone: with large n, every tiny difference is "significant" — false alarms.
            # - PSI alone: with small n, sampling noise inflates it — false alarms.
            # - Require BOTH: effect size (PSI) AND significance (KS) before flagging.
            drifted = psi_val > self.psi_threshold and ks_p < self.ks_pvalue
            report["features"][feat] = {"psi": round(psi_val, 4),
                                        "ks_pvalue": round(float(ks_p), 4),
                                        "drifted": bool(drifted)}
            if drifted:
                report["drifted"].append(feat)

        ref_rank = ref_attr.groupby("feature")["attribution"].mean().rank()
        cur_rank = cur_attr.groupby("feature")["attribution"].mean().rank()
        common = ref_rank.index.intersection(cur_rank.index)
        rho = float(stats.spearmanr(ref_rank[common], cur_rank[common]).statistic)
        report["rank_correlation"] = round(rho, 4)
        report["ranking_shifted"] = rho < self.rank_corr_threshold
        report["verdict"] = ("EXPLANATION DRIFT DETECTED"
                             if report["drifted"] or report["ranking_shifted"]
                             else "STABLE")
        return report
