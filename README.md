# 📉 Explanation Drift Monitor

[![tests](https://github.com/shashwatsaxena571/explanation-drift-monitor/actions/workflows/tests.yml/badge.svg)](https://github.com/shashwatsaxena571/explanation-drift-monitor/actions/workflows/tests.yml)

**Your model's accuracy is stable. Its *reasoning* changed. This tool catches that — with zero labels.**

## The problem

Model monitoring tools (Evidently and friends) watch **data drift** and **prediction drift**. Nobody watches **attribution drift** — the shift in *which features the model relies on*.

It is a silent failure mode: accuracy looks fine on your dashboard while the model has quietly switched from reasoning on `income` to reasoning on `loan_amount` — because an upstream table broke, a unit changed, or a silent `fillna(0)` swallowed a feature. In credit risk this matters even more: default labels arrive **months late**, so accuracy monitoring is blind exactly when you need it. Attribution drift needs **no labels** — the alarm can fire the same day the pipeline breaks.

## How it works

```
[Reference attributions] ──┐
                            ├── [DriftDetector] ──► DriftReport (JSON)
[Current attributions]   ──┘
```

Three statistical lenses per feature, combined deliberately:

1. **PSI** (Population Stability Index) — industry-standard distribution shift, quantile-binned with Laplace smoothing so small samples don't explode it
2. **KS two-sample test** — non-parametric significance
3. **Spearman rank correlation** — did the *ranking* of important features change?

**Design decision:** a feature is flagged only when **PSI > 0.2 AND KS p < 0.05**. KS alone false-alarms on large samples (everything becomes "significant"); PSI alone false-alarms on small samples (noise inflates it). Requiring effect size *and* significance kills both failure modes.

## Quick start

```bash
pip install -r requirements.txt
python demo.py            # importance-swap scenario: income → loan_amount takeover
python -m pytest tests/ -v
```

`demo.py` simulates a reference window where `income` dominates and a current window where `loan_amount` has taken over — accuracy-based monitoring would see nothing; the attribution lens flags it instantly.

## Roadmap

- [x] Phase 1 — core detector (PSI + KS + rank correlation) + demo
- [ ] Phase 2 — [ExplainOps](https://github.com/shashwatsaxena571/explainops) store integration, sliding windows
- [ ] Phase 3 — alerting (Slack webhook), HTML report
- [ ] Phase 4 — evaluation on real datasets + research write-up (working paper: *label-free attribution drift detection*)

## Part of a bigger thesis

This is one piece of my XAI × Data Engineering stack:

- ⚙️ [explainops](https://github.com/shashwatsaxena571/explainops) — generate & version explanations as pipeline artifacts
- 📉 **explanation-drift-monitor** *(this repo)* — detect when explanations shift
- 🔍 [lineage-explanation-tracer](https://github.com/shashwatsaxena571/lineage-explanation-tracer) — trace the shift to the corrupt upstream table

I'm a **Data Engineer at IBM** and a **PhD scholar in Trustworthy & Explainable AI**. 📰 Weekly newsletter: [**Explainable Pipelines**](https://www.linkedin.com/newsletters/7488207829871304704/) · 💼 [LinkedIn](https://www.linkedin.com/in/saxena-shashwat/)

## License

MIT
