# LossLens — P&C Claims Fraud Triage & Loss Analytics

Insurance analytics with testable claims: a synthetic auto book (6,000
policies, 897 claims) with **six planted fraud rings** whose membership is
known ground truth — so fraud-detection performance is measured
(precision@k against planted labels), never anecdotal. Plus the classical
P&C toolkit: accident-year loss ratios and paid development triangles with
age-to-age factors.

## Measured results (seeded, reproducible, pinned by 8 tests)

```
referral queue precision@50 (share of top-50 that are truly fraudulent):
  random baseline          9.8%
  rules only              86%
  rules+graph             84%
  rules+graph+anomaly     84%
  all four (+ notes)     100%
```

Two honest findings the ablation surfaces:
1. **Adjuster-note signals add the most** on top of rules — the planted ring
   notes carry patterns (identical statements, no police report, estimate-
   before-inspection) that numeric features can't see. This is the argument
   for LLM note triage in a real SIU workflow; here the note scorer is a
   deterministic lexicon (testable), with a live-LLM slot documented.
2. **Adding graph/anomaly under equal weighting slightly DILUTED rules**
   (86→84%) before notes recovered it — a real lesson about naive score
   averaging that a fitted combiner would hide (and on synthetic data, a
   fitted combiner would just memorize the generator — deliberately avoided,
   documented in triage.py).

Loss analytics: AY loss ratios in a plausible band, cumulative triangles
non-decreasing with age-to-age factors ≥ 1 — all asserted by tests.

## What's synthetic and why that's the method

Real claims data is PII-bound and proprietary. The generator IS the
methodology: fraud rings share phones and repair shops, file within 60 days
of inception, cluster in low-visibility hours, at inflated severities —
and `true_label` exists for EVALUATION only, never as a feature (guarded by
test). Earning is simplified to policy-year allocation (documented; pro-rata
daily earning is the extension).

## Run

```bash
pip install pandas scikit-learn pytest
PYTHONPATH=src python src/losslens/generate.py   # build the book
PYTHONPATH=src pytest tests/ -q                  # 8 tests
```

```
src/losslens/generate.py      the planted-ring book generator
src/losslens/triage.py        4 signal families + ablation eval
src/losslens/loss_metrics.py  AY/CY loss ratios, triangles, ATA factors
docs/POWERBI_SPEC.md          report spec: triangle matrix, referral queue, ring graph
```
