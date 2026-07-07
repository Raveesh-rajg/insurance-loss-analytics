"""Fraud triage: four signal families, combined and ABLATED.

The eval question is not "does the model work" but "what does each signal
family add" — precision@50 measured for rules alone, +graph, +anomaly,
+notes. Ground-truth labels exist because the generator planted them; the
labels never enter the features.

Signal families:
  R  rules      — early-inception filing, low-visibility hour, severity band
  G  graph      — shared-entity degree (same phone / repair shop across
                  claimants) computed as groupby counts; a full graph库 is
                  unnecessary for degree-1 features and this stays honest
  A  anomaly    — IsolationForest over numeric claim features
  N  notes      — lexicon scoring of adjuster notes (mock-LLM pattern:
                  deterministic, testable; a live LLM slots in one function)
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest

NOTE_FLAGS = [
    "word for word", "no police report", "before inspection", "same clinic",
    "unreachable", "before fnol", "pre-existing", "revised upward",
    "not consistent with photos", "requested specific appraiser",
    "injuries reported by all occupants",
]


def rule_score(claims: pd.DataFrame) -> pd.Series:
    s = (
        (claims.days_since_inception <= 60).astype(float) * 1.0
        + claims.loss_hour.isin([23, 0, 1, 2, 3, 4]).astype(float) * 1.0
        + (claims.paid_amount > claims.paid_amount.quantile(0.9)).astype(float) * 0.5
        + (pd.to_datetime(claims.report_date) - pd.to_datetime(claims.loss_date))
          .dt.days.le(1).astype(float) * 0.25
    )
    return s / s.max()


def graph_score(claims: pd.DataFrame) -> pd.Series:
    phone_deg = claims.groupby("claimant_phone")["policy_id"].transform("nunique")
    shop_deg = claims.groupby("repair_shop")["policy_id"].transform("nunique")
    shop_med = shop_deg.median()
    s = (phone_deg - 1).clip(lower=0) + ((shop_deg - shop_med).clip(lower=0) / shop_med)
    return (s / s.max()).fillna(0)


def anomaly_score(claims: pd.DataFrame, seed: int = 11) -> pd.Series:
    X = pd.DataFrame({
        "paid": claims.paid_amount,
        "dsi": claims.days_since_inception,
        "hour": claims.loss_hour,
    })
    iso = IsolationForest(n_estimators=200, random_state=seed, contamination="auto")
    raw = -iso.fit(X).score_samples(X)          # higher = more anomalous
    raw = pd.Series(raw, index=claims.index)
    return (raw - raw.min()) / (raw.max() - raw.min())


def note_score(notes: pd.DataFrame) -> pd.Series:
    txt = notes.set_index("claim_id").adjuster_note.str.lower()
    s = sum(txt.str.contains(f, regex=False).astype(float) for f in NOTE_FLAGS)
    return (s / max(s.max(), 1)).rename("note_score")


def build_scores(claims: pd.DataFrame, notes: pd.DataFrame) -> pd.DataFrame:
    out = claims[["claim_id", "true_label", "paid_amount"]].copy()
    out["rules"] = rule_score(claims)
    out["graph"] = graph_score(claims)
    out["anomaly"] = anomaly_score(claims)
    out = out.merge(note_score(notes).reset_index(), on="claim_id", how="left")
    out["note_score"] = out.note_score.fillna(0)
    # equal-weight combination — deliberately simple: the ablation table is
    # the argument, and a fitted meta-model over 4 features would overfit
    # the generator instead of demonstrating the method
    out["referral_score"] = out[["rules", "graph", "anomaly", "note_score"]].mean(axis=1)
    return out


def precision_at_k(scores: pd.DataFrame, cols: list[str], k: int = 50) -> float:
    s = scores.copy()
    s["combo"] = s[cols].mean(axis=1)
    top = s.nlargest(k, "combo")
    return float((top.true_label != "honest").mean())


def ablation(scores: pd.DataFrame, k: int = 50) -> dict[str, float]:
    return {
        "rules_only": precision_at_k(scores, ["rules"], k),
        "rules+graph": precision_at_k(scores, ["rules", "graph"], k),
        "rules+graph+anomaly": precision_at_k(scores, ["rules", "graph", "anomaly"], k),
        "all_four": precision_at_k(scores, ["rules", "graph", "anomaly", "note_score"], k),
        "baseline_random": float((scores.true_label != "honest").mean()),
    }
