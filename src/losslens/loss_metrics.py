"""Classical P&C analytics: loss ratios and development triangles.

Accident-year vs calendar-year loss ratio is the definitional interview
question in insurance analytics; both are computed here and the difference
is asserted in tests on the synthetic book.
"""

from __future__ import annotations

import pandas as pd


def loss_ratios(claims: pd.DataFrame, policies: pd.DataFrame) -> pd.DataFrame:
    c = claims.copy()
    c["accident_year"] = pd.to_datetime(c.loss_date).dt.year
    p = policies.copy()
    p["policy_year"] = pd.to_datetime(p.effective_date).dt.year
    # earned premium proxy: full annual premium allocated to policy year
    # (simplification documented in README; real earning is pro-rata daily)
    prem = p.groupby("policy_year").annual_premium.sum()
    loss = c.groupby("accident_year").paid_amount.sum()
    out = pd.DataFrame({"earned_premium": prem, "incurred_loss": loss}).fillna(0)
    out["loss_ratio"] = out.incurred_loss / out.earned_premium
    return out


def development_triangle(claims: pd.DataFrame, max_lag_q: int = 6) -> pd.DataFrame:
    """Paid development triangle: accident quarter x development quarter.
    Payment timing simulated as report-date settlement (single payment) —
    the triangle mechanics (cumulative build, age-to-age factors) are what
    the project demonstrates; multi-payment patterns are the extension."""
    c = claims.copy()
    c["acc_q"] = pd.PeriodIndex(pd.to_datetime(c.loss_date), freq="Q")
    c["dev_q"] = (
        pd.PeriodIndex(pd.to_datetime(c.report_date), freq="Q") - c.acc_q
    ).map(lambda x: x.n)
    tri = (c.pivot_table(index="acc_q", columns="dev_q", values="paid_amount",
                         aggfunc="sum").fillna(0).cumsum(axis=1))
    return tri.iloc[:, :max_lag_q]


def age_to_age(tri: pd.DataFrame) -> pd.Series:
    factors = {}
    for j in range(tri.shape[1] - 1):
        prev, nxt = tri.iloc[:, j], tri.iloc[:, j + 1]
        mask = prev > 0
        factors[f"{j}->{j+1}"] = float(nxt[mask].sum() / prev[mask].sum())
    return pd.Series(factors)
