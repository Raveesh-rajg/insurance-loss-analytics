"""Classical P&C analytics: loss ratios and development triangles.

Paid loss ratios use a simplified annual premium allocation. They are not
incurred loss ratios: the synthetic book has no case reserves or IBNR.
"""

from __future__ import annotations

import pandas as pd


def loss_ratios(claims: pd.DataFrame, policies: pd.DataFrame, basis: str = "accident") -> pd.DataFrame:
    if basis not in {"accident", "calendar"}:
        raise ValueError("basis must be accident or calendar")
    c = claims.copy()
    c["year"] = pd.to_datetime(c.loss_date if basis == "accident" else c.report_date).dt.year
    p = policies.copy()
    p["policy_year"] = pd.to_datetime(p.effective_date).dt.year
    # earned premium proxy: full annual premium allocated to policy year
    # (simplification documented in README; real earning is pro-rata daily)
    prem = p.groupby("policy_year").annual_premium.sum()
    loss = c.groupby("year").paid_amount.sum()
    out = pd.DataFrame({"earned_premium_proxy": prem, "paid_loss": loss}).fillna(0)
    out["loss_ratio"] = out.paid_loss / out.earned_premium_proxy.replace(0, float("nan"))
    out.index.name = "year"
    return out


def development_triangle(claims: pd.DataFrame, max_lag_q: int = 6, as_of=None) -> pd.DataFrame:
    """Paid development triangle: accident quarter x development quarter.
    Payment timing simulated as report-date settlement (single payment) —
    the triangle mechanics (cumulative build, age-to-age factors) are what
    the project demonstrates; multi-payment patterns are the extension."""
    c = claims.copy()
    cutoff = pd.Timestamp(as_of) if as_of is not None else pd.to_datetime(c.report_date).max()
    c = c[pd.to_datetime(c.loss_date) <= cutoff].copy()
    c["acc_q"] = pd.PeriodIndex(pd.to_datetime(c.loss_date), freq="Q")
    origins = sorted(c.acc_q.unique())
    c = c[pd.to_datetime(c.report_date) <= cutoff].copy()
    c["dev_q"] = (
        pd.PeriodIndex(pd.to_datetime(c.report_date), freq="Q") - c.acc_q
    ).map(lambda x: x.n)
    if (c.dev_q < 0).any():
        raise ValueError("Report date precedes accident quarter")
    tri = c.pivot_table(index="acc_q", columns="dev_q", values="paid_amount", aggfunc="sum")
    tri = tri.reindex(index=origins, columns=range(max_lag_q)).fillna(0).cumsum(axis=1)
    valuation = cutoff.to_period("Q")
    for origin in tri.index:
        for lag in tri.columns:
            if origin + lag > valuation:
                tri.loc[origin, lag] = float("nan")
    return tri


def age_to_age(tri: pd.DataFrame) -> pd.Series:
    factors = {}
    for j in range(tri.shape[1] - 1):
        prev, nxt = tri.iloc[:, j], tri.iloc[:, j + 1]
        mask = (prev > 0) & nxt.notna() & prev.notna()
        factors[f"{j}->{j+1}"] = float(nxt[mask].sum() / prev[mask].sum()) if mask.any() else float("nan")
    return pd.Series(factors)
