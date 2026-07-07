"""Synthetic P&C auto book with PLANTED fraud rings.

Real claims data is unobtainable (PII/proprietary), so the generator is the
methodology: defects and fraud are planted at known rates with known ring
membership, which turns detection claims into testable assertions —
precision@k against ground truth instead of anecdotes.

Planted structures:
  * 6 fraud RINGS: 4-9 claimants sharing phone numbers and repair shops,
    filing soon after policy inception, at higher severities;
  * staged-accident pattern: ring claims cluster in low-visibility hours;
  * opportunistic inflation: solo claims with severity padding (harder class);
  * honest book: everything else.
"""

from __future__ import annotations

import csv
import pathlib
import random
from datetime import date, timedelta

rng = random.Random(20260710)
OUT = pathlib.Path(__file__).resolve().parents[2] / "data"

N_POLICIES = 6_000
CLAIM_RATE = 0.14          # annual frequency
N_RINGS = 6
START = date(2024, 1, 1)
END = date(2025, 12, 31)


def d_between(a: date, b: date) -> date:
    return a + timedelta(days=rng.randint(0, (b - a).days))


def generate() -> dict:
    OUT.mkdir(exist_ok=True)
    states = ["NY", "NJ", "PA", "CT", "FL"]

    policies = []
    for i in range(N_POLICIES):
        eff = d_between(START, date(2025, 6, 30))
        policies.append({
            "policy_id": f"POL{100000+i}",
            "state": rng.choices(states, weights=[30, 20, 20, 10, 20])[0],
            "effective_date": eff.isoformat(),
            "annual_premium": round(rng.lognormvariate(7.0, 0.35), 2),  # ~$1.1k
            "vehicle_age": rng.randint(0, 15),
        })

    # ring infrastructure: shared identifiers
    rings = []
    for r in range(N_RINGS):
        rings.append({
            "ring_id": f"RING{r+1}",
            "phone": f"555-01{r:02d}",
            "shop": f"Shop {chr(65+r)} Auto Body",
            "members": rng.sample(range(N_POLICIES), rng.randint(4, 9)),
        })
    ring_of_policy = {}
    for ring in rings:
        for m in ring["members"]:
            ring_of_policy[m] = ring

    claims, notes = [], []
    cid = 50_000
    honest_shops = [f"{c} Collision" for c in
                    ("Maple", "Grand", "Union", "Bergen", "Astor", "Fulton",
                     "Sunrise", "Pioneer", "Summit", "Harbor")]

    for i, pol in enumerate(policies):
        eff = date.fromisoformat(pol["effective_date"])
        in_ring = i in ring_of_policy
        n_claims = 1 if in_ring else (1 if rng.random() < CLAIM_RATE else 0)
        for _ in range(n_claims):
            cid += 1
            if in_ring:
                ring = ring_of_policy[i]
                loss_dt = eff + timedelta(days=rng.randint(10, 60))   # soon after inception
                hour = rng.choice([1, 2, 3, 4, 23])                   # staged: low-visibility
                paid = round(rng.lognormvariate(9.3, 0.4), 2)         # ~$11k, inflated
                phone, shop = ring["phone"], ring["shop"]
                note = rng.choice([
                    "Claimant statement matches passenger statement word for word. No police report at scene. Repair estimate from shop received before inspection.",
                    "Injuries reported by all occupants despite minor bumper damage. Same clinic for all treatments. Claimant unreachable at listed address.",
                    "Second claim this quarter involving this repair facility. Photos show pre-existing rust at impact point. Tow initiated before FNOL call.",
                ])
                label = "ring"
            elif rng.random() < 0.06:
                loss_dt = d_between(max(eff, START), END)
                hour = rng.randint(6, 22)
                paid = round(rng.lognormvariate(8.9, 0.5), 2)         # padded
                phone = f"555-{rng.randint(2000, 9999)}"
                shop = rng.choice(honest_shops)
                note = "Estimate revised upward twice after initial appraisal. Additional parts claimed not consistent with photos. Claimant requested specific appraiser."
                label = "inflated"
            else:
                loss_dt = d_between(max(eff, START), END)
                hour = rng.randint(6, 22)
                paid = round(rng.lognormvariate(8.4, 0.7), 2)         # ~$4.5k honest
                phone = f"555-{rng.randint(2000, 9999)}"
                shop = rng.choice(honest_shops)
                note = rng.choice([
                    "Rear-end collision at signal, police report on file, photos consistent with damage. Standard repair authorized.",
                    "Weather-related single vehicle loss. Adjuster inspection matches estimate. No injuries reported.",
                    "Parking lot impact, third party accepted liability. Repair completed at network shop.",
                ])
                label = "honest"
            if loss_dt > END:
                loss_dt = END
            report_lag = rng.randint(0, 5) if label != "ring" else rng.randint(0, 1)
            claims.append({
                "claim_id": f"CLM{cid}",
                "policy_id": pol["policy_id"],
                "loss_date": loss_dt.isoformat(),
                "report_date": (loss_dt + timedelta(days=report_lag)).isoformat(),
                "loss_hour": hour,
                "paid_amount": paid,
                "claimant_phone": phone,
                "repair_shop": shop,
                "days_since_inception": (loss_dt - eff).days,
                "true_label": label,      # ground truth: stays OUT of model features
            })
            notes.append({"claim_id": f"CLM{cid}", "adjuster_note": note})

    for name, rows in (("policies", policies), ("claims", claims),
                       ("adjuster_notes", notes)):
        with open(OUT / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    n_ring = sum(1 for c in claims if c["true_label"] == "ring")
    n_infl = sum(1 for c in claims if c["true_label"] == "inflated")
    return {"claims": len(claims), "ring_claims": n_ring,
            "inflated_claims": n_infl, "rings": N_RINGS}


if __name__ == "__main__":
    print(generate())
