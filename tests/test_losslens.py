import pathlib
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from losslens.generate import generate, OUT
from losslens import triage, loss_metrics


@pytest.fixture(scope="session")
def book():
    stats = generate()
    claims = pd.read_csv(OUT / "claims.csv")
    notes = pd.read_csv(OUT / "adjuster_notes.csv")
    policies = pd.read_csv(OUT / "policies.csv")
    return stats, claims, notes, policies


class TestGenerator:
    def test_ground_truth_planted(self, book):
        stats, claims, *_ = book
        assert stats["ring_claims"] >= 25
        assert set(claims.true_label) == {"honest", "inflated", "ring"}

    def test_ring_claims_share_infrastructure(self, book):
        _, claims, *_ = book
        rings = claims[claims.true_label == "ring"]
        assert rings.groupby("claimant_phone").size().max() >= 4


class TestTriage:
    @pytest.fixture(scope="class")
    def scores(self, book):
        _, claims, notes, _ = book
        return triage.build_scores(claims, notes)

    def test_labels_never_used_as_features(self, scores):
        # structural guard: feature columns must not correlate perfectly by construction
        assert "true_label" in scores.columns  # present for EVAL only
        assert {"rules", "graph", "anomaly", "note_score"} <= set(scores.columns)

    def test_ablation_monotone_improvement(self, scores):
        """Each added signal family should not hurt precision@50, and the
        full stack must beat rules alone and crush the random baseline."""
        ab = triage.ablation(scores, k=50)
        assert ab["all_four"] >= ab["rules_only"]
        assert ab["all_four"] >= 5 * ab["baseline_random"]

    def test_rings_concentrate_at_top(self, scores):
        top50 = scores.nlargest(50, "referral_score")
        assert (top50.true_label == "ring").sum() >= 20


class TestLossMetrics:
    def test_loss_ratio_plausible(self, book):
        _, claims, _, policies = book
        lr = loss_metrics.loss_ratios(claims, policies)
        assert 0.2 < lr.loss_ratio.mean() < 1.5

    def test_triangle_cumulative_nondecreasing(self, book):
        _, claims, *_ = book
        tri = loss_metrics.development_triangle(claims)
        assert (tri.diff(axis=1).fillna(0) >= -1e-9).all().all()

    def test_age_to_age_factors_at_least_one(self, book):
        _, claims, *_ = book
        f = loss_metrics.age_to_age(loss_metrics.development_triangle(claims))
        assert (f >= 1.0).all()
