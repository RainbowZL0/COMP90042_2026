from src.utils.metrics import compute_eval_metrics


def test_compute_eval_metrics_matches_expected_single_claim():
    predictions = {
        "c1": {"claim_label": "SUPPORTS", "evidences": ["e1", "e2", "e3"]}
    }
    groundtruth = {
        "c1": {"claim_label": "SUPPORTS", "evidences": ["e1", "e3", "e4"]}
    }
    metrics = compute_eval_metrics(predictions, groundtruth)
    assert metrics.accuracy == 1.0
    assert round(metrics.evidence_f, 6) == round(2 / 3, 6)
    assert round(metrics.harmonic_mean, 6) == round(2 * (2 / 3) * 1 / ((2 / 3) + 1), 6)
