import json

from src.retrieval.reranker import repair_single_logit_config


def test_repair_single_logit_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"num_labels": 1, "problem_type": "single_label_classification"}),
        encoding="utf-8",
    )

    changed = repair_single_logit_config(tmp_path)

    assert changed is True
    repaired = json.loads(config_path.read_text(encoding="utf-8"))
    assert repaired["num_labels"] == 1
    assert repaired["problem_type"] == "regression"
    assert repaired["id2label"] == {"0": "RELEVANCE"}
    assert repaired["label2id"] == {"RELEVANCE": 0}
