"""Cross-encoder reranker.

This module is lazy about importing torch/transformers so the non-ML parts of
this project remain importable in lightweight test environments.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, Union

from src.data.schema import Evidence
from src.retrieval.rerank_data import RerankExample

PathLike = Union[str, Path]


def repair_single_logit_config(path: PathLike) -> bool:
    """Repair old local checkpoints saved with an invalid HF config.

    Earlier versions of this project used a single-logit BCE head but saved the
    HuggingFace config as ``problem_type='single_label_classification'`` with
    ``num_labels=1``. Newer transformers versions reject that combination while
    loading even the tokenizer. For our manual BCEWithLogitsLoss training, the
    correct compatible config is ``problem_type='regression'`` with one output
    logit. This function edits only local ``config.json`` files and is a no-op
    for remote model ids.
    """
    config_path = Path(path) / "config.json"
    if not config_path.exists():
        return False

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    changed = False
    if config.get("num_labels") == 1 and config.get("problem_type") == "single_label_classification":
        config["problem_type"] = "regression"
        changed = True

    # Keep labels explicit and harmless for one-logit relevance scoring.
    if config.get("num_labels") == 1:
        if config.get("id2label") != {"0": "RELEVANCE"}:
            config["id2label"] = {"0": "RELEVANCE"}
            changed = True
        if config.get("label2id") != {"RELEVANCE": 0}:
            config["label2id"] = {"RELEVANCE": 0}
            changed = True

    if changed:
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=True, indent=2)
            f.write("\n")
    return changed


class RerankDataset:  # torch Dataset at runtime, kept untyped to avoid hard import
    """Tokenized dataset for pointwise cross-encoder training."""

    def __init__(self, examples: Sequence[RerankExample], tokenizer, max_length: int):
        if max_length <= 0:
            raise ValueError("max_length must be > 0")
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        import torch

        ex = self.examples[idx]
        encoded = self.tokenizer(
            ex.claim_text,
            ex.evidence_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(float(ex.label), dtype=torch.float)
        return item


class CrossEncoderReranker:
    """Binary relevance reranker over (claim, evidence) text pairs.

    The model uses one output logit and we train it manually with
    BCEWithLogitsLoss. HuggingFace should therefore see the head as a
    one-dimensional regression head; we do not use HF's built-in loss.
    """

    def __init__(self, model, tokenizer, device: str = "cpu", max_length: int = 192):
        if max_length <= 0:
            raise ValueError("max_length must be > 0")
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.model.config.num_labels = 1
        self.model.config.problem_type = "regression"
        self.model.to(device)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: str = "cpu",
        max_length: int = 192,
    ) -> "CrossEncoderReranker":
        """Load a HuggingFace sequence-classification model with one logit."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers is required for CrossEncoderReranker. "
                "Install with: uv add transformers torch"
            ) from e

        # Automatically fix local checkpoints produced by the earlier buggy save.
        repair_single_logit_config(model_name_or_path)

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=1,
            problem_type="regression",
        )
        model.config.num_labels = 1
        model.config.problem_type = "regression"
        return cls(model=model, tokenizer=tokenizer, device=device, max_length=max_length)

    def save(self, path: PathLike) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.config.num_labels = 1
        self.model.config.problem_type = "regression"
        self.model.config.id2label = {0: "RELEVANCE"}
        self.model.config.label2id = {"RELEVANCE": 0}
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        repair_single_logit_config(path)

    @classmethod
    def load(
        cls,
        path: PathLike,
        device: str = "cpu",
        max_length: int = 192,
    ) -> "CrossEncoderReranker":
        return cls.from_pretrained(str(path), device=device, max_length=max_length)

    def predict_scores(
        self,
        claim_texts: Sequence[str],
        evidences: Sequence[Evidence],
        batch_size: int = 32,
    ) -> list[float]:
        """Return raw relevance logits, one per (claim_text, evidence) pair."""
        if len(claim_texts) != len(evidences):
            raise ValueError(
                f"claim_texts and evidences must have same length "
                f"({len(claim_texts)} vs {len(evidences)})"
            )
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not claim_texts:
            return []

        import torch

        self.model.eval()
        scores: list[float] = []
        with torch.no_grad():
            for start in range(0, len(claim_texts), batch_size):
                end = start + batch_size
                batch_claims = list(claim_texts[start:end])
                batch_evidence_texts = [ev.text for ev in evidences[start:end]]
                encoded = self.tokenizer(
                    batch_claims,
                    batch_evidence_texts,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                logits = self.model(**encoded).logits.view(-1)
                scores.extend(float(x) for x in logits.detach().cpu().tolist())
        return scores


def train_reranker(
    model: CrossEncoderReranker,
    train_examples: Sequence[RerankExample],
    dev_examples: Sequence[RerankExample] | None = None,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
) -> dict[str, list[float]]:
    """Simple, explicit training loop for pointwise BCE reranking."""
    if epochs <= 0:
        raise ValueError("epochs must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if not train_examples:
        raise ValueError("train_examples must be non-empty")

    import torch
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    from transformers import get_linear_schedule_with_warmup

    train_ds = RerankDataset(train_examples, model.tokenizer, model.max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        model.model.parameters(), lr=lr, weight_decay=weight_decay
    )
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps,
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    history: dict[str, list[float]] = {"train_loss": []}
    if dev_examples is not None:
        history["dev_loss"] = []

    for epoch in range(epochs):
        model.model.train()
        running = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"reranker epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            labels = batch.pop("labels").to(model.device)
            batch = {k: v.to(model.device) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            logits = model.model(**batch).logits.view(-1)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running += float(loss.detach().cpu())
            n_batches += 1
            pbar.set_postfix(loss=running / n_batches)

        history["train_loss"].append(running / max(n_batches, 1))
        if dev_examples is not None:
            history["dev_loss"].append(
                evaluate_reranker_loss(model, dev_examples, batch_size=batch_size)
            )
    return history


def evaluate_reranker_loss(
    model: CrossEncoderReranker,
    examples: Sequence[RerankExample],
    batch_size: int = 64,
) -> float:
    """Compute BCE loss on held-out rerank examples."""
    if not examples:
        raise ValueError("examples must be non-empty")

    import torch
    from torch.utils.data import DataLoader

    ds = RerankDataset(examples, model.tokenizer, model.max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.model.eval()

    total = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(model.device)
            batch = {k: v.to(model.device) for k, v in batch.items()}
            logits = model.model(**batch).logits.view(-1)
            loss = loss_fn(logits, labels)
            total += float(loss.detach().cpu())
            n_batches += 1
    return total / max(n_batches, 1)
