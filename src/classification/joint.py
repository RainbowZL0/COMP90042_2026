"""Joint cross-encoder classifier for 4-way claim verification."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, Union

from src.classification.base import Classifier
from src.classification.joint_data import (
    ID_TO_LABEL,
    LABELS,
    JointExample,
    build_evidence_bundle,
)
from src.data.schema import ClaimLabel, Evidence

PathLike = Union[str, Path]


class JointClassificationDataset:
    """Tokenized dataset for joint claim+evidence classification."""

    def __init__(self, examples: Sequence[JointExample], tokenizer, max_length: int):
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
        item["labels"] = torch.tensor(ex.label_id, dtype=torch.long)
        return item


class JointCrossEncoderClassifier(Classifier):
    """A Transformer cross-encoder over claim and bundled evidence text."""

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cpu",
        max_length: int = 256,
        max_evidences: int = 3,
    ):
        if max_length <= 0:
            raise ValueError("max_length must be > 0")
        if max_evidences <= 0:
            raise ValueError("max_evidences must be > 0")
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.max_evidences = max_evidences
        self.model.config.num_labels = len(LABELS)
        self.model.config.problem_type = "single_label_classification"
        self.model.config.id2label = {i: label.value for i, label in enumerate(LABELS)}
        self.model.config.label2id = {label.value: i for i, label in enumerate(LABELS)}
        self.model.to(device)

    @classmethod
    def from_pretrained_base(
        cls,
        model_name_or_path: str,
        device: str = "cpu",
        max_length: int = 256,
        max_evidences: int = 3,
    ) -> "JointCrossEncoderClassifier":
        """Initialise a 4-way classifier from a pretrained HF checkpoint."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers is required for JointCrossEncoderClassifier. "
                "Install with: uv add transformers torch"
            ) from e

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=len(LABELS),
            problem_type="single_label_classification",
            ignore_mismatched_sizes=True,
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=max_length,
            max_evidences=max_evidences,
        )

    @classmethod
    def load(
        cls,
        path: PathLike,
        device: str = "cpu",
        max_length: int = 256,
        max_evidences: int = 3,
    ) -> "JointCrossEncoderClassifier":
        return cls.from_pretrained_base(
            str(path),
            device=device,
            max_length=max_length,
            max_evidences=max_evidences,
        )

    def save(self, path: PathLike) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.config.num_labels = len(LABELS)
        self.model.config.problem_type = "single_label_classification"
        self.model.config.id2label = {i: label.value for i, label in enumerate(LABELS)}
        self.model.config.label2id = {label.value: i for i, label in enumerate(LABELS)}
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        meta = {"max_length": self.max_length, "max_evidences": self.max_evidences}
        with (path / "joint_classifier_meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=True, indent=2)
            f.write("\n")

    def classify_batch(
        self,
        claim_texts: Sequence[str],
        evidences_batch: Sequence[Sequence[Evidence]],
    ) -> list[ClaimLabel]:
        if len(claim_texts) != len(evidences_batch):
            raise ValueError(
                f"claim_texts and evidences_batch must have same length "
                f"({len(claim_texts)} vs {len(evidences_batch)})"
            )
        if not claim_texts:
            return []
        evidence_texts = [
            build_evidence_bundle(evidences, max_evidences=self.max_evidences)
            for evidences in evidences_batch
        ]
        return self.predict_labels(claim_texts, evidence_texts)

    def predict_labels(
        self,
        claim_texts: Sequence[str],
        evidence_texts: Sequence[str],
        batch_size: int = 32,
    ) -> list[ClaimLabel]:
        if len(claim_texts) != len(evidence_texts):
            raise ValueError("claim_texts and evidence_texts must have same length")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not claim_texts:
            return []

        import torch

        self.model.eval()
        out: list[ClaimLabel] = []
        with torch.no_grad():
            for start in range(0, len(claim_texts), batch_size):
                end = start + batch_size
                encoded = self.tokenizer(
                    list(claim_texts[start:end]),
                    list(evidence_texts[start:end]),
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                logits = self.model(**encoded).logits
                pred_ids = torch.argmax(logits, dim=-1).detach().cpu().tolist()
                out.extend(ID_TO_LABEL[int(i)] for i in pred_ids)
        return out


def train_joint_classifier(
    classifier: JointCrossEncoderClassifier,
    train_examples: Sequence[JointExample],
    dev_examples: Sequence[JointExample] | None = None,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    class_weights: Sequence[float] | None = None,
) -> dict[str, list[float]]:
    """Explicit training loop for 4-way claim classification."""
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

    train_ds = JointClassificationDataset(
        train_examples, classifier.tokenizer, classifier.max_length
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        classifier.model.parameters(), lr=lr, weight_decay=weight_decay
    )
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps,
    )
    if class_weights is None:
        weight_tensor = None
    else:
        weight_tensor = torch.tensor(list(class_weights), dtype=torch.float, device=classifier.device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)

    history: dict[str, list[float]] = {"train_loss": []}
    if dev_examples is not None:
        history["dev_loss"] = []
        history["dev_accuracy"] = []

    for epoch in range(epochs):
        classifier.model.train()
        running = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"joint classifier epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            labels = batch.pop("labels").to(classifier.device)
            batch = {k: v.to(classifier.device) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            logits = classifier.model(**batch).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running += float(loss.detach().cpu())
            n_batches += 1
            pbar.set_postfix(loss=running / n_batches)

        history["train_loss"].append(running / max(n_batches, 1))
        if dev_examples is not None:
            dev_loss, dev_acc = evaluate_joint_classifier(
                classifier, dev_examples, batch_size=batch_size, class_weights=class_weights
            )
            history["dev_loss"].append(dev_loss)
            history["dev_accuracy"].append(dev_acc)

    return history


def evaluate_joint_classifier(
    classifier: JointCrossEncoderClassifier,
    examples: Sequence[JointExample],
    batch_size: int = 32,
    class_weights: Sequence[float] | None = None,
) -> tuple[float, float]:
    """Return (loss, accuracy) on examples."""
    if not examples:
        raise ValueError("examples must be non-empty")

    import torch
    from torch.utils.data import DataLoader

    ds = JointClassificationDataset(examples, classifier.tokenizer, classifier.max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    if class_weights is None:
        weight_tensor = None
    else:
        weight_tensor = torch.tensor(list(class_weights), dtype=torch.float, device=classifier.device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)

    classifier.model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(classifier.device)
            batch = {k: v.to(classifier.device) for k, v in batch.items()}
            logits = classifier.model(**batch).logits
            loss = loss_fn(logits, labels)
            preds = torch.argmax(logits, dim=-1)
            total_loss += float(loss.detach().cpu())
            correct += int((preds == labels).sum().detach().cpu())
            total += int(labels.numel())
    return total_loss / max(len(loader), 1), correct / max(total, 1)
