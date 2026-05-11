"""Per-evidence stance encoder with attention aggregation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, Union

from src.classification.base import Classifier
from src.classification.joint_data import ID_TO_LABEL, LABELS
from src.classification.stance_attention_data import (
    STANCE_LABELS,
    StanceAttentionExample,
    _pad_evidences,
)
from src.data.schema import ClaimLabel, Evidence

PathLike = Union[str, Path]


class StanceAttentionDataset:
    def __init__(self, examples: Sequence[StanceAttentionExample], tokenizer, max_length: int, max_evidences: int):
        if max_length <= 0:
            raise ValueError("max_length must be > 0")
        if max_evidences <= 0:
            raise ValueError("max_evidences must be > 0")
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_evidences = max_evidences

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        import torch

        ex = self.examples[idx]
        claims = [ex.claim_text] * self.max_evidences
        evidences = list(ex.evidence_texts)
        encoded = self.tokenizer(
            claims,
            evidences,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v for k, v in encoded.items()}
        item["evidence_mask"] = torch.tensor(ex.evidence_mask, dtype=torch.bool)
        item["labels"] = torch.tensor(ex.label_id, dtype=torch.long)
        item["stance_labels"] = torch.tensor(ex.stance_labels, dtype=torch.long)
        item["stance_mask"] = torch.tensor(ex.stance_mask, dtype=torch.bool)
        return item


class StanceAttentionTorchModel:  # nn.Module created lazily to keep imports optional
    pass


def _build_torch_model(
    backbone, hidden_size: int, num_labels: int = 4, num_stance_labels: int = 3, dropout: float = 0.1
):
    import torch

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.dropout = torch.nn.Dropout(dropout)
            self.stance_head = torch.nn.Linear(hidden_size, num_stance_labels)
            self.attention = torch.nn.Linear(hidden_size, 1)
            self.classifier = torch.nn.Linear(hidden_size, num_labels)

        def forward(self, input_ids, attention_mask, evidence_mask, token_type_ids=None):
            bsz, k, seq_len = input_ids.shape
            flat_inputs = {
                "input_ids": input_ids.reshape(bsz * k, seq_len),
                "attention_mask": attention_mask.reshape(bsz * k, seq_len),
            }
            if token_type_ids is not None:
                flat_inputs["token_type_ids"] = token_type_ids.reshape(bsz * k, seq_len)
            outputs = self.backbone(**flat_inputs)
            cls = outputs.last_hidden_state[:, 0, :].reshape(bsz, k, -1)
            cls = self.dropout(cls)
            stance_logits = self.stance_head(cls)
            attn_logits = self.attention(cls).squeeze(-1)
            attn_logits = attn_logits.masked_fill(~evidence_mask, -1e4)
            weights = torch.softmax(attn_logits, dim=-1)
            pooled = torch.sum(cls * weights.unsqueeze(-1), dim=1)
            logits = self.classifier(self.dropout(pooled))
            return {"logits": logits, "stance_logits": stance_logits, "attention_weights": weights}

    return _Model()


class StanceAttentionClassifier(Classifier):
    """Claim classifier using per-evidence encodings and learned attention."""

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cpu",
        max_length: int = 192,
        max_evidences: int = 3,
        backbone_name_or_path: str | None = None,
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
        self.backbone_name_or_path = backbone_name_or_path
        self.model.to(device)

    @classmethod
    def from_pretrained_base(
        cls,
        model_name_or_path: str,
        device: str = "cpu",
        max_length: int = 192,
        max_evidences: int = 3,
        dropout: float = 0.1,
    ) -> "StanceAttentionClassifier":
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ImportError("transformers is required. Install with: uv add transformers torch") from e

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        backbone = AutoModel.from_pretrained(model_name_or_path)
        hidden_size = int(backbone.config.hidden_size)
        model = _build_torch_model(backbone, hidden_size=hidden_size, dropout=dropout)
        return cls(
            model, tokenizer, device=device, max_length=max_length, max_evidences=max_evidences,
            backbone_name_or_path=model_name_or_path
        )

    @classmethod
    def load(
        cls,
        path: PathLike,
        device: str = "cpu",
        max_length: int | None = None,
        max_evidences: int | None = None,
    ) -> "StanceAttentionClassifier":
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ImportError("transformers and torch are required.") from e

        path = Path(path)
        with (path / "stance_attention_meta.json").open("r", encoding="utf-8") as f:
            meta = json.load(f)
        tokenizer = AutoTokenizer.from_pretrained(path / "backbone")
        backbone = AutoModel.from_pretrained(path / "backbone")
        hidden_size = int(backbone.config.hidden_size)
        model = _build_torch_model(backbone, hidden_size=hidden_size, dropout=float(meta.get("dropout", 0.1)))
        heads_path = path / "stance_attention_heads.pt"
        state = torch.load(heads_path, map_location=device)
        model.load_state_dict(state)
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=int(max_length or meta["max_length"]),
            max_evidences=int(max_evidences or meta["max_evidences"]),
            backbone_name_or_path=meta.get("backbone_name_or_path"),
        )

    def save(self, path: PathLike, dropout: float = 0.1) -> None:
        import torch

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        backbone_dir = path / "backbone"
        self.model.backbone.save_pretrained(backbone_dir)
        self.tokenizer.save_pretrained(backbone_dir)
        torch.save(self.model.state_dict(), path / "stance_attention_heads.pt")
        meta = {
            "max_length": self.max_length,
            "max_evidences": self.max_evidences,
            "labels": [label.value for label in LABELS],
            "stance_labels": list(STANCE_LABELS),
            "backbone_name_or_path": self.backbone_name_or_path,
            "dropout": dropout,
        }
        with (path / "stance_attention_meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=True, indent=2)
            f.write("\n")

    def classify_batch(
        self,
        claim_texts: Sequence[str],
        evidences_batch: Sequence[Sequence[Evidence]],
    ) -> list[ClaimLabel]:
        if len(claim_texts) != len(evidences_batch):
            raise ValueError("claim_texts and evidences_batch must have same length")
        if not claim_texts:
            return []

        import torch

        self.model.eval()
        predictions: list[ClaimLabel] = []
        batch_size = 16
        with torch.no_grad():
            for start in range(0, len(claim_texts), batch_size):
                batch_claims = list(claim_texts[start:start + batch_size])
                batch_evs = evidences_batch[start:start + batch_size]
                all_claims: list[str] = []
                all_evidence_texts: list[str] = []
                masks: list[tuple[int, ...]] = []
                for claim_text, evidences in zip(batch_claims, batch_evs):
                    texts, mask = _pad_evidences(evidences, self.max_evidences)
                    all_claims.extend([claim_text] * self.max_evidences)
                    all_evidence_texts.extend(list(texts))
                    masks.append(mask)
                encoded = self.tokenizer(
                    all_claims,
                    all_evidence_texts,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                bsz = len(batch_claims)
                model_inputs = {}
                for key, val in encoded.items():
                    model_inputs[key] = val.reshape(bsz, self.max_evidences, -1).to(self.device)
                evidence_mask = torch.tensor(masks, dtype=torch.bool, device=self.device)
                outputs = self.model(evidence_mask=evidence_mask, **model_inputs)
                pred_ids = torch.argmax(outputs["logits"], dim=-1).detach().cpu().tolist()
                predictions.extend(ID_TO_LABEL[int(i)] for i in pred_ids)
        return predictions


def train_stance_attention_classifier(
    classifier: StanceAttentionClassifier,
    train_examples: Sequence[StanceAttentionExample],
    dev_examples: Sequence[StanceAttentionExample] | None = None,
    epochs: int = 6,
    batch_size: int = 8,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    class_weights: Sequence[float] | None = None,
    stance_loss_weight: float = 0.3,
) -> dict[str, list[float]]:
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

    train_ds = StanceAttentionDataset(
        train_examples, classifier.tokenizer, classifier.max_length, classifier.max_evidences
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(classifier.model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps,
    )
    weight_tensor = None if class_weights is None else torch.tensor(
        list(class_weights), dtype=torch.float, device=classifier.device
    )
    cls_loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)
    stance_loss_fn = torch.nn.CrossEntropyLoss()

    history: dict[str, list[float]] = {"train_loss": [], "train_cls_loss": [], "train_stance_loss": []}
    if dev_examples is not None:
        history["dev_loss"] = []
        history["dev_accuracy"] = []

    for epoch in range(epochs):
        classifier.model.train()
        running = 0.0
        running_cls = 0.0
        running_stance = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"stance-attention epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            labels = batch.pop("labels").to(classifier.device)
            stance_labels = batch.pop("stance_labels").to(classifier.device)
            stance_mask = batch.pop("stance_mask").to(classifier.device)
            batch = {k: v.to(classifier.device) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            outputs = classifier.model(**batch)
            cls_loss = cls_loss_fn(outputs["logits"], labels)
            if stance_mask.any():
                stance_logits = outputs["stance_logits"][stance_mask]
                masked_stance_labels = stance_labels[stance_mask]
                stance_loss = stance_loss_fn(stance_logits, masked_stance_labels)
            else:
                stance_loss = torch.tensor(0.0, device=classifier.device)
            loss = cls_loss + stance_loss_weight * stance_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running += float(loss.detach().cpu())
            running_cls += float(cls_loss.detach().cpu())
            running_stance += float(stance_loss.detach().cpu())
            n_batches += 1
            pbar.set_postfix(loss=running / n_batches, cls=running_cls / n_batches, stance=running_stance / n_batches)

        history["train_loss"].append(running / max(n_batches, 1))
        history["train_cls_loss"].append(running_cls / max(n_batches, 1))
        history["train_stance_loss"].append(running_stance / max(n_batches, 1))
        if dev_examples is not None:
            dev_loss, dev_acc = evaluate_stance_attention_classifier(
                classifier,
                dev_examples,
                batch_size=batch_size,
                class_weights=class_weights,
                stance_loss_weight=stance_loss_weight,
            )
            history["dev_loss"].append(dev_loss)
            history["dev_accuracy"].append(dev_acc)
    return history


def evaluate_stance_attention_classifier(
    classifier: StanceAttentionClassifier,
    examples: Sequence[StanceAttentionExample],
    batch_size: int = 16,
    class_weights: Sequence[float] | None = None,
    stance_loss_weight: float = 0.3,
) -> tuple[float, float]:
    if not examples:
        raise ValueError("examples must be non-empty")

    import torch
    from torch.utils.data import DataLoader

    ds = StanceAttentionDataset(examples, classifier.tokenizer, classifier.max_length, classifier.max_evidences)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    weight_tensor = None if class_weights is None else torch.tensor(
        list(class_weights), dtype=torch.float, device=classifier.device
    )
    cls_loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)
    stance_loss_fn = torch.nn.CrossEntropyLoss()

    classifier.model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(classifier.device)
            stance_labels = batch.pop("stance_labels").to(classifier.device)
            stance_mask = batch.pop("stance_mask").to(classifier.device)
            batch = {k: v.to(classifier.device) for k, v in batch.items()}
            outputs = classifier.model(**batch)
            cls_loss = cls_loss_fn(outputs["logits"], labels)
            if stance_mask.any():
                stance_loss = stance_loss_fn(outputs["stance_logits"][stance_mask], stance_labels[stance_mask])
            else:
                stance_loss = torch.tensor(0.0, device=classifier.device)
            loss = cls_loss + stance_loss_weight * stance_loss
            preds = torch.argmax(outputs["logits"], dim=-1)
            total_loss += float(loss.detach().cpu())
            correct += int((preds == labels).sum().detach().cpu())
            total += int(labels.numel())
    return total_loss / max(len(loader), 1), correct / max(total, 1)
