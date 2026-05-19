"""Build nlp_submission_v11.ipynb — leaderboard-ready single-pass system.

Pipeline:
  Retrieval :  BM25 (PP3, k1=1.2, b=0.5)
             ∪ Dense BGE-base
             → hybrid pool
             → R2 reranker  (mixed-neg, LR=5e-6, 5ep — v10 winner)
             → relative-δ δ=1.5

  Classification (ensemble):
    branch A — joint classifier
       cross-encoder/ms-marco-MiniLM-L-12-v2, max_len=512, W0 (uniform)
       trained on (claim, [SEP].join(top-K evidence)) with gold∪retrieved
    branch B — per-evidence stance + 9-stat aggregator
       cross-encoder/nli-MiniLM2-L6-H768 FROZEN
       per (claim, ev_i): 3-class NLI [contradict, entail, neutral]
       9 hand-crafted stats per claim → tiny MLP 9→16→4

    ensemble: softmax(A) · α + softmax(B) · (1-α), α grid-tuned on dev

  Final model: same hyperparams, retrained on train ∪ dev (leaderboard-allowed).
"""
from __future__ import annotations
import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []
def md(text: str) -> None: CELLS.append(("markdown", text))
def code(text: str) -> None: CELLS.append(("code", text))


# ================================================================ Title
md(r"""# COMP90042 2026 Project — v11 leaderboard submission

End-to-end system: hybrid retrieval → fine-tuned cross-encoder reranker →
ensemble of (joint 4-class classifier) + (per-evidence NLI stance head)
→ train+dev joint training for the final submission.

Design choices, each pinned to a prior experiment that validated it on dev:

| Component | Choice | Source |
|---|---|---|
| BM25 preprocessing × params | PP3 (lowercase+stopwords+Porter) × `k1=1.2, b=0.5` | v9 Stage 1 sweep (best of 15 cells) |
| Dense retriever | `BAAI/bge-base-en-v1.5` (CLS + L2 normalise) | v7/v8 cached embeddings |
| Reranker training | `ms-marco-MiniLM-L-12-v2`, LR=5e-6, 5 epochs, mixed BM25\Dense + Dense\BM25 negatives | v10 R2 winner |
| Scoring pool for reranker | BM25 ∪ Dense (full hybrid) | v10 (no longer crashes — see §6) |
| Selector | relative-δ δ=1.5 | v10 best-of-strategies |
| Joint classifier backbone | `ms-marco-MiniLM-L-12-v2`, `max_len=512` | v9 Stage 4 winner (with user's max_len=512 fix) |
| Classifier class weights | W0 (uniform) | v9 Stage 5: best DISPUTED recall at acceptable A |
| Stance head NLI backbone | `cross-encoder/nli-MiniLM2-L6-H768` — **frozen**, 3-class output kept | v8 §6.4 (avoid the 4-class reinit pathology) |
| Train+dev joint final | retrain reranker + joint + stance on 1382 examples | leaderboard convention; allowed by spec |

**Key novelty**: per-evidence NLI stance + 9-feature aggregator. v8/v9/v10
left REFUTES retrieval F stuck around 0.10 and REFUTES classification at
0/27 dev. The stance head reads directly from the 3-class entailment
output of a frozen NLI model — preserving the entail/contradict/neutral
signal that 4-class fine-tuning destroys (v8 §6.4) — and aggregates over
top-K evidence with hand-engineered statistics that capture DISPUTED's
defining feature (high variance of stance across evidence). The final
4-class output is a small trainable MLP, not a hand-crafted rule.
""")


# ================================================================ Section 1
md(r"""# 1. DataSet Processing
(You can add as many code blocks and text blocks as you need. However, YOU SHOULD NOT MODIFY the section title)
""")

code(r'''# Optional dependency installation (Colab-friendly; safe to skip on local).
import importlib.util, subprocess, sys
for import_name, pip_name in {
    "bm25s": "bm25s", "transformers": "transformers", "accelerate": "accelerate",
    "sklearn": "scikit-learn", "nltk": "nltk",
}.items():
    if importlib.util.find_spec(import_name) is None:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])
        except Exception as e:
            print(f"auto-install {pip_name} failed ({e}); please install manually.")
''')

code(r'''from pathlib import Path
import json, random, time, pickle, gc, math, re
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# -------- Reproducibility --------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -------- Paths --------
DATA_DIR    = Path("data")
V7_CACHE    = Path("outputs_notebook_v7/cache")     # evidence_bge_fp16.npy + dense top-500
V9_CACHE    = Path("outputs_notebook_v9/cache")     # BM25 top-500 (PP3/R-both)
V10_MODELS  = Path("outputs_notebook_v10/models")   # reranker_R2.pt
OUTPUT_DIR  = Path("outputs_notebook_v11")
CACHE_DIR   = OUTPUT_DIR / "cache"
MODEL_DIR   = OUTPUT_DIR / "models"
for d in (OUTPUT_DIR, CACHE_DIR, MODEL_DIR):
    d.mkdir(exist_ok=True, parents=True)

# -------- Hardware-aware precision (auto bf16 / fp16) --------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda" and torch.cuda.is_bf16_supported():
    AUTOCAST_DTYPE  = torch.bfloat16; USE_GRAD_SCALER = False
elif DEVICE.type == "cuda":
    AUTOCAST_DTYPE  = torch.float16;  USE_GRAD_SCALER = True
else:
    AUTOCAST_DTYPE  = torch.float32;  USE_GRAD_SCALER = False
if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
print(f"Device: {DEVICE}  autocast={AUTOCAST_DTYPE}  grad_scaler={USE_GRAD_SCALER}")

# -------- Task constants --------
LABELS    = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]
LABEL2ID  = {l: i for i, l in enumerate(LABELS)}
ID2LABEL  = {i: l for i, l in enumerate(LABELS)}

# -------- Hyperparameters --------
BM25_K1, BM25_B           = 1.2, 0.5
BM25_CANDIDATE_K          = 500
DENSE_CANDIDATE_K         = 500
DENSE_MODEL_NAME          = "BAAI/bge-base-en-v1.5"

RERANKER_MODEL_NAME       = "cross-encoder/ms-marco-MiniLM-L-12-v2"
RERANKER_MAX_LEN          = 256
RERANKER_BATCH_TRAIN      = 32
RERANKER_BATCH_EVAL       = 128
RERANKER_EPOCHS           = 5
RERANKER_LR               = 5e-6
NEGATIVES_PER_POSITIVE    = 4

CLASSIFIER_MODEL_NAME     = "cross-encoder/ms-marco-MiniLM-L-12-v2"
CLASSIFIER_MAX_LEN        = 512
CLASSIFIER_BATCH_TRAIN    = 8
CLASSIFIER_BATCH_EVAL     = 16
CLASSIFIER_EPOCHS         = 4
CLASSIFIER_LR             = 1e-5

NLI_MODEL_NAME            = "cross-encoder/nli-MiniLM2-L6-H768"
NLI_MAX_LEN               = 256
NLI_BATCH                 = 64

STANCE_MLP_HIDDEN         = 16
STANCE_MLP_DROPOUT        = 0.3
STANCE_EPOCHS             = 60
STANCE_LR                 = 1e-3
STANCE_BATCH              = 32

DELTA_FINAL               = 1.5
MIN_FINAL_K, MAX_FINAL_K  = 1, 5
WEIGHT_DECAY              = 0.01
WARMUP_RATIO              = 0.1
GRAD_CLIP                 = 1.0

ALPHA_GRID = [round(x, 2) for x in np.arange(0.0, 1.01, 0.05)]
''')


code(r'''def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

train_claims = load_json(DATA_DIR / "train-claims.json")
dev_claims   = load_json(DATA_DIR / "dev-claims.json")
test_claims  = load_json(DATA_DIR / "test-claims-unlabelled.json")
evidence     = load_json(DATA_DIR / "evidence.json")

evidence_ids = list(evidence.keys())
EID2IDX = {e: i for i, e in enumerate(evidence_ids)}

print(f"train={len(train_claims):,}  dev={len(dev_claims):,}  "
      f"test={len(test_claims):,}  evidence={len(evidence):,}")

label_dist = Counter(c["claim_label"] for c in train_claims.values())
for lbl in LABELS:
    print(f"  {lbl:20s} {label_dist[lbl]:5d} ({label_dist[lbl]/len(train_claims):6.2%})")
''')


code(r'''# Metrics (mirror eval.py).
def evidence_f1(pred_eids, gold_eids):
    if not pred_eids: return 0.0
    pred_set = set(pred_eids)
    correct = sum(1 for e in gold_eids if e in pred_set)
    if correct == 0: return 0.0
    p = correct / len(pred_eids); r = correct / len(gold_eids)
    return 2 * p * r / (p + r)

def retrieval_F(retr_dict, claims_dict):
    return float(np.mean([
        evidence_f1(retr_dict[cid], c["evidences"])
        for cid, c in claims_dict.items()
    ]))

def recall_at_k(cands_top500, claims_dict, k=500):
    rs = []
    for cid, c in claims_dict.items():
        cand = set(cands_top500[cid][:k])
        gold = c["evidences"]
        if not gold: continue
        rs.append(sum(1 for e in gold if e in cand) / len(gold))
    return float(np.mean(rs))

def evaluate_submission(preds, claims, verbose=True):
    f_scores, correct, total = [], 0, 0
    for cid, gold in claims.items():
        p = preds[cid]
        f_scores.append(evidence_f1(p["evidences"], gold["evidences"]))
        correct += int(p["claim_label"] == gold["claim_label"])
        total += 1
    F = float(np.mean(f_scores)); A = correct / total
    H = 0.0 if (F + A) == 0 else 2 * F * A / (F + A)
    if verbose:
        print(f"Evidence Retrieval F-score (F)    = {F:.6f}")
        print(f"Claim Classification Accuracy (A) = {A:.6f}")
        print(f"Harmonic Mean of F and A          = {H:.6f}")
    return {"F": F, "A": A, "H": H}

def write_predictions(preds, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2)
''')


# ---------------- BM25 retrieval ----------------
md(r"""## 1.1 Stage 1 — BM25 retrieval

PP3 preprocessing (lowercase + English stopwords + Porter stem) with
`k1=1.2, b=0.5`. Caches top-500 lists per split. Reuses v9's cache if
available; otherwise rebuilds from scratch.
""")

code(r'''import bm25s
EN_STOPWORDS = set(bm25s.tokenization.STOPWORDS_EN)
try:
    from nltk.stem import PorterStemmer
    _porter = PorterStemmer()
except Exception:
    _porter = None
    print("PorterStemmer unavailable; PP3 falls back to stopword-only.")

_TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")
def preprocess_text(text: str):
    toks = _TOKEN_PATTERN.findall(text.lower())
    toks = [t for t in toks if t not in EN_STOPWORDS]
    if _porter is not None:
        toks = [_porter.stem(t) for t in toks]
    return toks


def build_or_load_bm25_pool(claims_dict, split: str):
    """Return dict[cid] -> [(eid, bm25_score), ...] of top-500.
    Cache hierarchy: v11 → v9 → recompute."""
    for cache_path in [CACHE_DIR / f"{split}_bm25_top500.pkl",
                       V9_CACHE / f"{split}_bm25_top500.pkl"]:
        if cache_path.exists():
            print(f"  loading BM25 cache: {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
    # rebuild (slow; only happens on a fresh checkout)
    print(f"  no BM25 cache found for {split}; rebuilding ...")
    global _bm25_index, _corpus_tokens
    if "_bm25_index" not in globals():
        print("  tokenising corpus (one-time, ~30s on RTX 4060) ...")
        _corpus_tokens = [preprocess_text(t) for t in tqdm(
            (evidence[e] for e in evidence_ids), total=len(evidence_ids))]
        _bm25_index = bm25s.BM25(k1=BM25_K1, b=BM25_B)
        _bm25_index.index(_corpus_tokens, show_progress=False)
    cids = list(claims_dict.keys())
    queries = [claims_dict[cid]["claim_text"] for cid in cids]
    qtok = [preprocess_text(q) for q in queries]
    res, scores = _bm25_index.retrieve(qtok, k=BM25_CANDIDATE_K, show_progress=False)
    out = {}
    for r, cid in enumerate(cids):
        out[cid] = [(evidence_ids[int(res[r][j])], float(scores[r][j])) for j in range(len(res[r]))]
    cache_path = CACHE_DIR / f"{split}_bm25_top500.pkl"
    with open(cache_path, "wb") as f:
        pickle.dump(out, f)
    return out

train_bm25 = build_or_load_bm25_pool(train_claims, "train")
dev_bm25   = build_or_load_bm25_pool(dev_claims,   "dev")
test_bm25  = build_or_load_bm25_pool(test_claims,  "test")

dev_bm25_eids = {cid: [e for e, _ in v] for cid, v in dev_bm25.items()}
print(f"dev BM25 recall@500 = {recall_at_k(dev_bm25_eids, dev_claims):.4f}")
''')


# ---------------- Dense retrieval ----------------
md(r"""## 1.2 Stage 1b — Dense retrieval (BGE-base)

CLS pooling, L2 normalised. Caches the full corpus embedding (~1.86 GB
fp16) and top-500 lists per split. Loads from v7 cache if available;
otherwise rebuilds (one-time ~30 min on RTX 4060 / ~15 min on T4).
""")

code(r'''from transformers import AutoTokenizer, AutoModel

def _get_bge_evidence_embeddings():
    cache = V7_CACHE / "evidence_bge_fp16.npy"
    if cache.exists():
        print(f"  loading BGE corpus embeddings: {cache}")
        return np.load(cache)
    print("  encoding 1.2M evidence with BGE-base (one-time ~30 min on 4060) ...")
    tok = AutoTokenizer.from_pretrained(DENSE_MODEL_NAME)
    mdl = AutoModel.from_pretrained(DENSE_MODEL_NAME).to(DEVICE).eval()
    embs = []
    for i in tqdm(range(0, len(evidence_ids), 128)):
        chunk = [evidence[e] for e in evidence_ids[i:i+128]]
        enc = tok(chunk, padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            with torch.autocast(device_type="cuda" if DEVICE.type=="cuda" else "cpu",
                                dtype=AUTOCAST_DTYPE, enabled=(DEVICE.type=="cuda")):
                out = mdl(**enc).last_hidden_state[:, 0]
            out = F.normalize(out.float(), p=2, dim=1)
        embs.append(out.cpu().numpy().astype(np.float16))
    arr = np.concatenate(embs, axis=0)
    cache.parent.mkdir(exist_ok=True, parents=True)
    np.save(cache, arr)
    del mdl; torch.cuda.empty_cache() if DEVICE.type=="cuda" else None
    return arr

evidence_bge_np = _get_bge_evidence_embeddings()
print(f"  evidence embedding shape={evidence_bge_np.shape} dtype={evidence_bge_np.dtype}")


def _encode_claims_bge(claims_dict):
    """Encode claim texts with BGE; returns (cids, embeddings fp32)."""
    tok = AutoTokenizer.from_pretrained(DENSE_MODEL_NAME)
    mdl = AutoModel.from_pretrained(DENSE_MODEL_NAME).to(DEVICE).eval()
    cids = list(claims_dict.keys())
    embs = []
    for i in range(0, len(cids), 64):
        batch_cids = cids[i:i+64]
        texts = [claims_dict[c]["claim_text"] for c in batch_cids]
        enc = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            with torch.autocast(device_type="cuda" if DEVICE.type=="cuda" else "cpu",
                                dtype=AUTOCAST_DTYPE, enabled=(DEVICE.type=="cuda")):
                out = mdl(**enc).last_hidden_state[:, 0]
            out = F.normalize(out.float(), p=2, dim=1)
        embs.append(out.cpu().numpy())
    del mdl; torch.cuda.empty_cache() if DEVICE.type=="cuda" else None
    return cids, np.concatenate(embs, axis=0)


def build_or_load_dense_pool(claims_dict, split: str):
    """Return dict[cid] -> [(eid, cosine), ...] of top-500."""
    for cache_path in [CACHE_DIR / f"{split}_dense_top500.pkl",
                       V7_CACHE / f"{split}_dense_top500.pkl"]:
        if cache_path.exists():
            print(f"  loading Dense cache: {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
    print(f"  no Dense cache for {split}; computing top-500 on GPU ...")
    cids, claim_emb = _encode_claims_bge(claims_dict)
    ev_gpu = torch.from_numpy(evidence_bge_np).to(DEVICE)  # (N, 768) fp16
    out = {}
    for i, cid in enumerate(tqdm(cids)):
        q = torch.from_numpy(claim_emb[i]).to(DEVICE).half()
        sims = (ev_gpu @ q).float().cpu().numpy()  # (N,)
        idx = np.argpartition(-sims, DENSE_CANDIDATE_K)[:DENSE_CANDIDATE_K]
        idx = idx[np.argsort(-sims[idx])]
        out[cid] = [(evidence_ids[int(j)], float(sims[j])) for j in idx]
    del ev_gpu; torch.cuda.empty_cache() if DEVICE.type=="cuda" else None
    with open(CACHE_DIR / f"{split}_dense_top500.pkl", "wb") as f:
        pickle.dump(out, f)
    return out


train_dense = build_or_load_dense_pool(train_claims, "train")
dev_dense   = build_or_load_dense_pool(dev_claims,   "dev")
test_dense  = build_or_load_dense_pool(test_claims,  "test")

dev_dense_eids = {cid: [e for e, _ in v] for cid, v in dev_dense.items()}
print(f"dev Dense recall@500 = {recall_at_k(dev_dense_eids, dev_claims):.4f}")
''')


# ---------------- Hybrid pool ----------------
md(r"""## 1.3 Hybrid pool construction

`BM25 top-500 ∪ Dense top-500` per claim. BM25 entries come first in the
union list (lexical-discriminative cases first, then paraphrastic
fillers from dense).
""")

code(r'''def build_hybrid(bm25_split, dense_split):
    out = {}
    for cid, bm in bm25_split.items():
        ds = dense_split[cid]
        seen = set(); merged = []
        for e, s in bm:
            if e not in seen: seen.add(e); merged.append((e, "bm25", s))
        for e, s in ds:
            if e not in seen: seen.add(e); merged.append((e, "dense_only", s))
        out[cid] = merged
    return out

train_hybrid = build_hybrid(train_bm25, train_dense)
dev_hybrid   = build_hybrid(dev_bm25,   dev_dense)
test_hybrid  = build_hybrid(test_bm25,  test_dense)

sizes = [len(v) for v in dev_hybrid.values()]
print(f"dev hybrid pool: min={min(sizes)} median={int(np.median(sizes))} "
      f"max={max(sizes)} mean={np.mean(sizes):.0f}")
hybrid_eids_full = {cid:[e for e,_,_ in v] for cid,v in dev_hybrid.items()}
hybrid_k = max(len(v) for v in hybrid_eids_full.values())
print(f"dev hybrid recall@{hybrid_k} = "
      f"{recall_at_k(hybrid_eids_full, dev_claims, k=hybrid_k):.4f}")
''')


# ---------------- Reranker (R2) ----------------
md(r"""## 1.4 Stage 2 — Reranker (R2)

Cross-encoder `ms-marco-MiniLM-L-12-v2`, trained with mixed negative
sampling (2 from BM25\Dense + 2 from Dense\BM25 per positive), LR=5e-6,
5 epochs, BCE loss. Loads cached state from v10 if available; otherwise
trains from scratch.
""")

code(r'''from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

rerank_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)

class RerankerPairDataset(Dataset):
    def __init__(self, examples, claims_dict):
        self.examples, self.claims = examples, claims_dict
    def __len__(self): return len(self.examples)
    def __getitem__(self, i):
        ex = self.examples[i]
        enc = rerank_tokenizer(
            self.claims[ex["cid"]]["claim_text"], evidence[ex["eid"]],
            truncation=True, padding="max_length",
            max_length=RERANKER_MAX_LEN, return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(ex["label"], dtype=torch.float32)
        return item


def build_mixed_neg_examples(claims_dict, bm25_pool, dense_pool,
                             neg_per_pos=NEGATIVES_PER_POSITIVE):
    rng = random.Random(SEED)
    examples = []
    for cid, claim in claims_dict.items():
        gold = [g for g in claim["evidences"] if g in evidence]
        gold_set = set(gold)
        for g in gold:
            examples.append({"cid": cid, "eid": g, "label": 1.0})
        B = {e for e, _ in bm25_pool[cid] if e not in gold_set and e in evidence}
        D = {e for e, _ in dense_pool[cid] if e not in gold_set and e in evidence}
        B_only = list(B - D); D_only = list(D - B)
        n_pos = max(1, len(gold))
        half = neg_per_pos // 2
        other = neg_per_pos - half
        negs = rng.sample(B_only, min(half  * n_pos, len(B_only)))
        negs += rng.sample(D_only, min(other * n_pos, len(D_only)))
        for n in negs:
            examples.append({"cid": cid, "eid": n, "label": 0.0})
    rng.shuffle(examples)
    n_pos = sum(1 for e in examples if e["label"] == 1.0)
    return examples


def _scaler():
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=USE_GRAD_SCALER)
    return torch.cuda.amp.GradScaler(enabled=USE_GRAD_SCALER)


def train_reranker(examples, claims_dict, lr=RERANKER_LR, epochs=RERANKER_EPOCHS, tag=""):
    model = AutoModelForSequenceClassification.from_pretrained(
        RERANKER_MODEL_NAME, num_labels=1, ignore_mismatched_sizes=True,
    ).to(DEVICE)
    ds = RerankerPairDataset(examples, claims_dict)
    loader = DataLoader(ds, batch_size=RERANKER_BATCH_TRAIN, shuffle=True,
                         pin_memory=(DEVICE.type=="cuda"), num_workers=0)
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY, fused=True)
    except (TypeError, RuntimeError):
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    total_steps = len(loader) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP_RATIO*total_steps), total_steps)
    scaler = _scaler()
    loss_fn = nn.BCEWithLogitsLoss()
    history = []
    for epoch in range(1, epochs + 1):
        model.train(); losses = []; t0 = time.time()
        for batch in tqdm(loader, desc=f"  {tag} ep {epoch}/{epochs}"):
            labels = batch.pop("labels").to(DEVICE, non_blocking=True)
            batch  = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type="cuda" if DEVICE.type=="cuda" else "cpu",
                                dtype=AUTOCAST_DTYPE, enabled=(DEVICE.type=="cuda")):
                logits = model(**batch).logits.squeeze(-1)
                loss = loss_fn(logits, labels)
            if USE_GRAD_SCALER:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
            sched.step(); opt.zero_grad(set_to_none=True)
            losses.append(float(loss.item()))
        avg = float(np.mean(losses)); history.append(avg)
        print(f"  {tag} epoch {epoch}: loss={avg:.4f}  time={time.time()-t0:.1f}s")
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    del model; torch.cuda.empty_cache() if DEVICE.type=="cuda" else None; gc.collect()
    return state, history


# ---- Acquire R2 weights ----
R2_LOCAL  = MODEL_DIR / "reranker_R2.pt"
R2_FROMV10 = V10_MODELS / "reranker_R2.pt"

if R2_LOCAL.exists():
    print(f"Loading R2 from {R2_LOCAL}")
    state_R2 = torch.load(R2_LOCAL, map_location="cpu")
elif R2_FROMV10.exists():
    print(f"Loading R2 from {R2_FROMV10} (v10 winner)")
    state_R2 = torch.load(R2_FROMV10, map_location="cpu")
    torch.save(state_R2, R2_LOCAL)
else:
    print("R2 state not cached; training from scratch (5 epochs, ~10 min on 4060) ...")
    ex_R2 = build_mixed_neg_examples(train_claims, train_bm25, train_dense)
    print(f"  built {len(ex_R2):,} examples")
    state_R2, _ = train_reranker(ex_R2, train_claims, tag="R2")
    torch.save(state_R2, R2_LOCAL)
''')


# ---------------- Score hybrid + select ----------------
code(r'''@torch.inference_mode()
def score_hybrid(model, claims_dict, hybrid_split, split_name):
    model.eval()
    out = {}
    for cid, claim in tqdm(list(claims_dict.items()), desc=f"score {split_name}"):
        entries = hybrid_split[cid]
        eids = [e for e, _, _ in entries]
        logits_all = []
        for start in range(0, len(eids), RERANKER_BATCH_EVAL):
            batch_eids = eids[start:start+RERANKER_BATCH_EVAL]
            enc = rerank_tokenizer(
                [claim["claim_text"]] * len(batch_eids),
                [evidence[e] for e in batch_eids],
                truncation=True, padding=True, max_length=RERANKER_MAX_LEN, return_tensors="pt",
            ).to(DEVICE, non_blocking=True)
            with torch.autocast(device_type="cuda" if DEVICE.type=="cuda" else "cpu",
                                dtype=AUTOCAST_DTYPE, enabled=(DEVICE.type=="cuda")):
                logits = model(**enc).logits.squeeze(-1)
            logits_all.extend(logits.float().cpu().tolist())
        out[cid] = {"eids": eids, "ce_logit": np.asarray(logits_all, dtype=np.float32)}
    return out


def relative_delta_select(ce_cache, delta=DELTA_FINAL,
                          kmin=MIN_FINAL_K, kmax=MAX_FINAL_K):
    out = {}
    for cid, d in ce_cache.items():
        order = np.argsort(-d["ce_logit"])
        eids  = [d["eids"][i] for i in order]
        scores = d["ce_logit"][order]
        keep = [eids[0]]; top = scores[0]
        for j in range(1, min(kmax, len(eids))):
            if scores[j] >= top - delta:
                keep.append(eids[j])
            else:
                break
        out[cid] = keep[:kmax] if len(keep) >= kmin else keep
    return out


def score_and_select(state, hybrid_split, claims_dict, split_name, cache_tag="R2"):
    cache_path = CACHE_DIR / f"{cache_tag}_{split_name}_ce.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            ce = pickle.load(f)
        print(f"  loaded CE cache {cache_path}")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            RERANKER_MODEL_NAME, num_labels=1, ignore_mismatched_sizes=True,
        ).to(DEVICE)
        model.load_state_dict(state)
        ce = score_hybrid(model, claims_dict, hybrid_split, f"{cache_tag}/{split_name}")
        del model; torch.cuda.empty_cache() if DEVICE.type=="cuda" else None; gc.collect()
        with open(cache_path, "wb") as f:
            pickle.dump(ce, f)
    return ce, relative_delta_select(ce)


print("Scoring + selecting train/dev/test with R2 ...")
train_ce_R2, train_retrieval = score_and_select(state_R2, train_hybrid, train_claims, "train")
dev_ce_R2,   dev_retrieval   = score_and_select(state_R2, dev_hybrid,   dev_claims,   "dev")
test_ce_R2,  test_retrieval  = score_and_select(state_R2, test_hybrid,  test_claims,  "test")

print(f"\ndev retrieval F (R2/hybrid/relative-δ δ={DELTA_FINAL}) = "
      f"{retrieval_F(dev_retrieval, dev_claims):.4f}")
print(f"average evidence per claim: train={np.mean([len(v) for v in train_retrieval.values()]):.2f}  "
      f"dev={np.mean([len(v) for v in dev_retrieval.values()]):.2f}  "
      f"test={np.mean([len(v) for v in test_retrieval.values()]):.2f}")
''')


# ============================================================ Section 2
md(r"""# 2. Model Implementation
(You can add as many code blocks and text blocks as you need. However, YOU SHOULD NOT MODIFY the section title)

The classifier is an **ensemble of two branches**:
- Branch A (joint): a cross-encoder taking `[CLS] claim [SEP] ev1 [SEP] ev2 ...` and outputting 4 logits.
- Branch B (stance): a frozen 3-class NLI cross-encoder scoring each (claim, ev_i) pair separately, then aggregating to 4 logits through a tiny MLP on 9 hand-engineered statistics.

The α coefficient mixing them is selected on dev.
""")


# ---------------- Joint classifier ----------------
md(r"""## 2.1 Joint classifier

Backbone: `cross-encoder/ms-marco-MiniLM-L-12-v2`. `max_len=512` (256
truncated the concatenated evidence). Class weighting W0 (uniform): per
v9 Stage 5, this gave the best DISPUTED recall while accepting only a
0.006 drop in A vs the per-class optimum.
""")

code(r'''cls_tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL_NAME)

def build_classifier_evidence(cid, claim, retrieval_dict, max_k=MAX_FINAL_K,
                              include_gold=True):
    """Train-time: gold ∪ retrieved, dedup, cap at max_k. Inference: retrieved only."""
    gold = list(claim.get("evidences", [])) if include_gold else []
    retrieved = list(retrieval_dict.get(cid, []))
    seen, chosen = set(), []
    for eid in gold + retrieved:
        if eid in seen or eid not in evidence:
            continue
        seen.add(eid); chosen.append(eid)
        if len(chosen) >= max_k:
            break
    if not chosen and retrieved:
        chosen = retrieved[:1]
    return chosen


class ClaimEvidenceDataset(Dataset):
    def __init__(self, claims_dict, retrieval_dict, include_gold, max_len=CLASSIFIER_MAX_LEN):
        self.cids = list(claims_dict.keys())
        self.claims, self.retrieval = claims_dict, retrieval_dict
        self.include_gold = include_gold; self.max_len = max_len
        self.has_label = "claim_label" in next(iter(claims_dict.values()))
    def __len__(self): return len(self.cids)
    def __getitem__(self, i):
        cid = self.cids[i]; c = self.claims[cid]
        ev_ids = build_classifier_evidence(cid, c, self.retrieval, include_gold=self.include_gold)
        if not ev_ids:
            ev_ids = [next(iter(evidence.keys()))]
        sep = f" {cls_tokenizer.sep_token} "
        ev_text = sep.join(evidence[e] for e in ev_ids)
        enc = cls_tokenizer(c["claim_text"], ev_text, truncation=True,
                             padding="max_length", max_length=self.max_len, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.has_label:
            item["labels"] = torch.tensor(LABEL2ID[c["claim_label"]], dtype=torch.long)
        return item


def train_joint_classifier(train_claims_d, train_retr, dev_claims_d, dev_retr,
                            epochs=CLASSIFIER_EPOCHS, lr=CLASSIFIER_LR, tag="joint"):
    """Returns (best_state, best_dev_metrics, dev_logits_at_best)."""
    model = AutoModelForSequenceClassification.from_pretrained(
        CLASSIFIER_MODEL_NAME, num_labels=len(LABELS),
        id2label=ID2LABEL, label2id=LABEL2ID, ignore_mismatched_sizes=True,
    ).to(DEVICE)
    train_ds = ClaimEvidenceDataset(train_claims_d, train_retr, include_gold=True)
    train_loader = DataLoader(train_ds, batch_size=CLASSIFIER_BATCH_TRAIN, shuffle=True,
                               pin_memory=(DEVICE.type=="cuda"))
    dev_ds = ClaimEvidenceDataset(dev_claims_d, dev_retr, include_gold=False)
    dev_loader = DataLoader(dev_ds, batch_size=CLASSIFIER_BATCH_EVAL, shuffle=False,
                             pin_memory=(DEVICE.type=="cuda"))
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY, fused=True)
    except (TypeError, RuntimeError):
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP_RATIO*total_steps), total_steps)
    scaler = _scaler()
    loss_fn = nn.CrossEntropyLoss()  # W0: uniform weights

    best = {"H": -1.0, "state": None, "dev_logits": None, "epoch": -1}
    for epoch in range(1, epochs + 1):
        model.train(); losses = []; t0 = time.time()
        for batch in tqdm(train_loader, desc=f"  {tag} ep {epoch}/{epochs}"):
            batch  = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(device_type="cuda" if DEVICE.type=="cuda" else "cpu",
                                dtype=AUTOCAST_DTYPE, enabled=(DEVICE.type=="cuda")):
                logits = model(**batch).logits
                loss = loss_fn(logits, labels)
            if USE_GRAD_SCALER:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
            sched.step(); opt.zero_grad(set_to_none=True)
            losses.append(float(loss.item()))
        # ---- dev eval ----
        model.eval()
        dev_logits = []
        with torch.inference_mode():
            for batch in dev_loader:
                batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items() if k != "labels"}
                with torch.autocast(device_type="cuda" if DEVICE.type=="cuda" else "cpu",
                                    dtype=AUTOCAST_DTYPE, enabled=(DEVICE.type=="cuda")):
                    out = model(**batch).logits
                dev_logits.append(out.float().cpu().numpy())
        dev_logits = np.concatenate(dev_logits, axis=0)
        dev_preds = {dev_ds.cids[i]: ID2LABEL[int(dev_logits[i].argmax())]
                     for i in range(len(dev_ds.cids))}
        preds = {cid: {"claim_label": dev_preds[cid], "evidences": dev_retr[cid]}
                 for cid in dev_claims_d}
        m = evaluate_submission(preds, dev_claims_d, verbose=False)
        pred_counts = Counter(dev_preds.values())
        summary = " ".join(f"{l[0]}:{pred_counts.get(l,0)}" for l in LABELS)
        print(f"  {tag} epoch {epoch}: loss={np.mean(losses):.4f}  F={m['F']:.4f}  "
              f"A={m['A']:.4f}  H={m['H']:.4f}  preds=({summary})  {time.time()-t0:.1f}s")
        if m["H"] > best["H"]:
            best = {**m, "state": {k: v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    "dev_logits": dev_logits, "epoch": epoch, "dev_cids": list(dev_ds.cids)}
    del model; torch.cuda.empty_cache() if DEVICE.type=="cuda" else None; gc.collect()
    return best


print("\n=== Training joint classifier on TRAIN, eval on DEV ===")
joint_dev = train_joint_classifier(train_claims, train_retrieval, dev_claims, dev_retrieval,
                                    tag="joint-train")
print(f"Best joint dev epoch={joint_dev['epoch']} F={joint_dev['F']:.4f} "
      f"A={joint_dev['A']:.4f} H={joint_dev['H']:.4f}")
torch.save(joint_dev["state"], MODEL_DIR / "joint_devbest.pt")
''')


# ---------------- NLI stance scoring ----------------
md(r"""## 2.2 Per-evidence NLI stance scoring (frozen)

For every (claim, retrieved evidence) pair, run a frozen 3-class NLI
cross-encoder and store its softmax distribution over
`[contradiction, entailment, neutral]`.
Following v8 §6.4: we do **not** reinit the head into 4 classes — the
NLI inductive bias would be destroyed. The original 3-class output is
read directly and aggregated downstream.

Input convention: `(premise=evidence, hypothesis=claim)`. So
*contradiction* directly maps to "evidence refutes claim", *entailment*
to "evidence supports claim".
""")

code(r'''# Label order for cross-encoder/nli-MiniLM2-L6-H768:
#   index 0 -> contradiction,  1 -> entailment,  2 -> neutral
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME).to(DEVICE).eval()
for p in nli_model.parameters():
    p.requires_grad = False

@torch.inference_mode()
def nli_score_one_claim(claim_text, evidence_eids):
    """Return (K, 3) ndarray of [P_contradict, P_entail, P_neutral] for each evidence."""
    if not evidence_eids:
        return np.zeros((0, 3), dtype=np.float32)
    probs_all = []
    for start in range(0, len(evidence_eids), NLI_BATCH):
        batch_eids = evidence_eids[start:start+NLI_BATCH]
        ev_texts = [evidence[e] for e in batch_eids]
        enc = nli_tokenizer(
            ev_texts,                                  # premise = evidence
            [claim_text] * len(batch_eids),            # hypothesis = claim
            truncation=True, padding=True, max_length=NLI_MAX_LEN, return_tensors="pt",
        ).to(DEVICE)
        with torch.autocast(device_type="cuda" if DEVICE.type=="cuda" else "cpu",
                            dtype=AUTOCAST_DTYPE, enabled=(DEVICE.type=="cuda")):
            logits = nli_model(**enc).logits  # (B, 3)
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        probs_all.append(probs)
    return np.concatenate(probs_all, axis=0)  # (K, 3)


def compute_nli_cache(claims_dict, retrieval_dict, split_name):
    cache_path = CACHE_DIR / f"nli_{split_name}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            out = pickle.load(f)
        print(f"  loaded NLI cache {cache_path}: {len(out)} claims")
        return out
    out = {}
    for cid, claim in tqdm(list(claims_dict.items()), desc=f"NLI {split_name}"):
        ev_ids = retrieval_dict.get(cid, [])
        out[cid] = nli_score_one_claim(claim["claim_text"], ev_ids)
    with open(cache_path, "wb") as f:
        pickle.dump(out, f)
    return out


print("Computing NLI stance for train/dev/test retrieved evidence ...")
train_nli = compute_nli_cache(train_claims, train_retrieval, "train")
dev_nli   = compute_nli_cache(dev_claims,   dev_retrieval,   "dev")
test_nli  = compute_nli_cache(test_claims,  test_retrieval,  "test")
print(f"NLI per-claim shapes (train sample): {next(iter(train_nli.values())).shape}")
''')


# ---------------- Stance features + MLP ----------------
md(r"""## 2.3 Stance aggregator (9-feature MLP)

For each claim, the (K, 3) NLI tensor is summarised into 9 statistics:

| Feature | Captures |
|---|---|
| mean_entail, mean_contradict, mean_neutral | base rates |
| max_entail, max_contradict | the single strongest agreeing / disagreeing evidence |
| std_entail, std_contradict | **variance — DISPUTED's defining signature** |
| count(entail > 0.5), count(contradict > 0.5) | how many evidences strongly take each side |

Aggregator: `Linear(9 → 16) → ReLU → Dropout(0.3) → Linear(16 → 4)`.
About 200 trainable params; the 4-class output is **learnt**, not
hand-mapped from stance categories — this respects the no-rules
constraint while letting the NLI signal flow through.
""")

code(r'''def stance_features(nli_mat):
    """nli_mat: (K, 3) array of [P_contradict, P_entail, P_neutral]."""
    if len(nli_mat) == 0:
        return np.zeros(9, dtype=np.float32)
    c, e, n = nli_mat[:, 0], nli_mat[:, 1], nli_mat[:, 2]
    feats = np.array([
        e.mean(), c.mean(), n.mean(),
        e.max(),  c.max(),
        float(np.std(e)) if len(e) > 1 else 0.0,
        float(np.std(c)) if len(c) > 1 else 0.0,
        float((e > 0.5).sum()),
        float((c > 0.5).sum()),
    ], dtype=np.float32)
    return feats


class StanceMLP(nn.Module):
    def __init__(self, in_dim=9, hidden=STANCE_MLP_HIDDEN, n_classes=len(LABELS),
                 dropout=STANCE_MLP_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )
    def forward(self, x):
        return self.net(x)


def build_stance_xy(claims_dict, nli_cache):
    cids = list(claims_dict.keys())
    X = np.stack([stance_features(nli_cache[cid]) for cid in cids], axis=0)
    has_label = "claim_label" in next(iter(claims_dict.values()))
    y = (np.array([LABEL2ID[claims_dict[cid]["claim_label"]] for cid in cids], dtype=np.int64)
         if has_label else None)
    return cids, X, y


def train_stance_aggregator(train_claims_d, train_nli_c, dev_claims_d, dev_nli_c, tag="stance"):
    train_cids, X_train, y_train = build_stance_xy(train_claims_d, train_nli_c)
    dev_cids,   X_dev,   y_dev   = build_stance_xy(dev_claims_d,   dev_nli_c)
    X_train_t = torch.from_numpy(X_train).to(DEVICE)
    y_train_t = torch.from_numpy(y_train).to(DEVICE)
    X_dev_t   = torch.from_numpy(X_dev).to(DEVICE)
    y_dev_t   = torch.from_numpy(y_dev).to(DEVICE)
    mlp = StanceMLP().to(DEVICE)
    opt = torch.optim.AdamW(mlp.parameters(), lr=STANCE_LR, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    best = {"H": -1.0, "state": None, "epoch": -1, "dev_logits": None, "dev_cids": dev_cids}
    n = X_train_t.shape[0]
    for epoch in range(1, STANCE_EPOCHS + 1):
        mlp.train()
        perm = torch.randperm(n, device=DEVICE)
        losses = []
        for s in range(0, n, STANCE_BATCH):
            idx = perm[s:s+STANCE_BATCH]
            xb, yb = X_train_t[idx], y_train_t[idx]
            logits = mlp(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.item()))
        # dev eval (use dev retrieval already inside build_stance_xy via nli_cache)
        mlp.eval()
        with torch.inference_mode():
            dev_logits = mlp(X_dev_t).float().cpu().numpy()
        preds = {dev_cids[i]: {
            "claim_label": ID2LABEL[int(dev_logits[i].argmax())],
            "evidences":   dev_retrieval[dev_cids[i]],
        } for i in range(len(dev_cids))}
        m = evaluate_submission(preds, dev_claims_d, verbose=False)
        if epoch % 10 == 0 or epoch == 1 or m["H"] > best["H"]:
            print(f"  {tag} epoch {epoch:3d}: loss={np.mean(losses):.4f}  "
                  f"F={m['F']:.4f} A={m['A']:.4f} H={m['H']:.4f}")
        if m["H"] > best["H"]:
            best = {**m, "state": {k: v.detach().cpu().clone() for k,v in mlp.state_dict().items()},
                    "epoch": epoch, "dev_logits": dev_logits, "dev_cids": dev_cids}
    print(f"  {tag} BEST epoch {best['epoch']}: F={best['F']:.4f} A={best['A']:.4f} H={best['H']:.4f}")
    return best, mlp


print("\n=== Training stance MLP on TRAIN, eval on DEV ===")
stance_dev, _ = train_stance_aggregator(train_claims, train_nli, dev_claims, dev_nli,
                                          tag="stance-train")
torch.save(stance_dev["state"], MODEL_DIR / "stance_devbest.pt")
''')


# ---------------- Ensemble ----------------
md(r"""## 2.4 Ensemble of joint + stance

Soft-vote at the probability level:

```
p_final(c) = α · softmax(joint_logits)(c) + (1 − α) · softmax(stance_logits)(c)
pred = argmax_c p_final
```

`α` is selected on dev by scanning the grid `{0.00, 0.05, …, 1.00}` and
keeping the value that maximises the harmonic-mean H.
""")

code(r'''def ensemble_preds(joint_logits, stance_logits, alpha, cids, retr):
    """Returns dict[cid] -> prediction (label + evidences)."""
    p_joint  = torch.softmax(torch.from_numpy(joint_logits),  dim=-1).numpy()
    p_stance = torch.softmax(torch.from_numpy(stance_logits), dim=-1).numpy()
    p_final  = alpha * p_joint + (1.0 - alpha) * p_stance
    out = {}
    for i, cid in enumerate(cids):
        out[cid] = {"claim_label": ID2LABEL[int(p_final[i].argmax())],
                    "evidences":   retr[cid]}
    return out


# Sanity: both branches share the same dev claim order, but let's align defensively.
joint_cids  = joint_dev["dev_cids"];  joint_logits  = joint_dev["dev_logits"]
stance_cids = stance_dev["dev_cids"]; stance_logits = stance_dev["dev_logits"]
assert joint_cids == stance_cids, "dev cid order mismatch between branches"

rows = []
for a in ALPHA_GRID:
    preds = ensemble_preds(joint_logits, stance_logits, a, joint_cids, dev_retrieval)
    m = evaluate_submission(preds, dev_claims, verbose=False)
    rows.append({"alpha": a, **m})
alpha_df = pd.DataFrame(rows).sort_values("H", ascending=False).reset_index(drop=True)
display(alpha_df.head(10))

BEST_ALPHA = float(alpha_df.iloc[0]["alpha"])
print(f"\nBEST_ALPHA = {BEST_ALPHA}  →  "
      f"F={alpha_df.iloc[0]['F']:.4f} A={alpha_df.iloc[0]['A']:.4f} H={alpha_df.iloc[0]['H']:.4f}")
print(f"\nReference rows:")
for tag, (F, A, H) in [
    ("joint-only (α=1)",  (joint_dev["F"],  joint_dev["A"],  joint_dev["H"])),
    ("stance-only (α=0)", (stance_dev["F"], stance_dev["A"], stance_dev["H"])),
]:
    print(f"  {tag}:  F={F:.4f}  A={A:.4f}  H={H:.4f}")
''')


# ---------------- Dev evaluation summary ----------------
md(r"""## 2.5 Dev evaluation summary

All hyperparameters above are now locked. Next: retrain reranker +
joint + stance on `train ∪ dev` with the same hyperparams, then predict
test.
""")

code(r'''# Materialise the dev predictions at BEST_ALPHA and run eval.py for the record.
dev_preds_devmodel = ensemble_preds(joint_logits, stance_logits, BEST_ALPHA,
                                     joint_cids, dev_retrieval)
write_predictions(dev_preds_devmodel, OUTPUT_DIR / "dev-predictions-devmodel.json")
print("Dev predictions (dev-trained model) written.")
print("Final dev metrics:")
_ = evaluate_submission(dev_preds_devmodel, dev_claims)

eval_py = Path("eval.py")
if eval_py.exists():
    import subprocess
    cmd = [sys.executable, "eval.py",
           "--predictions", str(OUTPUT_DIR / "dev-predictions-devmodel.json"),
           "--groundtruth", str(DATA_DIR / "dev-claims.json")]
    print("\nOfficial eval.py:")
    completed = subprocess.run(cmd, text=True, capture_output=True)
    print(completed.stdout)
    if completed.stderr: print(completed.stderr)


# Per-class diagnostics (dev model)
def confusion_matrix_df(gold_claims, preds):
    mat = pd.DataFrame(0, index=LABELS, columns=LABELS)
    for cid, c in gold_claims.items():
        mat.loc[c["claim_label"], preds[cid]["claim_label"]] += 1
    return mat

print("\nConfusion matrix (dev, ensemble α=%.2f):" % BEST_ALPHA)
display(confusion_matrix_df(dev_claims, dev_preds_devmodel))

per_class = []
for lbl in LABELS:
    cids = [cid for cid, c in dev_claims.items() if c["claim_label"] == lbl]
    acc  = float(np.mean([dev_preds_devmodel[cid]["claim_label"] == lbl for cid in cids])) if cids else 0.0
    rF   = float(np.mean([evidence_f1(dev_retrieval[cid], dev_claims[cid]["evidences"]) for cid in cids])) if cids else 0.0
    per_class.append({"label": lbl, "n": len(cids), "class_acc": acc, "retrieval_F": rF})
display(pd.DataFrame(per_class))
''')


# ============================================================ Section 3
md(r"""# 3. Testing and Evaluation
(You can add as many code blocks and text blocks as you need. However, YOU SHOULD NOT MODIFY the section title)

## 3.1 Train + dev joint retraining (final model)

All hyperparameters (`BEST_ALPHA`, classifier max_len, weighting,
reranker LR/epochs, selector δ) were chosen on dev above. Now retrain
reranker + joint + stance on the union `train ∪ dev` and predict test.
**No further tuning** on the combined set — its metric is meaningless
(training data) and would leak dev into the model selection.
""")

code(r'''combined_claims = {**train_claims, **dev_claims}
combined_bm25    = {**train_bm25,   **dev_bm25}
combined_dense   = {**train_dense,  **dev_dense}
combined_hybrid  = {**train_hybrid, **dev_hybrid}

print(f"combined claims: {len(combined_claims):,}")

# ----- (1) retrain reranker on combined -----
print("\n=== Retraining reranker R2 on TRAIN ∪ DEV ===")
ex_R2_combined = build_mixed_neg_examples(combined_claims, combined_bm25, combined_dense)
print(f"  built {len(ex_R2_combined):,} examples")
state_R2_combined, _ = train_reranker(ex_R2_combined, combined_claims, tag="R2-combined")
torch.save(state_R2_combined, MODEL_DIR / "reranker_R2_combined.pt")
''')


code(r'''# ----- (2) re-score combined + test with the combined R2; rebuild retrieval -----
print("\nScoring combined + test with combined R2 ...")
# Force a fresh score: delete any stale per-claim cache that used dev-only weights.
for split in ("combined", "test"):
    p = CACHE_DIR / f"R2combined_{split}_ce.pkl"
    if p.exists(): p.unlink()
combined_ce, combined_retrieval = score_and_select(
    state_R2_combined, combined_hybrid, combined_claims, "combined", cache_tag="R2combined")
test_ce_final, test_retrieval_final = score_and_select(
    state_R2_combined, test_hybrid, test_claims, "test", cache_tag="R2combined")

# Diagnostic: combined retrieval F vs gold (training F, just a sanity check — model has seen it).
print(f"\ncombined retrieval F (in-sample, sanity only): "
      f"{retrieval_F(combined_retrieval, combined_claims):.4f}")
print(f"avg evidence per test claim: {np.mean([len(v) for v in test_retrieval_final.values()]):.2f}")
''')


code(r'''# ----- (3) retrain joint classifier on combined -----
# Note: there's no held-out dev now; we run a fixed number of epochs equal to
# the joint_dev best epoch + 1 (small safety margin), no early stopping.
JOINT_FINAL_EPOCHS = max(joint_dev["epoch"], 2)
print(f"\n=== Retraining joint classifier on TRAIN ∪ DEV for {JOINT_FINAL_EPOCHS} epochs ===")

def train_joint_no_eval(train_claims_d, train_retr, epochs, lr=CLASSIFIER_LR, tag="joint-combined"):
    model = AutoModelForSequenceClassification.from_pretrained(
        CLASSIFIER_MODEL_NAME, num_labels=len(LABELS),
        id2label=ID2LABEL, label2id=LABEL2ID, ignore_mismatched_sizes=True,
    ).to(DEVICE)
    ds = ClaimEvidenceDataset(train_claims_d, train_retr, include_gold=True)
    loader = DataLoader(ds, batch_size=CLASSIFIER_BATCH_TRAIN, shuffle=True,
                         pin_memory=(DEVICE.type=="cuda"))
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY, fused=True)
    except (TypeError, RuntimeError):
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    total_steps = len(loader) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP_RATIO*total_steps), total_steps)
    scaler = _scaler()
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        model.train(); losses = []; t0 = time.time()
        for batch in tqdm(loader, desc=f"  {tag} ep {epoch}/{epochs}"):
            batch  = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(device_type="cuda" if DEVICE.type=="cuda" else "cpu",
                                dtype=AUTOCAST_DTYPE, enabled=(DEVICE.type=="cuda")):
                logits = model(**batch).logits
                loss = loss_fn(logits, labels)
            if USE_GRAD_SCALER:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
            sched.step(); opt.zero_grad(set_to_none=True)
            losses.append(float(loss.item()))
        print(f"  {tag} epoch {epoch}: loss={np.mean(losses):.4f}  {time.time()-t0:.1f}s")
    state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
    del model; torch.cuda.empty_cache() if DEVICE.type=="cuda" else None; gc.collect()
    return state

joint_combined_state = train_joint_no_eval(combined_claims, combined_retrieval, JOINT_FINAL_EPOCHS)
torch.save(joint_combined_state, MODEL_DIR / "joint_combined.pt")
''')


code(r'''# ----- (4) re-compute NLI features for combined + test (retrieval changed!) -----
# (test retrieval also changed because R2 was retrained.)
for split in ("combined", "test"):
    p = CACHE_DIR / f"nli_{split}.pkl"
    if p.exists(): p.unlink()

print("\nRecomputing NLI for combined + test retrieved evidence ...")
combined_nli = compute_nli_cache(combined_claims, combined_retrieval, "combined")
test_nli_final = compute_nli_cache(test_claims,    test_retrieval_final, "test")
''')


code(r'''# ----- (5) retrain stance MLP on combined (no eval — fixed epochs) -----
STANCE_FINAL_EPOCHS = max(stance_dev["epoch"], 20)
print(f"\n=== Retraining stance MLP on TRAIN ∪ DEV for {STANCE_FINAL_EPOCHS} epochs ===")

cids_combined, X_combined, y_combined = build_stance_xy(combined_claims, combined_nli)
X_combined_t = torch.from_numpy(X_combined).to(DEVICE)
y_combined_t = torch.from_numpy(y_combined).to(DEVICE)
stance_combined = StanceMLP().to(DEVICE)
opt_s = torch.optim.AdamW(stance_combined.parameters(), lr=STANCE_LR, weight_decay=1e-3)
loss_fn_s = nn.CrossEntropyLoss()
for epoch in range(1, STANCE_FINAL_EPOCHS + 1):
    stance_combined.train()
    perm = torch.randperm(X_combined_t.shape[0], device=DEVICE)
    losses = []
    for s in range(0, X_combined_t.shape[0], STANCE_BATCH):
        idx = perm[s:s+STANCE_BATCH]
        logits = stance_combined(X_combined_t[idx])
        loss = loss_fn_s(logits, y_combined_t[idx])
        opt_s.zero_grad(); loss.backward(); opt_s.step()
        losses.append(float(loss.item()))
    if epoch % 10 == 0 or epoch == STANCE_FINAL_EPOCHS:
        print(f"  stance-combined epoch {epoch}: loss={np.mean(losses):.4f}")
torch.save(stance_combined.state_dict(), MODEL_DIR / "stance_combined.pt")
''')


# ---------------- Test predictions ----------------
md(r"""## 3.2 Test predictions

Apply the train+dev models to the test split, ensemble at the locked
`BEST_ALPHA`, write `test-output.json`.
""")

code(r'''def predict_test(joint_state, stance_module, test_claims_d, test_retr, test_nli_d,
                  alpha):
    # ----- joint logits on test -----
    model = AutoModelForSequenceClassification.from_pretrained(
        CLASSIFIER_MODEL_NAME, num_labels=len(LABELS),
        id2label=ID2LABEL, label2id=LABEL2ID, ignore_mismatched_sizes=True,
    ).to(DEVICE)
    model.load_state_dict(joint_state); model.eval()
    test_ds = ClaimEvidenceDataset(test_claims_d, test_retr, include_gold=False)
    loader = DataLoader(test_ds, batch_size=CLASSIFIER_BATCH_EVAL, shuffle=False,
                         pin_memory=(DEVICE.type=="cuda"))
    test_logits = []
    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items() if k != "labels"}
            with torch.autocast(device_type="cuda" if DEVICE.type=="cuda" else "cpu",
                                dtype=AUTOCAST_DTYPE, enabled=(DEVICE.type=="cuda")):
                out = model(**batch).logits
            test_logits.append(out.float().cpu().numpy())
    test_joint_logits = np.concatenate(test_logits, axis=0)
    del model; torch.cuda.empty_cache() if DEVICE.type=="cuda" else None
    # ----- stance logits on test -----
    test_cids, X_test, _ = build_stance_xy(test_claims_d, test_nli_d)
    with torch.inference_mode():
        stance_logits_test = stance_module(torch.from_numpy(X_test).to(DEVICE)).float().cpu().numpy()
    assert test_cids == test_ds.cids
    # ----- ensemble -----
    return ensemble_preds(test_joint_logits, stance_logits_test, alpha, test_cids, test_retr)


print("\nPredicting test with combined-trained ensemble ...")
test_predictions = predict_test(
    joint_combined_state, stance_combined,
    test_claims, test_retrieval_final, test_nli_final,
    alpha=BEST_ALPHA,
)
write_predictions(test_predictions, OUTPUT_DIR / "test-output.json")
print(f"Wrote {OUTPUT_DIR / 'test-output.json'} with {len(test_predictions)} predictions")

pred_dist = Counter(p["claim_label"] for p in test_predictions.values())
print("\nTest prediction label distribution:")
for lbl in LABELS:
    print(f"  {lbl:20s} {pred_dist.get(lbl, 0):4d} ({pred_dist.get(lbl, 0)/len(test_predictions):6.2%})")
''')


# ---------------- Summary ----------------
md(r"""## 3.3 Final summary
""")

code(r'''print("=" * 60)
print("Final configuration (locked on dev, retrained on train ∪ dev):")
print(f"  Retrieval     : BM25(PP3, k1={BM25_K1}, b={BM25_B}) ∪ Dense(BGE-base) → R2 → relative-δ δ={DELTA_FINAL}")
print(f"  Joint backbone: {CLASSIFIER_MODEL_NAME}  max_len={CLASSIFIER_MAX_LEN}  weighting=W0")
print(f"  Stance        : {NLI_MODEL_NAME} (frozen) + 9-stat MLP {STANCE_MLP_HIDDEN}h")
print(f"  Ensemble α    : {BEST_ALPHA}")
print()
print("Dev metrics (dev-only model, for reproducibility):")
print(f"  F = {alpha_df.iloc[0]['F']:.6f}")
print(f"  A = {alpha_df.iloc[0]['A']:.6f}")
print(f"  H = {alpha_df.iloc[0]['H']:.6f}")
print()
print(f"Test predictions: {OUTPUT_DIR / 'test-output.json'}")
''')


# ============================================================ Emit notebook
def _cell(kind: str, src: str, idx: int) -> dict:
    payload = {
        "cell_type": kind, "id": f"cell-{idx:03d}",
        "metadata": {}, "source": src.splitlines(keepends=True),
    }
    if kind == "code":
        payload["execution_count"] = None
        payload["outputs"] = []
    return payload

nb = {
    "cells": [_cell(k, s, i) for i, (k, s) in enumerate(CELLS)],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

out_path = Path(__file__).parent / "nlp_submission_v11.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {out_path}  ({len(CELLS)} cells)")
