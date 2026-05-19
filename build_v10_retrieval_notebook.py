"""Build nlp_retrieval_v10.ipynb — retrieval-only ablation notebook.

Pipeline ends at scoring strategy; no classifier, no test-output.

Reuses caches:
  outputs_notebook_v9/cache/{train,dev,test}_bm25_top500.pkl     (PP3 + R-both)
  outputs_notebook_v7/cache/{train,dev,test}_dense_top500.pkl    (BGE-base)
  outputs_notebook_v7/cache/evidence_bge_fp16.npy                (1.86 GB)
  outputs_notebook_v9/models/reranker_best.pt                    (v9 reranker = R0)
  outputs_notebook_v9/cache/dev_ce_scores.pkl                    (R0 dev scores on BM25 pool)

Reranker variants compared:
  R0  — v9 reranker (LR=1e-5, 3ep, BM25 random negs)               [load only]
  R1  — low LR baseline (LR=5e-6, 5ep, BM25 random negs)
  R2  — mixed negatives, no F-cos filter
  R3  — mixed negatives + F-cos filter
  R4  — hard-negative mining iter-1 (starting from R3)              [optional]

Scoring pools per reranker:  BM25-top-500  (S-B)  and  BM25 ∪ Dense  (S-H)
Selection / fusion strategies swept on each (reranker, pool) cache.
"""
from __future__ import annotations
import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []
def md(text: str) -> None: CELLS.append(("markdown", text))
def code(text: str) -> None: CELLS.append(("code", text))


# ============================================================ Title
md(r"""# v10 — Retrieval-stage ablation

Focused comparison of the retrieval pipeline. Pipeline ends at the
reranker + scoring/selection step; no classifier, no test predictions.

**What's varied**

| Axis | Values |
|---|---|
| Reranker training | R0 (v9, loaded) / R1 (low LR) / R2 (mixed neg) / R3 (mixed + F-cos) / R4 (R3 + hard-neg mining) |
| Reranker scoring pool | S-B (BM25 top-500) / S-H (BM25 ∪ Dense top-500) |
| Selection strategy | fixed-K / threshold / relative-δ / score-fusion 2-feat / score-fusion 3-feat / RRF |

**What's fixed (carried over from v9)**

- BM25 preprocessing + params: PP3 (lowercase + stopwords + Porter) × R-both (k1=1.2, b=0.5)
- Dense retrieval model: `BAAI/bge-base-en-v1.5` (cached embeddings)
- Reranker backbone: `cross-encoder/ms-marco-MiniLM-L-12-v2` (~33M params)
- Reranker max_len=256, batch=32 train / 128 eval
- Negatives per positive: 4

**Local metric**: dev evidence retrieval F (per-claim F1, averaged). This is
what selectors are tuned on; F is also the leaderboard-ranking metric's
first half.
""")

md(r"""## Key question being tested

v8 forbids the reranker from scoring dense-only candidates because the
training distribution was BM25-only — feeding dense-only items at inference
crashes dev F from 0.22 to 0.04 (v8 §6.1). v8's workaround was to keep
dense as a *selector feature* only.

**This notebook tests the principled alternative**: train the reranker on
mixed BM25 + Dense negatives (with optional false-negative filtering) so
that hybrid-pool scoring no longer goes out-of-distribution. If R2/R3 with
pool S-H matches or beats v9 (R0/S-B), we can ditch the fusion hack.
""")


# ============================================================ Section 1
md(r"""# 1. Setup""")

code(r'''import importlib.util, subprocess, sys
for import_name, pip_name in {
    "bm25s": "bm25s", "transformers": "transformers", "accelerate": "accelerate",
    "sklearn": "scikit-learn", "nltk": "nltk",
}.items():
    if importlib.util.find_spec(import_name) is None:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])
        except Exception as e:
            print(f"Auto-install of {pip_name} failed: {e}")
''')

code(r'''from pathlib import Path
import json, random, time, pickle, gc, math
from collections import Counter
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DATA_DIR    = Path("data")
V7_CACHE    = Path("outputs_notebook_v7/cache")
V9_CACHE    = Path("outputs_notebook_v9/cache")
V9_MODELS   = Path("outputs_notebook_v9/models")
OUTPUT_DIR  = Path("outputs_notebook_v10")
CACHE_DIR   = OUTPUT_DIR / "cache"
MODEL_DIR   = OUTPUT_DIR / "models"
for d in (OUTPUT_DIR, CACHE_DIR, MODEL_DIR):
    d.mkdir(exist_ok=True, parents=True)

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

# ---- Constants ----
BM25_CANDIDATE_K = 500
DENSE_CANDIDATE_K = 500
MIN_FINAL_K, MAX_FINAL_K = 1, 5
RERANKER_MODEL_NAME    = "cross-encoder/ms-marco-MiniLM-L-12-v2"
DENSE_MODEL_NAME       = "BAAI/bge-base-en-v1.5"
RERANKER_MAX_LEN       = 256
RERANKER_BATCH_TRAIN   = 32
RERANKER_BATCH_EVAL    = 128
NEGATIVES_PER_POSITIVE = 4
WEIGHT_DECAY           = 0.01
WARMUP_RATIO           = 0.1
GRAD_CLIP              = 1.0
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

print(f"train={len(train_claims)}  dev={len(dev_claims)}  test={len(test_claims)}  evidence={len(evidence)}")
''')

code(r'''# Metrics: per-claim F1 (eval.py style) and recall@K.
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
''')


# ============================================================ Section 2 BM25
md(r"""# 2. Stage 1 — BM25 retrieval (loaded from v9)

v9 selected `PP3 + R-both` (lowercase + Porter stem + stopwords;
BM25 k1=1.2, b=0.5) with dev recall@500 = 0.678. We reuse those cached
top-500 lists directly — no resweep.
""")

code(r'''def load_bm25_cache(split: str):
    path = V9_CACHE / f"{split}_bm25_top500.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run nlp_experiments_v9.ipynb's Stage 1 first "
            "to produce the BM25 caches."
        )
    with open(path, "rb") as f:
        return pickle.load(f)

train_bm25 = load_bm25_cache("train")
dev_bm25   = load_bm25_cache("dev")
test_bm25  = load_bm25_cache("test")

# Sanity: dev recall@500 should match v9's 0.678
dev_bm25_eids = {cid: [e for e, _ in pairs] for cid, pairs in dev_bm25.items()}
print(f"dev recall@500 (BM25 reuse) = {recall_at_k(dev_bm25_eids, dev_claims):.4f}")
''')


# ============================================================ Section 3 Dense
md(r"""# 3. Stage 1b — Dense retrieval (BGE cache from v7)

`BAAI/bge-base-en-v1.5` embeddings of the full 1.2 M evidence corpus
(fp16, 1.86 GB) and pre-computed top-500 lists per split.  Encoding the
corpus from scratch costs ~30–45 min on a 4060 — these caches avoid that.

We also build the **hybrid pool** per claim — `BM25 top-500 ∪ Dense top-500`
— and precompute dense cosines for every (claim, candidate) pair in that
pool, so fusion / RRF strategies can look them up without re-encoding.
""")

code(r'''BGE_EVIDENCE_PATH = V7_CACHE / "evidence_bge_fp16.npy"
if not BGE_EVIDENCE_PATH.exists():
    raise FileNotFoundError(
        f"{BGE_EVIDENCE_PATH} missing. This 1.86 GB cache must come from "
        "v7's Stage-1b dense indexing run."
    )
print(f"Loading BGE evidence embeddings from {BGE_EVIDENCE_PATH} ...")
t0 = time.time()
evidence_bge_np = np.load(BGE_EVIDENCE_PATH)  # (N, 768) fp16, L2-normalised
print(f"  shape={evidence_bge_np.shape}  dtype={evidence_bge_np.dtype}  "
      f"size={evidence_bge_np.nbytes/1e9:.2f} GB  load_time={time.time()-t0:.1f}s")

# Keep on CPU; transfer to GPU only when we need a cosine batch (saves VRAM during reranker training).
evidence_bge_torch_cpu = torch.from_numpy(evidence_bge_np)
''')

code(r'''def load_dense_cache(split: str):
    path = V7_CACHE / f"{split}_dense_top500.pkl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; v7 dense retrieval cache required.")
    with open(path, "rb") as f:
        return pickle.load(f)

train_dense = load_dense_cache("train")
dev_dense   = load_dense_cache("dev")
test_dense  = load_dense_cache("test")

dev_dense_eids = {cid: [e for e, _ in pairs] for cid, pairs in dev_dense.items()}
print(f"dev recall@500 (Dense)  = {recall_at_k(dev_dense_eids, dev_claims):.4f}")

# Hybrid pool per claim: ordered union with BM25 items first, then dense-only items.
def hybrid_pool(bm25_split, dense_split):
    out = {}
    for cid, bm in bm25_split.items():
        ds = dense_split[cid]
        seen = set()
        merged = []
        for e, s in bm:
            if e not in seen:
                seen.add(e); merged.append((e, "bm25", s))
        for e, s in ds:
            if e not in seen:
                seen.add(e); merged.append((e, "dense_only", s))
        out[cid] = merged
    return out

train_hybrid = hybrid_pool(train_bm25, train_dense)
dev_hybrid   = hybrid_pool(dev_bm25,   dev_dense)
test_hybrid  = hybrid_pool(test_bm25,  test_dense)

pool_sizes = [len(p) for p in dev_hybrid.values()]
print(f"dev hybrid pool size: min={min(pool_sizes)} median={int(np.median(pool_sizes))} "
      f"max={max(pool_sizes)} mean={np.mean(pool_sizes):.0f}")
hybrid_eids = {cid: [e for e, _, _ in p] for cid, p in dev_hybrid.items()}
# recall_at_k default k=500 was hiding the full hybrid pool — pass the actual pool depth.
hybrid_k = max(len(v) for v in hybrid_eids.values())
print(f"dev recall (hybrid union, k={hybrid_k}) = {recall_at_k(hybrid_eids, dev_claims, k=hybrid_k):.4f}")
''')

code(r'''# Encode claims with BGE so we can compute dense cosines for any (claim, eid).
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

print(f"Loading dense encoder {DENSE_MODEL_NAME} ...")
bge_tokenizer = AutoTokenizer.from_pretrained(DENSE_MODEL_NAME)
bge_model = AutoModel.from_pretrained(DENSE_MODEL_NAME).to(DEVICE).eval()

@torch.inference_mode()
def encode_claims_bge(texts, batch_size=64):
    embs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        enc = bge_tokenizer(chunk, padding=True, truncation=True, max_length=256,
                            return_tensors="pt").to(DEVICE)
        with torch.autocast(device_type="cuda" if DEVICE.type=="cuda" else "cpu",
                            dtype=AUTOCAST_DTYPE, enabled=(DEVICE.type=="cuda")):
            out = bge_model(**enc).last_hidden_state[:, 0]
        out = F.normalize(out.float(), p=2, dim=1)
        embs.append(out.cpu().numpy())
    return np.concatenate(embs, axis=0)

def claims_to_emb(claims_dict):
    cids = list(claims_dict.keys())
    texts = [claims_dict[cid]["claim_text"] for cid in cids]
    embs = encode_claims_bge(texts)
    return cids, embs

print("Encoding claims (train/dev/test) ...")
t0 = time.time()
train_cids, train_claim_emb = claims_to_emb(train_claims)
dev_cids,   dev_claim_emb   = claims_to_emb(dev_claims)
test_cids,  test_claim_emb  = claims_to_emb(test_claims)
print(f"  done in {time.time()-t0:.1f}s")

# Free dense encoder before reranker training.
del bge_model
torch.cuda.empty_cache() if DEVICE.type == "cuda" else None
gc.collect()
''')

code(r'''# Precompute dense cosines for every (claim, eid) in each claim's hybrid pool.
# This is what the selector / fusion strategies will read.
def build_dense_cos_lookup(cids, claim_embs, hybrid_split):
    """Returns dict[cid][eid] -> cosine."""
    out = {}
    # Move evidence embeddings to GPU for fast lookup (1.86 GB on fp16; 4060 has 8 GB).
    ev_gpu = evidence_bge_torch_cpu.to(DEVICE)  # (N, 768) fp16
    for i, cid in enumerate(tqdm(cids, desc="dense cos lookup")):
        pool = hybrid_split[cid]
        idxs = np.array([EID2IDX[e] for e, _, _ in pool], dtype=np.int64)
        idx_t = torch.from_numpy(idxs).to(DEVICE)
        cand_emb = ev_gpu.index_select(0, idx_t).float()  # (P, 768)
        claim_v = torch.from_numpy(claim_embs[i]).to(DEVICE)  # (768,) fp32 already
        cos = (cand_emb @ claim_v).cpu().numpy()             # (P,)
        out[cid] = {pool[j][0]: float(cos[j]) for j in range(len(pool))}
    del ev_gpu
    torch.cuda.empty_cache() if DEVICE.type == "cuda" else None
    return out

print("Precomputing dense cosines for hybrid pools ...")
train_dense_cos = build_dense_cos_lookup(train_cids, train_claim_emb, train_hybrid)
dev_dense_cos   = build_dense_cos_lookup(dev_cids,   dev_claim_emb,   dev_hybrid)
test_dense_cos  = build_dense_cos_lookup(test_cids,  test_claim_emb,  test_hybrid)
print("done.")
''')


# ============================================================ Section 3.5 — Diagnostics on dense / BM25 pool quality
md(r"""## 3.5 Diagnostics — is dense-only enough? does dense need K=500?

Two cheap sanity checks before the reranker work:

**Q1 — dense-only F@K**: if we skipped the reranker entirely and just
returned the dense top-K, what F would we get? This sets a hard lower
bound for "no reranker" and lets us quote the reranker's contribution in
the report.

**Q2 — dense / BM25 recall@K curves**: dense top-500's recall is 0.82 on
dev, but how much of that recall is concentrated in the top-50 vs the
tail? If recall@50 ≈ recall@500, the back half of the dense pool is
mostly noise — exactly the kind of OOD garbage that crashed v8 §6.1.
Truncating dense at small K is then a free win.
""")

code(r'''# Q1 + Q2 in one cell, using the cached BM25 / Dense top-500 lists.
def f_at_k_topk_only(top500_dict, claims, k):
    return retrieval_F({cid: [e for e, _ in pairs[:k]] for cid, pairs in top500_dict.items()}, claims)

def recall_at_k_topk_only(top500_dict, claims, k):
    return recall_at_k({cid: [e for e, _ in pairs] for cid, pairs in top500_dict.items()}, claims, k=k)

K_GRID = [5, 10, 20, 50, 100, 200, 500]
rows = []
for k in K_GRID:
    rows.append({
        "K": k,
        "dense_recall@K":   recall_at_k_topk_only(dev_dense, dev_claims, k),
        "bm25_recall@K":    recall_at_k_topk_only(dev_bm25,  dev_claims, k),
        "dense_F_topK":     f_at_k_topk_only(dev_dense, dev_claims, min(k, MAX_FINAL_K)),
        "bm25_F_topK":      f_at_k_topk_only(dev_bm25,  dev_claims, min(k, MAX_FINAL_K)),
    })
diagnostics_df = pd.DataFrame(rows)
display(diagnostics_df)

# Print marginal-gain comments so the K choice for S-D pool is data-driven.
print("\nDense recall@K marginal gain vs prev K:")
prev = 0.0
for r in rows:
    print(f"  K={r['K']:>3d}  recall={r['dense_recall@K']:.4f}  Δ vs prev = {r['dense_recall@K'] - prev:+.4f}")
    prev = r["dense_recall@K"]
''')

md(r"""### Dense-as-reranker-pool experiment (`S-D{K}`)

If the Q2 table shows dense recall plateauing around K=50–100, build a
**dense-only candidate pool of that depth** and feed it to each trained
reranker as a *third* scoring pool (S-D). The motivation:

- v8 said hybrid scoring crashed because dense-only candidates are OOD
  for a reranker trained on BM25 negatives. R2/R3 retrain on mixed
  negatives, so the dense-only items are *in distribution*.
- The 450 tail items of dense top-500 are mostly noise (low marginal
  recall). Truncating to top-K removes the worst OOD risk.
- If `R3 / S-D50 / relative-δ` ≥ `R0 / S-B / fuse-ce-dense`, the entire
  v8 architecture simplifies to **Dense top-50 → reranker → relative-δ**,
  no BM25 dependency at inference.

`DENSE_TOPK_FOR_RERANKER` defaults to 50; **adjust it after seeing the Q2
table** (look for where the marginal recall gain drops below ~0.01).
""")

code(r'''# Bumped from 50 → 200 after the Q2 diagnostic showed dense recall does NOT plateau
# (recall@50 = 0.51 vs recall@200 = 0.68). At K=200 the dense-only pool's recall
# approaches BM25 top-500's 0.68 ceiling, so dense-only scoring becomes a fair
# competitor to BM25 scoring.
DENSE_TOPK_FOR_RERANKER = 200

def dense_topK_pool(dense_split, K):
    return {cid: pairs[:K] for cid, pairs in dense_split.items()}

train_dense_topK = dense_topK_pool(train_dense, DENSE_TOPK_FOR_RERANKER)
dev_dense_topK   = dense_topK_pool(dev_dense,   DENSE_TOPK_FOR_RERANKER)
test_dense_topK  = dense_topK_pool(test_dense,  DENSE_TOPK_FOR_RERANKER)

dev_dense_topK_eids = {cid: [e for e, _ in v] for cid, v in dev_dense_topK.items()}
print(f"S-D{DENSE_TOPK_FOR_RERANKER}  dev recall@K = "
      f"{recall_at_k(dev_dense_topK_eids, dev_claims, k=DENSE_TOPK_FOR_RERANKER):.4f}")

# Precomputed dense cosines already cover all eids in the hybrid pool (which
# includes the dense top-500), so the dense-topK eids are guaranteed to be in
# the dense_cos lookup tables already built above. No extra precomputation.

# BM25 score lookup for fuse-3feat support on the dense-topK pool: dict[(cid,eid)] -> bm25.
bm25_score_lookup = {
    "train": {cid: dict(pairs) for cid, pairs in train_bm25.items()},
    "dev":   {cid: dict(pairs) for cid, pairs in dev_bm25.items()},
    "test":  {cid: dict(pairs) for cid, pairs in test_bm25.items()},
}
''')


# ============================================================ Section 4 Reranker
md(r"""# 4. Stage 2 — Reranker training variants

Four reranker variants — R0 is the v9 baseline (loaded from disk, not
retrained); R1, R2, R4 vary the training distribution. All scored on
**three** pools: BM25-only (S-B), hybrid union (S-H), and dense top-K (S-D).

| Variant | LR | Epochs | Negatives | F-cos filter |
|---|---|---|---|---|
| **R0** | 1e-5 | 3 | 4 random from BM25 top-500 | – |
| **R1** | 5e-6 | 5 | 4 random from BM25 top-500 | – |
| **R2** | 5e-6 | 5 | 2 from BM25\Dense + 2 from Dense\BM25 | – |
| **R4** | 5e-6 | 5 | 2 hard-mined (by R2 on train) + 2 random | drop hardest if cos to any gold > 0.75; skip top-10 hardest |

R3 was removed: in the previous run F-cos filter on random mixed
negatives made no measurable difference (R3 ≈ R2, ΔF = 0.0002). The
filter is repurposed inside R4 where it has more bite — the items R2
ranks highest among the non-gold are far more likely to be paraphrastic
unannotated gold than randomly-sampled dense candidates.
""")

code(r'''from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import torch.nn.functional as F

rerank_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)

class RerankerPairDataset(Dataset):
    def __init__(self, examples, claims_dict):
        self.examples = examples; self.claims = claims_dict
    def __len__(self): return len(self.examples)
    def __getitem__(self, i):
        ex = self.examples[i]
        enc = rerank_tokenizer(
            self.claims[ex["cid"]]["claim_text"], evidence[ex["eid"]],
            truncation=True, padding="max_length", max_length=RERANKER_MAX_LEN, return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(ex["label"], dtype=torch.float32)
        return item


def build_optimizer(model, lr, weight_decay=WEIGHT_DECAY):
    try:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, fused=True)
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def _scaler():
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=USE_GRAD_SCALER)
    return torch.cuda.amp.GradScaler(enabled=USE_GRAD_SCALER)


@torch.inference_mode()
def score_pool(model, claims_dict, pool_split, pool_kind: str, split_name: str,
               bm25_lookup: dict | None = None):
    """pool_kind ∈ {'bm25','hybrid','dense_topK'}.
    For 'dense_topK', bm25_lookup[cid][eid] is used to fill bm25 scores
    (NaN if the eid is not in BM25 top-500)."""
    model.eval()
    out = {}
    for cid, claim in tqdm(list(claims_dict.items()), desc=f"score {split_name}/{pool_kind}"):
        if pool_kind == "bm25":
            entries = pool_split[cid]                              # [(eid, bm25_score), ...]
            eids = [e for e, _ in entries]
            bm25_scores = np.array([s for _, s in entries], dtype=np.float32)
        elif pool_kind == "hybrid":
            entries = pool_split[cid]                              # [(eid, src, score), ...]
            eids = [e for e, _, _ in entries]
            bm25_scores = np.array([s if src == "bm25" else np.nan for _, src, s in entries], dtype=np.float32)
        elif pool_kind == "dense_topK":
            entries = pool_split[cid]                              # [(eid, dense_score), ...]
            eids = [e for e, _ in entries]
            lk = bm25_lookup[cid] if bm25_lookup is not None else {}
            bm25_scores = np.array([lk.get(e, np.nan) for e in eids], dtype=np.float32)
        else:
            raise ValueError(f"unknown pool_kind={pool_kind!r}")
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
        logits_np = np.asarray(logits_all, dtype=np.float32)
        out[cid] = {
            "eids":  eids,
            "bm25":  bm25_scores,
            "ce_logit": logits_np,
            "ce_prob":  1.0 / (1.0 + np.exp(-logits_np)),
        }
    return out


def train_reranker(examples, label_smoothing=0.0, lr=5e-6, epochs=5, init_state=None, tag=""):
    """Train one reranker from scratch (or from init_state) and return final state dict."""
    model = AutoModelForSequenceClassification.from_pretrained(
        RERANKER_MODEL_NAME, num_labels=1, ignore_mismatched_sizes=True,
    ).to(DEVICE)
    if init_state is not None:
        model.load_state_dict(init_state)
    ds = RerankerPairDataset(examples, train_claims)
    loader = DataLoader(ds, batch_size=RERANKER_BATCH_TRAIN, shuffle=True,
                         pin_memory=(DEVICE.type=="cuda"), num_workers=0)
    opt = build_optimizer(model, lr=lr)
    total_steps = len(loader) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP_RATIO * total_steps), total_steps)
    scaler = _scaler()
    loss_fn = nn.BCEWithLogitsLoss()

    losses_per_epoch = []
    for epoch in range(1, epochs + 1):
        model.train(); losses = []; t0 = time.time()
        for batch in tqdm(loader, desc=f"  {tag} ep {epoch}/{epochs}"):
            labels = batch.pop("labels").to(DEVICE, non_blocking=True)
            if label_smoothing > 0.0:
                labels = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
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
        avg = float(np.mean(losses))
        losses_per_epoch.append(avg)
        print(f"  {tag} epoch {epoch}: loss={avg:.4f}  time={time.time()-t0:.1f}s")
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    del model; torch.cuda.empty_cache() if DEVICE.type == "cuda" else None; gc.collect()
    return state, losses_per_epoch
''')


code(r'''# ------------- Negative-sampling strategies for the four learning variants ----------

def neg_random_bm25(cid, claim, bm25_pool, dense_pool, neg_per_pos, rng):
    """R0/R1: 4 random non-gold candidates from BM25 top-500."""
    gold = set(claim["evidences"])
    pool = [e for e, _ in bm25_pool[cid] if e not in gold and e in evidence]
    needed = neg_per_pos * max(1, sum(1 for g in claim["evidences"] if g in evidence))
    if len(pool) > needed:
        pool = rng.sample(pool, needed)
    return pool

def neg_mixed(cid, claim, bm25_pool, dense_pool, neg_per_pos, rng):
    """R2: 2 from BM25\Dense + 2 from Dense\BM25 per positive."""
    gold = set(claim["evidences"])
    B = {e for e, _ in bm25_pool[cid] if e not in gold and e in evidence}
    D = {e for e, _ in dense_pool[cid] if e not in gold and e in evidence}
    B_only = list(B - D); D_only = list(D - B)
    n_per_pos = neg_per_pos
    n_pos = max(1, sum(1 for g in claim["evidences"] if g in evidence))
    half = n_per_pos // 2
    other = n_per_pos - half
    out = []
    out += rng.sample(B_only, min(half  * n_pos, len(B_only)))
    out += rng.sample(D_only, min(other * n_pos, len(D_only)))
    return out

def neg_mixed_filtered(cid, claim, bm25_pool, dense_pool, neg_per_pos, rng,
                       cos_threshold=0.85):
    """R3: mixed negatives, then drop any candidate with cos(c, any_gold) > threshold
    (these are likely unannotated paraphrastic gold and would be false negatives)."""
    cands = neg_mixed(cid, claim, bm25_pool, dense_pool, neg_per_pos, rng)
    gold_eids = [g for g in claim["evidences"] if g in evidence]
    if not gold_eids or not cands:
        return cands
    gold_idx = np.array([EID2IDX[g] for g in gold_eids], dtype=np.int64)
    cand_idx = np.array([EID2IDX[c] for c in cands], dtype=np.int64)
    gold_emb = evidence_bge_np[gold_idx].astype(np.float32)
    cand_emb = evidence_bge_np[cand_idx].astype(np.float32)
    sims = cand_emb @ gold_emb.T               # (C, G)
    max_sim = sims.max(axis=1)
    kept = [c for c, s in zip(cands, max_sim) if s < cos_threshold]
    return kept


def build_examples(claims_dict, bm25_pool, dense_pool, neg_fn, neg_per_pos=NEGATIVES_PER_POSITIVE):
    rng = random.Random(SEED)
    examples = []
    n_dropped = 0
    for cid, claim in claims_dict.items():
        for g in claim["evidences"]:
            if g in evidence:
                examples.append({"cid": cid, "eid": g, "label": 1.0})
        wanted = neg_per_pos * max(1, sum(1 for g in claim["evidences"] if g in evidence))
        negs = neg_fn(cid, claim, bm25_pool, dense_pool, neg_per_pos, rng)
        if len(negs) < wanted: n_dropped += (wanted - len(negs))
        for n in negs:
            examples.append({"cid": cid, "eid": n, "label": 0.0})
    rng.shuffle(examples)
    n_pos = sum(1 for e in examples if e["label"] == 1.0)
    n_neg = len(examples) - n_pos
    print(f"  built {len(examples):,} examples (pos={n_pos:,} neg={n_neg:,} "
          f"neg/pos={n_neg/max(n_pos,1):.2f})  dropped={n_dropped}")
    return examples
''')


code(r'''# ============= Train R1, R2, R3 and cache CE scores on (BM25 pool, hybrid pool) × dev =============

reranker_results: dict[str, dict] = {}    # tag -> {"state": ..., "losses": [...], "ce": {pool: {split: cache}}}


def score_and_cache(state, tag: str, splits: tuple[str, ...] = ("dev",)):
    """Score the given reranker on BM25 / hybrid / dense_topK pools for the requested splits."""
    model = AutoModelForSequenceClassification.from_pretrained(
        RERANKER_MODEL_NAME, num_labels=1, ignore_mismatched_sizes=True,
    ).to(DEVICE)
    model.load_state_dict(state)
    out = {"bm25": {}, "hybrid": {}, "dense_topK": {}}
    pool_map = {
        "bm25":       {"train": train_bm25,        "dev": dev_bm25,        "test": test_bm25},
        "hybrid":     {"train": train_hybrid,      "dev": dev_hybrid,      "test": test_hybrid},
        "dense_topK": {"train": train_dense_topK,  "dev": dev_dense_topK,  "test": test_dense_topK},
    }
    claims_map = {"train": train_claims, "dev": dev_claims, "test": test_claims}
    for split in splits:
        claims_d = claims_map[split]
        out["bm25"][split]       = score_pool(model, claims_d, pool_map["bm25"][split],       "bm25",       f"{tag}/{split}")
        out["hybrid"][split]     = score_pool(model, claims_d, pool_map["hybrid"][split],     "hybrid",     f"{tag}/{split}")
        out["dense_topK"][split] = score_pool(model, claims_d, pool_map["dense_topK"][split], "dense_topK", f"{tag}/{split}",
                                              bm25_lookup=bm25_score_lookup[split])
    del model; torch.cuda.empty_cache() if DEVICE.type=="cuda" else None; gc.collect()
    return out
''')


code(r'''# -------- R0: load v9's reranker (no retraining) --------
R0_STATE_PATH = V9_MODELS / "reranker_best.pt"
if R0_STATE_PATH.exists():
    print(f"Loading v9 reranker state from {R0_STATE_PATH} ...")
    r0_state = torch.load(R0_STATE_PATH, map_location="cpu")
    reranker_results["R0"] = {
        "state": r0_state, "losses": None,
        "ce": score_and_cache(r0_state, tag="R0"),
    }
    with open(CACHE_DIR / "R0_ce_dev.pkl", "wb") as f:
        pickle.dump(reranker_results["R0"]["ce"], f)
    print("R0 scored on (BM25, hybrid) × dev.")
else:
    print(f"v9 reranker not at {R0_STATE_PATH} — R0 will be skipped.")
    print("Run v9 Stage 2 first, or comment out R0 references downstream.")
''')


code(r'''# -------- R1: same neg sampling as v9, but LR=5e-6 and 5 epochs --------
R1_STATE_PATH = MODEL_DIR / "reranker_R1.pt"
if R1_STATE_PATH.exists():
    print(f"\n=== R1: state cached at {R1_STATE_PATH}, skipping training ===")
    state_R1 = torch.load(R1_STATE_PATH, map_location="cpu")
    losses_R1 = None
else:
    print("\n=== R1: BM25 random negatives, LR=5e-6, 5 epochs ===")
    ex_R1 = build_examples(train_claims, train_bm25, train_dense, neg_random_bm25)
    state_R1, losses_R1 = train_reranker(ex_R1, lr=5e-6, epochs=5, tag="R1")
    torch.save(state_R1, R1_STATE_PATH)
reranker_results["R1"] = {
    "state": state_R1, "losses": losses_R1,
    "ce": score_and_cache(state_R1, tag="R1"),
}
with open(CACHE_DIR / "R1_ce_dev.pkl", "wb") as f:
    pickle.dump(reranker_results["R1"]["ce"], f)
''')


code(r'''# -------- R2: mixed negatives (2 BM25\Dense + 2 Dense\BM25), no F-cos filter --------
R2_STATE_PATH = MODEL_DIR / "reranker_R2.pt"
if R2_STATE_PATH.exists():
    print(f"\n=== R2: state cached at {R2_STATE_PATH}, skipping training ===")
    state_R2 = torch.load(R2_STATE_PATH, map_location="cpu")
    losses_R2 = None
else:
    print("\n=== R2: mixed negatives, LR=5e-6, 5 epochs ===")
    ex_R2 = build_examples(train_claims, train_bm25, train_dense, neg_mixed)
    state_R2, losses_R2 = train_reranker(ex_R2, lr=5e-6, epochs=5, tag="R2")
    torch.save(state_R2, R2_STATE_PATH)
reranker_results["R2"] = {
    "state": state_R2, "losses": losses_R2,
    "ce": score_and_cache(state_R2, tag="R2"),
}
with open(CACHE_DIR / "R2_ce_dev.pkl", "wb") as f:
    pickle.dump(reranker_results["R2"]["ce"], f)
''')


code(r'''# -------- R3 removed: the previous run had R3 ≈ R2 (F=0.2646 vs 0.2648),
#   the F-cos filter at threshold 0.85 made no measurable difference, and
#   training on filtered examples is strictly slower than R2. The F-cos
#   filter is retained inside the R4 mining step where it has more bite
#   (mined hardest negatives are far more likely to be paraphrastic gold
#   than randomly-sampled dense candidates).
''')


code(r'''# -------- R4: hard-negative mining iter-1, FROM SCRATCH (not continuing from R3) --------
# Score R3 on train BM25 pool; for each claim, take its highest-scoring non-gold candidates
# (also subject to the F-cos filter against gold) as hard negatives. Train fresh from
# checkpoint base, 3 epochs.

print("\n=== R4: hard-neg mining iter-1, FROM SCRATCH ===")
# Previous-run failure mode: continuing R4 from R3 with the hardest-scoring
# train negatives drove loss UP (0.41 → 0.65) and dev F crashed to 0.16.
# Diagnosis: R3's "hardest" negatives are mostly paraphrastic near-gold
# items its decision boundary was already correctly hovering around. Continuing
# training on them as label=0 *reverses* R3's good representation. Fixes:
#   (1) train from scratch — model never learns to push these down
#       from a confident-correct position
#   (2) tighten the F-cos filter from 0.85 → 0.75 (drop more paraphrastic items)
#   (3) skip the top `SKIP_TOP_HARDEST` rank candidates per claim — the
#       single most "wins-rerank-confidence" non-gold items are usually
#       the worst false negatives
#   (4) mix the negative budget: half mined-hard + half random, so the
#       model still has clean signal to anchor on
R4_COS_THRESHOLD   = 0.75
R4_SKIP_TOP_HARDEST = 10
R4_HARD_FRACTION   = 0.5   # 50% mined + 50% random

print("Scoring R2 on train BM25 pool to mine hard negatives (use R2 not R3 because R3 was removed)...")
_tmp_model = AutoModelForSequenceClassification.from_pretrained(
    RERANKER_MODEL_NAME, num_labels=1, ignore_mismatched_sizes=True,
).to(DEVICE)
_tmp_model.load_state_dict(state_R2)
r2_train_ce = score_pool(_tmp_model, train_claims, train_bm25, "bm25", "R2/train")
del _tmp_model; torch.cuda.empty_cache() if DEVICE.type=="cuda" else None; gc.collect()


def mine_hard_negs(claims_dict, train_ce_cache, neg_per_pos=NEGATIVES_PER_POSITIVE,
                   cos_threshold=R4_COS_THRESHOLD, skip_top=R4_SKIP_TOP_HARDEST,
                   hard_fraction=R4_HARD_FRACTION):
    """Mine hard negatives subject to multiple safeguards (see R4 cell preamble)."""
    rng = random.Random(SEED)
    examples = []
    n_pos_total = n_hard = n_rand = 0
    for cid, claim in claims_dict.items():
        gold = [g for g in claim["evidences"] if g in evidence]
        for g in gold:
            examples.append({"cid": cid, "eid": g, "label": 1.0})
        n_pos_total += len(gold)
        d = train_ce_cache[cid]
        order = np.argsort(-d["ce_logit"])
        gold_set = set(claim["evidences"])
        wanted = neg_per_pos * max(1, len(gold))
        n_hard_target = int(round(wanted * hard_fraction))
        n_rand_target = wanted - n_hard_target

        gold_emb = None
        if gold:
            gold_idx = np.array([EID2IDX[g] for g in gold], dtype=np.int64)
            gold_emb = evidence_bge_np[gold_idx].astype(np.float32)

        # (1) hard negatives: skip top-`skip_top` hardest, then take next
        # `n_hard_target` after F-cos filtering.
        hard = []
        passed = 0
        for j in order:
            e = d["eids"][j]
            if e in gold_set or e not in evidence:
                continue
            if gold_emb is not None:
                c_emb = evidence_bge_np[EID2IDX[e]].astype(np.float32)
                if float((gold_emb @ c_emb).max()) > cos_threshold:
                    continue
            passed += 1
            if passed <= skip_top:
                continue
            hard.append(e)
            if len(hard) >= n_hard_target:
                break

        # (2) random negatives: from BM25 top-500, non-gold, not already picked.
        used = set(hard) | gold_set
        rand_pool = [e for e, _ in train_bm25[cid] if e not in used and e in evidence]
        rng.shuffle(rand_pool)
        rand_negs = rand_pool[:n_rand_target]

        for h in hard:
            examples.append({"cid": cid, "eid": h, "label": 0.0})
        for r in rand_negs:
            examples.append({"cid": cid, "eid": r, "label": 0.0})
        n_hard += len(hard); n_rand += len(rand_negs)

    rng.shuffle(examples)
    print(f"  R4 examples: {len(examples):,}  pos={n_pos_total}  hard={n_hard}  rand={n_rand}")
    return examples


ex_R4 = mine_hard_negs(train_claims, r2_train_ce)
state_R4, losses_R4 = train_reranker(ex_R4, lr=5e-6, epochs=5, init_state=None, tag="R4")
torch.save(state_R4, MODEL_DIR / "reranker_R4.pt")
reranker_results["R4"] = {
    "state": state_R4, "losses": losses_R4,
    "ce": score_and_cache(state_R4, tag="R4"),
}
with open(CACHE_DIR / "R4_ce_dev.pkl", "wb") as f:
    pickle.dump(reranker_results["R4"]["ce"], f)
''')


# ============================================================ Section 5 Selection
md(r"""# 5. Stage 3 — Scoring / selection / fusion strategies

For each (reranker variant × pool) cache, sweep a family of selection
strategies and report each family's best dev F. This is what the final
comparison table is built from.

Strategies:

| Tag | Function |
|---|---|
| `fixed-K` | top-K by `ce_logit` |
| `threshold` | keep `ce_prob ≥ τ`, fallback to top-1 |
| `relative-δ` | keep within `δ` of top `ce_logit` |
| `fuse-ce-dense` | `α·ce_prob + (1−α)·min_max_norm(dense_cos)` then relative-δ pick |
| `fuse-3feat` | `α·ce_prob + β·bm25_norm + γ·dense_norm` then relative-δ pick (only hybrid pool meaningfully uses bm25 source flag) |
| `RRF` | `Σ 1/(k + rank_X)` over X ∈ {BM25 rank, Dense rank, Reranker rank} |
""")


code(r'''DELTA_GRID = [0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0]
FIXED_K_GRID = [3, 4, 5]
THRESHOLD_GRID = [round(float(x), 2) for x in np.arange(0.02, 0.52, 0.02)] + [0.60, 0.70, 0.80, 0.90]
ALPHA_GRID = [1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1, 0.0]
BETA_GRID  = [0.0, 0.1, 0.2, 0.3]
GAMMA_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
RRF_K_GRID = [10, 30, 60, 100]


def _min_max(x):
    a = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros_like(a)
    lo = a[finite].min(); hi = a[finite].max()
    out = np.zeros_like(a)
    if hi > lo:
        out[finite] = (a[finite] - lo) / (hi - lo)
    # impute missing (NaN bm25 for dense-only candidates) with per-claim min = 0.
    return out


def _apply_relative_delta(eids, scores, delta, kmin=MIN_FINAL_K, kmax=MAX_FINAL_K):
    order = np.argsort(-scores)
    eids_sorted = [eids[i] for i in order]
    scores_sorted = scores[order]
    keep = [eids_sorted[0]]
    top = scores_sorted[0]
    for j in range(1, min(kmax, len(eids_sorted))):
        if scores_sorted[j] >= top - delta:
            keep.append(eids_sorted[j])
        else:
            break
    return keep[:kmax] if len(keep) >= kmin else keep


def strat_fixed_k(ce_cache, dense_cos, k):
    out = {}
    for cid, d in ce_cache.items():
        order = np.argsort(-d["ce_logit"])
        out[cid] = [d["eids"][i] for i in order[:min(k, MAX_FINAL_K)]]
    return out

def strat_threshold(ce_cache, dense_cos, tau):
    out = {}
    for cid, d in ce_cache.items():
        probs = d["ce_prob"]
        order = np.argsort(-probs)
        sel = [d["eids"][i] for i in order if probs[i] >= tau][:MAX_FINAL_K]
        if len(sel) < MIN_FINAL_K:
            sel = [d["eids"][int(order[0])]]
        out[cid] = sel
    return out

def strat_relative_delta(ce_cache, dense_cos, delta):
    out = {}
    for cid, d in ce_cache.items():
        out[cid] = _apply_relative_delta(d["eids"], d["ce_logit"], delta)
    return out

def strat_fuse_ce_dense(ce_cache, dense_cos, alpha, delta=0.25):
    out = {}
    for cid, d in ce_cache.items():
        dcos = np.array([dense_cos[cid].get(e, np.nan) for e in d["eids"]], dtype=np.float32)
        dnorm = _min_max(dcos)
        fused = alpha * d["ce_prob"] + (1.0 - alpha) * dnorm
        out[cid] = _apply_relative_delta(d["eids"], fused, delta)
    return out

def strat_fuse_3feat(ce_cache, dense_cos, alpha, beta, gamma, delta=0.25):
    out = {}
    for cid, d in ce_cache.items():
        bm   = _min_max(d["bm25"])
        dcos = np.array([dense_cos[cid].get(e, np.nan) for e in d["eids"]], dtype=np.float32)
        dn   = _min_max(dcos)
        fused = alpha * d["ce_prob"] + beta * bm + gamma * dn
        out[cid] = _apply_relative_delta(d["eids"], fused, delta)
    return out

def strat_rrf(ce_cache, dense_cos, bm25_split_eids, dense_split_eids, k_rrf,
              use_bm25_rank: bool = True):
    """Reciprocal Rank Fusion over (optional BM25 rank, Dense rank, Reranker rank).
    For dense-only pools, set use_bm25_rank=False — BM25 rank doesn't apply."""
    out = {}
    for cid, d in ce_cache.items():
        bm25_rank  = {e: r for r, e in enumerate(bm25_split_eids[cid])}  if use_bm25_rank else {}
        dense_rank = {e: r for r, e in enumerate(dense_split_eids[cid])}
        ce_order = np.argsort(-d["ce_logit"])
        ce_rank = {d["eids"][i]: r for r, i in enumerate(ce_order)}
        scores = np.zeros(len(d["eids"]), dtype=np.float32)
        for j, e in enumerate(d["eids"]):
            s = 0.0
            if use_bm25_rank and e in bm25_rank:
                s += 1.0 / (k_rrf + bm25_rank[e])
            if e in dense_rank:
                s += 1.0 / (k_rrf + dense_rank[e])
            s += 1.0 / (k_rrf + ce_rank[e])
            scores[j] = s
        out[cid] = _apply_relative_delta(d["eids"], scores, delta=1e-6)  # essentially top-by-score
    return out
''')


code(r'''# Build per-split ordered eid lists for RRF (BM25 and Dense rankings).
bm25_split_eids   = {cid: [e for e, _ in v] for cid, v in dev_bm25.items()}
dense_split_eids  = {cid: [e for e, _ in v] for cid, v in dev_dense.items()}


def sweep_strategies(ce_cache, dense_cos, pool_kind, label):
    """Returns a list of dicts: best F per strategy family."""
    rows = []

    # --- fixed-K ---
    best = max((retrieval_F(strat_fixed_k(ce_cache, dense_cos, k), dev_claims), k) for k in FIXED_K_GRID)
    rows.append({"strategy": "fixed-K", "best_param": f"K={best[1]}", "dev_F": best[0]})

    # --- threshold ---
    best = max((retrieval_F(strat_threshold(ce_cache, dense_cos, t), dev_claims), t) for t in THRESHOLD_GRID)
    rows.append({"strategy": "threshold", "best_param": f"τ={best[1]}", "dev_F": best[0]})

    # --- relative-δ ---
    best = max((retrieval_F(strat_relative_delta(ce_cache, dense_cos, d), dev_claims), d) for d in DELTA_GRID)
    rows.append({"strategy": "relative-δ", "best_param": f"δ={best[1]}", "dev_F": best[0]})

    # --- fuse-ce-dense ---
    fcd_rows = []
    for a in ALPHA_GRID:
        for d in DELTA_GRID:
            F = retrieval_F(strat_fuse_ce_dense(ce_cache, dense_cos, a, d), dev_claims)
            fcd_rows.append((F, a, d))
    F, a, d = max(fcd_rows)
    rows.append({"strategy": "fuse-ce-dense", "best_param": f"α={a},δ={d}", "dev_F": F})

    # --- fuse-3feat (only hybrid pool has reliable BM25 scores for most candidates) ---
    if pool_kind == "hybrid":
        f3_rows = []
        for a in ALPHA_GRID:
            for b in BETA_GRID:
                for g in GAMMA_GRID:
                    if a == 0 and b == 0 and g == 0:
                        continue
                    F = retrieval_F(strat_fuse_3feat(ce_cache, dense_cos, a, b, g), dev_claims)
                    f3_rows.append((F, a, b, g))
        F, a, b, g = max(f3_rows)
        rows.append({"strategy": "fuse-3feat", "best_param": f"α={a},β={b},γ={g}", "dev_F": F})

    # --- RRF ---
    # hybrid pool: 3-way RRF over (BM25 rank, Dense rank, Reranker rank).
    # dense_topK pool: 2-way RRF (Dense rank, Reranker rank) — no BM25 channel.
    # bm25 pool: 2-way RRF (BM25 rank, Reranker rank) — no dense channel, but we
    #            can still bring in Dense rank from the full dense top-500 for
    #            those eids that happen to be in dense top-500. Done below.
    if pool_kind in {"hybrid", "dense_topK", "bm25"}:
        use_bm25 = pool_kind in {"hybrid", "bm25"}
        rrf_rows = []
        for kk in RRF_K_GRID:
            F = retrieval_F(
                strat_rrf(ce_cache, dense_cos, bm25_split_eids, dense_split_eids, kk,
                          use_bm25_rank=use_bm25),
                dev_claims,
            )
            rrf_rows.append((F, kk))
        F, kk = max(rrf_rows)
        tag = "RRF(B+D+CE)" if pool_kind == "hybrid" else ("RRF(B+CE)" if pool_kind == "bm25" else "RRF(D+CE)")
        rows.append({"strategy": tag, "best_param": f"k_rrf={kk}", "dev_F": F})

    for r in rows:
        r["reranker"] = label.split("/")[0]
        r["pool"] = label.split("/")[1]
    return rows
''')


code(r'''# Run the full sweep across all (reranker, pool) combinations on dev.
all_rows = []
for r_tag, r_data in reranker_results.items():
    for pool_kind in ("bm25", "hybrid", "dense_topK"):
        ce_cache = r_data["ce"][pool_kind]["dev"]
        rows = sweep_strategies(ce_cache, dev_dense_cos, pool_kind, label=f"{r_tag}/{pool_kind}")
        all_rows.extend(rows)

summary_df = pd.DataFrame(all_rows)[["reranker", "pool", "strategy", "best_param", "dev_F"]]
summary_df = summary_df.sort_values("dev_F", ascending=False).reset_index(drop=True)
display(summary_df)
''')


code(r'''# Pivot: for each (reranker, pool), show the best strategy family + its dev F.
print("\nBest strategy per (reranker, pool):")
best_per_combo = (summary_df
                  .sort_values("dev_F", ascending=False)
                  .groupby(["reranker", "pool"], as_index=False).first())
display(best_per_combo)

print("\nOverall winning row:")
winner = summary_df.iloc[0]
print(f"  reranker  = {winner['reranker']}")
print(f"  pool      = {winner['pool']}")
print(f"  strategy  = {winner['strategy']}")
print(f"  param     = {winner['best_param']}")
print(f"  dev F     = {winner['dev_F']:.4f}")

# Save summary.
summary_df.to_csv(OUTPUT_DIR / "retrieval_summary.csv", index=False)
print(f"\nWrote {OUTPUT_DIR / 'retrieval_summary.csv'}")
''')


# ============================================================ Section 6 Diagnostics
md(r"""# 6. Diagnostics

For the winning (reranker, pool, strategy) combination — show per-class
retrieval F, average predicted evidence count, and the dev recall ceiling
implied by each scoring pool. These tell us *why* a particular row won.
""")


code(r'''# Re-materialise the winning selection so we can analyse it.
def realise_winner(winner_row, ce_cache, dense_cos, pool_kind):
    s = winner_row["strategy"]
    p = winner_row["best_param"]
    parts = dict(kv.split("=") for kv in p.split(","))
    if s == "fixed-K":
        return strat_fixed_k(ce_cache, dense_cos, int(parts["K"]))
    if s == "threshold":
        return strat_threshold(ce_cache, dense_cos, float(parts["τ"]))
    if s == "relative-δ":
        return strat_relative_delta(ce_cache, dense_cos, float(parts["δ"]))
    if s == "fuse-ce-dense":
        return strat_fuse_ce_dense(ce_cache, dense_cos, float(parts["α"]), float(parts["δ"]))
    if s == "fuse-3feat":
        return strat_fuse_3feat(ce_cache, dense_cos,
                                 float(parts["α"]), float(parts["β"]), float(parts["γ"]))
    if s.startswith("RRF"):
        use_bm25 = "B" in s.split("(")[1].split(")")[0].split("+")  # parse "RRF(B+D+CE)"
        return strat_rrf(ce_cache, dense_cos, bm25_split_eids, dense_split_eids,
                          int(parts["k_rrf"]), use_bm25_rank=use_bm25)
    raise ValueError(s)

winner_cache = reranker_results[winner["reranker"]]["ce"][winner["pool"]]["dev"]
winner_retr = realise_winner(winner, winner_cache, dev_dense_cos, winner["pool"])

# Per-class retrieval F
LABELS = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]
rows = []
for lbl in LABELS:
    cids = [cid for cid, c in dev_claims.items() if c["claim_label"] == lbl]
    if not cids: continue
    f = float(np.mean([evidence_f1(winner_retr[cid], dev_claims[cid]["evidences"]) for cid in cids]))
    avg_pred = float(np.mean([len(winner_retr[cid]) for cid in cids]))
    rows.append({"label": lbl, "n": len(cids), "retrieval_F": f, "avg_pred": avg_pred})
display(pd.DataFrame(rows))

print(f"Overall dev F = {retrieval_F(winner_retr, dev_claims):.4f}")
print(f"Average predicted evidence per claim = "
      f"{np.mean([len(v) for v in winner_retr.values()]):.2f}")
''')


code(r'''# Pool recall comparison.
print("Stage-1 recall ceilings (dev):")
print(f"  BM25 only     recall@500 = {recall_at_k({cid:[e for e,_ in v] for cid,v in dev_bm25.items()}, dev_claims):.4f}")
print(f"  Dense only    recall@500 = {recall_at_k({cid:[e for e,_ in v] for cid,v in dev_dense.items()}, dev_claims):.4f}")
hybrid_eids = {cid:[e for e,_,_ in v] for cid,v in dev_hybrid.items()}
hybrid_k = max(len(v) for v in hybrid_eids.values())
print(f"  Hybrid union  recall@{hybrid_k} = {recall_at_k(hybrid_eids, dev_claims, k=hybrid_k):.4f}")
print(f"  Dense top-{DENSE_TOPK_FOR_RERANKER}  recall = "
      f"{recall_at_k({cid:[e for e,_ in v] for cid,v in dev_dense_topK.items()}, dev_claims, k=DENSE_TOPK_FOR_RERANKER):.4f}")

print("\nReranker training loss trajectory:")
# Variants may have different epoch counts (R0 has no losses, R4 may be 3-5).
# pd.Series wraps each list so pandas pads with NaN instead of erroring on length mismatch.
loss_table = pd.DataFrame({tag: pd.Series(r["losses"] or [], dtype=float)
                            for tag, r in reranker_results.items()})
display(loss_table)
''')


# ============================================================ Footer
md(r"""## Run order / notes

- Section 2 (BM25) and Section 3 (Dense/BGE) reuse caches; they should
  finish in seconds. If the caches are missing, run `nlp_experiments_v9.ipynb`
  (Stage 1) and the v7 dense indexing first.
- Section 3.5: Q1+Q2 diagnostics. 5 seconds. Look at the dense recall@K
  table — if recall plateaus before K=50, lower `DENSE_TOPK_FOR_RERANKER`
  in the next cell and re-run from there.
- Section 4: R0 loads from disk (no training), R1–R3 each train for 5
  epochs (~10 min on RTX 4060), R4 trains 3 epochs continuing from R3
  (~6 min plus a train-pool scoring pass at ~6 min). Each variant adds
  CE scoring on 3 pools × dev — bm25 (~45s), hybrid (~80s), dense_topK
  at K=50 (~10s). Total scoring per reranker ≈ 2.5 min.
- Section 5: pure post-processing, < 1 min. Each (reranker, pool) gets a
  full strategy sweep.
- All caches land in `outputs_notebook_v10/`. CE-score caches per reranker
  let you re-run Section 5 sweeps without re-training.

Key comparisons in the final table:

| Winning row implies | Architectural conclusion |
|---|---|
| `R0 / bm25 / fuse-ce-dense` | v8 fusion workaround is still the best |
| `R3 / hybrid / *` ≥ above | mixed-neg training validates direct hybrid scoring; drop fusion |
| `R3 / dense_topK / *` ≥ above | drop BM25 entirely — Dense-top-K → reranker is enough |
| `R4 / * / *` ≥ above + non-trivial gain | iterative hard-neg mining was worth the extra training pass |
""")


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

out_path = Path(__file__).parent / "nlp_retrieval_v10.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {out_path}  ({len(CELLS)} cells)")
