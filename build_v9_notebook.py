"""Build nlp_experiments_v9.ipynb from logical sections.

Run once: `uv run python build_v9_notebook.py`. The output ipynb has empty
output cells; running the notebook in Jupyter / VSCode fills them in.
"""
from __future__ import annotations

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text))


def code(text: str) -> None:
    CELLS.append(("code", text))


# ---------------------------------------------------------------- Title / intro
md(r"""# 2026 COMP90042 Project — v9 ablation notebook

Sequential greedy ablation over five stages, building up the full retrieval +
classification pipeline for climate-claim fact-checking. Each stage tunes its
own module conditional on the best configuration of all upstream stages, and
reports a local metric that isolates the design decision being made.

| Stage | Variants | Local metric | Best output |
|---|---|---|---|
| 1. Preprocessing × BM25 | 3 PP × 5 retrieval = 15 | dev recall@500 | `BEST_BM25_CONFIG` + cached top-500 |
| 2. Reranker training | 1 run (`ms-marco-MiniLM-L-12-v2`) | dev retrieval F per epoch | reranker weights + CE-score caches |
| 3. Selection strategy | fixed-K / threshold / relative-δ | dev retrieval F | `BEST_SELECTION` + final evidence per split |
| 4. Classifier backbone | distilbert / MiniLM-L-12-marco / MiniLM-NLI | dev A | `BEST_CLASSIFIER` |
| 5. Class weighting | none / sqrt-inv-freq / inv-freq | DISPUTED recall + A non-regression | `BEST_WEIGHTING` |

The reranker is trained **once** on the candidate pool from the best BM25
config. Greedy sequential ablation is a deliberate simplification disclosed in
the report; re-training the reranker per upstream variant is out of budget.

All results are produced by running this notebook end-to-end with `SEED = 42`.
""")

md(r"""## Disclosed limitations

1. Sequential greedy ablation ignores upstream-downstream hyperparameter interaction.
2. The reranker is trained on a single BM25 candidate pool, not re-trained per upstream variant.
3. Single random seed; no variance estimation.
4. No external data; only assignment-provided splits.
""")

# ----------------------------------------------------------- Section 1 marker
md(r"""# 1. DataSet Processing
(You can add as many code blocks and text blocks as you need. However, YOU SHOULD NOT MODIFY the section title)
""")

# ---------------------------------------------------------------- Env + DEV_RUN
code(r'''# Environment detection (local vs. Colab) and the DEV_RUN smoke-test toggle.
# - In Colab: mount Drive, copy data to /content for fast evidence.json reads,
#   write outputs back to Drive so an interrupted session keeps its artifacts.
# - Locally: use ./data and ./outputs_notebook_v9 in the project root.
# - DEV_RUN=True subsamples claims, collapses Stage 1 to one config and caps
#   epochs at 1; use it for end-to-end smoke tests, not for reportable numbers.
import os
from pathlib import Path

# Stop TensorFlow / JAX from grabbing 75% of the GPU on their first CUDA touch.
# On Colab both are preinstalled and may be imported transitively by nltk /
# transformers / etc. — without these flags PyTorch sees ~11/15 GB already
# reserved and OOMs when loading even a 33M-param cross-encoder. Must be set
# BEFORE the offending library is imported.
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

DEV_RUN = False  # set True for a fast smoke test (~minutes instead of hours)

try:
    import google.colab  # noqa
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')

    DRIVE_ROOT = Path('/content/drive/MyDrive/COMP90042')

    LOCAL_DATA = Path('/content/data')
    LOCAL_DATA.mkdir(exist_ok=True)
    if not (LOCAL_DATA / 'evidence.json').exists():
        os.system(f'cp -r "{DRIVE_ROOT}/data/." "{LOCAL_DATA}/"')

    DATA_DIR   = LOCAL_DATA
    OUTPUT_DIR = DRIVE_ROOT / 'outputs_notebook_v9'
else:
    DATA_DIR   = Path('data')
    OUTPUT_DIR = Path('outputs_notebook_v9')

if IN_COLAB:
    # Quick install path on Colab; the cell below is a slower local fallback.
    os.system('pip install -q bm25s')

print(f"IN_COLAB={IN_COLAB}  DEV_RUN={DEV_RUN}")
print(f"DATA_DIR={DATA_DIR}  OUTPUT_DIR={OUTPUT_DIR}")
''')

# ---------------------------------------------------------------- Imports / config
code(r'''# Optional dependency installation. Safe in Colab; on a fresh uv venv this can
# fail because uv venvs ship without pip — wrap in try/except and surface a
# hint to install via `uv pip install` outside the notebook.
import importlib.util, subprocess, sys

REQUIRED = {
    "bm25s": "bm25s",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "sklearn": "scikit-learn",
    "nltk": "nltk",
}
for import_name, pip_name in REQUIRED.items():
    if importlib.util.find_spec(import_name) is None:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])
        except Exception as e:
            print(f"Could not auto-install {pip_name}: {e}")
            print(f"Run `uv pip install {pip_name}` and re-run this cell.")
''')

code(r'''from pathlib import Path
import json, random, time, pickle, math
from collections import Counter

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

# -------- Reproducibility --------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -------- Paths (DATA_DIR / OUTPUT_DIR set by the environment cell above) --------
CACHE_DIR   = OUTPUT_DIR / "cache"
MODEL_DIR   = OUTPUT_DIR / "models"
for d in (OUTPUT_DIR, CACHE_DIR, MODEL_DIR):
    d.mkdir(exist_ok=True, parents=True)

# -------- Hardware / precision (auto-select bf16 vs fp16) --------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda" and torch.cuda.is_bf16_supported():
    AUTOCAST_DTYPE   = torch.bfloat16     # Ada / Ampere / Hopper
    USE_GRAD_SCALER  = False
elif DEVICE.type == "cuda":
    AUTOCAST_DTYPE   = torch.float16      # Turing (T4) / Volta
    USE_GRAD_SCALER  = True
else:
    AUTOCAST_DTYPE   = torch.float32
    USE_GRAD_SCALER  = False

if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

print(f"Device: {DEVICE}  autocast_dtype={AUTOCAST_DTYPE}  grad_scaler={USE_GRAD_SCALER}")

# -------- Task constants --------
LABELS    = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]
LABEL2ID  = {l: i for i, l in enumerate(LABELS)}
ID2LABEL  = {i: l for i, l in enumerate(LABELS)}

# -------- Retrieval / model hyperparameters --------
BM25_CANDIDATE_K = 500
MIN_FINAL_K, MAX_FINAL_K = 1, 5

RERANKER_MODEL_NAME    = "cross-encoder/ms-marco-MiniLM-L-12-v2"
RERANKER_MAX_LEN       = 256
RERANKER_BATCH_TRAIN   = 32
RERANKER_BATCH_EVAL    = 128
RERANKER_EPOCHS        = 3
RERANKER_LR            = 1e-5
NEGATIVES_PER_POSITIVE = 4

CLASSIFIER_MAX_LEN     = 256
CLASSIFIER_BATCH_TRAIN = 16
CLASSIFIER_BATCH_EVAL  = 32
CLASSIFIER_EPOCHS      = 4
CLASSIFIER_LR          = 1e-5

WEIGHT_DECAY  = 0.01
WARMUP_RATIO  = 0.1
GRAD_CLIP     = 1.0

# DEV_RUN overrides: keep the pipeline runnable end-to-end on minutes of compute.
if DEV_RUN:
    RERANKER_EPOCHS   = 1
    CLASSIFIER_EPOCHS = 1
    print("[DEV_RUN] capped RERANKER_EPOCHS=1, CLASSIFIER_EPOCHS=1")
''')

code(r'''def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

train_claims = load_json(DATA_DIR / "train-claims.json")
dev_claims   = load_json(DATA_DIR / "dev-claims.json")
test_claims  = load_json(DATA_DIR / "test-claims-unlabelled.json")
evidence     = load_json(DATA_DIR / "evidence.json")

if DEV_RUN:
    # Subsample claims; keep evidence full (BM25/reranker indices are over evidence,
    # and dropping evidence would invalidate recall numbers entirely).
    train_claims = dict(list(train_claims.items())[:100])
    dev_claims   = dict(list(dev_claims.items())[:50])
    test_claims  = dict(list(test_claims.items())[:30])
    print(f"[DEV_RUN] subsampled claims: train={len(train_claims)} "
          f"dev={len(dev_claims)} test={len(test_claims)}")

evidence_ids   = list(evidence.keys())
evidence_texts = [evidence[eid] for eid in evidence_ids]
EID2IDX        = {eid: i for i, eid in enumerate(evidence_ids)}

print(f"train:    {len(train_claims):,}")
print(f"dev:      {len(dev_claims):,}")
print(f"test:     {len(test_claims):,}")
print(f"evidence: {len(evidence):,}")
''')

code(r'''# Lightweight EDA used to justify later choices.
label_dist = Counter(c["claim_label"] for c in train_claims.values())
print("Train label distribution:")
for lbl in LABELS:
    cnt = label_dist[lbl]
    print(f"  {lbl:20s} {cnt:5d} ({cnt / len(train_claims):6.2%})")

gt_counts = [len(c["evidences"]) for c in train_claims.values()]
print(f"\nGround-truth evidence count: min={min(gt_counts)} max={max(gt_counts)} "
      f"mean={np.mean(gt_counts):.3f}")
print("  exact counts:", sorted(Counter(gt_counts).items()))
''')

code(r'''# Official-style metrics; mirror eval.py so we can tune inside the notebook.
def evidence_f1_for_claim(pred_eids, gold_eids):
    pred_eids = list(pred_eids); gold_eids = list(gold_eids)
    if not pred_eids:
        return 0.0
    pred_set = set(pred_eids)
    correct = sum(1 for eid in gold_eids if eid in pred_set)
    if correct == 0:
        return 0.0
    p = correct / len(pred_eids)
    r = correct / len(gold_eids)
    return 2 * p * r / (p + r)


def evaluate_retrieval_only(retrieval, gold_claims):
    return float(np.mean([
        evidence_f1_for_claim(retrieval[cid], c["evidences"])
        for cid, c in gold_claims.items()
    ]))


def recall_at_k(candidates_top500, gold_claims, k=500):
    recs = []
    for cid, c in gold_claims.items():
        cand = set(candidates_top500[cid][:k])
        gold = c["evidences"]
        if not gold:
            continue
        recs.append(sum(1 for e in gold if e in cand) / len(gold))
    return float(np.mean(recs))


def evaluate_submission(predictions, gold_claims, verbose=True):
    f_scores, correct, total = [], 0, 0
    for cid, gold in gold_claims.items():
        p = predictions[cid]
        f_scores.append(evidence_f1_for_claim(p["evidences"], gold["evidences"]))
        correct += int(p["claim_label"] == gold["claim_label"])
        total   += 1
    F = float(np.mean(f_scores))
    A = correct / total
    H = 0.0 if (F + A) == 0 else 2 * F * A / (F + A)
    if verbose:
        print(f"Evidence Retrieval F-score (F)    = {F:.6f}")
        print(f"Claim Classification Accuracy (A) = {A:.6f}")
        print(f"Harmonic Mean of F and A          = {H:.6f}")
    return {"F": F, "A": A, "H": H}


def write_predictions(predictions, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)
''')

# ----------------------------------------------------- Stage 1
md(r"""## Stage 1 — Preprocessing × BM25 joint tuning

Cartesian product of **3 preprocessing variants × 5 retrieval configs = 15
experiments**. The local metric is `recall@500` on the dev set: this is the
upper bound for everything downstream, and is the most direct thing the
first-stage retriever can be evaluated on.

Preprocessing variants (`lowercase=True` is implicit in all three):

| ID  | Lowercase | Stopwords | Stemmer |
|-----|-----------|-----------|---------|
| PP1 | yes       | no        | no      |
| PP2 | yes       | yes (en)  | no      |
| PP3 | yes       | yes (en)  | Porter  |

Retrieval configs:

| ID         | Method | k1   | b    |
|------------|--------|------|------|
| R-TFIDF    | TF-IDF | n/a  | n/a  |
| R-default  | BM25   | 1.5  | 0.75 |
| R-b05      | BM25   | 1.5  | 0.50 |
| R-k12      | BM25   | 1.2  | 0.75 |
| R-both     | BM25   | 1.2  | 0.50 |
""")

code(r'''import re
import bm25s

# ---- Stopwords (English) ----
# We use the same default list bm25s ships, materialised so the TF-IDF branch
# can use the identical stopword set.
EN_STOPWORDS = set(bm25s.tokenization.STOPWORDS_EN)

# ---- Porter stemmer (nltk) ----
try:
    from nltk.stem import PorterStemmer
    _porter = PorterStemmer()
    def _porter_stem(words): return [_porter.stem(w) for w in words]
except Exception as e:
    _porter_stem = None
    print(f"PorterStemmer unavailable ({e}); PP3 will fall back to PP2 behavior.")

# Simple tokenizer matching bm25s's default token pattern.
_TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")
def _tokenize_raw(text: str):
    return _TOKEN_PATTERN.findall(text.lower())

def preprocess_text(text: str, use_stop: bool, use_stem: bool):
    toks = _tokenize_raw(text)
    if use_stop:
        toks = [t for t in toks if t not in EN_STOPWORDS]
    if use_stem and _porter_stem is not None:
        toks = _porter_stem(toks)
    return toks

PP_VARIANTS = {
    "PP1": dict(use_stop=False, use_stem=False),
    "PP2": dict(use_stop=True,  use_stem=False),
    "PP3": dict(use_stop=True,  use_stem=True),
}

BM25_VARIANTS = {
    "R-TFIDF":   dict(method="tfidf"),
    "R-default": dict(method="bm25", k1=1.5, b=0.75),
    "R-b05":     dict(method="bm25", k1=1.5, b=0.50),
    "R-k12":     dict(method="bm25", k1=1.2, b=0.75),
    "R-both":    dict(method="bm25", k1=1.2, b=0.50),
}

if DEV_RUN:
    # Collapse the 3x5 sweep to a single (PP, BM25) config so we still exercise
    # the cell end-to-end without spending tokenization budget on 15 corpora.
    PP_VARIANTS   = {"PP2": PP_VARIANTS["PP2"]}
    BM25_VARIANTS = {"R-b05": BM25_VARIANTS["R-b05"]}
    print("[DEV_RUN] Stage 1 sweep reduced to PP2 x R-b05.")
''')

code(r'''from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix

def _identity(x): return x

def tfidf_top_k(corpus_tokens, query_tokens, k=500, batch_size=8):
    """Return [(eid_idx, score)] per query. L2-normalized cosine via sparse dot."""
    vec = TfidfVectorizer(analyzer=_identity, lowercase=False, dtype=np.float32)
    X = vec.fit_transform(corpus_tokens)
    X = normalize(X, norm="l2", copy=False)
    Q = vec.transform(query_tokens)
    Q = normalize(Q, norm="l2", copy=False)
    # Batched sparse dot to avoid materialising the full Q @ X.T as dense.
    Xt = X.T.tocsr()
    all_idx, all_score = [], []
    for start in range(0, Q.shape[0], batch_size):
        chunk = Q[start:start + batch_size]
        sims = chunk @ Xt                # csr (b, N)
        sims_dense = sims.toarray()       # (b, N) float32 — N≈1.2M, b small
        # argpartition top-k per row
        idx_part = np.argpartition(-sims_dense, k, axis=1)[:, :k]
        for i in range(sims_dense.shape[0]):
            row_idx = idx_part[i]
            row_scores = sims_dense[i, row_idx]
            order = np.argsort(-row_scores)
            row_idx = row_idx[order]; row_scores = row_scores[order]
            all_idx.append(row_idx); all_score.append(row_scores)
        del sims, sims_dense
    return all_idx, all_score


def bm25_top_k(corpus_tokens, query_tokens, k=500, k1=1.5, b=0.75):
    retr = bm25s.BM25(k1=k1, b=b)
    retr.index(corpus_tokens, show_progress=False)
    res, scores = retr.retrieve(query_tokens, k=k, show_progress=False)
    return retr, res, scores
''')

code(r'''# ----- Run Stage 1 sweep. Tokenize corpus ONCE per preprocessing variant. -----
dev_cids   = list(dev_claims.keys())
dev_queries = [dev_claims[cid]["claim_text"] for cid in dev_cids]

stage1_rows = []
# Cache the BM25 retriever for the chosen winning config so we can reuse it
# for train/test retrieval below.
_best_artifacts = {}   # pp_id -> {"corpus_tokens": ...}

for pp_id, pp_kw in PP_VARIANTS.items():
    print(f"\n=== Tokenizing corpus under {pp_id} ({pp_kw}) ===")
    t0 = time.time()
    corpus_tokens = [preprocess_text(t, **pp_kw) for t in evidence_texts]
    query_tokens  = [preprocess_text(t, **pp_kw) for t in dev_queries]
    print(f"  tokenize: {time.time()-t0:.1f}s")
    _best_artifacts[pp_id] = {"corpus_tokens": corpus_tokens}

    for retr_id, retr_kw in BM25_VARIANTS.items():
        t0 = time.time()
        if retr_kw["method"] == "tfidf":
            idx_lists, _ = tfidf_top_k(corpus_tokens, query_tokens, k=BM25_CANDIDATE_K)
            cand_per_claim = {cid: [evidence_ids[int(i)] for i in idx_lists[r]]
                              for r, cid in enumerate(dev_cids)}
        else:
            _, res, _ = bm25_top_k(corpus_tokens, query_tokens, k=BM25_CANDIDATE_K,
                                   k1=retr_kw["k1"], b=retr_kw["b"])
            cand_per_claim = {cid: [evidence_ids[int(j)] for j in res[r]]
                              for r, cid in enumerate(dev_cids)}
        rec500 = recall_at_k(cand_per_claim, dev_claims, k=BM25_CANDIDATE_K)
        # diagnostic F at top-5
        F_top5 = evaluate_retrieval_only({cid: v[:5] for cid, v in cand_per_claim.items()}, dev_claims)
        elapsed = time.time() - t0
        stage1_rows.append({
            "preproc": pp_id, "retrieval": retr_id, **retr_kw,
            "dev_recall@500": rec500, "dev_F@5": F_top5, "time_s": round(elapsed, 1),
        })
        print(f"  {retr_id:10s} recall@500={rec500:.4f}  F@5={F_top5:.4f}  ({elapsed:.1f}s)")

stage1_df = pd.DataFrame(stage1_rows).sort_values("dev_recall@500", ascending=False).reset_index(drop=True)
display(stage1_df)
''')

code(r'''# Pick the best (PP, retrieval) combo.
best_row = stage1_df.iloc[0]
BEST_PP_ID    = best_row["preproc"]
BEST_RETR_ID  = best_row["retrieval"]
BEST_METHOD   = BM25_VARIANTS[BEST_RETR_ID]["method"]
BEST_K1       = BM25_VARIANTS[BEST_RETR_ID].get("k1")
BEST_B        = BM25_VARIANTS[BEST_RETR_ID].get("b")
BEST_PP_KW    = PP_VARIANTS[BEST_PP_ID]
print(f"BEST_BM25_CONFIG: preproc={BEST_PP_ID} retrieval={BEST_RETR_ID} method={BEST_METHOD} k1={BEST_K1} b={BEST_B}")

# Free other preprocessing variants' tokenized corpora; keep only the winner.
corpus_tokens_best = _best_artifacts[BEST_PP_ID]["corpus_tokens"]
for k in list(_best_artifacts.keys()):
    if k != BEST_PP_ID:
        del _best_artifacts[k]
''')

code(r'''# Build the candidate pools for train / dev / test under the winning config.

def _build_pool(claims_dict, split_name):
    cids = list(claims_dict.keys())
    queries = [claims_dict[cid]["claim_text"] for cid in cids]
    query_tokens = [preprocess_text(t, **BEST_PP_KW) for t in queries]
    if BEST_METHOD == "tfidf":
        idx_lists, score_lists = tfidf_top_k(corpus_tokens_best, query_tokens, k=BM25_CANDIDATE_K)
        out = {}
        for r, cid in enumerate(cids):
            out[cid] = [(evidence_ids[int(idx_lists[r][j])], float(score_lists[r][j]))
                        for j in range(len(idx_lists[r]))]
    else:
        retr = bm25s.BM25(k1=BEST_K1, b=BEST_B)
        retr.index(corpus_tokens_best, show_progress=False)
        res, scores = retr.retrieve(query_tokens, k=BM25_CANDIDATE_K, show_progress=False)
        out = {}
        for r, cid in enumerate(cids):
            out[cid] = [(evidence_ids[int(res[r][j])], float(scores[r][j]))
                        for j in range(len(res[r]))]
    return out

print("Building train pool ...");
t0 = time.time(); train_bm25 = _build_pool(train_claims, "train"); print(f"  {time.time()-t0:.1f}s")
print("Building dev pool ...");
t0 = time.time(); dev_bm25   = _build_pool(dev_claims,   "dev");   print(f"  {time.time()-t0:.1f}s")
print("Building test pool ...");
t0 = time.time(); test_bm25  = _build_pool(test_claims,  "test");  print(f"  {time.time()-t0:.1f}s")

for split, obj in [("train", train_bm25), ("dev", dev_bm25), ("test", test_bm25)]:
    with open(CACHE_DIR / f"{split}_bm25_top500.pkl", "wb") as f:
        pickle.dump(obj, f)

# Sanity: dev recall under the winning combo (should match stage1_df row 0).
dev_top500_eids = {cid: [eid for eid, _ in pairs] for cid, pairs in dev_bm25.items()}
print(f"dev recall@500 (winning combo) = {recall_at_k(dev_top500_eids, dev_claims):.4f}")

# Drop the tokenized corpus to free RAM before reranker / classifier work.
del corpus_tokens_best
import gc; gc.collect()
''')

# ----------------------------------------------------- Stage 2
md(r"""## Stage 2 — Reranker training (single run)

Cross-encoder `cross-encoder/ms-marco-MiniLM-L-12-v2` (~33M params), trained
once on positive (gold) + 4 hard negatives sampled from each claim's BM25
top-500. The reranker is **not** re-trained per upstream BM25 ablation: that
would multiply Stage 2 cost by 15. The greedy-ablation rationale is disclosed
in the report.

Training: BCE loss, no `pos_weight`, AdamW (or Muon for 2D weights when
available), linear warmup + decay, 3 epochs, autocast bf16/fp16. The best
epoch is chosen by dev retrieval F under the relative-δ selector at α=1
(reranker-only — no fusion yet).
""")

code(r'''from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

# -------- Build reranker training pairs (positives + 4 hard negs) --------
def build_reranker_examples(claims_dict, bm25_candidates, neg_per_pos=NEGATIVES_PER_POSITIVE):
    examples = []
    rng = random.Random(SEED)
    for cid, claim in claims_dict.items():
        gold = [eid for eid in claim["evidences"] if eid in evidence]
        gold_set = set(gold)
        for eid in gold:
            examples.append({"cid": cid, "eid": eid, "label": 1.0})
        cand_negs = [eid for eid, _ in bm25_candidates[cid][:BM25_CANDIDATE_K]
                     if eid not in gold_set and eid in evidence]
        needed = neg_per_pos * max(1, len(gold))
        if len(cand_negs) > needed:
            cand_negs = rng.sample(cand_negs, needed)
        for eid in cand_negs:
            examples.append({"cid": cid, "eid": eid, "label": 0.0})
    rng.shuffle(examples)
    return examples

reranker_train_examples = build_reranker_examples(train_claims, train_bm25)
n_pos = sum(1 for e in reranker_train_examples if e["label"] == 1.0)
n_neg = len(reranker_train_examples) - n_pos
print(f"Reranker training examples: {len(reranker_train_examples):,}  "
      f"pos={n_pos:,} neg={n_neg:,} neg/pos={n_neg/max(n_pos,1):.2f}")


class RerankerPairDataset(Dataset):
    def __init__(self, examples, claims_dict, evidence_dict, tokenizer, max_len=RERANKER_MAX_LEN):
        self.examples, self.claims, self.evidence = examples, claims_dict, evidence_dict
        self.tokenizer, self.max_len = tokenizer, max_len
    def __len__(self): return len(self.examples)
    def __getitem__(self, i):
        ex = self.examples[i]
        enc = self.tokenizer(
            self.claims[ex["cid"]]["claim_text"], self.evidence[ex["eid"]],
            truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(ex["label"], dtype=torch.float32)
        return item
''')

code(r'''# -------- Optimizer builder: Muon for 2D weights, AdamW for the rest. --------
SPECIAL_KEYWORDS = ("embedding", "norm", "classifier", "pooler", "score", "head")

def build_optimizers(model, lr=1e-5, weight_decay=WEIGHT_DECAY):
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        if p.ndim < 2 or any(k in lname for k in SPECIAL_KEYWORDS):
            adamw_params.append(p)
        else:
            muon_params.append(p)
    try:
        from muon import Muon
        opts = []
        if muon_params:
            opts.append(Muon(muon_params, lr=lr, momentum=0.95))
        if adamw_params:
            try:
                opts.append(torch.optim.AdamW(adamw_params, lr=lr, weight_decay=weight_decay, fused=True))
            except (TypeError, RuntimeError):
                opts.append(torch.optim.AdamW(adamw_params, lr=lr, weight_decay=weight_decay))
        return opts
    except ImportError:
        all_params = [p for p in model.parameters() if p.requires_grad]
        try:
            return [torch.optim.AdamW(all_params, lr=lr, weight_decay=weight_decay, fused=True)]
        except (TypeError, RuntimeError):
            return [torch.optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)]


def build_schedulers(opts, total_steps, warmup_ratio=WARMUP_RATIO):
    return [get_linear_schedule_with_warmup(
        o, num_warmup_steps=int(warmup_ratio * total_steps),
        num_training_steps=total_steps,
    ) for o in opts]


def step_optimizers(opts, scheds, scaler, loss, params):
    if USE_GRAD_SCALER:
        scaler.scale(loss).backward()
        for o in opts: scaler.unscale_(o)
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        for o in opts: scaler.step(o)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        for o in opts: o.step()
    for s in scheds: s.step()
    for o in opts: o.zero_grad(set_to_none=True)
''')

code(r'''# -------- Score a candidate pool with a (loaded) reranker model. --------
@torch.inference_mode()
def score_candidates_with_reranker(model, tokenizer, claims_dict, bm25_cands, split_name="dev"):
    model.eval()
    cache = {}
    items = list(claims_dict.items())
    for cid, claim in tqdm(items, desc=f"Scoring {split_name}"):
        cand_pairs = bm25_cands[cid]
        cand_eids  = [eid for eid, _ in cand_pairs]
        bm25_scores = np.array([s for _, s in cand_pairs], dtype=np.float32)
        logits_all = []
        for start in range(0, len(cand_eids), RERANKER_BATCH_EVAL):
            batch_eids = cand_eids[start:start + RERANKER_BATCH_EVAL]
            enc = tokenizer(
                [claim["claim_text"]] * len(batch_eids),
                [evidence[eid] for eid in batch_eids],
                truncation=True, padding=True, max_length=RERANKER_MAX_LEN, return_tensors="pt",
            ).to(DEVICE, non_blocking=True)
            with torch.autocast(device_type="cuda" if DEVICE.type=="cuda" else "cpu",
                                dtype=AUTOCAST_DTYPE, enabled=(DEVICE.type=="cuda")):
                logits = model(**enc).logits.squeeze(-1)
            logits_all.extend(logits.float().detach().cpu().tolist())
        logits_np = np.asarray(logits_all, dtype=np.float32)
        cache[cid] = {
            "eids": cand_eids,
            "bm25": bm25_scores,
            "ce_logit": logits_np,
            "ce_prob":  1.0 / (1.0 + np.exp(-logits_np)),
        }
    return cache


# -------- Quick relative-δ dev F for epoch selection (α=1, no fusion). --------
def _quick_dev_F_relative_delta(ce_cache, claims, delta=0.25, kmin=MIN_FINAL_K, kmax=MAX_FINAL_K):
    retr = {}
    for cid, d in ce_cache.items():
        order = np.argsort(-d["ce_logit"])
        eids  = [d["eids"][i] for i in order]
        logits = d["ce_logit"][order]
        keep = [eids[0]]
        top = logits[0]
        for j in range(1, min(kmax, len(eids))):
            if logits[j] >= top - delta:
                keep.append(eids[j])
            else:
                break
        retr[cid] = keep[:kmax] if len(keep) >= kmin else keep
    return evaluate_retrieval_only(retr, claims), retr
''')

code(r'''# -------- Train the reranker (3 epochs; keep best by dev F). --------
print("Loading reranker tokenizer + model:", RERANKER_MODEL_NAME)
reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
reranker_model = AutoModelForSequenceClassification.from_pretrained(
    RERANKER_MODEL_NAME, num_labels=1, ignore_mismatched_sizes=True,
).to(DEVICE)

reranker_ds = RerankerPairDataset(reranker_train_examples, train_claims, evidence, reranker_tokenizer)
reranker_loader = DataLoader(
    reranker_ds, batch_size=RERANKER_BATCH_TRAIN, shuffle=True,
    pin_memory=(DEVICE.type=="cuda"), num_workers=0, persistent_workers=False,
)

reranker_loss_fn = nn.BCEWithLogitsLoss()
reranker_opts    = build_optimizers(reranker_model, lr=RERANKER_LR)
print(f"Reranker optimizers: {[type(o).__name__ for o in reranker_opts]}")
reranker_total_steps = len(reranker_loader) * RERANKER_EPOCHS
reranker_scheds  = build_schedulers(reranker_opts, reranker_total_steps)
reranker_scaler  = (torch.amp.GradScaler("cuda", enabled=USE_GRAD_SCALER) if hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler(enabled=USE_GRAD_SCALER))

best_reranker_state = None
best_reranker_F     = -1.0
best_reranker_epoch = -1
reranker_history    = []

for epoch in range(1, RERANKER_EPOCHS + 1):
    reranker_model.train()
    losses = []
    t0 = time.time()
    for batch in tqdm(reranker_loader, desc=f"Reranker epoch {epoch}/{RERANKER_EPOCHS}"):
        labels = batch.pop("labels").to(DEVICE, non_blocking=True)
        batch  = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device_type="cuda" if DEVICE.type=="cuda" else "cpu",
                            dtype=AUTOCAST_DTYPE, enabled=(DEVICE.type=="cuda")):
            logits = reranker_model(**batch).logits.squeeze(-1)
            loss   = reranker_loss_fn(logits, labels)
        step_optimizers(reranker_opts, reranker_scheds, reranker_scaler, loss, reranker_model.parameters())
        losses.append(float(loss.item()))

    dev_ce_tmp = score_candidates_with_reranker(reranker_model, reranker_tokenizer, dev_claims, dev_bm25,
                                                split_name=f"dev_epoch{epoch}")
    F_dev, _ = _quick_dev_F_relative_delta(dev_ce_tmp, dev_claims, delta=0.25)
    elapsed = time.time() - t0
    reranker_history.append({"epoch": epoch, "loss": float(np.mean(losses)),
                             "dev_F": F_dev, "time_s": round(elapsed, 1)})
    print(f"Epoch {epoch}: loss={np.mean(losses):.4f}  dev_F(δ=0.25)={F_dev:.4f}  time={elapsed:.1f}s")
    if F_dev > best_reranker_F:
        best_reranker_F = F_dev
        best_reranker_epoch = epoch
        best_reranker_state = {k: v.detach().clone().cpu() for k, v in reranker_model.state_dict().items()}

print(f"\nBest reranker epoch = {best_reranker_epoch}  dev_F = {best_reranker_F:.4f}")
reranker_model.load_state_dict(best_reranker_state)
torch.save(best_reranker_state, MODEL_DIR / "reranker_best.pt")
pd.DataFrame(reranker_history)
''')

code(r'''# -------- Score all three splits with the best reranker, cache the scores. --------
def _cache_or_score(split_name, claims_dict, bm25_cands):
    path = CACHE_DIR / f"{split_name}_ce_scores.pkl"
    print(f"Scoring {split_name} ...")
    cache = score_candidates_with_reranker(reranker_model, reranker_tokenizer,
                                           claims_dict, bm25_cands, split_name=split_name)
    with open(path, "wb") as f:
        pickle.dump(cache, f)
    return cache

train_ce = _cache_or_score("train", train_claims, train_bm25)
dev_ce   = _cache_or_score("dev",   dev_claims,   dev_bm25)
test_ce  = _cache_or_score("test",  test_claims,  test_bm25)
''')

# ----------------------------------------------------- Stage 3
md(r"""## Stage 3 — Selection strategy comparison

Post-processing only — no additional training. Three strategies, grid-tuned
on dev. The reranker logits are not perfectly calibrated, so a per-claim
**relative-δ** selector is often more robust than a global probability
threshold; the sweep lets the data decide.

| Strategy        | Param grid                                          |
|-----------------|-----------------------------------------------------|
| `fixed-K`       | K ∈ {3, 4, 5}                                       |
| `threshold`     | τ ∈ {0.02, 0.04, ..., 0.50, 0.60, 0.70, 0.80, 0.90} |
| `relative-δ`    | δ ∈ {0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0}          |

Bounds: min K = 1, max K = 5. Threshold fallback: top-1 if no candidate clears τ.
""")

code(r'''FIXED_K_GRID = [3, 4, 5]
THRESHOLD_GRID = [round(float(x), 2) for x in np.arange(0.02, 0.52, 0.02)] + [0.60, 0.70, 0.80, 0.90]
DELTA_GRID = [0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0]

def select_fixed_k(ce_cache, k, kmax=MAX_FINAL_K):
    out = {}
    for cid, d in ce_cache.items():
        order = np.argsort(-d["ce_logit"])
        out[cid] = [d["eids"][i] for i in order[:min(k, kmax)]]
    return out

def select_threshold(ce_cache, tau, kmin=MIN_FINAL_K, kmax=MAX_FINAL_K):
    out = {}
    for cid, d in ce_cache.items():
        probs = d["ce_prob"]
        order = np.argsort(-probs)
        sel = [d["eids"][i] for i in order if probs[i] >= tau][:kmax]
        if len(sel) < kmin:
            sel = [d["eids"][int(order[0])]]
        out[cid] = sel
    return out

def select_relative_delta(ce_cache, delta, kmin=MIN_FINAL_K, kmax=MAX_FINAL_K):
    out = {}
    for cid, d in ce_cache.items():
        order = np.argsort(-d["ce_logit"])
        eids  = [d["eids"][i] for i in order]
        logits = d["ce_logit"][order]
        keep = [eids[0]]
        top = logits[0]
        for j in range(1, min(kmax, len(eids))):
            if logits[j] >= top - delta:
                keep.append(eids[j])
            else:
                break
        out[cid] = keep[:kmax] if len(keep) >= kmin else keep
    return out


stage3_rows = []
for k in FIXED_K_GRID:
    retr = select_fixed_k(dev_ce, k)
    F = evaluate_retrieval_only(retr, dev_claims)
    avg = float(np.mean([len(v) for v in retr.values()]))
    stage3_rows.append({"strategy": "fixed-K", "param": k, "dev_F": F, "avg_pred": avg})
for tau in THRESHOLD_GRID:
    retr = select_threshold(dev_ce, tau)
    F = evaluate_retrieval_only(retr, dev_claims)
    avg = float(np.mean([len(v) for v in retr.values()]))
    stage3_rows.append({"strategy": "threshold", "param": tau, "dev_F": F, "avg_pred": avg})
for d in DELTA_GRID:
    retr = select_relative_delta(dev_ce, d)
    F = evaluate_retrieval_only(retr, dev_claims)
    avg = float(np.mean([len(v) for v in retr.values()]))
    stage3_rows.append({"strategy": "relative-δ", "param": d, "dev_F": F, "avg_pred": avg})

stage3_df = pd.DataFrame(stage3_rows).sort_values("dev_F", ascending=False).reset_index(drop=True)
display(stage3_df.head(15))
''')

code(r'''# Lock in the best selection strategy.
best_sel = stage3_df.iloc[0]
BEST_STRATEGY = best_sel["strategy"]
BEST_PARAM    = best_sel["param"]
print(f"BEST_SELECTION: {BEST_STRATEGY} @ {BEST_PARAM}  (dev F = {best_sel['dev_F']:.4f}, "
      f"avg pred = {best_sel['avg_pred']:.2f})")

def apply_best_selection(ce_cache):
    if BEST_STRATEGY == "fixed-K":
        return select_fixed_k(ce_cache, int(BEST_PARAM))
    if BEST_STRATEGY == "threshold":
        return select_threshold(ce_cache, float(BEST_PARAM))
    if BEST_STRATEGY == "relative-δ":
        return select_relative_delta(ce_cache, float(BEST_PARAM))
    raise ValueError(BEST_STRATEGY)

train_retrieval = apply_best_selection(train_ce)
dev_retrieval   = apply_best_selection(dev_ce)
test_retrieval  = apply_best_selection(test_ce)

for split, obj in [("train", train_retrieval), ("dev", dev_retrieval), ("test", test_retrieval)]:
    with open(CACHE_DIR / f"{split}_final_evidence.pkl", "wb") as f:
        pickle.dump(obj, f)

dev_F = evaluate_retrieval_only(dev_retrieval, dev_claims)
print(f"Final dev retrieval F under BEST_SELECTION = {dev_F:.4f}")
''')

code(r'''# Retrieval-only baseline (majority label on retrieved evidence) — sanity check.
majority_label = Counter(c["claim_label"] for c in train_claims.values()).most_common(1)[0][0]
print("Majority label:", majority_label)

dev_majority_preds = {
    cid: {"claim_label": majority_label, "evidences": dev_retrieval[cid]}
    for cid in dev_claims
}
print("Dev score: BEST retrieval + majority label:")
_ = evaluate_submission(dev_majority_preds, dev_claims)
write_predictions(dev_majority_preds, OUTPUT_DIR / "dev-retrieval-majority.json")
''')

# ----------------------------------------------------- Section 2
md(r"""# 2. Model Implementation
(You can add as many code blocks and text blocks as you need. However, YOU SHOULD NOT MODIFY the section title)
""")

md(r"""## Stage 4 — Classifier backbone comparison

Three candidate backbones with different pre-training distributions. All
train under identical hyperparameters: lr=1e-5, AdamW (+Muon), warmup=10%,
4 epochs, batch=16/32, max_seq=256. Class weighting fixed to **W1**
(sqrt inverse-freq) here so the only varying factor is the backbone.

| ID  | Model                                   | Pretraining              | Params |
|-----|-----------------------------------------|--------------------------|--------|
| C1  | `distilbert-base-uncased`               | MLM only                 | ~66M   |
| C2  | `cross-encoder/ms-marco-MiniLM-L-12-v2` | MS-MARCO passage ranking | ~33M   |
| C3  | `cross-encoder/nli-MiniLM2-L6-H768`     | SNLI + MNLI              | ~33M   |

Training evidence per claim: gold ∪ retrieved (dedup, gold first, capped at
5). Inference evidence: retrieved-only (matches test-time condition).
""")

code(r'''CLASSIFIER_CANDIDATES = {
    "C1-distilbert":   "distilbert-base-uncased",
    "C2-minilm-marco": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "C3-minilm-nli":   "cross-encoder/nli-MiniLM2-L6-H768",
}


def build_classifier_evidence(cid, claim, retrieval_dict, max_k=MAX_FINAL_K, include_gold=True):
    """Gold ∪ retrieved (gold first), deduped, capped at max_k. Inference passes include_gold=False."""
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
    def __init__(self, claims_dict, retrieval_dict, tokenizer, include_gold, max_len=CLASSIFIER_MAX_LEN):
        self.cids = list(claims_dict.keys())
        self.claims, self.retrieval = claims_dict, retrieval_dict
        self.tokenizer, self.max_len = tokenizer, max_len
        self.include_gold = include_gold
        self.has_label = "claim_label" in next(iter(claims_dict.values()))
    def __len__(self): return len(self.cids)
    def __getitem__(self, i):
        cid = self.cids[i]; c = self.claims[cid]
        ev_ids = build_classifier_evidence(cid, c, self.retrieval, include_gold=self.include_gold)
        if not ev_ids:
            ev_ids = [next(iter(evidence.keys()))]
        sep = f" {self.tokenizer.sep_token} "
        ev_text = sep.join(evidence[e] for e in ev_ids)
        enc = self.tokenizer(c["claim_text"], ev_text, truncation=True,
                              padding="max_length", max_length=self.max_len, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.has_label:
            item["labels"] = torch.tensor(LABEL2ID[c["claim_label"]], dtype=torch.long)
        return item


def make_loader(claims_dict, retrieval_dict, tokenizer, batch_size, include_gold, shuffle=False):
    ds = ClaimEvidenceDataset(claims_dict, retrieval_dict, tokenizer, include_gold)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                    pin_memory=(DEVICE.type=="cuda"), num_workers=0)
    return ds, dl
''')

code(r'''def compute_class_weights(power: float) -> torch.Tensor:
    """power=0 → uniform; power=0.5 → sqrt inv-freq; power=1 → inv-freq.
    Re-normalised so sum(weights) = num_labels."""
    counts = np.array([label_dist[l] for l in LABELS], dtype=np.float64)
    if power == 0.0:
        w = np.ones_like(counts)
    else:
        inv_freq = counts.sum() / np.maximum(counts, 1)
        w = inv_freq ** power
    w = w * (len(LABELS) / w.sum())
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


@torch.inference_mode()
def predict_labels(model, loader, dataset):
    model.eval()
    preds = []
    for batch in loader:
        batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items() if k != "labels"}
        with torch.autocast(device_type="cuda" if DEVICE.type=="cuda" else "cpu",
                            dtype=AUTOCAST_DTYPE, enabled=(DEVICE.type=="cuda")):
            logits = model(**batch).logits
        preds.extend(logits.float().argmax(dim=-1).cpu().tolist())
    return {dataset.cids[i]: ID2LABEL[p] for i, p in enumerate(preds)}


def train_classifier(model_name, weighting_power, epochs=CLASSIFIER_EPOCHS, label_tag=""):
    print(f"\n--- Training {label_tag} ({model_name}) weighting_power={weighting_power} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params:,}")

    train_ds, train_loader = make_loader(train_claims, train_retrieval, tokenizer,
                                          CLASSIFIER_BATCH_TRAIN, include_gold=True, shuffle=True)
    dev_ds,   dev_loader   = make_loader(dev_claims,   dev_retrieval,   tokenizer,
                                          CLASSIFIER_BATCH_EVAL, include_gold=False, shuffle=False)

    class_weights = compute_class_weights(weighting_power)
    print(f"  class weights: {class_weights.cpu().tolist()}")
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    opts    = build_optimizers(model, lr=CLASSIFIER_LR)
    total_steps = len(train_loader) * epochs
    scheds  = build_schedulers(opts, total_steps)
    scaler  = (torch.amp.GradScaler("cuda", enabled=USE_GRAD_SCALER) if hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler(enabled=USE_GRAD_SCALER))

    best = {"epoch": -1, "F": 0.0, "A": 0.0, "H": -1.0, "state": None, "pred_labels": None}
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        t0 = time.time()
        for batch in tqdm(train_loader, desc=f"  {label_tag} epoch {epoch}/{epochs}"):
            batch  = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(device_type="cuda" if DEVICE.type=="cuda" else "cpu",
                                dtype=AUTOCAST_DTYPE, enabled=(DEVICE.type=="cuda")):
                logits = model(**batch).logits
                loss   = loss_fn(logits, labels)
            step_optimizers(opts, scheds, scaler, loss, model.parameters())
            losses.append(float(loss.item()))

        pred_labels = predict_labels(model, dev_loader, dev_ds)
        preds = {cid: {"claim_label": pred_labels[cid], "evidences": dev_retrieval[cid]}
                 for cid in dev_claims}
        m = evaluate_submission(preds, dev_claims, verbose=False)
        elapsed = time.time() - t0
        pred_counts = Counter(pred_labels.values())
        pred_summary = " ".join(f"{l[0]}:{pred_counts.get(l, 0)}" for l in LABELS)
        print(f"  epoch {epoch}: loss={np.mean(losses):.4f}  F={m['F']:.4f}  A={m['A']:.4f}  "
              f"H={m['H']:.4f}  preds=({pred_summary})  {elapsed:.1f}s")
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **m, "time_s": round(elapsed, 1)})
        if m["A"] > best["A"] or (m["A"] == best["A"] and m["H"] > best["H"]):
            best = {"epoch": epoch, **m,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "pred_labels": dict(pred_labels)}

    print(f"  best epoch {best['epoch']}: F={best['F']:.4f}  A={best['A']:.4f}  H={best['H']:.4f}")
    # Free GPU memory between runs.
    del model
    torch.cuda.empty_cache() if DEVICE.type == "cuda" else None
    return tokenizer, best, history
''')

code(r'''stage4_results = {}    # tag -> (tokenizer, best, history)
stage4_summary_rows = []

for tag, model_name in CLASSIFIER_CANDIDATES.items():
    tok, best, hist = train_classifier(model_name, weighting_power=0.5, label_tag=tag)
    stage4_results[tag] = (tok, best, hist)
    # Per-class diagnostics from best epoch's preds.
    pl = best["pred_labels"]
    per_class = {}
    for lbl in LABELS:
        cids = [cid for cid, c in dev_claims.items() if c["claim_label"] == lbl]
        per_class[lbl] = float(np.mean([pl[cid] == lbl for cid in cids])) if cids else 0.0
    stage4_summary_rows.append({"backbone": tag, "model": model_name,
                                **{f"acc_{l}": per_class[l] for l in LABELS},
                                "best_epoch": best["epoch"], "F": best["F"],
                                "A": best["A"], "H": best["H"]})

stage4_df = pd.DataFrame(stage4_summary_rows).sort_values("A", ascending=False).reset_index(drop=True)
display(stage4_df)

BEST_CLASSIFIER_TAG  = stage4_df.iloc[0]["backbone"]
BEST_CLASSIFIER_NAME = CLASSIFIER_CANDIDATES[BEST_CLASSIFIER_TAG]
print(f"BEST_CLASSIFIER: {BEST_CLASSIFIER_TAG}  ({BEST_CLASSIFIER_NAME})")
''')

# ----------------------------------------------------- Stage 5
md(r"""## Stage 5 — Class weighting ablation

Conditioned on `BEST_CLASSIFIER` from Stage 4. Three weighting schemes; the
local metric is **DISPUTED-class recall** (the class with no representation
in our current best predictions), gated by non-regression on overall
accuracy and harmonic mean.

| ID | Weighting             | `power` |
|----|-----------------------|---------|
| W0 | None (uniform)        | 0.0     |
| W1 | sqrt inverse-freq     | 0.5     |
| W2 | full inverse-freq     | 1.0     |
""")

code(r'''stage5_summary_rows = []
stage5_results = {}

for w_tag, power in [("W0", 0.0), ("W1", 0.5), ("W2", 1.0)]:
    tok, best, hist = train_classifier(BEST_CLASSIFIER_NAME, weighting_power=power,
                                        label_tag=f"{BEST_CLASSIFIER_TAG}_{w_tag}")
    stage5_results[w_tag] = (tok, best, hist)
    pl = best["pred_labels"]
    per_class_recall = {}
    for lbl in LABELS:
        cids = [cid for cid, c in dev_claims.items() if c["claim_label"] == lbl]
        per_class_recall[lbl] = float(np.mean([pl[cid] == lbl for cid in cids])) if cids else 0.0
    stage5_summary_rows.append({"weighting": w_tag, "power": power,
                                **{f"recall_{l}": per_class_recall[l] for l in LABELS},
                                "F": best["F"], "A": best["A"], "H": best["H"]})

stage5_df = pd.DataFrame(stage5_summary_rows)
display(stage5_df)
''')

code(r'''# Pick BEST_WEIGHTING by DISPUTED recall, with a non-regression gate on A.
# We accept the variant with highest DISPUTED recall whose A is within 0.02 of the best A in the table.
A_max = stage5_df["A"].max()
eligible = stage5_df[stage5_df["A"] >= A_max - 0.02].copy()
eligible = eligible.sort_values(["recall_DISPUTED", "H"], ascending=False)
BEST_WEIGHTING = eligible.iloc[0]["weighting"]
print(f"BEST_WEIGHTING = {BEST_WEIGHTING}  (eligible A >= {A_max - 0.02:.4f})")
print(eligible.to_string())

# The chosen model is the one already trained in Stage 5 under that weighting.
tokenizer_final, best_final, _ = stage5_results[BEST_WEIGHTING]
final_state = best_final["state"]
final_pred_labels_dev = best_final["pred_labels"]

# Save final model state.
torch.save(final_state, MODEL_DIR / f"classifier_{BEST_CLASSIFIER_TAG}_{BEST_WEIGHTING}.pt")
''')

# ----------------------------------------------------- Section 3
md(r"""# 3. Testing and Evaluation
(You can add as many code blocks and text blocks as you need. However, YOU SHOULD NOT MODIFY the section title)
""")

code(r'''# -------- Final dev evaluation + official eval.py invocation. --------
dev_predictions = {cid: {"claim_label": final_pred_labels_dev[cid],
                          "evidences": dev_retrieval[cid]}
                   for cid in dev_claims}
print("FINAL dev metrics:")
final_dev_metrics = evaluate_submission(dev_predictions, dev_claims)
write_predictions(dev_predictions, OUTPUT_DIR / "dev-predictions.json")

# Run the official evaluator subprocess.
import subprocess, sys
eval_py = Path("eval.py")
if eval_py.exists():
    cmd = [sys.executable, "eval.py",
           "--predictions", str(OUTPUT_DIR / "dev-predictions.json"),
           "--groundtruth", str(DATA_DIR / "dev-claims.json")]
    print("\nRunning:", " ".join(cmd))
    completed = subprocess.run(cmd, text=True, capture_output=True)
    print(completed.stdout)
    if completed.stderr:
        print(completed.stderr)
''')

code(r'''# Confusion matrix + per-class diagnostics.
def confusion_matrix_df(gold_claims, pred_labels):
    mat = pd.DataFrame(0, index=LABELS, columns=LABELS)
    for cid, c in gold_claims.items():
        mat.loc[c["claim_label"], pred_labels[cid]] += 1
    return mat

print("Confusion matrix (rows=gold, cols=pred):")
display(confusion_matrix_df(dev_claims, final_pred_labels_dev))

per_class_rows = []
for lbl in LABELS:
    cids = [cid for cid, c in dev_claims.items() if c["claim_label"] == lbl]
    acc = float(np.mean([final_pred_labels_dev[cid] == lbl for cid in cids])) if cids else 0.0
    retr_F = float(np.mean([
        evidence_f1_for_claim(dev_retrieval[cid], dev_claims[cid]["evidences"]) for cid in cids
    ])) if cids else 0.0
    per_class_rows.append({"label": lbl, "n": len(cids), "class_acc": acc, "retrieval_F": retr_F})
display(pd.DataFrame(per_class_rows))
''')

code(r'''# -------- Test-set predictions for leaderboard. --------
# Load the chosen classifier weights back onto a fresh model for test inference.
test_model = AutoModelForSequenceClassification.from_pretrained(
    BEST_CLASSIFIER_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID,
    ignore_mismatched_sizes=True,
).to(DEVICE)
test_model.load_state_dict(final_state)

_, test_loader = make_loader(test_claims, test_retrieval, tokenizer_final,
                              CLASSIFIER_BATCH_EVAL, include_gold=False, shuffle=False)
test_ds = test_loader.dataset
test_pred_labels = predict_labels(test_model, test_loader, test_ds)

test_predictions = {cid: {"claim_label": test_pred_labels[cid],
                           "evidences": test_retrieval[cid]}
                    for cid in test_claims}
write_predictions(test_predictions, OUTPUT_DIR / "test-output.json")
print("Wrote", OUTPUT_DIR / "test-output.json", "with", len(test_predictions), "predictions")

print("\nFinal configuration:")
print(f"  BEST_BM25_CONFIG : preproc={BEST_PP_ID} retrieval={BEST_RETR_ID}")
print(f"  BEST_SELECTION   : {BEST_STRATEGY} @ {BEST_PARAM}")
print(f"  BEST_CLASSIFIER  : {BEST_CLASSIFIER_TAG}  ({BEST_CLASSIFIER_NAME})")
print(f"  BEST_WEIGHTING   : {BEST_WEIGHTING}")
print(f"  dev metrics      : F={final_dev_metrics['F']:.4f}  A={final_dev_metrics['A']:.4f}  H={final_dev_metrics['H']:.4f}")
''')

md(r"""## Object Oriented Programming codes here

The dataset classes `RerankerPairDataset` and `ClaimEvidenceDataset` defined
above are the OOP components of the system. They subclass `torch.utils.data.Dataset`
and encapsulate per-claim evidence assembly and tokenization for the
reranker and classifier respectively.
""")

# ----------------------------------------------------- Emit notebook JSON
def _cell(kind: str, src: str, idx: int) -> dict:
    payload = {
        "cell_type": kind,
        "id": f"cell-{idx:03d}",
        "metadata": {},
        "source": src.splitlines(keepends=True),
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
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = Path(__file__).parent / "nlp_experiments_v9.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {out_path}  ({len(CELLS)} cells)")
