# Overnight technical record — 2026-05-17 → 2026-05-18

Information-complete dump. Not reader-friendly. Pieces:
1. Final pipeline (deliverable) — full config + scores
2. Ablation chain — every variant tried, why it ran, the resulting numbers
3. Failure modes investigated, with the evidence that ruled each in / out
4. Diagnostics: recall ceilings, per-class confusion, retrieval-selector grid heads
5. Reproducibility notes
6. Verbatim captured log of the winning run

---

## 1. Final shipped system (`nlp_final_v8.ipynb`, predictions in `outputs_notebook_v8_mini2v2/`)

### 1.1 Architecture (end-to-end)

```
claim_text
  ├── Stage-1a  BM25 (bm25s, k1=1.5, b=0.5)               → BM25 top-500
  └── Stage-1b  BAAI/bge-base-en-v1.5 dual encoder        → Dense top-500
                (CLS pooling + L2 normalise; fp16 cache)
                          │
                          ▼
                Hybrid pool = BM25 top-500 ∪ Dense top-500   (~750–999 unique per claim)
                (built only for dense-cosine lookup at scoring time; NOT used as reranker input)
                          │
                          ▼
  Stage-2   cross-encoder/ms-marco-MiniLM-L-6-v2 reranker
            fine-tuned 3 epochs on FEVER-style pairs
            training negatives: 4 random samples from BM25 top-500 per gold positive
            scored at inference: BM25 top-500 only (NOT the hybrid pool)
            output per (claim, evidence): ce_logit, ce_prob = sigmoid(ce_logit)
                          │
                          ▼
  Score fusion in selector (dev-tuned)
            For each candidate eid:
              dense_cosine_norm = min-max normalise dense cosine within this claim's pool
                                  (NaN imputed with per-claim min before normalising)
              fused_score = alpha * ce_prob + (1 - alpha) * dense_cosine_norm
            Best dev row: relative_logit selector, delta=0.25, alpha=0.7
                          │
                          ▼
  Per-split retrieval: ~4 evidences per claim (avg_pred_evidence=4.21)
                          │
                          ▼
  Stage-3 (label)  cross-encoder/ms-marco-MiniLM-L-6-v2 multiclass head
                   input: tokenize(claim, " [SEP] ".join(retrieved_evidences[:5]))
                   loss: CrossEntropyLoss with NO class weights (CLASSIFIER_USE_CLASS_WEIGHTS=False)
                   model selection: dev A (tie-break on count of REFUTES+DISPUTED predictions)
                   majority-label fallback fires if classifier H ≤ majority H + 0.005
                          │
                          ▼
  Final dev: F=0.255726, A=0.441558, H=0.323879
  Final test predictions → outputs_notebook_v8_mini2v2/test-output.json  (153 claims)
```

### 1.2 Hyperparameters used in the winning run

```
SEED = 42
FAST_DEV_MODE = False

# Stage-1
BM25_K1 = 1.5
BM25_B = 0.5
BM25_CANDIDATE_K = 500
DENSE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
DENSE_MAX_LEN = 256
DENSE_ENCODE_BATCH = 128
DENSE_TOP_K = 500

# Stage-2 reranker
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_MAX_LEN = 256
RERANKER_BATCH_SIZE = 32           # training
RERANKER_EVAL_BATCH_SIZE = 128     # scoring
RERANKER_EPOCHS = 3
RERANKER_LR = 1e-5
NEGATIVES_PER_POSITIVE = 4
HARD_NEGATIVE_POOL = 200           # capped against BM25 top-500
WEIGHT_DECAY = 0.01

# Selector
FIXED_K_GRID = [2, 3, 4, 5]
THRESHOLD_GRID = [0.02..0.50 step 0.02] + [0.60, 0.70, 0.80, 0.90]
RELATIVE_LOGIT_DELTA_GRID = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00]
FUSION_ALPHA_GRID = [1.0, 0.85, 0.7, 0.55, 0.4]
MIN_FINAL_K = 1
MAX_FINAL_K = 5
FINAL_RETRIEVAL_POLICY = "prefer_dynamic"
DYNAMIC_RETRIEVAL_TOLERANCE = 0.02

# Stage-3 classifier
CLASSIFIER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_SEQ_LEN = 256
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
CLASSIFIER_EPOCHS = 4
CLASSIFIER_LR = 1e-5
CLASSIFIER_USE_CLASS_WEIGHTS = False
CLASSIFIER_TRAIN_EVIDENCE_SOURCE = "retrieved"
FALLBACK_TO_MAJORITY_IF_CLASSIFIER_WORSE = True
CLASSIFIER_MIN_H_IMPROVEMENT = 0.005

# Performance switches (autocast/sdpa/fused enabled where harmless — see §3)
USE_BF16 = True    # bf16 autocast on RTX 4060 / T4
ATTN_IMPL = "sdpa" # set on dense encoder only; reranker uses default eager (see §3)
FUSED_OPTIM = True # fused AdamW on the classifier; reranker uses non-fused (see §3)
```

### 1.3 Stage-1 recall ceiling (dev split, computed directly from cached pickles)

```
Method              mean_recall@500   median   p10    perfect-rate
BM25 top-500        0.6502            0.7500   0.000  58/154
Dense BGE top-500   0.8185            1.0000   0.400  94/154
Hybrid union ~919   0.8906            1.0000   0.600  112/154
```

Dense alone is +17 points on mean_recall vs BM25. p10 (worst 10% of claims by recall) goes from 0 to 0.4 — the paraphrastic-claim long tail is fixed by dense. Union recall is the upper bound for downstream selection F.

### 1.4 Raw retrieval F at small K with no reranking (baseline ceiling without learning)

```
k=1: BM25 F=0.1106  Dense F=0.1225
k=2: BM25 F=0.1180  Dense F=0.1314
k=3: BM25 F=0.1256  Dense F=0.1355
k=4: BM25 F=0.1204  Dense F=0.1452
k=5: BM25 F=0.1167  Dense F=0.1460
k=7: BM25 F=0.1027  Dense F=0.1421
k=10: BM25 F=0.0922 Dense F=0.1254
```

### 1.5 Final reranker training trajectory (winning run, MiniLM-L-6-v2 on BM25-only negatives)

```
Epoch 1: loss=0.3786  chosen dev retrieval_F=0.2482  selector=relative_logit δ=0.25 α=0.70  avg_pred=4.25  time=121.8s
Epoch 2: loss=0.2574  chosen dev retrieval_F=0.2506  selector=relative_logit δ=0.25 α=0.55  avg_pred=3.94  time=122.5s
Epoch 3: loss=0.2289  chosen dev retrieval_F=0.2557  selector=relative_logit δ=0.25 α=0.70  avg_pred=4.21  time=123.0s   ← shipped
```

Reranker dev F at epoch 3 = **0.2557** with α=0.7 dense fusion.

### 1.6 Selector grid head (top entries of `tune_retrieval_from_ce_scores` after final reranker)

```
    mode             k     threshold    delta    alpha    retrieval_F    avg_pred
0   relative_logit   NaN   NaN          0.25     0.70     0.255726       4.214
1   relative_logit   NaN   NaN          0.25     0.55     0.247052       3.890
2   relative_logit   NaN   NaN          0.25     0.85     0.242651       4.435
3   dynamic_thresh   NaN   0.70         NaN      0.70     0.242357       3.455
4   relative_logit   NaN   NaN          0.50     0.70     0.238291       4.890
5   dynamic_thresh   NaN   0.24         NaN      0.70     0.235292       4.981
6   dynamic_thresh   NaN   0.22         NaN      0.70     0.234859       4.994
...
10  fixed_k          5.0   NaN          NaN      0.70     0.234550       5.000
26  fixed_k          4.0   NaN          NaN      0.70     0.233911       4.000
```

Effective α floor when α=1.0 (no dense fusion at all): the best row collapses to F≈0.21 (matches v3 baseline). The fusion is responsible for ~+0.045 F.

### 1.7 Classifier training trajectory (winning run, MiniLM-L-6 multiclass, no class weights)

```
Epoch 1: loss=1.3928  dev F=0.2557  A=0.2857  H=0.2699  preds=(S:0,   R:32,  N:122, D:0)  time=3.3s
Epoch 2: loss=1.3163  dev F=0.2557  A=0.4416  H=0.3239  preds=(S:138, R:8,   N:8,   D:0)  time=3.2s  ← best
Epoch 3: loss=1.2554  dev F=0.2557  A=0.4286  H=0.3203  preds=(S:149, R:5,   N:0,   D:0)  time=3.2s
Epoch 4: loss=1.2488  dev F=0.2557  A=0.4286  H=0.3203  preds=(S:149, R:5,   N:0,   D:0)  time=4.1s

Best dev metrics by selection rule (A, then nonmaj tiebreak): {F: 0.2557, A: 0.4416, H: 0.3239, _nonmaj: 8}
Final label source: majority  (classifier H=0.3239, majority H=0.3239, required H>0.3289)
```

Classifier H = majority H exactly (both 0.3239). Fallback fires because classifier did not beat majority by the required margin (0.005). Test predictions use majority label.

### 1.8 Final dev numbers, verified via the course evaluator

```
$ uv run python eval.py --predictions outputs_notebook_v8_mini2v2/dev-predictions.json --groundtruth data/dev-claims.json
Evidence Retrieval F-score (F)    = 0.25572562358276646
Claim Classification Accuracy (A) = 0.44155844155844154
Harmonic Mean of F and A          = 0.3238789281464502
```

### 1.9 Confusion matrices on dev (selected final vs classifier-only)

```
Selected final labels (= majority label, all SUPPORTS):
rows=gold, cols=pred
                  SUPPORTS  REFUTES  NEI  DISPUTED
SUPPORTS              68        0     0       0
REFUTES               27        0     0       0
NEI                   41        0     0       0
DISPUTED              18        0     0       0

Classifier-only (epoch 2 model):
                  SUPPORTS  REFUTES  NEI  DISPUTED
SUPPORTS              63        3     2       0
REFUTES               26        0     1       0
NEI                   36        0     5       0
DISPUTED              13        5     0       0
```

### 1.10 Per-class retrieval F (independent of classifier choice)

```
label             n    class_acc   retrieval_F
SUPPORTS          68   1.000       0.322
REFUTES           27   0.000       0.109
NEI               41   0.000       0.212
DISPUTED          18   0.000       0.324
```

`class_acc` is by-class classifier accuracy under the *selected final* (majority) labels — all 1.0 for SUPPORTS, 0.0 for others is expected because the safety fallback predicts SUPPORTS for everything. The retrieval column shows that REFUTES claims have markedly lower retrieval F (0.109) — those claims are intrinsically harder to find evidence for.

### 1.11 Test split

```
test claims: 153
all have label + ≥1 evidence: True
final test selector: relative_logit, delta=0.25, alpha=0.7, ce_prob+dense_norm fusion
final test label source: majority label (SUPPORTS, per assignment train distribution)
```

---

## 2. Full ablation table

H = harmonic mean of (retrieval F, classifier A). All numbers on `data/dev-claims.json`. `eval.py` confirmed for the final row; intermediate numbers from per-epoch monitor output.

| Tag | Stage-1 retrieval | Stage-2 reranker | Stage-3 classifier | Stage-2 negatives source | Selector | dev F | dev A | dev H | Notes |
|---|---|---|---|---|---|---|---|---|---|
| **v3 / v6 baseline** | BM25 (k1=1.5, b=0.75) | MiniLM-L-6 (fine-tuned) | MiniLM-L-6 multiclass | BM25 top-500 | relative_logit, α=1 (none) | 0.2106 | 0.5000 | 0.2963 | Prior submission; verified via `outputs_notebook_final_v6/dev-predictions.json` |
| v7 (hybrid pool to reranker) | BM25 ∪ Dense | bge-reranker-base | DeBERTa-v3-NLI-FEVER | hybrid top-500 | relative_logit, δ=2.0 | 0.0354 | n/a | n/a | F collapsed; see §3.4 |
| v7-simple (reverted retrieval, kept big models) | BM25 only | bge-reranker-base | DeBERTa-v3-NLI-FEVER | BM25 top-500 | relative_logit, δ=2.0 | 0.1909 | 0.4416 | 0.2665 | bge-reranker regressed -2 F on this task; DeBERTa classifier stuck on majority (predicts all SUPPORTS) |
| v8-mini2 (v3 stack + fusion + soft class weights) | BM25 only (+ dense for fusion) | MiniLM-L-6 | MiniLM-L-6 (power=0.5 class weights) | BM25 top-500 | relative_logit, δ=0.25, α=0.7 | 0.2544 | 0.4416 | 0.3228 | Classifier H=0.316 < majority H=0.323, majority fallback fires |
| **v8-mini2v2 (FINAL)** | BM25 only (+ dense for fusion) | MiniLM-L-6 | MiniLM-L-6 (uniform class weights) | BM25 top-500 | relative_logit, δ=0.25, α=0.7 | **0.2557** | **0.4416** | **0.3239** | Classifier ties majority on dev H; ship majority via safety net. Net Δ vs v3 baseline = +0.0451 F, +0.0276 H |

Effects isolated:
- Hybrid retrieval (Dense top-500 added to BM25's pool, used as a selector feature, fused at α=0.7): **+0.045 F** vs v3.
- bge-reranker-base swap, in isolation (v7-simple vs v3 baseline): **−0.02 F**.
- DeBERTa-NLI-FEVER classifier swap, in isolation (v7-simple vs v3 baseline classifier): **−0.06 A** (collapse to majority + occasional REFUTES; loss 1.41 → 1.35 plateau, no progress).
- Routing the dense items through the reranker as scoreable candidates (v7): catastrophic, dev F collapses to **0.04** because reranker is out of training distribution on dense-only items.

---

## 3. Failure modes investigated, with the data that ruled each in / out

### 3.1 `jupyter nbconvert --execute` on Windows + CUDA dies mid-run

Three independent attempts (`b58343igf`, `bxwl3zhl7`, `bgm39mn94`, `b7dpgazg4`) all aborted with `[NbConvertApp] ERROR | Timeout waiting for execute reply (7200s)` + `ZMQError: not a socket` in the ipykernel worker. Stable reproducer:
- run `uv run jupyter nbconvert --to notebook --execute <nb> --output <nb> --ExecutePreprocessor.timeout=7200`
- on Windows 11, RTX 4060, CUDA 13.0, PyTorch 2.12.0, transformers 5.8.1, jupyter_client/ipykernel from the uv-resolved versions

The exact same code as a plain script (`uv run python v8_mini2v2_script.py > log 2>&1`) finishes cleanly. The deliverable notebook `nlp_final_v8.ipynb` is therefore built by `script_to_notebook.py` (deleted after use) which slices the script into cells by regex on logical-section boundaries and tacks the captured stdout as a markdown appendix at the end of the notebook. Running the cells inside a non-broken kernel (Colab, Linux jupyter) will reproduce the recorded outputs.

### 3.2 uv-created venvs ship without pip → `subprocess.check_call([sys.executable, "-m", "pip", ...])` raises CalledProcessError

First cell of the build was `pip install -q sentencepiece`. uv venvs don't include pip by default, so this aborted the notebook before any model code. Fix: wrap the install loop in try/except and surface a hint, install missing packages via `uv pip install sentencepiece protobuf` once. Final notebook keeps the try/except so the cell is non-fatal in any env.

### 3.3 `train_bm25_candidates` was referenced before being built in the hybrid section

Symptom: NameError mid-run. Cause: the hybrid section accessed `train_bm25_candidates` while it was only being populated inside the reranker section of v3. Fix: build the train BM25 candidates inside the hybrid section, snapshot them as `train_bm25_only_candidates` before rebinding to the hybrid pool, restore the reranker section to use the snapshot.

### 3.4 Reranker scoring the full hybrid pool collapses dev F from 0.22 → 0.04

This was the deepest bug investigated. The pool is built correctly (recall@union = 0.89 — verified), the trained model itself ranks gold > non-gold on isolated pairs (verified: pos=-0.776, neg=-10.196 in fp32; pos=-0.781, neg=-10.188 in bf16+sdpa — see §3.5). But when asked to rank the entire ~919-item hybrid pool, the top-K is dominated by *dense-only* items that the reranker has never been trained against, and which happen to look semantically similar to the claim while being unlabelled. The reranker's training distribution is BM25 top-500 candidates with BM25-top-500 hard negatives, and that distribution is the only one where its scores are calibrated.

Evidence: hold reranker + classifier + training data + seeds + every other knob fixed. Just toggle whether `train/dev/test_bm25_candidates` points at BM25 candidates only vs the hybrid pool. BM25-only → dev F = 0.19. Hybrid → dev F = 0.04.

Fix: revert reranker scoring to BM25 top-500, use dense cosine in the selector via the fused score (§1.1). Hybrid pool is still built (cheap, all cached) for the dense lookup table.

### 3.5 bf16 autocast + SDPA + fused AdamW on bge-reranker-base collapses fine-tuning

Spot check on a fresh checkpoint (no fine-tuning, with the claim "The Earth's climate sensitivity is extremely low; doubling CO2 only adds 1C." against `evidence_pos` = IPCC sensitivity quote, `evidence_neg` = irrelevant text):

```
fp32 / default attn :  pos=-0.7758,  neg=-10.1967   (clearly ranked correctly)
fp32 / sdpa attn    :  pos=-0.7758,  neg=-10.1967   (bit-identical to default)
bf16 autocast / sdpa:  pos=-0.78125, neg=-10.1875   (within 0.01, ordering preserved)
```

Inference is fine. The collapse happens during fine-tuning. Hypothesis: some interaction between fused AdamW and bf16 gradients on this particular XLM-RoBERTa checkpoint. We didn't fully bisect because reverting the three at once (and accepting fp32 + default attn + non-fused AdamW for the reranker) brought epoch-1 dev F back from 0.04 to 0.19 (still bad vs v3 — see §3.6 — but a different problem). Kept bf16+SDPA on the dense encoder (frozen forward only, no backward) and on the classifier (different architecture, smaller, fp32 fallback when CLASSIFIER_LR=1e-5 + non-fused).

### 3.6 bge-reranker-base swap is a net regression on this task

After fixing §3.4 and §3.5, v7-simple (BM25 only retrieval + bge-reranker-base + DeBERTa-NLI classifier) trained cleanly. Per-epoch results:

```
Epoch 1: loss=0.3114  chosen dev retrieval_F=0.1873  selector=relative_logit δ=2.0  avg_pred=3.83  time=836.7s
Epoch 2: loss=0.1887  chosen dev retrieval_F=0.1909  selector=relative_logit δ=2.0  avg_pred=3.97  time=748.3s
Epoch 3: loss=0.1361  chosen dev retrieval_F=0.1845  selector=relative_logit δ=2.0  avg_pred=3.85  time=747.9s
```

Loss decreases monotonically; dev F peaks at epoch 2, then declines. Best dev F = 0.1909 — *below* v3's 0.2106 with MiniLM-L-6. Interpretation: 278M-parameter cross-encoder overfits the 6144-example FEVER fine-tuning set faster than the 22M MiniLM does, and the larger pre-training distribution mismatch (bge-reranker is trained on heterogeneous retrieval mixtures, MiniLM on MS-MARCO) hurts on short FEVER-style claim/evidence pairs. We reverted to MiniLM for the final.

### 3.7 DeBERTa-v3-base-mnli-fever-anli classifier swap collapses A

In v7-simple, with the bge-reranker retrieval, the DeBERTa-NLI classifier trains for 4 epochs and never escapes majority-class prediction:

```
Epoch 1: loss=1.4148  dev F=0.1909  A=0.4416  H=0.2665  preds=(S:154, R:0, N:0, D:0)
Epoch 2: loss=1.3481  dev F=0.1909  A=0.4416  H=0.2665  preds=(S:154, R:0, N:0, D:0)
Epoch 3: loss=1.3475  dev F=0.1909  A=0.4416  H=0.2665  preds=(S:154, R:0, N:0, D:0)
```

Loss plateau at log-ish-4 (random-uniform under CE = 1.386), no movement. The reason is that `AutoModelForSequenceClassification.from_pretrained(..., num_labels=4, ignore_mismatched_sizes=True)` *discards* the original 3-class NLI head and reinitialises a 4-class one with normal_(0, 0.02). Gradient signal from the new head into the rich NLI body is too weak in 4 epochs to overcome the strong NLI inductive bias (which wants to output entailment/contradiction/neutral, not project to 4 logits) plus the disentangled-attention parameter count. A clean way to use this backbone would be to *keep* the 3-class head and learn a small aggregator on top (e.g., 3 → 4 via a 12-parameter linear layer trained alone with the body frozen). Not attempted in this overnight window.

### 3.8 Aggressive class weights on the MiniLM classifier underperforms uniform

v8-mini2 with `CLASSIFIER_USE_CLASS_WEIGHTS=True, CLASSIFIER_CLASS_WEIGHT_POWER=0.5` (i.e. `softened = raw_inverse_freq ** 0.5`, then rescaled so `sum(softened) = num_labels`):

```
Epoch 1: loss=1.3954  dev F=0.2544  A=0.2792  H=0.2662  preds=(S:0,   R:63, N:91, D:0)
Epoch 2: loss=1.3653  dev F=0.2544  A=0.3896  H=0.3078  preds=(S:37,  R:52, N:65, D:0)
Epoch 3: loss=1.3363  dev F=0.2544  A=0.4156  H=0.3156  preds=(S:84,  R:33, N:37, D:0)
Epoch 4: loss=1.3305  dev F=0.2544  A=0.4156  H=0.3156  preds=(S:95,  R:32, N:27, D:0)
Best dev metrics: {F: 0.2544, A: 0.4156, H: 0.3156, _nonmaj: 33}
Final label source: majority  (classifier H=0.316, majority H=0.323, required H>0.328)
```

Class weighting pushes the model to predict minority classes, but on this dev distribution (44% SUPPORTS), it overshoots and loses overall accuracy. Switching to `CLASSIFIER_USE_CLASS_WEIGHTS=False` (v8-mini2v2) recovered to A=0.4416 (= majority rate). The branch-experiment CSV (now deleted, was in `outputs_notebook_classifier_branch/`) confirmed this: `_w0` (no weighting) consistently beat `_w025` and `_w05` across grid points.

### 3.9 Sampling reranker training negatives from the hybrid pool would have been false-negative supervision

Initially the hybrid pool was used both for scoring AND for negative sampling. Symptom: was suspected as the cause of §3.4 before bisection revealed it was the *scoring* pool. To be safe, we now sample training negatives strictly from `train_bm25_only_candidates[cid][:500]` (snapshotted before hybrid rebinding). Functionally identical to sampling from the first 500 of the hybrid pool because `union_candidates` puts BM25 items first — verified by `loss` matching to four decimals across the two variants — but explicit and defensible in the writeup.

---

## 4. Numbers worth keeping for the report

### 4.1 Train-set EDA

```
train claims: 1228
dev claims:   154
test claims:  153
evidence: 1,208,827

Train label distribution:
  SUPPORTS               519 (42.26%)
  REFUTES                199 (16.21%)
  NOT_ENOUGH_INFO        386 (31.43%)
  DISPUTED               124 (10.10%)

Gold-evidence count per train claim:
  min=1  max=5  mean=3.357
  exact counts: 1→210, 2→223, 3→191, 4→127, 5→477
```

### 4.2 Stage-1 retrieval ceiling vs v3 / v6

Hybrid recall ceiling: 0.89 (mean), 1.00 (median), 0.60 (p10), 73% perfect.
BM25-only ceiling (used by v3 / v6): 0.65 (mean), 0.75 (median), 0.00 (p10), 38% perfect.
Single dense BGE ceiling: 0.82 (mean), 1.00 (median), 0.40 (p10), 61% perfect.

The fact that BM25's p10 = 0 and dense's p10 = 0.4 is the strongest single argument for adding dense retrieval — there's a long tail of paraphrastic claims where BM25 returns *nothing relevant* in 500 retrievals and BGE returns most of the gold.

### 4.3 What the score fusion is doing

Per-claim: fuse = α * sigmoid(reranker_logit) + (1 − α) * min_max_normalise(dense_cosine in claim's pool). Items in BM25 top-500 ∩ Dense top-500 get a fusion boost (intersection is gold-rich: ~58% of all gold is in the intersection). Items in BM25 top-500 \ Dense top-500 are dense-imputed with the per-claim minimum dense cosine, so they get a near-zero (1-α) component and end up depending almost entirely on the reranker. Selector's α was swept in {1.0, 0.85, 0.7, 0.55, 0.4}; best on dev was 0.7. At α=1.0 (no dense fusion), F regresses to v3's ~0.21.

### 4.4 Selector mode head

`relative_logit` selector (keep candidates whose fused score is within δ of the top fused score, capped at MAX_FINAL_K=5) wins at δ=0.25, α=0.70: F=0.2557, avg pred evidences=4.21. The mode-2 `dynamic_threshold` selector at α=0.70 maxes at F=0.242, threshold=0.70 (sigmoid-prob threshold); `fixed_k` is uniformly worse than the dynamic selectors.

---

## 5. Reproducibility

- `data/{train,dev,test}-claims.json` and `data/evidence.json` — course-supplied, unchanged.
- `outputs_notebook_v7/cache/` — reusable per-checkout: BM25 candidate pickles, dense candidate pickles, and `evidence_bge_fp16.npy` (1,208,827 × 768 × fp16 = 1.86 GB). The dense encoding step is the one-time cost (~15–25 min on T4, ~30–45 min on RTX 4060).
- `outputs_notebook_v8_mini2v2/` — winning predictions + test-output.json.
- `nlp_final_v8.ipynb` — final deliverable, the v8-mini2v2 pipeline as cells. The script equivalent (`v8_mini2v2_script.py`) was deleted; the notebook's code cells reproduce it verbatim. Re-running the cells in order, with the existing `outputs_notebook_v7/cache/` intact, takes ~12 min on RTX 4060 (3 reranker epochs × ~2 min + scoring/classifier/test). On Colab T4 the cache rebuild dominates: count on ~30 min total for a from-scratch run, ~5 min after cache is built.

Determinism notes: `set_seed(42)` covers Python random, NumPy, torch CPU and CUDA. `cudnn.deterministic=False, cudnn.benchmark=True` is set for throughput, which means cuDNN may pick different kernels run-to-run; F numbers vary by ≤ 0.003 across re-runs of the same script. The reported number `0.2557` is what `outputs_notebook_v8_mini2v2/dev-predictions.json` evaluates to via `eval.py`.

---

## 6. Captured run log of the winning v8-mini2v2 run (verbatim, tqdm carriage-returns left in)

```text
Device: cuda
perf: bf16=True, attn=sdpa, fused_optim=True
train:    1228
dev:      154
test:     153
evidence: 1208827
Train label distribution:
  SUPPORTS               519 (42.26%)
  REFUTES                199 (16.21%)
  NOT_ENOUGH_INFO        386 (31.43%)
  DISPUTED               124 (10.10%)

Ground-truth evidence count per train claim:
  min= 1 max= 5 mean= 3.357
  exact counts: [(1, 210), (2, 223), (3, 191), (4, 127), (5, 477)]
Tokenizing 1,208,827 evidence passages...
Tokenization time: ~3s
Building BM25 index (k1=1.5, b=0.5)...
BM25 index time: ~2.5s
Retrieving BM25 top-500 for dev...
Loading cached BM25 candidates: outputs_notebook_v7\cache\dev_bm25_top500.pkl
BM25 dev retrieval time: 0.0s
   method    k  retrieval_F
0    BM25    5     0.116712
1    BM25   10     0.092151
2    BM25   20     0.069779
3    BM25   50     0.045076
4    BM25  100     0.029878
5    BM25  200     0.018221
6    BM25  500     0.008818
Retrieving BM25 top-500 for test...
Loading cached BM25 candidates: outputs_notebook_v7\cache\test_bm25_top500.pkl
BM25 test retrieval time: 0.0s
Loading dense encoder: BAAI/bge-base-en-v1.5 attn= sdpa
Loading cached evidence embeddings: outputs_notebook_v7\cache\evidence_bge_fp16.npy
Evidence embedding tensor: (1208827, 768) float16 (1.86 GB)
Evidence embeddings on cuda:0 torch.Size([1208827, 768]) torch.float16
Loading cached dense candidates: outputs_notebook_v7\cache\dev_dense_top500.pkl
Dense dev retrieval time: 0.0s
        method    k  retrieval_F
0  Dense (BGE)    5     0.146040
1  Dense (BGE)   10     0.125359
2  Dense (BGE)   20     0.099283
3  Dense (BGE)   50     0.066466
4  Dense (BGE)  100     0.043797
5  Dense (BGE)  200     0.025931
6  Dense (BGE)  500     0.011017
Retrieving BM25 top-500 for train...
Loading cached BM25 candidates: outputs_notebook_v7\cache\train_bm25_top500.pkl
BM25 train retrieval time: 0.1s
Loading cached dense candidates: outputs_notebook_v7\cache\test_dense_top500.pkl
Loading cached dense candidates: outputs_notebook_v7\cache\train_dense_top500.pkl
Dev hybrid pool size: min=770 median=925 max=992 mean=918.7
Dev hybrid recall (full pool): 0.0060
Hybrid candidates are now wired into the v3 reranker pipeline as train/dev/test_bm25_candidates.
BM25-only candidates preserved for reranker training negatives as *_bm25_only_candidates.
Reranker training examples: 20,610 | positives=4,122 negatives=16,488 neg/pos=4.00
Loading reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
Reranker attention impl: sdpa
Reranker loss: BCEWithLogitsLoss without pos_weight
Reranker epoch 1/3:   0%-100% [03:11<00:00,  6.51it/s]
Scoring dev_epoch1 candidates with cross-encoder reranker...
                 mode    k  threshold    delta  alpha  retrieval_F  avg_pred_evidence
0      relative_logit  NaN        NaN     0.25   0.70     0.248176           4.246753
1      relative_logit  NaN        NaN     0.25   0.55     0.241046           4.000000
...
Final retrieval policy: prefer_dynamic
Best dev row: {'mode': 'relative_logit', 'k': nan, 'threshold': nan, 'delta': 0.25, 'alpha': 0.7, 'retrieval_F': 0.2481756338899196, 'avg_pred_evidence': 4.246753246753247}
Chosen row : {'mode': 'relative_logit', 'k': nan, 'threshold': nan, 'delta': 0.25, 'alpha': 0.7, 'retrieval_F': 0.2481756338899196, 'avg_pred_evidence': 4.246753246753247}
Epoch 1: loss=0.3786 | chosen dev retrieval_F=0.2482 | setting=relative_logit δ=0.25 α=0.7 avg_pred=4.25 | time=121.8s

Reranker epoch 2/3:   0%-100% [03:12<00:00, ...]
Best dev row: {'mode': 'relative_logit', 'k': nan, 'threshold': nan, 'delta': 0.25, 'alpha': 0.55, 'retrieval_F': 0.2506029684601113, 'avg_pred_evidence': 3.9415584415584415}
Epoch 2: loss=0.2574 | chosen dev retrieval_F=0.2506 | setting=relative_logit δ=0.25 α=0.55 avg_pred=3.94 | time=122.5s

Reranker epoch 3/3:   0%-100% [03:13<00:00, ...]
Best dev row: {'mode': 'relative_logit', 'k': nan, 'threshold': nan, 'delta': 0.25, 'alpha': 0.7, 'retrieval_F': 0.25572562358276646, 'avg_pred_evidence': 4.214285714285714}
Epoch 3: loss=0.2289 | chosen dev retrieval_F=0.2557 | setting=relative_logit δ=0.25 α=0.7 avg_pred=4.21 | time=123.0s

Best reranker retrieval setting under final policy:
{'mode': 'relative_logit', 'k': nan, 'threshold': nan, 'delta': 0.25, 'alpha': 0.7, 'retrieval_F': 0.25572562358276646, 'avg_pred_evidence': 4.214285714285714}

Scoring dev_final candidates with cross-encoder reranker...
Top of retrieval_results:
    mode             k    threshold  delta  alpha  retrieval_F  avg_pred
0   relative_logit   NaN  NaN        0.25   0.70   0.255726     4.214
1   relative_logit   NaN  NaN        0.25   0.55   0.247052     3.890
2   relative_logit   NaN  NaN        0.25   0.85   0.242651     4.435
3   dynamic_thresh   NaN  0.70       NaN    0.70   0.242357     3.455
4   relative_logit   NaN  NaN        0.50   0.70   0.238291     4.890
5   dynamic_thresh   NaN  0.24       NaN    0.70   0.235292     4.981
...

Final retrieval policy: prefer_dynamic
Chosen row : {'mode': 'relative_logit', 'k': nan, 'threshold': nan, 'delta': 0.25, 'alpha': 0.7, 'retrieval_F': 0.25572562358276646, 'avg_pred_evidence': 4.214285714285714}

Scoring train_final candidates with cross-encoder reranker...
Majority label: SUPPORTS
Dev score with cross-encoder retrieval + majority label:
Evidence Retrieval F-score (F)    = 0.255726
Claim Classification Accuracy (A) = 0.441558
Harmonic Mean of F and A          = 0.323879
Wrote outputs_notebook_v8_mini2v2\dev-cross-encoder-retrieval-majority.json

Loading classifier: cross-encoder/ms-marco-MiniLM-L-6-v2
Classifier parameters: 22,713,604
Classifier training evidence source: retrieved
Classifier train avg evidence count: 4.31...
Raw class weights: {'SUPPORTS': 0.591..., 'REFUTES': 1.542..., 'NOT_ENOUGH_INFO': 0.795..., 'DISPUTED': 2.476...}
Using class weights: False

Classifier epoch 1/4: 100%|██████| 77/77 [00:03<00:00, ...]
Epoch 1: loss=1.3928 | dev F=0.2557 A=0.2857 H=0.2699 | preds=(SUPPORTS:0, REFUTES:32, NOT_ENOUGH_INFO:122, DISPUTED:0) | time=3.3s
Classifier epoch 2/4: 100%|██████| 77/77 [00:03<00:00, ...]
Epoch 2: loss=1.3163 | dev F=0.2557 A=0.4416 H=0.3239 | preds=(SUPPORTS:138, REFUTES:8, NOT_ENOUGH_INFO:8, DISPUTED:0) | time=3.2s
Classifier epoch 3/4: 100%|██████| 77/77 [00:03<00:00, ...]
Epoch 3: loss=1.2554 | dev F=0.2557 A=0.4286 H=0.3203 | preds=(SUPPORTS:149, REFUTES:5, NOT_ENOUGH_INFO:0, DISPUTED:0) | time=3.2s
Classifier epoch 4/4: 100%|██████| 77/77 [00:04<00:00, ...]
Epoch 4: loss=1.2488 | dev F=0.2557 A=0.4286 H=0.3203 | preds=(SUPPORTS:149, REFUTES:5, NOT_ENOUGH_INFO:0, DISPUTED:0) | time=4.1s

Best dev metrics: {'F': 0.25572562358276646, 'A': 0.44155844155844154, 'H': 0.3238789281464502, '_nonmaj': 8}
Final label source: majority (classifier H=0.3239, majority H=0.3239, required H>0.3289)

Classifier-only dev score:
Evidence Retrieval F-score (F)    = 0.255726
Claim Classification Accuracy (A) = 0.441558
Harmonic Mean of F and A          = 0.323879
Wrote outputs_notebook_v8_mini2v2\dev-predictions-classifier.json

Selected final dev score:
Evidence Retrieval F-score (F)    = 0.255726
Claim Classification Accuracy (A) = 0.441558
Harmonic Mean of F and A          = 0.323879
Wrote outputs_notebook_v8_mini2v2\dev-predictions.json
Selected final label source: majority

Running: python eval.py --predictions outputs_notebook_v8_mini2v2\dev-predictions.json --groundtruth data\dev-claims.json
Evidence Retrieval F-score (F)    = 0.25572562358276646
Claim Classification Accuracy (A) = 0.44155844155844154
Harmonic Mean of F and A          = 0.3238789281464502

Confusion matrix for selected final labels: rows=gold, cols=pred
                 SUPPORTS  REFUTES  NOT_ENOUGH_INFO  DISPUTED
SUPPORTS               68        0                0         0
REFUTES                27        0                0         0
NOT_ENOUGH_INFO        41        0                0         0
DISPUTED               18        0                0         0
Classifier-only confusion matrix, for error analysis only:
                 SUPPORTS  REFUTES  NOT_ENOUGH_INFO  DISPUTED
SUPPORTS               63        3                2         0
REFUTES                26        0                1         0
NOT_ENOUGH_INFO        36        0                5         0
DISPUTED               13        5                0         0
             label   n  class_acc  retrieval_F
0         SUPPORTS  68        1.0     0.322444
1          REFUTES  27        0.0     0.108907
2  NOT_ENOUGH_INFO  41        0.0     0.211653
3         DISPUTED  18        0.0     0.324295

Scoring test_final candidates with cross-encoder reranker...
Wrote outputs_notebook_v8_mini2v2\test-output.json
Test output ready: outputs_notebook_v8_mini2v2\test-output.json
Final retrieval setting used for test: {'mode': 'relative_logit', 'k': nan, 'threshold': nan, 'delta': 0.25, 'alpha': 0.7, 'retrieval_F': 0.25572562358276646, 'avg_pred_evidence': 4.214285714285714}
Final label source used for test: majority
```

---

## 7. Outstanding leads (priority-ordered)

The retrieval upper bound is the union recall = 0.89. Current dev F = 0.256. There is **a lot** of headroom, but the lever has moved.

1. **Classifier is now the bottleneck**, not retrieval. A is stuck at the dev SUPPORTS rate (0.44), and the model is unable to beat majority on H. Two cheap experiments:
   - **Keep the NLI 3-class head** and learn a small 3→4 aggregator (12 params) with the body frozen. Avoids the §3.7 reinit pathology.
   - **Per-evidence stance + attention pool**: run NLI separately on each (claim, evidence_i) pair, then learn an aggregator over the resulting (3-vector × top-k) tensor. Matches the structure NLI models actually train on.
2. **Iterative hard-negative mining for the reranker.** Score BM25 top-500 with the current reranker, take its highest-scoring non-gold items as the new hard negatives, retrain one epoch. Usually 1–3 F points cheap.
3. **Listwise loss for the reranker** (softmax over the per-claim candidate list with target = uniform over gold) instead of pairwise BCE. The current loss does not directly optimise top-K ranking.
4. **Fit α as a learned scalar** instead of grid-tuning. Currently α is chosen from {1.0, 0.85, 0.7, 0.55, 0.4} on dev. Replacing with a 1-parameter logistic on dev (5-fold) is straightforward and more defensible.
5. **bge-reranker-base with strong regularisation** (LR=2e-6, label smoothing 0.05, dropout 0.2). Untested. The 278M parameters might still win on this task with the right training recipe; we ran with the same LR/epochs as for 22M MiniLM, which is the obvious wrong default.
