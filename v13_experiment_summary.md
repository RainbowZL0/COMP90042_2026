# 实验进展总结：从上次计划到当前结果

最终版流水线

| 模块               | 当前采用设置                                    |
| ------------------ | ----------------------------------------------- |
| 预处理             | lowercase + stopwords removal + Porter stemming |
| 第一阶段检索       | BM25,`k1=1.2`, `b=0.5`, top-500 candidates  |
| Reranker           | `cross-encoder/ms-marco-MiniLM-L-12-v2`       |
| Evidence selection | relative-delta,`delta=1.5`                    |
| Classifier         | `cross-encoder/ms-marco-MiniLM-L-12-v2`       |
| Class weighting    | 不加 class weighting                            |

最终模型在验证集结果

| 指标                             |   数值 |
| -------------------------------- | -----: |
| Evidence Retrieval F-score, F    | 0.2448 |
| Claim Classification Accuracy, A | 0.4805 |
| Harmonic Mean, H                 | 0.3243 |

指标解释

| 指标 | 含义                                                                                                |
| ---- | --------------------------------------------------------------------------------------------------- |
| F    | evidence retrieval F-score，衡量找回来的 evidence 和 gold evidence 有多接近。越高说明证据找得越准。 |
| A    | claim classification accuracy，衡量 claim label 预测对了多少。                                      |
| H    | F 和 A 的 harmonic mean，综合考虑 evidence 和 label 两部分。任何一边很低，H 都会被拉低。            |

## 术语速查

| 术语                | 白话解释                                                                                                    |
| ------------------- | ----------------------------------------------------------------------------------------------------------- |
| claim               | 要判断真假的一句陈述。                                                                                      |
| evidence            | 数据集中可用来支持或反驳 claim 的证据文本。                                                                 |
| gold evidence       | 标注数据里给出的正确证据。                                                                                  |
| train/dev/test      | train 用来训练模型；dev 用来调参和比较实验；test 用来最终提交预测，不能用 gold label 调参。                 |
| retrieval / 检索    | 从大 evidence corpus 中找出可能相关证据的过程。                                                             |
| candidate pool      | 检索阶段先粗略找出来的一批候选证据，例如 top-500。                                                          |
| sparse retrieval    | 基于词面匹配的检索方法，不直接理解深层语义，但速度快。TF-IDF 和 BM25 都属于这一类。                         |
| TF-IDF              | 根据词在当前文档中是否重要、在全语料中是否稀有来打分的传统检索方法。                                        |
| BM25                | TF-IDF 的经典改进版，常用于搜索引擎。它会考虑词频饱和、文档长度等因素。                                     |
| recall@500          | 看 gold evidence 是否出现在检索出的前 500 条候选中。它衡量“有没有把正确证据捞进候选池”。                  |
| F@5                 | 只看前 5 条 evidence 时的 evidence F-score。它更接近最终提交时的证据质量，但在 Stage 1 不是主要选择标准。   |
| reranker / 精排     | 在粗排得到的一批候选证据上，用更强但更慢的模型重新排序。                                                    |
| cross-encoder       | 把 claim 和 evidence 拼在一起输入 Transformer，让模型直接判断两者是否相关。通常效果好，但速度比简单检索慢。 |
| Transformer         | 现代 NLP 常用的神经网络结构，BERT、DistilBERT、MiniLM 都属于这一类。                                        |
| DistilBERT          | BERT 的轻量版，速度较快，参数较少。                                                                         |
| MiniLM              | 小型 Transformer 模型，通常比 BERT 更轻，适合算力有限时使用。                                               |
| LSTM                | 比 Transformer 更早的一类序列模型。上次计划提到过，但当前实验没有实现。                                     |
| backbone            | 模型底座。比如 DistilBERT 和 MiniLM 是不同 backbone，再在上面接分类头做具体任务。                           |
| MS MARCO            | 一个检索/问答相关数据集。`ms-marco-MiniLM` 表示这个模型曾在检索相关任务上预训练或微调过。                 |
| NLI                 | Natural Language Inference，自然语言推理任务，判断两个句子之间是蕴含、矛盾还是无关。                        |
| classifier / 分类器 | 根据 claim 和 evidence 预测 label 的模型。                                                                  |
| logit               | 模型输出的原始分数。分数越高通常表示模型越倾向于某个判断，但不同 claim 之间的 logit 不一定能直接比较。      |
| threshold / 阈值    | 设一个固定分数线，只保留分数高于这条线的 evidence。                                                         |
| fixed-K             | 每个 claim 固定保留分数最高的 K 条 evidence。                                                               |
| relative-delta      | 每个 claim 先看最高分，再保留“离最高分不超过 delta”的 evidence。它是相对阈值，不是全局固定阈值。          |
| alpha fusion        | 把 BM25 分数和 reranker 分数按某个比例混合。当前实验没有做。                                                |
| class weighting     | 训练分类器时给少数类更高权重，试图缓解类别不平衡。                                                          |
| baseline            | 简单参照方法。它不一定强，但用来判断复杂模型是否真的有提升。                                                |
| confusion matrix    | 混淆矩阵。行是真实 label，列是预测 label，用来看模型具体把哪些类混成了哪些类。                              |
| seed variance       | 换不同随机种子重新训练，看结果波动有多大。                                                                  |
| oracle retrieval    | 诊断实验：假设 retrieval 完美，直接把 gold evidence 给 classifier，看分类器本身能做到什么程度。             |

## 和上次计划的对应关系

从计划到后续实际实现时做了一些调整。下面是对应关系。

| 上次计划                                                                                | 当前实际完成                                                                                                            |
| --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| 预处理：Lowercase + Stopwords；Lowercase + Stopwords + Lemmatization                    | 实际比较了 3 种：lowercase only、lowercase + stopwords、lowercase + stopwords + Porter stemming。没有做 lemmatization。 |
| 粗排：TF-IDF、BM25s、cross encoder                                                      | 第一阶段粗排比较了 TF-IDF 和 BM25。cross-encoder 放在第二阶段 reranker 中使用。                                         |
| BM25s 换参数：粗排 TopK、alpha 系数                                                     | 当前固定 first-stage top-500，调了 BM25 的 `k1` 和 `b`。没有做 BM25-reranker alpha fusion。                         |
| 精排模型用 MiniLM，得到每条 evidence 的相关分数 logit，再比较 fixed-K、threshold、delta | 已完成。最终 relative-delta `delta=1.5` 最好。                                                                        |
| 分类器：LSTM、DistilBERT、MiniLM                                                        | 实际比较了 DistilBERT 和两个 MiniLM-style transformer。没有做 LSTM。                                                    |

## 数据概况

| Split / item    |      数量 |
| --------------- | --------: |
| train claims    |     1,228 |
| dev claims      |       154 |
| test claims     |       153 |
| evidence corpus | 1,208,827 |

训练集 label 分布：

| Label           | Count |  Share |
| --------------- | ----: | -----: |
| SUPPORTS        |   519 | 42.26% |
| REFUTES         |   199 | 16.21% |
| NOT_ENOUGH_INFO |   386 | 31.43% |
| DISPUTED        |   124 | 10.10% |

Label 含义：

| Label           | 含义                                       |
| --------------- | ------------------------------------------ |
| SUPPORTS        | 证据支持 claim。                           |
| REFUTES         | 证据反驳 claim。                           |
| NOT_ENOUGH_INFO | 证据不足，无法判断。表格中有时缩写为 NEI。 |
| DISPUTED        | 证据之间存在争议或冲突。                   |

Gold evidence 数量：

| Statistic | Value |
| --------- | ----: |
| min       |     1 |
| max       |     5 |
| mean      | 3.357 |

这表示每个 claim 通常不止需要一条证据。平均每个 claim 有 3.357 条 gold evidence，所以只找一条证据往往不够。

## 1. 预处理与第一阶段检索

第一阶段目标是让 gold evidence 尽量进入 top-500 candidate pool。这里的指标主要看 dev recall@500，因为如果正确 evidence 没进入候选池，后面 reranker 和 classifier 基本无法补救。

这里的 top-500 可以理解为“先大范围捞候选”。我们不是直接提交 500 条 evidence，而是先保证正确证据有机会进入候选池，后面再用 reranker 和 selection 缩小到最多 5 条左右。

### 实际比较的预处理

| ID  | Lowercase | Stopwords | Stemmer |
| --- | --------- | --------- | ------- |
| PP1 | yes       | no        | no      |
| PP2 | yes       | yes       | no      |
| PP3 | yes       | yes       | Porter  |

说明：当前没有做 lemmatization，用 Porter stemming 代替了词形归一化方向的实验。

预处理概念解释：

| 操作              | 解释                                                                |
| ----------------- | ------------------------------------------------------------------- |
| lowercase         | 把所有词转成小写，例如 `Climate` 和 `climate` 视为同一个词。    |
| stopwords removal | 去掉常见但信息量低的词，例如 `the`, `is`, `of`。              |
| stemming          | 把词粗略截成词干，例如 `warming`, `warmed` 可能被归到相近形式。 |
| lemmatization     | 更精细的词形还原，例如把 `was` 还原为 `be`。这次没有实现。      |

### 实际比较的检索设置

| ID        | Method |  k1 |    b |
| --------- | ------ | --: | ---: |
| R-TFIDF   | TF-IDF |   - |    - |
| R-default | BM25   | 1.5 | 0.75 |
| R-b05     | BM25   | 1.5 | 0.50 |
| R-k12     | BM25   | 1.2 | 0.75 |
| R-both    | BM25   | 1.2 | 0.50 |

BM25 参数解释：

| 参数   | 含义                                                                                          |
| ------ | --------------------------------------------------------------------------------------------- |
| `k1` | 控制词频增长的影响。一个词在 evidence 中出现多次时，`k1` 决定“多出现几次”还能增加多少分。 |
| `b`  | 控制文档长度归一化。`b` 越高，越强烈惩罚过长 evidence；`b` 越低，长度惩罚越弱。           |

### 完整结果

| Rank | Preproc | Retrieval | Method | dev recall@500 | dev F@5 | Time s |  k1 |    b |
| ---: | ------- | --------- | ------ | -------------: | ------: | -----: | --: | ---: |
|    1 | PP3     | R-both    | BM25   |         0.6782 |  0.1234 |   21.4 | 1.2 | 0.50 |
|    2 | PP3     | R-b05     | BM25   |         0.6758 |  0.1217 |   24.5 | 1.5 | 0.50 |
|    3 | PP3     | R-k12     | BM25   |         0.6702 |  0.1266 |   26.1 | 1.2 | 0.75 |
|    4 | PP3     | R-default | BM25   |         0.6642 |  0.1177 |   24.7 | 1.5 | 0.75 |
|    5 | PP3     | R-TFIDF   | TF-IDF |         0.6513 |  0.0919 |    9.3 |   - |    - |
|    6 | PP2     | R-b05     | BM25   |         0.6502 |  0.1167 |   23.4 | 1.5 | 0.50 |
|    7 | PP2     | R-both    | BM25   |         0.6502 |  0.1209 |   23.6 | 1.2 | 0.50 |
|    8 | PP2     | R-k12     | BM25   |         0.6328 |  0.1140 |   24.8 | 1.2 | 0.75 |
|    9 | PP1     | R-k12     | BM25   |         0.6234 |  0.1234 |   28.2 | 1.2 | 0.75 |
|   10 | PP1     | R-b05     | BM25   |         0.6209 |  0.1258 |   29.7 | 1.5 | 0.50 |
|   11 | PP1     | R-default | BM25   |         0.6201 |  0.1070 |   23.2 | 1.5 | 0.75 |
|   12 | PP1     | R-both    | BM25   |         0.6194 |  0.1313 |   26.7 | 1.2 | 0.50 |
|   13 | PP2     | R-default | BM25   |         0.6179 |  0.1077 |   25.2 | 1.5 | 0.75 |
|   14 | PP1     | R-TFIDF   | TF-IDF |         0.6121 |  0.0900 |   12.2 |   - |    - |
|   15 | PP2     | R-TFIDF   | TF-IDF |         0.6026 |  0.0868 |    8.8 |   - |    - |

选择：

```text
Preprocessing = PP3
Retriever = BM25, k1=1.2, b=0.5
dev recall@500 = 0.6782
```

结论：

- BM25 整体优于 TF-IDF。
- 加 Porter stemming 后 recall@500 提升明显。
- 最终选择的是 PP3 + BM25 `k1=1.2`, `b=0.5`。
- 注意 dev F@5 不是这一阶段的选择标准。Stage 1 更重视 recall@500，因为它决定了后面模块的上限。

## 2. Reranker 训练

Reranker 使用第一阶段选出的 BM25 top-500 candidate pool。没有对每一种预处理或 BM25 参数重新训练 reranker，否则计算成本会成倍增加。

直观理解：BM25 像“粗筛”，速度快但不够懂语义；reranker 像“复核员”，它逐条看 claim 和 evidence 是否真的相关。因为 cross-encoder 要把 claim 和每条 evidence 拼起来跑模型，所以计算更贵，只能用在 BM25 已经筛出的候选上。

模型：

```text
cross-encoder/ms-marco-MiniLM-L-12-v2
```

训练 pair 构造：

| Item                   |  Count |
| ---------------------- | -----: |
| training pairs         | 20,610 |
| positive pairs         |  4,122 |
| negative pairs         | 16,488 |
| negatives per positive |   4.00 |

pair 构造解释：

| 类型          | 含义                                                                                        |
| ------------- | ------------------------------------------------------------------------------------------- |
| positive pair | claim 和它的 gold evidence 组成的正样本，表示“相关”。                                     |
| negative pair | claim 和 BM25 找到但不是 gold 的 evidence 组成的负样本，表示“不相关”。                    |
| hard negative | 粗排模型觉得相关、但标注上不是 gold evidence 的负样本。它比随机负样本更难，也更有训练价值。 |

训练结果：

| Epoch | dev F under quick relative-delta check |    Time s |
| ----: | -------------------------------------: | --------: |
|     1 |                                 0.1987 | about 121 |
|     2 |                                 0.1900 | about 120 |
|     3 |                                 0.1784 | about 120 |

选择：

```text
Best reranker epoch = 1
```

结论：

- 第 1 个 epoch 最好。
- 第 2、3 个 epoch 的 dev F 下降，说明 reranker 可能开始过拟合训练 pairs。

这里的 epoch 指完整看一遍训练数据。训练越久不一定越好；如果模型开始记住训练集模式，dev set 表现可能下降。

## 3. Evidence selection 策略比较

Reranker 会给每个候选 evidence 一个相关性 logit。之后需要决定保留哪些 evidence 作为最终预测证据。

为什么还需要 selection：reranker 只是排序或打分，但最终提交不能把 500 条候选都交上去。我们需要决定每个 claim 最终保留几条 evidence。

比较了三类策略：

| Strategy       | Grid                              |
| -------------- | --------------------------------- |
| fixed-K        | K = 3, 4, 5                       |
| threshold      | 多个全局阈值                      |
| relative-delta | 根据每个 claim 的最高分做相对筛选 |

三种策略白话解释：

| Strategy       | 解释                                                                                        |
| -------------- | ------------------------------------------------------------------------------------------- |
| fixed-K        | 不管分数差距多大，每个 claim 都固定保留 K 条。优点是简单，缺点是不灵活。                    |
| threshold      | 设置一个全局分数线，超过就保留。缺点是不同 claim 的分数分布可能不一样。                     |
| relative-delta | 以每个 claim 自己的最高分为参照，保留接近最高分的 evidence，更适合 logit 不完全校准的情况。 |

完整结果：

| Rank | Strategy       | Param |  dev F | Avg selected evidence |
| ---: | -------------- | ----: | -----: | --------------------: |
|    1 | relative-delta |  1.50 | 0.2448 |                  4.05 |
|    2 | relative-delta |  1.00 | 0.2329 |                  3.39 |
|    3 | relative-delta |  2.00 | 0.2321 |                  4.46 |
|    4 | relative-delta |  3.00 | 0.2261 |                  4.80 |
|    5 | relative-delta |  0.75 | 0.2203 |                  3.00 |
|    6 | fixed-K        |  3.00 | 0.2156 |                  3.00 |
|    7 | fixed-K        |  4.00 | 0.2149 |                  4.00 |
|    8 | relative-delta |  0.50 | 0.2084 |                  2.32 |
|    9 | threshold      |  0.90 | 0.2079 |                  2.74 |
|   10 | threshold      |  0.24 | 0.2060 |                  4.99 |
|   11 | threshold      |  0.08 | 0.2059 |                  5.00 |
|   12 | fixed-K        |  5.00 | 0.2059 |                  5.00 |
|   13 | threshold      |  0.06 | 0.2059 |                  5.00 |
|   14 | threshold      |  0.04 | 0.2059 |                  5.00 |
|   15 | threshold      |  0.02 | 0.2059 |                  5.00 |

选择：

```text
Strategy = relative-delta
delta = 1.5
Final dev retrieval F = 0.2448
Average selected evidence = 4.05
```

结论：

- relative-delta 最好。
- fixed-K 和 threshold 都不如 relative-delta。
- 这说明不同 claim 的 reranker logits 不完全可比，用全局阈值不够稳。

## 4. 分类器模型对比

上次计划里提到 LSTM、DistilBERT、MiniLM。目前实际做的是三个 transformer backbone 对比，没有做 LSTM。

这里的 classifier 做的不是找 evidence，而是判断 claim label。也就是说，retrieval/reranker 负责“找证据”，classifier 负责“读 claim 和证据后判断标签”。输入大致是：

```text
claim text + selected evidence text
```

输出是四个 label 之一：

| Label           | 含义                                           |
| --------------- | ---------------------------------------------- |
| SUPPORTS        | 证据支持 claim。                               |
| REFUTES         | 证据反驳 claim。                               |
| NOT_ENOUGH_INFO | 证据不足，无法判断。后文表格中有时缩写为 NEI。 |
| DISPUTED        | 证据之间存在争议或冲突。                       |

训练设置：

- 训练时 evidence 输入：gold evidence 和 retrieved evidence 合并，去重后最多 5 条。
- dev/test inference：只使用 retrieved evidence。
- learning rate：`1e-5`
- epochs：4
- batch size：train 16, eval 32
- class weighting：固定为 sqrt inverse frequency，用于公平比较 backbone。

这里的 backbone 可以理解为“预训练模型底座”。不同 backbone 在预训练任务、参数规模和文本理解能力上不同，因此同样的训练数据下表现会不同。

模型名补充：

| 模型         | 简单解释                                                                |
| ------------ | ----------------------------------------------------------------------- |
| DistilBERT   | BERT 的压缩版，比较通用，但不专门针对检索任务。                         |
| MiniLM-MARCO | MiniLM 在 MS MARCO 检索任务上训练过，更适合 claim-evidence 相关性场景。 |
| MiniLM-NLI   | MiniLM 在自然语言推理任务上训练过，理论上可能更擅长支持/反驳关系。      |

完整结果：

| Rank | Backbone        | Model                                     | SUPPORTS acc | REFUTES acc | NEI acc | DISPUTED acc | Best epoch |      F |      A |      H |
| ---: | --------------- | ----------------------------------------- | -----------: | ----------: | ------: | -----------: | ---------: | -----: | -----: | -----: |
|    1 | C2-minilm-marco | `cross-encoder/ms-marco-MiniLM-L-12-v2` |       0.9706 |      0.0000 |  0.2439 |       0.0000 |          2 | 0.2448 | 0.4935 | 0.3273 |
|    2 | C3-minilm-nli   | `cross-encoder/nli-MiniLM2-L6-H768`     |       0.8088 |      0.3333 |  0.1951 |       0.1111 |          4 | 0.2448 | 0.4805 | 0.3243 |
|    3 | C1-distilbert   | `distilbert-base-uncased`               |       0.8088 |      0.0370 |  0.3171 |       0.0000 |          3 | 0.2448 | 0.4481 | 0.3166 |

选择：

```text
Classifier = cross-encoder/ms-marco-MiniLM-L-12-v2
```

结论：

- MiniLM-MARCO 的 overall accuracy 最高，因此被选中。
- 但它强烈偏向 SUPPORTS，对 REFUTES 和 DISPUTED 很弱。
- NLI-MiniLM 的 overall A 略低，但对 REFUTES 和 DISPUTED 更好，可以作为报告里的 trade-off 分析。

## 5. Class weighting 对比

选定 MiniLM-MARCO classifier 后，比较不同 class weighting。

为什么需要 class weighting：训练集中 SUPPORTS 最多，DISPUTED 最少。如果不处理类别不平衡，模型容易学成“多数时候猜 SUPPORTS”。Class weighting 的想法是：少数类错了惩罚更大，让模型更重视少数类。

| ID | Weighting              | Power |
| -- | ---------------------- | ----: |
| W0 | no weighting           |   0.0 |
| W1 | sqrt inverse frequency |   0.5 |
| W2 | full inverse frequency |   1.0 |

权重强度解释：

| 设置 | 直观含义               |
| ---- | ---------------------- |
| W0   | 不调整，各类同等权重。 |
| W1   | 温和提高少数类权重。   |
| W2   | 强烈提高少数类权重。   |

完整结果：

| Weighting | Power | SUPPORTS recall | REFUTES recall | NEI recall | DISPUTED recall |      F |      A |      H |
| --------- | ----: | --------------: | -------------: | ---------: | --------------: | -----: | -----: | -----: |
| W0        |   0.0 |          0.9559 |         0.0000 |     0.0976 |          0.2778 | 0.2448 | 0.4805 | 0.3243 |
| W1        |   0.5 |          0.9706 |         0.0000 |     0.2195 |          0.0000 | 0.2448 | 0.4870 | 0.3258 |
| W2        |   1.0 |          0.1471 |         0.0741 |     0.2439 |          0.8889 | 0.2448 | 0.2468 | 0.2458 |

选择：

```text
Weighting = W0, no class weighting
```

选择原因：

- W1 的 A 和 H 略高，但 DISPUTED recall 为 0。
- W0 的 A 只略低，但 DISPUTED recall 有 0.2778。
- W2 虽然 DISPUTED recall 很高，但 overall A 崩到 0.2468。

结论：

- class weighting 没有彻底解决 minority class 问题。
- W0 是一个折中选择，避免整体准确率大幅下降，同时保留一些 DISPUTED recall。

## 6. 最终 dev set 表现

最终整体指标：

|      F |      A |      H |
| -----: | -----: | -----: |
| 0.2448 | 0.4805 | 0.3243 |

Confusion matrix，行是 gold label，列是 predicted label：

如何读 confusion matrix：看每一行。比如 REFUTES 行表示 gold label 是 REFUTES 的 27 条 claim，其中 24 条被预测成 SUPPORTS，3 条被预测成 NOT_ENOUGH_INFO，0 条被预测成 REFUTES。

| Gold \ Pred     | SUPPORTS | REFUTES | NOT_ENOUGH_INFO | DISPUTED |
| --------------- | -------: | ------: | --------------: | -------: |
| SUPPORTS        |       65 |       0 |               2 |        1 |
| REFUTES         |       24 |       0 |               3 |        0 |
| NOT_ENOUGH_INFO |       37 |       0 |               4 |        0 |
| DISPUTED        |       13 |       0 |               0 |        5 |

Per-class 结果：

| Label           |  n | Class acc | Retrieval F |
| --------------- | -: | --------: | ----------: |
| SUPPORTS        | 68 |    0.9559 |      0.3418 |
| REFUTES         | 27 |    0.0000 |      0.0558 |
| NOT_ENOUGH_INFO | 41 |    0.0976 |      0.1764 |
| DISPUTED        | 18 |    0.2778 |      0.3177 |

主要问题：

- SUPPORTS 表现很好，但模型明显偏向 SUPPORTS。
- REFUTES 完全没有预测对。
- NOT_ENOUGH_INFO 和 DISPUTED 仍然困难。

## 7. Baseline 对比

为了判断当前系统到底比简单 baseline 强多少，比较了两个 baseline：

1. 所有 claim 都预测 SUPPORTS，evidence 用当前最终 retrieval；
2. 所有 claim 都预测 SUPPORTS，evidence 用 BM25 top-5。

baseline 的作用不是追求最好，而是提供参照线。如果复杂系统只比 baseline 高一点，说明复杂模块的收益有限；如果明显高很多，说明新模块确实有价值。

结果：

| Run                            |      F |      A |      H |
| ------------------------------ | -----: | -----: | -----: |
| all SUPPORTS + final retrieval | 0.2448 | 0.4416 | 0.3150 |
| all SUPPORTS + BM25 top-5      | 0.1234 | 0.4416 | 0.1929 |
| final system                   | 0.2448 | 0.4805 | 0.3243 |

解释：

- 当前 retrieval 比 raw BM25 top-5 明显更好，F 从 0.1234 到 0.2448。
- classifier 比 all-SUPPORTS baseline 只提升了约 3.9 个 accuracy points。
- 所以最终系统的提升主要来自 retrieval pipeline，classification 部分还有较大空间。

为什么 all SUPPORTS 是一个有用 baseline：训练和 dev 数据里 SUPPORTS 占比最高，所以“永远猜 SUPPORTS”虽然很笨，但 accuracy 不会特别低。我们的 classifier 必须超过它，才说明确实学到了一些东西。

## 8. Seed variance

为了看 classifier 稳定性，用相同最终配置重新训练 3 个 seed。

seed 是随机种子。训练神经网络时，参数初始化、batch 顺序等都有随机性。换 seed 后，如果结果差很多，说明模型不够稳定，单次结果不能过度解读。

|        Seed | Best epoch |                F |                A |                H |
| ----------: | ---------: | ---------------: | ---------------: | ---------------: |
|          42 |          4 |           0.2448 |           0.5065 |           0.3301 |
|         123 |          3 |           0.2448 |           0.5000 |           0.3287 |
|        2024 |          3 |           0.2448 |           0.4610 |           0.3198 |
| mean ± std |          - | 0.2448 ± 0.0000 | 0.4892 ± 0.0201 | 0.3262 ± 0.0046 |

解释：

- F 不变，因为 retrieval 固定。
- A 的标准差约 0.0201，说明 classifier 有一定随机波动。
- 当前 final A=0.4805 处于正常 seed variance 范围内。

这里的 mean ± std 表示“平均值 ± 标准差”。标准差越大，说明结果越不稳定。

## 9. LLM-based DISPUTED cascade 尝试

尝试用一个小型 instruction LLM：

```text
Qwen/Qwen2.5-3B-Instruct
```

对于当前预测为 SUPPORTS 或 REFUTES、且至少有两条 retrieved evidence 的 claim，询问：

```text
Do the evidence pieces contradict each other regarding the claim? Answer YES or NO.
```

如果回答 YES，就把预测改成 DISPUTED。

这个实验的动机：DISPUTED 类通常和“证据之间互相冲突”有关，所以尝试让 LLM 做一个额外检查。如果 LLM 发现 evidence 之间有矛盾，就把原本 SUPPORTS/REFUTES 的预测改成 DISPUTED。

结果：

| Item                 |  Value |
| -------------------- | -----: |
| evaluated candidates |    130 |
| flipped claims       |      0 |
| F after cascade      | 0.2448 |
| A after cascade      | 0.4805 |
| H after cascade      | 0.3243 |

Per-class accuracy：

| Label           | Base acc | Cascade acc |
| --------------- | -------: | ----------: |
| SUPPORTS        |   0.9559 |      0.9559 |
| REFUTES         |   0.0000 |      0.0000 |
| NOT_ENOUGH_INFO |   0.0976 |      0.0976 |
| DISPUTED        |   0.2778 |      0.2778 |

解释：

- LLM 没有触发任何 label flip。
- 这说明这个简单 prompt 和模型配置没有带来改进。
- 可以作为 negative result，但不建议在报告里把它写成主要方法。

negative result 的意思是：实验跑了，但没有提升。它仍然有价值，因为它告诉我们这个方向在当前设置下不值得作为主方法。

## 10. Oracle retrieval 诊断

这个实验把 dev set 的 retrieval 替换成 gold evidence，用来判断“如果 evidence 完美，classifier 会怎样”。

oracle 在这里不是一个真实可用于 test set 的方法，而是诊断工具。因为 test set 没有 gold evidence，所以不能真正用 oracle retrieval 提交。它的作用是拆分错误来源：到底是 evidence 没找对，还是 evidence 给对了 classifier 也判断不好。

整体结果：

| Setting                 |      F |      A |      H |
| ----------------------- | -----: | -----: | -----: |
| current retrieval       | 0.2448 | 0.4805 | 0.3243 |
| gold evidence retrieval | 1.0000 | 0.4091 | 0.5806 |

Per-class accuracy：

| Label           | Current retrieval acc | Gold evidence acc |
| --------------- | --------------------: | ----------------: |
| SUPPORTS        |                0.9559 |            0.7206 |
| REFUTES         |                0.0000 |            0.0000 |
| NOT_ENOUGH_INFO |                0.0976 |            0.2927 |
| DISPUTED        |                0.2778 |            0.1111 |

解释：

- gold evidence 让 F=1，所以 H 明显升高。
- 但是 classifier accuracy 反而从 0.4805 降到 0.4091。
- 这说明 classifier 不只是依赖“证据是否正确”，还明显受输入分布影响。它可能学到了 noisy retrieval 下的模式或 label prior。
- 因此系统瓶颈不只是 retrieval，也包括 classifier 的 evidence reasoning 能力。

这里有一个反直觉点：gold evidence 更正确，但 classifier accuracy 下降。这并不表示 gold evidence 不好，而是说明 classifier 训练出来的决策方式不够稳，换一种 evidence 输入分布后就会变差。

## 11. Error bucket 分析

按两个条件分桶：

1. retrieval 是否至少命中一条 gold evidence；
2. claim label 是否预测正确。

结果：

| Bucket                 |  n | Mean retrieval F | Share |
| ---------------------- | -: | ---------------: | ----: |
| all_correct            | 43 |           0.5657 | 27.9% |
| classification_failure | 38 |           0.3519 | 24.7% |
| lucky_classification   | 31 |           0.0000 | 20.1% |
| retrieval_failure      | 42 |           0.0000 | 27.3% |

含义：

| Bucket                 | 含义                                                |
| ---------------------- | --------------------------------------------------- |
| all_correct            | retrieval 至少命中一条 gold evidence，且 label 正确 |
| classification_failure | retrieval 有命中，但 label 错                       |
| lucky_classification   | retrieval 没命中，但 label 正确                     |
| retrieval_failure      | retrieval 没命中，label 也错                        |

解释：

- 真正 retrieval 和 classification 都正确的只有 27.9%。
- 24.7% 是 classification failure，说明即使 evidence 有命中，classifier 仍会错。
- 27.3% 是 retrieval failure，说明检索仍然是大瓶颈。
- 20.1% 是 lucky classification，说明模型有时可能靠 label prior 或 claim text 猜对。

这个分桶的作用是把错误拆开看。如果很多错误是 retrieval_failure，那应该优先改检索；如果很多错误是 classification_failure，那说明证据已经部分找到了，但分类器不会用。当前两类问题都不少。

## 12. 可写进报告的主要结论

1. BM25 明显优于 TF-IDF；lowercase + stopwords + Porter stemming 的 recall@500 最好。
2. Cross-encoder reranker 提升了最终 evidence F，但训练超过 1 epoch 后 dev F 下降，有过拟合迹象。
3. Evidence selection 中，relative-delta 比 fixed-K 和 global threshold 更稳。
4. MiniLM-MARCO classifier 的 overall accuracy 最高，但严重偏向 SUPPORTS。
5. Class weighting 不能完全解决类别不平衡；W2 提高 DISPUTED recall 的代价是 overall A 大幅下降。
6. 当前系统比 majority-label baseline 有提升，但 classification 提升有限。
7. Oracle retrieval 说明 classifier 的 evidence reasoning 仍不稳定，正确 evidence 不一定带来更高 label accuracy。
8. Error bucket 显示 retrieval failure 和 classification failure 都很多，后续提升需要同时处理两端。

如果报告篇幅有限，最建议强调三点：第一，retrieval pipeline 确实比 raw BM25 top-5 强；第二，classifier 有明显 SUPPORTS 偏置；第三，oracle retrieval 和 error bucket 说明系统瓶颈是 retrieval 与 classification 共同造成的。

## 13. 建议报告叙事

报告中可以把方法描述为 sequential greedy ablation：

1. 用 dev recall@500 选择第一阶段 sparse retrieval；
2. 在最佳 BM25 candidate pool 上训练一个 cross-encoder reranker；
3. 用 dev evidence F 选择 evidence selection 策略；
4. 比较 transformer classifier backbone；
5. 对最终 classifier 做 class weighting ablation；
6. 用 baseline、seed variance、oracle retrieval 和 error bucket 分析系统局限。

需要主动说明：这不是 full factorial search，而是 sequential greedy ablation。原因是对每个预处理和检索组合都重新训练 reranker/classifier 的成本太高。

full factorial search 指把所有模块的所有设置组合都跑一遍，例如每个预处理 × 每个检索参数 × 每个 reranker 设置 × 每个 classifier 设置。这样最全面，但计算量巨大。Sequential greedy ablation 是先选当前阶段最好的设置，再固定它进入下一阶段；它更省算力，但可能错过某些跨模块组合的最优解。
