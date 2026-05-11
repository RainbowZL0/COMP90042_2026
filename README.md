# COMP90042 Project Description

Please check the **lecture recording first** if you have any questions about Assignment 3 – Project (**Lecture 14,
Friday, 24 April 2026**).

This assignment is to be completed in teams of three (3). We strongly encourage effective and respectful collaboration.
Please refer to
the [University of Melbourne Working in Groups Guide](https://students.unimelb.edu.au/academic-skills/resources/communicating-in-class/communicating-with-peers/working-in-groups).

If a team member is not contributing adequately, please contact the
lecturer ([Dr. Caren Han](mailto:caren.han@unimelb.edu.au?subject=[COMP90042]%20Project%20Group)) with a clear
description of each member’s contributions. We strongly recommend starting early to allow sufficient time to identify
challenges and seek support when needed.

You are free to design your system using any of the techniques covered in lectures and labs, provided that they comply
with the **Project Rules (see the Important Notes section below).**

For training and evaluation, a benchmark dataset is provided with training, validation, and test splits. You should use
the training and validation sets for model development and tuning, and the test set only for final leaderboard
submission.

For this assignment, **your primary goal is not simply to achieve the highest performance, but to demonstrate thoughtful
system design and clear reasoning behind your choices**. You are encouraged to explore novel approaches, justify your
design decisions, and critically analyse your results.

**NOTE: Grading will primarily focus on your research process, methodological soundness, and clarity of analysis**,
rather than raw system performance (see grading details below).

**Table of Contents**

- [0. Important Dates (Very Important!!)](https://github.com/drcarenhan/COMP90042_2026?tab=readme-ov-file#-0-important-dates)
- [1. DataSet](https://github.com/drcarenhan/COMP90042_2026?tab=readme-ov-file#-1-dataset)
- [2. Important Notes](https://github.com/drcarenhan/COMP90042_2026?tab=readme-ov-file#-2-important-notes)
- [3. Model Testing](https://github.com/drcarenhan/COMP90042_2026?tab=readme-ov-file#-3-testing-and-evaluation)
- [4. Report Writing](https://github.com/drcarenhan/COMP90042_2026?tab=readme-ov-file#-4-report-writing)
- [5. Project Submission Method and Grading](https://github.com/drcarenhan/COMP90042_2026?tab=readme-ov-file#-5-project-submission-method-and-grading)
- [6. Peer Review](https://github.com/drcarenhan/COMP90042_2026?tab=readme-ov-file#-6-peer-review)
- [7. Leaderboard](https://github.com/drcarenhan/COMP90042_2026?tab=readme-ov-file#-7-leaderboard)
- [8. FAQ](https://github.com/drcarenhan/COMP90042_2026?tab=readme-ov-file#-8-faq)

<br/>
<br/>

## <img src="https://em-content.zobj.net/thumbs/120/microsoft/319/calendar_1f4c5.png" width="30" /> 0. Important Dates

The important dates for the project are as follows:

- **Project Specification Release Date**: 24 April 2026
- **Project Group Release Date**: 25 April 2026
- **Project Final Report & Code Submission Due**:  22 May 2026 **(NO extension will be given less than 2 days before
  deadline)**
- **Peer Review – Part 1 Due**: 27 May 2026  
  *(Starts: 9:00 PM, 24 May 2026 — No extension)*
- **Peer Review – Part 2 Due**: 29 May 2026  
  *(Starts: 9:00 AM, 28 May 2026 — No extension)*

All deadlines are **11:59 PM (AEST, Melbourne Time)**.
<br/>
**Leaderboard (Optional)**  
The leaderboard is optional and will run from **1 May to 22 May 2026 (11:59 PM AEST)**.


<br/>

## <img src="https://em-content.zobj.net/thumbs/120/samsung/349/card-file-box_1f5c3-fe0f.png" width="30" /> 1. DataSet

| :exclamation:  You need to put the code that you conduct all actions for this section in the [ipynb template](https://colab.research.google.com/drive/1CjlVXdEsioH_iGOHUbmrhimTLRXGJIt0?usp=sharing) |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

The impact of climate change on humanity is a significant concern. However, the increase in unverified statements
regarding climate science has led to a distortion of public opinion, underscoring the importance of conducting
fact-checks on claims related to climate science. Consider the following claim and related evidence:

**Claim**: The Earth’s climate sensitivity is so low that a doubling of atmospheric CO2 will result in a surface
temperature change on the order of 1°C or less.

**Evidence:**

1. In his first paper on the matter, he estimated that global temperature would rise by around 5 to 6 °C (9.0 to 10.8
   °F) if the quantity of CO 2 was doubled.
2. The 1990 IPCC First Assessment Report estimated that equilibrium climate sensitivity to a doubling of CO2 lay between
   1.5 and 4.5 °C (2.7 and 8.1 °F), with a "best guess in the light of current knowledge" of 2.5 °C (4.5 °F).

It should not be difficult to see that the claim is not supported by the evidence passages, and assuming the source of
the evidence is reliable, such a claim is misleading.

### Task Description

The goal of this project is to develop an automated fact-checking system.  
Given a claim, your system must:

1. **Retrieve** the most relevant evidence passages from a corpus (the *knowledge source* - evidence.json), and
2. **Classify** the claim based on the retrieved evidence into one of the following labels:  
   `{SUPPORTS, REFUTES, NOT_ENOUGH_INFO, DISPUTED}`

More concretely, you will be provided a list of claims and a corpus containing a large number of evidence passages (the
“knowledge source”), and your system must: (1) search for the most related evidence passages from the knowledge source
given the claim; and (2) classify the status of the claim given the evidence in the following 4 classes: {SUPPORTS,
REFUTES, NOT_ENOUGH_INFO, DISPUTED}.

To build a successful system, it must be able to retrieve the correct set of evidence passages and classify the claim
correctly.

Besides system implementation, you must also write a report that describes your fact-checking system, e.g. how the
retrieval and classification components work, the reason behind the choices you made and the system’s performance. We
hope that you will enjoy the project. To make it more engaging, **we will run the task as a leaderboard (participation
is optional; more details below)**. You will be competing with other students in the class. The following sections give
more details on the data format, system evaluation, grading scheme and use of the leaderboard. Your assessment will be
graded based on your report, and your code.

### Project Components

You are provided with several files for the project:

* [train-claims,dev-claims].json: JSON files for the labelled training and development set;
* [test-claims-unlabelled].json: JSON file for the unlabelled test set;
* dev-claims-baseline.json: JSON file containing predictions of a baseline system on the development set;
* evidence.json: JSON file containing a number of evidence passages (i.e. the “knowledge source”);
* eval.py: Python script to evaluate system performance (see “Evaluation” below for more details).

### Data Format

For the labelled claim files (train-claims.json, dev-claims.json), each instance contains the claim ID, claim text,
claim label (one of the four classes: {SUPPORTS, REFUTES, NOT_ENOUGH_INFO, DISPUTED}), and a list of evidence IDs. The
unlabelled claim file (test-claims-unlabelled.json) has a similar structure, except that it only contains the claim ID
and claim text. More concretely, the labelled claim files has the following format:

```
{
  "claim-2967":
  {
    claim_text: "[South Australia] has the most expensive electricity in the world."
    claim_label: "SUPPORTS"
    evidences: ["evidence-67732", "evidence-572512"]
  },
  "claim-375":
  ...
}
```

The list of evidence IDs (e.g. evidence-67732, evidence-572512) are drawn from the evidence passages in evidence.json:

```
{
  "evidence-0": "John Bennet Lawes, English entrepreneur and agricultural scientist",
  "evidence-1": "Lindberg began his professional career at the age of 16, eventually ...",
  ...
}
```

Given a claim (e.g. claim-2967), your system needs to search and retrieve a list of the most relevant evidence passages
from evidence.json, and classify the claim (1 out of the 4 classes mentioned above). You should retrieve at least one
evidence passage. So, for each claim, your system must:

- Retrieve **at least one** relevant evidence passage
- Predict the correct claim label

### Data Usage Guidelines

- The **training set** (train-claims.json) should be used for building your models, e.g. for use in development of
  features, rules and heuristics, and for supervised/unsupervised learning. You are encouraged to inspect this data
  closely to fully understand the task.
- The **development set** (dev-claims.json) is formatted like the training set. This will help you make major
  implementation decisions (e.g. choosing optimal hyper-parameter configurations), and should also be used for detailed
  analysis of your system — both for measuring performance and for error analysis — in the report.
- You will use the **test set (test-claims-unlabelled.json)** to participate in the leaderboard **(optional)**. For this
  reason, no labels (i.e. the evidence passages and claim labels) are provided for this partition. You are allowed (and
  encouraged) to train your final system on both the training and development set so as to maximise performance on the
  test set, but you should not at any time manually inspect the test dataset; any sign that you have done so will result
  in loss of marks. In terms of the format of the system output, we have provided dev-claims-predictions.json for this.
  **Note: you’ll notice that it has the same format as the labelled claim files (train-claims.json or dev-claims.json),
  although the claim_text field is optional (i.e. we do not use this field during evaluation) and you’re free to omit.**

<br/>

## <img src="https://em-content.zobj.net/thumbs/120/whatsapp/326/desktop-computer_1f5a5-fe0f.png" width="30" /> 2. Important Notes

| :exclamation:  You need to put the code that you conduct all actions for this section in the [ipynb template](https://colab.research.google.com/drive/1CjlVXdEsioH_iGOHUbmrhimTLRXGJIt0?usp=sharing) |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

### You MUST follow the rules below. Any team found to violate these rules will receive **zero marks** for the project.

### 1) Model Design

You are encouraged to explore different models for the task.  
Your system MUST include at least one sequence modelling component based on one of the following architectures: RNN,
LSTM, GRU, or Transformer.

You may use deep learning libraries (e.g., PyTorch) to implement these components (i.e., you do not need to implement
them from scratch).

You are encouraged to read relevant publications to guide your design. However, **you MUST NOT directly copy open-source
project code** (e.g., from GitHub or published repositories). Your system must be implemented by your team.

### 2) Use of Large Language Models (LLMs)

You are allowed to use Large Language Models (LLMs) as part of your system, provided that the following conditions are
met:

- **Only open-source models are allowed.**  
  You MUST NOT use any closed-source APIs or proprietary systems (e.g., OpenAI GPT, Claude, Gemini, Copilot).
- **Models must be runnable on the free version of Google Colab.**  
  This means:
    - The model must fit within standard Colab resource limits (e.g., ~12GB RAM)
    - It must run without requiring paid APIs or external compute
- **You may use LLMs in any way**, including:
    - prompting / in-context learning
    - fine-tuning
    - parameter-efficient tuning (e.g., LoRA)
    - integration into hybrid architectures
- **However, your contribution must be clear and substantial.**  
  Simply using an off-the-shelf model (e.g., prompting a pretrained LLM without modification or design justification)
  will **not** be sufficient for full marks.
- You must clearly describe in your report:
    - how the LLM is used
    - what design decisions you made (e.g., model selection, training strategy, system architecture) and why
    - and what your **technical contribution** is beyond the base model
- We strongly encourage designing **hybrid systems** (e.g., retrieval + LLM, or structured reasoning + LLM) rather than
  relying solely on a single pretrained model.

### 3) Prohibited Methods

You **MUST NOT** use:

- Any closed-source APIs or models (e.g., OpenAI GPT-3/4, Claude, Gemini, Copilot)
- Any hand-crafted if-then rules for classification or prediction logic

### 4) Libraries and Code Usage

The following libraries **are allowed**:

- Deep learning: PyTorch, TensorFlow, Keras
- HuggingFace Transformers (for model loading, training, and inference)
- Standard Python libraries (e.g., NumPy, Matplotlib)
- NLP toolkits (e.g., NLTK, spaCy)

You may use **the code provided in workshops**

You **MUST NOT** use or submit external project implementations (e.g., copying full solutions or repositories from
GitHub).
You may refer to external resources for understanding or inspiration, but your final system must be implemented by your
team.

### 5) Reproducibility

The model described in your report **MUST match** your submitted code and results.

You MUST include:

- Running logs
- Reported results

in your submitted `.ipynb` file.

### 6) Integrity

You **MUST NOT** submit results that were not generated by your final submitted code.

**Post-hoc modifications (e.g., manual edits to predictions)** are strictly prohibited.

### 7) Compute Constraints

You **MUST NOT** use models that cannot be run on the free version of Google Colab.

### 8) Code Template

You **MUST** use the
provided [ipynb template](https://colab.research.google.com/drive/1CjlVXdEsioH_iGOHUbmrhimTLRXGJIt0?usp=sharing).  
You **may extend or restructure it, but core components** must remain intact for grading compatibility.

### 9) Data Usage

You **MUST use only the provided training and development datasets** for model training, tuning, and evaluation.

You **MUST NOT** use any additional external datasets for training or evaluation.

The use of **pretrained open-source models (e.g., LLMs)** is allowed, provided they comply with the LLM usage rules
above.

## <img src="https://em-content.zobj.net/source/skype/289/test-tube_1f9ea.png" width="30" /> 3. Testing and Evaluation

| :exclamation:  You need to put the code that you conduct all actions for this section to the [ipynb template](https://colab.research.google.com/drive/1CjlVXdEsioH_iGOHUbmrhimTLRXGJIt0?usp=sharing) |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

**IMPORTANT: please make sure that you check the IMPORTANT NOTES**

### Evaluation Overview

We provide a script (eval.py) for evaluating your system. This script takes two input files, the ground truth and your
predictions, and computes three metrics: (1) F-score for evidence retrieval; (2) accuracy for claim classification;
and (3) harmonic mean of the evidence retrieval F-score and claim classification accuracy. Shown below is the output
from running predictions of a baseline system on the development set:

```
$ python eval.py --predictions dev-claims-baseline.json --groundtruth dev-claims.json
Evidence Retrieval F-score (F)    = 0.3377705627705628
Claim Classification Accuracy (A) = 0.35064935064935066
Harmonic Mean of F and A          = 0.3440894901357093
```

### Metric Definitions

The **three metrics** are computed as follows:

1. **Evidence Retrieval F-score (F)**: computes how well the evidence passages retrieved by the system match the ground
   truth evidence passages. For each claim, our evaluation considers all the retrieved evidence passages, computes the
   precision, recall and F-score by comparing them against the ground truth passages, and aggregates the F-scores by
   averaging over all claims. e.g. given a claim if a system retrieves the following set {evidence-1, evidence-2,
   evidence-3, evidence-4, evidence-5}, and the ground truth set is {evidence-1, evidence-5, evidence-10}, then
   precision = 2/5, recall = 2/3, and F-score = 1/2. The aim of this metric is to measure how well the retrieval
   component of your fact checking system works.

2. **Claim Classification Accuracy (A)**: computes standard classification accuracy for claim label prediction, ignoring
   the set of evidence passages retrieved by the system. This metric assesses solely how well the system classifies the
   claim, and is designed to understand how well the classification component of your fact checking system works.

3. **Harmonic Mean of F and A**: computes the harmonic mean of the evidence retrieval F-score and claim classification
   accuracy. Note that this metric is computed at the end after we have obtained the aggregate (over all claims) F-score
   and accuracy. This metric is designed to assess both the retrieval and classification components of your system, and
   as such will be used as **the main metric for ranking systems on the leaderboard.**

The first two metrics (F-score and accuracy) are provided to help diagnose and develop your system. While they are not
used to rank your system on the leaderboard, you should document them in your report and use them to discuss the
strengths/weaknesses of your system.

A strong system should balance both retrieval quality and classification performance.

### Baseline System

The example prediction file, dev-claims-baseline.json, is the output of a baseline system on the development set. This
file will help you understand the required file format for creating your development output (for tuning your system
using eval.py) and your test output (for submission to the Leaderboard).

Note that this is not a realistic baseline, and you might find that your system performs worse than it. The reason for
this is that this baseline constructs its output in the following manner: (1) the claim labels are randomly selected;
and (2) the set of evidence passages combines several randomly selected ground truth passages and several randomly
selected passages from the knowledge source. We created such a ‘baseline’ because a true random baseline that selects a
random set of evidence passages will most probably produce a zero F-score for evidence retrieval (and consequently zero
for the harmonic mean of F-score and accuracy), and it won’t serve as a good diagnostic example to explain the metrics.
To clarify, this baseline will not be used in any way for ranking submitted systems on the leaderboard, and is provided
solely to illustrate the metrics and an example system output.


<br/>

## <img src="https://em-content.zobj.net/thumbs/120/facebook/355/page-facing-up_1f4c4.png" width="30" /> 4. Report Writing

| :exclamation:  You MUST use the [ACL template](https://github.com/acl-org/acl-style-files) when writing your report. 
|----------------------------------------------------------------------------------------------------------------------|

You must use LATEX for writing your report. You must include your group number under the title (using the \author field
in LATEX).  
Make sure to change the template setting from `\usepackage[review]{acl}` to `\usepackage[final]{acl}` to generate the
final version.
We will not accept reports that are longer than the stated limits below, or otherwise violate the style requirements.

The report should be submitted as a PDF and contain **no more than seven(7)** pages of content, excluding team
contribution and references. An appendix is NOT allowed. Therefore, you should carefully consider the information you
want to include in the report to build a coherent and concise narrative.


-----
Below is a suggested report structure:

**Title** The title of your project and Group Name

**Abstract**. An abstract should concisely (less than 300 words) motivate the problem, describe your aims, describe your
contribution, and highlight your main finding(s).

**Introduction** The introduction explains the problem, why it’s difficult, interesting, or important, how and why
current methods succeed/fail at the problem, and explains the key ideas of your approach and results. Though an
introduction covers similar material as an abstract, the introduction gives more space for motivation, detail, and
references to existing work and captures the reader’s interest.

**Approach** This section details your approach(es) to the problem. For example, this is where you describe the
architecture of your neural network(s), and any other key methods or algorithms.

* You should be specific when describing your main approaches – you probably want to include equations and figures.
* You should also describe your baseline(s). Depending on space constraints, and how standard your baseline is, you
  might do this in detail, or simply refer the reader to some other paper for the details.
* If any part of your approach is original, make it clear (so we can give you credit!). For models and techniques that
  aren’t yours, provide references.
* As you’re setting up equations, notation, and the like, be sure to agree on a fixed technical vocabulary (that you’ve
  defined, or is well-defined in the literature) before writing and use it consistently throughout the report! This will
  make it easier for the teaching team to follow and is nice practice for research writing in general.

**Experiments**. This section contains the following.

* **Evaluation method**: If you’re defining your own metrics (for diagnostic purposes), be clear as to what you’re
  hoping to measure with each evaluation method (whether quantitative or qualitative, automatic or human-defined!), and
  how it’s defined.
* **Experimental details**: Report how you ran your experiments (e.g., model configurations, learning rate, training
  time, etc.)

**Results**: Report the quantitative results that you have found so far. Use a table or plot to compare results and
compare against baselines. You must report dev results, and also test results if you participate in the leaderboard.
When analysing your results, consider the following: Are they what you expected?; Better than you expected?; Is It worse
than you expected?; Why do you think that is?; What does that tell you about your approach?

**Conclusion**. Summarise the main findings of your project, and what you have learnt. Highlight your achievements, and
note the primary limitations of your work. If you like, you can describe avenues for future work.

**Team contributions** (doesn't count towards the page limit) If you are a multi-person team, briefly describe the
contributions of each member of the team.

**References** (doesn't count towards the page limit) Your references section should be produced using BibTeX.

**Important:** The report must be a faithful description of the system you implemented.  
If there is a mismatch between the report and the submitted code, marks will be deducted.

A strong report not only presents results, but also provides clear insight into *why* the method works (or fails).

<br/>

## <img src="https://em-content.zobj.net/thumbs/120/whatsapp/326/envelope-with-arrow_1f4e9.png" width="30" /> 5. Project Submission Method and Grading

**Submission:** LMS Assignment Submission Box will be opened from 28 April 2026.

**You Must Submit Two Files:**

- **pdf file** (filename format: COMP90042_teamname.pdf): a report using
  the [ACL template](https://github.com/acl-org/acl-style-files).

- **zip file** (filename format: COMP90042_teamname_resource.zip): A zip file containing:
    1) ipynb file(s) (**You MUST use the
       provided [ipynb template](https://colab.research.google.com/drive/1CjlVXdEsioH_iGOHUbmrhimTLRXGJIt0?usp=sharing)
       **)
    2) a README file describing how to run the code (if it's not apparent from the documentation in the ipynb files) *(
       optional)*
    3) any shell scripts used to run your code (e.g., for package installation) *(optional)*

  **Note:** You **MUST NOT** upload any data files or trained model checkpoints. Your code must run successfully in the
  Colab environment and reproduce the reported results.

**Your submissions will be graded as follows:**
| Component | Criteria | Description | Marks |
| ------------- | ------------- | ------------- | ------------- |
| Writing | Clarity | Is the report well-written and well-structured? | 3 |
| Writing | Tables/Figures | Are tables and figures interpretable and used effectively? | 4 |
| Content | Soundness | Are the methods technically sound and well-justified? | 8 |
| Content | Substance | How much work is done? Is there sufficient depth and effort? | 5 |
| Content | Novelty | How novel or ambitious are the techniques or methods? How original and well-justified are the
design choices? | 6 |
| Content | Results | Are the results and findings convincing? Are the results critically analysed and interpreted? |
6 |
| Scholarship | Citation | Does the report cite relevant publications to drive decision making (e.g. to motivate design
choices or to support findings)? | 3 |
| **Total**  | | | **35**  |

**Note:** For projects using LLMs, marks will be awarded based on the originality of system design and the clarity of
justification, rather than simply using a pretrained model.
Submissions that rely solely on off-the-shelf models without clear system design and analysis will be considered weak.

**Leaderboard**

Participation in the leaderboard is optional and does not contribute to your final mark.

<br/>

## <img src="https://em-content.zobj.net/source/whatsapp/390/clipboard_1f4cb.png" width="30" /> 6. Peer Review

**Peer Review is worth a total of 8 marks.**

- This is a **separate assessment from Assignment 3 (35 marks** for project report and code submission).
- Please refer to the [subject handbook](https://handbook.unimelb.edu.au/subjects/comp90042/assessment) for full
  assessment details.

The peer review will take place **from 24 May to 29 May 2026**.

**The detailed procedure will be shared in Week 10.**

After the project submission, you will:

- **Review two reports** from other teams (Part 1)
- **Evaluate the reviews received for your report** (Part 2)

<br/>

## <img src="https://em-content.zobj.net/source/skype/289/trophy_1f3c6.png" width="30" /> 7. Leaderboard

Competition link: https://www.codabench.org/competitions/16039/?secret_key=99e31f4a-d0b5-4514-b473-77a46db9ffae

Participation in the leaderboard is **optional** and does **not** contribute to your final mark. It is provided as a
public benchmark so you can see how your system compares against other teams during the project window. We strongly
encourage you to take part — past students have found it a useful sanity check that their pipeline runs end-to-end on
unseen data, and it makes the project a lot more fun.

The leaderboard is hosted on **Codabench**. Each project team should create **one Codabench account** under a
recognisable team name (the same name you use for your LMS submission) and use that single account for all of the team's
submissions. Sharing accounts across teams or registering multiple accounts to evade submission caps is treated as an
integrity breach under the project rules in §2.

The competition runs in two phases that together span **1 May 2026 → 22 May 2026 (23:59 AEST)**:

- **Phase 1 — Ongoing Evaluation (1 May → 18 May 2026 AEST).** Public leaderboard with continuous feedback. You may
  submit up to **5 times per day**, with a total cap of **100** submissions across the phase. Scores are computed on a
  held-out subset of the test set and serve as ongoing diagnostics for your system.
- **Phase 2 — Final Evaluation (19 May → 22 May 2026 AEST).** Private leaderboard. You may submit up to **5 times per
  day**, with a total cap of **3** submissions. Scores are computed on the remaining (previously hidden) portion of the
  test set and determine your team's final position on the leaderboard.

To submit, package your prediction file as a `.zip` archive containing **one file named `test-output.json`** with
predictions for the **entire** unlabelled test set (`test-claims-unlabelled.json`). The format is identical to
`dev-claims-baseline.json`: a JSON dictionary mapping each claim ID to an object with `claim_label` (one of `SUPPORTS`,
`REFUTES`, `NOT_ENOUGH_INFO`, `DISPUTED`) and `evidences` (a non-empty list of `evidence-XXXXX` IDs); `claim_text` is
optional and ignored by the scorer. Upload the zip via the **My Submissions** tab on the Codabench competition page,
choosing the appropriate phase. Each submission is scored automatically; once it has finished, you can choose which
completed runs to publish to the public leaderboard.

The leaderboard ranks teams by the **Harmonic Mean of F and A (H_FA)**, with `F` (evidence retrieval F-score) and `A` (
claim-classification accuracy) shown alongside as diagnostic columns. See §3 for the precise metric definitions.

A final reminder on integrity: the same project rules from §2 apply to your leaderboard submissions. In particular, you
must not manually inspect the test set, you must not use closed-source LLM APIs, and you must not hand-edit predictions
after they are produced by your system. The submission you place on the leaderboard must be a faithful output of the
code you submit on the LMS — any mismatch between the two will be treated as a project-rule breach.

Participation in the leaderboard is optional and does not contribute to your final mark.
The Leaderboard link and Submission instructions were announced on 1 May.

<br/>

## <img src="https://em-content.zobj.net/thumbs/120/google/350/person-raising-hand_1f64b.png" width="30" /> 8. FAQ

The FAQ will be updated based on students’ questions.

To support your learning, we will summarise common questions and provide answers on the discussion board.

**FAQ Discussion Board Summary:** The link will be shared 2–3 days after the assignment specification release.
