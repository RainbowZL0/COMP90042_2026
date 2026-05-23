# Running the Submitted Code

## Environment

The code was tested with Python 3.12. The main package versions in the working environment were:

```text
accelerate==1.13.0
bm25s==0.3.9
ipykernel==7.2.0
jupytext==1.19.3
mlflow==3.12.0
nltk==3.9.4
numpy==2.4.5
pandas==2.3.3
scikit-learn==1.8.0
scipy==1.17.1
sentence-transformers==5.5.0
sentencepiece==0.2.1
torch==2.12.0+cu130
tqdm==4.67.3
transformers==5.8.1
```

If starting from a clean Python environment, install the required packages with `pip`:

```bash
pip install accelerate==1.13.0 bm25s==0.3.9 ipykernel==7.2.0 jupytext==1.19.3 mlflow==3.12.0 nltk==3.9.4 numpy==2.4.5 pandas==2.3.3 scikit-learn==1.8.0 scipy==1.17.1 sentence-transformers==5.5.0 sentencepiece==0.2.1 tqdm==4.67.3 transformers==5.8.1
```

## Running

Run all jupyter cells from top to bottom. The code creates its working outputs automatically and prints the development evaluation results during execution.

## Expected Development Scores

The final reported development scores are approximately:

```text
Evidence F-score: 0.2448
Label accuracy:  0.4805
Harmonic mean:   0.3243
```
