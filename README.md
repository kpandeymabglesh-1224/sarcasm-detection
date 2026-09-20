# Sarcasm Detection: Stacked ML Ensemble vs DistilBERT

Data Mining project comparing a traditional stacked ML ensemble with a fine-tuned
DistilBERT transformer on the News Headlines Sarcasm Dataset.

## Results (same 5,701 test headlines)

| Model | Accuracy | Precision | Recall | F1 | Train time |
|---|---|---|---|---|---|
| Stacked ML (corrected) | 78.99% | 0.782 | 0.774 | 0.778 | 543 s (CPU) |
| DistilBERT (fine-tuned) | **92.55%** | 0.914 | 0.931 | 0.922 | 169 s (RTX 3050) |

Precision / recall / F1 are for the sarcastic class, as in the presentation.
Presentation (slide 5) reported 80.3% and 92.7%.

- **Stacked ML:** running the original Colab logic exactly (`--original`) gives
  80.33%, reproducing the reported figure. The corrected pipeline (below) scores
  78.99%.
- **DistilBERT:** 92.55% vs 92.7% reported. No original DistilBERT code survived,
  so its settings are a reconstruction using standard fine-tuning values.

## Dataset

Misra, *News Headlines Dataset for Sarcasm Detection*: 28,619 headlines from
The Onion (sarcastic) and HuffPost (not sarcastic), 116 duplicates removed → 28,503.
Download into `data/`:

```powershell
Invoke-WebRequest "https://raw.githubusercontent.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection/master/Sarcasm_Headlines_Dataset.json" -OutFile "data\Sarcasm_Headlines_Dataset.json"
```

`article_link` is never used as a feature: its domain gives the label away.

## Setup

Tested with Anaconda base, Python 3.13.9, Windows, RTX 3050 (4 GB). See `requirements.txt`.
PyTorch must be the CUDA build:

```powershell
pip install --force-reinstall --no-deps torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126
pip install transformers accelerate
python check_env.py          # checks libraries, GPU, NLTK data and dataset
```

## Running

Run from the project folder, always as modules (`python -m ...`):

```powershell
python -m src.stacked_ml                  # Pipeline A, ~10 min on CPU
python -m src.stacked_ml --original       # optional: exact Colab replica (80.33%)
python -m src.distilbert_model --smoke    # optional: 1-minute check
python -m src.distilbert_model            # Pipeline B, ~3 min on GPU
python -m src.compare                     # table -> results/comparison.md
```

Trained models are not in the repository (DistilBERT alone is ~260 MB); the
commands above recreate them in `models/`. `data/split.json` is included so the
test set is exactly the same.

## Project structure

```
data/            dataset + split.json (shared train/test indices)
original/        the surviving Colab notebook export, unchanged
src/config.py    paths, seed (42), test size (0.2)
src/data.py      loading, de-duplication, shared stratified 80/20 split
src/metrics.py   evaluation used by both pipelines -> results/<name>.json
src/stacked_ml.py        Pipeline A
src/distilbert_model.py  Pipeline B
src/compare.py           comparison table
models/          saved models (stacked_ml/*.joblib, distilbert/final/)
results/         metrics JSON + comparison.md
```

## Pipelines

**A: Stacked ML** (from the surviving Colab code): lowercase, remove URLs,
punctuation and stopwords → TF-IDF (unigrams + bigrams, 5,000 features) + counts of
`!`, `?` and capital letters → SVM, Logistic Regression and Multinomial Naive Bayes →
StackingClassifier with a Logistic Regression meta-model (3-fold).

Changes from the Colab code: TF-IDF is fitted on the training set only (the Colab
code fitted it on all data before splitting, leaking test vocabulary); duplicates
are removed; the split is stratified and shared with DistilBERT.

**B: DistilBERT:** `distilbert-base-uncased` on the raw headline (no stopword or
punctuation removal), max 64 tokens, 3 epochs, learning rate 2e-5, batch 32,
weight decay 0.01, 10% warm-up, fp16. 10% of the training set is used for
validation to keep the best epoch (epoch 2 in our run); the test set is used once.

## Findings and limitations

- DistilBERT makes 425 errors vs 1,198 for Stacked ML: **65% fewer mistakes**.
- Every headline in the dataset is lowercase, so the CAPS feature is always 0 in
  training and contributes nothing. `!` (0.69% of headlines) and `?` (2.89%) are weak signals.
- Both models label "Oh great, another Monday morning!" as *not* sarcastic. The
  dataset's sarcasm is The Onion's deadpan news style, so the models learn that
  style rather than conversational irony, which motivates testing on Twitter/Reddit data.
- Results come from a single run with seed 42; a different seed or split would move
  them slightly.
