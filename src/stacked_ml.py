"""Pipeline A: Stacked ML ensemble (rebuilt from the surviving Colab code).

    python -m src.stacked_ml              # corrected run (use this for the comparison)
    python -m src.stacked_ml --original   # exact replica of the Colab code, to check the 80.3%

Methodology (unchanged from the Colab code / slide 4):
  preprocess -> TF-IDF (1-2 grams, 5000 features) + counts of '!', '?', CAPS
  -> SVM + Logistic Regression + Multinomial NB -> StackingClassifier (LR meta-model, cv=3)

Differences in the corrected run, and why:
  1. TF-IDF is fitted on the training set only. The Colab code fitted it on all
     headlines before splitting, so test-set vocabulary leaked into training.
  2. The 116 duplicate headlines are removed, so none sit in both train and test.
  3. The split is stratified and shared with the DistilBERT pipeline
     (data/split.json), so both models are scored on the same test headlines.
"""
import argparse
import re
import time

import joblib
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from src.config import MODELS_DIR, SEED, TEST_SIZE
from src.data import get_split, load_dataset
from src.metrics import evaluate

STOP_WORDS = set(stopwords.words("english"))


# ---------- Preprocessing & features (identical to the Colab code) ----------
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    return " ".join(w for w in tokens if w not in STOP_WORDS)


def handcrafted_features(texts) -> np.ndarray:
    """Counts of '!', '?', and uppercase letters, computed on the RAW text.

    Note: every headline in this dataset is lowercase, so the CAPS column is
    always 0 during training.
    """
    return np.array([[t.count("!"), t.count("?"), sum(c.isupper() for c in t)]
                     for t in texts])


def build_features(vectorizer, raw_texts, clean_texts):
    return hstack([vectorizer.transform(clean_texts),
                   csr_matrix(handcrafted_features(raw_texts))]).tocsr()


def build_model() -> StackingClassifier:
    return StackingClassifier(
        estimators=[
            ("svm", SVC(probability=True)),
            ("lr", LogisticRegression()),
            ("nb", MultinomialNB()),
        ],
        final_estimator=LogisticRegression(),
        cv=3,
    )


# ---------- Runs ----------
def run_original():
    """Exact replica of the Colab logic (leakage and all)."""
    df = load_dataset(dedupe=False)
    df["clean_text"] = df["text"].apply(preprocess)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X_text = vectorizer.fit_transform(df["clean_text"])          # fitted on ALL rows
    X = hstack([X_text, handcrafted_features(df["text"])])
    X_train, X_test, y_train, y_test = train_test_split(
        X, df["label"], test_size=TEST_SIZE, random_state=SEED)
    return vectorizer, X_train, X_test, y_train, y_test, "stacked_ml_original"


def run_corrected():
    df = load_dataset(dedupe=True)
    train, test = get_split(df)
    train_clean = train["text"].apply(preprocess)
    test_clean = test["text"].apply(preprocess)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    vectorizer.fit(train_clean)                                   # fitted on TRAIN only
    X_train = build_features(vectorizer, train["text"], train_clean)
    X_test = build_features(vectorizer, test["text"], test_clean)
    return vectorizer, X_train, X_test, train["label"], test["label"], "stacked_ml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", action="store_true",
                    help="replicate the Colab code exactly (to check the reported 80.3%%)")
    args = ap.parse_args()

    vectorizer, X_train, X_test, y_train, y_test, name = (
        run_original() if args.original else run_corrected())
    print(f"[{name}] train {X_train.shape}, test {X_test.shape}")

    model = build_model()
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"Training time: {train_time:.1f} s")

    evaluate(y_test, model.predict(X_test), name,
             extra={"train_time_s": round(train_time, 1)})

    out = MODELS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "stacking_model.joblib")
    joblib.dump(vectorizer, out / "tfidf_vectorizer.joblib")
    print(f"Saved model + vectorizer -> models/{name}/")

    # Same two sample sentences as the Colab notebook
    samples = ["Oh great, another Monday morning!",
               "The government passed a new law today."]
    clean = [preprocess(s) for s in samples]
    preds = model.predict(build_features(vectorizer, samples, clean))
    for s, p in zip(samples, preds):
        print(f"  [{'sarcastic' if p else 'not sarcastic'}] {s}")


if __name__ == "__main__":
    main()
