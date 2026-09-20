"""Stage 1+2 check: verifies libraries, GPU, NLTK data, and the dataset file.

Run from the project root:  python check_env.py
"""
import importlib
import sys
from pathlib import Path

# GitHub ships the 28,619-row file as "Sarcasm_Headlines_Dataset.json";
# Kaggle ships the same data as "..._v2.json". Either name is accepted.
CANDIDATES = [Path("data") / "Sarcasm_Headlines_Dataset_v2.json",
              Path("data") / "Sarcasm_Headlines_Dataset.json"]
DATA_PATH = next((p for p in CANDIDATES if p.exists()), CANDIDATES[0])

print(f"Python {sys.version.split()[0]}  ({sys.executable})\n")

# ---------- 1. Libraries ----------
print("== Libraries ==")
libs = ["numpy", "pandas", "scipy", "sklearn", "nltk", "joblib",
        "torch", "transformers", "accelerate"]
missing = []
for name in libs:
    try:
        mod = importlib.import_module(name)
        print(f"  OK       {name:<13} {getattr(mod, '__version__', '?')}")
    except ImportError:
        print(f"  MISSING  {name}")
        missing.append(name)

# ---------- 2. GPU ----------
print("\n== GPU ==")
try:
    import torch
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  CUDA OK: {props.name}, {props.total_memory / 1024**3:.1f} GB, "
              f"torch CUDA {torch.version.cuda}")
    else:
        print("  CUDA NOT available -> DistilBERT would train on CPU (very slow)")
except ImportError:
    print("  torch not installed")

# ---------- 3. NLTK data ----------
print("\n== NLTK data ==")
try:
    import nltk
    for res, path in [("punkt", "tokenizers/punkt"),
                      ("punkt_tab", "tokenizers/punkt_tab"),
                      ("stopwords", "corpora/stopwords")]:
        try:
            nltk.data.find(path)
            print(f"  OK       {res}")
        except LookupError:
            print(f"  fetching {res} ...")
            nltk.download(res, quiet=True)
except ImportError:
    print("  nltk not installed")

# ---------- 4. Dataset ----------
print("\n== Dataset ==")
if not DATA_PATH.exists():
    print(f"  NOT FOUND: {DATA_PATH.resolve()}")
else:
    import pandas as pd
    df = pd.read_json(DATA_PATH, lines=True)
    print(f"  file: {DATA_PATH.name}")
    print(f"  rows: {len(df)}   (expected 28619)")
    print(f"  columns: {list(df.columns)}")
    print(f"  label counts:\n{df['is_sarcastic'].value_counts().to_string()}")
    h = df["headline"]
    print(f"  duplicate headlines: {h.duplicated().sum()}")
    print(f"  headlines with any uppercase letter: {h.str.contains(r'[A-Z]').mean():.2%}")
    print(f"  headlines containing '!': {h.str.contains('!', regex=False).mean():.2%}")
    print(f"  headlines containing '?': {h.str.contains('?', regex=False).mean():.2%}")
    lens = h.str.split().str.len()
    print(f"  words per headline: mean {lens.mean():.1f}, 95th pct {lens.quantile(.95):.0f}, max {lens.max()}")
    print("  sample:")
    for _, r in df.sample(3, random_state=0).iterrows():
        print(f"    [{r['is_sarcastic']}] {r['headline']}")

print("\nMissing libraries:", missing if missing else "none")
