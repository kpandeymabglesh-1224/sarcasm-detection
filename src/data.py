"""Dataset loading and the shared train/test split.

Both pipelines (Stacked ML and DistilBERT) call get_split(), so they are
evaluated on exactly the same test headlines.
"""
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_FILE, SEED, SPLIT_FILE, TEST_SIZE


def load_dataset(dedupe: bool = True) -> pd.DataFrame:
    """Return a DataFrame with columns: text, label.

    article_link is dropped on purpose: its domain (theonion / huffpost)
    gives the label away.
    dedupe=True removes the 116 repeated headlines (none have conflicting
    labels) so no headline can appear in both train and test.
    """
    df = pd.read_json(DATA_FILE, lines=True)
    df = df[["headline", "is_sarcastic"]].rename(
        columns={"headline": "text", "is_sarcastic": "label"}
    )
    if dedupe:
        df = df.drop_duplicates(subset="text").reset_index(drop=True)
    return df


def get_split(df: pd.DataFrame):
    """Stratified 80/20 split, created once and saved to data/split.json.

    Later runs (and the DistilBERT pipeline) reload the same indices.
    """
    if SPLIT_FILE.exists():
        saved = json.loads(SPLIT_FILE.read_text())
        if saved["n_rows"] != len(df):
            raise ValueError(
                f"split.json was made for {saved['n_rows']} rows but the "
                f"dataset has {len(df)}. Delete data/split.json to rebuild it."
            )
        train_idx, test_idx = saved["train"], saved["test"]
    else:
        train_idx, test_idx = train_test_split(
            list(range(len(df))),
            test_size=TEST_SIZE,
            random_state=SEED,
            stratify=df["label"],
        )
        SPLIT_FILE.write_text(json.dumps(
            {"n_rows": len(df), "seed": SEED, "train": train_idx, "test": test_idx}
        ))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
