"""
Runs Chaturvedi & Chaturvedi's multiclass religion classifier (Hindu/Muslim/
Sikh/Christian/Jain/Buddhist) on every candidate in clean_names.csv, as an
independent cross-check on Phase 3's hand-curated marker lists.

MUST run under a Python 3.8 + scikit-learn==0.22.2.post1 environment
(the .venv-chaturvedi venv), NOT the project's main venv -- the model was
pickled with sklearn 0.22.2.post1 in 2023, and unpickling it under modern
sklearn (tested: 1.7.2) silently produces garbage: it doesn't crash, but
the TF-IDF feature computation breaks internally and the classifier
degenerates to predicting the majority class ("Hindu") for almost
everything -- confirmed by testing khan/ansari/singh/kaur/jain, all of
which came back "Hindu" under the version-mismatched load. Under the
matching old environment those same names correctly come back
Muslim/Muslim/Hindu(ambiguous)/Sikh/Jain.

Setup (one-time, see conversation history / README for the full deadsnakes
PPA steps):
    python3.8 -m venv .venv-chaturvedi
    source .venv-chaturvedi/bin/activate
    pip install numpy==1.19.5 scipy==1.5.4 scikit-learn==0.22.2.post1 pandas

Model files cached at ~/.chaturvedi/models/ (not in this repo -- see
CITATIONS.md for the source: Chaturvedi & Chaturvedi's official CC0
replication data, Harvard Dataverse DOI 10.7910/DVN/JOEVPN,
its_all_in_the_name.zip, 1.4GB, only distributed as one zip -- there's no
way to fetch just the models/ subfolder without the full download).

Uses the logistic-regression variant (not the SVM variant also present),
specifically because LogisticRegression exposes real predict_proba()
output -- LinearSVC's decision_function() is a margin, not a probability,
and the plan's Phase 3 step 2 wants an actual probability threshold
("Chaturvedi's max-class probability clears a threshold, start at 0.9").
concat_False (single name, no parent-name concatenation) matches our
single-name-at-a-time candidates.

Run:
    source .venv-chaturvedi-lr/bin/activate
    python src/chaturvedi_classify.py                       # scores clean_names.csv (default)
    python src/chaturvedi_classify.py names.txt out.csv      # scores an arbitrary newline-separated name list
"""
from __future__ import annotations

import os
import pickle
import re
import sys
import unicodedata

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
MODEL_DIR = os.path.expanduser("~/.chaturvedi/models/")


def clean_name(n: str) -> str:
    n = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode()
    n = n.upper()
    n = re.sub(r"[^A-Z ]", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def main() -> None:
    if not os.path.exists(MODEL_DIR):
        print(f"Model cache not found at {MODEL_DIR} -- see this script's docstring for setup.", file=sys.stderr)
        sys.exit(1)

    vectorizer = pickle.load(open(os.path.join(MODEL_DIR, "vectorizer_multiclass_lr_concat_False.sav"), "rb"))
    clf = pickle.load(open(os.path.join(MODEL_DIR, "model_multiclass_lr_concat_False.sav"), "rb"))
    _, idx2label = pickle.load(open(os.path.join(MODEL_DIR, "non_neural_label_encoding_multiclass.pkl"), "rb"))

    if len(sys.argv) >= 3:
        names_in, out_path = sys.argv[1], sys.argv[2]
        with open(names_in) as f:
            names = sorted({line.strip() for line in f if line.strip()})
        print(f"Classifying {len(names)} unique names from {names_in}...")
    else:
        clean_path = os.path.join(PROCESSED_DIR, "clean_names.csv")
        df = pd.read_csv(clean_path)
        names = df["name"].astype(str).unique()
        out_path = os.path.join(PROCESSED_DIR, "chaturvedi_predictions.csv")
        print(f"Classifying {len(names)} unique candidate names from clean_names.csv...")

    cleaned = [clean_name(n) for n in names]
    X = vectorizer.transform(cleaned)
    probs = clf.predict_proba(X)
    pred_idx = probs.argmax(axis=1)
    max_prob = probs.max(axis=1)

    out = pd.DataFrame({
        "name": names,
        "clf_predicted_religion": [idx2label[i] for i in pred_idx],
        "clf_max_prob": max_prob,
    })
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out)} rows)")
    print(out["clf_predicted_religion"].value_counts().to_string())


if __name__ == "__main__":
    main()
