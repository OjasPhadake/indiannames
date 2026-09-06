"""
Re-export data/processed/name_bank_expanded.csv (the flat, machine-built
corpus) into the same FIRST_NAME_BANK / REGION_SURNAME_MAP Python-dict shape
Ojas's own original lists used, so it's a drop-in replacement for his code.

Run after any edit to data/mappings/user_provided_names_raw.py + a re-run of
`python src/expand_corpus.py prepare` and `finalize` -- this script only
reads the finalized CSV, it does no selection logic of its own.

Run: python src/generate_dict_export.py
"""
from __future__ import annotations

import os

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
IN_PATH = os.path.join(PROCESSED_DIR, "name_bank_expanded.csv")
OUT_PATH = os.path.join(PROCESSED_DIR, "name_bank_expanded_dict_format.py")

RELIGION_ORDER = ["Hindu", "Muslim", "Christian", "Sikh"]
REGION_ORDER = ["North", "East", "South", "West"]


def sorted_names(sub: pd.DataFrame) -> list[str]:
    return sub.sort_values("n", ascending=False)["name_title"].tolist()


def format_entry(key: str, names: list[str]) -> str:
    body_lines = []
    for i in range(0, len(names), 5):
        chunk = names[i:i + 5]
        body_lines.append("        " + ", ".join(f'"{n}"' for n in chunk) + ",")
    body = "\n".join(body_lines)
    return f'    "{key}": [  # n={len(names)}\n{body}\n    ],'


def main() -> None:
    df = pd.read_csv(IN_PATH)
    df["name_title"] = df["name"].str.title()

    print("FIRST_NAME_BANK entries:")
    fn_out = ["FIRST_NAME_BANK = {"]
    for religion in RELIGION_ORDER:
        for gender in ["Female", "Male"]:
            g = gender[0]
            sub = df[(df["religion"] == religion) & (df["type"] == "first") & (df["gender"] == g)]
            names = sorted_names(sub)
            key = f"{religion}_{gender}"
            print(f"  {key}: {len(names)}")
            fn_out.append(format_entry(key, names))
    fn_out.append("}")
    fn_block = "\n".join(fn_out)

    print("\nREGION_SURNAME_MAP entries:")
    sn_out = ["REGION_SURNAME_MAP = {"]
    for religion in RELIGION_ORDER:
        if religion == "Sikh":
            sub = df[(df["religion"] == "Sikh") & (df["type"] == "last")]
            names = sorted_names(sub)
            print(f"  Sikh: {len(names)}")
            sn_out.append(format_entry("Sikh", names))
            continue
        for region in REGION_ORDER:
            sub = df[(df["religion"] == religion) & (df["type"] == "last") & (df["region"] == region)]
            names = sorted_names(sub)
            key = f"{religion}_{region}"
            print(f"  {key}: {len(names)}")
            sn_out.append(format_entry(key, names))
    sn_out.append("}")
    sn_block = "\n".join(sn_out)

    with open(OUT_PATH, "w") as f:
        f.write(fn_block + "\n\n" + sn_block + "\n")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
