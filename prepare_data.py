"""Clean the raw Kaggle export and build data/anime_clean.csv.

Merges two files from the same dataset:
  - anime.csv               : one row per anime (title, genres, type, score)
  - anime_with_synopsis.csv : plot summaries, keyed on MAL_ID
"""

from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/anime.csv")
SYNOPSIS_PATH = Path("data/anime_with_synopsis.csv")
CLEAN_PATH = Path("data/anime_clean.csv")

COLUMNS = {
    "MAL_ID": "mal_id",
    "Name": "name",
    "English name": "name_en",
    "Genres": "genre",
    "Type": "type",
    "Score": "rating",
}

# MyAnimeList fills empty entries with this sentence. Left in place, those 745
# identical texts would make unrelated shows look highly similar to each other.
PLACEHOLDER = "No synopsis information has been added"

# Trailing credits: "(Source: ANN)", "[Written by MAL Rewrite]".
BOILERPLATE = r"\((?:Source|Written by)[^)]*\)|\[(?:Source|Written by)[^\]]*\]"


def load_synopses() -> pd.DataFrame | None:
    """Return a (mal_id, synopsis) frame, or None if the file is absent."""
    if not SYNOPSIS_PATH.exists():
        return None

    df = pd.read_csv(SYNOPSIS_PATH)
    # The column name is misspelled "sypnopsis" in the original dataset.
    column = "sypnopsis" if "sypnopsis" in df.columns else "synopsis"

    text = df[column].fillna("").astype(str)
    text = text.str.replace(BOILERPLATE, " ", regex=True)
    text = text.str.replace(r"\s+", " ", regex=True).str.strip()
    text = text.mask(text.str.startswith(PLACEHOLDER), "")

    return pd.DataFrame({"mal_id": df["MAL_ID"], "synopsis": text})


def main() -> None:
    if not RAW_PATH.exists():
        raise SystemExit(
            f"{RAW_PATH} not found. Download the Kaggle dataset "
            "'Anime Recommendation Database 2020' and put anime.csv in data/."
        )

    df = pd.read_csv(RAW_PATH)

    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise SystemExit(f"Columns missing from the raw CSV: {sorted(missing)}")

    df = df[list(COLUMNS)].rename(columns=COLUMNS)
    before = len(df)

    # In this dataset missing values are the STRING "Unknown", not NaN,
    # so fillna() alone never catches them.
    for col in ("name", "name_en", "genre", "type"):
        df[col] = df[col].astype(str).str.strip().replace({"Unknown": "", "nan": ""})

    # "Unknown" -> NaN so the column becomes numeric (otherwise float() crashes later).
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # A distinct English title helps search ("Attack on Titan").
    df["name_en"] = df.apply(
        lambda r: "" if r["name_en"].casefold() == r["name"].casefold() else r["name_en"],
        axis=1,
    )

    synopses = load_synopses()
    if synopses is not None:
        df = df.merge(synopses, on="mal_id", how="left")
        df["synopsis"] = df["synopsis"].fillna("")
    else:
        df["synopsis"] = ""
        print(f"WARNING: {SYNOPSIS_PATH} not found - the semantic model will only")
        print("         see titles and genres, which limits it severely.")

    df["type"] = df["type"].replace({"": "Unknown"})
    df = df[(df["name"] != "") & (df["genre"] != "")]
    df = df.drop_duplicates(subset="name").reset_index(drop=True)

    df = df[["name", "name_en", "genre", "type", "rating", "synopsis"]]

    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)

    with_synopsis = int((df["synopsis"].str.len() > 0).sum())
    print(f"{CLEAN_PATH} generated: {len(df)} rows kept out of {before}.")
    print(f"  - with a synopsis  : {with_synopsis} ({with_synopsis / len(df):.0%})")
    print(f"  - missing ratings  : {df['rating'].isna().sum()}")
    print(f"  - types            : {sorted(df['type'].unique())}")


if __name__ == "__main__":
    main()
