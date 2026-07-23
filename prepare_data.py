"""Nettoie le dataset Kaggle brut et produit data/anime_clean.csv."""

from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/anime.csv")
CLEAN_PATH = Path("data/anime_clean.csv")

# Colonne brute -> colonne finale
COLUMNS = {
    "Name": "name",
    "English name": "name_en",
    "Genres": "genre",
    "Type": "type",
    "Score": "rating",
}


def main() -> None:
    if not RAW_PATH.exists():
        raise SystemExit(
            f"{RAW_PATH} introuvable. Télécharge le dataset Kaggle "
            "« Anime Recommendation Database 2020 » et place anime.csv dans data/."
        )

    df = pd.read_csv(RAW_PATH)

    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise SystemExit(f"Colonnes absentes du CSV brut : {sorted(missing)}")

    df = df[list(COLUMNS)].rename(columns=COLUMNS)
    before = len(df)

    # Dans ce dataset, les valeurs manquantes sont la CHAÎNE "Unknown",
    # pas NaN : fillna() seul ne les attrape pas.
    for col in ("name", "name_en", "genre", "type"):
        df[col] = df[col].astype(str).str.strip().replace({"Unknown": "", "nan": ""})

    # "Unknown" -> NaN, la colonne devient numérique (sinon float() plante côté app).
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # Un titre anglais différent aide la recherche ("Attack on Titan").
    df["name_en"] = df.apply(
        lambda r: "" if r["name_en"].casefold() == r["name"].casefold() else r["name_en"],
        axis=1,
    )

    df["type"] = df["type"].replace({"": "Unknown"})
    df = df[(df["name"] != "") & (df["genre"] != "")]
    df = df.drop_duplicates(subset="name").reset_index(drop=True)

    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)

    print(f"{CLEAN_PATH} généré : {len(df)} lignes conservées sur {before}.")
    print(f"  - notes manquantes : {df['rating'].isna().sum()}")
    print(f"  - types            : {sorted(df['type'].unique())}")


if __name__ == "__main__":
    main()
