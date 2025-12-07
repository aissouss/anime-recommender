# prepare_data.py
import pandas as pd

df = pd.read_csv("anime.csv")

# On garde les colonnes utiles
cols_to_keep = ["name", "genre", "type", "rating"]
df = df[cols_to_keep].copy()

# Nettoyage basique
df["name"] = df["name"].astype(str).str.strip()
df["genre"] = df["genre"].fillna("").astype(str)
df["type"] = df["type"].fillna("Unknown").astype(str)
df["rating"] = df["rating"].fillna(0)

# On enlève les lignes sans nom ou genre vide
df = df[df["name"] != ""]
df = df[df["genre"] != ""]

df.to_csv("data/anime_clean.csv", index=False)
print("✅ Fichier sauvegardé dans data/anime_clean.csv")
