from recommender import AnimeRecommender

reco = AnimeRecommender("anime.csv")

titre = input("Entrez un anime que vous aimez : ")
recommandations = reco.recommend(titre)

print("\nAnimes recommandés :")
if isinstance(recommandations, str):
    print(recommandations)
else:
    for anime in recommandations:
        print("→", anime)
