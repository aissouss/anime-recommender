import requests

def search_anime(query):
    url = f"https://api.jikan.moe/v4/anime?q={query}"
    response = requests.get(url)
    if response.status_code != 200:
        return []
    data = response.json().get("data", [])
    return [anime["title"] for anime in data[:5]]

def search_game(query, api_key):
    url = f"https://api.rawg.io/api/games?search={query}&key={api_key}"
    response = requests.get(url)
    if response.status_code != 200:
        return []
    data = response.json().get("results", [])
    return [game["name"] for game in data[:5]]

