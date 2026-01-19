import requests

url = "http://localhost:8080/predict"

player = {
  "league": "ENG-Premier League",
  "season": 2526,
  "team": "Liverpool",
  "player": "Trey Nyoni",
  "nation": "ENG",
  "pos": "FW",
  "age": 18,
  "born": 2007,
  "mp": 1,
  "starts": 0,
  "min": 1,
  "90s": 0.0,
  "gls": 0,
  "ast": 0,
  "g+a": 0,
  "g-pk": 0,
  "pk": 0,
  "pkatt": 0,
  "crdy": 0,
  "crdr": 0,
  "xg": 0.0,
  "npxg": 0.0,
  "xag": 0.0,
  "npxg+xag": 0.0,
  "prgc": 0,
  "prgp": 0,
  "prgr": 0,
  "gls_per90": 0.0,
  "ast_per90": 0.0,
  "g+a_per90": 0.0,
  "g-pk_per90": 0.0,
  "g+a-pk_per90": 0.0,
  "xg_per90": 0.0,
  "xag_per90": 0.0,
  "xg+xag_per90": 0.0,
  "npxg_per90": 0.0,
  "npxg+xag_per90": 0.0,
  "other positions": None,
  "contract expiration": "2026-06-30",
  "years_remaining": 0.5
}




response = requests.post(url, json=player)
price = response.json()

print('response:', price)

