import requests
import json

resp = requests.get("https://itunes.apple.com/search",params={"term":"beatles", "entity":"song"})
print(resp.text[:1500])