import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("your_actual_api_key")

print("API key found:", bool(API_KEY))

url = "https://newsapi.org/v2/top-headlines"

params = {
    "country": "us",
    "apiKey": API_KEY
}

response = requests.get(url, params=params)

print("Status code:", response.status_code)
print("Response:")
print(response.text[:1000])