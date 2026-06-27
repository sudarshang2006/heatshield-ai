"""
weather_loader.py - Live weather data from OpenWeatherMap
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

CITY_COORDS = {
    "Delhi":     {"lat": 28.6139, "lon": 77.2090},
    "Mumbai":    {"lat": 19.0760, "lon": 72.8777},
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714},
    "Bangalore": {"lat": 12.9716, "lon": 77.5946},
    "Chennai":   {"lat": 13.0827, "lon": 80.2707},
}


def get_live_weather(city_name):
    if city_name not in CITY_COORDS:
        return {"success": False, "error": f"Unknown city: {city_name}"}

    if not OPENWEATHER_API_KEY:
        return {"success": False, "error": "API key not set in .env file"}

    coords = CITY_COORDS[city_name]
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": coords["lat"],
        "lon": coords["lon"],
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return {
            "temperature": round(data["main"]["temp"], 1),
            "feels_like":  round(data["main"]["feels_like"], 1),
            "humidity":    data["main"]["humidity"],
            "wind_speed":  data["wind"]["speed"],
            "pressure":    data["main"]["pressure"],
            "condition":   data["weather"][0]["description"].title(),
            "icon":        data["weather"][0]["icon"],
            "success": True,
        }
    except requests.exceptions.HTTPError as e:
        if response.status_code == 401:
            return {"success": False, "error": "Invalid API key"}
        return {"success": False, "error": f"HTTP error: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_weather_icon_url(icon_code):
    return f"https://openweathermap.org/img/wn/{icon_code}@2x.png"