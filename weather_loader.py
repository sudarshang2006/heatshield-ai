"""
weather_loader.py
Fetches LIVE meteorological data (temperature, humidity, wind speed,
pressure, weather condition) from OpenWeatherMap free API.

This covers the "Meteorological Data Integration" requirement
(equivalent to ERA5 reanalysis data, but live and free).

SETUP REQUIRED:
    Get a free API key from https://openweathermap.org/api
    Paste it into OPENWEATHER_API_KEY below.
"""

import requests

# ⚠️ PASTE YOUR OPENWEATHERMAP API KEY HERE (between the quotes)
    api_key = "YOUR_API_KEY_HERE"

# City coordinates (same as data_loader.py, kept here too so this
# module works independently)
CITY_COORDS = {
    "Delhi": {"lat": 28.6139, "lon": 77.2090},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777},
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714},
    "Bangalore": {"lat": 12.9716, "lon": 77.5946},
    "Chennai": {"lat": 13.0827, "lon": 80.2707},
}


def get_live_weather(city_name):
    """
    Fetch current live weather for a city from OpenWeatherMap.

    Returns a dict like:
    {
        "temperature": 34.2,      # Celsius
        "feels_like": 37.8,       # Celsius
        "humidity": 65,           # percent
        "wind_speed": 3.4,        # m/s
        "pressure": 1008,         # hPa
        "condition": "Haze",      # short description
        "icon": "50d",            # OpenWeatherMap icon code
        "success": True
    }
    If the API call fails (bad key, no internet, etc.), returns
    {"success": False, "error": "<message>"} instead, so the app
    can show a friendly fallback message rather than crashing.
    """
    if city_name not in CITY_COORDS:
        return {"success": False, "error": f"Unknown city: {city_name}"}

    if not OPENWEATHER_API_KEY or OPENWEATHER_API_KEY == "PASTE_YOUR_API_KEY_HERE":
        return {"success": False, "error": "API key not set yet"}

    coords = CITY_COORDS[city_name]
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": coords["lat"],
        "lon": coords["lon"],
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",  # Celsius, m/s
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        return {
            "temperature": round(data["main"]["temp"], 1),
            "feels_like": round(data["main"]["feels_like"], 1),
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"],
            "pressure": data["main"]["pressure"],
            "condition": data["weather"][0]["description"].title(),
            "icon": data["weather"][0]["icon"],
            "success": True,
        }
    except requests.exceptions.HTTPError as e:
        if response.status_code == 401:
            return {"success": False, "error": "Invalid API key (may still be activating - wait up to 2 hours after signup)"}
        return {"success": False, "error": f"HTTP error: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_weather_icon_url(icon_code):
    """Returns the URL for OpenWeatherMap's weather icon image."""
    return f"https://openweathermap.org/img/wn/{icon_code}@2x.png"
