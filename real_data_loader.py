"""
real_data_loader.py
Fetches REAL satellite thermal data from Google Earth Engine (Landsat 8/9).

SETUP REQUIRED (one-time, on your PC terminal):
    pip install earthengine-api geemap
    earthengine authenticate

This replaces the simulated data_loader.py with actual satellite imagery.
"""

import ee
import numpy as np
import pandas as pd

# Same city coordinates as before
CITIES = {
    "Delhi": {"lat": 28.6139, "lon": 77.2090, "base_temp": 42},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777, "base_temp": 36},
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714, "base_temp": 44},
    "Bangalore": {"lat": 12.9716, "lon": 77.5946, "base_temp": 32},
    "Chennai": {"lat": 13.0827, "lon": 80.2707, "base_temp": 38},
}

_INITIALIZED = False


def initialize_earth_engine(project_id="heatshield-ai-499810"):
    """
    Initialize the Earth Engine connection.
    project_id: your GEE Cloud project ID (find it at code.earthengine.google.com)
    Call this ONCE at the start of your app.
    """
    global _INITIALIZED
    if not _INITIALIZED:
        try:
            ee.Initialize(project=project_id)
        except Exception:
            ee.Authenticate()
            ee.Initialize(project=project_id)
        _INITIALIZED = True


def get_landsat_lst(city_name, start_date="2024-04-01", end_date="2024-05-31"):
    """
    Fetch real Land Surface Temperature (LST) from Landsat 8/9
    for a given city during a date range (summer months recommended).

    Returns an Earth Engine Image with LST band in Celsius.
    """
    city = CITIES[city_name]
    point = ee.Geometry.Point([city["lon"], city["lat"]])
    region = point.buffer(15000)  # 15 km radius around city center

    # Landsat 8/9 Collection 2 Level 2 (already calibrated, includes ST_B10 = surface temp)
    collection = (
        ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUD_COVER", 20))
        .sort("CLOUD_COVER")
    )

    image = collection.first()

    # ST_B10 is scaled: Celsius = (DN * 0.00341802 + 149.0) - 273.15
    lst_celsius = (
        image.select("ST_B10")
        .multiply(0.00341802)
        .add(149.0)
        .subtract(273.15)
        .rename("LST")
    )

    # NDVI from SR bands (Red = SR_B4, NIR = SR_B5), scaled by 0.0000275 - 0.2
    sr_red = image.select("SR_B4").multiply(0.0000275).add(-0.2)
    sr_nir = image.select("SR_B5").multiply(0.0000275).add(-0.2)
    ndvi = sr_nir.subtract(sr_red).divide(sr_nir.add(sr_red)).rename("NDVI")

    combined = lst_celsius.addBands(ndvi)
    return combined, region


def extract_grid_data(city_name, grid_size=20, start_date="2024-04-01", end_date="2024-05-31"):
    """
    Sample the Landsat LST + NDVI image on a grid of points
    and return a DataFrame in the SAME format as the old simulated data_loader.py
    so the rest of the app (ml_model.py, recommender.py, app.py) works unchanged.
    """
    city = CITIES[city_name]
    image, region = get_landsat_lst(city_name, start_date, end_date)

    # Build a grid of sample points across the region
    lats = np.linspace(city["lat"] - 0.12, city["lat"] + 0.12, grid_size)
    lons = np.linspace(city["lon"] - 0.12, city["lon"] + 0.12, grid_size)

    points = []
    for lat in lats:
        for lon in lons:
            points.append(ee.Feature(ee.Geometry.Point([lon, lat]), {"lat": lat, "lon": lon}))

    fc = ee.FeatureCollection(points)

    # Sample the image at each point (scale=30m matches Landsat resolution)
    sampled = image.sampleRegions(collection=fc, scale=30, geometries=True)

    # Pull data from Earth Engine to local Python (this is the network call)
    features = sampled.getInfo()["features"]

    rows = []
    for f in features:
        props = f["properties"]
        lst = props.get("LST")
        ndvi = props.get("NDVI")
        if lst is None or ndvi is None:
            continue  # skip pixels with missing data (e.g. clouds)

        # Classify surface type using NDVI + temperature heuristics
        # (Real classification would use a trained model on spectral bands;
        #  this is a practical approximation good enough for a hackathon demo)
        if ndvi > 0.4:
            surface = "Vegetation"
        elif ndvi < -0.05 and lst > city["base_temp"] + 5:
            surface = "Dark Rooftop"
        elif -0.05 <= ndvi < 0.1 and lst > city["base_temp"] + 3:
            surface = "Road/Pavement"
        elif ndvi < -0.1:
            surface = "Water Body"
        elif 0.1 <= ndvi <= 0.4:
            surface = "Bare Ground"
        else:
            surface = "Concrete Building"

        diff = lst - city["base_temp"]
        if diff >= 8:
            risk = "🔴 Critical"
        elif diff >= 4:
            risk = "🟠 High"
        elif diff >= 0:
            risk = "🟡 Moderate"
        else:
            risk = "🟢 Low"

        rows.append({
            "lat": props["lat"],
            "lon": props["lon"],
            "temperature": round(lst, 1),
            "surface_type": surface,
            "ndvi": round(ndvi, 3),
            "heat_risk": risk,
        })

    df = pd.DataFrame(rows)
    return df


def get_city_stats(df):
    """Same stats function as before - works on real or simulated data identically."""
    stats = {
        "avg_temp": df["temperature"].mean().round(1),
        "max_temp": df["temperature"].max().round(1),
        "min_temp": df["temperature"].min().round(1),
        "critical_zones": (df["heat_risk"] == "🔴 Critical").sum(),
        "high_zones": (df["heat_risk"] == "🟠 High").sum(),
        "total_zones": len(df),
        "vegetation_pct": ((df["surface_type"] == "Vegetation").sum() / len(df) * 100).round(1),
    }
    return stats
