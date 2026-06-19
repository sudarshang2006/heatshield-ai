import numpy as np
import pandas as pd

# Indian cities with their lat/lon centers
CITIES = {
    "Delhi": {"lat": 28.6139, "lon": 77.2090, "base_temp": 42},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777, "base_temp": 36},
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714, "base_temp": 44},
    "Bangalore": {"lat": 12.9716, "lon": 77.5946, "base_temp": 32},
    "Chennai": {"lat": 13.0827, "lon": 80.2707, "base_temp": 38},
}

SURFACE_TYPES = ["Dark Rooftop", "Road/Pavement", "Bare Ground", "Vegetation", "Water Body", "Concrete Building"]

def generate_city_data(city_name, grid_size=20):
    """
    Simulate satellite thermal data for a city.
    Returns a DataFrame with pixel-level temperature and surface info.
    """
    city = CITIES[city_name]
    base_temp = city["base_temp"]
    np.random.seed(42)

    n = grid_size * grid_size
    lats = np.linspace(city["lat"] - 0.15, city["lat"] + 0.15, grid_size)
    lons = np.linspace(city["lon"] - 0.15, city["lon"] + 0.15, grid_size)

    lat_grid, lon_grid = np.meshgrid(lats, lons)

    # Simulate surface types with realistic distribution
    surface_weights = [0.20, 0.25, 0.10, 0.25, 0.05, 0.15]
    surfaces = np.random.choice(SURFACE_TYPES, size=n, p=surface_weights)

    # Temperature based on surface type
    temp_map = {
        "Dark Rooftop": base_temp + np.random.uniform(8, 14),
        "Road/Pavement": base_temp + np.random.uniform(6, 12),
        "Bare Ground": base_temp + np.random.uniform(4, 9),
        "Vegetation": base_temp - np.random.uniform(2, 6),
        "Water Body": base_temp - np.random.uniform(5, 10),
        "Concrete Building": base_temp + np.random.uniform(5, 10),
    }

    temperatures = np.array([
        temp_map[s] + np.random.normal(0, 1.5) for s in surfaces
    ])

    # NDVI: vegetation index (-1 to 1)
    ndvi_map = {
        "Dark Rooftop": -0.1,
        "Road/Pavement": -0.05,
        "Bare Ground": 0.1,
        "Vegetation": 0.65,
        "Water Body": 0.0,
        "Concrete Building": -0.15,
    }
    ndvi = np.array([ndvi_map[s] + np.random.normal(0, 0.05) for s in surfaces])

    # Heat risk category
    def heat_risk(t):
        if t >= base_temp + 8:
            return "🔴 Critical"
        elif t >= base_temp + 4:
            return "🟠 High"
        elif t >= base_temp:
            return "🟡 Moderate"
        else:
            return "🟢 Low"

    df = pd.DataFrame({
        "lat": lat_grid.flatten(),
        "lon": lon_grid.flatten(),
        "temperature": temperatures.round(1),
        "surface_type": surfaces,
        "ndvi": ndvi.round(3),
        "heat_risk": [heat_risk(t) for t in temperatures],
    })

    return df


def get_city_stats(df):
    """Returns summary statistics for the city."""
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
