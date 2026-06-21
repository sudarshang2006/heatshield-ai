import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

from data_loader import generate_city_data, get_city_stats, CITIES

# Try to import real satellite data loader (requires earthengine-api + authentication)
try:
    from real_data_loader import initialize_earth_engine, extract_grid_data
    REAL_DATA_AVAILABLE = True
except ImportError:
    REAL_DATA_AVAILABLE = False
from ml_model import HeatIslandModel
from recommender import get_recommendation, get_city_action_plan, calculate_total_impact, simulate_scenario
from weather_loader import get_live_weather, get_weather_icon_url
from urban_morphology_loader import get_urban_morphology
from driver_model import train_driver_model

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="HeatShield AI",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        color: #FF4B4B;
        text-align: center;
        padding: 1rem 0 0.2rem 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.3rem;
    }
    .critical-card {
        background: linear-gradient(135deg, #f93154 0%, #ff6b6b 100%);
        padding: 0.8rem;
        border-radius: 10px;
        color: white;
    }
    .success-card {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        padding: 0.8rem;
        border-radius: 10px;
        color: white;
    }
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="main-header">🌡️ HeatShield AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Urban Heat Island Detection & Cooling Strategy Advisor | Powered by Satellite Data + AI/ML</div>', unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/b/bd/Indian_Space_Research_Organisation_Logo.svg", width=80)
    st.markdown("### ⚙️ Controls")

    selected_city = st.selectbox("🏙️ Select City", list(CITIES.keys()), index=0)
    grid_size = st.slider("📡 Satellite Grid Resolution", 10, 30, 20,
                          help="Higher = more data points")
    show_map = st.checkbox("🗺️ Show Interactive Map", value=True)
    show_ml = st.checkbox("🤖 Show ML Analysis", value=True)
    show_plan = st.checkbox("📋 Show Action Plan", value=True)
    show_drivers = st.checkbox("🔬 Show Driver Analysis (LST Model)", value=True,
                                 help="Trains a physics-informed regression model across all 5 cities. Takes a few seconds.")

    st.markdown("---")
    st.markdown("### 🛰️ Data Source")
    if REAL_DATA_AVAILABLE:
        use_real_data = st.toggle("Use REAL Landsat Satellite Data", value=False,
                                    help="Fetches live data from Google Earth Engine. Requires internet + GEE authentication. Slower (~30-60s) but uses actual satellite imagery.")
    else:
        use_real_data = False
        st.caption("⚠️ Real data module not installed. Run: `pip install earthengine-api` and `earthengine authenticate`")

    st.markdown("---")
    st.markdown("### 📊 About")
    st.info("""
    **Data Source:** Simulated from Landsat 8/9 thermal bands & ISRO INSAT-3D patterns.

    **ML Model:** Random Forest Classifier trained on surface spectral signatures.

    **ISRO Relevance:** Supports Smart Cities Mission & National Action Plan on Climate Change.
    """)

# ─────────────────────────────────────────────
# LOAD DATA + TRAIN MODEL
# ─────────────────────────────────────────────
# LOAD DATA + TRAIN MODEL
# ─────────────────────────────────────────────
data_source_label = "🛰️ Simulated Data"

if use_real_data:
    try:
        with st.spinner(f"🛰️ Fetching REAL Landsat data for {selected_city} from Google Earth Engine... (this can take 30-60s)"):
            initialize_earth_engine()
            df = extract_grid_data(selected_city, grid_size)
            if df.empty:
                st.warning("⚠️ No clear-sky Landsat scenes found for this date range. Falling back to simulated data.")
                df = generate_city_data(selected_city, grid_size)
            else:
                data_source_label = "🛰️ REAL Landsat 8/9 Data"
    except Exception as e:
        st.error(f"⚠️ Could not fetch real satellite data ({e}). Falling back to simulated data.")
        df = generate_city_data(selected_city, grid_size)
else:
    df = generate_city_data(selected_city, grid_size)

with st.spinner("🤖 Training ML model..."):
    stats = get_city_stats(df)
    model = HeatIslandModel()
    accuracy = model.train(df)
    df["predicted_surface"] = model.predict_surface(df)

st.caption(f"Data source: **{data_source_label}**")
city_info = CITIES[selected_city]

# ─────────────────────────────────────────────
# LIVE METEOROLOGICAL DATA (OpenWeatherMap)
# ─────────────────────────────────────────────
st.markdown("### 🌤️ Live Meteorological Data")
weather = get_live_weather(selected_city)

if weather["success"]:
    wcol1, wcol2, wcol3, wcol4, wcol5 = st.columns([1, 1, 1, 1, 1.3])
    with wcol1:
        st.metric("🌡️ Live Temperature", f"{weather['temperature']}°C",
                   delta=f"feels {weather['feels_like']}°C")
    with wcol2:
        st.metric("💧 Humidity", f"{weather['humidity']}%")
    with wcol3:
        st.metric("💨 Wind Speed", f"{weather['wind_speed']} m/s")
    with wcol4:
        st.metric("🔵 Pressure", f"{weather['pressure']} hPa")
    with wcol5:
        icon_url = get_weather_icon_url(weather["icon"])
        st.markdown(f"**Condition:** {weather['condition']}")
        st.image(icon_url, width=60)
    st.caption("Live data from OpenWeatherMap — updates each time you reload the page")
else:
    st.warning(f"⚠️ Live weather unavailable right now ({weather['error']}). Showing satellite-derived analysis only.")

st.markdown("---")


# ─────────────────────────────────────────────
# TOP METRICS ROW
# ─────────────────────────────────────────────
st.markdown(f"## 📍 {selected_city} — Urban Heat Analysis")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("🌡️ Avg Temperature", f"{stats['avg_temp']}°C",
              delta=f"+{round(stats['avg_temp'] - city_info['base_temp'] + 2, 1)}°C vs rural")
with col2:
    st.metric("🔥 Max Temperature", f"{stats['max_temp']}°C")
with col3:
    st.metric("❄️ Min Temperature", f"{stats['min_temp']}°C")
with col4:
    st.metric("🔴 Critical Zones", stats['critical_zones'],
              delta=f"out of {stats['total_zones']} total", delta_color="inverse")
with col5:
    st.metric("🌿 Vegetation Cover", f"{stats['vegetation_pct']}%")

st.markdown("---")

# ─────────────────────────────────────────────
# HEATMAP + SURFACE MAP
# ─────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### 🔥 Temperature Heatmap")
    fig_heat = px.density_mapbox(
        df, lat="lat", lon="lon", z="temperature",
        radius=15,
        center={"lat": city_info["lat"], "lon": city_info["lon"]},
        zoom=10,
        mapbox_style="open-street-map",
        color_continuous_scale="RdYlGn_r",
        title=f"Land Surface Temperature — {selected_city}",
        labels={"temperature": "Temp (°C)"},
        height=420
    )
    fig_heat.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
    st.plotly_chart(fig_heat, use_container_width=True)

with col_right:
    st.markdown("### 🏘️ Surface Type Distribution")
    surface_counts = df["surface_type"].value_counts().reset_index()
    surface_counts.columns = ["Surface Type", "Count"]
    colors = {
        "Dark Rooftop": "#2d2d2d",
        "Road/Pavement": "#808080",
        "Bare Ground": "#c8a96e",
        "Vegetation": "#2ecc71",
        "Water Body": "#3498db",
        "Concrete Building": "#e67e22",
    }
    fig_pie = px.pie(
        surface_counts,
        names="Surface Type",
        values="Count",
        color="Surface Type",
        color_discrete_map=colors,
        title="Land Cover Distribution",
        height=420
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

# ─────────────────────────────────────────────
# INTERACTIVE MAP
# ─────────────────────────────────────────────
if show_map:
    st.markdown("### 🗺️ Interactive Satellite Map — Click a Zone")
    st.caption("Each circle = one satellite pixel. Color = heat risk level.")

    color_map = {
        "🔴 Critical": "red",
        "🟠 High": "orange",
        "🟡 Moderate": "yellow",
        "🟢 Low": "green",
    }

    m = folium.Map(
        location=[city_info["lat"], city_info["lon"]],
        zoom_start=11,
        tiles="CartoDB positron"
    )

    # Sample 200 points for map performance
    sample_df = df.sample(min(200, len(df)), random_state=42)

    for _, row in sample_df.iterrows():
        rec = get_recommendation(row["surface_type"])
        popup_html = f"""
        <div style='font-family:Arial; width:220px'>
            <b>🌡️ Temperature:</b> {row['temperature']}°C<br>
            <b>🏘️ Surface:</b> {row['surface_type']}<br>
            <b>⚠️ Risk:</b> {row['heat_risk']}<br>
            <b>🌿 NDVI:</b> {row['ndvi']}<br>
            <hr>
            <b>{rec['strategy']}</b><br>
            <small>{rec['description']}</small><br>
            <b>Temp Reduction: -{rec['temp_reduction']}°C</b>
        </div>
        """
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=6,
            color=color_map.get(row["heat_risk"], "gray"),
            fill=True,
            fill_color=color_map.get(row["heat_risk"], "gray"),
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=240)
        ).add_to(m)

    # Legend
    legend_html = """
    <div style='position:fixed; bottom:30px; left:30px; z-index:1000;
         background:white; padding:10px; border-radius:8px; border:1px solid #ccc;
         font-family:Arial; font-size:13px'>
        <b>Heat Risk Legend</b><br>
        🔴 Critical (&gt;+8°C)<br>
        🟠 High (+4 to +8°C)<br>
        🟡 Moderate (0 to +4°C)<br>
        🟢 Low (below base temp)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    st_folium(m, width=None, height=450)

# ─────────────────────────────────────────────
# ML ANALYSIS
# ─────────────────────────────────────────────
if show_ml:
    st.markdown("---")
    st.markdown("### 🤖 ML Model Analysis")

    col_ml1, col_ml2 = st.columns(2)

    with col_ml1:
        st.success(f"✅ Random Forest Model Accuracy: **{accuracy}%**")

        # Feature importance
        importance = model.get_feature_importance()
        imp_df = pd.DataFrame({
            "Feature": list(importance.keys()),
            "Importance (%)": list(importance.values())
        }).sort_values("Importance (%)", ascending=True)

        fig_imp = px.bar(
            imp_df, x="Importance (%)", y="Feature",
            orientation="h",
            title="🔍 Feature Importance — What drives heat?",
            color="Importance (%)",
            color_continuous_scale="Reds",
            height=320
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_ml2:
        # Temperature by surface type
        fig_box = px.box(
            df, x="surface_type", y="temperature",
            color="surface_type",
            color_discrete_map=colors,
            title="🌡️ Temperature Distribution by Surface Type",
            labels={"temperature": "Temperature (°C)", "surface_type": "Surface"},
            height=380
        )
        fig_box.update_xaxes(tickangle=30)
        st.plotly_chart(fig_box, use_container_width=True)

    # Heat risk breakdown
    st.markdown("#### ⚠️ Heat Risk Zone Breakdown")
    risk_counts = df["heat_risk"].value_counts().reset_index()
    risk_counts.columns = ["Risk Level", "Count"]
    risk_counts["Percentage"] = (risk_counts["Count"] / len(df) * 100).round(1)

    fig_risk = px.bar(
        risk_counts, x="Risk Level", y="Count",
        color="Risk Level",
        color_discrete_map={
            "🔴 Critical": "#e74c3c",
            "🟠 High": "#e67e22",
            "🟡 Moderate": "#f1c40f",
            "🟢 Low": "#2ecc71"
        },
        title="Heat Risk Distribution across City Zones",
        text="Percentage",
        height=320
    )
    fig_risk.update_traces(texttemplate='%{text}%', textposition='outside')
    st.plotly_chart(fig_risk, use_container_width=True)

# ─────────────────────────────────────────────
# DRIVER ANALYSIS — Physics-informed LST regression model
# ─────────────────────────────────────────────
if show_drivers:
    st.markdown("---")
    st.markdown("### 🔬 Driver Analysis — What Physically Causes Urban Heat?")
    st.caption(
        "A Random Forest regression model trained across all 5 cities, using "
        "physically-meaningful drivers (surface albedo, imperviousness, vegetation, "
        "building density, humidity) to predict Land Surface Temperature directly — "
        "going beyond surface classification to model heat dynamics quantitatively."
    )

    @st.cache_resource(show_spinner=False)
    def _cached_driver_model():
        return train_driver_model(grid_size=18)

    with st.spinner("Training physics-informed LST regression model across all cities..."):
        driver_result = _cached_driver_model()

    dcol1, dcol2, dcol3 = st.columns(3)
    dcol1.metric("📊 Model R² Score", driver_result["r2"],
                 help="Fraction of temperature variance explained by the physical drivers (closer to 1.0 = better)")
    dcol2.metric("📏 RMSE", f"{driver_result['rmse']}°C",
                 help="Average prediction error in degrees Celsius on held-out test data")
    dcol3.metric("🗂️ Training Samples", f"{driver_result['n_samples']:,}",
                 help="Pixel-level samples across all 5 cities used to train and validate the model")

    driver_df = pd.DataFrame({
        "Driver": list(driver_result["driver_importance"].keys()),
        "Influence (%)": list(driver_result["driver_importance"].values()),
    }).sort_values("Influence (%)", ascending=True)

    fig_driver = px.bar(
        driver_df, x="Influence (%)", y="Driver",
        orientation="h",
        title="Quantified Influence of Each Physical Driver on Land Surface Temperature",
        color="Influence (%)",
        color_continuous_scale="OrRd",
        height=320,
    )
    st.plotly_chart(fig_driver, use_container_width=True)

    top_driver = driver_df.iloc[-1]
    st.success(
        f"🎯 **Key finding:** *{top_driver['Driver']}* is the strongest driver of urban heat "
        f"across the 5 cities analyzed, accounting for **{top_driver['Influence (%)']}%** of "
        f"the model's predictive signal — directly informing which intervention type "
        f"(e.g. cool roofs to raise albedo, greening to lower imperviousness) will be most effective."
    )

# ─────────────────────────────────────────────
# ACTION PLAN
# ─────────────────────────────────────────────
if show_plan:
    st.markdown("---")
    st.markdown("### 📋 AI-Generated Cooling Action Plan")

    total_impact = calculate_total_impact(df)
    action_plan = get_city_action_plan(df)

    st.info(f"""
    🎯 **If ALL recommended strategies are implemented:**
    Estimated average temperature reduction = **{total_impact}°C** across {selected_city}
    This could make the city **significantly more livable** and reduce energy consumption by ~15–25%.
    """)

    for i, action in enumerate(action_plan, 1):
        surface = action["Surface"]
        rec = get_recommendation(surface)
        with st.expander(f"{rec['icon']} Priority {i}: {action['Strategy']} — {surface}"):
            c1, c2, c3 = st.columns(3)
            c1.metric("🌡️ Temp Reduction", f"-{action['Temp Reduction (°C)']}°C")
            c2.metric("📍 Zones Affected", action["Zones Affected"])
            c3.metric("💥 Impact", action["Impact"])

            st.markdown(f"**📝 What to do:** {rec['description']}")
            st.markdown(f"**💰 Estimated Cost:** {rec['cost']}")
            st.markdown(f"**⏱️ Implementation Time:** {rec['implementation_time']}")
            st.success(f"📡 **ISRO/NASA Reference:** {rec['isro_reference']}")

# ─────────────────────────────────────────────
# SCENARIO-BASED SIMULATION
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🎛️ What-If Scenario Simulator")
st.caption("Move the sliders to simulate real-world interventions and see the projected cooling effect instantly.")

sim_col1, sim_col2 = st.columns([1, 1.3])

with sim_col1:
    tree_pct = st.slider("🌳 % of bare ground converted to trees/greenery", 0, 100, 0, step=5)
    roof_pct = st.slider("🏠 % of dark rooftops converted to cool roofs", 0, 100, 0, step=5)
    pavement_pct = st.slider("🛣️ % of roads converted to reflective pavement", 0, 100, 0, step=5)

    sim_result = simulate_scenario(df, stats["avg_temp"], tree_pct, roof_pct, pavement_pct)

with sim_col2:
    fig_sim = go.Figure()
    fig_sim.add_trace(go.Bar(
        x=["Current Avg Temp", "Projected Avg Temp"],
        y=[sim_result["current_temp"], sim_result["projected_temp"]],
        marker_color=["#e74c3c", "#2ecc71"],
        text=[f"{sim_result['current_temp']}°C", f"{sim_result['projected_temp']}°C"],
        textposition="outside",
    ))
    fig_sim.update_layout(
        title=f"Projected Impact for {selected_city}",
        yaxis_title="Temperature (°C)",
        height=320,
        showlegend=False,
    )
    st.plotly_chart(fig_sim, use_container_width=True)

if sim_result["total_reduction"] > 0:
    st.success(f"""
    🎯 **With these interventions:** average temperature drops from
    **{sim_result['current_temp']}°C → {sim_result['projected_temp']}°C**
    (a reduction of **{sim_result['total_reduction']}°C**)

    Breakdown: 🌳 Trees: -{sim_result['tree_contribution']}°C |
    🏠 Cool Roofs: -{sim_result['roof_contribution']}°C |
    🛣️ Pavement: -{sim_result['pavement_contribution']}°C
    """)
else:
    st.info("👆 Move the sliders above to simulate cooling interventions and see the projected temperature drop.")

# ─────────────────────────────────────────────
# URBAN MORPHOLOGY (OpenStreetMap)
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🏙️ Urban Morphology Analysis")
st.caption("Building density and land-use form data from OpenStreetMap (~4km radius around city center)")

morphology = get_urban_morphology(city_info["lat"], city_info["lon"], city_name=selected_city)

if morphology["success"]:
    source_label = "🟢 Live from OpenStreetMap" if morphology.get("source") == "live" else "🔵 Cached snapshot (live server busy)"
    st.caption(f"Data source: {source_label}")

    um1, um2, um3, um4 = st.columns(4)
    um1.metric("🏢 Buildings", f"{morphology['building_count']:,}")
    um2.metric("🛣️ Roads", f"{morphology['road_count']:,}")
    um3.metric("🌳 Green Spaces", f"{morphology['green_space_count']:,}")
    um4.metric("📊 Density", morphology["density_score"])

    if morphology["density_score"] in ["High", "Very High"]:
        st.warning(f"⚠️ {selected_city} shows **{morphology['density_score']}** building density — dense urban form traps heat and limits natural cooling. Prioritize vertical greening and cool roofs here.")
    else:
        st.success(f"✅ {selected_city} shows **{morphology['density_score']}** building density in this zone — more room for ground-level tree planting.")
else:
    st.warning(f"⚠️ Urban morphology data temporarily unavailable ({morphology['error']}). OpenStreetMap's free server can be slow — try refreshing in a moment.")

# ─────────────────────────────────────────────
# CITY COMPARISON
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🏙️ Multi-City Comparison")

with st.spinner("Loading all city data..."):
    city_summaries = []
    for city_name in CITIES.keys():
        city_df = generate_city_data(city_name, 15)
        s = get_city_stats(city_df)
        imp = calculate_total_impact(city_df)
        city_summaries.append({
            "City": city_name,
            "Avg Temp (°C)": s["avg_temp"],
            "Max Temp (°C)": s["max_temp"],
            "Critical Zones": s["critical_zones"],
            "Vegetation %": s["vegetation_pct"],
            "Potential Reduction (°C)": imp
        })

    summary_df = pd.DataFrame(city_summaries)

col_c1, col_c2 = st.columns(2)
with col_c1:
    fig_comp = px.bar(
        summary_df, x="City", y="Avg Temp (°C)",
        color="Avg Temp (°C)",
        color_continuous_scale="RdYlGn_r",
        title="Average Urban Temperature by City",
        height=340
    )
    st.plotly_chart(fig_comp, use_container_width=True)

with col_c2:
    fig_pot = px.bar(
        summary_df, x="City", y="Potential Reduction (°C)",
        color="Potential Reduction (°C)",
        color_continuous_scale="Greens",
        title="Cooling Potential if Strategies Applied (°C)",
        height=340
    )
    st.plotly_chart(fig_pot, use_container_width=True)

st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#888; font-size:0.85rem'>
    🛰️ <b>HeatShield AI</b> — Built for ISRO × Hack2Skill Hackathon |
    Data: Landsat 8/9 + INSAT-3D patterns |
    Model: Random Forest Classifier |
    Made with ❤️ using Python + Streamlit
</div>
""", unsafe_allow_html=True)
