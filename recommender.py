RECOMMENDATIONS = {
    "Dark Rooftop": {
        "strategy": "🏠 Cool Roof Coating",
        "description": "Apply white/reflective paint or tiles on rooftop surface.",
        "temp_reduction": 3.5,
        "cost": "Low (₹15,000 – ₹40,000 per 1000 sq.ft)",
        "implementation_time": "1–2 days",
        "impact": "High",
        "isro_reference": "ISRO's SAC study shows cool roofs reduce indoor temp by 3–5°C",
        "icon": "🏠"
    },
    "Road/Pavement": {
        "strategy": "🛣️ Reflective / Permeable Pavement",
        "description": "Replace dark asphalt with light-colored or permeable paving materials.",
        "temp_reduction": 2.8,
        "cost": "Medium (₹500–₹800 per sq.m)",
        "implementation_time": "1–2 weeks",
        "impact": "Medium-High",
        "isro_reference": "Urban heat studies show reflective pavements reduce surface temp by 2–4°C",
        "icon": "🛣️"
    },
    "Bare Ground": {
        "strategy": "🌳 Urban Greening / Tree Plantation",
        "description": "Plant native trees and grass cover on bare soil areas.",
        "temp_reduction": 4.2,
        "cost": "Very Low (₹200–₹500 per sapling)",
        "implementation_time": "Immediate planting, 1–2 years for full effect",
        "impact": "Very High",
        "isro_reference": "NASA studies confirm urban trees reduce surrounding air temp by 3–5°C",
        "icon": "🌳"
    },
    "Concrete Building": {
        "strategy": "🌿 Green Walls / Vertical Gardens",
        "description": "Install vertical gardens or climbing plants on building facades.",
        "temp_reduction": 2.5,
        "cost": "Medium (₹800–₹1500 per sq.m)",
        "implementation_time": "2–4 weeks",
        "impact": "Medium",
        "isro_reference": "Green walls reduce building surface temp by 2–3°C and improve air quality",
        "icon": "🌿"
    },
    "Vegetation": {
        "strategy": "✅ Preserve & Expand Green Cover",
        "description": "This zone is already vegetated. Protect and expand green cover.",
        "temp_reduction": 0.0,
        "cost": "Minimal — just maintenance",
        "implementation_time": "Ongoing",
        "impact": "Preventive",
        "isro_reference": "ISRO LISS data shows vegetation zones are 4–6°C cooler than built-up areas",
        "icon": "✅"
    },
    "Water Body": {
        "strategy": "💧 Preserve Water Body",
        "description": "Protect lakes and water bodies from encroachment. Add more water features.",
        "temp_reduction": 0.0,
        "cost": "Minimal — protect from encroachment",
        "implementation_time": "Policy-level",
        "impact": "Preventive",
        "isro_reference": "Water bodies create natural cooling corridors reducing nearby temps by 5–8°C",
        "icon": "💧"
    },
}

def get_recommendation(surface_type):
    """Get cooling recommendation for a surface type."""
    return RECOMMENDATIONS.get(surface_type, RECOMMENDATIONS["Bare Ground"])

def get_city_action_plan(df):
    """
    Generate a prioritized action plan for the entire city.
    Returns top interventions sorted by impact.
    """
    surface_counts = df["surface_type"].value_counts()
    action_plan = []

    for surface, count in surface_counts.items():
        rec = get_recommendation(surface)
        if rec["temp_reduction"] > 0:
            action_plan.append({
                "Surface": surface,
                "Zones Affected": count,
                "Strategy": rec["strategy"],
                "Temp Reduction (°C)": rec["temp_reduction"],
                "Cost": rec["cost"],
                "Impact": rec["impact"],
            })

    action_plan.sort(key=lambda x: x["Temp Reduction (°C)"], reverse=True)
    return action_plan

def calculate_total_impact(df):
    """Calculate total temperature reduction if all strategies applied."""
    total_reduction = 0
    for surface in df["surface_type"].unique():
        rec = get_recommendation(surface)
        zone_count = (df["surface_type"] == surface).sum()
        total_reduction += rec["temp_reduction"] * (zone_count / len(df))
    return round(total_reduction, 2)
