import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class HeatIslandModel:
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoder = LabelEncoder()
        self.is_trained = False

    def prepare_features(self, df):
        """Extract ML features from dataframe."""
        features = pd.DataFrame({
            "temperature": df["temperature"],
            "ndvi": df["ndvi"],
            "lat": df["lat"],
            "lon": df["lon"],
            # Derived features
            "temp_ndvi_ratio": df["temperature"] / (df["ndvi"].abs() + 0.1),
            "is_vegetated": (df["ndvi"] > 0.3).astype(int),
        })
        return features

    def train(self, df):
        """Train the Random Forest classifier."""
        X = self.prepare_features(df)
        y = self.label_encoder.fit_transform(df["surface_type"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        self.is_trained = True
        return round(acc * 100, 1)

    def predict_surface(self, df):
        """Predict surface type for new data."""
        if not self.is_trained:
            return df["surface_type"]
        X = self.prepare_features(df)
        preds = self.classifier.predict(X)
        return self.label_encoder.inverse_transform(preds)

    def get_feature_importance(self):
        """Return feature importance for explainability."""
        if not self.is_trained:
            return {}
        features = ["temperature", "ndvi", "lat", "lon", "temp_ndvi_ratio", "is_vegetated"]
        importance = self.classifier.feature_importances_
        return dict(zip(features, (importance * 100).round(1)))

    def predict_heat_risk(self, temperature, city_base_temp):
        """Predict heat risk level for a given temperature."""
        diff = temperature - city_base_temp
        if diff >= 8:
            return "🔴 Critical", diff
        elif diff >= 4:
            return "🟠 High", diff
        elif diff >= 0:
            return "🟡 Moderate", diff
        else:
            return "🟢 Low", diff
