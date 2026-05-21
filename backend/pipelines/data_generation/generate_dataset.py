import numpy as np
import random
from typing import List, Dict, Any
from backend.utils.config import config

class DataGenerator:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.py_rng = random.Random(seed)
        self.regions = config.regions
        self.region_names = list(self.regions.keys())

    def _get_weather(self, season: str) -> Dict[str, float]:
        """Generate realistic weather for Moldova based on season."""
        if season == "Summer":
            temp = self.rng.uniform(20, 38)
            humidity = self.rng.uniform(20, 50)
            rainfall = self.rng.choice([0, 0, 0, 5], p=[0.8, 0.1, 0.05, 0.05])
        elif season == "Winter":
            temp = self.rng.uniform(-15, 5)
            humidity = self.rng.uniform(60, 95)
            rainfall = self.rng.choice([0, 2, 10], p=[0.7, 0.2, 0.1]) # Rain or snow
        elif season == "Spring":
            temp = self.rng.uniform(5, 20)
            humidity = self.rng.uniform(40, 70)
            rainfall = self.rng.uniform(0, 5)
        else: # Autumn
            temp = self.rng.uniform(5, 15)
            humidity = self.rng.uniform(50, 80)
            rainfall = self.rng.uniform(0, 8)
        
        wind_speed = self.rng.rayleigh(3.0) # Realistic wind distribution
        return {
            "temperature": round(temp, 1),
            "humidity": round(humidity, 1),
            "wind_speed": round(wind_speed, 1),
            "rainfall": round(rainfall, 1)
        }

    def _get_coords(self, region_name: str) -> Dict[str, float]:
        """Generate lat/lng within region radius."""
        reg = self.regions[region_name]
        # Very rough approx: 1 deg ~ 111km
        radius_deg = reg['radius_km'] / 111.0
        lat = reg['lat'] + self.rng.uniform(-radius_deg, radius_deg)
        lng = reg['lng'] + self.rng.uniform(-radius_deg, radius_deg)
        return {"latitude": round(lat, 4), "longitude": round(lng, 4)}

    def generate_row(self) -> Dict[str, Any]:
        region = self.rng.choice(self.region_names)
        season = self.rng.choice(["Spring", "Summer", "Autumn", "Winter"])
        weather = self._get_weather(season)
        coords = self._get_coords(region)
        
        hour = self.rng.randint(0, 24)
        day_of_week = self.rng.randint(0, 7)
        sensor_id = f"SN-{region[:3].upper()}-{self.rng.randint(100, 999)}"

        # Audio confidence scores with correlations
        # Chainsaw more likely in Codrii
        chainsaw_base = 0.05 if region == "Codrii" else 0.01
        chainsaw_conf = np.clip(self.rng.exponential(chainsaw_base), 0, 1)
        
        # Fire more likely in Summer, South/North
        fire_base = 0.08 if season == "Summer" and region in ["South", "North"] else 0.01
        fire_conf = np.clip(self.rng.exponential(fire_base), 0, 1)
        
        # Gunshot random but slightly higher in forest/agricultural
        gunshot_base = 0.03 if region in ["Codrii", "North"] else 0.01
        gunshot_conf = np.clip(self.rng.exponential(gunshot_base), 0, 1)

        # Occasional high spikes for events
        if self.rng.random() < 0.4: # 40% chance of something interesting
            event_type = self.rng.choice(["chainsaw", "fire", "gunshot", "anomaly", "rolling_fire"])
            if event_type == "chainsaw" and region == "Codrii":
                chainsaw_conf = self.rng.uniform(0.7, 0.95)
            elif event_type == "fire" and weather["humidity"] < 50:
                fire_conf = self.rng.uniform(0.75, 0.98)
            elif event_type == "gunshot" and region != "Chisinau":
                gunshot_conf = self.rng.uniform(0.7, 0.9)
            elif event_type == "rolling_fire":
                fire_conf = self.rng.uniform(0.5, 0.8)
                # This will trigger rolling average elevation in the next step
        
        # Simulated rolling averages (based on current values with noise)
        def get_rolling(current, window_mins, event_type=None):
            # If current is high, rolling should be elevated
            if current > 0.5:
                return np.clip(current * self.rng.uniform(0.6, 0.9), 0, 1)
            if event_type == "rolling_fire" and "fire" in str(current): # simplified check
                 return self.rng.uniform(0.45, 0.6)
            return np.clip(self.rng.exponential(0.02), 0, 1)

        # Re-evaluating event_type for rolling
        event_type = None
        if self.rng.random() < 0.4: # same probability as above
             event_type = self.rng.choice(["chainsaw", "fire", "gunshot", "anomaly", "rolling_fire"])

        row = {
            **weather,
            "hour_of_day": hour,
            "day_of_week": day_of_week,
            "season": season,
            "sensor_id": sensor_id,
            "region": region,
            **coords,
            "chainsaw_confidence": round(float(chainsaw_conf), 4),
            "fire_confidence": round(float(fire_conf), 4),
            "gunshot_confidence": round(float(gunshot_conf), 4),
            "wildlife_activity_score": round(float(self.rng.beta(2, 5)), 4),
            "rolling_chainsaw_5m": round(float(get_rolling(chainsaw_conf, 5)), 4),
            "rolling_chainsaw_10m": round(float(get_rolling(chainsaw_conf, 10)), 4),
            "rolling_chainsaw_30m": round(float(get_rolling(chainsaw_conf, 30)), 4),
            "rolling_fire_5m": round(float(get_rolling(fire_conf, 5)), 4),
            "rolling_fire_10m": round(float(get_rolling(fire_conf, 10)), 4),
            "rolling_fire_30m": round(float(get_rolling(fire_conf, 30, event_type)), 4),
            "detection_frequency_1h": int(self.rng.randint(0, 15)),
            "anomaly_score": round(float(self.rng.uniform(0, 1)), 4)
        }
        
        # Ensure high anomaly score if we have high confidences
        if max(chainsaw_conf, fire_conf, gunshot_conf) > 0.6:
            row["anomaly_score"] = round(float(self.rng.uniform(0.7, 0.99)), 4)

        return row
