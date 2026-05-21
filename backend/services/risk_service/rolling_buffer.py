import threading
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class RollingBuffer:
    """
    In-memory rolling window manager (per sensor_id).
    Maintains 5m / 10m / 30m rolling averages of audio confidence scores.
    """
    def __init__(self, max_window_mins: int = 30):
        self.lock = threading.Lock()
        self.data: Dict[str, deque] = {} # sensor_id -> deque of (timestamp, confidences_dict)
        self.max_window = timedelta(minutes=max_window_mins)

    def add_observation(self, sensor_id: str, confidences: Dict[str, float]):
        with self.lock:
            if sensor_id not in self.data:
                self.data[sensor_id] = deque()
            
            now = datetime.now()
            self.data[sensor_id].append((now, confidences))
            self._cleanup(sensor_id, now)

    def _cleanup(self, sensor_id: str, now: datetime):
        # Remove old observations outside the max window
        cutoff = now - self.max_window
        while self.data[sensor_id] and self.data[sensor_id][0][0] < cutoff:
            self.data[sensor_id].popleft()

    def get_averages(self, sensor_id: str, minutes: int) -> Dict[str, float]:
        with self.lock:
            if sensor_id not in self.data:
                return {"fire": 0.0, "chainsaw": 0.0, "gunshot": 0.0}
            
            now = datetime.now()
            cutoff = now - timedelta(minutes=minutes)
            
            # Filter observations within the requested window
            relevant = [obs for ts, obs in self.data[sensor_id] if ts >= cutoff]
            
            if not relevant:
                return {"fire": 0.0, "chainsaw": 0.0, "gunshot": 0.0}
            
            # Calculate averages for key classes
            avg_fire = sum(r.get("fire", 0.0) for r in relevant) / len(relevant)
            avg_chainsaw = sum(r.get("chainsaw", 0.0) for r in relevant) / len(relevant)
            avg_gunshot = sum(r.get("gunshot", 0.0) for r in relevant) / len(relevant)
            
            return {
                "fire": round(avg_fire, 4),
                "chainsaw": round(avg_chainsaw, 4),
                "gunshot": round(avg_gunshot, 4)
            }

# Global instance
rolling_buffer = RollingBuffer()
