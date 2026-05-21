import os
import yaml
from pathlib import Path
from typing import Dict, Any

class Config:
    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        # Find config.yaml relative to this file
        config_path = Path(__file__).parent.parent / "config.yaml"
        if not config_path.exists():
            # Fallback for different execution contexts
            config_path = Path("backend/config.yaml")
        
        if config_path.exists():
            with open(config_path, "r") as f:
                self._config = yaml.safe_load(f)
        else:
            print(f"Warning: Configuration file not found at {config_path}")

    @property
    def models(self) -> Dict[str, str]:
        return self._config.get("models", {})

    @property
    def thresholds(self) -> Dict[str, float]:
        return self._config.get("thresholds", {})

    @property
    def regions(self) -> Dict[str, Dict[str, Any]]:
        return self._config.get("regions", {})

    @property
    def paths(self) -> Dict[str, str]:
        return self._config.get("paths", {})

    @property
    def system(self) -> Dict[str, Any]:
        return self._config.get("system", {})

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

# Global config instance
config = Config()

# Expose typed constants for easier access
AUDIO_MODEL_PATH = config.models.get("audio_model_path", "backend/ml/audio_model/best_checkpoint.pth")
CONFIDENCE_THRESHOLD = config.thresholds.get("confidence_threshold", 0.3)
RANDOM_SEED = config.system.get("random_seed", 42)
LOGGING_LEVEL = config.system.get("logging_level", "INFO")
DATASET_OUTPUT_PATH = config.paths.get("dataset_output_path", "Data/Harvested/")
