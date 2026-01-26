"""
F1PA Streamlit - Configuration
"""
import os

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_USERNAME = os.getenv("API_USERNAME", "f1pa")
API_PASSWORD = os.getenv("API_PASSWORD", "f1pa")

# External Links
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5000")
GITHUB_URL = "https://github.com/Aurelien-L/F1PA"
EVIDENTLY_URL = os.getenv("EVIDENTLY_URL", "http://localhost:8080")  # Placeholder

# Default values for prediction
DEFAULT_TEMP = 25.0
DEFAULT_RHUM = 50.0
DEFAULT_PRES = 1013.0
DEFAULT_LAP_NUMBER = 1
DEFAULT_YEAR = 2025

# Speed ranges (km/h)
SPEED_MIN = 150.0
SPEED_MAX = 370.0
DEFAULT_ST_SPEED = 310.0
DEFAULT_I1_SPEED = 295.0
DEFAULT_I2_SPEED = 285.0
