"""
ETL data quality validation tests
"""
import os
import pytest
import pandas as pd


@pytest.fixture(scope="module")
def dataset_ml():
    """Load final ML dataset"""
    csv_path = "data/processed/dataset_ml_lap_level_2023_2024_2025.csv"

    if not os.path.exists(csv_path):
        pytest.skip(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def test_dataset_exists():
    """Test: final ML dataset exists"""
    csv_path = "data/processed/dataset_ml_lap_level_2023_2024_2025.csv"
    assert os.path.exists(csv_path), f"Dataset ML not found: {csv_path}"


def test_dataset_not_empty(dataset_ml):
    """Test: dataset contains data"""
    assert len(dataset_ml) > 0, "Dataset is empty"
    assert len(dataset_ml) > 1000, f"Dataset too small: {len(dataset_ml)} rows"


def test_dataset_schema(dataset_ml):
    """Test: ML dataset schema has required columns"""
    required_columns = [
        "year", "meeting_key", "session_key", "circuit_key", "driver_number", "lap_number",
        "session_name", "session_type", "location", "country_name", "date_start_session",
        "st_speed", "i1_speed", "i2_speed",
        "duration_sector_1", "duration_sector_2", "duration_sector_3",
        "temp", "rhum", "pres",
        "lap_duration"
    ]

    missing_cols = [col for col in required_columns if col not in dataset_ml.columns]
    assert len(missing_cols) == 0, f"Missing columns: {missing_cols}"


def test_speed_ranges(dataset_ml):
    """Test: speeds within realistic F1 ranges (20-380 km/h)"""
    speed_columns = ["st_speed", "i1_speed", "i2_speed"]

    for col in speed_columns:
        non_null = dataset_ml[col].dropna()
        assert len(non_null) > 0, f"All {col} values are null"

        min_speed = non_null.min()
        max_speed = non_null.max()

        assert min_speed >= 20, f"{col} min too low: {min_speed} km/h"
        assert max_speed <= 380, f"{col} max too high: {max_speed} km/h"

        in_range = non_null.between(150, 370).sum()
        percentage = (in_range / len(non_null)) * 100
        assert percentage >= 75, f"{col}: only {percentage:.1f}% in racing range 150-370 km/h"


def test_lap_duration_ranges(dataset_ml):
    """Test: lap_duration within realistic F1 ranges"""
    lap_durations = dataset_ml["lap_duration"].dropna()

    assert len(lap_durations) > 0, "All lap_duration values are null"

    min_duration = lap_durations.min()
    max_duration = lap_durations.max()

    assert min_duration >= 50, f"lap_duration min too low: {min_duration}s"
    assert max_duration <= 1200, f"lap_duration max too high: {max_duration}s"

    in_range = lap_durations.between(60, 130).sum()
    percentage = (in_range / len(lap_durations)) * 100
    assert percentage >= 60, f"Only {percentage:.1f}% in racing range 60-130s"


def test_lap_duration_outliers(dataset_ml):
    """
    Test: detects extreme outliers (> 300s).

    ETL already removes outliers using per-session quantiles (Q0.01-Q0.99).
    This test detects remaining extreme values from anomalous sessions (red flags, incidents).
    Random Forest models are robust to these rare outliers (< 0.1% of data).
    """
    outliers = dataset_ml[dataset_ml["lap_duration"] > 300]
    outlier_percentage = (len(outliers) / len(dataset_ml)) * 100

    assert outlier_percentage < 0.1, f"{outlier_percentage:.2f}% outliers > 300s (typically from incident sessions)"


def test_no_critical_nan(dataset_ml):
    """Test: limited NaN on critical features"""
    strictly_critical = ["lap_duration", "driver_number", "circuit_key"]

    for col in strictly_critical:
        null_count = dataset_ml[col].isna().sum()
        null_percentage = (null_count / len(dataset_ml)) * 100
        assert null_percentage < 1, f"{col}: {null_percentage:.1f}% null (strictly critical)"

    important_features = {"st_speed": 10, "i1_speed": 20, "i2_speed": 20}

    for col, max_null_pct in important_features.items():
        null_count = dataset_ml[col].isna().sum()
        null_percentage = (null_count / len(dataset_ml)) * 100
        assert null_percentage < max_null_pct, f"{col}: {null_percentage:.1f}% null (max {max_null_pct}%)"


def test_sector_durations_coherence(dataset_ml):
    """Test: sector durations coherent with lap_duration"""
    df_complete = dataset_ml.dropna(subset=["duration_sector_1", "duration_sector_2", "duration_sector_3", "lap_duration"])

    if len(df_complete) > 0:
        total_sectors = (
            df_complete["duration_sector_1"] +
            df_complete["duration_sector_2"] +
            df_complete["duration_sector_3"]
        )

        diff = (total_sectors - df_complete["lap_duration"]).abs()

        coherent = (diff < 2.0).sum()
        percentage = (coherent / len(df_complete)) * 100

        assert percentage >= 85, f"Only {percentage:.1f}% laps have coherent sector durations"


def test_weather_ranges(dataset_ml):
    """Test: weather data within realistic ranges"""
    weather_checks = {
        "temp": (-10, 50, "°C"),
        "rhum": (0, 100, "%"),
        "pres": (900, 1100, "hPa"),
    }

    for col, (min_val, max_val, unit) in weather_checks.items():
        if col in dataset_ml.columns:
            non_null = dataset_ml[col].dropna()

            if len(non_null) > 0:
                col_min = non_null.min()
                col_max = non_null.max()

                assert col_min >= min_val, f"{col} out of range: {col_min}{unit}"
                assert col_max <= max_val, f"{col} out of range: {col_max}{unit}"


def test_driver_circuit_positive(dataset_ml):
    """Test: driver_number and circuit_key are positive"""
    assert (dataset_ml["driver_number"] > 0).all(), "Some driver_number values are <= 0"
    assert (dataset_ml["circuit_key"] > 0).all(), "Some circuit_key values are <= 0"


def test_year_validity(dataset_ml):
    """Test: years are coherent (2023-2025)"""
    years = dataset_ml["year"].unique()

    for year in years:
        assert 2023 <= year <= 2025, f"Invalid year: {year}"


def test_data_types(dataset_ml):
    """Test: appropriate data types"""
    numeric_columns = ["st_speed", "i1_speed", "i2_speed", "lap_duration", "temp", "pres"]

    for col in numeric_columns:
        if col in dataset_ml.columns:
            assert pd.api.types.is_numeric_dtype(dataset_ml[col]), f"{col} is not numeric type"


def test_no_duplicate_laps(dataset_ml):
    """Test: no duplicate laps"""
    key_columns = ["session_key", "driver_number", "lap_number"]

    duplicates = dataset_ml.duplicated(subset=key_columns, keep=False)
    duplicate_count = duplicates.sum()

    assert duplicate_count == 0, f"Found {duplicate_count} duplicate laps"


def test_lap_numbers_sequential(dataset_ml):
    """Test: lap numbers are coherent"""
    assert (dataset_ml["lap_number"] >= 1).all(), "Some lap_number values are < 1"

    sample_session = dataset_ml.groupby(["session_key", "driver_number"]).head(10)

    for (session, driver), group in sample_session.groupby(["session_key", "driver_number"]):
        laps = sorted(group["lap_number"].unique())

        if len(laps) > 1:
            max_gap = max([laps[i+1] - laps[i] for i in range(len(laps)-1)])
            assert max_gap <= 20, f"Session {session}, driver {driver}: lap gap too large ({max_gap})"
