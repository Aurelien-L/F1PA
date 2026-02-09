"""
Simple LOAD pipeline using docker exec and SQL COPY.
Works around Windows psycopg2 connection issues.
"""
import subprocess
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTAINER = "f1pa_postgres"


def log(msg: str) -> None:
    print(f"[LOAD] {msg}", flush=True)


def sql(command: str) -> str:
    """Execute SQL via docker exec."""
    result = subprocess.run(
        ["docker", "exec", "-i", CONTAINER, "psql", "-U", "f1pa", "-d", "f1pa_db", "-c", command],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"SQL failed: {result.stderr}")
    return result.stdout


def copy_csv_to_table(df: pd.DataFrame, table: str, columns: list[str], temp_name: str) -> None:
    """Copy pandas DataFrame into PostgreSQL table via docker."""
    temp_csv = PROJECT_ROOT / f"temp_{temp_name}.csv"
    # Use empty string for NULL and handle it in PostgreSQL COPY command
    df[columns].to_csv(temp_csv, index=False, header=False)

    try:
        # Copy file into container
        subprocess.run(
            ["docker", "cp", str(temp_csv), f"{CONTAINER}:/tmp/{temp_name}.csv"],
            check=True
        )

        # COPY into table (empty string = NULL)
        cols_str = ", ".join(columns)
        sql(f"COPY {table} ({cols_str}) FROM '/tmp/{temp_name}.csv' WITH (FORMAT CSV, NULL '')")
        log(f"Loaded {len(df)} rows into {table}")
    finally:
        temp_csv.unlink(missing_ok=True)


def main():
    log("=" * 80)
    log("F1PA LOAD PIPELINE")
    log("=" * 80)

    # Paths
    sessions_csv = PROJECT_ROOT / "data/transform/sessions_scope_2023_2024_2025.csv"
    drivers_csv = PROJECT_ROOT / "data/extract/openf1/openf1_drivers_2023_2024_2025.csv"
    dataset_csv = PROJECT_ROOT / "data/processed/dataset_ml_lap_level_2023_2024_2025.csv"

    # 1. dim_circuits
    log("\n[1/4] Loading dim_circuits...")
    df = pd.read_csv(sessions_csv)

    # Select available columns
    circuit_cols = ["circuit_key", "circuit_short_name", "location", "country_name", "country_code", "wikipedia_circuit_url", "station_id"]
    available_cols = [c for c in circuit_cols if c in df.columns]
    circuits = df[available_cols].drop_duplicates("circuit_key").sort_values("circuit_key")

    # Add missing columns as None
    for col in circuit_cols:
        if col not in circuits.columns:
            circuits[col] = None
    sql("TRUNCATE TABLE dim_circuits CASCADE")
    copy_csv_to_table(
        circuits, "dim_circuits",
        ["circuit_key", "circuit_short_name", "location", "country_name", "country_code", "wikipedia_circuit_url", "station_id"],
        "circuits"
    )

    # 2. dim_drivers
    log("\n[2/4] Loading dim_drivers...")
    drivers = pd.read_csv(drivers_csv)[[
        "driver_number", "full_name", "broadcast_name", "name_acronym",
        "first_name", "last_name", "country_code", "team_name", "team_colour", "headshot_url"
    ]].drop_duplicates("driver_number").sort_values("driver_number")
    sql("TRUNCATE TABLE dim_drivers CASCADE")
    copy_csv_to_table(
        drivers, "dim_drivers",
        ["driver_number", "full_name", "broadcast_name", "name_acronym", "first_name", "last_name", "country_code", "team_name", "team_colour", "headshot_url"],
        "drivers"
    )

    # 3. dim_sessions
    log("\n[3/4] Loading dim_sessions...")
    sessions = df[[
        "session_key", "meeting_key", "year", "session_name", "session_type",
        "date_start", "date_end", "gmt_offset", "circuit_key"
    ]].drop_duplicates("session_key").sort_values(["year", "meeting_key"])
    sql("TRUNCATE TABLE dim_sessions CASCADE")
    copy_csv_to_table(
        sessions, "dim_sessions",
        ["session_key", "meeting_key", "year", "session_name", "session_type", "date_start", "date_end", "gmt_offset", "circuit_key"],
        "sessions"
    )

    # 4. fact_laps
    log("\n[4/4] Loading fact_laps...")
    laps = pd.read_csv(dataset_csv)

    # Select columns
    lap_cols = [
        "meeting_key", "session_key", "driver_number", "lap_number",
        "year", "circuit_key", "lap_hour_utc",
        "st_speed", "i1_speed", "i2_speed",
        "duration_sector_1", "duration_sector_2", "duration_sector_3",
        "temp", "rhum", "pres", "wspd", "wdir", "prcp", "cldc",
        "lap_duration"
    ]
    laps_filtered = laps[lap_cols + ["__source_file"]].copy()
    laps_filtered = laps_filtered.rename(columns={"__source_file": "source_file"})
    laps_filtered = laps_filtered[laps_filtered["lap_duration"].notna()]

    log(f"Filtered to {len(laps_filtered)} laps with valid target")

    sql("TRUNCATE TABLE fact_laps CASCADE")
    copy_csv_to_table(
        laps_filtered, "fact_laps",
        lap_cols + ["source_file"],
        "laps"
    )

    # Stats
    log("\n" + "=" * 80)
    log("SUMMARY")
    log("=" * 80)
    stats = sql("SELECT COUNT(*) FROM dim_circuits")
    log(f"dim_circuits: {stats.strip().split()[2]} rows")
    stats = sql("SELECT COUNT(*) FROM dim_drivers")
    log(f"dim_drivers: {stats.strip().split()[2]} rows")
    stats = sql("SELECT COUNT(*) FROM dim_sessions")
    log(f"dim_sessions: {stats.strip().split()[2]} rows")
    stats = sql("SELECT COUNT(*) FROM fact_laps")
    log(f"fact_laps: {stats.strip().split()[2]} rows")

    log("\nDatabase ready!")


if __name__ == "__main__":
    main()
