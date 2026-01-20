-- ===========================================================================
-- F1PA - PostgreSQL Database Schema
-- ===========================================================================
--
-- Purpose: Load lap-level F1 data for ML predictions
-- Granularity: lap-level (meeting_key, session_key, driver_number, lap_number)
-- Target: lap_duration (seconds)
--
-- Architecture: Star Schema simplified
-- - fact_laps: central fact table with all lap metrics
-- - dim_drivers: driver reference data
-- - dim_sessions: session context
-- - dim_circuits: circuit reference data
--
-- ===========================================================================

-- Drop existing tables (cascade to handle foreign keys)
DROP TABLE IF EXISTS fact_laps CASCADE;
DROP TABLE IF EXISTS dim_drivers CASCADE;
DROP TABLE IF EXISTS dim_sessions CASCADE;
DROP TABLE IF EXISTS dim_circuits CASCADE;

-- ===========================================================================
-- DIMENSION: dim_circuits
-- ===========================================================================
-- Reference table for F1 circuits
-- Populated from sessions data + wikipedia enrichment
-- ===========================================================================

CREATE TABLE dim_circuits (
    circuit_key INTEGER PRIMARY KEY,
    circuit_short_name VARCHAR(100),
    location VARCHAR(200),
    country_name VARCHAR(100),
    country_code VARCHAR(10),
    wikipedia_circuit_url TEXT,
    station_id VARCHAR(20)  -- Meteostat weather station ID
);

COMMENT ON TABLE dim_circuits IS 'F1 circuits reference dimension';
COMMENT ON COLUMN dim_circuits.circuit_key IS 'OpenF1 circuit unique identifier';
COMMENT ON COLUMN dim_circuits.station_id IS 'Meteostat weather station mapped to this circuit';

-- ===========================================================================
-- DIMENSION: dim_drivers
-- ===========================================================================
-- Driver reference data (from OpenF1 drivers endpoint)
-- One row per unique driver_number
-- ===========================================================================

CREATE TABLE dim_drivers (
    driver_number INTEGER PRIMARY KEY,
    full_name VARCHAR(200),
    broadcast_name VARCHAR(200),
    name_acronym VARCHAR(5),
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    country_code VARCHAR(10),
    team_name VARCHAR(200),
    team_colour VARCHAR(10),
    headshot_url TEXT
);

COMMENT ON TABLE dim_drivers IS 'F1 drivers reference dimension';
COMMENT ON COLUMN dim_drivers.driver_number IS 'Driver racing number (primary key)';
COMMENT ON COLUMN dim_drivers.name_acronym IS 'Three-letter driver code (e.g., VER, HAM)';

-- ===========================================================================
-- DIMENSION: dim_sessions
-- ===========================================================================
-- Session context (Race sessions only for this MVP)
-- Links to circuits via circuit_key
-- ===========================================================================

CREATE TABLE dim_sessions (
    session_key INTEGER PRIMARY KEY,
    meeting_key INTEGER NOT NULL,
    year INTEGER NOT NULL,
    session_name VARCHAR(100),
    session_type VARCHAR(50),
    date_start TIMESTAMP WITH TIME ZONE,
    date_end TIMESTAMP WITH TIME ZONE,
    gmt_offset INTERVAL,
    circuit_key INTEGER REFERENCES dim_circuits(circuit_key)
);

COMMENT ON TABLE dim_sessions IS 'F1 sessions reference dimension (Race only)';
COMMENT ON COLUMN dim_sessions.session_key IS 'OpenF1 session unique identifier';
COMMENT ON COLUMN dim_sessions.meeting_key IS 'OpenF1 meeting (Grand Prix event) identifier';

CREATE INDEX idx_sessions_year ON dim_sessions(year);
CREATE INDEX idx_sessions_circuit ON dim_sessions(circuit_key);

-- ===========================================================================
-- FACT: fact_laps
-- ===========================================================================
-- Central fact table at lap-level granularity
-- Contains:
-- - Identifiers (meeting, session, driver, lap)
-- - Sport features (speeds, sector durations)
-- - Weather features (temp, wind, pressure, etc.)
-- - Target variable (lap_duration)
-- ===========================================================================

CREATE TABLE fact_laps (
    -- Primary key (composite)
    meeting_key INTEGER NOT NULL,
    session_key INTEGER NOT NULL REFERENCES dim_sessions(session_key),
    driver_number INTEGER NOT NULL REFERENCES dim_drivers(driver_number),
    lap_number INTEGER NOT NULL,

    -- Context metadata
    year INTEGER NOT NULL,
    circuit_key INTEGER NOT NULL REFERENCES dim_circuits(circuit_key),
    lap_hour_utc TIMESTAMP WITH TIME ZONE,

    -- Sport features (from OpenF1 laps)
    st_speed NUMERIC(10, 2),  -- Speed trap
    i1_speed NUMERIC(10, 2),  -- Intermediate 1 speed
    i2_speed NUMERIC(10, 2),  -- Intermediate 2 speed
    duration_sector_1 NUMERIC(10, 3),
    duration_sector_2 NUMERIC(10, 3),
    duration_sector_3 NUMERIC(10, 3),

    -- Weather features (from Meteostat hourly)
    temp NUMERIC(10, 2),      -- Temperature (C)
    rhum NUMERIC(10, 2),      -- Relative humidity (%)
    pres NUMERIC(10, 2),      -- Atmospheric pressure (hPa)
    wspd NUMERIC(10, 2),      -- Wind speed (km/h)
    wdir NUMERIC(10, 2),      -- Wind direction (degrees)
    prcp NUMERIC(10, 2),      -- Precipitation (mm)
    cldc NUMERIC(10, 2),      -- Cloud cover (%)

    -- Target variable
    lap_duration NUMERIC(10, 3) NOT NULL,  -- Lap time in seconds

    -- Metadata
    source_file VARCHAR(200),  -- Original CSV file name for traceability

    -- Primary key constraint
    CONSTRAINT pk_fact_laps PRIMARY KEY (meeting_key, session_key, driver_number, lap_number)
);

COMMENT ON TABLE fact_laps IS 'F1 lap-level fact table with sport and weather features';
COMMENT ON COLUMN fact_laps.lap_duration IS 'Target variable: lap time in seconds';
COMMENT ON COLUMN fact_laps.lap_hour_utc IS 'Lap timestamp (hour granularity UTC) used for weather join';

-- Indexes for query performance
CREATE INDEX idx_laps_session ON fact_laps(session_key);
CREATE INDEX idx_laps_driver ON fact_laps(driver_number);
CREATE INDEX idx_laps_circuit ON fact_laps(circuit_key);
CREATE INDEX idx_laps_year ON fact_laps(year);
CREATE INDEX idx_laps_hour ON fact_laps(lap_hour_utc);

-- ===========================================================================
-- QUALITY CHECKS
-- ===========================================================================
-- Simple constraints to enforce data quality at DB level
-- ===========================================================================

-- Lap duration must be positive (relaxed for safety car/slow laps)
ALTER TABLE fact_laps ADD CONSTRAINT chk_lap_duration_positive
    CHECK (lap_duration > 0 AND lap_duration < 3600);  -- Max 1 hour (extremely permissive)

-- Year must be reasonable
ALTER TABLE dim_sessions ADD CONSTRAINT chk_year_valid
    CHECK (year >= 2020 AND year <= 2030);

-- Driver number must be positive
ALTER TABLE dim_drivers ADD CONSTRAINT chk_driver_number_positive
    CHECK (driver_number > 0);

-- ===========================================================================
-- END OF SCHEMA
-- ===========================================================================
