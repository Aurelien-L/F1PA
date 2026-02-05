"""
F1PA API - Database Service

Service for accessing F1PA PostgreSQL database.
Supports both direct SQLAlchemy connection and Docker exec fallback for Windows.
"""
import subprocess
import json
import sys
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError


# Cache for circuit typical max_lap (circuits have stable race lengths)
_circuit_typical_max_lap_cache: Dict[int, int] = {}


class DBService:
    """Service for database access."""

    def __init__(self):
        self.engine: Optional[Engine] = None
        self._initialized = False
        self._use_docker = False  # Fallback mode for Windows
        self._container = "f1pa_postgres"

    def connect(self, database_url: str) -> bool:
        """
        Connect to the PostgreSQL database.

        Tries SQLAlchemy first, falls back to docker exec on Windows if needed.
        """
        # Try SQLAlchemy connection first
        try:
            self.engine = create_engine(database_url, pool_pre_ping=True)
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            self._initialized = True
            self._use_docker = False
            return True
        except Exception as e:
            print(f"SQLAlchemy connection failed: {e}")

        # Fallback: try docker exec (for Windows)
        if sys.platform == 'win32':
            try:
                result = self._docker_sql("SELECT 1 AS test")
                if result:
                    print("Using Docker exec fallback for database access")
                    self._initialized = True
                    self._use_docker = True
                    return True
            except Exception as e:
                print(f"Docker exec fallback also failed: {e}")

        self._initialized = False
        return False

    def _docker_sql(self, query: str) -> List[Dict[str, Any]]:
        """Execute SQL via docker exec and return results as list of dicts."""
        # Use JSON output format for easy parsing
        cmd = [
            "docker", "exec", "-i", self._container,
            "psql", "-U", "f1pa", "-d", "f1pa_db",
            "-t", "-A", "-F", ",",  # Tuples only, unaligned, comma separator
            "-c", query
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"SQL failed: {result.stderr}")
        return result.stdout.strip()

    def _docker_query(self, query: str, columns: List[str]) -> List[Dict[str, Any]]:
        """Execute query via docker and parse results into list of dicts."""
        output = self._docker_sql(query)
        if not output:
            return []

        rows = []
        for line in output.split('\n'):
            if line.strip():
                values = line.split(',')
                if len(values) == len(columns):
                    row = {}
                    for col, val in zip(columns, values):
                        # Convert types
                        if val == '' or val == 'NULL':
                            row[col] = None
                        elif col in ('circuit_key', 'driver_number', 'session_key', 'meeting_key',
                                     'lap_number', 'year', 'total_laps', 'total_circuits',
                                     'total_drivers', 'total_sessions'):
                            row[col] = int(val) if val else None
                        elif col in ('lap_duration', 'st_speed', 'i1_speed', 'i2_speed',
                                     'duration_sector_1', 'duration_sector_2', 'duration_sector_3',
                                     'temp', 'rhum', 'pres', 'wspd', 'wdir', 'prcp', 'cldc'):
                            row[col] = float(val) if val else None
                        else:
                            row[col] = val
                    rows.append(row)
        return rows

    def _docker_scalar(self, query: str) -> Any:
        """Execute query via docker and return single scalar value."""
        output = self._docker_sql(query)
        if output:
            val = output.strip()
            try:
                return int(val)
            except ValueError:
                try:
                    return float(val)
                except ValueError:
                    return val
        return None

    def is_ready(self) -> bool:
        """Check if database is connected."""
        return self._initialized

    @contextmanager
    def get_connection(self):
        """Get a database connection context (SQLAlchemy mode only)."""
        if not self.is_ready():
            raise RuntimeError("Database not connected. Call connect() first.")
        if self._use_docker:
            raise RuntimeError("Using Docker mode - use specific query methods instead")
        with self.engine.connect() as conn:
            yield conn

    # =========================================================================
    # CIRCUITS
    # =========================================================================

    def get_circuits(self) -> List[Dict[str, Any]]:
        """Get all circuits."""
        query = """
            SELECT circuit_key, circuit_short_name, location, country_name, country_code
            FROM dim_circuits
            ORDER BY circuit_short_name
        """
        if self._use_docker:
            return self._docker_query(query, ['circuit_key', 'circuit_short_name', 'location', 'country_name', 'country_code'])

        with self.get_connection() as conn:
            result = conn.execute(text(query))
            return [dict(row._mapping) for row in result]

    def get_circuit(self, circuit_key: int) -> Optional[Dict[str, Any]]:
        """Get a specific circuit by key."""
        query = f"""
            SELECT circuit_key, circuit_short_name, location, country_name, country_code
            FROM dim_circuits
            WHERE circuit_key = {circuit_key}
        """
        if self._use_docker:
            rows = self._docker_query(query, ['circuit_key', 'circuit_short_name', 'location', 'country_name', 'country_code'])
            return rows[0] if rows else None

        with self.get_connection() as conn:
            result = conn.execute(text(query.replace(str(circuit_key), ':circuit_key')), {"circuit_key": circuit_key})
            row = result.fetchone()
            return dict(row._mapping) if row else None

    # =========================================================================
    # DRIVERS
    # =========================================================================

    def get_drivers(self) -> List[Dict[str, Any]]:
        """Get all drivers."""
        query = """
            SELECT driver_number, full_name, name_acronym, team_name, team_colour, country_code, headshot_url
            FROM dim_drivers
            ORDER BY full_name
        """
        if self._use_docker:
            return self._docker_query(query, ['driver_number', 'full_name', 'name_acronym', 'team_name', 'team_colour', 'country_code', 'headshot_url'])

        with self.get_connection() as conn:
            result = conn.execute(text(query))
            return [dict(row._mapping) for row in result]

    def get_driver(self, driver_number: int) -> Optional[Dict[str, Any]]:
        """Get a specific driver by number."""
        query = f"""
            SELECT driver_number, full_name, name_acronym, team_name, team_colour, country_code, headshot_url
            FROM dim_drivers
            WHERE driver_number = {driver_number}
        """
        if self._use_docker:
            rows = self._docker_query(query, ['driver_number', 'full_name', 'name_acronym', 'team_name', 'team_colour', 'country_code', 'headshot_url'])
            return rows[0] if rows else None

        with self.get_connection() as conn:
            result = conn.execute(text(query.replace(str(driver_number), ':driver_number')), {"driver_number": driver_number})
            row = result.fetchone()
            return dict(row._mapping) if row else None

    # =========================================================================
    # SESSIONS
    # =========================================================================

    def get_sessions(
        self,
        year: Optional[int] = None,
        circuit_key: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get sessions with optional filters."""
        conditions = []
        if year is not None:
            conditions.append(f"year = {year}")
        if circuit_key is not None:
            conditions.append(f"circuit_key = {circuit_key}")

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"""
            SELECT session_key, meeting_key, year, session_name, session_type,
                   circuit_key, date_start::text
            FROM dim_sessions
            WHERE {where_clause}
            ORDER BY date_start DESC
            LIMIT {limit}
        """

        if self._use_docker:
            return self._docker_query(query, ['session_key', 'meeting_key', 'year', 'session_name', 'session_type', 'circuit_key', 'date_start'])

        with self.get_connection() as conn:
            result = conn.execute(text(query))
            return [dict(row._mapping) for row in result]

    # =========================================================================
    # LAPS
    # =========================================================================

    def get_laps(
        self,
        year: Optional[int] = None,
        circuit_key: Optional[int] = None,
        driver_number: Optional[int] = None,
        session_key: Optional[int] = None,
        page: int = 1,
        page_size: int = 100
    ) -> Dict[str, Any]:
        """Get laps with optional filters and pagination."""
        conditions = []
        if year is not None:
            conditions.append(f"year = {year}")
        if circuit_key is not None:
            conditions.append(f"circuit_key = {circuit_key}")
        if driver_number is not None:
            conditions.append(f"driver_number = {driver_number}")
        if session_key is not None:
            conditions.append(f"session_key = {session_key}")

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        offset = (page - 1) * page_size

        # Count total
        count_query = f"SELECT COUNT(*) FROM fact_laps WHERE {where_clause}"

        if self._use_docker:
            total = self._docker_scalar(count_query) or 0
        else:
            with self.get_connection() as conn:
                result = conn.execute(text(count_query))
                total = result.scalar()

        # Get paginated data
        columns = ['meeting_key', 'session_key', 'driver_number', 'lap_number', 'year',
                   'circuit_key', 'lap_duration', 'st_speed', 'i1_speed', 'i2_speed',
                   'duration_sector_1', 'duration_sector_2', 'duration_sector_3',
                   'temp', 'rhum', 'pres']

        data_query = f"""
            SELECT {', '.join(columns)}
            FROM fact_laps
            WHERE {where_clause}
            ORDER BY session_key, driver_number, lap_number
            LIMIT {page_size} OFFSET {offset}
        """

        if self._use_docker:
            data = self._docker_query(data_query, columns)
        else:
            with self.get_connection() as conn:
                result = conn.execute(text(data_query))
                data = [dict(row._mapping) for row in result]

        total_pages = (total + page_size - 1) // page_size if total else 0

        return {
            "data": data,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages
        }

    def get_driver_laps(
        self,
        driver_number: int,
        year: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get laps for a specific driver."""
        conditions = [f"driver_number = {driver_number}"]
        if year is not None:
            conditions.append(f"year = {year}")

        where_clause = " AND ".join(conditions)
        columns = ['meeting_key', 'session_key', 'driver_number', 'lap_number', 'year', 'circuit_key',
                   'lap_duration', 'st_speed', 'i1_speed', 'i2_speed',
                   'duration_sector_1', 'duration_sector_2', 'duration_sector_3',
                   'temp', 'rhum', 'pres']

        query = f"""
            SELECT {', '.join(columns)}
            FROM fact_laps
            WHERE {where_clause}
            ORDER BY session_key DESC, lap_number
            LIMIT {limit}
        """

        if self._use_docker:
            return self._docker_query(query, columns)

        with self.get_connection() as conn:
            result = conn.execute(text(query))
            return [dict(row._mapping) for row in result]

    def get_circuit_laps(
        self,
        circuit_key: int,
        year: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get laps for a specific circuit."""
        conditions = [f"circuit_key = {circuit_key}"]
        if year is not None:
            conditions.append(f"year = {year}")

        where_clause = " AND ".join(conditions)
        columns = ['meeting_key', 'session_key', 'driver_number', 'lap_number', 'year',
                   'circuit_key', 'lap_duration', 'st_speed', 'i1_speed', 'i2_speed',
                   'duration_sector_1', 'duration_sector_2', 'duration_sector_3',
                   'temp', 'rhum', 'pres']

        query = f"""
            SELECT {', '.join(columns)}
            FROM fact_laps
            WHERE {where_clause}
            ORDER BY lap_duration ASC
            LIMIT {limit}
        """

        if self._use_docker:
            return self._docker_query(query, columns)

        with self.get_connection() as conn:
            result = conn.execute(text(query))
            return [dict(row._mapping) for row in result]

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get overall dataset statistics."""
        if self._use_docker:
            stats = {
                "total_laps": self._docker_scalar("SELECT COUNT(*) FROM fact_laps"),
                "total_circuits": self._docker_scalar("SELECT COUNT(*) FROM dim_circuits"),
                "total_drivers": self._docker_scalar("SELECT COUNT(*) FROM dim_drivers"),
                "total_sessions": self._docker_scalar("SELECT COUNT(*) FROM dim_sessions"),
            }

            # Get years
            years_output = self._docker_sql("SELECT DISTINCT year FROM fact_laps ORDER BY year")
            stats["years"] = [int(y) for y in years_output.split('\n') if y.strip()]

            # Get date range
            date_output = self._docker_sql("SELECT MIN(date_start)::text, MAX(date_start)::text FROM dim_sessions")
            if date_output:
                parts = date_output.split(',')
                stats["date_range"] = {
                    "min": parts[0] if len(parts) > 0 else None,
                    "max": parts[1] if len(parts) > 1 else None
                }
            else:
                stats["date_range"] = {"min": None, "max": None}

            return stats

        # SQLAlchemy mode
        queries = {
            "total_laps": "SELECT COUNT(*) FROM fact_laps",
            "total_circuits": "SELECT COUNT(*) FROM dim_circuits",
            "total_drivers": "SELECT COUNT(*) FROM dim_drivers",
            "total_sessions": "SELECT COUNT(*) FROM dim_sessions",
        }

        stats = {}
        with self.get_connection() as conn:
            for key, query in queries.items():
                result = conn.execute(text(query))
                stats[key] = result.scalar()

            result = conn.execute(text("SELECT DISTINCT year FROM fact_laps ORDER BY year"))
            stats["years"] = [row[0] for row in result]

            result = conn.execute(text("""
                SELECT MIN(date_start)::text, MAX(date_start)::text
                FROM dim_sessions
            """))
            row = result.fetchone()
            stats["date_range"] = {
                "min": row[0] if row else None,
                "max": row[1] if row else None
            }

        return stats

    def get_circuit_avg_laptime(self, circuit_key: int) -> Optional[float]:
        """Get average lap time for a circuit (for predictions)."""
        query = f"SELECT AVG(lap_duration) FROM fact_laps WHERE circuit_key = {circuit_key}"

        if self._use_docker:
            return self._docker_scalar(query)

        with self.get_connection() as conn:
            result = conn.execute(text(query))
            return result.scalar()

    def get_circuit_typical_max_lap(self, circuit_key: int) -> int:
        """
        Get typical maximum lap number for a circuit (averaged across sessions).

        Uses average of max laps across all sessions for this circuit.
        Falls back to 70 if circuit not found or has no data.

        Results are cached since circuit configurations are stable.
        """
        # Check cache first
        if circuit_key in _circuit_typical_max_lap_cache:
            return _circuit_typical_max_lap_cache[circuit_key]

        # Query: average of max laps per session for this circuit
        query = f"""
            SELECT ROUND(AVG(max_laps))::int
            FROM (
                SELECT MAX(lap_number) as max_laps
                FROM fact_laps
                WHERE circuit_key = {circuit_key}
                GROUP BY session_key
            ) subquery
        """

        if self._use_docker:
            result = self._docker_scalar(query)
            typical_max_lap = int(result) if result else 70
        else:
            with self.get_connection() as conn:
                result = conn.execute(text(query))
                max_lap_value = result.scalar()
                typical_max_lap = int(max_lap_value) if max_lap_value else 70

        # Cache the result
        _circuit_typical_max_lap_cache[circuit_key] = typical_max_lap
        return typical_max_lap


# Global service instance
db_service = DBService()
