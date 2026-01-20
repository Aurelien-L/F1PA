import psycopg2
import sys

try:
    print("Attempting to connect to PostgreSQL...")
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="f1pa_db",
        user="f1pa",
        password="f1pa",
        connect_timeout=10
    )
    print("OK: Connection successful!", flush=True)
    cursor = conn.cursor()
    cursor.execute("SELECT version();")
    version = cursor.fetchone()
    print(f"PostgreSQL version: {version[0]}", flush=True)
    cursor.close()
    conn.close()
except Exception as e:
    print("ERROR: Connection failed!", flush=True)
    print(f"Error type: {type(e).__name__}")
    print(f"Error details: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
