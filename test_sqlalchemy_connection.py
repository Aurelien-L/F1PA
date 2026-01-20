from sqlalchemy import create_engine, text

try:
    print("Testing SQLAlchemy connection...", flush=True)

    conn_str = "postgresql://f1pa:f1pa@localhost:5432/f1pa_db"
    print(f"Connection string: {conn_str}", flush=True)

    engine = create_engine(conn_str, echo=True)

    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1 AS test"))
        row = result.fetchone()
        print(f"Query result: {row[0]}", flush=True)
        print("OK: SQLAlchemy connection successful!", flush=True)

except Exception as e:
    print(f"ERROR: {type(e).__name__}", flush=True)
    print(f"Message: {str(e)}", flush=True)
    import traceback
    traceback.print_exc()
