#!/bin/bash
# Run LOAD pipeline from inside a Docker container with network access to PostgreSQL

set -e

echo "Running F1PA LOAD pipeline in Docker container..."

# Run Python in temporary container connected to same network as postgres
docker run --rm \
    --network f1pa_default \
    -v "$(pwd)/../..:/app" \
    -w /app/etl/load \
    python:3.10-slim \
    bash -c "
        pip install --quiet pandas sqlalchemy psycopg2-binary &&
        python run_load_all.py --host f1pa_postgres
    "

echo "LOAD pipeline complete!"
