set -euo pipefail

# Check PostgreSQL connection
until pg_isready -h localhost -p 5432; do
    echo "Waiting for PostgreSQL..."
    sleep 1
done

: "${BACKEND_STORE_URI:=postgresql://mlflow:mlflow@localhost:5432/mlflow}"
: "${ARTIFACT_ROOT:=./mlartifacts}"
: "${HOST:=0.0.0.0}"
: "${PORT:=5001}"

echo "Launching MLflow server on ${HOST}:${PORT}"
mlflow server \
  --host "${HOST}" \
  --port "${PORT}" \
  --backend-store-uri "${BACKEND_STORE_URI}" \
  --default-artifact-root "${ARTIFACT_ROOT}" \
  --serve-artifacts
