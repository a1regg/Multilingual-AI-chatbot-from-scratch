set -euo pipefail

# Kill existing MLflow processes
echo "Cleaning up existing MLflow processes..."
lsof -ti:5001 | xargs kill -9 2>/dev/null || true

# Make scripts executable
chmod +x setup-db.sh start-mlflow.sh entrypoint.sh

# Function to cleanup background processes
cleanup() {
    echo "Cleaning up..."
    kill $(jobs -p) 2>/dev/null || true
}

# Set trap for cleanup
trap cleanup EXIT

# Setup the database
./setup-db.sh

# Start MLflow server in background
./start-mlflow.sh &

# Wait for MLflow to start
echo "Waiting for MLflow server to start..."
sleep 5

# Run the main application
./entrypoint.sh
