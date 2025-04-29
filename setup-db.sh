set -euo pipefail

echo "Starting PostgreSQL setup..."

# Create postgres superuser if doesn't exist
createuser -s postgres 2>/dev/null || true

# Create mlflow role and database
psql -U postgres -d postgres <<EOF
    DO \$\$
    BEGIN
        -- Create role if not exists
        IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'mlflow') THEN
            CREATE ROLE mlflow WITH LOGIN PASSWORD 'mlflow';
            RAISE NOTICE 'Created mlflow role';
        END IF;

        -- Create database if not exists
        IF NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow') THEN
            CREATE DATABASE mlflow OWNER mlflow;
            RAISE NOTICE 'Created mlflow database';
        END IF;
    END
    \$\$;
EOF

echo "PostgreSQL setup completed"
