# ChatBot Bob Project Documentation

## 1. Overview

ChatBot Bob is an AI-powered chatbot designed for normal conversations. It utilizes machine learning models (PyTorch) for intent recognition and response generation, supporting both English and Ukrainian languages simultaneously. The project integrates with MLflow for experiment tracking and model management, and uses Docker for containerized deployment.

## 2. Project Structure

```
.
├── data/                     # Training and vocabulary data
│   ├── intents.json          # Core intents and responses data (EN/UK)
│   ├── tag_to_idx.txt        # Mapping from intent tags to indices
│   ├── vocab_en.txt          # English vocabulary file
│   └── vocab_uk.txt          # Ukrainian vocabulary file
├── mlartifacts/              # MLflow artifacts (models, etc.) - auto-generated, volume-mounted
├── src/                      # Source code for the chatbot
│   ├── config.py             # Configuration (MLflow setup, env vars)
│   ├── data_preprocessing.py # Scripts for preparing data
│   ├── gui.py                # Graphical User Interface code (if applicable)
│   ├── main.py               # Main application entry point (likely used by entrypoint.sh)
│   ├── model.py              # Chatbot model definition (PyTorch)
│   └── training.py           # Model training script
├── tests/                    # Unit and integration tests
│   ├── test_config.py
│   └── test_data_preprocessing.py
├── .env                      # Environment variables (MLFLOW_TRACKING_URI, etc. - **Create this!**)
├── .gitignore                # Specifies intentionally untracked files (like mlruns, .env)
├── .pre-commit-config.yaml   # Configuration for pre-commit hooks (linting, formatting), make sure you have specified the correct python version.
├── docker-compose.yaml       # Docker Compose configuration for services (app, mlflow, db)
├── Dockerfile                # Dockerfile for the main application ('app' service)
├── Dockerfile.mlflow         # Dockerfile for the MLflow server ('mlflow' service)
├── entrypoint.sh             # Entry script run inside the app container / locally via run.sh
├── pytest.ini                # Pytest configuration
├── README.md                 # Documentation
├── requirements.txt          # Python dependencies
├── run.sh                    # Script to run the project locally (sets up DB, MLflow, runs app)
├── setup-db.sh               # Script to set up the database (used only by run.sh for local setup)
└── start-mlflow.sh           # Script to start the MLflow server locally (used by run.sh)
```

## 3. Prerequisites

*   **Docker & Docker Compose:** Required for the default containerized setup. ([Install Docker](https://docs.docker.com/get-docker/))
*   **Python:** Required for local execution (Version 3.10+ recommended).
*   **PostgreSQL:** Required for local execution (`run.sh` uses `setup-db.sh` and `start-mlflow.sh` which depend on a local PostgreSQL server). Ensure it's installed and running if you don't want to use Docker.
*   **Git:** For cloning the repository.

## 4. Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/a1regg/Multilingual-AI-chatbot.git
    cd Multilingual-AI-chatbot
    ```

2.  **Create Environment File (`.env`):**
    This project relies heavily on environment variables defined in a `.env` file in the project root. Create this file if it doesn't exist. **Do not commit this file to Git** (it should be in your `.gitignore`).

    ```dotenv
    # .env file content

    # --- MLflow Configuration ---
    # For Docker (default): Use the service name 'mlflow' and its internal port
    MLFLOW_TRACKING_URI=http://mlflow:5000

    # For Local execution (if running MLflow locally via run.sh):
    # MLFLOW_TRACKING_URI=http://127.0.0.1:5001

    MLFLOW_EXPERIMENT_NAME=ChatBot_Experiment

    # --- Docker Compose Port Mapping ---
    # Port on your host machine to access MLflow UI
    MLFLOW_HOST_PORT=5001
    # Port inside the MLflow container (should match Dockerfile.mlflow expose/cmd)
    MLFLOW_CONTAINER_PORT=5000

    # Add any other necessary environment variables here
    # e.g., database credentials if not hardcoded (though defaults are in docker-compose.yaml)
    ```
    **Important:** Choose the correct `MLFLOW_TRACKING_URI` based on how you intend to run the project (Docker vs. Local). The default is set up for Docker.

## 5. Running the Project

There are two primary ways to run the chatbot: using Docker (recommended for consistency) or running locally. Ensure your `.env` file is configured correctly for the chosen method, especially the `MLFLOW_TRACKING_URI`.

### 5.1 Docker

Docker and Docker Compose handle the environment, dependencies, and service orchestration (application, MLflow server, PostgreSQL database).

**Option 1: All-in-One (Foreground)**

This command builds the necessary Docker images and starts all services (`app`, `mlflow`, `mlflow-db`) defined in `docker-compose.yaml`. The logs from all services will be streamed to your terminal.

```bash
docker-compose up --build
```

*   The `--build` flag ensures images are rebuilt if Dockerfiles or context change. Omit it for faster startups if no changes were made.
*   The application (`app` service) should start after MLflow is ready.
*   Access the MLflow UI in your browser at `http://localhost:5001` (or the `MLFLOW_HOST_PORT` defined in your `.env`).
*   To interact with the chatbot, you might need to attach to the running `app` container:
    ```bash
    # Find the container name (i.e., multilingualaichatbot-app-1)
    docker ps
    # Attach to it
    docker attach multilingualaichatbot-app-1
    ```
*   Press `Ctrl+C` in the terminal where `docker-compose up` is running to stop all services.

**Option 2: Background Services + Interactive App**

This approach starts the database and MLflow server in the background, then runs the main application interactively in the foreground, attached to your terminal.

1.  **Start Background Services:**
    ```bash
    # Start database and MLflow detached (-d)
    docker-compose up -d mlflow-db mlflow
    ```
    *   Wait a few moments for the services to initialize.
    *   Check status: `docker-compose ps`
    *   Access MLflow UI: `http://localhost:5001` (or your `MLFLOW_HOST_PORT`).

2.  **Run the Application Interactively:**
    ```bash
    # Build the app image if needed, then run the app service
    # --rm removes the container on exit
    docker-compose run --rm app
    ```
    *   This executes the `entrypoint.sh` script inside a new `app` container.
    *   Your terminal is directly connected to the application's standard input/output.
    *   When the application finishes or you exit (e.g., `Ctrl+C`), the container is automatically removed.
    *   The background services (`mlflow`, `mlflow-db`) remain running until stopped (`docker-compose down`).

### 5.2 Local Execution

This method runs the application directly on your host machine using your local Python environment.

1.  **Configure `.env` for Local:**
    Make sure `MLFLOW_TRACKING_URI` in your `.env` file points to the local MLflow address:
    ```dotenv
    # .env (for local execution)
    MLFLOW_TRACKING_URI=http://127.0.0.1:5001
    # Other variables as needed...
    ```

2.  **Install Dependencies:**
    It's highly recommended to use a Python virtual environment.
    ```bash
    # Create virtual environment (if you haven't)
    python3 -m venv .venv
    # Activate it
    source .venv/bin/activate  # Linux/macOS
    # .\.venv\Scripts\activate  # Windows

    # Install requirements
    pip install -r requirements.txt
    ```

3.  **Make Scripts Executable (Linux/macOS):**
    Ensure the shell scripts have execute permissions.
    ```bash
    chmod +x run.sh # run.sh ensures all necessary files have execute permissions
    ```

4.  **Run the Application:**
    Execute the main run script:
    ```bash
    ./run.sh
    ```
    This script performs the following steps:
    *   Cleans up any previous MLflow processes running on port 5001.
    *   Ensures necessary scripts are executable.
    *   Sets up the required PostgreSQL user and database using `setup-db.sh`. **Requires a running local PostgreSQL instance.**
    *   Starts the MLflow server in the background using `start-mlflow.sh`, typically listening on `http://127.0.0.1:5001` and configured to use the local PostgreSQL database as its backend.
    *   Waits briefly for MLflow to initialize.
    *   Runs the main application logic via `entrypoint.sh`.

    * Also you can run PyQT6 gui instead of console chat.
    * To do this, you need first install PyQt6:
    ```
    python3 -m pip install PyQt6    # Linux/MacOS
    python -m pip install PyQt6     # Windows
    ```
    * Then run python3 src/gui.py` or
    ```
    cd src
    python3 gui.py
    ```
## 6. MLflow Integration

*   **Purpose:** MLflow is used for managing the machine learning lifecycle, including:
    *   Tracking experiments (parameters, metrics, code versions).
    *   Logging and storing model artifacts (trained models, vocabulary files).
    *   Packaging models for deployment.
*   **Configuration:** Managed in `src/config.py`, which reads the `MLFLOW_TRACKING_URI` and `MLFLOW_EXPERIMENT_NAME` from environment variables (via `.env`).
*   **Server:**
    *   **Docker:** Runs as the `mlflow` service, accessible via `http://mlflow:5000` internally and `http://localhost:5001` (default) externally. Uses PostgreSQL (`mlflow-db` service) as a backend store and mounts `./mlartifacts` for artifact storage.
    *   **Local:** Started by `run.sh` via `start-mlflow.sh`, listening on `http://127.0.0.1:5001`. Uses the local PostgreSQL database (set up by `setup-db.sh`) as the backend store and `./mlartifacts` for artifact storage. The `mlruns/` directory is **not** used when PostgreSQL is the backend.

## 7. Data

The `data/` directory contains essential files for training and running the chatbot:
*   `intents.json`: The core dataset defining chatbot intents, example user patterns (in English and Ukrainian), and corresponding responses.
*   `vocab_en.txt`, `vocab_uk.txt`: Vocabulary lists generated during preprocessing for each language.
*   `tag_to_idx.txt`: A mapping file converting intent tags (strings) into numerical indices used by the model.

## 8. Testing

Tests are located in the `tests/` directory and use the `pytest` framework.

1.  **Ensure `pytest` is installed:** It should be included in `requirements.txt`.
2.  **Navigate to the project root directory.**
3.  **Run tests:**
    ```bash
    pytest
    ```
    *   The tests are designed to be self-contained or use mocking (`unittest.mock`) for external services like MLflow. They do not require a running MLflow instance or specific environment variables to be set.

## 9. Code Quality

This project uses `pre-commit` hooks configured in `.pre-commit-config.yaml` to automatically check and format code before commits. This includes tools like `black`, `isort`, `flake8`, and `mypy` to ensure consistent style, import sorting, linting, and static type checking.

*   **Setup (first time):**
    ```bash
    pip install pre-commit
    pre-commit install --hook-type pre-commit --hook-type pre-push
    ```
*   **Usage:** Hooks run automatically when you `git commit`. If they modify files, you'll need to `git add` the changes and commit again.
