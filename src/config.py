import os

import mlflow
from dotenv import load_dotenv

load_dotenv()
# Read directly from environment - Docker Compose MUST provide this
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "ChatBot_Experiment")

print(f"Attempting to connect to MLflow at: {MLFLOW_TRACKING_URI}")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def get_or_create_experiment_id(experiment_name):
    """
    Gets the experiment ID for a given name,
    creating the experiment if it doesn't exist.
    """
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found. Creating a new one.")
        experiment_id = client.create_experiment(experiment_name)
        print(f"Created experiment '{experiment_name}' with ID: {experiment_id}")
        return experiment_id
    else:
        print(
            f"Found existing experiment '{experiment_name}' "
            f"with ID: {experiment.experiment_id}"
        )
        return experiment.experiment_id


def get_latest_run_id(experiment_id):
    """Gets the ID of the most recent run within a given experiment."""
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["start_time DESC"],  # Order by start time, descending
        max_results=1,  # Get only the latest run
    )
    if not runs:
        raise ValueError(f"No runs found for experiment ID: {experiment_id}")
    latest_run_id = runs[0].info.run_id
    print(f"Found latest run ID: {latest_run_id} for experiment ID: {experiment_id}")
    return latest_run_id


def start_run(**kwargs):
    """
    Wrapper around mlflow.start_run that ensures the right
    experiment is always used, creating it if necessary.
    """
    experiment_id = get_or_create_experiment_id(MLFLOW_EXPERIMENT_NAME)
    return mlflow.start_run(experiment_id=experiment_id, **kwargs)
