from unittest.mock import MagicMock

import pytest

from src import config


# Mock experiment and run info objects similar to what MlflowClient returns
class MockExperiment:
    def __init__(self, experiment_id, name):
        self.experiment_id = experiment_id
        self.name = name


class MockRunInfo:
    def __init__(self, run_id):
        self.run_id = run_id


class MockRun:
    def __init__(self, run_id):
        self.info = MockRunInfo(run_id)


@pytest.fixture
def mock_mlflow_client(mocker):
    """Fixture to mock the MlflowClient."""
    mock_client = MagicMock()
    # Patch the MlflowClient constructor to return our mock
    mocker.patch("mlflow.tracking.MlflowClient", return_value=mock_client)
    return mock_client


def test_get_or_create_experiment_id_exists(mock_mlflow_client):
    """Test getting an existing experiment ID."""
    experiment_name = "Test_Experiment"
    expected_id = "123"
    mock_mlflow_client.get_experiment_by_name.return_value = MockExperiment(
        expected_id, experiment_name
    )

    actual_id = config.get_or_create_experiment_id(experiment_name)

    mock_mlflow_client.get_experiment_by_name.assert_called_once_with(experiment_name)
    mock_mlflow_client.create_experiment.assert_not_called()
    assert actual_id == expected_id


def test_get_or_create_experiment_id_creates(mock_mlflow_client):
    """Test creating a new experiment ID."""
    experiment_name = "New_Experiment"
    expected_id = "456"
    # Simulate experiment not found
    mock_mlflow_client.get_experiment_by_name.return_value = None
    # Simulate create_experiment returning the new ID
    mock_mlflow_client.create_experiment.return_value = expected_id

    actual_id = config.get_or_create_experiment_id(experiment_name)

    mock_mlflow_client.get_experiment_by_name.assert_called_once_with(experiment_name)
    mock_mlflow_client.create_experiment.assert_called_once_with(experiment_name)
    assert actual_id == expected_id


def test_get_latest_run_id_found(mock_mlflow_client):
    """Test getting the latest run ID when runs exist."""
    experiment_id = "123"
    expected_run_id = "run_abc"
    mock_runs = [
        MockRun(expected_run_id),
        MockRun("run_xyz"),
    ]  # search_runs returns a list
    mock_mlflow_client.search_runs.return_value = mock_runs

    actual_run_id = config.get_latest_run_id(experiment_id)

    mock_mlflow_client.search_runs.assert_called_once_with(
        experiment_ids=[experiment_id], order_by=["start_time DESC"], max_results=1
    )
    assert actual_run_id == expected_run_id


def test_get_latest_run_id_not_found(mock_mlflow_client):
    """Test getting the latest run ID when no runs exist."""
    experiment_id = "123"
    # Simulate no runs found
    mock_mlflow_client.search_runs.return_value = []

    with pytest.raises(
        ValueError, match=f"No runs found for experiment ID: {experiment_id}"
    ):
        config.get_latest_run_id(experiment_id)

    mock_mlflow_client.search_runs.assert_called_once_with(
        experiment_ids=[experiment_id], order_by=["start_time DESC"], max_results=1
    )
