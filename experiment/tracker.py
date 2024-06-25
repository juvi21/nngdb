import mlflow
import torch
from typing import Dict, Any

class ExperimentTracker:
    def __init__(self, experiment_name: str):
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str):
        mlflow.start_run(run_name=run_name)

    def log_params(self, params: Dict[str, Any]):
        mlflow.log_params(params)

    def log_metric(self, key: str, value: float):
        mlflow.log_metric(key, value)

    def log_model(self, model: torch.nn.Module, artifact_path: str):
        mlflow.pytorch.log_model(model, artifact_path)

    def end_run(self):
        mlflow.end_run()