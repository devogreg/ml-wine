from __future__ import annotations

import os
from datetime import datetime

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

COMPOSE_NETWORK = os.environ.get("COMPOSE_NETWORK", "mlops_default")
HOST_DATA_DIR = os.environ.get("HOST_DATA_DIR", "/home/gregorius/projects/ml-wine/data")

with DAG(
    dag_id="wine_training_pipeline",
    start_date=datetime(2025, 11, 1),
    schedule=None,
    catchup=False,
    tags=["wine", "mlflow", "docker"],
) as dag:
    train = DockerOperator(
        task_id="train_model_both",
        image="wineclf:latest",
        api_version="auto",
        auto_remove=True,
        command="python src/wineclf/train.py --model both",
        docker_url="unix://var/run/docker.sock",
        network_mode=COMPOSE_NETWORK,
        environment={
            "MLFLOW_TRACKING_URI": "http://host.docker.internal:5000",
            "GIT_PYTHON_REFRESH": "quiet",
        },
        mounts=[
            Mount(source=HOST_DATA_DIR, target="/app/data", type="bind", read_only=False)
        ],
    )
