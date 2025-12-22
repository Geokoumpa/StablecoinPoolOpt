# Local Orchestration & Parity

This project uses a "Poor Man's Parity" approach to ensure that the local development environment matches the production GCP environment as closely as possible.

## Architecture

-   **Production**: GCP Workflows (orchestrator) -> Cloud Run Jobs (execution).
-   **Local**: `local_orchestrator.py` (orchestrator) -> Docker Container (execution).

### Components

1.  **`docker-compose.yml`**: Defines the local stack.
    -   `pipeline-runner`: A container running the same image as production (based on `Dockerfile.lightweight`). It mounts the local source code, so changes are reflected immediately.
    -   `db`: Local PostgreSQL instance.
2.  **`local_orchestrator.py`**: A custom script that parses the production `workflow.yaml`.
    -   It iterates through the steps defined in the YAML.
    -   For each `run_cloud_run_job`, it runs the corresponding command inside the `pipeline-runner` container using `docker compose exec`.
3.  **`run_local.sh`**: Helper script to start the stack and run the pipeline.

## How to Run Locally

1.  **Start the Environment**:
    ```bash
    ./run_local.sh
    ```
    This script will:
    -   Start `docker compose` if not running.
    -   Install `PyYAML` locally if missing.
    -   Run `local_orchestrator.py`.

## Adding New Steps

1.  **Modify `workflow.yaml`**: Add your new step using the standard GCP Workflows syntax.
    ```yaml
    - run_my_new_step:
        call: run_cloud_run_job
        args:
          step_key: "my_new_script_name" # mapped to script filename
    ```
2.  **Run Locally**: Just run `./run_local.sh`. The orchestrator will pick up the new step and try to execute `python pipeline_runner.py my_new_script_name` inside the container.

## Extensibility (e.g., Spark/Dataproc)

Currently, the `local_orchestrator.py` handles `run_cloud_run_job` calls by executing them in the `pipeline-runner` container.

To support **Dataproc (Spark)** jobs:

1.  Add a Spark container to `docker-compose.yml` (e.g., `bitnami/spark`).
2.  Update `local_orchestrator.py`:
    -   Detect `call: run_dataproc_batch_job`.
    -   Extract `main_python_file` from args.
    -   Execute a `docker compose exec spark-container spark-submit ...` command instead of the default python command.
