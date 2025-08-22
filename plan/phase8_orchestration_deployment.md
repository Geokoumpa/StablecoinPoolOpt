# Phase 8: Orchestration & Deployment

This phase focuses on orchestrating the entire data pipeline, containerizing components, and deploying them to GCP Cloud Run using Terraform.

## Detailed Tasks:

### 8.1 Implement a Central Orchestration Mechanism
- Create a central orchestration script (e.g., `main_pipeline.py`) or define a GCP Workflow to coordinate the execution of individual pipeline scripts in the defined sequence.
- Ensure proper data dependencies are met between scripts.
- Implement robust error handling, logging, and alerting mechanisms for the entire pipeline.
- Consider using a workflow management system like Apache Airflow (if not using GCP Workflows) for more complex scheduling and monitoring.

### 8.2 Set up GCP Cloud Scheduler for Daily Pipeline Runs
- Configure GCP Cloud Scheduler to trigger the central orchestration mechanism (e.g., a Cloud Run job or Workflow) daily at 00:01 EST.
- Define the target for the scheduler job (e.g., HTTP target for a Cloud Run service, or Pub/Sub topic triggering a Workflow).

### 8.3 Containerize Pipeline Components using Docker
- Create Dockerfiles for each modular Python script (or logical groups of scripts) to containerize them.
- Ensure all necessary dependencies (Python libraries, OS packages) are included in the Docker images.
- Optimize Docker images for size and build time.
- Push Docker images to Google Container Registry (GCR) or Artifact Registry.

### 8.4 Define and Deploy GCP Cloud Run Jobs using Terraform
- Write Terraform configurations to define and deploy the containerized pipeline components as GCP Cloud Run jobs.
- Configure Cloud Run jobs with appropriate memory, CPU, and timeout settings.
- Define environment variables for API keys, database connection strings, and other configurable parameters.
- Implement secrets management (e.g., using GCP Secret Manager) for sensitive information.
- Define IAM roles and permissions for Cloud Run service accounts to access necessary GCP resources (Cloud SQL, GCR, Cloud Scheduler, etc.).
- Use Terraform to manage the entire infrastructure setup, ensuring reproducibility and version control.