resource "google_workflows_workflow" "main" {
  name            = "defi-pipeline-workflow"
  region          = var.region
  service_account = google_service_account.workflow_sa.email

  source_contents = file("${path.module}/../workflow.yaml")

  user_env_vars = {
    PROJECT_ID = var.project_id
    REGION     = var.region
  }
}

# =============================================================================
# Cloud Run Permissions for Workflow
# =============================================================================

resource "google_project_iam_member" "workflow_cloud_run_invoker" {
  project = var.project_id
  role    = "roles/run.invoker"
  member  = "serviceAccount:${google_service_account.workflow_sa.email}"
}

resource "google_project_iam_member" "workflow_logs_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.workflow_sa.email}"
}

resource "google_project_iam_member" "workflow_cloud_run_viewer" {
  project = var.project_id
  role    = "roles/run.viewer"
  member  = "serviceAccount:${google_service_account.workflow_sa.email}"
}

# =============================================================================
# Dataproc Serverless Permissions for Workflow
# =============================================================================

# Grant Dataproc Editor role for creating and managing batch jobs
resource "google_project_iam_member" "workflow_dataproc_editor" {
  project = var.project_id
  role    = "roles/dataproc.editor"
  member  = "serviceAccount:${google_service_account.workflow_sa.email}"
}

# Grant Storage Object Viewer for reading Spark scripts from GCS
resource "google_project_iam_member" "workflow_storage_viewer" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.workflow_sa.email}"
}

# Grant Service Account User to allow workflow to act as the Dataproc SA
resource "google_service_account_iam_member" "workflow_can_use_dataproc_sa" {
  service_account_id = google_service_account.dataproc_sa.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.workflow_sa.email}"
}

# =============================================================================
# Cloud Scheduler Job
# =============================================================================

resource "google_cloud_scheduler_job" "daily_pipeline_trigger" {
  name        = "daily-defi-pipeline-trigger"
  description = "Triggers the DeFi pipeline workflow daily"
  schedule    = "1 0 * * *" # 00:01 UTC every day
  time_zone   = "UTC"

  http_target {
    http_method = "POST"
    uri         = "https://workflowexecutions.googleapis.com/v1/projects/${var.project_id}/locations/${var.region}/workflows/${google_workflows_workflow.main.name}/executions"
    oauth_token {
      service_account_email = google_service_account.workflow_sa.email
    }
    headers = {
      "Content-Type" = "application/json"
    }
    body = base64encode(jsonencode({
      argument = jsonencode({
        run_mode = "daily_pipeline"
      })
    }))
  }
}

# =============================================================================
# Outputs for Dataproc Integration
# =============================================================================

output "dataproc_bucket_name" {
  description = "GCS bucket for Dataproc (scripts, temp data)"
  value       = google_storage_bucket.dataproc.name
}

output "dataproc_service_account_email" {
  description = "Service account email for Dataproc Serverless"
  value       = google_service_account.dataproc_sa.email
}
