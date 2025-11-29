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
