resource "google_workflows_workflow" "main" {
  name            = "defi-pipeline-workflow"
  region          = var.region
  service_account = google_service_account.workflow_sa.email

  source_contents = file("${path.module}/../workflow.yaml")

  user_env_vars = {
    PIPELINE_STEP_APPLY_MIGRATIONS_ID                  = google_cloud_run_v2_job.pipeline_step["apply_migrations"].id
    PIPELINE_STEP_CREATE_ALLOCATION_SNAPSHOTS_ID       = google_cloud_run_v2_job.pipeline_step["create_allocation_snapshots"].id
    PIPELINE_STEP_FETCH_OHLCV_COINMARKETCAP_ID         = google_cloud_run_v2_job.pipeline_step["fetch_ohlcv_coinmarketcap"].id
    PIPELINE_STEP_FETCH_GAS_ETHGASTRACKER_ID           = google_cloud_run_v2_job.pipeline_step["fetch_gas_ethgastracker"].id
    PIPELINE_STEP_FETCH_DEFILLAMA_POOLS_ID             = google_cloud_run_v2_job.pipeline_step["fetch_defillama_pools"].id
    PIPELINE_STEP_FETCH_ACCOUNT_DATA_ETHERSCAN_ID      = google_cloud_run_v2_job.pipeline_step["fetch_account_data_etherscan"].id
    PIPELINE_STEP_FILTER_POOLS_PRE_ID                  = google_cloud_run_v2_job.pipeline_step["filter_pools_pre"].id
    PIPELINE_STEP_FETCH_FILTERED_POOL_HISTORIES_ID     = google_cloud_run_v2_job.pipeline_step["fetch_filtered_pool_histories"].id
    PIPELINE_STEP_CALCULATE_POOL_METRICS_ID            = google_cloud_run_v2_job.pipeline_step["calculate_pool_metrics"].id
    PIPELINE_STEP_APPLY_POOL_GROUPING_ID               = google_cloud_run_v2_job.pipeline_step["apply_pool_grouping"].id
    PIPELINE_STEP_PROCESS_ICEBOX_LOGIC_ID              = google_cloud_run_v2_job.pipeline_step["process_icebox_logic"].id
    PIPELINE_STEP_UPDATE_ALLOCATION_SNAPSHOTS_ID       = google_cloud_run_v2_job.pipeline_step["update_allocation_snapshots"].id
    PIPELINE_STEP_FILTER_POOLS_FINAL_ID                = google_cloud_run_v2_job.pipeline_step["filter_pools_final"].id
    PIPELINE_STEP_FORECAST_POOLS_ID                    = google_cloud_run_v2_job.pipeline_step["forecast_pools"].id
    PIPELINE_STEP_FORECAST_GAS_FEES_ID                 = google_cloud_run_v2_job.pipeline_step["forecast_gas_fees"].id
    PIPELINE_STEP_OPTIMIZE_ALLOCATIONS_ID              = google_cloud_run_v2_job.pipeline_step["optimize_allocations"].id
    PIPELINE_STEP_MANAGE_LEDGER_ID                     = google_cloud_run_v2_job.pipeline_step["manage_ledger"].id
    PIPELINE_STEP_POST_SLACK_NOTIFICATION_ID           = google_cloud_run_v2_job.pipeline_step["post_slack_notification"].id
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

