# Outputs for GCP Cloud SQL PostgreSQL setup
output "instance_connection_name" {
  description = "The connection name of the Cloud SQL instance."
  value       = google_sql_database_instance.main_instance.connection_name
}

output "database_name" {
  description = "The name of the created database."
  value       = google_sql_database.defiyieldopt_database.name
}

output "database_user" {
  description = "The username for the database."
  value       = google_sql_user.defiyieldopt_user.name
}

output "database_host" {
  description = "The IP address of the Cloud SQL instance."
  value       = google_sql_database_instance.main_instance.public_ip_address
}

# Outputs for Cloud Run pipeline jobs
output "pipeline_step_job_ids" {
  description = "The full resource IDs of the Cloud Run pipeline jobs."
  value       = { for k, v in google_cloud_run_v2_job.pipeline_step : k => v.id }
}

# Outputs for Cloud Workflows and Schedulers
output "workflow_id" {
  description = "The full resource ID of the Cloud Workflows workflow."
  value       = google_workflows_workflow.main.id
}

output "daily_scheduler_id" {
  description = "The resource ID of the daily pipeline scheduler job."
  value       = google_cloud_scheduler_job.daily_pipeline_trigger.id
}


# Output for VPC Connector
output "vpc_connector_id" {
  description = "The resource ID of the VPC Access Connector."
  value       = google_vpc_access_connector.connector.id
}

# Output for Service Accounts
output "cloud_run_service_account_email" {
  description = "The email of the Cloud Run service account."
  value       = google_service_account.cloud_run_sa.email
}

output "workflow_service_account_email" {
  description = "The email of the Cloud Workflows service account."
  value       = google_service_account.workflow_sa.email
}