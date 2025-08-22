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