# VPC Access Connector for Cloud Run to Cloud SQL private connectivity
resource "google_vpc_access_connector" "cloud_sql_connector" {
  name          = "cloud-sql-connector"
  region        = var.region
  ip_cidr_range = "10.8.0.0/28"  # Small range for connector
  network       = "default"
  
  # Minimum capacity for cost optimization
  min_throughput = 200
  max_throughput = 300

  depends_on = [
    google_project_service.project_services["vpcaccess.googleapis.com"]
  ]
}

# Output for connector reference
output "vpc_connector_name" {
  description = "The name of the VPC access connector"
  value       = google_vpc_access_connector.cloud_sql_connector.name
}

output "vpc_connector_id" {
  description = "The full resource ID of the VPC access connector"
  value       = google_vpc_access_connector.cloud_sql_connector.id
}