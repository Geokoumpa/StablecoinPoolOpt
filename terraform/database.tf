# Private Service Connection - using existing psc-range
# No need to create new resources

resource "google_service_networking_connection" "private_vpc_connection" {
  provider = google-beta
  network = "projects/${var.project_id}/global/networks/default"
  service = "servicenetworking.googleapis.com"

  reserved_peering_ranges = ["psc-range"]

  depends_on = [
    google_project_service.project_services["servicenetworking.googleapis.com"]
  ]
}

# Cloud SQL PostgreSQL instance
resource "google_sql_database_instance" "main_instance" {
  name             = "defiyieldopt-db-instance"
  database_version = var.database_version
  region           = var.region
  deletion_protection = false

  settings {
    tier = var.instance_tier
    disk_size = var.disk_size
    disk_type = "PD_SSD"
    disk_autoresize = true
    disk_autoresize_limit = 100

    backup_configuration {
      enabled                        = true
      start_time                     = "23:00"
      point_in_time_recovery_enabled = true
      transaction_log_retention_days = 7
      backup_retention_settings {
        retained_backups = 30
      }
    }

    ip_configuration {
      ipv4_enabled    = false  # Disable public IP for security
      private_network = "projects/${var.project_id}/global/networks/default"
    }

    database_flags {
      name  = "log_checkpoints"
      value = "on"
    }
    database_flags {
      name  = "log_connections"
      value = "on"
    }
    database_flags {
      name  = "log_disconnections"
      value = "on"
    }
    database_flags {
      name  = "log_lock_waits"
      value = "on"
    }
  }

  depends_on = [
    google_project_service.project_services["sqladmin.googleapis.com"],
    google_service_networking_connection.private_vpc_connection
  ]
}

# Database
resource "google_sql_database" "defiyieldopt_database" {
  name     = var.database_name
  instance = google_sql_database_instance.main_instance.name
}

# Database user
resource "google_sql_user" "defiyieldopt_user" {
  name     = var.database_user
  instance = google_sql_database_instance.main_instance.name
  password = var.database_password
}