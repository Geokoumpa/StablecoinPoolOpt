terraform {
  backend "gcs" {
    bucket = "defiyieldopt-terraform-state"
    prefix = "terraform/state"
  }
}

# Configure the Google Cloud provider
provider "google" {
  project = var.project_id
  region  = var.region
}

# Enable the Cloud SQL Admin API
resource "google_project_service" "sqladmin_api" {
  project = var.project_id
  service = "sqladmin.googleapis.com"
  disable_on_destroy = false
}

# Enable the Cloud Storage API
resource "google_project_service" "storage_api" {
  project = var.project_id
  service = "storage.googleapis.com"
  disable_on_destroy = false
}

# Create a Cloud SQL PostgreSQL instance
resource "google_sql_database_instance" "main_instance" {
  database_version = var.database_version
  name             = "defiyieldopt-sql-instance"
  project          = var.project_id
  region           = var.region
  settings {
    tier = var.instance_tier
    disk_size = var.disk_size
    backup_configuration {
      enabled            = true
      start_time         = "03:00"
    }
    ip_configuration {
      ipv4_enabled = true
      # You might want to restrict authorized networks for production
      # authorized_networks {
      #   value = "0.0.0.0/0" # Replace with your IP range
      # }
    }
    database_flags {
      name  = "cloudsql.iam_authentication"
      value = "On"
    }
  }
  depends_on = [google_project_service.sqladmin_api]
}

# Create a PostgreSQL database
resource "google_sql_database" "defiyieldopt_database" {
  name     = var.database_name
  instance = google_sql_database_instance.main_instance.name
  project  = var.project_id
  charset  = "UTF8"
  collation = "en_US.UTF8"
}

# Create a PostgreSQL user
resource "google_sql_user" "defiyieldopt_user" {
  name     = var.database_user
  instance = google_sql_database_instance.main_instance.name
  project  = var.project_id
  password = var.database_password
}

# Create a GCS bucket for model persistence
resource "google_storage_bucket" "model_bucket" {
  name          = "${var.project_id}-forecasting-models" # Unique bucket name
  location      = var.region
  project       = var.project_id
  force_destroy = false # Set to true for easier testing, but be careful in production

  uniform_bucket_level_access = true

  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      age = 30 # Delete objects older than 30 days
    }
  }

  depends_on = [google_project_service.storage_api]
}