terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 7.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 7.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.1"
    }
  }

  backend "gcs" {
    bucket = "defiyieldopt-terraform-state"
    prefix = "terraform/state"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_project_service" "project_services" {
  for_each = toset([
    "run.googleapis.com",
    "sqladmin.googleapis.com",
    "secretmanager.googleapis.com",
    "workflows.googleapis.com",
    "servicenetworking.googleapis.com", # Added for Private Service Connection
    "compute.googleapis.com", # Added for Private Service Connection
    "cloudresourcemanager.googleapis.com", # Added for project info
    "iam.googleapis.com", # Added for service account creation
    "cloudfunctions.googleapis.com", # Potentially needed for future extensions
    "cloudscheduler.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "vpcaccess.googleapis.com" # Required for Serverless VPC Access
  ])
  service                    = each.key
  disable_dependent_services = true
}

data "google_project" "project" {
  project_id = var.project_id
}

resource "google_service_account" "cloud_run_sa" {
  account_id   = "cloud-run-sa"
  display_name = "Service Account for Cloud Run services"
  project      = var.project_id
}

resource "google_service_account" "workflow_sa" {
  account_id   = "workflow-sa"
  display_name = "Service Account for Cloud Workflows"
  project      = var.project_id
}

resource "google_project_iam_member" "cloud_run_sa_roles" {
  for_each = toset([
    "roles/run.invoker",
    "roles/secretmanager.secretAccessor",
    "roles/cloudsql.client",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter"
  ])
  project = var.project_id
  role    = each.key
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

resource "google_project_iam_member" "workflow_sa_roles" {
  for_each = toset([
    "roles/workflows.invoker",
    "roles/run.invoker",
    "roles/run.developer",
    "roles/logging.logWriter"
  ])
  project = var.project_id
  role    = each.key
  member  = "serviceAccount:${google_service_account.workflow_sa.email}"
}

# Cloud Router for Cloud NAT
resource "google_compute_router" "nat_router" {
  name    = "nat-router-${var.region}"
  region  = var.region
  network = "default" # Assuming your Cloud Run connector uses the 'default' VPC network
}

# Cloud NAT Gateway
resource "google_compute_router_nat" "nat_gateway" {
  name                          = "nat-gateway-${var.region}"
  router                        = google_compute_router.nat_router.name
  region                        = google_compute_router.nat_router.region
  nat_ip_allocate_option        = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"

  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}

# Cloud Build Trigger for Docker image builds on git push to main
# Commented out temporarily due to configuration issues
# resource "google_cloudbuild_trigger" "docker_build_trigger" {
#   name        = "defi-pipeline-build-trigger"
#   description = "Build and push defi-pipeline Docker images on main branch push"
#   filename    = "cloudbuild.yaml"
#
#   github {
#     owner = "your-github-username"  # Replace with actual GitHub username
#     name  = "stablecoin-yield-pipeline"  # Replace with actual repo name
#     push {
#       branch = "^master$"
#     }
#   }
#
#   depends_on = [google_project_service.project_services["cloudbuild.googleapis.com"]]
# }