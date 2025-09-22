# Secret Manager secrets for all API keys
resource "google_secret_manager_secret" "db_password" {
  secret_id = "db-password"

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "db_password_version" {
  secret      = google_secret_manager_secret.db_password.id
  secret_data = var.database_password
}

resource "google_secret_manager_secret" "slack_webhook_url" {
  secret_id = "slack-webhook-url"

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "slack_webhook_url_version" {
  secret      = google_secret_manager_secret.slack_webhook_url.id
  secret_data = var.slack_webhook_url
}

resource "google_secret_manager_secret" "ethplorer_api_key" {
  secret_id = "ethplorer-api-key"

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "ethplorer_api_key_version" {
  secret      = google_secret_manager_secret.ethplorer_api_key.id
  secret_data = var.ethplorer_api_key
}

resource "google_secret_manager_secret" "coinmarketcap_api_key" {
  secret_id = "coinmarketcap-api-key"

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "coinmarketcap_api_key_version" {
  secret      = google_secret_manager_secret.coinmarketcap_api_key.id
  secret_data = var.coinmarketcap_api_key
}

resource "google_secret_manager_secret" "ethgastracker_api_key" {
  secret_id = "ethgastracker-api-key"

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "ethgastracker_api_key_version" {
  secret      = google_secret_manager_secret.ethgastracker_api_key.id
  secret_data = var.ethgastracker_api_key
}

resource "google_secret_manager_secret" "main_asset_holding_address" {
  secret_id = "main-asset-holding-address"

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "main_asset_holding_address_version" {
  secret      = google_secret_manager_secret.main_asset_holding_address.id
  secret_data = var.main_asset_holding_address
}

# VPC Access Connector for private Cloud SQL access

# Enable Cloud Run API
resource "google_project_service" "run_api" {
  project                    = var.project_id
  service                    = "run.googleapis.com"
  disable_dependent_services = true
}

# Enable Secret Manager API
resource "google_project_service" "secret_manager_api" {
  project                    = var.project_id
  service                    = "secretmanager.googleapis.com"
  disable_dependent_services = true
}

# Cloud Run Jobs for each pipeline step
resource "google_project_iam_member" "cloud_run_vpc_access" {
  project = var.project_id
  role    = "roles/vpcaccess.user"
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

resource "google_cloud_run_v2_job" "pipeline_step" {
  for_each = toset([
    "apply_migrations",
    "create_allocation_snapshots",
    "fetch_ohlcv_coinmarketcap",
    "fetch_gas_ethgastracker",
    "fetch_defillama_pools",
    "fetch_account_transactions",
    "filter_pools_pre",
    "fetch_filtered_pool_histories",
    "calculate_pool_metrics",
    "apply_pool_grouping",
    "process_icebox_logic",
    "update_allocation_snapshots",
    "filter_pools_final",
    "forecast_pools",
    "forecast_gas_fees",
    "optimize_allocations",
    "manage_ledger",
    "post_slack_notification",
    "process_account_transactions"
  ])

  name               = "pipeline-step-${replace(each.key, "_", "-")}"
  location           = var.region
  project            = var.project_id
  deletion_protection = false

  template {
    template {
      service_account = google_service_account.cloud_run_sa.email
      
      # VPC Access for private Cloud SQL connectivity
      vpc_access {
        connector = google_vpc_access_connector.cloud_sql_connector.id
        egress    = "PRIVATE_RANGES_ONLY"
      }
      
      containers {
        image = "gcr.io/${var.project_id}/defi-pipeline:latest"

        env {
          name  = "SCRIPT_NAME"
          value = each.key
        }
        env {
          name  = "DB_USER"
          value = google_sql_user.defiyieldopt_user.name
        }
        env {
          name  = "DB_HOST"
          value = google_sql_database_instance.main_instance.private_ip_address
        }
        env {
          name  = "DB_PORT"
          value = "5432"
        }
        env {
          name  = "DB_NAME"
          value = google_sql_database.defiyieldopt_database.name
        }
        env {
          name = "DB_PASSWORD"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.db_password.secret_id
              version = "latest"
            }
          }
        }

        # Conditional API keys for specific jobs
        dynamic "env" {
          for_each = contains(["fetch_ohlcv_coinmarketcap"], each.key) ? [1] : []
          content {
            name  = "COINMARKETCAP_API_KEY"
            value_source {
              secret_key_ref {
                secret  = google_secret_manager_secret.coinmarketcap_api_key.id
                version = "latest"
              }
            }
          }
        }

        dynamic "env" {
          for_each = contains(["fetch_gas_ethgastracker"], each.key) ? [1] : []
          content {
            name  = "ETHGASTRACKER_API_KEY"
            value_source {
              secret_key_ref {
                secret  = google_secret_manager_secret.ethgastracker_api_key.id
                version = "latest"
              }
            }
          }
        }

        dynamic "env" {
          for_each = contains(["fetch_account_transactions"], each.key) ? [1] : []
          content {
            name  = "MAIN_ASSET_HOLDING_ADDRESS"
            value_source {
              secret_key_ref {
                secret  = google_secret_manager_secret.main_asset_holding_address.id
                version = "latest"
              }
            }
          }
        }

        dynamic "env" {
          for_each = contains(["fetch_account_transactions"], each.key) ? [1] : []
          content {
            name  = "ETHPLORER_API_KEY"
            value_source {
              secret_key_ref {
                secret  = google_secret_manager_secret.ethplorer_api_key.id
                version = "latest"
              }
            }
          }
        }

        dynamic "env" {
          for_each = contains(["process_account_transactions"], each.key) ? [1] : []
          content {
            name  = "ETHPLORER_API_KEY"
            value_source {
              secret_key_ref {
                secret  = google_secret_manager_secret.ethplorer_api_key.id
                version = "latest"
              }
            }
          }
        }

        dynamic "env" {
          for_each = contains(["post_slack_notification"], each.key) ? [1] : []
          content {
            name  = "SLACK_WEBHOOK_URL"
            value_source {
              secret_key_ref {
                secret  = google_secret_manager_secret.slack_webhook_url.id
                version = "latest"
              }
            }
          }
        }
      }
      timeout = "1800s" # Set timeout to 30 minutes
    }
  }

  depends_on = [
    google_project_service.run_api,
    google_vpc_access_connector.cloud_sql_connector
  ]
}
