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
resource "google_secret_manager_secret" "fred_api_key" {
  secret_id = "fred-api-key"

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "fred_api_key_version" {
  secret      = google_secret_manager_secret.fred_api_key.id
  secret_data = var.fred_api_key
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

resource "google_secret_manager_secret" "cold_wallet_address" {
  secret_id = "cold-wallet-address"

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "cold_wallet_address_version" {
  secret      = google_secret_manager_secret.cold_wallet_address.id
  secret_data = var.cold_wallet_address
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
# Note: roles/vpcaccess.user is no longer needed with Direct VPC Egress

locals {
  job_profiles = {
    "fetch_ohlcv_coinmarketcap"      = { cpu = "1", memory = "1Gi" }
    "fetch_gas_ethgastracker"        = { cpu = "1", memory = "1Gi" }
    "fetch_defillama_pools"          = { cpu = "1", memory = "2Gi" }
    "fetch_defillama_pool_addresses" = { cpu = "2", memory = "4Gi" }
    "fetch_account_transactions"     = { cpu = "1", memory = "1Gi" }
    "fetch_macroeconomic_data"       = { cpu = "1", memory = "1Gi" }
    "filter_pools_pre"               = { cpu = "1", memory = "2Gi" }
    "fetch_filtered_pool_histories"  = { cpu = "1", memory = "2Gi" }
    "calculate_pool_metrics"         = { cpu = "2", memory = "4Gi" }
    "apply_pool_grouping"            = { cpu = "1", memory = "2Gi" }
    "process_icebox_logic"           = { cpu = "1", memory = "2Gi" }
    "update_allocation_snapshots"    = { cpu = "1", memory = "2Gi" }
    "forecast_pools"                 = { cpu = "2", memory = "4Gi" }
    "forecast_gas_fees"              = { cpu = "1", memory = "2Gi" }
    "filter_pools_final"             = { cpu = "1", memory = "2Gi" }
    "process_account_transactions"   = { cpu = "1", memory = "2Gi" }
    "manage_ledger"                  = { cpu = "1", memory = "1Gi" }
    "optimize_allocations"           = { cpu = "2", memory = "4Gi" }
    "post_slack_notification"        = { cpu = "1", memory = "512Mi" }
    "apply_migrations"               = { cpu = "1", memory = "1Gi" }
    "create_allocation_snapshots"    = { cpu = "1", memory = "1Gi" }
  }

  # Image mapping for specialized Docker images
  image_mapping = {
    # Web Scraping Jobs
    "fetch_defillama_pool_addresses" = "defi-pipeline-web-scraping"
    
    # ML/Forecasting Jobs
    "forecast_pools" = "defi-pipeline-ml-science"
    "forecast_gas_fees" = "defi-pipeline-ml-science"
    "optimize_allocations" = "defi-pipeline-ml-science"
    "calculate_pool_metrics" = "defi-pipeline-ml-science"
    
    # Database Operations Jobs
    "apply_migrations" = "defi-pipeline-database"
    "create_allocation_snapshots" = "defi-pipeline-database"
    "manage_ledger" = "defi-pipeline-database"
    "post_slack_notification" = "defi-pipeline-database"
    
    # Lightweight Data Processing Jobs (default)
    "fetch_ohlcv_coinmarketcap" = "defi-pipeline-lightweight"
    "fetch_gas_ethgastracker" = "defi-pipeline-lightweight"
    "fetch_defillama_pools" = "defi-pipeline-lightweight"
    "fetch_account_transactions" = "defi-pipeline-lightweight"
    "fetch_macroeconomic_data" = "defi-pipeline-lightweight"
    "filter_pools_pre" = "defi-pipeline-lightweight"
    "fetch_filtered_pool_histories" = "defi-pipeline-lightweight"
    "apply_pool_grouping" = "defi-pipeline-lightweight"
    "process_icebox_logic" = "defi-pipeline-lightweight"
    "update_allocation_snapshots" = "defi-pipeline-lightweight"
    "filter_pools_final" = "defi-pipeline-lightweight"
    "process_account_transactions" = "defi-pipeline-lightweight"
  }
}

resource "google_cloud_run_v2_job" "pipeline_step" {
  for_each = toset([
    "apply_migrations",
    "create_allocation_snapshots",
    "fetch_ohlcv_coinmarketcap",
    "fetch_gas_ethgastracker",
    "fetch_defillama_pools",
    "fetch_defillama_pool_addresses",
    "fetch_account_transactions",
    "fetch_macroeconomic_data",
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

  depends_on = [
    google_project_service.run_api
  ]

  template {
    template {
      service_account = google_service_account.cloud_run_sa.email
      
      # Direct VPC Egress for private Cloud SQL connectivity
      vpc_access {
        network_interfaces {
          network    = "default"
          subnetwork = "default"
        }
        egress = "PRIVATE_RANGES_ONLY"
      }
      
      containers {
        image = "gcr.io/${var.project_id}/${lookup(local.image_mapping, each.key, "defi-pipeline-lightweight")}:latest"

        env {
          name  = "SCRIPT_NAME"
          value = each.key
        }
        env {
          name  = "ENVIRONMENT"
          value = "production"
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

        dynamic "env" {
          for_each = contains(["fetch_macroeconomic_data"], each.key) ? [1] : []
          content {
            name  = "FRED_API_KEY"
            value_source {
              secret_key_ref {
                secret  = google_secret_manager_secret.fred_api_key.id
                version = "latest"
              }
            }
          }
        }
        dynamic "env" {
          for_each = contains(["manage_ledger"], each.key) ? [1] : []
          content {
            name  = "COLD_WALLET_ADDRESS"
            value_source {
              secret_key_ref {
                secret  = google_secret_manager_secret.cold_wallet_address.id
                version = "latest"
              }
            }
          }
        }

        dynamic "env" {
          for_each = contains(["manage_ledger"], each.key) ? [1] : []
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
          for_each = contains(["optimize_allocations"], each.key) ? [1] : []
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

        resources {
          limits = {
            cpu    = lookup(local.job_profiles, each.key, { cpu = "1", memory = "2Gi" }).cpu
            memory = lookup(local.job_profiles, each.key, { cpu = "1", memory = "2Gi" }).memory
          }
        }
      }
      timeout = "3600s" # Set timeout to 60 minutes
    }
  }
}
