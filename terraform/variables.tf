# Variables for GCP Cloud SQL PostgreSQL setup
variable "project_id" {
  description = "The GCP project ID."
  type        = string
  default     = "innate-concept-430315-q2"
}

variable "region" {
  description = "The GCP region for the Cloud SQL instance."
  type        = string
  default     = "us-central1"
}

variable "database_version" {
  description = "The PostgreSQL database version."
  type        = string
  default     = "POSTGRES_14"
}

variable "instance_tier" {
  description = "The machine type for the Cloud SQL instance."
  type        = string
  default     = "db-g1-small"
}

variable "disk_size" {
  description = "The disk size for the Cloud SQL instance in GB."
  type        = number
  default     = 20
}

variable "database_name" {
  description = "The name of the PostgreSQL database to create."
  type        = string
  default     = "defiyieldopt_db"
}

variable "database_user" {
  description = "The username for the PostgreSQL database."
  type        = string
  default     = "defiyieldopt_user"
}

variable "database_password" {
  description = "The password for the PostgreSQL database user."
  type        = string
  sensitive   = true
}

variable "ethplorer_api_key" {
  description = "The Ethplorer API key for fetching account and gas data."
  type        = string
  sensitive   = true
}

variable "coinmarketcap_api_key" {
  description = "The CoinMarketCap API key for fetching OHLCV data."
  type        = string
  sensitive   = true
}

variable "main_asset_holding_address" {
  description = "The main asset holding Ethereum address."
  type        = string
  sensitive   = true
}

variable "ethgastracker_api_key" {
  description = "The ETH Gas Tracker API key."
  type        = string
  sensitive   = true
}

variable "slack_webhook_url" {
  description = "The Slack webhook URL for notifications."
  type        = string
  sensitive   = true
}