# Random suffix for unique bucket naming
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# Bucket for Terraform state (if not already exists)
resource "google_storage_bucket" "terraform_state" {
  name          = "${var.project_id}-terraform-state-${random_id.bucket_suffix.hex}"
  location      = "US-CENTRAL1"
  force_destroy = false

  public_access_prevention = "enforced"
  
  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }
}