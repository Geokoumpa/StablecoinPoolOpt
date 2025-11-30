# Cloud Run Rollback Procedures using gcloud Commands

This document outlines the manual rollback procedures for the DeFi pipeline using gcloud commands when you need to quickly revert to a previous Docker image version without changing the code repository.

## Prerequisites

- Google Cloud SDK installed and authenticated
- Appropriate IAM permissions for Cloud Run operations
- Project ID and region information

## Step 1: Identify Available Image Tags

List the available Docker image tags to identify the version you want to rollback to:

```bash
# List all available image tags (most recent first)
gcloud container images list-tags gcr.io/$PROJECT_ID/defi-pipeline \
  --limit=10 \
  --format="table(tags, digest, timestamp.datetime)" \
  --sort-by="~timestamp"

# Example output:
# TAGS                      DIGEST                                    TIMESTAMP
# latest, abc123def456       sha256:abc123...                           2024-01-15 10:30:00
# previous, xyz789uvw012    sha256:xyz789...                           2024-01-14 15:45:00
# v1.2.0                    sha256:uvw012...                           2024-01-13 09:20:00
```

## Step 2: Select Target Image Tag

Choose the image tag you want to rollback to. Common options:
- Previous commit SHA: `xyz789uvw012`
- Version tag: `v1.2.0`
- Previous stable tag: `previous`

## Step 3: Rollback Individual Cloud Run Jobs

### Method A: Rollback Single Job

```bash
# Update a specific Cloud Run job to use the previous image
gcloud run jobs update pipeline-step-fetch-defillama-pools \
  --image="gcr.io/$PROJECT_ID/defi-pipeline:xyz789uvw012" \
  --region="$REGION" \
  --quiet

# Verify the update
gcloud run jobs describe pipeline-step-fetch-defillama-pools \
  --region="$REGION" \
  --format="value(spec.template.spec.containers[0].image)"
```

### Method B: Rollback All Jobs (Recommended)

Create a script to rollback all 22 pipeline jobs:

```bash
#!/bin/bash
# rollback-all-jobs.sh

set -e

PROJECT_ID="$1"
REGION="$2"
TARGET_TAG="$3"

if [ -z "$PROJECT_ID" ] || [ -z "$REGION" ] || [ -z "$TARGET_TAG" ]; then
  echo "Usage: $0 <PROJECT_ID> <REGION> <TARGET_TAG>"
  echo "Example: $0 my-project us-central1 xyz789uvw012"
  exit 1
fi

echo "Rolling back all Cloud Run jobs to: gcr.io/$PROJECT_ID/defi-pipeline:$TARGET_TAG"

# List of all pipeline steps (matches terraform/cloud_run.tf)
STEPS=(
  "apply_migrations"
  "create_allocation_snapshots"
  "fetch_ohlcv_coinmarketcap"
  "fetch_gas_ethgastracker"
  "fetch_defillama_pools"
  "fetch_defillama_pool_addresses"
  "fetch_account_transactions"
  "fetch_macroeconomic_data"
  "filter_pools_pre"
  "fetch_filtered_pool_histories"
  "calculate_pool_metrics"
  "apply_pool_grouping"
  "process_icebox_logic"
  "update_allocation_snapshots"
  "filter_pools_final"
  "forecast_pools"
  "forecast_gas_fees"
  "optimize_allocations"
  "manage_ledger"
  "post_slack_notification"
  "process_account_transactions"
)

# Update each Cloud Run Job
SUCCESS_COUNT=0
FAILED_COUNT=0

for step in "${STEPS[@]}"; do
  job_name="pipeline-step-${step//_/-}"
  echo "Updating job: $job_name"
  
  if gcloud run jobs update "$job_name" \
    --image="gcr.io/$PROJECT_ID/defi-pipeline:$TARGET_TAG" \
    --region="$REGION" \
    --quiet; then
    echo "✓ Successfully updated $job_name"
    ((SUCCESS_COUNT++))
  else
    echo "✗ Failed to update $job_name"
    ((FAILED_COUNT++))
  fi
done

echo ""
echo "Rollback Summary:"
echo "✓ Successfully updated: $SUCCESS_COUNT jobs"
echo "✗ Failed to update: $FAILED_COUNT jobs"

if [ $FAILED_COUNT -gt 0 ]; then
  echo "⚠️  Some jobs failed to update. Please check the errors above."
  exit 1
else
  echo "🎉 All jobs successfully rolled back!"
fi
```

### Execute the Rollback Script

```bash
# Make the script executable
chmod +x rollback-all-jobs.sh

# Execute the rollback
./rollback-all-jobs.sh $PROJECT_ID $REGION xyz789uvw012
```

## Step 4: Verify Rollback

### Verify Individual Jobs

```bash
# Check the image being used by a specific job
gcloud run jobs describe pipeline-step-fetch-defillama-pools \
  --region="$REGION" \
  --format="value(spec.template.spec.containers[0].image)"

# Check all jobs' images
for step in apply_migrations create_allocation_snapshots fetch_ohlcv_coinmarketcap; do
  job_name="pipeline-step-${step//_/-}"
  image=$(gcloud run jobs describe "$job_name" \
    --region="$REGION" \
    --format="value(spec.template.spec.containers[0].image)")
  echo "$job_name: $image"
done
```

### Test the Rollback

```bash
# Execute a test job to verify the rollback
gcloud run jobs execute pipeline-step-fetch-defillama-pools \
  --region="$REGION" \
  --wait

# Check the execution logs
gcloud run jobs executions list \
  --job=pipeline-step-fetch-defillama-pools \
  --region="$REGION" \
  --limit=1 \
  --format="value(metadata.name)" | \
xargs gcloud run jobs executions logs read \
  --region="$REGION" \
  --execution
```

## Step 5: Monitor Pipeline Health

After rollback, monitor the pipeline execution:

```bash
# Check recent workflow executions
gcloud workflows executions list \
  --workflow=defi-pipeline-workflow \
  --region="$REGION" \
  --limit=5 \
  --format="table(name,state,startTime,startTime)"

# Check specific execution logs
gcloud workflows executions describe \
  --workflow=defi-pipeline-workflow \
  --region="$REGION" \
  --execution=<EXECUTION_ID>
```

## Emergency Rollback Procedures

### Quick Rollback to Previous Version

```bash
# Get the previous image tag (excluding current latest)
PREVIOUS_TAG=$(gcloud container images list-tags gcr.io/$PROJECT_ID/defi-pipeline \
  --limit=2 \
  --format="get(tags)" \
  --filter="tags!=latest" | \
  tr ',' '\n' | \
  grep -v "latest" | \
  head -1)

echo "Rolling back to: $PREVIOUS_TAG"

# Quick rollback (copy-paste ready)
gcloud run jobs update pipeline-step-fetch-defillama-pools \
  --image="gcr.io/$PROJECT_ID/defi-pipeline:$PREVIOUS_TAG" \
  --region="$REGION" \
  --quiet
```

### Rollback All Jobs to Previous Version

```bash
# One-liner for emergency rollback
PREVIOUS_TAG=$(gcloud container images list-tags gcr.io/$PROJECT_ID/defi-pipeline \
  --limit=2 \
  --format="get(tags)" \
  --filter="tags!=latest" | \
  tr ',' '\n' | \
  grep -v "latest" | \
  head -1)

for step in apply_migrations create_allocation_snapshots fetch_ohlcv_coinmarketcap fetch_gas_ethgastracker fetch_defillama_pools fetch_defillama_pool_addresses fetch_account_transactions fetch_macroeconomic_data filter_pools_pre fetch_filtered_pool_histories calculate_pool_metrics apply_pool_grouping process_icebox_logic update_allocation_snapshots filter_pools_final forecast_pools forecast_gas_fees optimize_allocations manage_ledger post_slack_notification process_account_transactions; do
  job_name="pipeline-step-${step//_/-}"
  echo "Rolling back $job_name to $PREVIOUS_TAG"
  gcloud run jobs update "$job_name" \
    --image="gcr.io/$PROJECT_ID/defi-pipeline:$PREVIOUS_TAG" \
    --region="$REGION" \
    --quiet
done
```

## Troubleshooting

### Common Issues

1. **Permission Denied**
   ```bash
   # Ensure you have the right permissions
   gcloud projects add-iam-policy-binding $PROJECT_ID \
     --member="user:your-email@example.com" \
     --role="roles/run.admin"
   ```

2. **Job Not Found**
   ```bash
   # List all available jobs
   gcloud run jobs list --region="$REGION"
   ```

3. **Image Not Found**
   ```bash
   # Verify the image exists
   gcloud container images describe gcr.io/$PROJECT_ID/defi-pipeline:$TARGET_TAG
   ```

### Rollback Validation Checklist

- [ ] All Cloud Run jobs updated to target image
- [ ] Test job execution succeeds
- [ ] Pipeline workflow completes successfully
- [ ] Data processing results are correct
- [ ] No error logs in recent executions
- [ ] Performance metrics are within expected ranges

## Automation Scripts

### Create Rollback Script Template

```bash
# Create reusable rollback script
cat > rollback-template.sh << 'EOF'
#!/bin/bash
set -e

PROJECT_ID="${PROJECT_ID}"
REGION="${REGION:-us-central1}"
TARGET_TAG="${1}"

if [ -z "$TARGET_TAG" ]; then
  echo "Usage: $0 <TARGET_TAG>"
  echo "Available tags:"
  gcloud container images list-tags gcr.io/$PROJECT_ID/defi-pipeline \
    --limit=10 \
    --format="value(tags)" | tr ',' '\n'
  exit 1
fi

echo "Rolling back to: gcr.io/$PROJECT_ID/defi-pipeline:$TARGET_TAG"

# Add your rollback logic here
EOF

chmod +x rollback-template.sh
```

## Best Practices

1. **Always test rollback on a single job first**
2. **Monitor job execution after rollback**
3. **Keep a record of rollback operations**
4. **Have rollback procedures documented and accessible**
5. **Use specific commit SHA tags rather than generic tags**
6. **Implement health checks before and after rollback**

## Related Documentation

- [Cloud Run Jobs Documentation](https://cloud.google.com/run/docs/reference/rest/v2/namespaces.jobs)
- [gcloud CLI Reference](https://cloud.google.com/sdk/gcloud/reference/run/jobs)
- [Container Registry Documentation](https://cloud.google.com/container-registry/docs)