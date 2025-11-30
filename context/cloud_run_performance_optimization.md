# Cloud Run Performance Optimization Plan

## Overview
This plan addresses performance bottlenecks in our DeFi pipeline Cloud Run jobs, focusing on reducing cold start times (currently 8-15s), optimizing resource allocation, and improving overall execution performance.

## Current State Assessment
- **Cold Start Time**: 8-15 seconds
- **Primary Bottlenecks**: Large container image (1GB+), heavy Python imports, Playwright dependencies
- **Current Resource Allocation**: 1 CPU, 2Gi memory (uniform across jobs)
- **Network Architecture**: VPC connector for Cloud SQL access

## Performance Analysis

### Resource Allocation Assessment
Current uniform allocation (1 CPU, 2Gi) is inefficient for diverse workloads:

| Job Category | Examples | Current Allocation | Recommended Allocation | Issue |
|---------------|-------------|-------------------|-------------------|--------|
| API Data Fetchers | fetch_ohlcv, fetch_gas | 1 CPU, 2Gi | 0.5 CPU, 1Gi | Over-provisioned |
| Simple Processing | filter_pools_pre | 1 CPU, 2Gi | 1 CPU, 2Gi | Appropriate |
| Heavy Processing | calculate_pool_metrics | 1 CPU, 2Gi | 2 CPU, 4Gi | Under-provisioned |
| ML/Forecasting | forecast_pools, optimize_allocations | 1 CPU, 2Gi | 2 CPU, 4Gi | Under-provisioned |
| Database Ops | apply_migrations, manage_ledger | 1 CPU, 2Gi | 0.5 CPU, 1Gi | Over-provisioned |
| Browser Ops | fetch_defillama_pool_addresses | 2 CPU, 4Gi | 2 CPU, 4Gi | Appropriate |

## Optimization Strategy

### Phase 1: Quick Wins (1-2 days implementation)

#### 1.1 Enable CPU Boost for Startup
- Add `startup_cpu_boost = true` to all Cloud Run job configurations
- Expected impact: 30-40% reduction in cold start time
- Cost implication: Minimal additional cost during startup only

#### 1.2 Implement Lazy Import Pattern
- Refactor Python scripts to import heavy libraries only when needed
- Move imports like `xgboost`, `lightgbm`, `playwright` inside functions
- Expected impact: 2-3 seconds faster script initialization

#### 1.3 Right-Size Resources by Job Type
- Implement job-specific resource profiles in Terraform
- Reduce resources for I/O-bound jobs
- Increase resources for CPU-intensive ML jobs
- Expected impact: 30-40% cost reduction, 50% performance improvement for ML jobs

### Phase 2: Container Architecture Improvements (1 week implementation)

#### 2.1 Split Container Images by Function
Create specialized images for different job types:
- **Web Scraping Image**: Playwright + browser dependencies
- **Data Science Image**: ML libraries (xgboost, lightgbm, etc.)
- **Lightweight Image**: Simple data processing jobs
- **Database Image**: Minimal dependencies for SQL operations

Expected impact: 40-50% reduction in image size for non-browser jobs

#### 2.2 Implement Multi-Stage Docker Builds
- Separate build environment from runtime environment
- Remove build-time dependencies from final image
- Optimize layer caching for faster builds

#### 2.3 Optimize Base Image Strategy
- Evaluate `python:3.11-slim` for performance improvements
- Consider distroless images for security and size reduction
- Benchmark different base images for startup performance

### Phase 3: Advanced Optimizations (2-3 weeks implementation)

#### 3.1 Dynamic Resource Allocation Framework
Implement Terraform module for job-specific resources:

```hcl
# Job resource profiles
locals {
  job_profiles = {
    "fetch_ohlcv_coinmarketcap"    = { cpu = "0.5", memory = "1Gi", image = "lightweight" }
    "fetch_defillama_pools"        = { cpu = "1",   memory = "2Gi", image = "standard" }
    "calculate_pool_metrics"        = { cpu = "2",   memory = "4Gi", image = "datascience" }
    "forecast_pools"               = { cpu = "2",   memory = "4Gi", image = "datascience" }
    "fetch_defillama_pool_addresses" = { cpu = "2",   memory = "4Gi", image = "browser" }
    "apply_migrations"              = { cpu = "0.5", memory = "1Gi", image = "database" }
  }
}
```

#### 3.2 Container Reuse Strategy
- For frequently run jobs, consider `min_instance_count = 1`
- Implement container warming for critical path jobs
- Cost-benefit analysis of always-on instances

#### 3.3 Optimize VPC Connector Configuration
- Review and tune VPC connector throughput settings
- Consider direct internet egress for non-sensitive operations
- Implement connection pooling at the network level

### Phase 4: Monitoring and Continuous Optimization

#### 4.1 Performance Monitoring Setup
- Implement detailed startup time tracking
- Set up alerts for performance degradation
- Create performance dashboards
- Monitor resource utilization by job type

#### 4.2 A/B Testing Framework
- Implement gradual rollout of optimizations
- Measure impact of each optimization
- Establish performance baselines

## Implementation Priority Matrix

| Optimization | Startup Impact | Performance Impact | Cost Impact | Effort | Priority |
|--------------|------------------|-------------------|---------------|----------|----------|
| CPU Boost | High | Low | Low | Low | 1 |
| Resource Right-Sizing | Low | High | High | Low | 2 |
| Lazy Imports | Medium | Medium | Low | Low | 3 |
| Image Splitting | High | Medium | Medium | Medium | 4 |
| Dynamic Allocation | Low | High | High | Medium | 5 |

## Expected Outcomes

### Performance Improvements
- **Cold Start Time**: 8-15s → 3-6s (60% improvement)
- **ML Job Performance**: 50-60% faster execution
- **Data Processing**: 20-30% faster completion
- **Overall Pipeline**: 25-35% reduction in total execution time

### Cost Optimization
- **Compute Costs**: 30-40% reduction through right-sizing
- **Storage Costs**: 20% reduction through smaller images
- **Network Costs**: 10-15% reduction through optimized VPC usage

### Resource Efficiency
- **CPU Utilization**: 40-60% improvement
- **Memory Utilization**: 50-70% improvement
- **Job Success Rate**: Improved through better resource matching

## Risk Assessment

| Risk Category | Low Risk | Medium Risk | High Risk |
|---------------|------------|--------------|------------|
| Quick Wins | CPU boost, lazy imports, basic monitoring | Resource right-sizing | |
| Architecture | | Image splitting, multi-stage builds | Major architectural changes |
| Advanced | | | Container reuse, dynamic allocation |

## Success Metrics
1. **Startup Performance**
   - Average cold start time < 6 seconds
   - 95th percentile cold start time < 8 seconds

2. **Execution Performance**
   - ML job completion time reduced by 50%
   - Overall pipeline time reduced by 25%

3. **Cost Efficiency**
   - Compute cost per job reduced by 30%
   - No increase in job failure rate

4. **Resource Utilization**
   - Average CPU utilization > 60%
   - Average memory utilization > 70%

## Implementation Timeline

```mermaid
gantt
    title Cloud Run Optimization Timeline
    dateFormat  YYYY-MM-DD
    section Phase 1
    Quick Wins           :done, phase1a, 2024-01-01, 2d
    Resource Right-Sizing   :done, phase1b, 2024-01-03, 1d
    
    section Phase 2
    Container Architecture  :active, phase2, 2024-01-05, 5d
    Image Splitting       :phase2, 2024-01-05, 3d
    Multi-Stage Builds    :phase2, 2024-01-08, 2d
    
    section Phase 3
    Advanced Optimizations :phase3, 2024-01-10, 10d
    Dynamic Allocation    :phase3, 2024-01-10, 5d
    Container Reuse      :phase3, 2024-01-15, 3d
    
    section Phase 4
    Monitoring Setup      :phase4, 2024-01-20, 3d
    A/B Testing         :phase4, 2024-01-23, 5d