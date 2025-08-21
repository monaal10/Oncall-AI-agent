# GCP Cloud Integration

This module provides integration with Google Cloud Platform services for the OnCall AI Agent.

## Features

### Cloud Logging
- Query logs using Cloud Logging filter expressions
- Search across multiple log types and resources
- Support for structured and unstructured logs
- Resource-specific log helpers (App Engine, GKE, Cloud Functions)
- Severity-based filtering

### Cloud Monitoring
- Query metrics from GCP resources
- Support for Compute Engine, GKE, Cloud Functions, and other services
- Time-series data with configurable aggregations
- Alert policy information
- Resource-specific metric helpers

## Configuration

### Authentication Methods

1. **Default Application Credentials (Recommended)**
   ```bash
   gcloud auth application-default login
   ```

2. **Service Account Key File**
   ```bash
   # Create service account
   gcloud iam service-accounts create oncall-agent
   
   # Create key file
   gcloud iam service-accounts keys create key.json \
     --iam-account=oncall-agent@PROJECT_ID.iam.gserviceaccount.com
   
   # Set environment variable
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
   ```

3. **Automatic (GCE/GKE/Cloud Run)**
   - Service accounts are automatically available on Google Cloud services

### Required IAM Roles

```bash
# Grant required roles
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:oncall-agent@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/logging.viewer"

gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:oncall-agent@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/monitoring.viewer"
```

- **roles/logging.viewer**: Read access to Cloud Logging
- **roles/monitoring.viewer**: Read access to Cloud Monitoring
- **roles/project.viewer**: Basic project access (optional)

### Configuration Example

```yaml
integrations:
  log_provider:
    type: "gcp_logging"
    config:
      project_id: "${GCP_PROJECT_ID}"
      # Optional: specify service account file
      credentials_path: "/path/to/service-account.json"
      
  metrics_provider:
    type: "gcp_monitoring"
    config:
      project_id: "${GCP_PROJECT_ID}"
```

## Usage Examples

### Cloud Logging

```python
from oncall_agent.integrations.gcp import GCPCloudLoggingProvider, create_gcp_config

config = create_gcp_config(project_id="your-project-id")
logs_provider = GCPCloudLoggingProvider(config)

# Filter query
logs = await logs_provider.fetch_logs(
    query='severity >= "ERROR" AND resource.type="gce_instance"',
    start_time=start_time,
    end_time=end_time
)

# Pattern search
errors = await logs_provider.search_logs_by_pattern(
    pattern="exception",
    start_time=start_time,
    end_time=end_time
)

# Service-specific logs
gke_logs = await logs_provider.get_gke_logs(
    cluster_name="my-cluster",
    start_time=start_time,
    end_time=end_time
)
```

### Cloud Monitoring

```python
from oncall_agent.integrations.gcp import GCPCloudMonitoringProvider

metrics_provider = GCPCloudMonitoringProvider(config)

# Compute Engine metrics
instance_metrics = await metrics_provider.get_compute_instance_metrics(
    instance_name="my-instance",
    zone="us-central1-a",
    hours=1
)

# Specific metric
cpu_data = await metrics_provider.get_metric_data(
    metric_name="compute.googleapis.com/instance/cpu/utilization",
    namespace="gce_instance",
    start_time=start_time,
    end_time=end_time,
    dimensions={"instance_name": "my-instance", "zone": "us-central1-a"}
)

# Alert policies
alerts = await metrics_provider.get_alarms()
```

## Resource Types

### Cloud Logging Resources
- `gce_instance`: Compute Engine instances
- `gke_cluster`: Google Kubernetes Engine clusters
- `k8s_container`: Kubernetes containers
- `cloud_function`: Cloud Functions
- `gae_app`: App Engine applications
- `cloud_run_revision`: Cloud Run services

### Common Log Names
- `cloudsql.googleapis.com%2Fmysql.err`: Cloud SQL MySQL errors
- `compute.googleapis.com%2Factivity_log`: Compute Engine activity
- `container.googleapis.com%2Fcluster-autoscaler-visibility`: GKE autoscaler
- `run.googleapis.com%2Frequests`: Cloud Run requests
- `appengine.googleapis.com%2Frequest_log`: App Engine requests

## Common Metrics

### Compute Engine
- `compute.googleapis.com/instance/cpu/utilization`: CPU usage
- `compute.googleapis.com/instance/disk/read_bytes_count`: Disk reads
- `compute.googleapis.com/instance/disk/write_bytes_count`: Disk writes
- `compute.googleapis.com/instance/network/received_bytes_count`: Network in
- `compute.googleapis.com/instance/network/sent_bytes_count`: Network out

### Google Kubernetes Engine
- `kubernetes.io/container/cpu/core_usage_time`: Container CPU
- `kubernetes.io/container/memory/used_bytes`: Container memory
- `kubernetes.io/node/cpu/core_usage_time`: Node CPU
- `kubernetes.io/node/memory/used_bytes`: Node memory

### Cloud Functions
- `cloudfunctions.googleapis.com/function/execution_count`: Executions
- `cloudfunctions.googleapis.com/function/execution_times`: Duration
- `cloudfunctions.googleapis.com/function/user_memory_bytes`: Memory usage

### Cloud Run
- `run.googleapis.com/container/cpu/utilizations`: CPU utilization
- `run.googleapis.com/container/memory/utilizations`: Memory utilization
- `run.googleapis.com/request_count`: Request count

## Filter Expressions

Cloud Logging uses a powerful filter syntax:

```
# Severity filtering
severity >= "ERROR"

# Resource type filtering
resource.type="gce_instance"

# Text search
textPayload:"database connection failed"

# JSON payload search
jsonPayload.message:"error"

# Time range (handled automatically)
timestamp >= "2024-01-01T00:00:00Z"

# Combining filters
severity >= "WARNING" AND resource.type="k8s_container" AND resource.labels.cluster_name="prod-cluster"
```

## Error Handling

The integration handles common GCP-specific errors:

- **Authentication errors**: Invalid credentials or missing service account
- **Permission errors**: Insufficient IAM roles
- **API errors**: Quota exceeded or service unavailable
- **Resource errors**: Non-existent projects or resources

## Limitations

1. **Alert History**: Incident history requires additional API integration
2. **Cross-project**: Each provider instance works with a single project
3. **Quota Limits**: Subject to Cloud Logging and Monitoring API quotas
4. **Log Retention**: Limited by Cloud Logging retention policies

## Dependencies

```bash
pip install google-cloud-logging google-cloud-monitoring
```

## References

- [Cloud Logging Documentation](https://cloud.google.com/logging/docs)
- [Cloud Monitoring Documentation](https://cloud.google.com/monitoring/docs)
- [Logging Filter Syntax](https://cloud.google.com/logging/docs/view/logging-query-language)
- [Google Cloud Python SDK](https://github.com/googleapis/google-cloud-python)
