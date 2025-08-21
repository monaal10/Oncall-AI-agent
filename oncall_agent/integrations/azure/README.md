# Azure Monitor Integration

This module provides integration with Azure Monitor services for the OnCall AI Agent.

## Features

### Azure Monitor Logs
- Query logs using KQL (Kusto Query Language)
- Search across multiple Log Analytics tables
- Filter by time ranges and patterns
- Support for Application Insights logs
- Extract structured log data with metadata

### Azure Monitor Metrics
- Query metrics from Azure resources
- Support for Virtual Machines, App Services, and other resources
- Time-series data with configurable aggregations
- Resource-specific metric helpers

## Configuration

### Authentication Methods

1. **Default Azure Credentials (Recommended)**
   ```bash
   az login
   ```

2. **Service Principal**
   ```bash
   # Create service principal
   az ad sp create-for-rbac --name "oncall-agent"
   
   # Set environment variables
   export AZURE_TENANT_ID="your-tenant-id"
   export AZURE_CLIENT_ID="your-client-id"
   export AZURE_CLIENT_SECRET="your-client-secret"
   ```

3. **Managed Identity**
   - Automatically available on Azure VMs, App Services, etc.

### Required Permissions

- **Log Analytics Reader** role on Log Analytics workspaces
- **Monitoring Reader** role on subscription or resource groups
- **Reader** role on specific resources for metrics

### Configuration Example

```yaml
integrations:
  log_provider:
    type: "azure_monitor"
    config:
      subscription_id: "${AZURE_SUBSCRIPTION_ID}"
      workspace_id: "${AZURE_WORKSPACE_ID}"
      # Optional for service principal auth
      tenant_id: "${AZURE_TENANT_ID}"
      client_id: "${AZURE_CLIENT_ID}"
      client_secret: "${AZURE_CLIENT_SECRET}"
```

## Usage Examples

### Logs

```python
from oncall_agent.integrations.azure import AzureMonitorLogsProvider, create_azure_config

config = create_azure_config(
    subscription_id="your-subscription-id",
    workspace_id="your-workspace-id"
)

logs_provider = AzureMonitorLogsProvider(config)

# KQL query
logs = await logs_provider.fetch_logs(
    query="AzureActivity | where Level == 'Error' | take 10",
    start_time=start_time,
    end_time=end_time
)

# Pattern search
errors = await logs_provider.search_logs_by_pattern(
    pattern="exception",
    start_time=start_time,
    end_time=end_time
)
```

### Metrics

```python
from oncall_agent.integrations.azure import AzureMonitorMetricsProvider

metrics_provider = AzureMonitorMetricsProvider(config)

# VM metrics
vm_metrics = await metrics_provider.get_vm_metrics(
    resource_group="my-rg",
    vm_name="my-vm",
    hours=1
)

# Specific metric
cpu_data = await metrics_provider.get_metric_data(
    metric_name="Percentage CPU",
    namespace="/subscriptions/.../virtualMachines/my-vm",
    start_time=start_time,
    end_time=end_time
)
```

## Common Log Tables

- **AzureActivity**: Azure Resource Manager operations
- **AppServiceConsoleLogs**: App Service application logs
- **AppServiceHTTPLogs**: App Service HTTP access logs
- **ContainerLog**: Container instance logs
- **Syslog**: Linux system logs
- **Event**: Windows event logs
- **SecurityEvent**: Windows security events

## Common Metrics

### Virtual Machines
- `Percentage CPU`
- `Network In`
- `Network Out`
- `Disk Read Bytes`
- `Disk Write Bytes`

### App Services
- `CpuTime`
- `Requests`
- `BytesReceived`
- `BytesSent`
- `Http2xx`, `Http3xx`, `Http4xx`, `Http5xx`

### Storage Accounts
- `UsedCapacity`
- `Transactions`
- `Ingress`
- `Egress`

## Error Handling

The integration handles common Azure-specific errors:

- **Authentication errors**: Invalid credentials or expired tokens
- **Permission errors**: Insufficient roles or access rights
- **API errors**: Azure service limits or temporary failures
- **Resource errors**: Non-existent workspaces or resources

## Limitations

1. **Alert Rules**: Full alert rule querying requires Azure Resource Manager API integration
2. **Alert History**: Requires Activity Log API integration
3. **Cross-subscription**: Each provider instance works with a single subscription
4. **Rate Limits**: Subject to Azure Monitor API rate limits

## Dependencies

```bash
pip install azure-identity azure-monitor-query
```

## References

- [Azure Monitor Documentation](https://docs.microsoft.com/en-us/azure/azure-monitor/)
- [KQL Reference](https://docs.microsoft.com/en-us/azure/data-explorer/kusto/query/)
- [Azure Monitor Python SDK](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/monitor)
