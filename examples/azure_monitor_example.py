"""Example usage of Azure Monitor integrations."""

import asyncio
from datetime import datetime, timedelta
from oncall_agent.integrations.azure import (
    AzureMonitorLogsProvider,
    AzureMonitorMetricsProvider,
    create_azure_config
)


async def azure_monitor_logs_example():
    """Example of using Azure Monitor Logs integration."""
    print("=== Azure Monitor Logs Example ===")
    
    # Create Azure configuration
    config = create_azure_config(
        subscription_id="your-subscription-id",
        workspace_id="your-workspace-id",
        # Optionally specify service principal credentials
        # tenant_id="your-tenant-id",
        # client_id="your-client-id",
        # client_secret="your-client-secret"
    )
    
    # Initialize Azure Monitor Logs provider
    logs_provider = AzureMonitorLogsProvider(config)
    
    try:
        # Test connection
        is_healthy = await logs_provider.health_check()
        print(f"Azure Monitor Logs connection: {'✓ Healthy' if is_healthy else '✗ Failed'}")
        
        if not is_healthy:
            return
        
        # Get available tables
        print("\n--- Available Log Tables ---")
        tables = await logs_provider.get_log_groups()
        for table in tables[:5]:  # Show first 5
            print(f"  - {table}")
        print(f"  ... and {len(tables) - 5} more" if len(tables) > 5 else "")
        
        # Search for recent errors
        print("\n--- Recent Error Logs ---")
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)
        
        error_logs = await logs_provider.get_recent_errors(hours=1)
        
        for log in error_logs[:5]:  # Show first 5
            print(f"  [{log['timestamp']}] {log['source']}: {log['message'][:80]}...")
        
        if not error_logs:
            print("  No error logs found in the last hour")
        
        # Example KQL query
        print("\n--- KQL Query Example ---")
        kql_query = """
        AzureActivity
        | where TimeGenerated > ago(1h)
        | where Level == "Error"
        | project TimeGenerated, Caller, OperationName, ActivityStatus
        | order by TimeGenerated desc
        """
        
        query_results = await logs_provider.fetch_logs(
            query=kql_query,
            start_time=start_time,
            end_time=end_time,
            limit=5
        )
        
        for log in query_results:
            print(f"  [{log['timestamp']}] {log['message'][:60]}...")
        
        if not query_results:
            print("  No results from KQL query")
        
        # Application Insights logs (if available)
        print("\n--- Application Insights Logs ---")
        try:
            app_insights_logs = await logs_provider.get_application_insights_logs(
                start_time=start_time,
                end_time=end_time,
                limit=3
            )
            
            for log in app_insights_logs:
                print(f"  [{log['timestamp']}] {log['level']}: {log['message'][:60]}...")
            
            if not app_insights_logs:
                print("  No Application Insights logs found")
                
        except Exception as e:
            print(f"  Could not fetch Application Insights logs: {e}")
            
    except Exception as e:
        print(f"Error: {e}")


async def azure_monitor_metrics_example():
    """Example of using Azure Monitor Metrics integration."""
    print("\n=== Azure Monitor Metrics Example ===")
    
    # Create Azure configuration
    config = create_azure_config(subscription_id="your-subscription-id")
    
    # Initialize Azure Monitor Metrics provider
    metrics_provider = AzureMonitorMetricsProvider(config)
    
    try:
        # Test connection
        is_healthy = await metrics_provider.health_check()
        print(f"Azure Monitor Metrics connection: {'✓ Healthy' if is_healthy else '✗ Failed'}")
        
        if not is_healthy:
            return
        
        # Example: Get VM metrics
        print("\n--- Virtual Machine Metrics Example ---")
        try:
            vm_metrics = await metrics_provider.get_vm_metrics(
                resource_group="your-resource-group",
                vm_name="your-vm-name",
                hours=1
            )
            
            print(f"  VM: {vm_metrics['resource_uri']}")
            print(f"  Available metrics: {vm_metrics['total_available_metrics']}")
            
            for metric_name, data_points in vm_metrics['sample_data'].items():
                if data_points:
                    latest = data_points[-1]
                    print(f"    {metric_name}: {latest['value']:.2f} {latest['unit']}")
                    
        except Exception as e:
            print(f"  Could not fetch VM metrics: {e}")
        
        # Example: Get App Service metrics
        print("\n--- App Service Metrics Example ---")
        try:
            app_metrics = await metrics_provider.get_app_service_metrics(
                resource_group="your-resource-group",
                app_name="your-app-name",
                hours=1
            )
            
            print(f"  App Service: {app_metrics['resource_uri']}")
            print(f"  Available metrics: {app_metrics['total_available_metrics']}")
            
            for metric_name, data_points in app_metrics['sample_data'].items():
                if data_points:
                    latest = data_points[-1]
                    print(f"    {metric_name}: {latest['value']:.2f} {latest['unit']}")
                    
        except Exception as e:
            print(f"  Could not fetch App Service metrics: {e}")
        
        # Example: List available metrics for a resource
        print("\n--- Available Metrics for Resource ---")
        try:
            # Example resource URI (replace with actual resource)
            resource_uri = "/subscriptions/your-subscription-id/resourceGroups/your-rg/providers/Microsoft.Compute/virtualMachines/your-vm"
            
            available_metrics = await metrics_provider.list_metrics(namespace=resource_uri)
            
            print(f"  Found {len(available_metrics)} available metrics")
            for metric in available_metrics[:5]:  # Show first 5
                print(f"    - {metric['metric_name']}: {metric['description'][:50]}...")
                
        except Exception as e:
            print(f"  Could not list metrics: {e}")
        
        # Example: Get specific metric data
        print("\n--- Specific Metric Data ---")
        try:
            # Example: CPU utilization for a VM
            resource_uri = "/subscriptions/your-subscription-id/resourceGroups/your-rg/providers/Microsoft.Compute/virtualMachines/your-vm"
            
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=2)
            
            cpu_data = await metrics_provider.get_metric_data(
                metric_name="Percentage CPU",
                namespace=resource_uri,
                start_time=start_time,
                end_time=end_time,
                statistic="Average",
                period=300
            )
            
            if cpu_data:
                print(f"  Found {len(cpu_data)} CPU data points")
                for point in cpu_data[-3:]:  # Show latest 3 points
                    print(f"    [{point['timestamp']}] {point['value']:.2f}% CPU")
            else:
                print("  No CPU data available")
                
        except Exception as e:
            print(f"  Could not fetch CPU data: {e}")
            
    except Exception as e:
        print(f"Error: {e}")


async def main():
    """Run all Azure Monitor examples."""
    print("Azure Monitor Integration Examples")
    print("=" * 50)
    
    await azure_monitor_logs_example()
    await azure_monitor_metrics_example()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nNote: Make sure you have Azure credentials configured:")
    print("  - Azure CLI: az login")
    print("  - Service Principal: Set AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID")
    print("  - Managed Identity (for Azure VMs/App Services)")
    print("  - Update subscription_id and workspace_id in the examples")


if __name__ == "__main__":
    asyncio.run(main())
