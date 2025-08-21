"""Example usage of GCP Cloud integrations."""

import asyncio
from datetime import datetime, timedelta
from oncall_agent.integrations.gcp import (
    GCPCloudLoggingProvider,
    GCPCloudMonitoringProvider,
    create_gcp_config
)


async def gcp_cloud_logging_example():
    """Example of using GCP Cloud Logging integration."""
    print("=== GCP Cloud Logging Example ===")
    
    # Create GCP configuration
    config = create_gcp_config(
        project_id="your-project-id",
        # Optionally specify service account file
        # credentials_path="/path/to/service-account.json"
    )
    
    # Initialize GCP Cloud Logging provider
    logs_provider = GCPCloudLoggingProvider(config)
    
    try:
        # Test connection
        is_healthy = await logs_provider.health_check()
        print(f"GCP Cloud Logging connection: {'✓ Healthy' if is_healthy else '✗ Failed'}")
        
        if not is_healthy:
            return
        
        # Get available log names
        print("\n--- Available Log Names ---")
        log_names = await logs_provider.get_log_groups()
        for log_name in log_names[:5]:  # Show first 5
            print(f"  - {log_name}")
        print(f"  ... and {len(log_names) - 5} more" if len(log_names) > 5 else "")
        
        # Search for recent errors
        print("\n--- Recent Error Logs ---")
        error_logs = await logs_provider.get_recent_errors(hours=1)
        
        for log in error_logs[:5]:  # Show first 5
            print(f"  [{log['timestamp']}] {log['level']}: {log['message'][:80]}...")
        
        if not error_logs:
            print("  No error logs found in the last hour")
        
        # Example Cloud Logging filter query
        print("\n--- Cloud Logging Filter Query Example ---")
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)
        
        filter_query = 'severity >= "WARNING" AND resource.type="gce_instance"'
        
        query_results = await logs_provider.fetch_logs(
            query=filter_query,
            start_time=start_time,
            end_time=end_time,
            limit=5
        )
        
        for log in query_results:
            print(f"  [{log['timestamp']}] {log['level']}: {log['message'][:60]}...")
        
        if not query_results:
            print("  No results from filter query")
        
        # App Engine logs (if available)
        print("\n--- App Engine Logs ---")
        try:
            app_engine_logs = await logs_provider.get_app_engine_logs(
                start_time=start_time,
                end_time=end_time,
                limit=3
            )
            
            for log in app_engine_logs:
                print(f"  [{log['timestamp']}] {log['level']}: {log['message'][:60]}...")
            
            if not app_engine_logs:
                print("  No App Engine logs found")
                
        except Exception as e:
            print(f"  Could not fetch App Engine logs: {e}")
        
        # GKE logs (if available)
        print("\n--- GKE Logs ---")
        try:
            gke_logs = await logs_provider.get_gke_logs(
                cluster_name="your-cluster-name",  # Replace with actual cluster
                start_time=start_time,
                end_time=end_time,
                limit=3
            )
            
            for log in gke_logs:
                print(f"  [{log['timestamp']}] {log['level']}: {log['message'][:60]}...")
            
            if not gke_logs:
                print("  No GKE logs found (or cluster doesn't exist)")
                
        except Exception as e:
            print(f"  Could not fetch GKE logs: {e}")
        
        # Cloud Functions logs (if available)
        print("\n--- Cloud Functions Logs ---")
        try:
            cf_logs = await logs_provider.get_cloud_function_logs(
                function_name="your-function-name",  # Replace with actual function
                start_time=start_time,
                end_time=end_time,
                limit=3
            )
            
            for log in cf_logs:
                print(f"  [{log['timestamp']}] {log['level']}: {log['message'][:60]}...")
            
            if not cf_logs:
                print("  No Cloud Functions logs found (or function doesn't exist)")
                
        except Exception as e:
            print(f"  Could not fetch Cloud Functions logs: {e}")
            
    except Exception as e:
        print(f"Error: {e}")


async def gcp_cloud_monitoring_example():
    """Example of using GCP Cloud Monitoring integration."""
    print("\n=== GCP Cloud Monitoring Example ===")
    
    # Create GCP configuration
    config = create_gcp_config(project_id="your-project-id")
    
    # Initialize GCP Cloud Monitoring provider
    metrics_provider = GCPCloudMonitoringProvider(config)
    
    try:
        # Test connection
        is_healthy = await metrics_provider.health_check()
        print(f"GCP Cloud Monitoring connection: {'✓ Healthy' if is_healthy else '✗ Failed'}")
        
        if not is_healthy:
            return
        
        # Get alert policies
        print("\n--- Alert Policies ---")
        try:
            alert_policies = await metrics_provider.get_alarms()
            
            if alert_policies:
                for policy in alert_policies[:5]:  # Show first 5
                    status_icon = "✅" if policy["state"] == "enabled" else "❌"
                    print(f"  {status_icon} {policy['display_name']}: {policy['state']}")
                    if policy['conditions']:
                        print(f"     Conditions: {len(policy['conditions'])}")
            else:
                print("  No alert policies found")
                
        except Exception as e:
            print(f"  Could not fetch alert policies: {e}")
        
        # Example: Get Compute Engine metrics
        print("\n--- Compute Engine Metrics Example ---")
        try:
            instance_metrics = await metrics_provider.get_compute_instance_metrics(
                instance_name="your-instance-name",  # Replace with actual instance
                zone="us-central1-a",  # Replace with actual zone
                hours=1
            )
            
            print(f"  Instance: {instance_metrics['instance_name']}")
            print(f"  Zone: {instance_metrics['zone']}")
            
            for metric_name, data_points in instance_metrics['metrics'].items():
                if data_points:
                    latest = data_points[-1]
                    metric_short = metric_name.split('/')[-1]
                    print(f"    {metric_short}: {latest['value']:.2f} {latest['unit']}")
                    
        except Exception as e:
            print(f"  Could not fetch Compute Engine metrics: {e}")
        
        # Example: Get GKE metrics
        print("\n--- GKE Cluster Metrics Example ---")
        try:
            cluster_metrics = await metrics_provider.get_gke_cluster_metrics(
                cluster_name="your-cluster-name",  # Replace with actual cluster
                location="us-central1-a",  # Replace with actual location
                hours=1
            )
            
            print(f"  Cluster: {cluster_metrics['cluster_name']}")
            print(f"  Location: {cluster_metrics['location']}")
            
            for metric_name, data_points in cluster_metrics['metrics'].items():
                if data_points:
                    latest = data_points[-1]
                    metric_short = metric_name.split('/')[-1]
                    print(f"    {metric_short}: {latest['value']:.2f} {latest['unit']}")
                    
        except Exception as e:
            print(f"  Could not fetch GKE metrics: {e}")
        
        # Example: Get Cloud Functions metrics
        print("\n--- Cloud Functions Metrics Example ---")
        try:
            function_metrics = await metrics_provider.get_cloud_function_metrics(
                function_name="your-function-name",  # Replace with actual function
                region="us-central1",  # Replace with actual region
                hours=1
            )
            
            print(f"  Function: {function_metrics['function_name']}")
            print(f"  Region: {function_metrics['region']}")
            
            for metric_name, data_points in function_metrics['metrics'].items():
                if data_points:
                    latest = data_points[-1]
                    metric_short = metric_name.split('/')[-1]
                    print(f"    {metric_short}: {latest['value']:.2f} {latest['unit']}")
                    
        except Exception as e:
            print(f"  Could not fetch Cloud Functions metrics: {e}")
        
        # Example: List available metrics
        print("\n--- Available Metrics (Sample) ---")
        try:
            available_metrics = await metrics_provider.list_metrics()
            
            print(f"  Found {len(available_metrics)} total metrics")
            
            # Group by namespace and show samples
            namespaces = {}
            for metric in available_metrics:
                namespace = metric['namespace']
                if namespace not in namespaces:
                    namespaces[namespace] = []
                namespaces[namespace].append(metric['metric_name'])
            
            for namespace, metrics in list(namespaces.items())[:5]:  # Show first 5 namespaces
                print(f"    {namespace}: {len(metrics)} metrics")
                for metric in metrics[:2]:  # Show first 2 metrics per namespace
                    print(f"      - {metric}")
                    
        except Exception as e:
            print(f"  Could not list metrics: {e}")
        
        # Example: Get specific metric data
        print("\n--- Specific Metric Data ---")
        try:
            # Example: CPU utilization for Compute Engine
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=2)
            
            cpu_data = await metrics_provider.get_metric_data(
                metric_name="compute.googleapis.com/instance/cpu/utilization",
                namespace="gce_instance",
                start_time=start_time,
                end_time=end_time,
                dimensions={"instance_name": "your-instance-name", "zone": "us-central1-a"},
                statistic="Average",
                period=300
            )
            
            if cpu_data:
                print(f"  Found {len(cpu_data)} CPU data points")
                for point in cpu_data[-3:]:  # Show latest 3 points
                    print(f"    [{point['timestamp']}] {point['value']:.4f} (utilization ratio)")
            else:
                print("  No CPU data available (instance might not exist)")
                
        except Exception as e:
            print(f"  Could not fetch CPU data: {e}")
            
    except Exception as e:
        print(f"Error: {e}")


async def main():
    """Run all GCP Cloud examples."""
    print("GCP Cloud Integration Examples")
    print("=" * 50)
    
    await gcp_cloud_logging_example()
    await gcp_cloud_monitoring_example()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nNote: Make sure you have GCP credentials configured:")
    print("  - gcloud CLI: gcloud auth application-default login")
    print("  - Service Account: Set GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json")
    print("  - Managed Identity (for GCE/GKE/Cloud Run)")
    print("  - Update project_id and resource names in the examples")
    print("\nRequired IAM roles:")
    print("  - roles/logging.viewer (for Cloud Logging)")
    print("  - roles/monitoring.viewer (for Cloud Monitoring)")


if __name__ == "__main__":
    asyncio.run(main())
