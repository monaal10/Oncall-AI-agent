"""Example usage of AWS CloudWatch integrations."""

import asyncio
from datetime import datetime, timedelta
from oncall_agent.integrations.aws import (
    CloudWatchLogsProvider,
    CloudWatchMetricsProvider,
    create_aws_config
)


async def cloudwatch_logs_example():
    """Example of using CloudWatch Logs integration."""
    print("=== CloudWatch Logs Example ===")
    
    # Create AWS configuration
    config = create_aws_config(
        region="us-west-2",
        # Optionally specify credentials or use default AWS credentials
        # access_key_id="your-access-key",
        # secret_access_key="your-secret-key"
    )
    
    # Initialize CloudWatch Logs provider
    logs_provider = CloudWatchLogsProvider(config)
    
    try:
        # Test connection
        is_healthy = await logs_provider.health_check()
        print(f"CloudWatch Logs connection: {'✓ Healthy' if is_healthy else '✗ Failed'}")
        
        if not is_healthy:
            return
        
        # Get available log groups
        print("\n--- Available Log Groups ---")
        log_groups = await logs_provider.get_log_groups()
        for group in log_groups[:5]:  # Show first 5
            print(f"  - {group}")
        print(f"  ... and {len(log_groups) - 5} more" if len(log_groups) > 5 else "")
        
        # Search for recent errors
        print("\n--- Recent Error Logs ---")
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)
        
        error_logs = await logs_provider.search_logs_by_pattern(
            pattern="ERROR",
            start_time=start_time,
            end_time=end_time,
            limit=5
        )
        
        for log in error_logs:
            print(f"  [{log['timestamp']}] {log['source']}: {log['message'][:100]}...")
        
        if not error_logs:
            print("  No error logs found in the last hour")
        
        # Example CloudWatch Insights query
        print("\n--- CloudWatch Insights Query Example ---")
        query = """
        fields @timestamp, @message, @logStream
        | filter @message like /ERROR/
        | sort @timestamp desc
        | limit 3
        """
        
        insights_logs = await logs_provider.fetch_logs(
            query=query,
            start_time=start_time,
            end_time=end_time,
            limit=3
        )
        
        for log in insights_logs:
            print(f"  [{log['timestamp']}] {log['message'][:80]}...")
        
        if not insights_logs:
            print("  No results from Insights query")
            
    except Exception as e:
        print(f"Error: {e}")


async def cloudwatch_metrics_example():
    """Example of using CloudWatch Metrics integration."""
    print("\n=== CloudWatch Metrics Example ===")
    
    # Create AWS configuration
    config = create_aws_config(region="us-west-2")
    
    # Initialize CloudWatch Metrics provider
    metrics_provider = CloudWatchMetricsProvider(config)
    
    try:
        # Test connection
        is_healthy = await metrics_provider.health_check()
        print(f"CloudWatch Metrics connection: {'✓ Healthy' if is_healthy else '✗ Failed'}")
        
        if not is_healthy:
            return
        
        # Get current alarms
        print("\n--- Current Alarms ---")
        alarms = await metrics_provider.get_alarms()
        
        if alarms:
            for alarm in alarms[:5]:  # Show first 5
                state_icon = "🚨" if alarm["state"] == "ALARM" else "✅" if alarm["state"] == "OK" else "⚠️"
                print(f"  {state_icon} {alarm['name']}: {alarm['state']} - {alarm['reason']}")
        else:
            print("  No alarms found")
        
        # Get alarming metrics
        print("\n--- Metrics in ALARM State ---")
        alarming_metrics = await metrics_provider.get_alarming_metrics()
        
        for metric in alarming_metrics:
            print(f"  🚨 {metric['namespace']}/{metric['metric_name']}")
            print(f"     Alarm: {metric['alarm_name']}")
            print(f"     Reason: {metric['alarm_reason']}")
            print(f"     Threshold: {metric['comparison_operator']} {metric['threshold']}")
        
        if not alarming_metrics:
            print("  No metrics currently in ALARM state")
        
        # Example: Get EC2 CPU utilization
        print("\n--- Sample Metric Data (EC2 CPU) ---")
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=2)
        
        try:
            cpu_data = await metrics_provider.get_metric_data(
                metric_name="CPUUtilization",
                namespace="AWS/EC2",
                start_time=start_time,
                end_time=end_time,
                statistic="Average",
                period=300  # 5 minutes
            )
            
            if cpu_data:
                print(f"  Found {len(cpu_data)} data points")
                # Show latest few points
                for point in cpu_data[-3:]:
                    print(f"    [{point['timestamp']}] {point['value']:.2f}% CPU")
            else:
                print("  No EC2 CPU data available (no EC2 instances or no data)")
                
        except Exception as e:
            print(f"  Could not fetch EC2 CPU data: {e}")
        
        # Get insights for a namespace
        print("\n--- Namespace Insights (AWS/Lambda) ---")
        try:
            insights = await metrics_provider.get_metric_insights("AWS/Lambda", hours=24)
            print(f"  Total metrics: {insights['total_metrics']}")
            print(f"  Total alarms: {insights['total_alarms']}")
            print(f"  Active alarms: {insights['active_alarms']}")
            
            if insights['active_alarms'] > 0:
                print("  Active alarm details:")
                for alarm in insights['alarm_details']:
                    print(f"    - {alarm['name']}: {alarm['reason']}")
                    
        except Exception as e:
            print(f"  Could not get Lambda insights: {e}")
            
    except Exception as e:
        print(f"Error: {e}")


async def main():
    """Run all CloudWatch examples."""
    print("AWS CloudWatch Integration Examples")
    print("=" * 50)
    
    await cloudwatch_logs_example()
    await cloudwatch_metrics_example()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nNote: Make sure you have AWS credentials configured:")
    print("  - AWS CLI: aws configure")
    print("  - Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
    print("  - IAM roles (for EC2/Lambda)")
    print("  - AWS profiles")


if __name__ == "__main__":
    asyncio.run(main())
