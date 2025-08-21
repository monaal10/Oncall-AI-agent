"""AWS CloudWatch Logs integration."""

import asyncio
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from ..base.log_provider import LogProvider


class CloudWatchLogsProvider(LogProvider):
    """AWS CloudWatch Logs provider implementation.
    
    Provides integration with AWS CloudWatch Logs service for fetching
    and searching log data.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize CloudWatch Logs provider.
        
        Args:
            config: Configuration dictionary containing:
                - region: AWS region (required)
                - access_key_id: AWS access key (optional, uses default credentials)
                - secret_access_key: AWS secret key (optional, uses default credentials)
                - session_token: AWS session token (optional)
                - log_groups: List of log groups to search (optional)
        """
        super().__init__(config)
        self._setup_client()

    def _validate_config(self) -> None:
        """Validate AWS CloudWatch Logs configuration.
        
        Raises:
            ValueError: If required configuration is missing
        """
        if "region" not in self.config:
            raise ValueError("AWS region is required for CloudWatch Logs provider")

    def _setup_client(self) -> None:
        """Set up AWS CloudWatch Logs client.
        
        Raises:
            NoCredentialsError: If AWS credentials are not available
        """
        session_kwargs = {"region_name": self.config["region"]}
        
        if "access_key_id" in self.config:
            session_kwargs["aws_access_key_id"] = self.config["access_key_id"]
        if "secret_access_key" in self.config:
            session_kwargs["aws_secret_access_key"] = self.config["secret_access_key"]
        if "session_token" in self.config:
            session_kwargs["aws_session_token"] = self.config["session_token"]

        self.client = boto3.client("logs", **session_kwargs)

    async def fetch_logs(
        self,
        query: str,
        start_time: datetime,
        end_time: datetime,
        limit: Optional[int] = 1000,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Fetch logs using CloudWatch Logs Insights query.
        
        Args:
            query: CloudWatch Logs Insights query string
            start_time: Start of time range to search
            end_time: End of time range to search
            limit: Maximum number of log entries to return
            **kwargs: Additional parameters:
                - log_groups: List of log groups to query
                
        Returns:
            List of log entries, each containing:
            - timestamp: Log entry timestamp as datetime
            - message: Log message content
            - level: Log level extracted from message
            - source: Log group name
            - metadata: Additional fields from the log entry
            
        Raises:
            ConnectionError: If unable to connect to CloudWatch Logs
            ValueError: If query parameters are invalid
        """
        try:
            log_groups = kwargs.get("log_groups", self.config.get("log_groups", []))
            if not log_groups:
                # If no log groups specified, get all available ones
                log_groups = await self.get_log_groups()

            # Convert datetime to epoch milliseconds
            start_timestamp = int(start_time.timestamp() * 1000)
            end_timestamp = int(end_time.timestamp() * 1000)

            # Start the query
            response = await asyncio.to_thread(
                self.client.start_query,
                logGroupNames=log_groups,
                startTime=start_timestamp,
                endTime=end_timestamp,
                queryString=query,
                limit=limit or 1000
            )

            query_id = response["queryId"]

            # Wait for query completion
            while True:
                result = await asyncio.to_thread(
                    self.client.get_query_results,
                    queryId=query_id
                )
                
                if result["status"] in ["Complete", "Failed", "Cancelled"]:
                    break
                
                await asyncio.sleep(1)

            if result["status"] != "Complete":
                raise ConnectionError(f"Query failed with status: {result['status']}")

            # Process results
            logs = []
            for result_row in result["results"]:
                log_entry = self._process_log_entry(result_row)
                logs.append(log_entry)

            return logs

        except ClientError as e:
            raise ConnectionError(f"AWS CloudWatch Logs error: {e}")
        except Exception as e:
            raise ConnectionError(f"Failed to fetch logs: {e}")

    async def search_logs_by_pattern(
        self,
        pattern: str,
        start_time: datetime,
        end_time: datetime,
        log_groups: Optional[List[str]] = None,
        limit: Optional[int] = 1000
    ) -> List[Dict[str, Any]]:
        """Search logs using pattern matching with CloudWatch filter patterns.
        
        Args:
            pattern: CloudWatch filter pattern or simple text search
            start_time: Start of time range to search
            end_time: End of time range to search
            log_groups: Specific log groups to search
            limit: Maximum number of matches to return
            
        Returns:
            List of matching log entries with same structure as fetch_logs
            
        Raises:
            ConnectionError: If unable to connect to CloudWatch Logs
            ValueError: If pattern or parameters are invalid
        """
        try:
            target_log_groups = log_groups or self.config.get("log_groups", [])
            if not target_log_groups:
                target_log_groups = await self.get_log_groups()

            # Convert datetime to epoch milliseconds
            start_timestamp = int(start_time.timestamp() * 1000)
            end_timestamp = int(end_time.timestamp() * 1000)

            all_logs = []
            
            for log_group in target_log_groups:
                try:
                    response = await asyncio.to_thread(
                        self.client.filter_log_events,
                        logGroupName=log_group,
                        startTime=start_timestamp,
                        endTime=end_timestamp,
                        filterPattern=pattern,
                        limit=limit or 1000
                    )

                    for event in response.get("events", []):
                        log_entry = {
                            "timestamp": datetime.fromtimestamp(event["timestamp"] / 1000),
                            "message": event["message"],
                            "level": self._extract_log_level(event["message"]),
                            "source": log_group,
                            "metadata": {
                                "log_stream": event.get("logStreamName"),
                                "event_id": event.get("eventId"),
                                "ingestion_time": datetime.fromtimestamp(
                                    event["ingestionTime"] / 1000
                                ) if "ingestionTime" in event else None
                            }
                        }
                        all_logs.append(log_entry)

                except ClientError as e:
                    # Log the error but continue with other log groups
                    print(f"Error searching log group {log_group}: {e}")
                    continue

            # Sort by timestamp and apply limit
            all_logs.sort(key=lambda x: x["timestamp"])
            return all_logs[:limit] if limit else all_logs

        except ClientError as e:
            raise ConnectionError(f"AWS CloudWatch Logs error: {e}")
        except Exception as e:
            raise ConnectionError(f"Failed to search logs: {e}")

    async def get_log_groups(self) -> List[str]:
        """Get available CloudWatch log groups.
        
        Returns:
            List of log group names
            
        Raises:
            ConnectionError: If unable to connect to CloudWatch Logs
        """
        try:
            log_groups = []
            next_token = None

            while True:
                kwargs = {}
                if next_token:
                    kwargs["nextToken"] = next_token

                response = await asyncio.to_thread(
                    self.client.describe_log_groups,
                    **kwargs
                )

                for log_group in response.get("logGroups", []):
                    log_groups.append(log_group["logGroupName"])

                next_token = response.get("nextToken")
                if not next_token:
                    break

            return log_groups

        except ClientError as e:
            raise ConnectionError(f"AWS CloudWatch Logs error: {e}")
        except Exception as e:
            raise ConnectionError(f"Failed to get log groups: {e}")

    def _process_log_entry(self, result_row: List[Dict[str, str]]) -> Dict[str, Any]:
        """Process a CloudWatch Logs Insights query result row.
        
        Args:
            result_row: Raw result row from CloudWatch Logs Insights
            
        Returns:
            Processed log entry dictionary
        """
        log_entry = {
            "timestamp": None,
            "message": "",
            "level": "INFO",
            "source": "",
            "metadata": {}
        }

        for field in result_row:
            field_name = field.get("field", "")
            field_value = field.get("value", "")

            if field_name == "@timestamp":
                try:
                    log_entry["timestamp"] = datetime.fromisoformat(
                        field_value.replace("Z", "+00:00")
                    )
                except ValueError:
                    log_entry["timestamp"] = datetime.now()
            elif field_name == "@message":
                log_entry["message"] = field_value
                log_entry["level"] = self._extract_log_level(field_value)
            elif field_name == "@logStream":
                log_entry["source"] = field_value
            else:
                log_entry["metadata"][field_name] = field_value

        return log_entry

    def _extract_log_level(self, message: str) -> str:
        """Extract log level from log message.
        
        Args:
            message: Log message content
            
        Returns:
            Extracted log level (ERROR, WARN, INFO, DEBUG, TRACE)
        """
        message_upper = message.upper()
        
        if any(level in message_upper for level in ["ERROR", "FATAL", "EXCEPTION"]):
            return "ERROR"
        elif any(level in message_upper for level in ["WARN", "WARNING"]):
            return "WARN"
        elif "DEBUG" in message_upper:
            return "DEBUG"
        elif "TRACE" in message_upper:
            return "TRACE"
        else:
            return "INFO"

    async def get_recent_errors(
        self,
        hours: int = 1,
        log_groups: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get recent error logs from specified time range.
        
        Args:
            hours: Number of hours to look back
            log_groups: Specific log groups to search
            
        Returns:
            List of error log entries
            
        Raises:
            ConnectionError: If unable to connect to CloudWatch Logs
        """
        from datetime import timedelta
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        error_pattern = "ERROR"
        
        return await self.search_logs_by_pattern(
            pattern=error_pattern,
            start_time=start_time,
            end_time=end_time,
            log_groups=log_groups,
            limit=100
        )
