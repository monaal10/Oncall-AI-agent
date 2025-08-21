"""Azure Monitor Logs integration."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from azure.monitor.query import LogsQueryClient, LogsQueryStatus
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError

from ..base.log_provider import LogProvider
from .client import AzureClientManager


class AzureMonitorLogsProvider(LogProvider):
    """Azure Monitor Logs provider implementation.
    
    Provides integration with Azure Monitor Logs (Log Analytics) service
    for fetching and searching log data using KQL (Kusto Query Language).
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize Azure Monitor Logs provider.
        
        Args:
            config: Configuration dictionary containing:
                - subscription_id: Azure subscription ID (required)
                - workspace_id: Log Analytics workspace ID (required)
                - tenant_id: Azure AD tenant ID (optional, for service principal)
                - client_id: Azure AD client ID (optional, for service principal)
                - client_secret: Azure AD client secret (optional, for service principal)
                - tables: List of log tables to search (optional)
        """
        super().__init__(config)
        self.client_manager = AzureClientManager(config)
        self.client = self.client_manager.create_logs_client()

    def _validate_config(self) -> None:
        """Validate Azure Monitor Logs configuration.
        
        Raises:
            ValueError: If required configuration is missing
        """
        if "subscription_id" not in self.config:
            raise ValueError("Azure subscription_id is required for Azure Monitor Logs provider")
        if "workspace_id" not in self.config:
            raise ValueError("Log Analytics workspace_id is required for Azure Monitor Logs provider")

    async def fetch_logs(
        self,
        query: str,
        start_time: datetime,
        end_time: datetime,
        limit: Optional[int] = 1000,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Fetch logs using KQL (Kusto Query Language) query.
        
        Args:
            query: KQL query string
            start_time: Start of time range to search
            end_time: End of time range to search
            limit: Maximum number of log entries to return
            **kwargs: Additional parameters (ignored for Azure)
                
        Returns:
            List of log entries, each containing:
            - timestamp: Log entry timestamp as datetime
            - message: Log message content
            - level: Log level extracted from message
            - source: Log table name
            - metadata: Additional fields from the log entry
            
        Raises:
            ConnectionError: If unable to connect to Azure Monitor Logs
            ValueError: If query parameters are invalid
        """
        try:
            # Add limit to query if specified
            if limit and "take" not in query.lower() and "top" not in query.lower():
                query = f"{query} | take {limit}"

            response = await asyncio.to_thread(
                self.client.query_workspace,
                workspace_id=self.config["workspace_id"],
                query=query,
                timespan=(start_time, end_time)
            )

            if response.status != LogsQueryStatus.SUCCESS:
                raise ConnectionError(f"Query failed with status: {response.status}")

            # Process results
            logs = []
            for table in response.tables:
                for row in table.rows:
                    log_entry = self._process_log_entry(row, table.columns)
                    logs.append(log_entry)

            return logs

        except ClientAuthenticationError as e:
            raise ConnectionError(f"Azure authentication error: {e}")
        except HttpResponseError as e:
            raise ConnectionError(f"Azure Monitor Logs API error: {e}")
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
        """Search logs using pattern matching with KQL contains operator.
        
        Args:
            pattern: Text pattern to search for
            start_time: Start of time range to search
            end_time: End of time range to search
            log_groups: Specific tables to search (if not specified, searches common tables)
            limit: Maximum number of matches to return
            
        Returns:
            List of matching log entries with same structure as fetch_logs
            
        Raises:
            ConnectionError: If unable to connect to Azure Monitor Logs
            ValueError: If pattern or parameters are invalid
        """
        try:
            # Use specified tables or default common tables
            tables = log_groups or self.config.get("tables", [
                "AzureActivity",
                "AppServiceConsoleLogs", 
                "AppServiceHTTPLogs",
                "ContainerLog",
                "Syslog",
                "Event"
            ])

            all_logs = []
            
            for table in tables:
                try:
                    # Build KQL query for pattern search
                    query = f"""
                    {table}
                    | where TimeGenerated between(datetime({start_time.isoformat()}) .. datetime({end_time.isoformat()}))
                    | where * contains "{pattern}"
                    | order by TimeGenerated desc
                    """
                    
                    if limit:
                        query += f" | take {limit}"

                    response = await asyncio.to_thread(
                        self.client.query_workspace,
                        workspace_id=self.config["workspace_id"],
                        query=query,
                        timespan=(start_time, end_time)
                    )

                    if response.status == LogsQueryStatus.SUCCESS:
                        for result_table in response.tables:
                            for row in result_table.rows:
                                log_entry = self._process_log_entry(row, result_table.columns, table)
                                all_logs.append(log_entry)

                except Exception as e:
                    # Log the error but continue with other tables
                    print(f"Error searching table {table}: {e}")
                    continue

            # Sort by timestamp and apply limit
            all_logs.sort(key=lambda x: x["timestamp"], reverse=True)
            return all_logs[:limit] if limit else all_logs

        except ClientAuthenticationError as e:
            raise ConnectionError(f"Azure authentication error: {e}")
        except Exception as e:
            raise ConnectionError(f"Failed to search logs: {e}")

    async def get_log_groups(self) -> List[str]:
        """Get available Log Analytics tables.
        
        Returns:
            List of table names available in the workspace
            
        Raises:
            ConnectionError: If unable to connect to Azure Monitor Logs
        """
        try:
            # Query to get all tables in the workspace
            query = """
            search *
            | distinct $table
            | order by $table asc
            """

            end_time = datetime.now()
            start_time = end_time - timedelta(days=1)

            response = await asyncio.to_thread(
                self.client.query_workspace,
                workspace_id=self.config["workspace_id"],
                query=query,
                timespan=(start_time, end_time)
            )

            if response.status != LogsQueryStatus.SUCCESS:
                # Fallback to common table names if query fails
                return [
                    "AzureActivity",
                    "AppServiceConsoleLogs",
                    "AppServiceHTTPLogs", 
                    "ContainerLog",
                    "Syslog",
                    "Event",
                    "SecurityEvent",
                    "Heartbeat"
                ]

            tables = []
            for table in response.tables:
                for row in table.rows:
                    if row and len(row) > 0:
                        tables.append(str(row[0]))

            return sorted(tables)

        except Exception as e:
            # Return common tables as fallback
            return [
                "AzureActivity",
                "AppServiceConsoleLogs",
                "AppServiceHTTPLogs",
                "ContainerLog", 
                "Syslog",
                "Event"
            ]

    def _process_log_entry(
        self, 
        row: List[Any], 
        columns: List[Any],
        table_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a Log Analytics query result row.
        
        Args:
            row: Raw result row from Log Analytics
            columns: Column definitions from the query result
            table_name: Name of the source table
            
        Returns:
            Processed log entry dictionary
        """
        log_entry = {
            "timestamp": datetime.now(),
            "message": "",
            "level": "INFO",
            "source": table_name or "Unknown",
            "metadata": {}
        }

        # Map row values to column names
        for i, column in enumerate(columns):
            if i >= len(row):
                break
                
            column_name = column.name
            column_value = row[i]

            if column_name in ["TimeGenerated", "Timestamp", "EventTime"]:
                if column_value:
                    log_entry["timestamp"] = column_value
            elif column_name in ["Message", "RenderedDescription", "Activity", "OperationName"]:
                if column_value:
                    log_entry["message"] = str(column_value)
                    log_entry["level"] = self._extract_log_level(str(column_value))
            elif column_name == "Level":
                if column_value:
                    log_entry["level"] = str(column_value)
            elif column_name in ["Computer", "Resource", "ResourceGroup", "SubscriptionId"]:
                if column_value:
                    log_entry["source"] = str(column_value)
            else:
                # Add other fields to metadata
                if column_value is not None:
                    log_entry["metadata"][column_name] = column_value

        return log_entry

    def _extract_log_level(self, message: str) -> str:
        """Extract log level from log message.
        
        Args:
            message: Log message content
            
        Returns:
            Extracted log level (ERROR, WARNING, INFO, DEBUG, VERBOSE)
        """
        message_upper = message.upper()
        
        if any(level in message_upper for level in ["ERROR", "FATAL", "EXCEPTION", "FAILED"]):
            return "ERROR"
        elif any(level in message_upper for level in ["WARN", "WARNING"]):
            return "WARNING"
        elif any(level in message_upper for level in ["DEBUG", "TRACE"]):
            return "DEBUG"
        elif "VERBOSE" in message_upper:
            return "VERBOSE"
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
            log_groups: Specific tables to search
            
        Returns:
            List of error log entries
            
        Raises:
            ConnectionError: If unable to connect to Azure Monitor Logs
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # Build KQL query for errors
        tables = log_groups or ["AzureActivity", "AppServiceConsoleLogs", "ContainerLog"]
        
        queries = []
        for table in tables:
            queries.append(f"""
            {table}
            | where TimeGenerated between(datetime({start_time.isoformat()}) .. datetime({end_time.isoformat()}))
            | where * contains "error" or * contains "ERROR" or * contains "exception" or * contains "EXCEPTION"
            """)
        
        # Union all table queries
        union_query = " | union ".join([f"({query})" for query in queries])
        union_query += " | order by TimeGenerated desc | take 100"
        
        return await self.fetch_logs(
            query=union_query,
            start_time=start_time,
            end_time=end_time,
            limit=100
        )

    async def get_application_insights_logs(
        self,
        start_time: datetime,
        end_time: datetime,
        limit: Optional[int] = 1000
    ) -> List[Dict[str, Any]]:
        """Get Application Insights logs (traces, exceptions, requests).
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            limit: Maximum number of entries to return
            
        Returns:
            List of Application Insights log entries
            
        Raises:
            ConnectionError: If unable to connect to Azure Monitor Logs
        """
        query = f"""
        union traces, exceptions, requests
        | where timestamp between(datetime({start_time.isoformat()}) .. datetime({end_time.isoformat()}))
        | order by timestamp desc
        """
        
        if limit:
            query += f" | take {limit}"
        
        return await self.fetch_logs(
            query=query,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
