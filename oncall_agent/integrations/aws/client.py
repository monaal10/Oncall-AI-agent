"""AWS client utilities and configuration."""

from typing import Dict, Any, Optional
import boto3
from botocore.exceptions import NoCredentialsError, ClientError


class AWSClientManager:
    """Manages AWS client creation and configuration.
    
    Provides centralized AWS client creation with proper credential handling
    and error management.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize AWS client manager.
        
        Args:
            config: AWS configuration containing:
                - region: AWS region (required)
                - access_key_id: AWS access key (optional)
                - secret_access_key: AWS secret key (optional)
                - session_token: AWS session token (optional)
                - profile_name: AWS profile name (optional)
        """
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate AWS configuration.
        
        Raises:
            ValueError: If required configuration is missing
        """
        if "region" not in self.config:
            raise ValueError("AWS region is required")

    def create_client(self, service_name: str) -> Any:
        """Create AWS service client.
        
        Args:
            service_name: AWS service name (e.g., 'logs', 'cloudwatch', 's3')
            
        Returns:
            Configured boto3 client for the specified service
            
        Raises:
            NoCredentialsError: If AWS credentials are not available
            ClientError: If client creation fails
        """
        try:
            session_kwargs = {"region_name": self.config["region"]}
            
            # Add credentials if provided
            if "access_key_id" in self.config:
                session_kwargs["aws_access_key_id"] = self.config["access_key_id"]
            if "secret_access_key" in self.config:
                session_kwargs["aws_secret_access_key"] = self.config["secret_access_key"]
            if "session_token" in self.config:
                session_kwargs["aws_session_token"] = self.config["session_token"]

            # Create session with profile if specified
            if "profile_name" in self.config:
                session = boto3.Session(
                    profile_name=self.config["profile_name"],
                    region_name=self.config["region"]
                )
                return session.client(service_name)
            else:
                return boto3.client(service_name, **session_kwargs)

        except NoCredentialsError:
            raise NoCredentialsError(
                "AWS credentials not found. Please configure credentials using "
                "AWS CLI, environment variables, or IAM roles."
            )
        except Exception as e:
            raise ClientError(f"Failed to create AWS {service_name} client: {e}")

    async def test_credentials(self) -> Dict[str, Any]:
        """Test AWS credentials by making a simple API call.
        
        Returns:
            Dictionary containing:
            - valid: Whether credentials are valid
            - account_id: AWS account ID (if valid)
            - region: Configured region
            - error: Error message (if invalid)
            
        Raises:
            ConnectionError: If unable to test credentials
        """
        try:
            sts_client = self.create_client("sts")
            response = sts_client.get_caller_identity()
            
            return {
                "valid": True,
                "account_id": response.get("Account"),
                "user_id": response.get("UserId"),
                "arn": response.get("Arn"),
                "region": self.config["region"],
                "error": None
            }
            
        except NoCredentialsError as e:
            return {
                "valid": False,
                "account_id": None,
                "user_id": None,
                "arn": None,
                "region": self.config["region"],
                "error": str(e)
            }
        except ClientError as e:
            return {
                "valid": False,
                "account_id": None,
                "user_id": None,
                "arn": None,
                "region": self.config["region"],
                "error": f"AWS API error: {e}"
            }
        except Exception as e:
            return {
                "valid": False,
                "account_id": None,
                "user_id": None,
                "arn": None,
                "region": self.config["region"],
                "error": f"Unexpected error: {e}"
            }

    def get_available_regions(self, service_name: str = "ec2") -> list[str]:
        """Get list of available AWS regions for a service.
        
        Args:
            service_name: AWS service name to check regions for
            
        Returns:
            List of available region names
            
        Raises:
            ClientError: If unable to get regions
        """
        try:
            session = boto3.Session()
            return session.get_available_regions(service_name)
        except Exception as e:
            raise ClientError(f"Failed to get available regions: {e}")


def create_aws_config(
    region: str,
    access_key_id: Optional[str] = None,
    secret_access_key: Optional[str] = None,
    session_token: Optional[str] = None,
    profile_name: Optional[str] = None
) -> Dict[str, Any]:
    """Create AWS configuration dictionary.
    
    Args:
        region: AWS region
        access_key_id: AWS access key ID (optional)
        secret_access_key: AWS secret access key (optional)
        session_token: AWS session token (optional)
        profile_name: AWS profile name (optional)
        
    Returns:
        AWS configuration dictionary
    """
    config = {"region": region}
    
    if access_key_id:
        config["access_key_id"] = access_key_id
    if secret_access_key:
        config["secret_access_key"] = secret_access_key
    if session_token:
        config["session_token"] = session_token
    if profile_name:
        config["profile_name"] = profile_name
    
    return config
