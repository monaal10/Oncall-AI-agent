"""OnCall AI Agent - AI-powered incident resolution for DevOps teams."""

from .core import SetupManager, RuntimeInterface
from .utils.config_validator import ConfigValidator

__version__ = "0.1.0"

__all__ = [
    "SetupManager",
    "RuntimeInterface",
    "ConfigValidator"
]


async def setup_oncall_agent(config: dict) -> RuntimeInterface:
    """Quick setup function for OnCall AI Agent.
    
    Args:
        config: User configuration dictionary
        
    Returns:
        Configured RuntimeInterface with unified functions
        
    Example:
        >>> config = {
        ...     "aws": {"region": "us-west-2"},
        ...     "github": {"repositories": ["myorg/repo"]},
        ...     "openai": {"model": "gpt-4"}
        ... }
        >>> runtime = await setup_oncall_agent(config)
        >>> logs = await runtime.get_logs("ERROR", (start_time, end_time))
    """
    setup_manager = SetupManager()
    return await setup_manager.setup_from_config(config)
