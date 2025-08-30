"""
MCP Attack Selector - Intelligent Red Teaming Attack Selection System

This package provides an MCP (Model Context Protocol) server that intelligently
selects the most appropriate red teaming attack method based on target model
characteristics and available resources.

Author: Anonymous
Version: 1.0
"""

__version__ = "1.0.0"
__author__ = "Anonymous"

from .attack_registry import AttackMethod, ATTACK_REGISTRY, list_attack_methods
from .model_profiler import ModelProfile, MODEL_PROFILES, get_model_profile, list_models
from .attack_selector import AttackSelector
from .mcp_server import create_mcp_server
from .execution_engine import ExecutionEngine
from .agent_system import AgentSystem, AgentContext, ToolCall
from .tool_descriptions import ToolDescriptor

__all__ = [
    "AttackMethod",
    "ATTACK_REGISTRY", 
    "list_attack_methods",
    "ModelProfile",
    "MODEL_PROFILES",
    "get_model_profile",
    "list_models",
    "AttackSelector",
    "create_mcp_server",
    "ExecutionEngine",
    "AgentSystem",
    "AgentContext",
    "ToolCall",
    "ToolDescriptor"
]