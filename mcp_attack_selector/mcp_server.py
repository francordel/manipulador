"""
MCP Server - Model Context Protocol server for intelligent attack selection.

This module implements the MCP server that exposes tools to AI agents for 
selecting and executing red teaming attacks based on target model characteristics.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import asdict

from .attack_selector import AttackSelector
from .model_profiler import get_model_profile, list_models
from .attack_registry import list_attack_methods, get_attack_method
from .execution_engine import ExecutionEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPAttackServer:
    """MCP server for intelligent red teaming attack selection and execution."""
    
    def __init__(self):
        self.attack_selector = AttackSelector()
        self.execution_engine = ExecutionEngine()
        self.server_info = {
            "name": "MCP Attack Selector",
            "version": "1.0.0",
            "description": "Intelligent red teaming attack selection and execution"
        }
        
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get list of available MCP tools."""
        return [
            {
                "name": "select_attack_method",
                "description": "Intelligently select the best attack method based on target model and behavior",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "target_model": {
                            "type": "string",
                            "description": "Name of the target model (e.g., 'openchat_3_5_1210')"
                        },
                        "behavior_type": {
                            "type": "string", 
                            "description": "Type of behavior to test (e.g., 'illegal', 'misinformation', 'harassment')"
                        },
                        "available_resources": {
                            "type": "object",
                            "description": "Available resources including GPU memory and time constraints",
                            "properties": {
                                "gpu_memory_gb": {"type": "number"},
                                "max_time_minutes": {"type": "number"}
                            }
                        },
                        "preferences": {
                            "type": "object",
                            "description": "User preferences for attack selection (optional)",
                            "properties": {
                                "safety_weight": {"type": "number"},
                                "resource_weight": {"type": "number"},
                                "behavior_weight": {"type": "number"}
                            }
                        }
                    },
                    "required": ["target_model", "behavior_type", "available_resources"]
                }
            },
            {
                "name": "execute_attack",
                "description": "Execute the selected attack method with appropriate parameters",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "attack_method": {
                            "type": "string",
                            "enum": ["DirectRequest", "FewShot", "GPTFuzz"],
                            "description": "Selected attack method"
                        },
                        "target_model": {
                            "type": "string",
                            "description": "Target model name"
                        },
                        "behavior_file": {
                            "type": "string",
                            "description": "Path to behavior file or behavior IDs"
                        },
                        "results_dir": {
                            "type": "string",
                            "description": "Directory to save results (optional)",
                            "default": "./results"
                        },
                        "custom_params": {
                            "type": "object",
                            "description": "Custom parameters for the attack method (optional)"
                        }
                    },
                    "required": ["attack_method", "target_model", "behavior_file"]
                }
            },
            {
                "name": "get_model_characteristics",
                "description": "Get detailed characteristics of a target model",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "model_name": {
                            "type": "string",
                            "description": "Name of the model to analyze"
                        }
                    },
                    "required": ["model_name"]
                }
            },
            {
                "name": "list_available_models",
                "description": "Get list of all available target models",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "list_available_attacks",
                "description": "Get list of all available attack methods",
                "inputSchema": {
                    "type": "object", 
                    "properties": {}
                }
            },
            {
                "name": "get_attack_recommendations",
                "description": "Get attack recommendations for multiple behavior types",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "target_model": {
                            "type": "string",
                            "description": "Target model name"
                        },
                        "behavior_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of behavior types to test"
                        },
                        "available_resources": {
                            "type": "object",
                            "description": "Available resources",
                            "properties": {
                                "gpu_memory_gb": {"type": "number"},
                                "max_time_minutes": {"type": "number"}
                            }
                        }
                    },
                    "required": ["target_model", "behavior_types", "available_resources"]
                }
            },
            {
                "name": "get_selection_statistics",
                "description": "Get statistics about previous attack selections",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
    
    def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP tool calls."""
        try:
            if tool_name == "select_attack_method":
                return self._select_attack_method(**arguments)
            elif tool_name == "execute_attack":
                return self._execute_attack(**arguments)
            elif tool_name == "get_model_characteristics":
                return self._get_model_characteristics(**arguments)
            elif tool_name == "list_available_models":
                return self._list_available_models()
            elif tool_name == "list_available_attacks":
                return self._list_available_attacks()
            elif tool_name == "get_attack_recommendations":
                return self._get_attack_recommendations(**arguments)
            elif tool_name == "get_selection_statistics":
                return self._get_selection_statistics()
            else:
                return {"error": f"Unknown tool: {tool_name}"}
                
        except Exception as e:
            logger.error(f"Error handling tool call {tool_name}: {str(e)}")
            return {"error": f"Tool execution failed: {str(e)}"}
    
    def _select_attack_method(
        self,
        target_model: str,
        behavior_type: str,
        available_resources: Dict[str, Any],
        preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Select the optimal attack method."""
        logger.info(f"Selecting attack method for {target_model}, behavior: {behavior_type}")
        
        result = self.attack_selector.select_attack_method(
            target_model=target_model,
            behavior_type=behavior_type,
            available_resources=available_resources,
            preferences=preferences
        )
        
        return {
            "status": "success",
            "selected_method": result["selected_method"],
            "confidence_score": result["confidence_score"],
            "reasoning": result["reasoning"],
            "alternative_methods": result["alternative_methods"],
            "model_analysis": result["model_analysis"],
            "attack_details": result["attack_details"]
        }
    
    def _execute_attack(
        self,
        attack_method: str,
        target_model: str,
        behavior_file: str,
        results_dir: str = "./results",
        custom_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute the selected attack method."""
        logger.info(f"Executing {attack_method} attack on {target_model}")
        
        result = self.execution_engine.execute_attack(
            attack_method=attack_method,
            target_model=target_model,
            behavior_file=behavior_file,
            results_dir=results_dir,
            custom_params=custom_params or {}
        )
        
        return result
    
    def _get_model_characteristics(self, model_name: str) -> Dict[str, Any]:
        """Get model characteristics."""
        logger.info(f"Getting characteristics for model: {model_name}")
        
        try:
            profile = get_model_profile(model_name)
            return {
                "status": "success",
                "model_profile": asdict(profile)
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to get model profile: {str(e)}"
            }
    
    def _list_available_models(self) -> Dict[str, Any]:
        """List all available models."""
        models = list_models()
        return {
            "status": "success",
            "available_models": models,
            "count": len(models)
        }
    
    def _list_available_attacks(self) -> Dict[str, Any]:
        """List all available attack methods."""
        attacks = list_attack_methods()
        attack_details = {}
        
        for attack_name in attacks:
            method = get_attack_method(attack_name)
            attack_details[attack_name] = {
                "description": method.description,
                "complexity_level": method.complexity_level,
                "resource_requirements": method.resource_requirements
            }
        
        return {
            "status": "success",
            "available_attacks": attacks,
            "attack_details": attack_details,
            "count": len(attacks)
        }
    
    def _get_attack_recommendations(
        self,
        target_model: str,
        behavior_types: List[str],
        available_resources: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get attack recommendations for multiple behavior types."""
        logger.info(f"Getting recommendations for {target_model}, behaviors: {behavior_types}")
        
        result = self.attack_selector.get_attack_recommendations(
            target_model=target_model,
            behavior_types=behavior_types,
            available_resources=available_resources
        )
        
        return {
            "status": "success",
            "recommendations": result["recommendations"],
            "summary": result["summary"]
        }
    
    def _get_selection_statistics(self) -> Dict[str, Any]:
        """Get selection statistics."""
        stats = self.attack_selector.get_selection_statistics()
        return {
            "status": "success",
            "statistics": stats
        }


def create_mcp_server() -> MCPAttackServer:
    """Create and configure MCP server instance."""
    server = MCPAttackServer()
    logger.info("MCP Attack Selector server created successfully")
    return server


# Example usage for testing
if __name__ == "__main__":
    # Create server
    server = create_mcp_server()
    
    # Test tool listing
    tools = server.get_tools()
    print(f"Available tools: {[tool['name'] for tool in tools]}")
    
    # Test model characteristics
    result = server.handle_tool_call("get_model_characteristics", {"model_name": "openchat_3_5_1210"})
    print(f"\nModel characteristics result: {json.dumps(result, indent=2)}")
    
    # Test attack selection
    selection_result = server.handle_tool_call("select_attack_method", {
        "target_model": "openchat_3_5_1210",
        "behavior_type": "illegal",
        "available_resources": {"gpu_memory_gb": 20, "max_time_minutes": 60}
    })
    print(f"\nAttack selection result: {json.dumps(selection_result, indent=2)}")