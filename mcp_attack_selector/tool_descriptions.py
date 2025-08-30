"""
Tool Descriptions - Function-based tool descriptions for MCP Attack Selector.

This module provides structured, function-based descriptions of available tools
that can be used by LLM agents for intelligent tool selection.

Author: Anonymous
"""

import json
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum


class ToolCategory(Enum):
    """Categories of available tools."""
    SELECTION = "selection"
    EXECUTION = "execution"
    ANALYSIS = "analysis"
    UTILITY = "utility"


@dataclass
class ToolParameter:
    """Description of a tool parameter."""
    name: str
    type: str
    description: str
    required: bool = True
    default: Optional[Any] = None
    constraints: Optional[Dict[str, Any]] = None


@dataclass
class ToolFunction:
    """Complete description of a tool function."""
    name: str
    category: ToolCategory
    description: str
    parameters: List[ToolParameter]
    returns: str
    examples: List[Dict[str, Any]]
    use_cases: List[str]
    resource_requirements: Dict[str, Any]
    execution_time: str
    success_factors: List[str]
    limitations: List[str]


class ToolDescriptor:
    """Provides structured descriptions of available MCP tools."""
    
    def __init__(self):
        self.tools = self._initialize_tool_descriptions()
    
    def _initialize_tool_descriptions(self) -> Dict[str, ToolFunction]:
        """Initialize all tool descriptions."""
        return {
            "select_attack_method": self._describe_select_attack_method(),
            "execute_attack": self._describe_execute_attack(),
            "get_model_characteristics": self._describe_get_model_characteristics(),
            "list_available_models": self._describe_list_available_models(),
            "list_available_attacks": self._describe_list_available_attacks(),
            "get_attack_recommendations": self._describe_get_attack_recommendations(),
            "get_selection_statistics": self._describe_get_selection_statistics(),
        }
    
    def _describe_select_attack_method(self) -> ToolFunction:
        """Describe the select_attack_method tool."""
        return ToolFunction(
            name="select_attack_method",
            category=ToolCategory.SELECTION,
            description="Intelligently selects the optimal red teaming attack method based on target model characteristics, behavior type, and available resources. Uses multi-factor scoring algorithm.",
            parameters=[
                ToolParameter(
                    name="target_model",
                    type="string",
                    description="Name of the target model to attack (e.g., 'openchat_3_5_1210', 'llama2_7b')",
                    required=True,
                    constraints={"enum": ["openchat_3_5_1210", "llama2_7b", "llama2_13b", "llama2_70b", "gpt-4", "vicuna_7b_v1_5", "mistral_7b_v2"]}
                ),
                ToolParameter(
                    name="behavior_type",
                    type="string",
                    description="Type of harmful behavior to test (e.g., 'illegal', 'misinformation', 'harassment', 'violence')",
                    required=True,
                    constraints={"examples": ["illegal", "misinformation", "harassment", "violence", "simple"]}
                ),
                ToolParameter(
                    name="available_resources",
                    type="object",
                    description="Available computational resources",
                    required=True,
                    constraints={
                        "properties": {
                            "gpu_memory_gb": {"type": "number", "minimum": 1, "maximum": 200},
                            "max_time_minutes": {"type": "number", "minimum": 5, "maximum": 1440}
                        }
                    }
                ),
                ToolParameter(
                    name="preferences",
                    type="object",
                    description="User preferences for attack selection weighting",
                    required=False,
                    constraints={
                        "properties": {
                            "safety_weight": {"type": "number", "minimum": 0, "maximum": 1},
                            "resource_weight": {"type": "number", "minimum": 0, "maximum": 1},
                            "behavior_weight": {"type": "number", "minimum": 0, "maximum": 1}
                        }
                    }
                )
            ],
            returns="Dictionary containing selected method, confidence score, detailed reasoning, alternative methods, and attack specifications",
            examples=[
                {
                    "input": {
                        "target_model": "openchat_3_5_1210",
                        "behavior_type": "illegal",
                        "available_resources": {"gpu_memory_gb": 20, "max_time_minutes": 60}
                    },
                    "output": {
                        "selected_method": "GPTFuzz",
                        "confidence_score": 0.85,
                        "reasoning": "Medium safety model with adequate resources requires sophisticated attack"
                    }
                }
            ],
            use_cases=[
                "Automated attack method selection for red teaming workflows",
                "Resource-constrained attack optimization",
                "Behavior-specific attack methodology selection",
                "Model vulnerability assessment planning"
            ],
            resource_requirements={
                "gpu_memory_gb": 0.1,
                "execution_time_seconds": 1,
                "cpu_cores": 1
            },
            execution_time="~1 second",
            success_factors=[
                "Accurate model profiling data",
                "Clear behavior type specification",
                "Realistic resource constraints",
                "Up-to-date attack method characteristics"
            ],
            limitations=[
                "Selection quality depends on model profile accuracy",
                "May not account for recent model updates",
                "Requires manual validation for novel attack scenarios"
            ]
        )
    
    def _describe_execute_attack(self) -> ToolFunction:
        """Describe the execute_attack tool."""
        return ToolFunction(
            name="execute_attack",
            category=ToolCategory.EXECUTION,
            description="Executes a selected red teaming attack method with specified parameters. Handles resource management, timeout control, and result processing.",
            parameters=[
                ToolParameter(
                    name="attack_method",
                    type="string",
                    description="Attack method to execute",
                    required=True,
                    constraints={"enum": ["DirectRequest", "FewShot", "GPTFuzz"]}
                ),
                ToolParameter(
                    name="target_model",
                    type="string",
                    description="Target model name",
                    required=True
                ),
                ToolParameter(
                    name="behavior_file",
                    type="string",
                    description="Path to behavior file or behavior specification",
                    required=True
                ),
                ToolParameter(
                    name="results_dir",
                    type="string",
                    description="Directory to save execution results",
                    required=False,
                    default="./results"
                ),
                ToolParameter(
                    name="custom_params",
                    type="object",
                    description="Custom parameters for the attack method",
                    required=False
                )
            ],
            returns="Execution results including success metrics, generated test cases, logs, and performance data",
            examples=[
                {
                    "input": {
                        "attack_method": "GPTFuzz",
                        "target_model": "openchat_3_5_1210",
                        "behavior_file": "./five_behaviors.txt"
                    },
                    "output": {
                        "status": "success",
                        "success_rate": 0.8,
                        "jailbreaks_found": 4,
                        "execution_time_minutes": 45
                    }
                }
            ],
            use_cases=[
                "Automated red teaming attack execution",
                "Batch testing of multiple behaviors",
                "Attack method performance evaluation",
                "Vulnerability assessment automation"
            ],
            resource_requirements={
                "gpu_memory_gb": "1-50 (depends on attack method)",
                "execution_time_minutes": "5-180 (depends on attack method)",
                "cpu_cores": "1-4"
            },
            execution_time="5 minutes to 3 hours (method-dependent)",
            success_factors=[
                "Sufficient GPU memory for chosen attack method",
                "Valid behavior file format",
                "Proper model configuration",
                "Adequate time allocation"
            ],
            limitations=[
                "Resource-intensive for advanced attacks",
                "May require significant execution time",
                "Success depends on model vulnerabilities",
                "Results may vary between runs"
            ]
        )
    
    def _describe_get_model_characteristics(self) -> ToolFunction:
        """Describe the get_model_characteristics tool."""
        return ToolFunction(
            name="get_model_characteristics",
            category=ToolCategory.ANALYSIS,
            description="Retrieves comprehensive characteristics of a target model including safety level, capabilities, vulnerabilities, and recommended attack methods.",
            parameters=[
                ToolParameter(
                    name="model_name",
                    type="string",
                    description="Name of the model to analyze",
                    required=True
                )
            ],
            returns="Detailed model profile with safety metrics, capabilities, vulnerabilities, and attack recommendations",
            examples=[
                {
                    "input": {"model_name": "openchat_3_5_1210"},
                    "output": {
                        "safety_level": "medium",
                        "instruction_following": "excellent",
                        "known_vulnerabilities": ["Few-shot prompt injection", "Context manipulation"],
                        "recommended_attacks": ["FewShot", "GPTFuzz"]
                    }
                }
            ],
            use_cases=[
                "Model vulnerability assessment",
                "Attack strategy planning",
                "Safety evaluation preparation",
                "Research and analysis"
            ],
            resource_requirements={
                "gpu_memory_gb": 0,
                "execution_time_seconds": 0.1,
                "cpu_cores": 1
            },
            execution_time="Instant",
            success_factors=[
                "Model exists in profile database",
                "Up-to-date vulnerability information",
                "Accurate safety assessments"
            ],
            limitations=[
                "Limited to pre-profiled models",
                "May not reflect recent model updates",
                "Profile accuracy depends on available data"
            ]
        )
    
    def _describe_list_available_models(self) -> ToolFunction:
        """Describe the list_available_models tool."""
        return ToolFunction(
            name="list_available_models",
            category=ToolCategory.UTILITY,
            description="Lists all models available for red teaming evaluation with their basic characteristics.",
            parameters=[],
            returns="List of available model names with count and basic information",
            examples=[
                {
                    "input": {},
                    "output": {
                        "available_models": ["openchat_3_5_1210", "llama2_7b", "gpt-4"],
                        "count": 3
                    }
                }
            ],
            use_cases=[
                "Discovery of available models",
                "Planning evaluation campaigns",
                "Model selection for testing",
                "Inventory management"
            ],
            resource_requirements={
                "gpu_memory_gb": 0,
                "execution_time_seconds": 0.1,
                "cpu_cores": 1
            },
            execution_time="Instant",
            success_factors=[
                "Up-to-date model registry",
                "Proper model catalog maintenance"
            ],
            limitations=[
                "Only shows pre-registered models",
                "May not include latest model releases"
            ]
        )
    
    def _describe_list_available_attacks(self) -> ToolFunction:
        """Describe the list_available_attacks tool."""
        return ToolFunction(
            name="list_available_attacks",
            category=ToolCategory.UTILITY,
            description="Lists all available attack methods with their characteristics and resource requirements.",
            parameters=[],
            returns="List of attack methods with descriptions, complexity levels, and resource requirements",
            examples=[
                {
                    "input": {},
                    "output": {
                        "available_attacks": ["DirectRequest", "FewShot", "GPTFuzz"],
                        "attack_details": {
                            "GPTFuzz": {
                                "complexity_level": "high",
                                "resource_requirements": {"gpu_memory_gb": 20}
                            }
                        }
                    }
                }
            ],
            use_cases=[
                "Attack method discovery",
                "Resource planning",
                "Method comparison",
                "Capability assessment"
            ],
            resource_requirements={
                "gpu_memory_gb": 0,
                "execution_time_seconds": 0.1,
                "cpu_cores": 1
            },
            execution_time="Instant",
            success_factors=[
                "Complete attack registry",
                "Accurate resource specifications"
            ],
            limitations=[
                "Static attack method list",
                "May not include experimental methods"
            ]
        )
    
    def _describe_get_attack_recommendations(self) -> ToolFunction:
        """Describe the get_attack_recommendations tool."""
        return ToolFunction(
            name="get_attack_recommendations",
            category=ToolCategory.SELECTION,
            description="Generates attack method recommendations for multiple behavior types simultaneously, with summary statistics and optimization suggestions.",
            parameters=[
                ToolParameter(
                    name="target_model",
                    type="string",
                    description="Target model name",
                    required=True
                ),
                ToolParameter(
                    name="behavior_types",
                    type="array",
                    description="List of behavior types to analyze",
                    required=True,
                    constraints={"items": {"type": "string"}}
                ),
                ToolParameter(
                    name="available_resources",
                    type="object",
                    description="Available computational resources",
                    required=True
                )
            ],
            returns="Recommendations for each behavior type with summary statistics and optimization suggestions",
            examples=[
                {
                    "input": {
                        "target_model": "openchat_3_5_1210",
                        "behavior_types": ["illegal", "misinformation"],
                        "available_resources": {"gpu_memory_gb": 20, "max_time_minutes": 120}
                    },
                    "output": {
                        "recommendations": {
                            "illegal": {"selected_method": "GPTFuzz", "confidence": 0.9},
                            "misinformation": {"selected_method": "FewShot", "confidence": 0.8}
                        },
                        "summary": {"most_recommended": "GPTFuzz", "total_behaviors": 2}
                    }
                }
            ],
            use_cases=[
                "Batch behavior analysis",
                "Campaign planning",
                "Resource optimization",
                "Comprehensive evaluation setup"
            ],
            resource_requirements={
                "gpu_memory_gb": 0.1,
                "execution_time_seconds": 2,
                "cpu_cores": 1
            },
            execution_time="~2 seconds",
            success_factors=[
                "Diverse behavior type coverage",
                "Realistic resource constraints",
                "Clear evaluation objectives"
            ],
            limitations=[
                "Limited to predefined behavior types",
                "Recommendations may overlap",
                "Resource allocation complexity"
            ]
        )
    
    def _describe_get_selection_statistics(self) -> ToolFunction:
        """Describe the get_selection_statistics tool."""
        return ToolFunction(
            name="get_selection_statistics",
            category=ToolCategory.ANALYSIS,
            description="Provides statistical analysis of previous attack method selections including success rates, method distribution, and performance trends.",
            parameters=[],
            returns="Statistical summary of attack selections with performance metrics and trends",
            examples=[
                {
                    "input": {},
                    "output": {
                        "total_selections": 150,
                        "method_distribution": {"GPTFuzz": 0.6, "FewShot": 0.3, "DirectRequest": 0.1},
                        "average_confidence": 0.82
                    }
                }
            ],
            use_cases=[
                "Performance analysis",
                "Method effectiveness evaluation",
                "Selection algorithm tuning",
                "Research and reporting"
            ],
            resource_requirements={
                "gpu_memory_gb": 0,
                "execution_time_seconds": 0.5,
                "cpu_cores": 1
            },
            execution_time="~0.5 seconds",
            success_factors=[
                "Sufficient selection history",
                "Accurate performance tracking",
                "Diverse evaluation scenarios"
            ],
            limitations=[
                "Requires historical data",
                "May not reflect recent changes",
                "Limited to tracked metrics"
            ]
        )
    
    def get_tool_description(self, tool_name: str) -> Optional[ToolFunction]:
        """Get description for a specific tool."""
        return self.tools.get(tool_name)
    
    def get_all_tools(self) -> List[ToolFunction]:
        """Get all tool descriptions."""
        return list(self.tools.values())
    
    def get_tools_by_category(self, category: ToolCategory) -> List[ToolFunction]:
        """Get tools filtered by category."""
        return [tool for tool in self.tools.values() if tool.category == category]
    
    def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """Get tool descriptions formatted for LLM consumption."""
        llm_tools = []
        
        for tool in self.tools.values():
            llm_tool = {
                "name": tool.name,
                "description": tool.description,
                "category": tool.category.value,
                "parameters": [asdict(param) for param in tool.parameters],
                "returns": tool.returns,
                "use_cases": tool.use_cases,
                "resource_requirements": tool.resource_requirements,
                "execution_time": tool.execution_time,
                "success_factors": tool.success_factors,
                "limitations": tool.limitations
            }
            llm_tools.append(llm_tool)
        
        return llm_tools
    
    def generate_tool_summary(self) -> str:
        """Generate a human-readable summary of all tools."""
        summary = "## MCP Attack Selector Tools\n\n"
        
        for category in ToolCategory:
            category_tools = self.get_tools_by_category(category)
            if category_tools:
                summary += f"### {category.value.title()} Tools\n\n"
                
                for tool in category_tools:
                    summary += f"**{tool.name}**\n"
                    summary += f"- {tool.description}\n"
                    summary += f"- Execution time: {tool.execution_time}\n"
                    summary += f"- Use cases: {', '.join(tool.use_cases[:2])}\n\n"
        
        return summary


# Example usage and testing
if __name__ == "__main__":
    descriptor = ToolDescriptor()
    
    # Test individual tool description
    select_tool = descriptor.get_tool_description("select_attack_method")
    print(f"Tool: {select_tool.name}")
    print(f"Description: {select_tool.description}")
    print(f"Parameters: {len(select_tool.parameters)}")
    
    # Test LLM-formatted tools
    llm_tools = descriptor.get_tools_for_llm()
    print(f"\nLLM tools: {len(llm_tools)}")
    
    # Test summary generation
    summary = descriptor.generate_tool_summary()
    print(f"\nSummary length: {len(summary)} characters")