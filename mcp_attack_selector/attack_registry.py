"""
Attack Registry - Defines characteristics and capabilities of each attack method.

This module contains the detailed specifications for each available attack method,
including their strengths, weaknesses, resource requirements, and optimal use cases.
"""

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class AttackMethod:
    """Comprehensive definition of an attack method's characteristics."""
    name: str
    description: str
    strengths: List[str]
    weaknesses: List[str]
    resource_requirements: Dict[str, Any]
    success_factors: List[str]
    typical_use_cases: List[str]
    complexity_level: str  # "low", "medium", "high"
    execution_script: str
    default_params: Dict[str, Any]


# Complete attack method registry
ATTACK_REGISTRY = {
    "DirectRequest": AttackMethod(
        name="DirectRequest",
        description="Simple direct prompting without sophistication - tests basic safety filters",
        strengths=[
            "Fast execution (seconds to minutes)",
            "Minimal resource usage",
            "Good baseline for testing",
            "Simple implementation",
            "Low complexity barrier"
        ],
        weaknesses=[
            "Low success rate against modern models",
            "Easily blocked by basic safety filters",
            "Limited sophistication",
            "No adaptive capabilities",
            "Poor against RLHF models"
        ],
        resource_requirements={
            "gpu_memory_gb": 1,
            "execution_time_minutes": 5,
            "complexity": "low",
            "cpu_cores": 1
        },
        success_factors=[
            "Weak safety filters",
            "Older model architectures",
            "Simple behavior types",
            "Basic prompt injection vulnerabilities",
            "Untuned base models"
        ],
        typical_use_cases=[
            "Quick baseline testing",
            "Resource-constrained environments",
            "Simple behavior validation",
            "Initial vulnerability assessment",
            "Educational demonstrations"
        ],
        complexity_level="low",
        execution_script="run_directrequest.sh",
        default_params={
            "timeout_minutes": 10,
            "max_retries": 3
        }
    ),
    
    "FewShot": AttackMethod(
        name="FewShot",
        description="Uses few-shot examples to guide model behavior and exploit instruction-following capabilities",
        strengths=[
            "Good with instruction-following models",
            "Moderate success rate",
            "Leverages model's learning from examples",
            "Contextual awareness",
            "Balanced resource usage"
        ],
        weaknesses=[
            "Requires example generation",
            "Moderate resource usage",
            "Limited by example quality",
            "May not work on simple base models",
            "Depends on context understanding"
        ],
        resource_requirements={
            "gpu_memory_gb": 8,
            "execution_time_minutes": 30,
            "complexity": "medium",
            "cpu_cores": 2
        },
        success_factors=[
            "Instruction-tuned models",
            "Context-aware behaviors",
            "Models with strong pattern recognition",
            "Few-shot learning capabilities",
            "Good instruction following"
        ],
        typical_use_cases=[
            "Instruction-following model testing",
            "Context-dependent behaviors",
            "Moderate sophistication attacks",
            "Educational content generation",
            "Balanced testing approach"
        ],
        complexity_level="medium",
        execution_script="run_fewshot.sh",
        default_params={
            "n_shots": 5,
            "sample_size": 1000,
            "num_steps": 5,
            "timeout_minutes": 60
        }
    ),
    
    "GPTFuzz": AttackMethod(
        name="GPTFuzz",
        description="Advanced fuzzing with iterative mutations, feedback loops, and sophisticated prompt evolution",
        strengths=[
            "High success rate against modern models",
            "Adaptive mutations based on feedback",
            "Sophisticated attack strategies",
            "Handles complex safety mechanisms",
            "Iterative improvement"
        ],
        weaknesses=[
            "High resource usage (10-50GB GPU memory)",
            "Long execution time (hours)",
            "Complex implementation",
            "Requires significant computational resources",
            "May be overkill for simple targets"
        ],
        resource_requirements={
            "gpu_memory_gb": 20,
            "execution_time_minutes": 120,
            "complexity": "high",
            "cpu_cores": 4
        },
        success_factors=[
            "Stubborn safety filters",
            "Modern RLHF-trained models",
            "Complex behavior types",
            "Well-defended models",
            "Sophisticated safety mechanisms"
        ],
        typical_use_cases=[
            "Advanced red teaming",
            "Sophisticated model testing",
            "Research environments",
            "High-security model evaluation",
            "Comprehensive vulnerability assessment"
        ],
        complexity_level="high",
        execution_script="run_gptfuzz.sh",
        default_params={
            "max_query": 1000,
            "max_jailbreak": 1,
            "max_iteration": 100,
            "energy": 1,
            "mutation_model": "openchat-3.5",
            "dtype": "float16",
            "timeout_minutes": 180
        }
    )
}


def get_attack_method(name: str) -> AttackMethod:
    """Get attack method by name."""
    if name not in ATTACK_REGISTRY:
        raise ValueError(f"Unknown attack method: {name}. Available: {list(ATTACK_REGISTRY.keys())}")
    return ATTACK_REGISTRY[name]


def list_attack_methods() -> List[str]:
    """Get list of available attack method names."""
    return list(ATTACK_REGISTRY.keys())


def get_attack_methods_by_complexity(complexity: str) -> List[AttackMethod]:
    """Get attack methods filtered by complexity level."""
    return [method for method in ATTACK_REGISTRY.values() if method.complexity_level == complexity]


def get_suitable_attacks_for_resources(max_gpu_gb: float, max_time_minutes: int) -> List[str]:
    """Get attack methods that fit within resource constraints."""
    suitable = []
    for name, method in ATTACK_REGISTRY.items():
        if (method.resource_requirements["gpu_memory_gb"] <= max_gpu_gb and 
            method.resource_requirements["execution_time_minutes"] <= max_time_minutes):
            suitable.append(name)
    return suitable


if __name__ == "__main__":
    # Test the registry
    print("Available Attack Methods:")
    for name, method in ATTACK_REGISTRY.items():
        print(f"\n{name}:")
        print(f"  Description: {method.description}")
        print(f"  Complexity: {method.complexity_level}")
        print(f"  GPU Memory: {method.resource_requirements['gpu_memory_gb']}GB")
        print(f"  Time: {method.resource_requirements['execution_time_minutes']} minutes")