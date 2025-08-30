"""
Model Profiler - Analyzes target model characteristics and vulnerabilities.

This module contains detailed profiles of various AI models including their
safety levels, capabilities, and known vulnerabilities to inform attack selection.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class ModelProfile:
    """Comprehensive profile of a target model's characteristics."""
    model_name: str
    model_type: str  # "open_source", "closed_source"
    safety_level: str  # "low", "medium", "high"
    instruction_following: str  # "weak", "good", "excellent"
    context_awareness: str  # "low", "medium", "high"
    resource_requirements: Dict[str, Any]
    known_vulnerabilities: List[str]
    training_approach: str  # "base", "instruction_tuned", "rlhf"
    model_family: str
    release_year: int
    parameter_count: str
    strengths: List[str]
    weaknesses: List[str]
    recommended_attacks: List[str]


# Comprehensive model registry
MODEL_PROFILES = {
    "openchat_3_5_1210": ModelProfile(
        model_name="openchat_3_5_1210",
        model_type="open_source",
        safety_level="medium",
        instruction_following="excellent",
        context_awareness="high",
        resource_requirements={
            "gpu_memory_gb": 16,
            "inference_speed": "fast",
            "context_length": 8192
        },
        known_vulnerabilities=[
            "Few-shot prompt injection",
            "Context manipulation",
            "Instruction confusion",
            "Role-playing exploitation"
        ],
        training_approach="instruction_tuned",
        model_family="OpenChat",
        release_year=2023,
        parameter_count="7B",
        strengths=[
            "Strong instruction following",
            "Good conversational abilities",
            "Fast inference",
            "Moderate safety training"
        ],
        weaknesses=[
            "Vulnerable to sophisticated prompts",
            "Limited safety mechanisms",
            "Context manipulation susceptible",
            "No RLHF training"
        ],
        recommended_attacks=["FewShot", "GPTFuzz", "DirectRequest"]
    ),
    
    "llama2_7b": ModelProfile(
        model_name="llama2_7b",
        model_type="open_source", 
        safety_level="medium",
        instruction_following="good",
        context_awareness="medium",
        resource_requirements={
            "gpu_memory_gb": 14,
            "inference_speed": "fast",
            "context_length": 4096
        },
        known_vulnerabilities=[
            "System prompt leakage",
            "Role confusion",
            "Context window attacks",
            "Instruction bypassing"
        ],
        training_approach="rlhf",
        model_family="Llama",
        release_year=2023,
        parameter_count="7B",
        strengths=[
            "RLHF safety training",
            "Good general capabilities",
            "Robust architecture",
            "Well-documented behavior"
        ],
        weaknesses=[
            "Moderate safety gaps",
            "Context manipulation vulnerable",
            "Limited instruction tuning",
            "Prompt injection susceptible"
        ],
        recommended_attacks=["GPTFuzz", "FewShot", "DirectRequest"]
    ),
    
    "llama2_13b": ModelProfile(
        model_name="llama2_13b",
        model_type="open_source",
        safety_level="medium",
        instruction_following="good", 
        context_awareness="medium",
        resource_requirements={
            "gpu_memory_gb": 26,
            "inference_speed": "medium",
            "context_length": 4096
        },
        known_vulnerabilities=[
            "System prompt leakage",
            "Role confusion",
            "Context window attacks",
            "Instruction bypassing"
        ],
        training_approach="rlhf",
        model_family="Llama",
        release_year=2023,
        parameter_count="13B",
        strengths=[
            "RLHF safety training",
            "Better reasoning than 7B",
            "More robust responses",
            "Improved safety awareness"
        ],
        weaknesses=[
            "Higher resource requirements",
            "Still vulnerable to sophisticated attacks",
            "Context manipulation susceptible",
            "Prompt injection possible"
        ],
        recommended_attacks=["GPTFuzz", "FewShot", "DirectRequest"]
    ),
    
    "llama2_70b": ModelProfile(
        model_name="llama2_70b",
        model_type="open_source",
        safety_level="high",
        instruction_following="excellent",
        context_awareness="high",
        resource_requirements={
            "gpu_memory_gb": 140,
            "inference_speed": "slow",
            "context_length": 4096
        },
        known_vulnerabilities=[
            "Advanced prompt injection",
            "Multi-turn exploitation",
            "Context manipulation",
            "Sophisticated role-playing"
        ],
        training_approach="rlhf",
        model_family="Llama",
        release_year=2023,
        parameter_count="70B",
        strengths=[
            "Strong RLHF safety training",
            "Excellent reasoning capabilities",
            "High safety awareness",
            "Robust to simple attacks"
        ],
        weaknesses=[
            "Very high resource requirements",
            "Complex deployment",
            "Still vulnerable to advanced techniques",
            "Expensive to run"
        ],
        recommended_attacks=["GPTFuzz", "FewShot"]
    ),
    
    "vicuna_7b_v1_5": ModelProfile(
        model_name="vicuna_7b_v1_5",
        model_type="open_source",
        safety_level="low",
        instruction_following="excellent",
        context_awareness="high",
        resource_requirements={
            "gpu_memory_gb": 14,
            "inference_speed": "fast",
            "context_length": 2048
        },
        known_vulnerabilities=[
            "Minimal safety training",
            "Easy prompt injection",
            "Role-playing exploitation",
            "Direct instruction following"
        ],
        training_approach="instruction_tuned",
        model_family="Vicuna",
        release_year=2023,
        parameter_count="7B",
        strengths=[
            "Excellent instruction following",
            "Good conversational abilities",
            "Fast inference",
            "Open and accessible"
        ],
        weaknesses=[
            "Very limited safety mechanisms",
            "Highly vulnerable to simple attacks",
            "No safety training",
            "Follows harmful instructions easily"
        ],
        recommended_attacks=["DirectRequest", "FewShot", "GPTFuzz"]
    ),
    
    "gpt-4": ModelProfile(
        model_name="gpt-4",
        model_type="closed_source",
        safety_level="high",
        instruction_following="excellent",
        context_awareness="excellent",
        resource_requirements={
            "gpu_memory_gb": 0,  # API-based
            "inference_speed": "medium",
            "context_length": 32768
        },
        known_vulnerabilities=[
            "Advanced prompt injection",
            "Multi-turn jailbreaking",
            "Context manipulation",
            "Sophisticated role-playing",
            "System prompt confusion"
        ],
        training_approach="rlhf",
        model_family="GPT",
        release_year=2023,
        parameter_count="Unknown (Large)",
        strengths=[
            "Extensive safety training",
            "Sophisticated reasoning",
            "Strong refusal mechanisms",
            "Advanced context understanding"
        ],
        weaknesses=[
            "API rate limits",
            "Cost per request",
            "Still vulnerable to advanced techniques",
            "Complex prompt patterns can bypass"
        ],
        recommended_attacks=["GPTFuzz", "FewShot"]
    ),
    
    "mistral_7b_v2": ModelProfile(
        model_name="mistral_7b_v2",
        model_type="open_source",
        safety_level="low",
        instruction_following="good",
        context_awareness="medium",
        resource_requirements={
            "gpu_memory_gb": 14,
            "inference_speed": "fast",
            "context_length": 32768
        },
        known_vulnerabilities=[
            "Limited safety training",
            "Prompt injection",
            "Role confusion",
            "Context manipulation"
        ],
        training_approach="instruction_tuned",
        model_family="Mistral",
        release_year=2023,
        parameter_count="7B",
        strengths=[
            "Long context length",
            "Good performance",
            "Fast inference",
            "Efficient architecture"
        ],
        weaknesses=[
            "Minimal safety mechanisms",
            "Vulnerable to attacks",
            "Limited refusal training",
            "Follows harmful instructions"
        ],
        recommended_attacks=["DirectRequest", "FewShot", "GPTFuzz"]
    )
}


def get_model_profile(model_name: str) -> ModelProfile:
    """Get model profile by name."""
    if model_name not in MODEL_PROFILES:
        # Return default profile for unknown models
        return ModelProfile(
            model_name=model_name,
            model_type="unknown",
            safety_level="medium",
            instruction_following="good",
            context_awareness="medium",
            resource_requirements={"gpu_memory_gb": 16, "inference_speed": "medium"},
            known_vulnerabilities=["Unknown"],
            training_approach="unknown",
            model_family="Unknown",
            release_year=2023,
            parameter_count="Unknown",
            strengths=["Unknown"],
            weaknesses=["Unknown"],
            recommended_attacks=["FewShot", "DirectRequest", "GPTFuzz"]
        )
    return MODEL_PROFILES[model_name]


def list_models() -> List[str]:
    """Get list of available model names."""
    return list(MODEL_PROFILES.keys())


def get_models_by_safety_level(safety_level: str) -> List[str]:
    """Get models filtered by safety level."""
    return [name for name, profile in MODEL_PROFILES.items() 
            if profile.safety_level == safety_level]


def get_models_by_type(model_type: str) -> List[str]:
    """Get models filtered by type (open_source/closed_source)."""
    return [name for name, profile in MODEL_PROFILES.items() 
            if profile.model_type == model_type]


def analyze_model_for_attack_selection(model_name: str) -> Dict[str, Any]:
    """Analyze model characteristics for attack method selection."""
    profile = get_model_profile(model_name)
    
    analysis = {
        "model_name": model_name,
        "safety_score": {"low": 1, "medium": 2, "high": 3}[profile.safety_level],
        "instruction_score": {"weak": 1, "good": 2, "excellent": 3}[profile.instruction_following],
        "context_score": {"low": 1, "medium": 2, "high": 3}[profile.context_awareness],
        "vulnerability_count": len(profile.known_vulnerabilities),
        "resource_requirements": profile.resource_requirements,
        "recommended_attacks": profile.recommended_attacks,
        "attack_difficulty": profile.safety_level,
        "best_attack_types": []
    }
    
    # Determine best attack types based on characteristics
    if profile.safety_level == "low":
        analysis["best_attack_types"] = ["DirectRequest", "FewShot"]
    elif profile.safety_level == "medium":
        if profile.instruction_following == "excellent":
            analysis["best_attack_types"] = ["FewShot", "GPTFuzz"]
        else:
            analysis["best_attack_types"] = ["DirectRequest", "FewShot"]
    else:  # high safety
        analysis["best_attack_types"] = ["GPTFuzz", "FewShot"]
    
    return analysis


if __name__ == "__main__":
    # Test the profiler
    print("Available Models:")
    for name in list_models():
        profile = get_model_profile(name)
        analysis = analyze_model_for_attack_selection(name)
        print(f"\n{name}:")
        print(f"  Safety Level: {profile.safety_level}")
        print(f"  Best Attacks: {analysis['best_attack_types']}")
        print(f"  GPU Memory: {profile.resource_requirements.get('gpu_memory_gb', 'Unknown')}GB")