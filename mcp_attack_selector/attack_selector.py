"""
Attack Selector - Intelligent decision-making logic for attack method selection.

This module implements the core algorithm that selects the most appropriate
attack method based on target model characteristics, available resources,
and behavior complexity.
"""

import json
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import asdict

from .attack_registry import ATTACK_REGISTRY, get_attack_method
from .model_profiler import get_model_profile, analyze_model_for_attack_selection


class AttackSelector:
    """Intelligent attack method selection engine."""
    
    def __init__(self):
        self.selection_history = []
        self.performance_data = {}
        
    def select_attack_method(
        self,
        target_model: str,
        behavior_type: str,
        available_resources: Dict[str, Any],
        preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Select the optimal attack method based on multiple factors.
        
        Args:
            target_model: Name of the target model
            behavior_type: Type of behavior to test
            available_resources: Available GPU memory, time constraints, etc.
            preferences: User preferences for attack selection
            
        Returns:
            Dict containing selected method, reasoning, and confidence score
        """
        # Get model analysis
        model_analysis = analyze_model_for_attack_selection(target_model)
        model_profile = get_model_profile(target_model)
        
        # Calculate scores for each attack method
        scores = {}
        reasoning = {}
        
        for attack_name in ATTACK_REGISTRY.keys():
            score, rationale = self._calculate_attack_score(
                attack_name,
                model_analysis,
                behavior_type,
                available_resources,
                preferences or {}
            )
            scores[attack_name] = score
            reasoning[attack_name] = rationale
        
        # Select best attack
        best_attack = max(scores, key=scores.get)
        confidence = scores[best_attack]
        
        # Generate comprehensive reasoning
        selection_reasoning = self._generate_selection_reasoning(
            best_attack,
            model_analysis,
            behavior_type,
            available_resources,
            scores,
            reasoning
        )
        
        # Record selection for learning
        selection_record = {
            "target_model": target_model,
            "behavior_type": behavior_type,
            "selected_attack": best_attack,
            "confidence": confidence,
            "model_safety": model_profile.safety_level,
            "resources_used": available_resources
        }
        self.selection_history.append(selection_record)
        
        return {
            "selected_method": best_attack,
            "confidence_score": confidence,
            "reasoning": selection_reasoning,
            "alternative_methods": {k: v for k, v in scores.items() if k != best_attack},
            "model_analysis": model_analysis,
            "attack_details": asdict(get_attack_method(best_attack))
        }
    
    def _calculate_attack_score(
        self,
        attack_name: str,
        model_analysis: Dict[str, Any],
        behavior_type: str,
        available_resources: Dict[str, Any],
        preferences: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        """Calculate score for a specific attack method."""
        attack_method = get_attack_method(attack_name)
        rationale = []
        
        # Base scores for different factors
        
        safety_score = self._calculate_safety_score(attack_method, model_analysis, rationale)
        resource_score = self._calculate_resource_score(attack_method, available_resources, rationale)
        behavior_score = self._calculate_behavior_score(attack_method, behavior_type, rationale)
        model_type_score = self._calculate_model_type_score(attack_method, model_analysis, rationale)
        historical_score = self._calculate_historical_score(attack_name, model_analysis, rationale)
        
        # Weights for different factors (can be adjusted based on preferences)
        weights = {
            "safety": preferences.get("safety_weight", 0.25),
            "resources": preferences.get("resource_weight", 0.25),
            "behavior": preferences.get("behavior_weight", 0.20),
            "model_type": preferences.get("model_type_weight", 0.20),
            "historical": preferences.get("historical_weight", 0.10)
        }
        
        # Calculate weighted score
        total_score = (
            safety_score * weights["safety"] +
            resource_score * weights["resources"] +
            behavior_score * weights["behavior"] +
            model_type_score * weights["model_type"] +
            historical_score * weights["historical"]
        )
        
        return total_score, rationale
    
    def _calculate_safety_score(self, attack_method, model_analysis: Dict, rationale: List[str]) -> float:
        """Calculate score based on model safety level vs attack sophistication."""
        safety_level = model_analysis["safety_score"]  # 1=low, 2=medium, 3=high
        complexity_map = {"low": 1, "medium": 2, "high": 3}
        attack_complexity = complexity_map[attack_method.complexity_level]
        
        # Higher safety models need more sophisticated attacks
        if safety_level == 3:  # High safety
            if attack_complexity == 3:  # GPTFuzz
                score = 1.0
                rationale.append("High safety model requires sophisticated attack (GPTFuzz)")
            elif attack_complexity == 2:  # FewShot
                score = 0.7
                rationale.append("High safety model partially matches medium complexity attack")
            else:  # DirectRequest
                score = 0.3
                rationale.append("High safety model poorly matches simple attack")
        elif safety_level == 2:  # Medium safety
            if attack_complexity == 2:  # FewShot
                score = 1.0
                rationale.append("Medium safety model well-suited for moderate attack complexity")
            elif attack_complexity == 3:  # GPTFuzz
                score = 0.8
                rationale.append("Medium safety model compatible with sophisticated attack")
            else:  # DirectRequest
                score = 0.6
                rationale.append("Medium safety model may work with simple attack")
        else:  # Low safety
            if attack_complexity == 1:  # DirectRequest
                score = 1.0
                rationale.append("Low safety model ideal for simple direct attacks")
            elif attack_complexity == 2:  # FewShot
                score = 0.8
                rationale.append("Low safety model works well with moderate attacks")
            else:  # GPTFuzz
                score = 0.5
                rationale.append("Low safety model may be overkill for sophisticated attack")
        
        return score
    
    def _calculate_resource_score(self, attack_method, available_resources: Dict, rationale: List[str]) -> float:
        """Calculate score based on resource availability vs requirements."""
        required_gpu = attack_method.resource_requirements["gpu_memory_gb"]
        required_time = attack_method.resource_requirements["execution_time_minutes"]
        
        available_gpu = available_resources.get("gpu_memory_gb", 16)
        available_time = available_resources.get("max_time_minutes", 120)
        
        # GPU memory score
        if required_gpu <= available_gpu * 0.5:
            gpu_score = 1.0
            rationale.append(f"Plenty of GPU memory available ({available_gpu}GB vs {required_gpu}GB needed)")
        elif required_gpu <= available_gpu * 0.8:
            gpu_score = 0.8
            rationale.append(f"Adequate GPU memory ({available_gpu}GB vs {required_gpu}GB needed)")
        elif required_gpu <= available_gpu:
            gpu_score = 0.6
            rationale.append(f"Tight GPU memory fit ({available_gpu}GB vs {required_gpu}GB needed)")
        else:
            gpu_score = 0.1
            rationale.append(f"Insufficient GPU memory ({available_gpu}GB vs {required_gpu}GB needed)")
        
        # Time score
        if required_time <= available_time * 0.5:
            time_score = 1.0
            rationale.append(f"Plenty of time available ({available_time}min vs {required_time}min needed)")
        elif required_time <= available_time * 0.8:
            time_score = 0.8
            rationale.append(f"Adequate time ({available_time}min vs {required_time}min needed)")
        elif required_time <= available_time:
            time_score = 0.6
            rationale.append(f"Tight time constraint ({available_time}min vs {required_time}min needed)")
        else:
            time_score = 0.1
            rationale.append(f"Insufficient time ({available_time}min vs {required_time}min needed)")
        
        return (gpu_score + time_score) / 2
    
    def _calculate_behavior_score(self, attack_method, behavior_type: str, rationale: List[str]) -> float:
        """Calculate score based on behavior type complexity."""
        behavior_complexity_map = {
            "simple": 1,
            "illegal": 2,
            "misinformation": 2,
            "harassment": 2,
            "violence": 3,
            "complex": 3,
            "sophisticated": 3
        }
        
        behavior_complexity = behavior_complexity_map.get(behavior_type.lower(), 2)
        complexity_map = {"low": 1, "medium": 2, "high": 3}
        attack_complexity = complexity_map[attack_method.complexity_level]
        
        # Match complexity levels
        if behavior_complexity == attack_complexity:
            score = 1.0
            rationale.append(f"Attack complexity perfectly matches behavior type ({behavior_type})")
        elif abs(behavior_complexity - attack_complexity) == 1:
            score = 0.7
            rationale.append(f"Attack complexity reasonably matches behavior type ({behavior_type})")
        else:
            score = 0.4
            rationale.append(f"Attack complexity mismatch with behavior type ({behavior_type})")
        
        return score
    
    def _calculate_model_type_score(self, attack_method, model_analysis: Dict, rationale: List[str]) -> float:
        """Calculate score based on model type and characteristics."""
        instruction_score = model_analysis["instruction_score"]  # 1=weak, 2=good, 3=excellent
        context_score = model_analysis["context_score"]  # 1=low, 2=medium, 3=high
        
        if attack_method.name == "FewShot":
            # FewShot works best with instruction-following models
            if instruction_score >= 2:
                score = 1.0
                rationale.append("FewShot ideal for instruction-following model")
            else:
                score = 0.5
                rationale.append("FewShot suboptimal for weak instruction-following")
        elif attack_method.name == "GPTFuzz":
            # GPTFuzz works well with any model type but especially sophisticated ones
            if instruction_score >= 2 and context_score >= 2:
                score = 1.0
                rationale.append("GPTFuzz excellent for sophisticated models")
            else:
                score = 0.8
                rationale.append("GPTFuzz still effective for simpler models")
        else:  # DirectRequest
            # DirectRequest works universally but better with simpler models
            if instruction_score <= 2:
                score = 1.0
                rationale.append("DirectRequest ideal for simple models")
            else:
                score = 0.7
                rationale.append("DirectRequest works but may be insufficient for sophisticated models")
        
        return score
    
    def _calculate_historical_score(self, attack_name: str, model_analysis: Dict, rationale: List[str]) -> float:
        """Calculate score based on historical performance."""
        # This would use actual performance data in a real implementation
        # For now, use some default scoring based on attack reputation
        
        historical_success_rates = {
            "DirectRequest": 0.3,
            "FewShot": 0.6,
            "GPTFuzz": 0.8
        }
        
        base_score = historical_success_rates.get(attack_name, 0.5)
        
        # Adjust based on model safety (harder models have lower success rates)
        safety_penalty = (model_analysis["safety_score"] - 1) * 0.1
        adjusted_score = max(0.1, base_score - safety_penalty)
        
        rationale.append(f"Historical success rate for {attack_name}: {adjusted_score:.1%}")
        
        return adjusted_score
    
    def _generate_selection_reasoning(
        self,
        selected_attack: str,
        model_analysis: Dict,
        behavior_type: str,
        available_resources: Dict,
        all_scores: Dict[str, float],
        all_reasoning: Dict[str, List[str]]
    ) -> str:
        """Generate comprehensive reasoning for the attack selection."""
        model_name = model_analysis["model_name"]
        safety_level = ["low", "medium", "high"][model_analysis["safety_score"] - 1]
        
        reasoning_parts = [
            f"Selected {selected_attack} for {model_name} (Safety: {safety_level}, Behavior: {behavior_type})",
            "",
            "Selection Factors:",
        ]
        
        # Add specific reasoning for selected attack
        for reason in all_reasoning[selected_attack]:
            reasoning_parts.append(f"  ✓ {reason}")
        
        reasoning_parts.extend([
            "",
            f"Confidence Score: {all_scores[selected_attack]:.2f}/1.0",
            "",
            "Alternative Options:"
        ])
        
        # Add alternatives
        sorted_alternatives = sorted(
            [(k, v) for k, v in all_scores.items() if k != selected_attack],
            key=lambda x: x[1],
            reverse=True
        )
        
        for alt_attack, alt_score in sorted_alternatives:
            reasoning_parts.append(f"  • {alt_attack}: {alt_score:.2f}/1.0")
        
        return "\n".join(reasoning_parts)
    
    def get_attack_recommendations(
        self,
        target_model: str,
        behavior_types: List[str],
        available_resources: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get attack recommendations for multiple behavior types."""
        recommendations = {}
        
        for behavior_type in behavior_types:
            selection = self.select_attack_method(
                target_model, behavior_type, available_resources
            )
            recommendations[behavior_type] = selection
        
        # Summary statistics
        method_counts = {}
        for rec in recommendations.values():
            method = rec["selected_method"]
            method_counts[method] = method_counts.get(method, 0) + 1
        
        return {
            "recommendations": recommendations,
            "summary": {
                "total_behaviors": len(behavior_types),
                "method_distribution": method_counts,
                "most_recommended": max(method_counts, key=method_counts.get) if method_counts else None
            }
        }
    
    def update_performance_data(self, attack_name: str, model_name: str, success_rate: float):
        """Update historical performance data for learning."""
        key = f"{attack_name}_{model_name}"
        if key not in self.performance_data:
            self.performance_data[key] = []
        self.performance_data[key].append(success_rate)
    
    def get_selection_statistics(self) -> Dict[str, Any]:
        """Get statistics about attack selections."""
        if not self.selection_history:
            return {"message": "No selections recorded yet"}
        
        total_selections = len(self.selection_history)
        method_counts = {}
        safety_level_counts = {}
        
        for record in self.selection_history:
            method = record["selected_attack"]
            safety = record["model_safety"]
            
            method_counts[method] = method_counts.get(method, 0) + 1
            safety_level_counts[safety] = safety_level_counts.get(safety, 0) + 1
        
        return {
            "total_selections": total_selections,
            "method_distribution": {k: v/total_selections for k, v in method_counts.items()},
            "safety_level_distribution": safety_level_counts,
            "average_confidence": sum(r["confidence"] for r in self.selection_history) / total_selections
        }


if __name__ == "__main__":
    # Test the selector
    selector = AttackSelector()
    
    # Test selection
    result = selector.select_attack_method(
        target_model="openchat_3_5_1210",
        behavior_type="illegal",
        available_resources={"gpu_memory_gb": 20, "max_time_minutes": 60}
    )
    
    print("Attack Selection Result:")
    print(f"Selected Method: {result['selected_method']}")
    print(f"Confidence: {result['confidence_score']:.2f}")
    print(f"Reasoning:\n{result['reasoning']}")