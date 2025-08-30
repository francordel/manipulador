import json
import os
from typing import Dict, Any, Optional, List
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Ruta absoluta al archivo tools.json dentro de la carpeta configs/tools
TOOLS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "tools", "tools.json")
TOOLS_PATH = os.path.abspath(TOOLS_PATH)

# Cargar tools.json incluyendo todos los campos (name, description, overview, etc.)
with open(TOOLS_PATH, "r", encoding="utf-8") as f:
    TOOLS = json.load(f)

# Import the behavior-aware agent system
try:
    from ..agent_system import AgentSystem, BehaviorDatasetManager
    _AGENT_SYSTEM = None
    _BEHAVIOR_MANAGER = None
    
    def get_agent_system() -> AgentSystem:
        """Get or create the global agent system instance."""
        global _AGENT_SYSTEM
        if _AGENT_SYSTEM is None:
            _AGENT_SYSTEM = AgentSystem()
            logger.info("Initialized behavior-aware agent system")
        return _AGENT_SYSTEM
    
    def get_behavior_manager() -> BehaviorDatasetManager:
        """Get or create the global behavior manager instance."""
        global _BEHAVIOR_MANAGER
        if _BEHAVIOR_MANAGER is None:
            _BEHAVIOR_MANAGER = BehaviorDatasetManager()
            logger.info("Initialized behavior dataset manager")
        return _BEHAVIOR_MANAGER
        
except ImportError as e:
    logger.warning(f"Could not import behavior-aware components: {e}")
    _AGENT_SYSTEM = None
    _BEHAVIOR_MANAGER = None
    
    def get_agent_system():
        return None
    
    def get_behavior_manager():
        return None


# Enhanced tool selection functions
def select_attack_method_behavior_aware(target_model: str,
                                      behavior_name: Optional[str] = None,
                                      behavior_type: Optional[str] = None,
                                      gpu_memory_gb: float = 16.0,
                                      max_time_minutes: int = 60,
                                      use_historical_data: bool = True,
                                      preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Enhanced attack method selection using behavior-aware intelligence.
    
    Args:
        target_model: Name of the target model
        behavior_name: Specific behavior name (e.g., 'fauci_lab_leak_involvement')
        behavior_type: General behavior type (e.g., 'misinformation') 
        gpu_memory_gb: Available GPU memory
        max_time_minutes: Time limit in minutes
        use_historical_data: Whether to use historical performance data
        preferences: Additional user preferences
        
    Returns:
        Dictionary containing selection results and recommendations
    """
    
    agent_system = get_agent_system()
    if not agent_system or not use_historical_data:
        # Fallback to legacy selection
        return _legacy_method_selection(target_model, behavior_type or behavior_name,
                                      gpu_memory_gb, max_time_minutes)
    
    try:
        # Create behavior-aware context
        context = agent_system.create_context(
            target_model=target_model,
            behavior_type=behavior_type or "unknown",
            behavior_name=behavior_name,
            gpu_memory_gb=gpu_memory_gb,
            time_limit_minutes=max_time_minutes,
            available_tools=[
                {"name": "select_attack_method", "description": "Behavior-aware attack method selection"},
                {"name": "execute_attack", "description": "Execute selected attack method"}
            ],
            user_preferences=preferences
        )
        
        # Get optimal method directly for best performance
        if context.behavior_name:
            selected_method, confidence, reasoning = agent_system.select_optimal_method_direct(
                behavior_name=context.behavior_name,
                target_model=target_model,
                gpu_memory_gb=gpu_memory_gb,
                time_limit_minutes=max_time_minutes
            )
            
            # Get additional analysis
            recommendations = agent_system.get_behavior_recommendations(context.behavior_name)
            
            return {
                "selected_method": selected_method,
                "confidence": confidence,
                "reasoning": reasoning,
                "behavior_name": context.behavior_name,
                "available_methods": context.available_methods,
                "method_performance": context.method_performance,
                "recommendations": recommendations,
                "target_model": target_model,
                "resource_constraints": {
                    "gpu_memory_gb": gpu_memory_gb,
                    "max_time_minutes": max_time_minutes
                },
                "success": True,
                "method": "behavior_aware_selection"
            }
        else:
            # No specific behavior found, use LLM agent
            tool_call = agent_system.select_tool(context)
            
            return {
                "selected_method": tool_call.args.get("selected_method", "DirectRequest"),
                "confidence": tool_call.confidence,
                "reasoning": tool_call.reasoning,
                "behavior_name": behavior_name or behavior_type,
                "available_methods": [],
                "method_performance": {},
                "target_model": target_model,
                "resource_constraints": {
                    "gpu_memory_gb": gpu_memory_gb,
                    "max_time_minutes": max_time_minutes
                },
                "success": True,
                "method": "llm_agent_selection"
            }
            
    except Exception as e:
        logger.error(f"Behavior-aware selection failed: {e}")
        return _legacy_method_selection(target_model, behavior_type or behavior_name,
                                      gpu_memory_gb, max_time_minutes)


def _legacy_method_selection(target_model: str, behavior: str, 
                           gpu_memory: float, time_limit: int) -> Dict[str, Any]:
    """Legacy method selection for fallback."""
    
    # Simple heuristic-based selection
    if gpu_memory >= 20 and time_limit >= 90:
        method = "GPTFuzz"
        confidence = 0.7
        reasoning = "High resources available, using sophisticated fuzzing method"
    elif gpu_memory >= 8 and time_limit >= 30:
        method = "FewShot"
        confidence = 0.8
        reasoning = "Medium resources, using few-shot prompting for efficiency"
    else:
        method = "DirectRequest"
        confidence = 0.6
        reasoning = "Limited resources, using direct request method"
    
    return {
        "selected_method": method,
        "confidence": confidence,
        "reasoning": reasoning,
        "behavior_name": behavior,
        "available_methods": ["DirectRequest", "FewShot", "GPTFuzz"],
        "method_performance": {},
        "target_model": target_model,
        "resource_constraints": {
            "gpu_memory_gb": gpu_memory,
            "max_time_minutes": time_limit
        },
        "success": True,
        "method": "legacy_heuristic"
    }


def get_behavior_analysis(behavior_name: str) -> Dict[str, Any]:
    """Get detailed analysis for a specific behavior.
    
    Args:
        behavior_name: Name of the behavior to analyze
        
    Returns:
        Dictionary containing behavior analysis and method recommendations
    """
    
    behavior_manager = get_behavior_manager()
    agent_system = get_agent_system()
    
    if not behavior_manager or not agent_system:
        return {"error": "Behavior analysis not available", "behavior_name": behavior_name}
    
    try:
        # Get comprehensive recommendations
        recommendations = agent_system.get_behavior_recommendations(behavior_name)
        
        if "error" in recommendations:
            return recommendations
        
        # Add success rates and method rankings
        success_rates = recommendations.get('success_rates', {})
        method_ranking = sorted(success_rates.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate overall statistics
        avg_asr = sum(success_rates.values()) / len(success_rates) if success_rates else 0
        best_asr = max(success_rates.values()) if success_rates else 0
        worst_asr = min(success_rates.values()) if success_rates else 0
        
        return {
            "behavior_name": behavior_name,
            "behavior_data": recommendations.get('behavior_data', {}),
            "method_ranking": method_ranking,
            "statistics": {
                "avg_asr": avg_asr,
                "best_asr": best_asr,
                "worst_asr": worst_asr,
                "method_count": len(success_rates)
            },
            "method_analysis": recommendations.get('method_analysis', {}),
            "scenario_recommendations": recommendations.get('scenario_recommendations', {}),
            "overall_best_method": recommendations.get('overall_best_method'),
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Failed to get behavior analysis for {behavior_name}: {e}")
        return {"error": str(e), "behavior_name": behavior_name, "success": False}


def list_available_behaviors(category: Optional[str] = None, 
                           min_asr: Optional[float] = None,
                           method: Optional[str] = None) -> Dict[str, Any]:
    """List available behaviors with optional filtering.
    
    Args:
        category: Filter by behavior category
        min_asr: Minimum ASR threshold for any method
        method: Filter by available method
        
    Returns:
        Dictionary containing filtered behavior list
    """
    
    behavior_manager = get_behavior_manager()
    if not behavior_manager:
        return {"error": "Behavior manager not available", "behaviors": []}
    
    try:
        all_behaviors = behavior_manager.behavior_data
        filtered_behaviors = {}
        
        for behavior_name, behavior_data in all_behaviors.items():
            # Apply filters
            if category and behavior_data.get('category', '').lower() != category.lower():
                continue
                
            if method and method not in behavior_data.get('available_methods', []):
                continue
                
            if min_asr is not None:
                # Check if any method meets the ASR threshold
                performance_metrics = behavior_data.get('performance_metrics', {})
                max_asr = max([metrics.get('asr', 0) for metrics in performance_metrics.values()], 
                            default=0)
                if max_asr < min_asr:
                    continue
            
            # Include this behavior
            filtered_behaviors[behavior_name] = {
                'behavior': behavior_data.get('behavior', ''),
                'category': behavior_data.get('category', ''),
                'complexity': behavior_data.get('complexity', ''),
                'available_methods': behavior_data.get('available_methods', []),
                'best_asr': max([metrics.get('asr', 0) 
                               for metrics in behavior_data.get('performance_metrics', {}).values()], 
                              default=0)
            }
        
        return {
            "behaviors": filtered_behaviors,
            "count": len(filtered_behaviors),
            "filters_applied": {
                "category": category,
                "min_asr": min_asr,
                "method": method
            },
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Failed to list behaviors: {e}")
        return {"error": str(e), "behaviors": {}, "success": False}
