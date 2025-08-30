"""
Execution Engine - Command execution wrapper for running red teaming attacks.

This module provides a standardized interface for executing different attack
methods with proper parameter handling, logging, and result processing.
"""

import os
import json
import logging
import subprocess
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import asdict

from .attack_registry import get_attack_method
from .model_profiler import get_model_profile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExecutionEngine:
    """Standardized execution engine for red teaming attacks."""
    
    def __init__(self, scripts_dir: Optional[str] = None):
        """Initialize execution engine.
        
        Args:
            scripts_dir: Directory containing attack execution scripts
        """
        if scripts_dir is None:
            scripts_dir = Path(__file__).parent / "scripts"
        
        self.scripts_dir = Path(scripts_dir)
        self.scripts_dir.mkdir(exist_ok=True)
        
        # Map attack methods to their execution scripts
        self.script_mapping = {
            "DirectRequest": "run_directrequest.sh",
            "FewShot": "run_fewshot.sh", 
            "GPTFuzz": "run_gptfuzz.sh"
        }
        
        # Default parameters for each attack method
        self.default_params = {
            "DirectRequest": {
                "timeout_minutes": 10,
                "max_retries": 3
            },
            "FewShot": {
                "n_shots": 5,
                "sample_size": 1000,
                "num_steps": 5,
                "timeout_minutes": 60
            },
            "GPTFuzz": {
                "max_query": 1000,
                "max_jailbreak": 1,
                "max_iteration": 100,
                "energy": 1,
                "mutation_model": "openchat-3.5",
                "dtype": "float16",
                "timeout_minutes": 180
            }
        }
    
    def execute_attack(
        self,
        attack_method: str,
        target_model: str,
        behavior_file: str,
        results_dir: str = "./results",
        custom_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a red teaming attack.
        
        Args:
            attack_method: Name of attack method to execute
            target_model: Target model name
            behavior_file: Path to behavior file or behavior IDs
            results_dir: Directory to save results
            custom_params: Custom parameters for the attack
            
        Returns:
            Dict containing execution results and metadata
        """
        logger.info(f"Executing {attack_method} attack on {target_model}")
        
        # Validate attack method
        if attack_method not in self.script_mapping:
            return {
                "status": "error",
                "error": f"Unknown attack method: {attack_method}",
                "available_methods": list(self.script_mapping.keys())
            }
        
        # Get attack method details
        try:
            method_details = get_attack_method(attack_method)
            model_profile = get_model_profile(target_model)
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to get method/model details: {str(e)}"
            }
        
        # Prepare execution parameters
        execution_params = self._prepare_execution_params(
            attack_method=attack_method,
            target_model=target_model,
            behavior_file=behavior_file,
            results_dir=results_dir,
            custom_params=custom_params or {}
        )
        
        # Check resource requirements
        resource_check = self._check_resource_requirements(
            method_details, execution_params.get("available_resources", {})
        )
        if not resource_check["sufficient"]:
            return {
                "status": "error",
                "error": "Insufficient resources for attack execution",
                "resource_check": resource_check
            }
        
        # Execute the attack
        start_time = time.time()
        
        try:
            execution_result = self._run_attack_script(
                attack_method=attack_method,
                execution_params=execution_params
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Process results
            results = self._process_attack_results(
                attack_method=attack_method,
                execution_result=execution_result,
                execution_params=execution_params,
                execution_time=execution_time
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Attack execution failed: {str(e)}")
            return {
                "status": "error",
                "error": f"Attack execution failed: {str(e)}",
                "execution_time": time.time() - start_time
            }
    
    def _prepare_execution_params(
        self,
        attack_method: str,
        target_model: str,
        behavior_file: str,
        results_dir: str,
        custom_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare parameters for attack execution."""
        
        # Get default parameters for the attack method
        default_params = self.default_params.get(attack_method, {})
        
        # Merge with custom parameters
        merged_params = {**default_params, **custom_params}
        
        # Prepare base execution parameters
        execution_params = {
            "attack_method": attack_method,
            "target_model": target_model,
            "behavior_file": behavior_file,
            "results_dir": results_dir,
            "script_path": self.scripts_dir / self.script_mapping[attack_method],
            **merged_params
        }
        
        # Add method-specific parameters
        if attack_method == "GPTFuzz":
            execution_params.update({
                "mutation_model": merged_params.get("mutation_model", "openchat-3.5"),
                "dtype": merged_params.get("dtype", "float16"),
                "use_local_model": True
            })
        elif attack_method == "FewShot":
            execution_params.update({
                "n_shots": merged_params.get("n_shots", 5),
                "sample_size": merged_params.get("sample_size", 1000)
            })
        
        return execution_params
    
    def _check_resource_requirements(
        self,
        method_details,
        available_resources: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if available resources meet attack requirements."""
        
        required_gpu = method_details.resource_requirements.get("gpu_memory_gb", 0)
        required_time = method_details.resource_requirements.get("execution_time_minutes", 0)
        
        available_gpu = available_resources.get("gpu_memory_gb", 0)
        available_time = available_resources.get("max_time_minutes", float('inf'))
        
        gpu_sufficient = available_gpu >= required_gpu
        time_sufficient = available_time >= required_time
        
        return {
            "sufficient": gpu_sufficient and time_sufficient,
            "gpu_check": {
                "required": required_gpu,
                "available": available_gpu,
                "sufficient": gpu_sufficient
            },
            "time_check": {
                "required": required_time,
                "available": available_time,
                "sufficient": time_sufficient
            }
        }
    
    def _run_attack_script(
        self,
        attack_method: str,
        execution_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run the attack execution script."""
        
        script_path = execution_params["script_path"]
        
        # Ensure script exists and is executable
        if not script_path.exists():
            raise FileNotFoundError(f"Attack script not found: {script_path}")
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        # Prepare command arguments based on attack method
        if attack_method == "DirectRequest":
            cmd = [
                str(script_path),
                execution_params["target_model"],
                execution_params["behavior_file"],
                execution_params["results_dir"]
            ]
        elif attack_method == "FewShot":
            cmd = [
                str(script_path),
                execution_params["target_model"],
                execution_params["behavior_file"],
                execution_params["results_dir"],
                str(execution_params.get("n_shots", 5)),
                str(execution_params.get("sample_size", 1000))
            ]
        elif attack_method == "GPTFuzz":
            cmd = [
                str(script_path),
                execution_params["target_model"],
                execution_params["behavior_file"],
                execution_params["results_dir"],
                execution_params.get("mutation_model", "openchat-3.5"),
                execution_params.get("dtype", "float16")
            ]
        else:
            raise ValueError(f"Unknown attack method: {attack_method}")
        
        # Set timeout
        timeout = execution_params.get("timeout_minutes", 60) * 60  # Convert to seconds
        
        logger.info(f"Executing command: {' '.join(cmd)}")
        
        # Execute the script
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=Path.cwd()
            )
            
            return {
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": cmd,
                "timeout": timeout
            }
            
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Attack execution timed out after {timeout} seconds")
        except Exception as e:
            raise RuntimeError(f"Failed to execute attack script: {str(e)}")
    
    def _process_attack_results(
        self,
        attack_method: str,
        execution_result: Dict[str, Any],
        execution_params: Dict[str, Any],
        execution_time: float
    ) -> Dict[str, Any]:
        """Process and format attack execution results."""
        
        success = execution_result["return_code"] == 0
        
        # Basic result structure
        results = {
            "status": "success" if success else "failed",
            "attack_method": attack_method,
            "target_model": execution_params["target_model"],
            "behavior_file": execution_params["behavior_file"],
            "execution_time_seconds": execution_time,
            "execution_time_minutes": execution_time / 60,
            "return_code": execution_result["return_code"],
            "stdout": execution_result["stdout"],
            "stderr": execution_result["stderr"]
        }
        
        # Try to parse attack-specific results from stdout/results files
        try:
            parsed_results = self._parse_attack_output(
                attack_method, execution_params["results_dir"], execution_result["stdout"]
            )
            results.update(parsed_results)
        except Exception as e:
            logger.warning(f"Failed to parse attack results: {str(e)}")
            results["parsing_error"] = str(e)
        
        return results
    
    def _parse_attack_output(
        self,
        attack_method: str,
        results_dir: str,
        stdout: str
    ) -> Dict[str, Any]:
        """Parse attack-specific output and result files."""
        
        parsed = {
            "success_rate": 0.0,
            "total_attempts": 0,
            "successful_attacks": 0,
            "jailbreaks_found": 0,
            "details": {}
        }
        
        # Look for result files in the results directory
        results_path = Path(results_dir)
        if results_path.exists():
            # Look for common result file patterns
            for file_pattern in ["*results*.json", "*output*.json", "*report*.json"]:
                result_files = list(results_path.glob(file_pattern))
                for result_file in result_files:
                    try:
                        with open(result_file, 'r') as f:
                            file_data = json.load(f)
                            parsed["details"][str(result_file)] = file_data
                            
                            # Extract common metrics
                            if "success_rate" in file_data:
                                parsed["success_rate"] = file_data["success_rate"]
                            if "total_queries" in file_data:
                                parsed["total_attempts"] = file_data["total_queries"]
                            if "jailbreaks" in file_data:
                                parsed["jailbreaks_found"] = len(file_data["jailbreaks"])
                                
                    except Exception as e:
                        logger.warning(f"Failed to parse result file {result_file}: {str(e)}")
        
        # Parse stdout for additional information
        if "jailbreak" in stdout.lower():
            # Try to extract jailbreak count from stdout
            import re
            jailbreak_matches = re.findall(r'(\d+)\s*jailbreak', stdout.lower())
            if jailbreak_matches:
                parsed["jailbreaks_found"] = max(parsed["jailbreaks_found"], int(jailbreak_matches[-1]))
        
        # Calculate success rate if we have the data
        if parsed["total_attempts"] > 0:
            parsed["success_rate"] = parsed["jailbreaks_found"] / parsed["total_attempts"]
        
        return parsed
    
    def list_available_scripts(self) -> Dict[str, Any]:
        """List available attack execution scripts."""
        scripts = {}
        
        for attack_method, script_name in self.script_mapping.items():
            script_path = self.scripts_dir / script_name
            scripts[attack_method] = {
                "script_name": script_name,
                "script_path": str(script_path),
                "exists": script_path.exists(),
                "executable": script_path.exists() and os.access(script_path, os.X_OK)
            }
        
        return scripts
    
    def validate_setup(self) -> Dict[str, Any]:
        """Validate execution engine setup."""
        validation = {
            "scripts_directory": str(self.scripts_dir),
            "scripts_directory_exists": self.scripts_dir.exists(),
            "scripts": self.list_available_scripts(),
            "issues": []
        }
        
        # Check for issues
        if not self.scripts_dir.exists():
            validation["issues"].append("Scripts directory does not exist")
        
        for attack_method, script_info in validation["scripts"].items():
            if not script_info["exists"]:
                validation["issues"].append(f"Missing script for {attack_method}: {script_info['script_name']}")
            elif not script_info["executable"]:
                validation["issues"].append(f"Script not executable for {attack_method}: {script_info['script_name']}")
        
        validation["valid"] = len(validation["issues"]) == 0
        
        return validation


if __name__ == "__main__":
    # Test the execution engine
    engine = ExecutionEngine()
    
    # Validate setup
    validation = engine.validate_setup()
    print(f"Setup validation: {json.dumps(validation, indent=2)}")
    
    # List available scripts
    scripts = engine.list_available_scripts()
    print(f"\nAvailable scripts: {json.dumps(scripts, indent=2)}")