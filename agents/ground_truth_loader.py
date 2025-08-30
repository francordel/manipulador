#!/usr/bin/env python3
"""
Ground Truth Loader for Real Agentic Evaluation

This module loads actual ground truth labels from the experimental result files
for scoring AFTER agent decisions are made (clean separation principle).

IMPORTANT: This is ONLY used for scoring, NEVER for agent decision making!
"""

import json
import os
import glob
from typing import Dict, List, Optional, Tuple
import warnings

# Label key priorities for each method
LABEL_KEYS = {
    "FewShot": ["manual_label", "fewshot_label", "label", "success"],
    "DirectRequest": ["manual_label", "directrequest_label", "label", "success"], 
    "GPTFuzz": ["claude_label", "label", "manual_label", "success"],
}

# Behavior key priorities
BEHAVIOR_KEYS = ["behavior", "behavior_name", "prompt"]

# Model folder name to agent name mapping
MODEL_FOLDER_TO_AGENT = {
    "granite_2b_small": "granite_3_3_2b_instruct",
    "granite_8b_medium": "granite_3_3_8b_instruct",
    "llama_1b_small": "llama_3_2_1b",
    "llama_3b_medium": "llama_3_2_3b_instruct", 
    "llama_70b_large": "meta_llama_3_70b",
    "mistral_7b_v03_medium": "mistral_7b_v03_medium",
    "mixtral_8x7b_large": "mixtral_8x7b_v01",
    "qwen2_0_5b_micro": "qwen2_5_0_5b_instruct",  # Special mapping for paper
    "qwen2_1_5b_small": "qwen2_1_5b_instruct",
    "qwen2_7b_medium": "qwen2_7b_instruct"
}

# Reverse mapping for lookups
AGENT_TO_MODEL_FOLDER = {v: k for k, v in MODEL_FOLDER_TO_AGENT.items()}

class GroundTruthLoader:
    """Loads ground truth labels for scoring evaluation results."""
    
    def __init__(self, base_path: str = "."):
        self.base_path = base_path
        self.ground_truth_cache = {}  # Cache loaded results
        self.missing_files = []  # Track missing files for reporting
    
    def glob_paths(self, method: str, folder: str) -> List[str]:
        """Get file paths for a method-model combination."""
        if method == "FewShot":
            pattern = f"{self.base_path}/results/FewShot/{folder}/results/{folder}_complete_manual_batch*.json"
            return sorted(glob.glob(pattern))
        elif method == "DirectRequest":
            pattern = f"{self.base_path}/results/DirectRequest/{folder}/DirectRequest/default/results/{folder}_manual_corrections_batch*.json"
            return sorted(glob.glob(pattern))
        elif method == "GPTFuzz":
            path = f"{self.base_path}/results/GPTFuzz/{folder}/results/{folder}_claude_personal_evaluation.json"
            return [path] if os.path.exists(path) else []
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def read_label(self, item: dict, method: str) -> int:
        """Extract label from item with fallback priority."""
        for key in LABEL_KEYS[method]:
            if key in item:
                value = item[key]
                # Convert to 0/1
                if value in (1, True, "1", "true", "True"):
                    return 1
                elif value in (0, False, "0", "false", "False"):
                    return 0
                else:
                    warnings.warn(f"Unexpected label value: {value} for key {key}")
                    return int(bool(value))  # Fallback conversion
        
        raise KeyError(f"No label key found in {list(item.keys())} for method {method}")
    
    def read_behavior(self, item: dict) -> str:
        """Extract behavior name from item with fallback priority."""
        for key in BEHAVIOR_KEYS:
            if key in item:
                return item[key]
        
        raise KeyError(f"No behavior key found in {list(item.keys())}")
    
    def load_ground_truth(self, method: str, folder: str) -> Dict[str, int]:
        """Load ground truth for a method-model combination."""
        cache_key = f"{method}_{folder}"
        
        if cache_key in self.ground_truth_cache:
            return self.ground_truth_cache[cache_key]
        
        ground_truth = {}
        file_paths = self.glob_paths(method, folder)
        
        if not file_paths:
            self.missing_files.append(f"{method}/{folder}")
            warnings.warn(f"No files found for {method}/{folder}")
            self.ground_truth_cache[cache_key] = {}
            return {}
        
        files_loaded = 0
        for path in file_paths:
            if not os.path.exists(path):
                self.missing_files.append(path)
                warnings.warn(f"Missing file: {path}")
                continue
                
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if not isinstance(data, dict):
                    warnings.warn(f"Expected dict in {path}, got {type(data)}")
                    continue
                
                # Handle different file structures based on method
                if method == "FewShot":
                    # FewShot files have "manual_evaluations" section
                    evaluations = data.get("manual_evaluations", {})
                    for key, item in evaluations.items():
                        if not isinstance(item, dict):
                            continue
                        try:
                            # Extract behavior name from key like "khmer_rouge_beneficial_regime_case_0"
                            behavior = key.replace("_case_0", "").replace("_case_", "_case")
                            if behavior.endswith("_case"):
                                behavior = behavior[:-5]  # Remove "_case" suffix
                            label = self.read_label(item, method)
                            ground_truth[behavior] = label
                        except KeyError as e:
                            warnings.warn(f"Skipping item {key} in {path}: {e}")
                            continue
                
                elif method == "DirectRequest":
                    # DirectRequest files have "corrections" section
                    corrections = data.get("corrections", {})
                    for behavior, item in corrections.items():
                        if not isinstance(item, dict):
                            continue
                        try:
                            label = self.read_label(item, method)
                            ground_truth[behavior] = label
                        except KeyError as e:
                            warnings.warn(f"Skipping item {behavior} in {path}: {e}")
                            continue
                
                elif method == "GPTFuzz":
                    # GPTFuzz files have behavior names as top-level keys
                    for behavior, items in data.items():
                        if not isinstance(items, list) or not items:
                            continue
                        try:
                            # Use first item in list
                            item = items[0] if isinstance(items[0], dict) else {}
                            # For GPTFuzz, use claude_personal_label or claude_label
                            label = None
                            for key in ["claude_personal_label", "claude_label", "label"]:
                                if key in item:
                                    value = item[key]
                                    label = 1 if value in (1, True, "1", "true", "True") else 0
                                    break
                            if label is not None:
                                ground_truth[behavior] = label
                        except Exception as e:
                            warnings.warn(f"Skipping item {behavior} in {path}: {e}")
                            continue
                
                files_loaded += 1
                
            except Exception as e:
                self.missing_files.append(path)
                warnings.warn(f"Error loading {path}: {e}")
                continue
        
        if files_loaded > 0:
            print(f"✅ Loaded {len(ground_truth)} behaviors for {method}/{folder} from {files_loaded} files")
        else:
            warnings.warn(f"No files successfully loaded for {method}/{folder}")
        
        self.ground_truth_cache[cache_key] = ground_truth
        return ground_truth
    
    def get_label(self, agent_model: str, behavior: str, method: str) -> Optional[int]:
        """Get ground truth label for agent_model, behavior, method combination."""
        # Map agent model name to folder name
        if agent_model not in AGENT_TO_MODEL_FOLDER:
            warnings.warn(f"Unknown agent model: {agent_model}")
            return None
        
        folder = AGENT_TO_MODEL_FOLDER[agent_model]
        ground_truth = self.load_ground_truth(method, folder)
        
        return ground_truth.get(behavior)
    
    def get_all_labels(self, agent_model: str) -> Dict[str, Dict[str, int]]:
        """Get all ground truth labels for a model across all methods."""
        if agent_model not in AGENT_TO_MODEL_FOLDER:
            warnings.warn(f"Unknown agent model: {agent_model}")
            return {}
        
        folder = AGENT_TO_MODEL_FOLDER[agent_model]
        results = {}
        
        for method in ["FewShot", "DirectRequest", "GPTFuzz"]:
            results[method] = self.load_ground_truth(method, folder)
        
        return results
    
    def get_coverage_report(self) -> Dict[str, any]:
        """Generate coverage report showing what data is available."""
        coverage = {}
        
        for agent_model, folder in AGENT_TO_MODEL_FOLDER.items():
            model_coverage = {}
            
            for method in ["FewShot", "DirectRequest", "GPTFuzz"]:
                ground_truth = self.load_ground_truth(method, folder)
                model_coverage[method] = {
                    "behaviors_count": len(ground_truth),
                    "behaviors": list(ground_truth.keys()) if len(ground_truth) <= 10 else list(ground_truth.keys())[:10] + ["..."]
                }
            
            coverage[agent_model] = model_coverage
        
        return {
            "model_coverage": coverage,
            "missing_files": self.missing_files,
            "total_missing": len(self.missing_files)
        }
    
    def validate_behavior_consistency(self) -> Dict[str, any]:
        """Validate that behavior names are consistent across methods."""
        all_behaviors = set()
        method_behaviors = {}
        
        for agent_model, folder in AGENT_TO_MODEL_FOLDER.items():
            for method in ["FewShot", "DirectRequest", "GPTFuzz"]:
                ground_truth = self.load_ground_truth(method, folder)
                key = f"{agent_model}_{method}"
                method_behaviors[key] = set(ground_truth.keys())
                all_behaviors.update(ground_truth.keys())
        
        # Find common behaviors across all method-model combinations
        common_behaviors = all_behaviors.copy()
        for behaviors in method_behaviors.values():
            common_behaviors &= behaviors
        
        # Find missing behaviors per method-model combination
        missing_report = {}
        for key, behaviors in method_behaviors.items():
            missing = all_behaviors - behaviors
            if missing:
                missing_report[key] = list(missing)
        
        return {
            "total_unique_behaviors": len(all_behaviors),
            "common_behaviors_count": len(common_behaviors),
            "common_behaviors": sorted(list(common_behaviors)),
            "missing_behaviors_by_method_model": missing_report
        }

def main():
    """Test the ground truth loader."""
    print("🧪 Testing Ground Truth Loader")
    print("=" * 50)
    
    loader = GroundTruthLoader()
    
    # Test loading for one model
    print("\n📊 Testing single model load:")
    test_model = "mistral_7b_v03_medium"
    all_labels = loader.get_all_labels(test_model)
    
    for method, labels in all_labels.items():
        print(f"  {method}: {len(labels)} behaviors")
        if labels:
            sample_behavior = next(iter(labels))
            sample_label = labels[sample_behavior]
            print(f"    Sample: {sample_behavior} → {sample_label}")
    
    # Generate coverage report
    print("\n📋 Coverage Report:")
    coverage = loader.get_coverage_report()
    
    for model, methods in coverage["model_coverage"].items():
        print(f"  {model}:")
        for method, info in methods.items():
            print(f"    {method}: {info['behaviors_count']} behaviors")
    
    if coverage["missing_files"]:
        print(f"\n⚠️  Missing files ({coverage['total_missing']}):")
        for missing_file in coverage["missing_files"][:5]:  # Show first 5
            print(f"    {missing_file}")
        if coverage["total_missing"] > 5:
            print(f"    ... and {coverage['total_missing'] - 5} more")
    
    # Validate behavior consistency
    print("\n🔍 Behavior Consistency Check:")
    consistency = loader.validate_behavior_consistency()
    print(f"  Total unique behaviors: {consistency['total_unique_behaviors']}")
    print(f"  Common across all: {consistency['common_behaviors_count']}")
    
    if consistency["missing_behaviors_by_method_model"]:
        print(f"  Missing data detected:")
        for key, missing in list(consistency["missing_behaviors_by_method_model"].items())[:3]:
            print(f"    {key}: {len(missing)} missing behaviors")

if __name__ == "__main__":
    main()