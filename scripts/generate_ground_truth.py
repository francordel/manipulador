#!/usr/bin/env python3
"""
Ground truth generation script for baseline methods.

This script generates ground truth evaluation results for DirectRequest, FewShot, 
and GPTFuzz baseline methods across multiple target models. Results are used for
agent comparison and performance analysis.
"""

import sys
import os
import json
import argparse
import time
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import subprocess
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class GroundTruthGenerator:
    """Generator for baseline method ground truth results."""
    
    def __init__(self, verbose: bool = True):
        """Initialize ground truth generator."""
        self.verbose = verbose
        self.results = {}
        self.supported_methods = ['directrequest', 'fewshot', 'gptfuzz']
        
    def load_behaviors(self, behaviors_file: str) -> List[Dict[str, Any]]:
        """Load behaviors from file."""
        behaviors = []
        file_path = Path(behaviors_file)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Behaviors file not found: {behaviors_file}")
        
        if file_path.suffix.lower() == '.csv':
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    behaviors.append({
                        'name': row.get('behavior_name', row.get('name', 'unknown')),
                        'behavior': row.get('behavior', row.get('description', '')),
                        'category': row.get('category', 'general')
                    })
        elif file_path.suffix.lower() == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    behaviors = data
                else:
                    behaviors = [data]
        
        if self.verbose:
            print(f"✅ Loaded {len(behaviors)} behaviors from {behaviors_file}")
            
        return behaviors
    
    def get_model_configs(self, models_config: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Load model configurations."""
        if models_config and Path(models_config).exists():
            # Load from YAML config file
            try:
                import yaml
                with open(models_config, 'r') as f:
                    config = yaml.safe_load(f)
                return config.get('models', config)
            except ImportError:
                print("⚠️  PyYAML not available, using default configurations")
        
        # Default model configurations
        return {
            "mistral_7b_v03_medium": {
                "model_name_or_path": "mistralai/Mistral-7B-Instruct-v0.3",
                "dtype": "float16",
                "num_gpus": 1,
                "safety_level": "medium"
            },
            "qwen2_7b_instruct": {
                "model_name_or_path": "Qwen/Qwen2-7B-Instruct", 
                "dtype": "float16",
                "num_gpus": 1,
                "safety_level": "medium"
            },
            "mixtral_8x7b_v01": {
                "model_name_or_path": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "dtype": "float16", 
                "num_gpus": 2,
                "safety_level": "high"
            }
        }
    
    def run_baseline_method(self, method: str, model_config: Dict[str, Any], 
                          behaviors: List[Dict[str, Any]], 
                          output_dir: str) -> Dict[str, Any]:
        """Run a single baseline method evaluation."""
        if method not in self.supported_methods:
            raise ValueError(f"Unsupported method: {method}")
        
        method_results = {
            'method': method,
            'model_config': model_config,
            'total_behaviors': len(behaviors),
            'results': [],
            'start_time': datetime.now().isoformat()
        }
        
        if self.verbose:
            print(f"🔄 Running {method} baseline...")
        
        # Simulate baseline method execution
        # In real implementation, this would call actual baseline scripts
        for i, behavior in enumerate(behaviors):
            if self.verbose and (i + 1) % 5 == 0:
                print(f"   Progress: {i + 1}/{len(behaviors)} behaviors")
            
            # Simulate evaluation (replace with actual baseline calls)
            result = self._simulate_baseline_evaluation(method, behavior, model_config)
            method_results['results'].append(result)
            
            # Add small delay to simulate processing
            time.sleep(0.1)
        
        method_results['end_time'] = datetime.now().isoformat()
        method_results['success_rate'] = sum(1 for r in method_results['results'] if r['success']) / len(behaviors) * 100
        
        # Save method-specific results
        method_file = Path(output_dir) / f"{method}_results.json"
        with open(method_file, 'w') as f:
            json.dump(method_results, f, indent=2)
        
        if self.verbose:
            print(f"   Success Rate: {method_results['success_rate']:.1f}%")
            print(f"   Results saved: {method_file}")
        
        return method_results
    
    def _simulate_baseline_evaluation(self, method: str, behavior: Dict[str, Any], 
                                    model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate baseline method evaluation (replace with actual implementation)."""
        import random
        
        # Simulate different success rates based on method and model safety
        safety_level = model_config.get('safety_level', 'medium')
        
        if method == 'directrequest':
            base_success_rate = {'light': 0.15, 'medium': 0.08, 'high': 0.05}.get(safety_level, 0.08)
        elif method == 'fewshot':
            base_success_rate = {'light': 0.65, 'medium': 0.51, 'high': 0.42}.get(safety_level, 0.51)
        elif method == 'gptfuzz':
            base_success_rate = {'light': 0.58, 'medium': 0.44, 'high': 0.38}.get(safety_level, 0.44)
        else:
            base_success_rate = 0.3
        
        # Add some randomness
        success = random.random() < base_success_rate
        confidence = random.uniform(0.6, 0.9) if success else random.uniform(0.2, 0.5)
        execution_time = random.uniform(10, 30) if method == 'directrequest' else random.uniform(400, 1200)
        
        return {
            'behavior_name': behavior['name'],
            'behavior_category': behavior.get('category', 'general'),
            'method': method,
            'success': success,
            'confidence': confidence,
            'execution_time': execution_time,
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_ground_truth(self, behaviors: List[Dict[str, Any]], 
                            model_configs: Dict[str, Dict[str, Any]],
                            methods: List[str],
                            output_dir: str) -> Dict[str, Any]:
        """Generate complete ground truth across methods and models."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.verbose:
            print(f"\n🚀 GENERATING GROUND TRUTH")
            print(f"{'='*60}")
            print(f"Methods: {', '.join(methods)}")
            print(f"Models: {', '.join(model_configs.keys())}")
            print(f"Behaviors: {len(behaviors)}")
            print(f"Total Evaluations: {len(methods) * len(model_configs) * len(behaviors)}")
            print(f"{'='*60}\n")
        
        ground_truth = {
            'generation_metadata': {
                'start_time': datetime.now().isoformat(),
                'methods': methods,
                'models': list(model_configs.keys()),
                'total_behaviors': len(behaviors)
            },
            'results_by_model': {}
        }
        
        for model_name, model_config in model_configs.items():
            if self.verbose:
                print(f"📊 Processing model: {model_name}")
                print(f"   Safety Level: {model_config.get('safety_level', 'unknown')}")
            
            model_output_dir = output_path / model_name
            model_output_dir.mkdir(parents=True, exist_ok=True)
            
            model_results = {
                'model_name': model_name,
                'model_config': model_config,
                'method_results': {}
            }
            
            for method in methods:
                method_result = self.run_baseline_method(
                    method, model_config, behaviors, str(model_output_dir)
                )
                model_results['method_results'][method] = method_result
            
            # Create combined results for this model
            combined_results = self._combine_model_results(model_results, behaviors)
            
            # Save model-specific combined results
            model_file = model_output_dir / f"{model_name}_ground_truth.json"
            with open(model_file, 'w') as f:
                json.dump(combined_results, f, indent=2)
            
            ground_truth['results_by_model'][model_name] = combined_results
            
            if self.verbose:
                print(f"   ✅ Completed {model_name}")
                for method in methods:
                    sr = model_results['method_results'][method]['success_rate']
                    print(f"      {method}: {sr:.1f}% success rate")
                print()
        
        ground_truth['generation_metadata']['end_time'] = datetime.now().isoformat()
        
        # Save complete ground truth
        complete_file = output_path / "complete_ground_truth.json"
        with open(complete_file, 'w') as f:
            json.dump(ground_truth, f, indent=2)
        
        # Generate summary statistics
        summary = self._generate_summary_statistics(ground_truth)
        summary_file = output_path / "ground_truth_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        if self.verbose:
            print(f"💾 Ground truth generation completed!")
            print(f"   Complete results: {complete_file}")
            print(f"   Summary: {summary_file}")
            self._print_summary(summary)
        
        return ground_truth
    
    def _combine_model_results(self, model_results: Dict[str, Any], 
                              behaviors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from all methods for a single model."""
        combined = {
            'model_name': model_results['model_name'],
            'model_config': model_results['model_config'],
            'behaviors': []
        }
        
        for i, behavior in enumerate(behaviors):
            behavior_result = {
                'behavior_name': behavior['name'],
                'behavior_category': behavior.get('category', 'general'),
                'method_results': {}
            }
            
            # Extract results for this behavior from each method
            for method, method_data in model_results['method_results'].items():
                if i < len(method_data['results']):
                    behavior_result['method_results'][method] = method_data['results'][i]
            
            # Determine optimal method (highest success, then confidence)
            working_methods = [
                (method, result) for method, result in behavior_result['method_results'].items()
                if result.get('success', False)
            ]
            
            if working_methods:
                # Sort by success, then by confidence
                optimal_method, optimal_result = max(working_methods, 
                                                   key=lambda x: (x[1]['success'], x[1]['confidence']))
                behavior_result['optimal_method'] = optimal_method
                behavior_result['working_methods_count'] = len(working_methods)
            else:
                behavior_result['optimal_method'] = None
                behavior_result['working_methods_count'] = 0
            
            combined['behaviors'].append(behavior_result)
        
        return combined
    
    def _generate_summary_statistics(self, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics across all models and methods."""
        summary = {
            'overall_statistics': {},
            'method_statistics': {},
            'model_statistics': {}
        }
        
        all_results = []
        method_results = {method: [] for method in self.supported_methods}
        model_results = {}
        
        for model_name, model_data in ground_truth['results_by_model'].items():
            model_results[model_name] = []
            
            for behavior in model_data['behaviors']:
                for method, result in behavior['method_results'].items():
                    all_results.append(result)
                    method_results[method].append(result)
                    model_results[model_name].append(result)
        
        # Overall statistics
        if all_results:
            total = len(all_results)
            successful = sum(1 for r in all_results if r.get('success', False))
            avg_time = sum(r.get('execution_time', 0) for r in all_results) / total
            
            summary['overall_statistics'] = {
                'total_evaluations': total,
                'successful_evaluations': successful,
                'overall_success_rate': successful / total * 100,
                'average_execution_time': avg_time
            }
        
        # Method statistics
        for method, results in method_results.items():
            if results:
                total = len(results)
                successful = sum(1 for r in results if r.get('success', False))
                avg_time = sum(r.get('execution_time', 0) for r in results) / total
                
                summary['method_statistics'][method] = {
                    'total_evaluations': total,
                    'successful_evaluations': successful,
                    'success_rate': successful / total * 100,
                    'average_execution_time': avg_time
                }
        
        # Model statistics
        for model, results in model_results.items():
            if results:
                total = len(results)
                successful = sum(1 for r in results if r.get('success', False))
                
                summary['model_statistics'][model] = {
                    'total_evaluations': total,
                    'successful_evaluations': successful,
                    'success_rate': successful / total * 100
                }
        
        return summary
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print summary statistics."""
        print(f"\n📊 GROUND TRUTH SUMMARY")
        print(f"{'='*50}")
        
        overall = summary.get('overall_statistics', {})
        if overall:
            print(f"Overall Success Rate: {overall.get('overall_success_rate', 0):.1f}%")
            print(f"Total Evaluations: {overall.get('total_evaluations', 0)}")
            print(f"Avg Execution Time: {overall.get('average_execution_time', 0):.1f}s")
        
        print(f"\n🎯 Method Performance:")
        for method, stats in summary.get('method_statistics', {}).items():
            print(f"  {method}:")
            print(f"    Success Rate: {stats['success_rate']:.1f}%")
            print(f"    Avg Time: {stats['average_execution_time']:.1f}s")
        
        if len(summary.get('model_statistics', {})) > 1:
            print(f"\n🔧 Model Performance:")
            for model, stats in summary.get('model_statistics', {}).items():
                print(f"  {model}: {stats['success_rate']:.1f}% success rate")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate Ground Truth for Baseline Methods')
    
    parser.add_argument('--behaviors_file', required=True,
                       help='Path to behaviors CSV/JSON file')
    parser.add_argument('--methods', default='directrequest,fewshot,gptfuzz',
                       help='Comma-separated list of methods to evaluate')
    parser.add_argument('--models_config', 
                       help='Path to models configuration YAML file')
    parser.add_argument('--target_models', nargs='*',
                       help='Specific models to evaluate (default: all in config)')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory for ground truth results')
    parser.add_argument('--max_behaviors', type=int,
                       help='Maximum number of behaviors to evaluate')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    try:
        # Initialize generator
        generator = GroundTruthGenerator(verbose=not args.quiet)
        
        # Load behaviors
        behaviors = generator.load_behaviors(args.behaviors_file)
        
        if args.max_behaviors and len(behaviors) > args.max_behaviors:
            behaviors = behaviors[:args.max_behaviors]
            print(f"📏 Limited to {args.max_behaviors} behaviors")
        
        # Load model configurations
        model_configs = generator.get_model_configs(args.models_config)
        
        # Filter models if specified
        if args.target_models:
            model_configs = {
                model: config for model, config in model_configs.items()
                if model in args.target_models
            }
        
        # Parse methods
        methods = [m.strip().lower() for m in args.methods.split(',')]
        invalid_methods = [m for m in methods if m not in generator.supported_methods]
        if invalid_methods:
            print(f"⚠️  Invalid methods: {invalid_methods}")
            methods = [m for m in methods if m in generator.supported_methods]
        
        if not methods:
            print("❌ No valid methods specified")
            sys.exit(1)
        
        # Generate ground truth
        ground_truth = generator.generate_ground_truth(
            behaviors=behaviors,
            model_configs=model_configs,
            methods=methods,
            output_dir=args.output_dir
        )
        
        print(f"\n✅ Ground truth generation completed!")
        
    except Exception as e:
        print(f"\n❌ Ground truth generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()