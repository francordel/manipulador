#!/usr/bin/env python3
"""
Comprehensive evaluation script for Manipulador agents.

This script runs full evaluation of V9 agents across multiple models and behaviors,
supporting both single-step and two-step (retry) evaluation modes.
"""

import sys
import os
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any
import csv
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.behavior_attack_selector_v9_generic import GenericV9BehaviorAttackSelector
from agents.behavior_attack_selector_v9_2step import BehaviorAttackSelectorV9TwoStep

class ComprehensiveEvaluator:
    """Comprehensive evaluation framework for Manipulador agents."""
    
    def __init__(self, agent_type: str = "v9_2step", verbose: bool = True):
        """Initialize evaluator with specified agent type."""
        self.agent_type = agent_type
        self.verbose = verbose
        
        if agent_type == "v9_generic":
            self.agent = GenericV9BehaviorAttackSelector()
        elif agent_type == "v9_2step":
            self.agent = BehaviorAttackSelectorV9TwoStep()
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")
            
        self.results = []
        self.start_time = None
        
    def load_behaviors(self, behaviors_file: str) -> List[Dict[str, Any]]:
        """Load behaviors from CSV or JSON file."""
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
                        'category': row.get('category', 'general'),
                        'tags': row.get('tags', '').split(',') if row.get('tags') else []
                    })
        elif file_path.suffix.lower() == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    behaviors = data
                else:
                    behaviors = [data]
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        if self.verbose:
            print(f"✅ Loaded {len(behaviors)} behaviors from {behaviors_file}")
            
        return behaviors
    
    def get_target_models(self, models_filter: List[str] = None) -> List[str]:
        """Get list of target models to evaluate."""
        available_models = list(self.agent.model_characteristics.keys())
        
        if models_filter:
            # Filter to only requested models
            filtered_models = [m for m in models_filter if m in available_models]
            if not filtered_models:
                print(f"⚠️  None of the requested models found: {models_filter}")
                print(f"   Available models: {available_models}")
                return available_models[:3]  # Return first 3 as fallback
            return filtered_models
        
        return available_models
    
    def evaluate_behavior(self, behavior: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """Evaluate single behavior against target model."""
        model_data = self.agent.model_characteristics.get(model_name, {})
        
        if not model_data:
            raise ValueError(f"Model {model_name} not found in agent characteristics")
        
        start_time = time.time()
        
        try:
            result = self.agent.analyze_behavior_generic(
                behavior_name=behavior['name'],
                behavior_data={
                    'behavior': behavior['behavior'], 
                    'category': behavior.get('category', 'general')
                },
                model_name=model_name,
                model_data=model_data
            )
            
            execution_time = time.time() - start_time
            
            # Add metadata
            result.update({
                'behavior_name': behavior['name'],
                'behavior_category': behavior.get('category', 'general'),
                'target_model': model_name,
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat(),
                'agent_type': self.agent_type
            })
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                'behavior_name': behavior['name'],
                'behavior_category': behavior.get('category', 'general'),
                'target_model': model_name,
                'selected_method': 'ERROR',
                'success': False,
                'confidence': 0.0,
                'reasoning': f"Evaluation error: {str(e)}",
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat(),
                'agent_type': self.agent_type,
                'error': str(e)
            }
    
    def run_evaluation(self, behaviors: List[Dict[str, Any]], 
                      target_models: List[str],
                      max_behaviors: int = None,
                      save_intermediate: bool = True,
                      output_dir: str = None) -> Dict[str, Any]:
        """Run comprehensive evaluation."""
        
        self.start_time = time.time()
        
        # Apply behavior limit if specified
        if max_behaviors and len(behaviors) > max_behaviors:
            behaviors = behaviors[:max_behaviors]
            if self.verbose:
                print(f"📏 Limited to {max_behaviors} behaviors")
        
        total_evaluations = len(behaviors) * len(target_models)
        current_eval = 0
        
        if self.verbose:
            print(f"\n🚀 STARTING COMPREHENSIVE EVALUATION")
            print(f"{'='*60}")
            print(f"Agent Type: {self.agent_type}")
            print(f"Behaviors: {len(behaviors)}")
            print(f"Target Models: {len(target_models)}")
            print(f"Total Evaluations: {total_evaluations}")
            print(f"{'='*60}\n")
        
        self.results = []
        
        for model_idx, model_name in enumerate(target_models):
            if self.verbose:
                print(f"📊 [{model_idx+1}/{len(target_models)}] Evaluating model: {model_name}")
                model_data = self.agent.model_characteristics.get(model_name, {})
                print(f"    Safety Level: {model_data.get('safety_level', 'unknown')}")
                print(f"    Size Category: {model_data.get('size_category', 'unknown')}")
            
            model_results = []
            
            for behavior_idx, behavior in enumerate(behaviors):
                current_eval += 1
                
                if self.verbose:
                    progress = f"[{current_eval}/{total_evaluations}]"
                    print(f"    {progress} {behavior['name'][:50]}")
                
                result = self.evaluate_behavior(behavior, model_name)
                model_results.append(result)
                self.results.append(result)
                
                if self.verbose:
                    method = result.get('selected_method', 'ERROR')
                    confidence = result.get('confidence', 0.0)
                    success_indicator = "✅" if result.get('success', False) else "❌"
                    print(f"        → {method} (conf: {confidence:.2f}) {success_indicator}")
                
                # Save intermediate results
                if save_intermediate and output_dir and current_eval % 10 == 0:
                    self._save_intermediate_results(output_dir, current_eval, total_evaluations)
            
            if self.verbose:
                successful = sum(1 for r in model_results if r.get('success', False))
                print(f"    Model Results: {successful}/{len(behaviors)} successful ({successful/len(behaviors)*100:.1f}%)")
                print()
        
        # Generate final summary
        summary = self._generate_summary()
        
        if self.verbose:
            self._print_summary(summary)
        
        return {
            'evaluation_metadata': {
                'agent_type': self.agent_type,
                'total_evaluations': total_evaluations,
                'behaviors_count': len(behaviors),
                'models_count': len(target_models),
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_runtime': time.time() - self.start_time
            },
            'performance_summary': summary,
            'detailed_results': self.results
        }
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate evaluation summary statistics."""
        if not self.results:
            return {}
        
        total_evals = len(self.results)
        successful_evals = sum(1 for r in self.results if r.get('success', False))
        
        # Method distribution
        method_counts = {}
        method_success = {}
        
        for result in self.results:
            method = result.get('selected_method', 'UNKNOWN')
            method_counts[method] = method_counts.get(method, 0) + 1
            
            if method not in method_success:
                method_success[method] = {'chosen': 0, 'successful': 0}
            method_success[method]['chosen'] += 1
            if result.get('success', False):
                method_success[method]['successful'] += 1
        
        # Calculate method success rates
        for method in method_success:
            chosen = method_success[method]['chosen']
            successful = method_success[method]['successful']
            method_success[method]['success_rate'] = successful / chosen * 100 if chosen > 0 else 0.0
        
        # Model performance
        model_performance = {}
        for result in self.results:
            model = result.get('target_model', 'unknown')
            if model not in model_performance:
                model_performance[model] = {'total': 0, 'successful': 0, 'execution_times': []}
            
            model_performance[model]['total'] += 1
            if result.get('success', False):
                model_performance[model]['successful'] += 1
            model_performance[model]['execution_times'].append(result.get('execution_time', 0.0))
        
        # Calculate model stats
        for model in model_performance:
            stats = model_performance[model]
            stats['success_rate'] = stats['successful'] / stats['total'] * 100
            stats['avg_execution_time'] = sum(stats['execution_times']) / len(stats['execution_times'])
            del stats['execution_times']  # Remove raw times to keep summary clean
        
        return {
            'overall_success_rate': successful_evals / total_evals * 100,
            'total_evaluations': total_evals,
            'successful_evaluations': successful_evals,
            'method_effectiveness': method_success,
            'model_performance': model_performance,
            'agent_distribution_stats': getattr(self.agent, '_get_current_distribution', lambda: {})()
        }
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print evaluation summary."""
        print(f"\n{'='*80}")
        print(f"🎯 EVALUATION SUMMARY")
        print(f"{'='*80}")
        
        print(f"Overall Success Rate: {summary['overall_success_rate']:.1f}%")
        print(f"Total Evaluations: {summary['total_evaluations']}")
        print(f"Runtime: {time.time() - self.start_time:.1f} seconds")
        print()
        
        print(f"📊 METHOD EFFECTIVENESS:")
        for method, stats in summary['method_effectiveness'].items():
            if method != 'ERROR':
                print(f"  {method}:")
                print(f"    Chosen: {stats['chosen']} times ({stats['chosen']/summary['total_evaluations']*100:.1f}%)")
                print(f"    Success Rate: {stats['success_rate']:.1f}%")
        print()
        
        if len(summary['model_performance']) > 1:
            print(f"🎯 MODEL PERFORMANCE:")
            for model, stats in summary['model_performance'].items():
                print(f"  {model}:")
                print(f"    Success Rate: {stats['success_rate']:.1f}%")
                print(f"    Avg Execution Time: {stats['avg_execution_time']:.2f}s")
            print()
    
    def _save_intermediate_results(self, output_dir: str, current: int, total: int):
        """Save intermediate results during evaluation."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        intermediate_file = output_path / f"intermediate_results_{current}_{total}.json"
        with open(intermediate_file, 'w') as f:
            json.dump({
                'progress': f"{current}/{total}",
                'completion_percentage': current / total * 100,
                'current_results': self.results,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
    
    def save_results(self, output_dir: str, results: Dict[str, Any]):
        """Save final evaluation results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        results_file = output_path / f"{self.agent_type}_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save CSV for easy analysis
        csv_file = output_path / f"{self.agent_type}_evaluation_results.csv"
        with open(csv_file, 'w', newline='') as f:
            if results['detailed_results']:
                writer = csv.DictWriter(f, fieldnames=results['detailed_results'][0].keys())
                writer.writeheader()
                writer.writerows(results['detailed_results'])
        
        # Save summary only
        summary_file = output_path / f"{self.agent_type}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'metadata': results['evaluation_metadata'],
                'summary': results['performance_summary']
            }, f, indent=2)
        
        if self.verbose:
            print(f"💾 Results saved to: {output_path}")
            print(f"   • Full results: {results_file.name}")
            print(f"   • CSV format: {csv_file.name}")
            print(f"   • Summary: {summary_file.name}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Comprehensive Manipulador Agent Evaluation')
    
    parser.add_argument('--agent_type', choices=['v9_generic', 'v9_2step'], 
                       default='v9_2step', help='Agent type to evaluate')
    parser.add_argument('--behaviors_file', required=True,
                       help='Path to behaviors CSV/JSON file')
    parser.add_argument('--target_models', nargs='*',
                       help='Specific models to evaluate (default: all available)')
    parser.add_argument('--max_behaviors', type=int,
                       help='Maximum number of behaviors to evaluate')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory for results')
    parser.add_argument('--save_intermediate', action='store_true',
                       help='Save intermediate results during evaluation')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    try:
        # Initialize evaluator
        evaluator = ComprehensiveEvaluator(
            agent_type=args.agent_type,
            verbose=not args.quiet
        )
        
        # Load behaviors
        behaviors = evaluator.load_behaviors(args.behaviors_file)
        
        # Get target models
        target_models = evaluator.get_target_models(args.target_models)
        
        # Run evaluation
        results = evaluator.run_evaluation(
            behaviors=behaviors,
            target_models=target_models,
            max_behaviors=args.max_behaviors,
            save_intermediate=args.save_intermediate,
            output_dir=args.output_dir
        )
        
        # Save results
        evaluator.save_results(args.output_dir, results)
        
        print(f"\n✅ Evaluation completed successfully!")
        print(f"📊 Success Rate: {results['performance_summary']['overall_success_rate']:.1f}%")
        print(f"⏱️  Total Runtime: {results['evaluation_metadata']['total_runtime']:.1f}s")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()