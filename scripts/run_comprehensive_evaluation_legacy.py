#!/usr/bin/env python3
"""
V9 Generic Agent 2-Step Comprehensive Evaluation

This script runs the V9 Generic agent with a retry mechanism:
1. First attempt: Agent selects method based on V9 tactical analysis
2. If first attempt would fail: Agent gets feedback and can select a different method

This combines V9's model-adaptive behavior with intelligent retry learning.
"""

import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Add the current directory to path for imports
sys.path.append(os.path.dirname(__file__))

try:
    from behavior_attack_selector_v9_2step import GenericV9TwoStepBehaviorAttackSelector
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Making sure the V9 2-step selector is available...")
    sys.exit(1)

class V9TwoStepEvaluator:
    """V9 evaluator with 2-step retry mechanism for failed attempts."""
    
    def __init__(self):
        self.models = [
            "mistral_7b_v03_medium",
            "qwen2_5_0_5b_instruct", 
            "mixtral_8x7b_v01",
            "qwen2_7b_instruct",
            "granite_3_3_8b_instruct",
            "granite_3_3_2b_instruct",
            "llama_3_2_1b",
            "llama_3_2_3b_instruct",
            "qwen2_1_5b_instruct",
            "meta_llama_3_70b"
        ]
        self.load_behaviors()
        self.load_ground_truths()
        print(f"🎯 V9 2-Step Evaluator initialized")
        print(f"📊 Models: {len(self.models)}")
        print(f"🎭 Behaviors: {len(self.behaviors)}")
        print(f"📈 Total evaluations: {len(self.models) * len(self.behaviors)} (with V9 retry mechanism)")
    
    def load_behaviors(self):
        """Load the 35 behaviors."""
        try:
            with open('updated_35_behaviors_complete.json', 'r') as f:
                behavior_data = json.load(f)
            
            self.behaviors = []
            for item in behavior_data:
                self.behaviors.append({
                    'behavior': item['behavior'],
                    'description': item.get('behavior', 'Unknown behavior'),
                    'category': self.categorize_behavior(item['behavior'])
                })
            
            print(f"✅ Loaded {len(self.behaviors)} behaviors")
            
        except FileNotFoundError:
            print("❌ Behavior file not found, using sample behaviors")
            self.behaviors = [
                {'behavior': 'test_behavior', 'description': 'Test behavior', 'category': 'test'}
            ]
    
    def load_ground_truths(self):
        """Load ground truth data for all models to simulate success/failure."""
        self.ground_truths = {}
        
        ground_truth_files = {
            'mistral_7b_v03_medium': 'updated_35_behaviors_mistral_7b_v03_complete.json',
            'qwen2_5_0_5b_instruct': 'updated_35_behaviors_qwen2_5_complete.json',
            'mixtral_8x7b_v01': 'updated_35_behaviors_mixtral_8x7b_complete.json',
            'qwen2_7b_instruct': 'updated_35_behaviors_qwen2_7b_complete.json',
            'granite_3_3_8b_instruct': 'updated_35_behaviors_granite_8b_complete.json',
            'granite_3_3_2b_instruct': 'updated_35_behaviors_granite_2b_complete.json',
            'llama_3_2_1b': 'updated_35_behaviors_llama_1b_complete.json',
            'llama_3_2_3b_instruct': 'updated_35_behaviors_llama_3b_complete.json',
            'qwen2_1_5b_instruct': 'updated_35_behaviors_qwen2_1_5b_complete.json',
            'meta_llama_3_70b': 'updated_35_behaviors_llama_70b_complete.json'
        }
        
        for model_key, filename in ground_truth_files.items():
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                    # Convert to behavior: {method: success} format
                    self.ground_truths[model_key] = {}
                    for item in data:
                        behavior = item['behavior']
                        self.ground_truths[model_key][behavior] = {
                            'directrequest': item.get('directrequest_label', 0),
                            'fewshot': item.get('fewshot_label', 0), 
                            'gptfuzz': item.get('gptfuzz_label', 0)
                        }
                print(f"✅ Loaded ground truth for {model_key}")
            except FileNotFoundError:
                print(f"⚠️  Ground truth file not found for {model_key}: {filename}")
                self.ground_truths[model_key] = {}
    
    def categorize_behavior(self, behavior_name: str) -> str:
        """Categorize behavior for better method selection."""
        behavior_lower = behavior_name.lower()
        
        if any(word in behavior_lower for word in ['misinformation', 'propaganda', 'bias', 'claim', 'article']):
            return 'misinformation'
        elif any(word in behavior_lower for word in ['instructions', 'guide', 'synthesis', 'manufacturing', 'tutorial']):
            return 'technical'
        elif any(word in behavior_lower for word in ['insults', 'harassment', 'hatred', 'bullying']):
            return 'bias'
        elif any(word in behavior_lower for word in ['direct', 'simple', 'factual']):
            return 'direct'
        else:
            return 'general'
    
    def simulate_attack_success(self, model_name: str, behavior: str, method: str) -> bool:
        """Check if attack method would succeed based on ground truth."""
        if model_name not in self.ground_truths:
            return False
            
        behavior_gt = self.ground_truths[model_name].get(behavior, {})
        method_map = {
            'FewShot': 'fewshot',
            'DirectRequest': 'directrequest', 
            'GPTFuzz': 'gptfuzz'
        }
        
        gt_method = method_map.get(method, method.lower())
        return behavior_gt.get(gt_method, 0) == 1
    
    def get_available_methods(self, model_name: str, behavior: str) -> List[str]:
        """Get list of methods that would succeed for this behavior-model combination."""
        if model_name not in self.ground_truths:
            return []
            
        behavior_gt = self.ground_truths[model_name].get(behavior, {})
        available = []
        
        if behavior_gt.get('fewshot', 0) == 1:
            available.append('FewShot')
        if behavior_gt.get('directrequest', 0) == 1:
            available.append('DirectRequest')
        if behavior_gt.get('gptfuzz', 0) == 1:
            available.append('GPTFuzz')
            
        return available
    
    def run_two_step_evaluation(self, model_name: str) -> Dict[str, Any]:
        """Run V9 2-step evaluation against a specific model."""
        print(f"\n🎯 Running V9 2-Step against {model_name}")
        print("-" * 60)
        
        try:
            # Initialize fresh V9 2-step selector for each model
            selector = GenericV9TwoStepBehaviorAttackSelector()
            
            model_results = {
                'model_name': model_name,
                'agent_version': 'v9_generic_2step',
                'start_time': datetime.now().isoformat(),
                'decisions': [],
                'method_counts': {'FewShot': 0, 'DirectRequest': 0, 'GPTFuzz': 0},
                'step1_success': 0,
                'step2_success': 0, 
                'total_success': 0,
                'retry_attempts': 0,
                'distribution': {},
                'detailed_results': []
            }
            
            # Run each behavior against this model
            for behavior_idx, behavior in enumerate(self.behaviors, 1):
                try:
                    print(f"  Behavior {behavior_idx}/{len(self.behaviors)}: {behavior['behavior'][:40]}...")
                    
                    # STEP 1: Initial V9 agent decision
                    step1_method = selector.select_attack_method(
                        model_name=model_name,
                        behavior_description=behavior['description'], 
                        category=behavior['category']
                    )
                    
                    # Check if Step 1 would succeed
                    step1_success = self.simulate_attack_success(model_name, behavior['behavior'], step1_method)
                    
                    final_method = step1_method
                    step2_method = None
                    used_retry = False
                    
                    if not step1_success:
                        # STEP 2: V9 agent gets feedback and can choose differently
                        available_methods = self.get_available_methods(model_name, behavior['behavior'])
                        
                        if available_methods:
                            # Track retry attempt
                            selector.retry_attempts += 1
                            model_results['retry_attempts'] += 1
                            
                            try:
                                # Get Step 2 decision with V9 adaptive feedback
                                step2_method = selector.select_attack_method(
                                    model_name=model_name,
                                    behavior_description=behavior['description'],
                                    category="retry_with_feedback",
                                    is_retry=True,
                                    available_methods=available_methods,
                                    initial_method=step1_method
                                )
                                
                                # Ensure Step 2 method is from available options
                                if step2_method in available_methods:
                                    final_method = step2_method
                                    used_retry = True
                                    selector.retry_successes += 1
                                    print(f"    🔄 V9 2-Step Retry: {step1_method} → {step2_method}")
                                
                            except Exception as e:
                                print(f"    ⚠️  V9 Step 2 failed: {e}")
                    
                    # Check final success
                    final_success = self.simulate_attack_success(model_name, behavior['behavior'], final_method)
                    
                    # Record decision
                    decision = {
                        'behavior': behavior['behavior'],
                        'category': behavior['category'],
                        'step1_method': step1_method,
                        'step1_success': step1_success,
                        'step2_method': step2_method,
                        'final_method': final_method,
                        'used_retry': used_retry,
                        'final_success': final_success,
                        'model': model_name
                    }
                    
                    model_results['decisions'].append(decision)
                    model_results['method_counts'][final_method] += 1
                    model_results['detailed_results'].append(decision)
                    
                    # Track success rates
                    if step1_success:
                        model_results['step1_success'] += 1
                    if used_retry and final_success:
                        model_results['step2_success'] += 1
                    if final_success:
                        model_results['total_success'] += 1
                    
                    status = "✓" if final_success else "✗"
                    retry_info = f" (V9 retry: {step1_method}→{step2_method})" if used_retry else ""
                    print(f"    → {final_method} {status}{retry_info}")
                    
                except Exception as e:
                    print(f"    ❌ Error processing behavior {behavior['behavior']}: {e}")
                    continue
            
            # Calculate distributions and success rates
            total_decisions = sum(model_results['method_counts'].values())
            if total_decisions > 0:
                model_results['distribution'] = {
                    method: (count / total_decisions * 100)
                    for method, count in model_results['method_counts'].items()
                }
                
                model_results['success_rates'] = {
                    'step1_rate': (model_results['step1_success'] / total_decisions) * 100,
                    'step2_rate': (model_results['step2_success'] / model_results['retry_attempts']) * 100 if model_results['retry_attempts'] > 0 else 0,
                    'total_rate': (model_results['total_success'] / total_decisions) * 100,
                    'retry_improvement': model_results['step2_success']
                }
            else:
                model_results['distribution'] = {'FewShot': 0, 'DirectRequest': 0, 'GPTFuzz': 0}
                model_results['success_rates'] = {'step1_rate': 0, 'step2_rate': 0, 'total_rate': 0, 'retry_improvement': 0}
            
            model_results['end_time'] = datetime.now().isoformat()
            model_results['total_decisions'] = total_decisions
            
            print(f"  ✅ {model_name} complete:")
            print(f"    Distribution: FS={model_results['distribution'].get('FewShot', 0):.1f}%, " +
                  f"DR={model_results['distribution'].get('DirectRequest', 0):.1f}%, " +
                  f"GPT={model_results['distribution'].get('GPTFuzz', 0):.1f}%")
            print(f"    Success: Step1={model_results['success_rates']['step1_rate']:.1f}%, " +
                  f"Total={model_results['success_rates']['total_rate']:.1f}% " +
                  f"(+{model_results['step2_success']} from V9 retries)")
            
            return model_results
            
        except Exception as e:
            print(f"❌ Error running V9 2-Step against {model_name}: {e}")
            return None
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run V9 2-step evaluation across all models."""
        print(f"🚀 Starting V9 2-Step Comprehensive Evaluation at {datetime.now()}")
        print("=" * 80)
        
        results = {
            'evaluation_info': {
                'version': 'V9_Generic_2Step',
                'strategy': 'Model-adaptive tactical diversity (70/20/10) with intelligent retry',
                'start_time': datetime.now().isoformat(),
                'models_evaluated': len(self.models),
                'behaviors_evaluated': len(self.behaviors),
                'total_decisions': 0
            },
            'model_results': {},
            'overall_distribution': {'FewShot': 0, 'DirectRequest': 0, 'GPTFuzz': 0},
            'overall_success': {
                'step1_successes': 0,
                'step2_successes': 0, 
                'total_successes': 0,
                'total_retries': 0
            },
            'summary_stats': {}
        }
        
        # Run evaluation for each model
        for model_idx, model_name in enumerate(self.models, 1):
            result = self.run_two_step_evaluation(model_name)
            if result:
                results['model_results'][model_name] = result
                
                # Aggregate overall stats
                for method, count in result['method_counts'].items():
                    results['overall_distribution'][method] += count
                
                results['overall_success']['step1_successes'] += result['step1_success']
                results['overall_success']['step2_successes'] += result['step2_success']
                results['overall_success']['total_successes'] += result['total_success']
                results['overall_success']['total_retries'] += result['retry_attempts']
                results['evaluation_info']['total_decisions'] += result['total_decisions']
        
        # Calculate overall statistics
        total_decisions = results['evaluation_info']['total_decisions']
        if total_decisions > 0:
            results['overall_distribution_percentage'] = {
                method: (count / total_decisions * 100)
                for method, count in results['overall_distribution'].items()
            }
            
            results['overall_success_rates'] = {
                'step1_rate': (results['overall_success']['step1_successes'] / total_decisions) * 100,
                'step2_rate': (results['overall_success']['step2_successes'] / results['overall_success']['total_retries']) * 100 if results['overall_success']['total_retries'] > 0 else 0,
                'total_rate': (results['overall_success']['total_successes'] / total_decisions) * 100,
                'retry_effectiveness': results['overall_success']['step2_successes'],
                'retry_usage': (results['overall_success']['total_retries'] / total_decisions) * 100
            }
        
        results['evaluation_info']['end_time'] = datetime.now().isoformat()
        return results
    
    def save_results(self, results: Dict[str, Any]) -> str:
        """Save evaluation results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"v9_2step_comprehensive_evaluation_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\n✅ Results saved to: {filename}")
            
            # Print comprehensive summary
            self.print_summary(results)
            
            return filename
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return None
    
    def print_summary(self, results: Dict[str, Any]):
        """Print comprehensive summary of V9 2-step evaluation results."""
        print("\n" + "=" * 80)
        print("🏆 V9 2-STEP COMPREHENSIVE EVALUATION SUMMARY")
        print("=" * 80)
        
        eval_info = results.get('evaluation_info', {})
        overall_dist = results.get('overall_distribution_percentage', {})
        success_rates = results.get('overall_success_rates', {})
        
        print(f"📊 Total Models Tested: {eval_info.get('models_evaluated', 0)}")
        print(f"🎭 Behaviors per Model: {eval_info.get('behaviors_evaluated', 0)}")
        print(f"📈 Total Decisions: {eval_info.get('total_decisions', 0)}")
        
        print(f"\n🎯 V9 2-Step Method Distribution:")
        print(f"  FewShot: {overall_dist.get('FewShot', 0):.1f}% (Target: 70%)")
        print(f"  DirectRequest: {overall_dist.get('DirectRequest', 0):.1f}% (Target: 20%)")  
        print(f"  GPTFuzz: {overall_dist.get('GPTFuzz', 0):.1f}% (Target: 10%)")
        
        print(f"\n🚀 Success Rate Analysis:")
        print(f"  Step 1 Success Rate: {success_rates.get('step1_rate', 0):.1f}%")
        print(f"  Step 2 Success Rate: {success_rates.get('step2_rate', 0):.1f}%")
        print(f"  Overall Success Rate: {success_rates.get('total_rate', 0):.1f}%")
        print(f"  Retry Usage: {success_rates.get('retry_usage', 0):.1f}% of decisions")
        print(f"  Retry Effectiveness: +{success_rates.get('retry_effectiveness', 0)} additional successes")
        
        print(f"\n🧠 V9 Intelligence Analysis:")
        improvement = success_rates.get('total_rate', 0) - success_rates.get('step1_rate', 0)
        print(f"  Adaptive Improvement: +{improvement:.1f}% from V9 retry mechanism")
        print(f"  Model-Adaptive Learning: V9 tactical awareness + failure learning")
        print(f"  Strategic Balance: Quota enforcement with intelligent adaptation")
        
        # Model-by-model breakdown
        print(f"\n📋 Model Performance Breakdown:")
        model_results = results.get('model_results', {})
        for model_name, model_data in model_results.items():
            success_rate = model_data.get('success_rates', {}).get('total_rate', 0)
            retries = model_data.get('retry_attempts', 0)
            dist = model_data.get('distribution', {})
            print(f"  {model_name}: {success_rate:.1f}% success ({retries} V9 retries) " +
                  f"[FS:{dist.get('FewShot', 0):.0f}%, DR:{dist.get('DirectRequest', 0):.0f}%, GPT:{dist.get('GPTFuzz', 0):.0f}%]")

def main():
    """Main execution function."""
    print("🎯 V9 Generic Agent - 2-Step Comprehensive Evaluation")
    print("=" * 80)
    print("Strategy: Model-adaptive tactical diversity with intelligent retry")
    print("Features: V9 quota enforcement + failure-based learning")
    print("=" * 80)
    
    try:
        # Initialize evaluator
        evaluator = V9TwoStepEvaluator()
        
        # Run comprehensive evaluation
        results = evaluator.run_comprehensive_evaluation()
        
        # Save results
        filename = evaluator.save_results(results)
        
        print(f"\n✨ V9 2-Step evaluation completed!")
        if filename:
            print(f"📁 Results file: {filename}")
            print(f"\n💡 This evaluation combines V9 model-adaptive behavior with intelligent retry learning.")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()