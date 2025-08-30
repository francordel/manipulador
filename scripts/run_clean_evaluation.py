#!/usr/bin/env python3
"""
Clean Evaluation Script for Manipulador OSS

This script provides a clean interface for running V9 REAL evaluations
with proper argument parsing and logging for publication-ready results.
"""

import os
import sys
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.run_v9_real_measured_evaluation import V9RealMeasuredEvaluator
from scripts.run_v9_real_2step_evaluation import V9RealEvaluator


def setup_logging(output_dir: str, log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    log_file = os.path.join(output_dir, f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger('manipulador_evaluation')
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def run_measured_evaluation(args, logger):
    """Run V9 REAL evaluation with measured ASR."""
    logger.info("Starting V9 REAL Measured ASR Evaluation")
    logger.info(f"Configuration: {vars(args)}")
    
    evaluator = V9RealMeasuredEvaluator()
    
    if args.single_model:
        # Single model evaluation
        logger.info(f"Running single model evaluation: {args.single_model}")
        result = evaluator.run_measured_evaluation(args.single_model)
        
        # Save results
        output_file = os.path.join(args.output_dir, f"{args.single_model}_measured_evaluation.json")
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Results saved to: {output_file}")
        return result
    
    else:
        # Full evaluation across all models
        logger.info("Running full evaluation across all models")
        results = evaluator.run_full_evaluation()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(args.output_dir, f"v9_real_measured_evaluation_{timestamp}.json")
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Full evaluation completed. Results saved to: {output_file}")
        return results


def run_2step_evaluation(args, logger):
    """Run V9 REAL 2-step evaluation."""
    logger.info("Starting V9 REAL 2-Step Evaluation")
    logger.info(f"Configuration: {vars(args)}")
    
    evaluator = V9RealEvaluator()
    
    if args.single_model:
        # Single model evaluation
        logger.info(f"Running single model evaluation: {args.single_model}")
        result = evaluator.run_legitimate_evaluation(args.single_model)
        
        # Save results
        output_file = os.path.join(args.output_dir, f"{args.single_model}_2step_evaluation.json")
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Results saved to: {output_file}")
        return result
    
    else:
        # Full evaluation across all models
        logger.info("Running full 2-step evaluation across all models")
        results = evaluator.run_full_evaluation()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(args.output_dir, f"v9_real_2step_evaluation_{timestamp}.json")
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Full 2-step evaluation completed. Results saved to: {output_file}")
        return results


def analyze_and_report(results, args, logger):
    """Generate analysis and performance report."""
    logger.info("Generating analysis report")
    
    try:
        if 'model_results' in results:
            # Full evaluation results
            model_results = results['model_results']
            
            print("\n" + "="*70)
            print("📊 EVALUATION SUMMARY")
            print("="*70)
            
            print(f"Evaluation Type: {results['evaluation_info']['version']}")
            print(f"Total Models: {len(model_results)}")
            print(f"Duration: {results['evaluation_info'].get('duration', 'Unknown')}")
            
            if 'overall_metrics' in results:
                metrics = results['overall_metrics']
                print(f"\n🎯 Overall Performance:")
                
                if 'macro_averages' in metrics:
                    macro = metrics['macro_averages']
                    print(f"  Step-1 ASR: {macro.get('step1_asr', 0):.1f}%")
                    print(f"  Final ASR: {macro.get('final_asr', 0):.1f}%")
                    print(f"  Retry Rate: {macro.get('retry_rate', 0):.1f}%")
                    print(f"  Retry Precision: {macro.get('retry_precision', 0):.1f}%")
                
                if 'method_distribution' in metrics:
                    dist = metrics['method_distribution']
                    print(f"\n📈 Method Distribution:")
                    print(f"  FewShot: {dist.get('FewShot', 0):.1f}%")
                    print(f"  DirectRequest: {dist.get('DirectRequest', 0):.1f}%") 
                    print(f"  GPTFuzz: {dist.get('GPTFuzz', 0):.1f}%")
            
            # Top performing models
            if len(model_results) > 1:
                print(f"\n🏆 Top Performing Models:")
                sorted_models = sorted(
                    [(m['model_name'], m.get('metrics', {}).get('final_asr', 0)) 
                     for m in model_results],
                    key=lambda x: x[1], reverse=True
                )
                for i, (model, asr) in enumerate(sorted_models[:5]):
                    print(f"  {i+1}. {model}: {asr:.1f}% ASR")
        
        else:
            # Single model result
            print(f"\n📊 Single Model Results: {results.get('model_name', 'Unknown')}")
            if 'metrics' in results:
                metrics = results['metrics']
                print(f"  Step-1 ASR: {metrics.get('step1_asr', 0):.1f}%")
                print(f"  Final ASR: {metrics.get('final_asr', 0):.1f}%")
                print(f"  Retry Rate: {metrics.get('retry_rate', 0):.1f}%")
        
        print("\n" + "="*70)
        
    except Exception as e:
        logger.error(f"Error generating analysis report: {e}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Manipulador V9 REAL Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full measured evaluation across all models
  python run_clean_evaluation.py --evaluation-type measured
  
  # Single model evaluation
  python run_clean_evaluation.py --evaluation-type measured --single-model mistral_7b_v03_medium
  
  # 2-step evaluation with custom output
  python run_clean_evaluation.py --evaluation-type 2step --output-dir ./my_results
        """
    )
    
    parser.add_argument(
        '--evaluation-type',
        choices=['measured', '2step'],
        default='measured',
        help='Type of evaluation to run (default: measured)'
    )
    
    parser.add_argument(
        '--single-model',
        type=str,
        help='Run evaluation on single model (e.g., mistral_7b_v03_medium)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='Output directory for results (default: ./results)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--no-analysis',
        action='store_true',
        help='Skip analysis report generation'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir, args.log_level)
    
    try:
        logger.info("="*50)
        logger.info("MANIPULADOR V9 REAL EVALUATION")
        logger.info("="*50)
        
        # Run evaluation based on type
        if args.evaluation_type == 'measured':
            results = run_measured_evaluation(args, logger)
        elif args.evaluation_type == '2step':
            results = run_2step_evaluation(args, logger)
        else:
            raise ValueError(f"Unknown evaluation type: {args.evaluation_type}")
        
        # Generate analysis report
        if not args.no_analysis:
            analyze_and_report(results, args, logger)
        
        logger.info("Evaluation completed successfully!")
        print(f"\n✅ Evaluation completed! Check {args.output_dir} for results.")
        
        return results
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        print("\n⚠️ Evaluation interrupted by user")
        return None
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"\n❌ Evaluation failed: {e}")
        print("Check the log file for detailed error information")
        raise


if __name__ == "__main__":
    main()