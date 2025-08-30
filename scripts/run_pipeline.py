import argparse
import yaml
import os
import csv
import subprocess
import re
import torch
import signal
import ray
from tqdm import tqdm
import sys
sys.path.extend(['./baselines', '../baselines'])
from model_utils import _init_ray

# (non-exhaustive) methods list: AutoDAN,AutoPrompt,DirectRequest,GCG-Multi,GCG-Transfer,FewShot,GBDA,GCG,HumanJailbreaks,PAIR,PAP-top5,PEZ,TAP,TAP-Transfer,UAT,ZeroShot
# (non-exhaustive) models list: llama2_7b,llama2_13b,llama2_70b,vicuna_7b_v1_5,vicuna_13b_v1_5,koala_7b,koala_13b,orca_2_7b,orca_2_13b,solar_10_7b_instruct,openchat_3_5_1210,starling_7b,mistral_7b_v2,mixtral_8x7b,zephyr_7b_robust6_data2,zephyr_7b,baichuan2_7b,baichuan2_13b,qwen_7b_chat,qwen_14b_chat,qwen_72b_chat,gpt-3.5-turbo-1106,gpt-3.5-turbo-0613,gpt-4-1106-preview,gpt-4-0613,gpt-4-vision-preview,claude-instant-1,claude-2.1,claude-2,gemini,mistral-medium,LLaVA,InstructBLIP,Qwen,GPT4V

def detect_gpu_info():
    """Detect GPU information and return recommendations."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,compute_cap,memory.total', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpus = []
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip():
                    name, compute_cap, memory = [x.strip() for x in line.split(',')]
                    gpus.append({
                        'name': name,
                        'compute_cap': float(compute_cap),
                        'memory': memory
                    })
            return gpus
    except:
        pass
    return []

def get_dtype_recommendation(gpus):
    """Get dtype recommendation based on GPU capabilities."""
    if not gpus:
        return "float16", "No GPU detected, float16 is safest"
    
    min_compute_cap = min(gpu['compute_cap'] for gpu in gpus)
    gpu_names = [gpu['name'] for gpu in gpus]
    
    if min_compute_cap >= 8.0:
        return "bfloat16", f"Your GPU(s) support bfloat16 (compute capability {min_compute_cap})"
    else:
        return "float16", f"Your GPU(s) require float16 (compute capability {min_compute_cap} < 8.0 needed for bfloat16)"

def interactive_dtype_selection(mutation_model):
    """Interactive dtype selection for GPTFuzz."""
    print("\n" + "="*60)
    print("🔧 GPTFuzz Model Configuration")
    print("="*60)
    
    # Detect GPU info
    gpus = detect_gpu_info()
    recommended_dtype, reason = get_dtype_recommendation(gpus)
    
    if gpus:
        print(f"\n📊 GPU Information:")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu['name']}")
            print(f"    Compute Capability: {gpu['compute_cap']}")
            print(f"    Memory: {gpu['memory']}")
    else:
        print("\n⚠️  Could not detect GPU information")
    
    print(f"\n🤖 Mutation Model: {mutation_model}")
    print(f"💡 Recommended dtype: {recommended_dtype}")
    print(f"📝 Reason: {reason}")
    
    print(f"\n🎯 Available dtype options:")
    print(f"  1. float16/half  - Best for V100, RTX 20xx series (safe choice)")
    print(f"  2. bfloat16      - Best for RTX 30xx/40xx, H100, A100 (faster)")
    print(f"  3. float32       - Highest precision (slower)")
    print(f"  4. auto          - Let the system decide")
    
    while True:
        choice = input(f"\nSelect dtype (1-4) or press Enter for recommended [{recommended_dtype}]: ").strip()
        
        if choice == "" or choice == "0":
            return recommended_dtype
        elif choice == "1":
            return "float16"
        elif choice == "2":
            if recommended_dtype == "float16":
                confirm = input("⚠️  Your GPU may not support bfloat16. Continue anyway? (y/N): ").strip().lower()
                if confirm not in ['y', 'yes']:
                    continue
            return "bfloat16"
        elif choice == "3":
            return "float32"
        elif choice == "4":
            return "auto"
        else:
            print("❌ Invalid choice. Please select 1-4 or press Enter for recommended.")

def parse_args():
    """
    Parse all command-line arguments for controlling the pipeline execution.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed arguments containing all pipeline options, such as paths, methods, models,
            pipeline steps, execution mode, and step-specific flags (e.g., overwrite, verbose).
    """
    parser = argparse.ArgumentParser(description="Running red teaming with baseline methods.")
    # Arguments for all steps
    parser.add_argument("--base_save_dir", type=str, default='./results',
                    help="The base directory for saving test cases, completions, and results for a specific set of behaviors")
    parser.add_argument("--base_log_dir", type=str, default='./slurm_logs',
                    help="The base directory for saving slurm logs")
    parser.add_argument("--pipeline_config_path", type=str, default='./configs/pipeline_configs/run_pipeline.yaml',
                    help="The path to the pipeline config file used to schedule/run jobs")
    parser.add_argument("--models_config_path", type=str, default='./configs/model_configs/models.yaml',
                        help="The path to the config file with model hyperparameters")
    parser.add_argument("--methods", type=str, default='GCG',
                        help="A list of red teaming method names (choices are in run_pipeline.yaml)")
    parser.add_argument("--models", type=str, default='llama2_7b',
                        help="A list of target models (from models.yaml)")
    parser.add_argument("--behaviors_path", type=str, default='./data/behavior_datasets/harmbench_behaviors_text_all.csv',
                        help="The path to the behaviors file")
    parser.add_argument("--step", type=str, default="1",
                        help="The step of the pipeline to run",
                        choices=["1", "1.5", "2", "3", "2_and_3", "all"])
    parser.add_argument("--mode", type=str, default="slurm",
                        help="Whether to run the pipeline in 'local', 'slurm', or 'local_parallel' mode",
                        choices=["local", "slurm", "local_parallel"])
    parser.add_argument("--partition", type=str, default="cais",
                        help="The partition to use for submitting slurm jobs")
    # Arguments for step 1
    parser.add_argument("--behavior_ids_subset", type=str, default='',
                        help="An optional comma-separated list of behavior IDs, or a path to a newline-separated list of behavior IDs. If provided, test cases will only be generated for the specified behaviors. (only used in step 1 of the pipeline)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Whether to overwrite existing saved test cases (only used in step 1 of the pipeline)")
    parser.add_argument("--verbose", action="store_true",
                        help="Whether to print out intermediate results while generating test cases (only used in step 1 of the pipeline)")
    # Arguments for step 2
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="The maximum number of new tokens to generate for each completion (only used in step 2 of the pipeline)")
    parser.add_argument("--incremental_update", action="store_true",
                        help="Whether to incrementally update completions or generate a new completions file from scratch  (only used in step 2 of the pipeline)")
    # Arguments for step 3
    parser.add_argument("--cls_path", type=str, default="cais/HarmBench-Llama-2-13b-cls",
                        help="The path to the classifier used for evaluating completions (only used in step 3 of the pipeline)")
    parser.add_argument("--generate_report", action="store_true",
                        help="Generate comprehensive evaluation report after pipeline completion")
    parser.add_argument("--report_formats", nargs='+', default=['json', 'html'],
                        help="Output formats for the evaluation report")
    parser.add_argument("--include_advbench_metric", action="store_true",
                        help="Whether to include the AdvBench refusal metric in evaluation (step 3 only)")
    
    # GPTFuzz specific arguments
    parser.add_argument("--gptfuzz_mutation_model", type=str, default="gpt-3.5-turbo",
                        help="The mutation model to use for GPTFuzz attack")
    parser.add_argument("--gptfuzz_use_local_model", action="store_true",
                        help="Whether to use a local model for GPTFuzz mutation instead of OpenAI")
    parser.add_argument("--gptfuzz_openai_key", type=str, default=None,
                        help="OpenAI API key for GPTFuzz mutation model")
    parser.add_argument("--gptfuzz_dtype", type=str, default="auto", choices=["auto", "float16", "half", "bfloat16"],
                        help="Data type for GPTFuzz local model (auto=detect GPU capability, half=float16 for V100)")
    args = parser.parse_args()
    print("args ", args)
    return args


def get_expanded_pipeline_configs(pipeline_config, model_configs, proper_method_names, models):
    """
    Expand the pipeline configuration to create method-specific experiment mappings for all selected models.

    Args:
        pipeline_config (dict): The pipeline configuration dictionary loaded from YAML, keyed by method name.
        model_configs (dict): The model configuration dictionary loaded from YAML, keyed by model name.
        proper_method_names (list): List of selected method names (keys in pipeline_config) to expand.
        models (list): List of selected model names (keys in model_configs) to include in the expansion.

    Returns:
        dict: Dictionary of expanded pipeline configs for the specified methods and models,
              where each method config contains experiment definitions and mappings for job scheduling and analysis.
    """
    expanded_pipeline_configs = {}
    for proper_method_name in proper_method_names:
        expanded_pipeline_config = pipeline_config[proper_method_name]
        experiment_name_template = expanded_pipeline_config['experiment_name_template']
        expanded_pipeline_config['experiments'] = []
        expanded_pipeline_config['model_to_experiment'] = {}
        if '<model_name>' in experiment_name_template:
            for model in models:
                if model_configs[model]['model_type'] not in expanded_pipeline_config['allowed_target_model_types']:
                    continue  # Skip models that are not allowed for this method
                experiment_name = experiment_name_template.replace('<model_name>', model)
                expanded_pipeline_config['experiments'].append({
                    'experiment_name': experiment_name,
                    'expected_target_models': [model],
                    'experiment_num_gpus': model_configs[model].get('num_gpus', 0),  # closed-source model configs don't have the num_gpus field, so we default to 0
                    'behavior_chunk_size': expanded_pipeline_config.get('behavior_chunk_size', 5)
                    })
                expanded_pipeline_config['model_to_experiment'][model] = experiment_name
        else:
            experiment_name = experiment_name_template
            expanded_pipeline_config['experiments'].append({
                'experiment_name': experiment_name,
                'expected_target_models': models,
                'experiment_num_gpus': 0,
                'behavior_chunk_size': expanded_pipeline_config.get('behavior_chunk_size', 5)
                })
            for model in models:
                if model_configs[model]['model_type'] not in expanded_pipeline_config['allowed_target_model_types']:
                    continue  # Skip models that are not allowed for this method
                expanded_pipeline_config['model_to_experiment'][model] = experiment_name

        expanded_pipeline_configs[proper_method_name] = expanded_pipeline_config
    
    return expanded_pipeline_configs


def run_subprocess_slurm(command):
    """
    Execute a SLURM job using the provided command and retrieve the resulting job ID.

    Args:
        command (str): The shell command to submit a SLURM job, typically via sbatch.

    Returns:
        str or None: The SLURM job ID if successfully parsed from output, otherwise None.
    """
    # Execute the sbatch command
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Print the output and error, if any
    print("command:", command)
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)

    # Get the job ID for linking dependencies
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if match:
        job_id = match.group(1)
    else:
        job_id = None

    return job_id

@ray.remote
def run_subprocess_ray(command):
    """
    Execute a command as a Ray remote subprocess, using available GPU resources.

    Args:
        command (str): The shell command to execute in a subprocess, typically for parallel job execution.

    Returns:
        None
    """
    # Set the visible gpus for each command where resources={'visible_gpus': [(id, num_gpus)], ....}
    resources = ray.get_runtime_context().worker.core_worker.resource_ids()['visible_gpus']
    gpu_ids = [str(r[0]) for r in resources]

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_ids)

    print(f"gpus_ids={gpu_ids} | command={command}")
    process = subprocess.Popen(command, shell=True, env=env, preexec_fn=os.setsid)
    signal.signal(signal.SIGTERM, lambda signum, frame: os.killpg(os.getpgid(process.pid), signal.SIGTERM))
    signal.signal(signal.SIGINT, lambda signum, frame: os.killpg(os.getpgid(process.pid), signal.SIGTERM))
    process.wait()

def run_step1_single_job(
    mode, partition, job_name, num_gpus, output_log, method_name, experiment_name, behaviors_path, save_dir,
    start_idx, end_idx, behavior_ids_subset, run_id, overwrite, verbose, gptfuzz_args=None
):
    """
    Submit or execute a single job for generating test cases (step 1) based on the provided configuration.

    Args:
        mode (str): Execution mode ("slurm", "local", or "local_parallel").
        partition (str): SLURM partition name for job submission.
        job_name (str): Name for the job (for SLURM or logging).
        num_gpus (int): Number of GPUs to allocate for the job.
        output_log (str): Path for the output log file.
        method_name (str): Name of the red teaming method.
        experiment_name (str): Name of the experiment or experiment group.
        behaviors_path (str): Path to the behaviors dataset file.
        save_dir (str): Directory to save generated test cases.
        start_idx (str or int): Start index for behavior chunking.
        end_idx (str or int): End index for behavior chunking.
        behavior_ids_subset (str): Comma-separated or file path containing behavior IDs to restrict generation.
        run_id (str): Optional identifier for the run (for repeated runs).
        overwrite (bool): Whether to overwrite existing test cases.
        verbose (bool): Whether to print intermediate results during generation.

    Returns:
        str or Ray ObjectRef or None: SLURM job ID, Ray future, or None depending on mode.
    """
    if mode == "slurm":
        overwrite_string = "True" if overwrite else "False"
        verbose_string = "True" if verbose else "False"
        behavior_ids_subset = '""' if not behavior_ids_subset else behavior_ids_subset
        run_id_string = '""' if not run_id else run_id
        # Construct the sbatch command with all parameters and script arguments
        command = f'''sbatch --partition={partition} --job-name={job_name} --nodes=1 --gpus-per-node={num_gpus} --time=24:00:00 --output={output_log} \
            scripts/generate_test_cases.sh {method_name} {experiment_name} {behaviors_path} {save_dir} {start_idx} {end_idx} {behavior_ids_subset} {run_id_string} {overwrite_string} {verbose_string}'''
        # Run the sbatch command and get the job ID
        job_id = run_subprocess_slurm(command)
        return job_id
    
    elif mode in ["local", "local_parallel"]:
        # Construct the local command with all parameters and script arguments (with -u in case the output is piped to another file)
        options = [f"--{opt}" for opt, val in zip(['overwrite', 'verbose'], [overwrite, verbose]) if val]
        command = f"python -u generate_test_cases.py --method_name={method_name} --experiment_name={experiment_name} --behaviors_path={behaviors_path} \
            --save_dir={save_dir} --behavior_start_idx={start_idx} --behavior_end_idx={end_idx} --behavior_ids_subset={behavior_ids_subset} --run_id={run_id} " + ' '.join(options)
        
        # Add GPTFuzz specific arguments if provided
        if gptfuzz_args:
            if gptfuzz_args.get('mutation_model'):
                command += f" --gptfuzz_mutation_model={gptfuzz_args['mutation_model']}"
            if gptfuzz_args.get('use_local_model'):
                command += " --gptfuzz_use_local_model"
            if gptfuzz_args.get('openai_key'):
                command += f" --gptfuzz_openai_key={gptfuzz_args['openai_key']}"
            if gptfuzz_args.get('dtype'):
                command += f" --gptfuzz_dtype={gptfuzz_args['dtype']}"
        # Execute the local command
        if mode == "local":
            subprocess.run(command, shell=True, capture_output=False, text=True)
        # Return command to be executed parallel
        elif mode == "local_parallel":
            future = run_subprocess_ray.options(resources={"visible_gpus": num_gpus}).remote(command)
            return future

def run_step1_5_single_job(mode, partition, job_name, output_log, method_name, save_dir, dependency_job_ids=None):
    """
    Submit or execute a single job to merge all generated test case shards into a single file (step 1.5).

    Args:
        mode (str): Execution mode ("slurm", "local", or "local_parallel").
        partition (str): SLURM partition name for job submission.
        job_name (str): Name for the job (for SLURM or logging).
        output_log (str): Path for the output log file.
        method_name (str): Name of the red teaming method.
        save_dir (str): Directory containing test case shards to merge.
        dependency_job_ids (list or None): List of SLURM job IDs that must complete before this job (SLURM only).

    Returns:
        str or None: SLURM job ID if submitted, otherwise None.
    """
    if mode == "slurm":
        # Construct the sbatch command with all parameters and script arguments
        command = f"sbatch --partition={partition} --job-name={job_name} --nodes=1 --gpus-per-node=0 --time=00:10:00 --output={output_log}"
        if dependency_job_ids:
            command += f" --dependency=afterok:{':'.join(dependency_job_ids)}"
        command += f" scripts/merge_test_cases.sh {method_name} {save_dir}"
        # Run the sbatch command and get the job ID
        job_id = run_subprocess_slurm(command)
        return job_id
    elif mode in ["local", "local_parallel"]:
        # Construct the local command with all parameters and script arguments (with -u in case the output is piped to another file)
        command = f"python -u merge_test_cases.py --method_name={method_name} --save_dir={save_dir}"
        # Execute the local command
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(result.stdout)

def run_step2_single_job(
    mode, partition, job_name, num_gpus, output_log, model_name, behaviors_path, test_cases_path, save_path,
    max_new_tokens, incremental_update, dependency_job_ids=None
):
    """
    Submit or execute a single job to generate completions from a model for each test case prompt (step 2).

    Args:
        mode (str): Execution mode ("slurm", "local", or "local_parallel").
        partition (str): SLURM partition name for job submission.
        job_name (str): Name for the job (for SLURM or logging).
        num_gpus (int): Number of GPUs to allocate for the job.
        output_log (str): Path for the output log file.
        model_name (str): Name of the model used for completion generation.
        behaviors_path (str): Path to the behaviors dataset file.
        test_cases_path (str): Path to the generated test cases file.
        save_path (str): Path to save the generated completions.
        max_new_tokens (int): Maximum number of tokens to generate per completion.
        incremental_update (bool): Whether to update completions incrementally or regenerate from scratch.
        dependency_job_ids (list or None): List of SLURM job IDs that must complete before this job (SLURM only).

    Returns:
        str or Ray ObjectRef or None: SLURM job ID, Ray future, or None depending on mode.
    """
    if mode == "slurm":
        incremental_update_str = "True" if incremental_update else "False"
        # Construct the sbatch command with all parameters and script arguments
        command = f"sbatch --partition={partition} --job-name={job_name} --nodes=1 --gpus-per-node={num_gpus} --time=5:00:00 --output={output_log}"
        if dependency_job_ids:
            command += f" --dependency=afterok:{':'.join(dependency_job_ids)}"
        command += f" scripts/generate_completions.sh {model_name} {behaviors_path} {test_cases_path} {save_path} {max_new_tokens} {incremental_update_str}"
        # Run the sbatch command and get the job ID
        job_id = run_subprocess_slurm(command)
        return job_id
    
    elif mode in ["local", "local_parallel"]:
        # Construct the local command with all parameters and script arguments (with -u in case the output is piped to another file)
        incremental_update_str = "--incremental_update" if incremental_update else ""
        command = f"python -u generate_completions.py --model_name={model_name} --behaviors_path={behaviors_path} --test_cases_path={test_cases_path} \
            --save_path={save_path} --max_new_tokens={max_new_tokens} {incremental_update_str}"
        # Execute the local command
        if mode == 'local':
            print(f"[DEBUG] Ejecutando comando:\n{command}\n")
            subprocess.run(command, shell=True, stdout=sys.stdout, stderr=sys.stderr)
        else:
            future = run_subprocess_ray.options(resources={"visible_gpus": num_gpus}).remote(command)
            return future

def run_step3_single_job(mode, partition, job_name, output_log, cls_path, behaviors_path, completions_path, results_path, include_advbench_metric=False):
    """
    Submit or execute a single job to evaluate model completions using a classifier (step 3).

    Args:
        mode (str): Execution mode ("slurm", "local", or "local_parallel").
        partition (str): SLURM partition name for job submission.
        job_name (str): Name for the job (for SLURM or logging).
        output_log (str): Path for the output log file.
        cls_path (str): Path to the classifier model or directory.
        behaviors_path (str): Path to the behaviors dataset file.
        completions_path (str): Path to the generated completions file.
        save_path (str): Path to save the evaluation results.
        dependency_job_ids (list or None): List of SLURM job IDs that must complete before this job (SLURM only).
        include_advbench_metric (bool): Whether to include the AdvBench refusal metric in evaluation.

    Returns:
        str or Ray ObjectRef or None: SLURM job ID, Ray future, or None depending on mode.
    """
    if mode == "slurm":
        # Construct the sbatch command with all parameters and script arguments
        command = f"sbatch --partition={partition} --job-name={job_name} --nodes=1 --gpus-per-node=1 --time=5:00:00 --output={output_log}"
        if dependency_job_ids:
            command += f" --dependency=afterok:{':'.join(dependency_job_ids)}"
        command += f" scripts/evaluate_completions.sh {cls_path} {behaviors_path} {completions_path} {results_path}"
        # Run the sbatch command and get the job ID
        job_id = run_subprocess_slurm(command)
        return job_id
    
    elif mode in ["local", "local_parallel"]:
        # Construct the local command with all parameters and script arguments (with -u in case the output is piped to another file)
        command = f"python -u evaluate_completions.py --cls_path={cls_path} --behaviors_path={behaviors_path} --completions_path={completions_path} \
            --save_path={results_path}"
        if include_advbench_metric:
            command += " --include_advbench_metric"
        
        # Execute the local command
        if mode == "local":
            print("Comando: ", command)
            subprocess.run(command, shell=True, capture_output=False, text=True)
        # Return command to be executed parallel
        elif mode == "local_parallel":
            future = run_subprocess_ray.options(resources={"visible_gpus": 1}).remote(command)
            return future


def run_step1(args, expanded_pipeline_configs):
    """
    Manage and launch jobs to generate all test cases across methods and experiments (step 1).

    Args:
        args (argparse.Namespace): Parsed command-line arguments with pipeline options.
        expanded_pipeline_configs (dict): Expanded pipeline configuration for selected methods and models.

    Returns:
        dict or None: Nested dictionary of job IDs for submitted jobs, or None if no jobs were submitted.
    """
    # Prepare GPTFuzz arguments
    gptfuzz_args = None
    if hasattr(args, 'gptfuzz_mutation_model') and args.gptfuzz_mutation_model:
        # Interactive dtype selection if not specified and using local model
        dtype_to_use = args.gptfuzz_dtype
        if args.gptfuzz_use_local_model and (dtype_to_use is None or dtype_to_use == "auto"):
            print(f"\n🚀 Setting up GPTFuzz with local model...")
            dtype_to_use = "bfloat16"
            #interactive_dtype_selection(args.gptfuzz_mutation_model)
            print(f"✅ Selected dtype: {dtype_to_use}")
        elif dtype_to_use is None:
            dtype_to_use = "auto"  # Default for non-interactive mode
            
        gptfuzz_args = {
            'mutation_model': args.gptfuzz_mutation_model,
            'use_local_model': args.gptfuzz_use_local_model,
            'openai_key': args.gptfuzz_openai_key,
            'dtype': dtype_to_use
        }
    
    # Compute num_behaviors to get end_idx
    with open(args.behaviors_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        behaviors = list(reader)
    num_behaviors = len(behaviors)

    # Handling behavior_ids_subset
    if args.behavior_ids_subset:
        if os.path.exists(args.behavior_ids_subset):
            with open(args.behavior_ids_subset, 'r') as f:
                behavior_ids_subset = f.read().splitlines()
        else:
            behavior_ids_subset = args.behavior_ids_subset.split(',')
        
        behaviors = [b for b in behaviors if b['BehaviorID'] in behavior_ids_subset]
        num_behaviors = len(behaviors)

    # Running jobs
    job_ids = {}
    num_job_ids = 0

    for expanded_pipeline_config in expanded_pipeline_configs.values():
        method_name = expanded_pipeline_config['class_name']
        if method_name not in job_ids:
            job_ids[method_name] = {}
        base_num_gpus = expanded_pipeline_config['base_num_gpus']

        for experiment in expanded_pipeline_config['experiments']:
            experiment_name = experiment['experiment_name']
            if experiment_name not in job_ids[method_name]:
                job_ids[method_name][experiment_name] = []
            num_gpus = base_num_gpus + experiment['experiment_num_gpus']
            save_dir=os.path.join(args.base_save_dir, method_name, experiment_name, 'test_cases')

            if experiment['behavior_chunk_size'] == 'all_behaviors':
                behavior_chunk_size = num_behaviors
            else:
                behavior_chunk_size = experiment['behavior_chunk_size']

            for start_idx in tqdm(range(0, num_behaviors, behavior_chunk_size)):
                end_idx = str(start_idx + behavior_chunk_size)
                start_idx = str(start_idx)
                if (behavior_ids_subset := args.behavior_ids_subset):
                    behavior_ids_subset = ','.join([b['BehaviorID'] for b in behaviors[int(start_idx):int(end_idx)]])
                output_log = os.path.join(args.base_log_dir, 'generate_test_cases', method_name, experiment_name, f'start_idx_{start_idx}.log')
                run_ids = expanded_pipeline_config.get('run_ids', [''])

                for run_id in run_ids:
                    
                    if run_id != '':
                        job_name=f"generate_test_cases_{method_name}_{experiment_name}_{start_idx}_{end_idx}_{run_id}"
                    else:
                        job_name=f"generate_test_cases_{method_name}_{experiment_name}_{start_idx}_{end_idx}"

                    job_id = run_step1_single_job(args.mode, args.partition, job_name, num_gpus, output_log, method_name, experiment_name,
                                                  args.behaviors_path, save_dir, start_idx, end_idx, behavior_ids_subset, run_id,
                                                  args.overwrite, args.verbose, gptfuzz_args)
                    if job_id:
                        job_ids[method_name][experiment_name].append(job_id)
                        num_job_ids += 1

    if num_job_ids == 0:
        return None
    
    # ======== parallel with ray for local_paralllel (NOTE: experimental mode)=====
    if args.mode == "local_parallel":
        # flatten all jobs
        futures = [job_id for method in job_ids.values() for experiment in method.values() for job_id in experiment]
        ray.get(futures)
        return None
    
    return job_ids


def run_step1_5(args, expanded_pipeline_configs, dependency_job_ids=None):
    """
    Merge all generated test case files across methods and experiments into a single test_cases.json (step 1.5).

    Args:
        args (argparse.Namespace): Parsed command-line arguments with pipeline options.
        expanded_pipeline_configs (dict): Expanded pipeline configuration for selected methods and models.
        dependency_job_ids (dict or None): Nested dictionary of job IDs for dependency management (SLURM only).

    Returns:
        dict or None: Nested dictionary of job IDs for submitted merge jobs, or None if no jobs were submitted.
    """
    job_ids = {}
    num_job_ids = 0
    for expanded_pipeline_config in expanded_pipeline_configs.values():
        method_name = expanded_pipeline_config['class_name']
        if method_name not in job_ids:
            job_ids[method_name] = {}

        for experiment in expanded_pipeline_config['experiments']:
            experiment_name = experiment['experiment_name']
            if experiment_name not in job_ids[method_name]:
                job_ids[method_name][experiment_name] = []
            save_dir=os.path.join(args.base_save_dir, method_name, experiment_name, 'test_cases')
            job_name=f"merge_test_cases_{method_name}_{experiment_name}"
            output_log = os.path.join(args.base_log_dir, 'merge_test_cases', method_name, f'{experiment_name}.log')
            if dependency_job_ids and method_name in dependency_job_ids and experiment_name in dependency_job_ids[method_name]:
                current_dep_job_ids = dependency_job_ids[method_name][experiment_name]
            else:
                current_dep_job_ids = None
            job_id = run_step1_5_single_job(args.mode, args.partition, job_name, output_log, method_name, save_dir,
                                            dependency_job_ids=current_dep_job_ids)
            if job_id:
                job_ids[method_name][experiment_name].append(job_id)
                num_job_ids += 1
    
    if num_job_ids == 0:
        job_ids = None
    return job_ids


def run_step2(args, expanded_pipeline_configs, model_configs, dependency_job_ids=None):
    """
    Launch jobs to generate completions for all models and methods using the generated test cases (step 2).

    Args:
        args (argparse.Namespace): Parsed command-line arguments with pipeline options.
        expanded_pipeline_configs (dict): Expanded pipeline configuration for selected methods and models.
        model_configs (dict): Model configuration dictionary loaded from YAML.
        dependency_job_ids (dict or None): Nested dictionary of job IDs for dependency management (SLURM only).

    Returns:
        dict or None: Nested dictionary of job IDs for submitted completion generation jobs, or None if no jobs were submitted.
    """
    job_ids = {}
    num_job_ids = 0

    for expanded_pipeline_config in expanded_pipeline_configs.values():
        method_name = expanded_pipeline_config['class_name']
        if method_name not in job_ids:
            job_ids[method_name] = {}

        for experiment in expanded_pipeline_config['experiments']:
            experiment_name = experiment['experiment_name']
            if experiment_name not in job_ids[method_name]:
                job_ids[method_name][experiment_name] = []

            for model_name in experiment['expected_target_models']:
                num_gpus = model_configs[model_name].get('num_gpus', 0)
                test_cases_path = os.path.join(args.base_save_dir, method_name, experiment_name, 'test_cases', 'test_cases.json')
                if args.max_new_tokens != 512:
                    save_path = os.path.join(args.base_save_dir, method_name, experiment_name, 'completions',
                                             f'{args.max_new_tokens}_tokens', f'{model_name}.json')
                else:
                    save_path = os.path.join(args.base_save_dir, method_name, experiment_name, 'completions', f'{model_name}.json')

                job_name=f"generate_completions_{method_name}_{experiment_name}_{model_name}"
                output_log = os.path.join(args.base_log_dir, 'generate_completions', method_name, experiment_name, f'{model_name}.log')

                if dependency_job_ids and method_name in dependency_job_ids and experiment_name in dependency_job_ids[method_name]:
                    current_dep_job_ids = dependency_job_ids[method_name][experiment_name]
                else:
                    current_dep_job_ids = None
                job_id = run_step2_single_job(args.mode, args.partition, job_name, num_gpus, output_log, model_name, args.behaviors_path,
                                              test_cases_path, save_path, args.max_new_tokens, args.incremental_update,
                                              dependency_job_ids=current_dep_job_ids)
                if job_id:
                    job_ids[method_name][experiment_name].append(job_id)
                    num_job_ids += 1
    
    if num_job_ids == 0:
        job_ids = None

    if args.mode == "local_parallel":
        # flatten all jobs
        futures = [job_id for method in job_ids.values() for experiment in method.values() for job_id in experiment]
        ray.get(futures)
        return None
    
    return job_ids


def run_step3(args, expanded_pipeline_configs, dependency_job_ids=None):
    """
    Launch jobs to evaluate all generated completions with the classifier (step 3).

    Args:
        args (argparse.Namespace): Parsed command-line arguments with pipeline options.
        expanded_pipeline_configs (dict): Expanded pipeline configuration for selected methods and models.
        dependency_job_ids (dict or None): Nested dictionary of job IDs for dependency management (SLURM only).

    Returns:
        None
    """
    job_ids = {}
    num_job_ids = 0
    print("Step 3: ")
    for expanded_pipeline_config in expanded_pipeline_configs.values():
        method_name = expanded_pipeline_config['class_name']
        if method_name not in job_ids:
            job_ids[method_name] = {}

        for experiment in expanded_pipeline_config['experiments']:
            experiment_name = experiment['experiment_name']
            if experiment_name not in job_ids[method_name]:
                job_ids[method_name][experiment_name] = []

            for model_name in experiment['expected_target_models']:
                if args.max_new_tokens != 512:
                    completions_path = os.path.join(args.base_save_dir, method_name, experiment_name, 'completions',
                                                    f'{args.max_new_tokens}_tokens', f'{model_name}.json')
                    results_path = os.path.join(args.base_save_dir, method_name, experiment_name, 'results',
                                                f'{args.max_new_tokens}_tokens', f'{model_name}.json')
                else:
                    completions_path = os.path.join(args.base_save_dir, method_name, experiment_name, 'completions', f'{model_name}.json')
                    results_path = os.path.join(args.base_save_dir, method_name, experiment_name, 'results', f'{model_name}.json')

                job_name=f"evaluate_completions_{method_name}_{experiment_name}_{model_name}"
                output_log = os.path.join(args.base_log_dir, 'evaluate_completions', method_name, experiment_name, f'{model_name}.log')

                if dependency_job_ids and method_name in dependency_job_ids and experiment_name in dependency_job_ids[method_name]:
                    current_dep_job_ids = dependency_job_ids[method_name][experiment_name]
                else:
                    current_dep_job_ids = None
                print("Llamada a step3 single job:")
                job_id = run_step3_single_job(
                    args.mode, args.partition, job_name, output_log,
                    args.cls_path, args.behaviors_path, completions_path, results_path,
                    include_advbench_metric=args.include_advbench_metric
                )
                job_ids[method_name][experiment_name].append(job_id)
    
    if args.mode == "local_parallel":
        # flatten all jobs
        futures = [job_id for method in job_ids.values() for experiment in method.values() for job_id in experiment]
        ray.get(futures)

def main():
    """
    Orchestrate the pipeline workflow according to the selected step and user arguments.
    """
    print("main", flush=True)
    args = parse_args()
    
    # ==================== LOG COMPREHENSIVE SYSTEM INFORMATION ==================== #
    try:
        sys.path.append('.')  # Add current directory to import system_info_logger
        from system_info_logger import log_system_info
        import datetime
        
        # Create system info log with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"system_info_{timestamp}.json"
        
        print("🔍 Logging comprehensive system information...")
        system_info = log_system_info(output_file=log_filename, console_output=True)
        
        print(f"📁 System information saved to: {log_filename}")
        print("-" * 80)
        
    except Exception as e:
        print(f"⚠️ Warning: Could not log system information: {e}")
        print("Continuing with pipeline execution...")
        print("-" * 80)

    with open(args.pipeline_config_path) as file:
        pipeline_config = yaml.full_load(file)
    
    with open(args.models_config_path) as file:
        model_configs = yaml.full_load(file)        
    # Get the pipeline config for each method
    if args.methods == "all":
        methods = list(pipeline_config.keys())
    else:
        methods = args.methods.split(',')
        for method in methods:
            assert method in pipeline_config, f"Method {method} not found in pipeline_config.yaml"
    
    if args.models == "all":
        models = list(model_configs.keys())
    else:
        models = args.models.split(',')
        for model in models:
            assert model in model_configs, f"Model {model} not found in models.yaml"

    expanded_pipeline_configs = get_expanded_pipeline_configs(pipeline_config, model_configs, methods, models)
    
    # ===================== Run the pipeline ===================== #
    parallel_mode = args.mode == "local_parallel"
    if parallel_mode:
        _init_ray(num_cpus=8, reinit=False, resources={"visible_gpus": torch.cuda.device_count()})
    
    if args.step == "1":
        run_step1(args, expanded_pipeline_configs)
    elif args.step == "1.5":
        run_step1_5(args, expanded_pipeline_configs)
    elif args.step == "2":
        run_step2(args, expanded_pipeline_configs, model_configs)
    elif args.step == "3":
        print("runing step3")
        run_step3(args, expanded_pipeline_configs)
    elif args.step == "2_and_3":
        dependency_job_ids = run_step2(args, expanded_pipeline_configs, model_configs)
        run_step3(args, expanded_pipeline_configs, dependency_job_ids)
    elif args.step == "all":
        dependency_job_ids = run_step1(args, expanded_pipeline_configs)
        dependency_job_ids = run_step1_5(args, expanded_pipeline_configs, dependency_job_ids)
        dependency_job_ids = run_step2(args, expanded_pipeline_configs, model_configs, dependency_job_ids)
        run_step3(args, expanded_pipeline_configs, dependency_job_ids)
    
    # ===================== Generate comprehensive report ===================== #
    if args.generate_report and args.step in ["3", "all", "2_and_3"] and args.mode != "slurm":
        print("\n🔄 Generating comprehensive evaluation report...")
        try:
            import sys
            sys.path.append('..')
            from report_generator import EvaluationReportGenerator
            
            report_generator = EvaluationReportGenerator()
            report = report_generator.generate_report(
                methods=methods,
                models=models,
                behaviors_path=args.behaviors_path,
                output_dir="./reports",
                formats=args.report_formats
            )
            
            print(f"✅ Comprehensive evaluation report generated: {report.report_id}")
            print(f"📊 Overall ASR: {report.summary_statistics.get('overall_asr', 0):.2%}")
            print(f"📋 Behaviors analyzed: {report.summary_statistics.get('total_behaviors', 0)}")
            print(f"📁 Report saved to: ./reports/")
            
        except Exception as e:
            print(f"❌ Error generating evaluation report: {e}")
            print("⚠️  Pipeline completed successfully, but report generation failed")

if __name__ == "__main__":
    main()
