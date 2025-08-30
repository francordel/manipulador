import torch
from torch.nn import CrossEntropyLoss
import numpy as np
from tqdm import tqdm
from ..baseline import SingleBehaviorRedTeamingMethod
from ..model_utils import get_template
import json
import os

# Set environment variables for memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid tokenizer warnings


from fastchat.model import add_model_args
import argparse
import pandas as pd
from .gptfuzzer.fuzzer.selection import MCTSExploreSelectPolicy
from .gptfuzzer.fuzzer.mutator import (
    MutateRandomSinglePolicy, OpenAIMutatorCrossOver, OpenAIMutatorExpand,
    OpenAIMutatorGenerateSimilar, OpenAIMutatorRephrase, OpenAIMutatorShorten)
from .gptfuzzer.fuzzer import GPTFuzzer
from .gptfuzzer.llm import OpenAILLM, LocalLLM, PaLM2LLM, ClaudeLLM, MutationLLM
from .gptfuzzer.utils.predict import RoBERTaPredictor
from .gptfuzzer.utils.template import synthesis_message
import random
random.seed(100)
import logging
httpx_logger: logging.Logger = logging.getLogger("httpx")
# disable httpx logging
httpx_logger.setLevel(logging.WARNING)

# ============================== GBDA CLASS DEFINITION ============================== #
class GPTFuzz(SingleBehaviorRedTeamingMethod):
    def __init__(self,
                target_model,
                mutate_model_path,
                openai_key=None,
                use_local_model=False,
                max_query=1000,
                max_jailbreak=1,
                max_iteration=100,
                energy=1,
                seed_selection_strategy='round_robin',
                max_new_tokens=512,
                seed_path='baselines/gptfuzzer/GPTFuzzer.csv',
                targets_path=None,
                num_gpus=1,
                dtype=None,
                **model_kwargs):
        """
        :param max_query: the maximum number of queries to use
        :param max_jailbreak: the maximum number of jailbreaks to use
        :param energy: the energy to use
        :param seed_selection_strategy: the strategy to use for selecting seeds
        :param max_new_tokens: the maximum number of new tokens to use
        :param seed_path: the path to the seed file
        """
        # This causes OOM in VLLM
        # super().__init__(target_model, **model_kwargs)
        self.max_query = max_query
        self.max_jailbreak = max_jailbreak
        self.energy = energy
        self.seed_selection_strategy = seed_selection_strategy
        self.max_new_tokens = max_new_tokens
        self.max_iteration = max_iteration

        if targets_path is None:
            raise ValueError("targets_path must be specified")
        with open(targets_path, 'r', encoding='utf-8') as file:
            self.behavior_id_to_target = json.load(file)
        self.targets_path = targets_path

        template = get_template(**target_model)['prompt']
        self.template = template
        self.before_tc, self.after_tc = template.split("{instruction}")

        self.initial_seed = pd.read_csv(seed_path)['text'].tolist()

        # Initialize mutation model based on configuration
        print(f"Using local model: {use_local_model}")
        if use_local_model:
            print(f"Using local mutation model from {mutate_model_path}")
            print(f"Using dtype: {dtype}")
            # Convert dtype parameter to torch dtype
            torch_dtype = torch.float16  # Default for V100 compatibility
            if dtype == "auto":
                torch_dtype = torch.float16  # Safe default for V100
            elif dtype == "float16" or dtype == "half":
                torch_dtype = torch.float16
            elif dtype == "bfloat16":
                print(f"PROBLEM: bfloat16 is not supported on V100, torch.bfloat16 = {torch.bfloat16}")
                torch_dtype = torch.bfloat16
            elif dtype == "float32":
                torch_dtype = torch.float32
            print(f"Using torch dtype: {torch_dtype}")
            # Add memory optimization for V100
            print("🔄 Loading mutation model with memory optimizations...")
            self.mutation_model = MutationLLM(
                mutate_model_path, 
                dtype=torch_dtype, 
                num_gpus=num_gpus,
                max_gpu_memory="2GB",  # Very conservative for shared GPU
                load_8bit=True  # Enable 8-bit quantization for memory efficiency
            )
            
            # Clear cache after loading mutation model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("🧹 Cleared CUDA cache after loading mutation model")
        else:
            if openai_key is None:
                raise ValueError("OpenAI API key is required when not using local model")
            self.mutation_model = OpenAILLM(mutate_model_path, openai_key)
        # target_model = PaLM2LLM(args.target_model, args.palm_key)
        # target_model = ClaudeLLM(args.target_model, args.claude_key)
        # target_model = LocalLLM(args.target_model) # we suggest using LocalVLLM for better performance, however if you are facing difficulties in installing vl
        target_model['num_gpus'] = num_gpus
        print(f"Using target model with num_gpus: {num_gpus}")
        
        # Add dtype configuration for target model (V100 compatibility)
        # Override the model config dtype if user specified one via --gptfuzz_dtype
        if use_local_model and dtype is not None:
            original_dtype = target_model.get('dtype', 'not set')
            if dtype == "auto":
                target_model['dtype'] = "float16"  # Safe default for V100
            elif dtype == "float16" or dtype == "half":
                target_model['dtype'] = "float16"
            elif dtype == "bfloat16":
                target_model['dtype'] = "bfloat16"
            elif dtype == "float32":
                target_model['dtype'] = "float32"
            else:
                target_model['dtype'] = "float16"  # Safe fallback
            
            print(f"🔄 Target model dtype override: {original_dtype} -> {target_model['dtype']}")
            
            target_model['gpu_memory_utilization'] = 0.6  # Conservative for V100
            target_model['max_model_len'] = 4096  # Reduce context length
            target_model['enforce_eager'] = True  # Avoid compilation overhead
            
            print(f"💾 Target model memory optimizations:")
            print(f"   gpu_memory_utilization: {target_model['gpu_memory_utilization']}")
            print(f"   max_model_len: {target_model['max_model_len']}")
            print(f"   enforce_eager: {target_model['enforce_eager']}")
            
        elif 'dtype' in target_model:
            print(f"📋 Using model config dtype: {target_model['dtype']}")
        
        # Use transformers instead of vLLM to avoid compatibility issues
        print("🔄 Using transformers instead of vLLM for target model")
        
        # Add memory management and cleanup before loading target model
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Check available memory
            free_memory = torch.cuda.mem_get_info()[0] / 1024**3  # Convert to GB
            total_memory = torch.cuda.mem_get_info()[1] / 1024**3
            used_memory = total_memory - free_memory
            
            print(f"🔍 GPU Memory Status:")
            print(f"   Total: {total_memory:.2f} GB")
            print(f"   Used: {used_memory:.2f} GB")
            print(f"   Free: {free_memory:.2f} GB")
            
            if free_memory < 10.0:  # Less than 10GB free
                print(f"⚠️  WARNING: Low GPU memory available ({free_memory:.2f} GB)")
                print(f"   Other processes may be using GPU memory")
                print(f"   Consider killing other processes or using a different GPU")
                
            if free_memory < 5.0:  # Less than 5GB free
                print(f"🚨 CRITICAL: Very low GPU memory ({free_memory:.2f} GB)")
                print(f"   GPTFuzz may fail with OOM errors")
                
            print(f"🧹 Cleared CUDA cache before loading target model")
        
        from ..model_utils import load_model_and_tokenizer
        
        # Configure memory-efficient loading for target model
        target_model_config = target_model.copy()
        target_model_config.update({
            'low_cpu_mem_usage': True,
            'torch_dtype': torch.float16,  # Force float16 for memory efficiency
            'device_map': 'auto',
            'max_memory': {0: '4GB'} if torch.cuda.is_available() else None,  # Very conservative limit
        })
        
        self.target_model_obj, self.target_tokenizer = load_model_and_tokenizer(**target_model_config)
        # --- FIX: ensure pad_token for target tokenizer and set padding side ---
        if self.target_tokenizer.pad_token is None:
            _eos = getattr(self.target_tokenizer, "eos_token", None)
            if _eos is not None:
                self.target_tokenizer.pad_token = _eos
            else:
                self.target_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                self.target_tokenizer.pad_token = "[PAD]"

        # Prefer left padding for decoder-only; right for encoder-decoder
        try:
            from transformers import AutoConfig
            _cfg = AutoConfig.from_pretrained(target_model_config['model_name_or_path'], trust_remote_code=True)
            if getattr(_cfg, "is_encoder_decoder", False):
                self.target_tokenizer.padding_side = "right"
            else:
                self.target_tokenizer.padding_side = "left"
        except Exception:
            self.target_tokenizer.padding_side = "left"

        # If we added a new special token, ensure embeddings are resized
        try:
            self.target_model_obj.resize_token_embeddings(len(self.target_tokenizer))
        except Exception as _e:
            print(f"⚠️ Warning: could not resize embeddings after adding PAD: {_e}")

        self.target_model = TransformersWrapper(self.target_model_obj, self.target_tokenizer, target_model_config)
        
        # Use CPU for RoBERTa to save GPU memory (V100 optimization)
        if use_local_model:
            print("💾 Using CPU for RoBERTa model to save GPU memory")
            roberta_device = 'cpu'
        else:
            # If not using local models, we can use GPU for RoBERTa
            extra_device = f'cuda:{torch.cuda.device_count()-1}' if torch.cuda.device_count() > 1 else 'cuda:0'
            roberta_device = extra_device
            
        self.roberta_model = RoBERTaPredictor('hubert233/GPTFuzz', device=roberta_device)

    def generate_test_cases_single_behavior(self, behavior, num_generate, verbose=False):
        # ========== Memory Management ==========
        # Clear CUDA cache and run garbage collection before each test case
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"🧹 Cleared memory before generating test case")
        
        # ========== Behavior and Target str ==========
        # get required variables from behavior dictionary
        behavior_dict = behavior
        behavior = behavior_dict['Behavior']
        context_str = behavior_dict['ContextString']
        behavior_id = behavior_dict['BehaviorID']
    
        target = self.behavior_id_to_target[behavior_id]
        behavior += ' '
        # handle setup for behaviors with context strings
        if context_str:
            behavior = f"{context_str}\n\n---\n\n{behavior}"
            
        # ========== Enhanced Logging ==========
        print(f"🎯 [BEHAVIOR] Starting GPTFuzz for: {behavior_id}")
        print(f"📝 [BEHAVIOR] Description: {behavior[:100]}..." if len(behavior) > 100 else f"📝 [BEHAVIOR] Description: {behavior}")
        print(f"🎯 [TARGET] Expected target response: {target[:50]}..." if len(target) > 50 else f"🎯 [TARGET] Expected target response: {target}")
        print(f"🌱 [SEEDS] Available initial seeds: {len(self.initial_seed)} seeds")
        print(f"🌱 [SEEDS] Sample seeds: {self.initial_seed[:3] if len(self.initial_seed) >= 3 else self.initial_seed}")
        print(f"⚙️  [CONFIG] Max iterations: {self.max_iteration}, Max queries: {self.max_query}, Energy: {self.energy}")

        fuzzer = GPTFuzzer(
            questions=[behavior],
            # target_model=openai_model,
            target=self.target_model,
            predictor=self.roberta_model,
            initial_seed=self.initial_seed,
            mutate_policy=MutateRandomSinglePolicy([
                OpenAIMutatorCrossOver(self.mutation_model, temperature=0.0),  # for reproduction only, if you want better performance, use temperature>0
                OpenAIMutatorExpand(self.mutation_model, temperature=0.0),
                OpenAIMutatorGenerateSimilar(self.mutation_model, temperature=0.0),
                OpenAIMutatorRephrase(self.mutation_model, temperature=0.0),
                OpenAIMutatorShorten(self.mutation_model, temperature=0.0)],
                concatentate=True,
            ),
            select_policy=MCTSExploreSelectPolicy(),
            energy=self.energy,
            max_jailbreak=self.max_jailbreak,
            max_query=self.max_query,
            max_iteration=self.max_iteration,
            generate_in_batch=False,
        )

        fuzzer.run()
        logs = []
        test_cases = []
        for i in range(len(fuzzer.prompt_history)):
            test_case = synthesis_message(behavior, fuzzer.prompt_history[i])
            if test_case is not None:  # Only add valid test cases
                test_cases.append(test_case)
                logs.append(dict(loss=0, test_case=test_case))
            else:
                print(f"⚠️ WARNING: synthesis_message returned None for prompt: {fuzzer.prompt_history[i][:50]}...")
        
        # Handle case where no test cases were generated
        if len(test_cases) == 0:
            print(f"⚠️ WARNING: GPTFuzz generated no test cases for behavior. Returning empty list.")
            test_cases = []
        
        # Clean up memory after test case generation
        del fuzzer
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"🧹 Cleared memory after generating test case")
            
        return test_cases, logs


class TransformersWrapper:
    """Wrapper class to replace LocalVLLM functionality using transformers."""
    
    def __init__(self, model, tokenizer, model_config):
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.template = get_template(model_config['model_name_or_path'], 
                                   chat_template=model_config.get('chat_template', None))

    def generate(self, prompt, temperature=0, max_tokens=512):
        """Generate single response using transformers."""
        prompts = [prompt]
        return self.generate_batch(prompts, temperature, max_tokens)[0]

    def generate_batch(self, prompts, temperature=0, max_tokens=512):
        """Generate batch responses using transformers with memory management."""
        # Process prompts in smaller batches to avoid OOM
        batch_size = min(len(prompts), 4)  # Limit batch size for memory efficiency
        all_generated_texts = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # Format prompts using template
            formatted_prompts = [self.template['prompt'].format(instruction=p if p is not None else "") for p in batch_prompts]
            
            # Tokenize inputs with truncation for memory safety
            inputs = self.tokenizer(
                formatted_prompts, 
                return_tensors='pt', 
                padding=True, 
                truncation=True,
                max_length=1024  # Limit input length to prevent OOM
            )
            
            # Move to device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate with memory management
            with torch.no_grad():
                try:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=min(max_tokens, 256),  # Limit output length
                        temperature=temperature if temperature > 0 else 0.01,
                        do_sample=temperature > 0,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=False,  # Disable KV cache to save memory
                    )
                    
                    # Decode outputs
                    batch_generated_texts = []
                    for j, output in enumerate(outputs):
                        # Skip input tokens
                        generated_tokens = output[inputs['input_ids'][j].shape[0]:]
                        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                        if generated_text is not None:
                            batch_generated_texts.append(generated_text.strip())
                        else:
                            batch_generated_texts.append("")  # Use empty string instead of None
                    
                    all_generated_texts.extend(batch_generated_texts)
                    
                except torch.cuda.OutOfMemoryError:
                    print("⚠️  CUDA OOM during generation, clearing cache and retrying with smaller batch...")
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    # Retry with batch size 1
                    for single_prompt in batch_prompts:
                        single_formatted = [self.template['prompt'].format(instruction=single_prompt if single_prompt is not None else "")]
                        single_inputs = self.tokenizer(
                            single_formatted, 
                            return_tensors='pt', 
                            truncation=True,
                            max_length=512  # Even smaller for single prompts
                        )
                        single_inputs = {k: v.to(device) for k, v in single_inputs.items()}
                        
                        with torch.no_grad():
                            single_outputs = self.model.generate(
                                **single_inputs,
                                max_new_tokens=128,  # Very conservative
                                temperature=0.01,
                                do_sample=False,
                                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                                eos_token_id=self.tokenizer.eos_token_id,
                                use_cache=False,
                            )
                            
                            generated_tokens = single_outputs[0][single_inputs['input_ids'][0].shape[0]:]
                            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                            if generated_text is not None:
                                all_generated_texts.append(generated_text.strip())
                            else:
                                all_generated_texts.append("")  # Use empty string instead of None
                    
                finally:
                    # Clear memory after each batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        return all_generated_texts
