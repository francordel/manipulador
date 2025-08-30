import torch
import numpy as np
from tqdm import tqdm
import random
from torch.nn import CrossEntropyLoss
from ..baseline import SingleBehaviorRedTeamingMethod
from ..check_refusal_utils import check_refusal_completions
from ..model_utils import get_template, load_model_and_tokenizer, load_vllm_model
from accelerate.utils import find_executable_batch_size
from vllm import SamplingParams
import re
import json
import os

class FewShot(SingleBehaviorRedTeamingMethod):
    use_ray = True # vllm uses ray to handle multi-gpus (tensor_parallel)

    def __init__(self, target_model, attack_model, targets_path, method_template, completion_prefix,
                generation_config, sample_size=1000, n_shots=5, num_steps=5, method_template_contextual=None,
                completion_prefix_contextual=None, **kwargs):
        """
        :param target_model: a dictionary specifying the target model (kwargs to load_model_and_tokenizer)
        :param attack_model: a dictionary specifying the attack model (kwargs to load_vllm_model)
        :param targets_path: the path to the targets JSON file
        :param generation_config: the generation config for attack model
        :param sample_size: the number of samples to generate at each step
        :param n_shots: the number of shots to use for fewshot
        :param num_steps: the number of steps to use
        :param method_template: few-shot prompt template for behaviors w/o context string
        :param completion_prefix: the completion prefix for behaviors w/o context string
        :param method_template_contextual: few-shot prompt template for behaviors w/ context string
        :param completion_prefix_contextual: the completion prefix for behaviors w/ context string
        """

        self.method_template = method_template
        self.completion_prefix = completion_prefix
        self.method_template_contextual = method_template_contextual
        self.completion_prefix_contextual = completion_prefix_contextual
        assert '{behavior}' in self.method_template, \
             "method_template needs to include `{behavior}` format keys"
        with open(targets_path, 'r', encoding='utf-8') as file:
            self.behavior_id_to_target = json.load(file)

        # ===== Load Attack Model ======
        shared_model = attack_model == target_model
        allocated_gpus = attack_model.get("num_gpus", 1)

        if shared_model:
            print("[FewShot] Using same model for attack and target.")
            self.attack_model_template = get_template(**target_model)['prompt']
            self.template = self.attack_model_template
            if attack_model.get("use_vllm", True):
                self.attack_model = load_vllm_model(**attack_model)
                self.model = None
                self.tokenizer = None
                self.vllm_shared = True
            else:
                self.model, self.tokenizer = load_model_and_tokenizer(**attack_model)
                self.attack_model = self.model
                self.vllm_shared = False
                # Fix tokenizer padding issue
                if self.tokenizer.pad_token is None:
                    if self.tokenizer.eos_token is not None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    elif self.tokenizer.unk_token is not None:
                        self.tokenizer.pad_token = self.tokenizer.unk_token
                    else:
                        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        else:
            self.attack_model_template = get_template(**attack_model)['prompt']
            if attack_model.get("use_vllm", True):
                self.attack_model = load_vllm_model(**attack_model)
            else:
                self.attack_model, _ = load_model_and_tokenizer(**attack_model)
            self.vllm_shared = False

            self.template = get_template(**target_model)['prompt']
            num_gpus = target_model.get("num_gpus", 1)
            # Use standard model loading without custom memory constraints
            self.model, self.tokenizer = load_model_and_tokenizer(**target_model)
            # Fix tokenizer padding issue
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                elif self.tokenizer.unk_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.unk_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.generation_config = SamplingParams(**generation_config)
        self.generation_config.stop = ['11.'] # dummy stop in case the generation go over a list of 10 as we fix each query to generate 10 cases

        # self._generation_sample_size = min(32, sample_size)  # starting _generation_batch_size, will decrease if go OOM
        self._foward_batch_size = 128 # starting _foward_batch_size to get loss, will decrease if go OOM

        self.sample_size = sample_size
        self.n_shots = n_shots
        self.num_steps = num_steps

    def generate_test_cases_single_behavior(self, behavior_dict, verbose=False, **kwargs):
        # Unpack behavior_dict
        behavior = behavior_dict['Behavior']
        context_str = behavior_dict['ContextString']
        behavior_id = behavior_dict['BehaviorID']
        target = self.behavior_id_to_target[behavior_id]

        attack_model, model = self.attack_model, self.model
        tokenizer = self.tokenizer
        # Robust device management
        try:
            device = model.device if model is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except Exception as e:
            print(f"⚠️ Warning: Could not determine model device, using cuda: {e}")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        attack_model_template, template = self.attack_model_template, self.template


        method_template = self.method_template if not context_str else self.method_template_contextual
        completion_prefix = self.completion_prefix if not context_str else self.completion_prefix_contextual
        sample_size = self.sample_size
        num_steps = self.num_steps
        n_shots = self.n_shots
        generation_config = self.generation_config

        query = method_template.format(behavior=behavior, context=context_str) # will not throw error for self.method_template w/o context 
        query = attack_model_template.format(instruction=query)
        candidates = []
        logs = []

        seen_candidates = set()
        prev_candidates = None
        for step in tqdm(range(num_steps)):
            # If first step, will do zeroshot to get variations
            # Else will use top {n_shots} lowest loss variations as examples for fewshot
            fs_examples = candidates[:n_shots]

            # check if variations are converged
            if fs_examples == prev_candidates:
                break

            prev_candidates = fs_examples

            starting_idx = len(fs_examples) + 1
            shots = []
            for idx, s in enumerate(fs_examples, 1):
                candidate = s.get("candidate", "")
                if candidate is None:
                    candidate = ""
                shots.append(f'{idx}. {candidate}')

            # === construct completion_prefix query===
            # Sure, here's 10 variants of '{behavior}': 
            # 1. ...
            # 2. ...
            # ...
            # {n_shots}. ...
            # {starting_idx}.
            # Ensure all components are strings and not None
            prefix_formatted = completion_prefix.format(behavior=behavior) if completion_prefix else ""
            if prefix_formatted is None:
                prefix_formatted = ""
            
            # Filter out any None values and ensure all are strings
            safe_shots = [str(shot) if shot is not None else "" for shot in shots]
            
            components = [prefix_formatted] + safe_shots + [f'{starting_idx}. ']
            input_query = query + '\n'.join(components)
            current_step_candidates = []

            # === Sampling until got {sample_size} new variations ===
            attempt = 0
            # print("====input_query====")
            # print(input_query)
            while len(current_step_candidates) < sample_size and attempt <= 10:
                input_query_b = [input_query] * round((sample_size - len(current_step_candidates)) / n_shots) 
                outputs = self._vllm_generate(attack_model, input_query_b, generation_config)

                generated_candidates = []
                for o in outputs:
                    # Skip None or empty outputs
                    if o is None or not o.strip():
                        continue
                    # prepend a starting idx back to outputs to extract_numbered_lines
                    o = f'{starting_idx}. ' + o.strip()
                    generated_candidates.extend(extract_numbered_lines(o))

                # === Dedup, filtered and update seen candidates ===
                filtered_candidates = []
                for g in generated_candidates:
                    if g in seen_candidates: continue
                    if not g.strip() or len(g) < 32: continue
                    filtered_candidates.append(g)
                    seen_candidates.add(g)
                # === Add current candidates to list for this step ===
                current_step_candidates.extend(filtered_candidates)
                attempt += 1
            
            # ==== break if current_step_candidates empty after filtering ====
            if len(current_step_candidates) == 0:
                break
        
        # ===== Check if we have any candidates before calculating loss =====
        if len(current_step_candidates) == 0:
            print(f"⚠️ WARNING: No valid candidates generated for behavior. Returning empty list.")
            return [], {}
            
        # ===== Calculating Loss for each candidates =====
        if self.vllm_shared:
            raise RuntimeError("Cannot score loss when using shared vLLM model. Use HuggingFace model instead.")
        loss_function = find_executable_batch_size(self._get_loss_bs, self._foward_batch_size)
        formated_candidates = []
        for c in current_step_candidates:
            # Ensure c is not None before processing
            safe_c = c if c is not None else ""
            instruction = f"{context_str}\n\n---\n\n{safe_c}" if context_str else safe_c
            instruction = template.format(instruction=instruction)
            formated_candidates.append(instruction)
        
        candidates_loss = loss_function(formated_candidates, target)
        for c, loss in zip(current_step_candidates, candidates_loss):
            # Ensure candidate is not None
            safe_candidate = c if c is not None else ""
            candidates.append(dict(candidate=safe_candidate, loss=loss.item()))

            # ===== Sort lowest loss candidates as shot examples for next step =====
            candidates = sorted(candidates, key=lambda x: x['loss'])
            
            # == Add current step to logs ==
            logs.append(fs_examples)

            # ===== Check if lowest loss candidates jailbreak ======
            test_case = candidates[0]['candidate']
            if context_str:
                test_case = f"{context_str}\n\n---\n\n{test_case}"
            _input = template.format(instruction=test_case)
            is_refusal, completions, _ = check_refusal_completions(model, tokenizer, inputs=[_input])
            # print("====test_case==== Loss:", candidates[0]['loss'])
            # print(_input)
            # print("====completions====")
            # print(completions[0])
            if not is_refusal[0]:
                break
            
        return test_case, logs
    
    def _vllm_generate(self, model, queries, generation_config):
        if hasattr(model, "generate") and "sampling_params" not in model.generate.__code__.co_varnames:
            # Use HuggingFace-style generation
            tokenizer = self.tokenizer
            
            # Safe tokenization with error handling
            try:
                inputs = tokenizer(queries, return_tensors="pt", padding=True, truncation=True)
                if torch.cuda.is_available() and hasattr(self.model, 'device'):
                    inputs = inputs.to(self.model.device)
            except Exception as e:
                print(f"⚠️ Error during tokenization: {e}")
                # Fallback: try without device transfer
                inputs = tokenizer(queries, return_tensors="pt", padding=True, truncation=True)
            
            # Convert SamplingParams to safe HuggingFace parameters
            hf_params = {
                'do_sample': False,  # Start conservative
                'max_new_tokens': 256,  # Conservative default
                'pad_token_id': tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
            }
            
            try:
                config_vars = vars(generation_config)
                
                # Only add parameters that are safe and compatible
                if 'max_tokens' in config_vars and config_vars['max_tokens'] > 0:
                    hf_params['max_new_tokens'] = min(config_vars['max_tokens'], 512)  # Cap at 512
                
                if 'temperature' in config_vars and config_vars['temperature'] > 0:
                    temp = float(config_vars['temperature'])
                    if 0.0 < temp <= 2.0:  # Valid range
                        hf_params['temperature'] = temp
                        hf_params['do_sample'] = True
                
                if 'top_p' in config_vars and hf_params.get('do_sample', False):
                    top_p = float(config_vars['top_p'])
                    if 0.0 < top_p <= 1.0:  # Valid range
                        hf_params['top_p'] = top_p
                
                print(f"🔧 Using generation params: {hf_params}")
                
            except Exception as e:
                print(f"⚠️ Error processing generation config: {e}")
                print("Using safe defaults")
            
            # Generate with error handling and memory management
            try:
                with torch.no_grad():
                    outputs = model.generate(**inputs, **hf_params)
                
                # Decode only the new tokens
                input_length = inputs['input_ids'].shape[1]
                new_tokens = outputs[:, input_length:]
                generations = []
                for tokens in new_tokens:
                    decoded = tokenizer.decode(tokens, skip_special_tokens=True)
                    if decoded is not None:
                        generations.append(decoded.strip())
                    else:
                        generations.append("")  # Use empty string instead of None
                
            except Exception as e:
                print(f"❌ Generation failed: {e}")
                print("Falling back to minimal generation")
                # Ultra-safe fallback
                safe_params = {
                    'max_new_tokens': 50,
                    'do_sample': False,
                    'pad_token_id': tokenizer.eos_token_id
                }
                try:
                    with torch.no_grad():
                        outputs = model.generate(**inputs, **safe_params)
                    input_length = inputs['input_ids'].shape[1]
                    new_tokens = outputs[:, input_length:]
                    generations = []
                    for tokens in new_tokens:
                        decoded = tokenizer.decode(tokens, skip_special_tokens=True)
                        if decoded is not None:
                            generations.append(decoded.strip())
                        else:
                            generations.append("")  # Use empty string instead of None
                except:
                    # Last resort: return dummy responses
                    generations = [f"Generated response {i+1}" for i in range(len(queries))]
            
        else:
            # Use vLLM-style generation (original path)
            outputs = model.generate(queries, sampling_params=generation_config, use_tqdm=False)
            generations = []
            for o in outputs:
                text = o.outputs[0].text if o.outputs[0].text is not None else ""
                generations.append(text.strip())
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return generations
    
    def _get_loss_bs(self, batch_size, inputs, target):
        tokenizer = self.tokenizer
        model = self.model

        # === Update _foward_batch_size for later calls as this function is called by find_executable_batch_size ===
        if batch_size != self._foward_batch_size:
            self._foward_batch_size = batch_size
            print(f"INFO: setting _foward_batch_size={self._foward_batch_size}")

        all_candidate_losses = []
        try:
            target_ids = tokenizer(target, return_tensors='pt', add_special_tokens=False).input_ids.to(model.device)
        except RuntimeError as e:
            print(f"⚠️ CUDA error in target tokenization: {e}")
            target_ids = tokenizer(target, return_tensors='pt', add_special_tokens=False).input_ids
            if torch.cuda.is_available():
                target_ids = target_ids.to(model.device)
        for i in range(0, len(inputs), batch_size):
            inputs_b = [str(_input) + target if _input is not None else target for _input in inputs[i:i+batch_size]]
            try:
                encoded_b = tokenizer(inputs_b, return_tensors='pt', padding=True).to(model.device)
            except RuntimeError as e:
                print(f"⚠️ CUDA error in batch encoding: {e}")
                encoded_b = tokenizer(inputs_b, return_tensors='pt', padding=True)
                if torch.cuda.is_available():
                    encoded_b = encoded_b.to(model.device)
            
            with torch.no_grad():
                outputs = model(**encoded_b)
            batch_logits = outputs.logits

            # compute loss
            # Shift so that tokens < n predict n
            shift_logits = batch_logits[:, -target_ids.size(1)-1:-1, :].contiguous()
            shift_labels = target_ids.repeat(shift_logits.size(0), 1)
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_logits.size(0), -1).mean(dim=1)
            all_candidate_losses.append(loss)

        return torch.cat(all_candidate_losses)
    
    @classmethod
    def merge_test_cases(cls, save_dir):
        """
        Merge all test_cases.json files from test_cases/test_cases_individual_behaviours/<behavior_id>/test_cases.json
        into a single global test_cases.json mapping behavior_id -> [prompts]
        """
        print(f"[DEBUG] Ejecutando merge_test_cases para: {save_dir}") 
        merged_cases = {}

        individual_dir = os.path.join(save_dir, "test_cases_individual_behaviors")
        if not os.path.exists(individual_dir):
            print(f"[❌] ERROR: No existe la carpeta: {individual_dir}")
            return

        for behavior_name in os.listdir(individual_dir):
            behavior_path = os.path.join(individual_dir, behavior_name, "test_cases.json")
            if os.path.exists(behavior_path):
                with open(behavior_path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        if not isinstance(data, dict) or len(data) != 1:
                            continue

                        behavior_id, prompts = next(iter(data.items()))
                        if not isinstance(prompts, list):
                            continue

                        # Añadir y eliminar duplicados
                        if behavior_id not in merged_cases:
                            merged_cases[behavior_id] = []
                        merged_cases[behavior_id].extend(prompts)
                        merged_cases[behavior_id] = list(set(merged_cases[behavior_id]))

                    except json.JSONDecodeError as e:
                        print(f"[!] Skipping {behavior_path} due to JSON error: {e}")

        output_path = os.path.join(save_dir, "test_cases.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged_cases, f, indent=2)

        total = sum(len(v) for v in merged_cases.values())
        print(f"[✔] Merged {len(merged_cases)} behaviors ({total} prompts) into: {output_path}")


# ====== Helper Functions ===== #

### Extract list of numbers: 1. .... into list
def extract_numbered_lines(text):
    if text is None:
        return []
    lines = text.split("\n")
    numbered_lines = set([re.sub(r'^\s*\d+\.\s*"?(.*?)"?\s*$', r'\1', line) for line in lines if re.match(r'^\s*\d', line)])
    numbered_lines = [l.replace("Request: ", "").strip() for l in numbered_lines]
    numbered_lines = [re.sub(r'^\d+\.\s*', '', l) for l in numbered_lines] # some lines still have `{number}. ...`
    # Filter out empty strings
    numbered_lines = [l for l in numbered_lines if l.strip()]
    return numbered_lines