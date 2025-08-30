import torch
from openai import OpenAI
from fastchat.model import load_model, get_conversation_template
import logging
import time
import concurrent.futures
from vllm import LLM as vllm
from vllm import SamplingParams
try:
    import google.generativeai as palm
except ImportError:
    palm = None
try:
    from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
except ImportError:
    Anthropic = None
    HUMAN_PROMPT = None
    AI_PROMPT = None
from baselines import get_template, load_model_and_tokenizer, load_vllm_model
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLM:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def generate(self, prompt):
        raise NotImplementedError("LLM must implement generate method.")

    def predict(self, sequences):
        raise NotImplementedError("LLM must implement predict method.")


class LocalLLM(LLM):
    def __init__(self,
                 model_path,
                 device='cuda',
                 num_gpus=1,
                 max_gpu_memory=None,
                 dtype=torch.float16,
                 load_8bit=False,
                 cpu_offloading=False,
                 revision=None,
                 debug=False,
                 system_message=None
                 ):
        super().__init__()

        self.model, self.tokenizer = self.create_model(
            model_path,
            device,
            num_gpus,
            max_gpu_memory,
            dtype,
            load_8bit,
            cpu_offloading,
            revision=revision,
            debug=debug,
        )
        self.model_path = model_path

        if system_message is None and 'Llama-2' in model_path:
            # monkey patch for latest FastChat to use llama2's official system message
            self.system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. " \
            "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. " \
            "Please ensure that your responses are socially unbiased and positive in nature.\n\n" \
            "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. " \
            "If you don't know the answer to a question, please don't share false information."
        else:
            self.system_message = system_message

    @torch.inference_mode()
    def create_model(self, model_path,
                     device='cuda',
                     num_gpus=1,
                     max_gpu_memory=None,
                     dtype=torch.float16,
                     load_8bit=False,
                     cpu_offloading=False,
                     revision=None,
                     debug=False):
        model, tokenizer = load_model(
            model_path,
            device,
            num_gpus,
            max_gpu_memory,
            dtype,
            load_8bit,
            cpu_offloading,
            revision=revision,
            debug=debug,
        )

        return model, tokenizer

    def set_system_message(self, conv_temp):
        if self.system_message is not None:
            conv_temp.set_system_message(self.system_message)

    @torch.inference_mode()
    def generate(self, prompt, temperature=0.01, max_tokens=512, repetition_penalty=1.0):
        conv_temp = get_conversation_template(self.model_path)
        self.set_system_message(conv_temp)

        conv_temp.append_message(conv_temp.roles[0], prompt)
        conv_temp.append_message(conv_temp.roles[1], None)

        prompt_input = conv_temp.get_prompt()
        input_ids = self.tokenizer([prompt_input]).input_ids
        output_ids = self.model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=False,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_tokens
        )

        if self.model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(input_ids[0]):]

        return self.tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )

    @torch.inference_mode()
    def generate_batch(self, prompts, temperature=0.01, max_tokens=512, repetition_penalty=1.0, batch_size=16):
        prompt_inputs = []
        for prompt in prompts:
            conv_temp = get_conversation_template(self.model_path)
            self.set_system_message(conv_temp)

            conv_temp.append_message(conv_temp.roles[0], prompt)
            conv_temp.append_message(conv_temp.roles[1], None)

            prompt_input = conv_temp.get_prompt()
            prompt_inputs.append(prompt_input)

        if self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        input_ids = self.tokenizer(prompt_inputs, padding=True).input_ids
        # load the input_ids batch by batch to avoid OOM
        outputs = []
        for i in range(0, len(input_ids), batch_size):
            output_ids = self.model.generate(
                torch.as_tensor(input_ids[i:i+batch_size]).cuda(),
                do_sample=False,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_tokens,
            )
            output_ids = output_ids[:, len(input_ids[0]):]
            outputs.extend(self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True, spaces_between_special_tokens=False))
        return outputs


class LocalVLLMOriginal(LLM):
    def __init__(self,
                 model_path,
                 gpu_memory_utilization=0.6,
                 system_message=None
                 ):
        super().__init__()
        self.model_path = model_path
        self.model = vllm(
            self.model_path, trust_remote_code=True, gpu_memory_utilization=gpu_memory_utilization)
        
        if system_message is None and 'Llama-2' in model_path:
            # monkey patch for latest FastChat to use llama2's official system message
            self.system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. " \
            "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. " \
            "Please ensure that your responses are socially unbiased and positive in nature.\n\n" \
            "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. " \
            "If you don't know the answer to a question, please don't share false information."
        else:
            self.system_message = system_message

    def set_system_message(self, conv_temp):
        if self.system_message is not None:
            conv_temp.set_system_message(self.system_message)

    def generate(self, prompt, temperature=0, max_tokens=512):
        prompts = [prompt]
        return self.generate_batch(prompts, temperature, max_tokens)

    def generate_batch(self, prompts, temperature=0, max_tokens=512):
        prompt_inputs = []
        for prompt in prompts:
            conv_temp = get_conversation_template(self.model_path)
            self.set_system_message(conv_temp)

            conv_temp.append_message(conv_temp.roles[0], prompt)
            conv_temp.append_message(conv_temp.roles[1], None)

            prompt_input = conv_temp.get_prompt()
            prompt_inputs.append(prompt_input)

        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        results = self.model.generate(
            prompt_inputs, sampling_params, use_tqdm=False)
        outputs = []
        for result in results:
            outputs.append(result.outputs[0].text)
        return outputs


class LocalVLLM(LLM):
    def __init__(self, model_config):
        super().__init__()
        self.model = load_vllm_model(**model_config)
        self.template = get_template(model_config['model_name_or_path'], chat_template=model_config.get('chat_template', None))

    def generate(self, prompt, temperature=0, max_tokens=512):
        prompts = [prompt]
        return self.generate_batch(prompts, temperature, max_tokens)

    def generate_batch(self, prompts, temperature=0, max_tokens=512):
        sp = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        generation_kwargs = dict(sampling_params=sp, use_tqdm=False)
        inputs = [self.template['prompt'].format(instruction=s if s is not None else "") for s in prompts]
        outputs = self.model.generate(inputs, **generation_kwargs)
        generations = []
        for o in outputs:
            text = o.outputs[0].text if o.outputs[0].text is not None else ""
            generations.append(text.strip())
        return generations


class BardLLM(LLM):
    def generate(self, prompt):
        return

class PaLM2LLM(LLM):
    def __init__(self,
                 model_path='chat-bison-001',
                 api_key=None,
                 system_message=None
                ):
        super().__init__()
        
        if palm is None:
            raise ImportError("google.generativeai is not installed. Please install it with: pip install google-generativeai")
        
        if len(api_key) != 39:
            raise ValueError('invalid PaLM2 API key')
        
        palm.configure(api_key=api_key)
        available_models = [m for m in palm.list_models()]
        for model in available_models:
            if model.name == model_path:
                self.model_path = model
                break
        self.system_message = system_message
        # The PaLM-2 has a great rescriction on the number of input tokens, so I will release the short jailbreak prompts later
        
    def generate(self, prompt, temperature=0, n=1, max_trials=1, failure_sleep_time=1):
        for _ in range(max_trials):
            try:
                results = palm.chat(
                    model=self.model_path,
                    prompt=prompt,
                    temperature=temperature,
                    candidate_count=n,
                )
                return [results.candidates[i]['content'] for i in range(n)]
            except Exception as e:
                logging.warning(
                    f"PaLM2 API call failed due to {e}. Retrying {_+1} / {max_trials} times...")
                time.sleep(failure_sleep_time)

        return [" " for _ in range(n)]
    
    def generate_batch(self, prompts, temperature=0, n=1, max_trials=1, failure_sleep_time=1):
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.generate, prompt, temperature, n,
                                       max_trials, failure_sleep_time): prompt for prompt in prompts}
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        return results

class ClaudeLLM(LLM):
    def __init__(self,
                 model_path='claude-instant-1.2',
                 api_key=None
                ):
        super().__init__()
        
        if Anthropic is None:
            raise ImportError("anthropic is not installed. Please install it with: pip install anthropic")
        
        if len(api_key) != 108:
            raise ValueError('invalid Claude API key')
        
        self.model_path = model_path
        self.api_key = api_key
        self.anthropic = Anthropic(
            api_key=self.api_key
        )

    def generate(self, prompt, max_tokens=512, max_trials=1, failure_sleep_time=1):
        
        for _ in range(max_trials):
            try:
                completion = self.anthropic.completions.create(
                    model=self.model_path,
                    max_tokens_to_sample=300,
                    prompt=f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}",
                )
                return [completion.completion]
            except Exception as e:
                logging.warning(
                    f"Claude API call failed due to {e}. Retrying {_+1} / {max_trials} times...")
                time.sleep(failure_sleep_time)

        return [" "]
    
    def generate_batch(self, prompts, max_tokens=512, max_trials=1, failure_sleep_time=1):
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.generate, prompt, max_tokens,
                                       max_trials, failure_sleep_time): prompt for prompt in prompts}
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        return results

class MutationLLM(LLM):
    """Enhanced LocalLLM implementation specifically for GPTFuzz mutation models."""
    
    def __init__(self,
                 model_path,
                 device='cuda',
                 num_gpus=1,
                 max_gpu_memory=None,
                 dtype=torch.float16,  # Default to float16 for V100 compatibility
                 load_8bit=False,
                 cpu_offloading=False,
                 revision=None,
                 debug=False,
                 system_message=None
                 ):
        super().__init__()
        
        # Popular models mapping for easy selection
        self.popular_models = {
            'llama3-8b': 'meta-llama/Meta-Llama-3-8B-Instruct',
            'llama3-70b': 'meta-llama/Meta-Llama-3-70B-Instruct',
            'llama3.1-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
            'llama3.1-70b': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
            'llama2-7b': 'meta-llama/Llama-2-7b-chat-hf',
            'llama2-13b': 'meta-llama/Llama-2-13b-chat-hf',
            'mixtral-8x7b': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
            'mixtral-8x22b': 'mistralai/Mixtral-8x22B-Instruct-v0.1',
            'codellama-7b': 'codellama/CodeLlama-7b-Instruct-hf',
            'codellama-13b': 'codellama/CodeLlama-13b-Instruct-hf',
            'vicuna-7b': 'lmsys/vicuna-7b-v1.5',
            'vicuna-13b': 'lmsys/vicuna-13b-v1.5',
            'openchat-3.5': 'openchat/openchat-3.5-1210',
            'openchat_3_5_1210': 'openchat/openchat-3.5-1210',
            'zephyr-7b': 'HuggingFaceH4/zephyr-7b-beta'
        }
        
        # Store original model path for reference
        self.original_model_path = model_path
        
        # Resolve model path if it's a popular model name
        if model_path in self.popular_models:
            resolved_model_path = self.popular_models[model_path]
            print(f"🔄 Resolving {model_path} -> {resolved_model_path}")
            model_path = resolved_model_path
        
        # Auto-detect V100 compatibility and adjust dtype if needed
        if device == 'cuda' and dtype == torch.bfloat16:
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    compute_caps = [float(cap.strip()) for cap in result.stdout.split('\n') if cap.strip()]
                    if any(cap < 8.0 for cap in compute_caps):
                        print(f"⚠️  GPU compute capability {min(compute_caps)} < 8.0, switching from bfloat16 to float16")
                        dtype = torch.float16
            except:
                # If we can't detect GPU info, default to float16 for safety
                print("⚠️  Could not detect GPU compute capability, defaulting to float16")
                dtype = torch.float16
        print(f"Using dtype: {dtype}")    
        self.model, self.tokenizer = self.create_model(
            model_path,
            device,
            num_gpus,
            max_gpu_memory,
            dtype,
            load_8bit,
            cpu_offloading,
            revision=revision,
            debug=debug,
        )
        self.model_path = model_path
        
        # Default system message for mutation tasks
        if system_message is None:
            self.system_message = "You are a helpful AI assistant specialized in text transformation and mutation. Follow instructions precisely."
        else:
            self.system_message = system_message
            
    @staticmethod
    def get_popular_models():
        """Get list of popular models for CLI selection."""
        return {
            'llama3-8b': 'Llama 3 8B Instruct',
            'llama3-70b': 'Llama 3 70B Instruct',
            'llama3.1-8b': 'Llama 3.1 8B Instruct',
            'llama3.1-70b': 'Llama 3.1 70B Instruct',
            'mixtral-8x7b': 'Mixtral 8x7B Instruct',
            'mixtral-8x22b': 'Mixtral 8x22B Instruct',
            'codellama-7b': 'Code Llama 7B Instruct',
            'codellama-13b': 'Code Llama 13B Instruct',
            'vicuna-7b': 'Vicuna 7B v1.5',
            'vicuna-13b': 'Vicuna 13B v1.5',
            'openchat-3.5': 'OpenChat 3.5',
            'openchat_3_5_1210': 'OpenChat 3.5 1210',
            'zephyr-7b': 'Zephyr 7B Beta'
        }
    
    @staticmethod
    def download_model(model_name):
        """Download model from HuggingFace."""
        try:
            print(f"🔄 Downloading {model_name} from HuggingFace...")
            
            # Download tokenizer first
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print(f"✅ Tokenizer downloaded successfully")
            
            # Download model
            model = AutoModelForCausalLM.from_pretrained(model_name)
            print(f"✅ Model downloaded successfully")
            
            return True
        except Exception as e:
            print(f"❌ Error downloading {model_name}: {e}")
            return False
    
    @torch.inference_mode()
    def create_model(self, model_path,
                     device='cuda',
                     num_gpus=1,
                     max_gpu_memory=None,
                     dtype=torch.float16,
                     load_8bit=False,
                     cpu_offloading=False,
                     revision=None,
                     debug=False):
        """Create and load the model with specified configurations using standard transformers."""
        print(f"🔄 Creating model with path: {model_path} and dtype: {dtype}")
        print(f"📝 Using standard transformers instead of FastChat to avoid compression issues")
        
        try:
            # Use standard transformers loading instead of FastChat
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch.nn as nn
            
            # Load tokenizer
            print(f"🔤 Loading tokenizer from {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                revision=revision
            )

            # --- FIX: ensure pad_token ---
            if tokenizer.pad_token is None:
                # Use EOS as PAD for decoder-only models
                eos = getattr(tokenizer, "eos_token", None)
                if eos is not None:
                    tokenizer.pad_token = eos
                else:
                    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                    tokenizer.pad_token = "[PAD]"

            # Optional: for causal attention models, left padding works better
            tokenizer.padding_side = "left"
            
            # Determine device placement
            if device == 'cuda' and torch.cuda.is_available():
                device_map = "auto" if num_gpus > 1 else "cuda:0"
                print(f"🎮 Using GPU device map: {device_map}")
            else:
                device_map = "cpu"
                print(f"🖥️  Using CPU device")
                
            # Configure memory limits
            max_memory_config = None
            if max_gpu_memory and device != 'cpu':
                max_memory_config = {0: max_gpu_memory}
                print(f"💾 GPU memory limit: {max_gpu_memory}")
                
            # Load model with memory-efficient settings
            print(f"🤖 Loading model from {model_path}")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device_map,
                max_memory=max_memory_config,
                trust_remote_code=True,
                revision=revision,
                low_cpu_mem_usage=True,
                load_in_8bit=load_8bit,
                offload_folder="./offload" if cpu_offloading else None,
            )
            
            # Additional memory optimizations
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                print(f"✅ Enabled gradient checkpointing for memory efficiency")
                
            print(f"✅ Model loaded successfully!")
            print(f"📊 Model dtype: {model.dtype}")
            print(f"🎯 Model device: {next(model.parameters()).device}")
            
            # Clear cache after loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print(f"🧹 Cleared CUDA cache after model loading")
            
            return model, tokenizer
            
        except Exception as e:
            print(f"❌ Error loading model with transformers: {e}")
            print(f"🔄 Attempting fallback loading method...")
            
            # Fallback: try with basic settings
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,  # Force float16 for compatibility
                    device_map="auto" if torch.cuda.is_available() else "cpu",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                
                print(f"✅ Fallback loading succeeded!")
                return model, tokenizer
                
            except Exception as fallback_error:
                print(f"❌ Fallback also failed: {fallback_error}")
                raise RuntimeError(f"Failed to load model {model_path} with both primary and fallback methods: {e}")                
    
    def set_system_message(self, system_message=None):
        """Set or update the system message for this mutation model."""
        if system_message is not None:
            self.system_message = system_message
        elif hasattr(self, 'system_message') and self.system_message is None:
            self.system_message = "You are a helpful AI assistant specialized in text transformation and mutation. Follow instructions precisely."

    @torch.inference_mode()
    def generate(self, prompt, temperature=0.01, max_tokens=512, repetition_penalty=1.0):
        """Generate response using standard transformers instead of FastChat conversation templates."""
        try:
            # Create a simple chat template instead of using FastChat
            if hasattr(self, 'system_message') and self.system_message:
                full_prompt = f"System: {self.system_message}\n\nUser: {prompt}\n\nAssistant:"
            else:
                full_prompt = f"User: {prompt}\n\nAssistant:"
            
            # Tokenize input
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048  # Limit input length
            )
            
            # Move to appropriate device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate with memory management
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else 0.01,
                    max_new_tokens=max_tokens,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=False,  # Save memory
                )
            
            # Decode only the new tokens (excluding input)
            input_length = inputs['input_ids'].shape[1]
            new_tokens = output_ids[0][input_length:]
            
            decoded_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            generated_text = decoded_text.strip() if decoded_text is not None else ""
            
            # Clear cache after generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return generated_text
            
        except Exception as e:
            print(f"❌ Error in generate: {e}")
            print(f"🔄 Attempting simplified generation...")
            
            # Fallback to very simple generation
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=min(max_tokens, 256),
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        use_cache=False,
                    )
                
                input_length = inputs['input_ids'].shape[1]
                decoded_text = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
                generated_text = decoded_text.strip() if decoded_text is not None else ""
                
                return generated_text
                
            except Exception as fallback_error:
                print(f"❌ Fallback generation also failed: {fallback_error}")
                return f"Error: Could not generate response - {str(e)}"

    @torch.inference_mode()
    def generate_batch(self, prompts, temperature=0.01, max_tokens=512, repetition_penalty=1.0, batch_size=4):
        """Generate batch responses using standard transformers with memory management."""
        try:
            # Reduce batch size for memory efficiency
            batch_size = min(batch_size, 4, len(prompts))
            
            # Format prompts with simple chat template
            prompt_inputs = []
            for prompt in prompts:
                if hasattr(self, 'system_message') and self.system_message:
                    full_prompt = f"System: {self.system_message}\n\nUser: {prompt}\n\nAssistant:"
                else:
                    full_prompt = f"User: {prompt}\n\nAssistant:"
                prompt_inputs.append(full_prompt)
            
            # Ensure tokenizer has pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"🔧 Set pad_token to eos_token for batch processing")
            
            self.tokenizer.padding_side = "left"  # Left padding for batch generation
            
            all_outputs = []
            device = next(self.model.parameters()).device
            
            # Process in smaller batches to avoid OOM
            for i in range(0, len(prompt_inputs), batch_size):
                batch_prompts = prompt_inputs[i:i+batch_size]
                
                # Tokenize batch with truncation
                inputs = self.tokenizer(
                    batch_prompts, 
                    padding=True, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=1024  # Conservative limit
                )
                
                # Move to device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                input_length = inputs['input_ids'].shape[1]
                
                # Generate batch
                with torch.no_grad():
                    try:
                        output_ids = self.model.generate(
                            **inputs,
                            do_sample=temperature > 0,
                            temperature=temperature if temperature > 0 else 0.01,
                            max_new_tokens=min(max_tokens, 256),  # Conservative limit
                            repetition_penalty=repetition_penalty,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            use_cache=False,  # Save memory
                        )
                        
                        # Decode only new tokens for each sequence
                        batch_outputs = []
                        for j, output in enumerate(output_ids):
                            # Get the length of the specific input (accounting for different lengths due to padding)
                            actual_input_length = (inputs['attention_mask'][j] == 1).sum().item()
                            new_tokens = output[actual_input_length:]
                            decoded_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                            generated_text = decoded_text.strip() if decoded_text is not None else ""
                            batch_outputs.append(generated_text)
                        
                        all_outputs.extend(batch_outputs)
                        
                    except torch.cuda.OutOfMemoryError:
                        print(f"⚠️  CUDA OOM in batch {i//batch_size + 1}, falling back to single generation...")
                        # Fallback: generate one by one
                        for single_prompt in batch_prompts:
                            single_output = self.generate(single_prompt, temperature, max_tokens, repetition_penalty)
                            all_outputs.append(single_output)
                    
                    finally:
                        # Clear memory after each batch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
            
            return all_outputs
            
        except Exception as e:
            print(f"❌ Error in batch generation: {e}")
            print(f"🔄 Falling back to individual generation...")
            
            # Ultimate fallback: generate each prompt individually
            fallback_outputs = []
            for prompt in prompts:
                output = self.generate(prompt, temperature, max_tokens, repetition_penalty)
                fallback_outputs.append(output)
            
            return fallback_outputs


class OpenAILLM(LLM):
    def __init__(self,
                 model_path,
                 api_key=None,
                 system_message=None
                ):
        super().__init__()

        if not api_key.startswith('sk-'):
            raise ValueError('OpenAI API key should start with sk-')
        if model_path not in ['gpt-3.5-turbo', 'gpt-4']:
            raise ValueError(
                'OpenAI model path should be gpt-3.5-turbo or gpt-4')
        self.client = OpenAI(api_key = api_key)
        self.model_path = model_path
        self.system_message = system_message if system_message is not None else "You are a helpful assistant."

    def generate(self, prompt, temperature=0, max_tokens=512, n=1, max_trials=10, failure_sleep_time=5):
        for _ in range(max_trials):
            try:
                results = self.client.chat.completions.create(
                    model=self.model_path,
                    messages=[
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                )
                return [results.choices[i].message.content for i in range(n)]
            except Exception as e:
                logging.warning(
                    f"OpenAI API call failed due to {e}. Retrying {_+1} / {max_trials} times...")
                time.sleep(failure_sleep_time)

        return [" " for _ in range(n)]

    def generate_batch(self, prompts, temperature=0, max_tokens=512, n=1, max_trials=10, failure_sleep_time=5):
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.generate, prompt, temperature, max_tokens, n,
                                       max_trials, failure_sleep_time): prompt for prompt in prompts}
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        return results
