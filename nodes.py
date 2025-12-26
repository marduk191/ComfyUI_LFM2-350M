import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import folder_paths

class LFM2Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "repo_id": ("STRING", {"default": "marduk191/lfm2-350m-dp-marduk191"}),
                "precision": (["bf16", "fp16", "fp32", "auto"], {"default": "bf16"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            },
            "optional": {
                "local_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("LFM2_MODEL", "LFM2_TOKENIZER")
    RETURN_NAMES = ("model", "tokenizer")
    FUNCTION = "load_model"
    CATEGORY = "LiquidAI/LFM2"

    def load_model(self, repo_id, precision, device, local_path=""):
        path_to_load = repo_id
        if local_path and os.path.exists(local_path):
            print(f"Loading LFM2 model from local path: {local_path}")
            path_to_load = local_path
        else:
            print(f"Loading LFM2 model from Hugging Face: {repo_id}")

        dtype = torch.bfloat16
        if precision == "fp16":
            dtype = torch.float16
        elif precision == "fp32":
            dtype = torch.float32
        elif precision == "auto":
            dtype = "auto"

        try:
            tokenizer = AutoTokenizer.from_pretrained(path_to_load, trust_remote_code=True)
            
            # Check if this is a local path with potentially compiled weights
            print(f"LFM2 Debug - local_path arg: '{local_path}'")
            print(f"LFM2 Debug - path_to_load: '{path_to_load}'")
            print(f"LFM2 Debug - local_path exists: {os.path.exists(local_path) if local_path else 'N/A'}")
            
            if local_path and os.path.exists(local_path):
                from safetensors import safe_open
                import glob
                
                # Find safetensors files
                safetensor_files = glob.glob(os.path.join(path_to_load, "*.safetensors"))
                print(f"LFM2 Debug - Found safetensor files: {safetensor_files}")
                
                if safetensor_files:
                    # Check for _orig_mod in keys (just for logging)
                    with safe_open(safetensor_files[0], framework="pt") as f:
                        keys = list(f.keys())
                        print(f"LFM2 Debug - First 5 weight keys: {keys[:5]}")
                    
                    # ALWAYS apply the fix for local models - strip _orig_mod if present
                    print("Loading local model with weight key correction...")
                    
                    # Load model structure first (will have random weights)
                    model = AutoModelForCausalLM.from_pretrained(
                        path_to_load,
                        torch_dtype=dtype,
                        device_map="cpu",  # Load to CPU first
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                    
                    # Load and fix state dict
                    from safetensors.torch import load_file
                    state_dict = {}
                    for file in safetensor_files:
                        sd = load_file(file)
                        for k, v in sd.items():
                            # Strip _orig_mod. from anywhere in the key
                            new_key = k.replace("_orig_mod.", "")
                            state_dict[new_key] = v
                            
                    print(f"LFM2 Debug - First 5 corrected keys: {list(state_dict.keys())[:5]}")
                    
                    # Load corrected weights
                    print(f"LFM2 Debug - Loading {len(state_dict)} weight keys...")
                    result = model.load_state_dict(state_dict, strict=False)
                    print(f"LFM2 Debug - Missing keys count: {len(result.missing_keys)}")
                    print(f"LFM2 Debug - Unexpected keys count: {len(result.unexpected_keys)}")
                    if result.missing_keys:
                        print(f"LFM2 Debug - First 3 missing: {result.missing_keys[:3]}")
                    if result.unexpected_keys:
                        print(f"LFM2 Debug - First 3 unexpected: {result.unexpected_keys[:3]}")
                    model = model.to(device)
                    print("Successfully loaded model with weight key correction.")
                else:
                    # No safetensors, use normal loading
                    model = AutoModelForCausalLM.from_pretrained(
                        path_to_load,
                        torch_dtype=dtype,
                        device_map=device,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
            else:
                # Loading from HF - check if weights need _orig_mod fix
                print("Loading from HuggingFace, checking for weight key issues...")
                
                # First load normally
                model = AutoModelForCausalLM.from_pretrained(
                    path_to_load,
                    torch_dtype=dtype,
                    device_map="cpu",  # Load to CPU first to check weights
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
                # Check if model weights seem uninitialized (all weights have similar small values)
                # by trying to load and fix from the HF hub directly
                from huggingface_hub import hf_hub_download
                import os
                
                try:
                    # Download the safetensors file
                    safetensor_path = hf_hub_download(repo_id=path_to_load, filename="model.safetensors")
                    print(f"LFM2 Debug - Downloaded safetensors to: {safetensor_path}")
                    
                    from safetensors import safe_open
                    from safetensors.torch import load_file
                    
                    with safe_open(safetensor_path, framework="pt") as f:
                        keys = list(f.keys())
                        print(f"LFM2 Debug - HF model first 5 keys: {keys[:5]}")
                        has_orig_mod = any("_orig_mod" in k for k in keys)
                    
                    if has_orig_mod:
                        print("Detected _orig_mod prefix in HF model, applying fix...")
                        state_dict = {}
                        sd = load_file(safetensor_path)
                        for k, v in sd.items():
                            new_key = k.replace("_orig_mod.", "")
                            state_dict[new_key] = v
                        
                        print(f"LFM2 Debug - First 5 corrected keys: {list(state_dict.keys())[:5]}")
                        result = model.load_state_dict(state_dict, strict=False)
                        print(f"LFM2 Debug - Missing keys: {len(result.missing_keys)}, Unexpected: {len(result.unexpected_keys)}")
                        
                except Exception as e:
                    print(f"LFM2 Debug - Could not check/fix HF weights: {e}")
                
                model = model.to(device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

        return ({"model": model, "device": device}, tokenizer)


class LFM2Generator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_context": ("LFM2_MODEL",),
                "tokenizer": ("LFM2_TOKENIZER",),
                "system_prompt": ("STRING", {"default": "Transform the following image description into a detailed prompt for Z-Image Turbo. Use rich details, lighting info, and camera specs. Do not include negative prompts or ending explanation starting with 'this detailed prompt should'", "multiline": True}),
                "prompt": ("STRING", {"default": "Hello, how are you?", "multiline": True}),
                "max_new_tokens": ("INT", {"default": 256, "min": 1, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 50, "min": 0, "max": 200}),
                "min_p": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 1.0, "max": 2.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"
    CATEGORY = "LiquidAI/LFM2"

    def generate(self, model_context, tokenizer, system_prompt, prompt, max_new_tokens, temperature, top_p, top_k, min_p, repetition_penalty):
        model = model_context["model"]
        device = model_context["device"]

        # Prepare messages
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            # Use tokenize=True to get IDs directly and avoid special token parsing issues
            inputs = tokenizer.apply_chat_template(
                messages, 
                tokenize=True, 
                add_generation_prompt=True, 
                return_tensors="pt",
                return_dict=True
            ).to(device)
            
            # If inputs is just a tensor (older transformers versions), wrap it
            if isinstance(inputs, torch.Tensor):
                 inputs = {"input_ids": inputs}
                 
            print(f"LFM2 Debug - Input Token IDs: {inputs['input_ids'][0].tolist()}")

        except Exception as e:
            print(f"LFM2 Debug - Chat Template Failed: {e}")
            # Fallback: simple encoding of prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
        print(f"LFM2 Debug - Input Token IDs: {inputs['input_ids'][0].tolist()}")
        print(f"LFM2 Debug - Special Tokens Map: {tokenizer.special_tokens_map}")
        
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print("LFM2 Debug - pad_token_id was None, set to eos_token_id")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode only the new tokens
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        print(f"LFM2 Debug - Generated Token IDs: {generated_ids.tolist()}")
        
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"LFM2 Debug - Output Text: {output_text}")

        return (output_text,)
