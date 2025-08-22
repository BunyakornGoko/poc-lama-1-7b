#!/usr/bin/env python3
"""
Simple usage example for AdaptLLM/finance-LLM
Colab-friendly script with memory optimization to prevent GPU OOM errors.
"""

import sys
import subprocess
import os
import gc
from typing import Optional


def _ensure_deps_installed() -> None:
    """Install required packages if they are missing (useful for Google Colab)."""
    required = [
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "accelerate>=0.20.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0",
        "numpy>=1.24.0",
        "safetensors>=0.4.0",
        "bitsandbytes>=0.41.0",  # For quantization
    ]
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + required)


def optimize_gpu_memory():
    """Set environment variables to optimize GPU memory usage."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def simple_finance_llm(question: Optional[str] = None, force_cpu: bool = False) -> None:
    """Run a simple prompt against AdaptLLM/finance-LLM with memory optimization."""

    _ensure_deps_installed()
    optimize_gpu_memory()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Clear any existing GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # Determine device and settings based on available memory
    if torch.cuda.is_available() and not force_cpu:
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🔍 GPU Memory: {gpu_memory_gb:.1f} GB")
        
        if gpu_memory_gb < 16:  # For T4/lower-end GPUs
            device = "cuda"
            dtype = torch.float16
            use_8bit = True
            max_tokens = 500  # Increased from 250
            print("⚡ Using 8-bit quantization for low GPU memory")
        else:  # For higher-end GPUs
            device = "cuda"
            dtype = torch.float16
            use_8bit = False
            max_tokens = 800  # Increased from 400
            print("🚀 Using full precision on high-memory GPU")
    else:
        device = "cpu"
        dtype = torch.float32
        use_8bit = False
        max_tokens = 600  # Increased from 300
        print("🖥️ Using CPU (slower but reliable)")

    print(f"Loading model on {device}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "AdaptLLM/finance-LLM",
            use_fast=False,
            legacy=False,
        )

        # Avoid pad token issues during generation
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        # Model loading with memory optimization
        model_kwargs = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        
        if device == "cuda":
            model_kwargs["device_map"] = "auto"
            
            # Add quantization for low memory GPUs
            if use_8bit:
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0,
                        llm_int8_has_fp16_weight=False,
                    )
                    model_kwargs["quantization_config"] = quantization_config
                except ImportError:
                    print("⚠️ BitsAndBytes not available, using 16-bit")

        model = AutoModelForCausalLM.from_pretrained(
            "AdaptLLM/finance-LLM",
            **model_kwargs
        )
        model.eval()

        # Default question if none provided
        if not question:
            question = "What is inflation?"

        prompt = question

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
            padding=False,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Clear cache before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        # Slice off the prompt to get only generated tokens
        prompt_length = inputs["input_ids"].shape[-1]
        generated = outputs[0][prompt_length:]
        response = tokenizer.decode(generated, skip_special_tokens=True)

        print(f"Question: {question}")
        print(f"Answer: {response.strip()}")
        
        # Clean up memory after generation
        del outputs, inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("❌ GPU out of memory!")
            if not force_cpu:
                print("🔄 Retrying with CPU...")
                return simple_finance_llm(question, force_cpu=True)
            else:
                print("💡 Try reducing max_tokens or using a smaller model")
                raise e
        else:
            raise e
    except Exception as e:
        print(f"❌ Error: {e}")
        if not force_cpu and torch.cuda.is_available():
            print("🔄 Retrying with CPU...")
            return simple_finance_llm(question, force_cpu=True)
        else:
            raise e


if __name__ == "__main__":
    simple_finance_llm()
