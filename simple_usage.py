#!/usr/bin/env python3
"""
Simple usage example for AdaptLLM/finance-LLM
Colab-friendly script with auto dependency install and GPU/CPU handling.
"""

import sys
import subprocess
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
    ]
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + required)


def simple_finance_llm(question: Optional[str] = None) -> None:
    """Run a simple prompt against AdaptLLM/finance-LLM and print the response."""

    _ensure_deps_installed()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading model on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        "AdaptLLM/finance-LLM",
        use_fast=False,
        legacy=False,
    )

    # Avoid pad token issues during generation
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "AdaptLLM/finance-LLM",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if device == "cuda" else None,
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

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Slice off the prompt to get only generated tokens
    prompt_length = inputs["input_ids"].shape[-1]
    generated = outputs[0][prompt_length:]
    response = tokenizer.decode(generated, skip_special_tokens=True)

    print(f"Question: {question}")
    print(f"Answer: {response.strip()}")


if __name__ == "__main__":
    simple_finance_llm()
