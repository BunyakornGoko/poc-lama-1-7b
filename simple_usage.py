#!/usr/bin/env python3
"""
Simple usage example for AdaptLLM/finance-LLM
Minimal code to get started quickly
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def simple_finance_llm():
    """Simple example of using the finance LLM"""
    
    # Load model and tokenizer
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained("AdaptLLM/finance-LLM")
    tokenizer = AutoTokenizer.from_pretrained("AdaptLLM/finance-LLM", use_fast=False)
    
    # Your finance question
    question = "What is a inflation?"
    
    # Prepare prompt
    prompt = question
    
    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    outputs = model.generate(input_ids=inputs.input_ids, max_length=200)
    
    # Get the response
    answer_start = int(inputs.input_ids.shape[-1])
    response = tokenizer.decode(outputs[0][answer_start:], skip_special_tokens=True)
    
    print(f"Question: {question}")
    print(f"Answer: {response}")

if __name__ == "__main__":
    simple_finance_llm()
