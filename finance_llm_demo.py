#!/usr/bin/env python3
"""
AdaptLLM Finance-LLM Demo
A comprehensive demonstration of how to use the AdaptLLM/finance-LLM model
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import warnings
warnings.filterwarnings("ignore")

class FinanceLLMDemo:
    def __init__(self, model_name="AdaptLLM/finance-LLM", device=None):
        """
        Initialize the Finance LLM Demo
        
        Args:
            model_name (str): Hugging Face model name
            device (str): Device to run the model on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🚀 Initializing {model_name}")
        print(f"📱 Using device: {self.device}")
        
        # Load model and tokenizer
        self._load_model()
        
    def _load_model(self):
        """Load the model and tokenizer"""
        try:
            print("📥 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                use_fast=False,
                trust_remote_code=True
            )
            
            print("📥 Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
                
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def generate_response(self, prompt, max_length=512, temperature=0.7, top_p=0.9):
        """
        Generate a response from the model
        
        Args:
            prompt (str): Input prompt
            max_length (int): Maximum length of generated text
            temperature (float): Sampling temperature
            top_p (float): Top-p sampling parameter
            
        Returns:
            str: Generated response
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                add_special_tokens=False
            ).input_ids.to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            answer_start = int(inputs.shape[-1])
            response = self.tokenizer.decode(
                outputs[answer_start:], 
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return f"Error: {e}"
    
    def interactive_chat(self):
        """Interactive chat interface"""
        print("\n💬 Interactive Finance LLM Chat")
        print("Type 'quit' to exit, 'clear' to clear conversation")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'clear':
                    print("🧹 Conversation cleared")
                    continue
                elif not user_input:
                    continue
                
                print("🤖 AI: Generating response...")
                start_time = time.time()
                
                response = self.generate_response(user_input)
                
                end_time = time.time()
                print(f"🤖 AI: {response}")
                print(f"⏱️  Response time: {end_time - start_time:.2f}s")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def run_demo_examples():
    """Run predefined demo examples"""
    print("🎯 Finance LLM Demo Examples")
    print("=" * 50)
    
    # Initialize the model
    demo = FinanceLLMDemo()
    
    # Example 1: Basic financial question
    print("\n📊 Example 1: Basic Financial Question")
    print("-" * 40)
    prompt1 = "What is the difference between a stock and a bond?"
    print(f"👤 Question: {prompt1}")
    
    response1 = demo.generate_response(prompt1)
    print(f"🤖 Answer: {response1}")
    
    # Example 2: Financial document analysis
    print("\n📄 Example 2: Financial Document Analysis")
    print("-" * 40)
    prompt2 = """Use this fact to answer the question: 
    Title of each class Trading Symbol(s) Name of each exchange on which registered
    Common Stock, Par Value $.01 Per Share MMM New York Stock Exchange
    MMM Chicago Stock Exchange, Inc.
    1.500% Notes due 2026 MMM26 New York Stock Exchange
    1.750% Notes due 2030 MMM30 New York Stock Exchange
    1.500% Notes due 2031 MMM31 New York Stock Exchange

    Which debt securities are registered to trade on a national securities exchange under 3M's name as of Q2 of 2023?"""
    
    print(f"👤 Question: {prompt2}")
    response2 = demo.generate_response(prompt2, max_length=256)
    print(f"🤖 Answer: {response2}")
    
    # Example 3: Financial terminology
    print("\n💼 Example 3: Financial Terminology")
    print("-" * 40)
    prompt3 = "Explain what a derivative is in simple terms."
    print(f"👤 Question: {prompt3}")
    
    response3 = demo.generate_response(prompt3)
    print(f"🤖 Answer: {response3}")
    
    return demo

def main():
    """Main function"""
    print("🏦 Welcome to AdaptLLM Finance-LLM Demo!")
    print("=" * 50)
    
    try:
        # Run demo examples
        demo = run_demo_examples()
        
        # Ask if user wants interactive chat
        print("\n" + "=" * 50)
        chat_choice = input("Would you like to start an interactive chat? (y/n): ").strip().lower()
        
        if chat_choice in ['y', 'yes']:
            demo.interactive_chat()
        else:
            print("👋 Thanks for using the demo!")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("💡 Make sure you have the required dependencies installed:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()
