# AdaptLLM Finance-LLM Implementation

This repository contains a complete implementation for using the [AdaptLLM/finance-LLM](https://huggingface.co/AdaptLLM/finance-LLM) model, a specialized language model for finance domain knowledge.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Demo

```bash
python finance_llm_demo.py
```

### 3. Or Use the Simple Example

```bash
python simple_usage.py
```

## 📁 Files Overview

- **`requirements.txt`** - All necessary Python dependencies
- **`finance_llm_demo.py`** - Comprehensive demo with interactive chat
- **`simple_usage.py`** - Minimal example for quick testing
- **`README.md`** - This documentation file

## 🔧 Features

### FinanceLLMDemo Class

- **Automatic device detection** (CUDA/CPU)
- **Interactive chat interface**
- **Pre-built examples** for finance questions
- **Configurable generation parameters**
- **Error handling and logging**

### Key Capabilities

- ✅ Load and run the 6.74B parameter finance model
- ✅ Handle financial document analysis
- ✅ Answer finance terminology questions
- ✅ Interactive conversation mode
- ✅ Optimized for both GPU and CPU usage

## 💡 Usage Examples

### Basic Usage

```python
from finance_llm_demo import FinanceLLMDemo

# Initialize the model
demo = FinanceLLMDemo()

# Ask a finance question
response = demo.generate_response("What is a derivative?")
print(response)
```

### Interactive Chat

```python
# Start interactive chat
demo.interactive_chat()
```

### Custom Parameters

```python
response = demo.generate_response(
    prompt="Explain compound interest",
    max_length=300,
    temperature=0.8,
    top_p=0.9
)
```

## 🎯 Model Information

- **Model**: AdaptLLM/finance-LLM
- **Base**: LLaMA-1-7B
- **Domain**: Finance
- **Parameters**: 6.74B
- **Performance**: Competes with BloombergGPT-50B

## ⚠️ Important Notes

1. **First Run**: The model will be downloaded (~13GB) on first use
2. **Memory**: Requires at least 16GB RAM for CPU usage
3. **GPU**: CUDA support for faster inference
4. **Internet**: Required for initial model download

## 🐛 Troubleshooting

### Common Issues

**Out of Memory Error**

- Use CPU mode: `demo = FinanceLLMDemo(device="cpu")`
- Reduce `max_length` parameter

**Model Loading Error**

- Check internet connection
- Verify sufficient disk space (~13GB)
- Ensure all dependencies are installed

**Slow Performance**

- Use GPU if available
- Reduce `max_length` for faster responses
- Lower `temperature` for more focused answers

## 🔗 References

- [Original Model](https://huggingface.co/AdaptLLM/finance-LLM)
- [Research Paper](https://openreview.net/forum?id=y886UXPEZ0)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)

## 📝 License

This implementation is provided as-is for educational and research purposes. The underlying model follows the license of the original LLaMA model.

---

**Happy Finance AI-ing! 🏦🤖**
# poc-lama-1-7b
