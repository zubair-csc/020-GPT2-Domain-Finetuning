# 🔥 GPT-2 Domain Fine-tuning - Advanced Language Model Customization

## 📋 Project Overview
GPT-2 Domain Fine-tuning is a comprehensive and production-ready implementation for fine-tuning GPT-2 language models on domain-specific text data using PyTorch and Transformers. Built with clean architecture and modular design, it includes custom dataset handling, multiple training strategies, real-time monitoring, text generation capabilities, and model evaluation tools. The system supports both Hugging Face Trainer and manual PyTorch training loops with extensive customization options for different domains and use cases.

## 🎯 Objectives
- Fine-tune pre-trained GPT-2 models on custom domain-specific text datasets
- Implement robust training pipelines with both automated and manual training options
- Support multiple data formats (CSV, TXT, JSON) with flexible preprocessing
- Provide comprehensive evaluation metrics including perplexity calculation
- Enable high-quality text generation with configurable sampling strategies
- Offer model comparison tools between original and fine-tuned versions

## 📊 Dataset Information
| Attribute | Details |
|-----------|---------|
| **Supported Formats** | CSV, TXT, JSON with flexible column mapping |
| **Input Processing** | Automatic tokenization, padding, and truncation |
| **Max Sequence Length** | Configurable (default: 512 tokens) |
| **Data Splitting** | Automatic train/validation split with stratification |
| **Preprocessing** | Text cleaning, encoding, and batch processing |

## 🔧 Technical Implementation

### 🔌 Model Architectures
- **Base Models**: GPT-2 (117M), GPT-2 Medium (345M), GPT-2 Large (762M), GPT-2 XL (1.5B)
- **Tokenization**: GPT-2 BPE tokenizer with proper padding and special tokens
- **Fine-tuning Strategy**: Causal language modeling with next-token prediction
- **Architecture**: Transformer decoder with attention mechanisms and positional encoding
- **Parameter Updates**: Full model fine-tuning or parameter-efficient methods

### 🧹 Data Preprocessing
**Preprocessing Pipeline:**
- Automatic text cleaning and normalization
- Tokenization with vocabulary mapping and special token handling
- Sequence truncation and padding for consistent batch sizes
- Train/validation splitting with reproducible random seeds
- Memory-efficient data loading with PyTorch DataLoader

### ⚙️ Training Architecture
**Fine-tuning Process:**
1. **Data Preparation**: Load and preprocess domain-specific text data
   - Text tokenization and encoding
   - Dataset splitting and validation setup
   - Batch creation with attention masks

2. **Model Training**: Fine-tune GPT-2 on domain data
   - Causal language modeling objective
   - AdamW optimizer with learning rate scheduling
   - Gradient accumulation for effective larger batch sizes
   - Regular validation and checkpoint saving

3. **Training Features**:
   - Cross-entropy loss for next-token prediction
   - Learning rate warmup and decay scheduling
   - Early stopping based on validation loss
   - Gradient clipping for training stability

### 🔍 Training Features
**Monitoring and Evaluation:**
- Real-time training and validation loss tracking
- Perplexity calculation for model quality assessment
- Text generation samples during training progress
- Training metrics logging and visualization

**Model Management:**
- Automatic model and tokenizer saving
- Load pre-trained fine-tuned models for inference
- Generate text with various sampling strategies
- Model comparison between original and fine-tuned versions

## 📊 Visualizations
- **Loss Curves**: Training and validation loss progression over epochs
- **Perplexity Tracking**: Model quality metrics throughout training
- **Text Generation Samples**: Quality assessment through generated examples
- **Model Comparison**: Side-by-side comparison of original vs fine-tuned outputs

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Required libraries: PyTorch, Transformers, pandas, numpy, matplotlib, scikit-learn

### Installation
1. Clone the repository:
```bash
git clone https://github.com/zubair-csc/020-GPT2-Domain-Finetuning.git
cd 020-GPT2-Domain-Finetuning
```

2. Install required libraries:
```bash
pip install torch transformers pandas numpy matplotlib scikit-learn tqdm
```

### Dataset Setup
Prepare your domain-specific text data:
- **CSV Format**: Include a 'text' column with your text data
- **TXT Format**: One text sample per line
- **JSON Format**: Array of objects with text field

### Running the Training
Execute the Python script:
```python
python gpt2_finetuning.py
```

This will:
- Load sample domain-specific data automatically
- Initialize GPT-2 model and tokenizer
- Train the model for specified epochs
- Display training progress and loss curves
- Save fine-tuned model and generate sample text

## 📈 Usage Examples

### Basic Training with Sample Data
```python
# Quick start with built-in sample data
simple_example()
```

### Custom Dataset Training
```python
# Load your own data and train
texts = load_domain_specific_data('your_data.csv', text_column='text')
trainer = GPT2Trainer(model_name='gpt2', max_length=512)
train_dataset, val_dataset = trainer.prepare_data(texts)
trainer.train_manual(train_dataset, val_dataset)
```

### Generate New Text
```python
# Generate text from fine-tuned model
generated_texts = trainer.generate_text(
    prompt="In machine learning,",
    max_length=100,
    num_return_sequences=3,
    temperature=0.8
)
```

### Model Evaluation
```python
# Calculate perplexity
perplexity = trainer.evaluate_perplexity(val_dataset)
print(f"Model perplexity: {perplexity:.2f}")
```

### Load Pre-trained Fine-tuned Model
```python
# Load saved model for inference
tokenizer = GPT2Tokenizer.from_pretrained('./gpt2-finetuned')
model = GPT2LMHeadModel.from_pretrained('./gpt2-finetuned')
```

### Custom Hyperparameters
```python
# Modify training parameters
trainer.train_manual(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    num_epochs=5,
    batch_size=8,
    learning_rate=3e-5
)
```

## 🔮 Future Enhancements
- Parameter-efficient fine-tuning methods (LoRA, Adapters)
- Multi-GPU distributed training support
- Advanced text generation strategies (beam search, nucleus sampling)
- Integration with Weights & Biases for experiment tracking
- Automated hyperparameter optimization
- Support for other language models (GPT-3, T5, BERT)
- Quantization and model compression techniques
- Web interface for interactive text generation

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 🙌 Acknowledgments
- **Hugging Face** for the Transformers library and pre-trained models
- **OpenAI** for the original GPT-2 model architecture
- **PyTorch** for the deep learning framework
- Open source community for continuous support and inspiration

## 📞 Contact
Zubair - [GitHub Profile](https://github.com/zubair-csc)

Project Link: [https://github.com/zubair-csc/020-GPT2-Domain-Finetuning](https://github.com/zubair-csc/020-GPT2-Domain-Finetuning)

⭐ Star this repository if you found it helpful!
