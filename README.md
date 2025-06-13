# Transformer Implementation

A PyTorch implementation of a decoder-only transformer model trained on Wikipedia data.
This project was made in December 2024, but I decided to upload this to github for archiving purpose.

## Features

- **Decoder-only Transformer**: Custom implementation with multi-head attention and feed-forward layers
- **Wikipedia Training**: Pre-tokenized Wikipedia dataset for language modeling
- **Configurable Architecture**: Adjustable model dimensions, attention heads, and layers
- **Modern PyTorch**: Uses DataLoader, proper device handling, and modular design

## Files

- `decoder_transformer.py` - Main decoder-only transformer model
- `my_transformer.py` - Core transformer components (attention, layers, etc.)
- `tokenizer.py` - Dataset wrapper for tokenized Wikipedia data
- `main.py` - Training script and model configuration
- `test.py` - Testing utilities

## Quick Start

1. Ensure you have the tokenized Wikipedia dataset in `./tokenized_wiki/`
2. Run the main training script:
   ```bash
   python main.py
   ```

## Model Configuration

- **Vocabulary Size**: DistilBERT tokenizer vocabulary
- **Model Dimension**: 128
- **Attention Heads**: 4
- **Layers**: 4
- **Feed-Forward Dimension**: 1024
- **Max Sequence Length**: 128
- **Batch Size**: 64

## Requirements

- PyTorch
- Transformers (HuggingFace)
- Datasets
- NumPy