# Transformers from Scratch

This repository contains my implementation of Transformers from scratch, inspired by Andrej Karpathy's video. While the original implementation focused on a decoder-only model, I have extended it by adding an encoder as well. The goal of this project is to deeply understand the inner workings of Transformers by building them step by step without relying on high-level libraries like Hugging Face Transformers.

## Features

- Implements a full Transformer model (Encoder-Decoder architecture)
- Single-file implementation (`gpt.py`) for simplicity
- Includes essential components:
  - Token Embeddings
  - Positional Encodings
  - Multi-Head Self-Attention
  - Feedforward Layers
  - Layer Normalization
  - Encoder and Decoder Blocks
- Trained on sample text data to demonstrate functionality

## Installation

To run the implementation, clone this repository and install the required dependencies:

```bash
git clone https://github.com/ahmetz3lka/transformers_from_scratch.git
cd transformers-from-scratch
pip install -r requirements.txt
```

## Usage

Run the script with:

```bash
python gpt.py
```

Modify `gpt.py` to experiment with different model hyperparameters.

## Understanding the Code

The entire Transformer model is implemented within a single file, `gpt.py`, to keep things simple and easy to follow. The key sections include:

- **Embedding Layer**: Converts tokens into dense vector representations.
- **Self-Attention Mechanism**: Captures relationships between tokens.
- **Feedforward Network**: Adds non-linearity and depth.
- **Encoder-Decoder Architecture**: Implements both parts of the Transformer model.

## Future Improvements

- Fine-tune on larger datasets
- Experiment with different tokenization techniques
- Optimize performance and efficiency

## References

- [Andrej Karpathy's Video](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## License

This project is open-source under the MIT License.
