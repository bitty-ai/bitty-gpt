<p align="center">
  <img src="assets/bitty-gpt.png" alt="Bitty GPT Logo" width="600"/>
</p>

# 🤖 Bitty GPT: A Minimalist Implementation of the GPT Architecture

Welcome to **Bitty GPT**, a simplified, educational, and highly customizable implementation of the Generative Pre-trained Transformer (GPT) architecture. This project is designed to be a learning resource, allowing you to easily understand and modify the core components of a modern language model pipeline.

## 🚀 Getting Started

Follow these steps to set up the environment and start training your own model.

### 1. Data Setup

First, download and prepare the required dataset by running the following script from your terminal:

```bash
sh data-download.sh
```

### 2. Training the Model

Once the data is ready, you can start the training process by running the main training script:

```bash
python src/train.py
```

## ✨ Customization and Training Flags

The training pipeline in [`src/train.py`](src/train.py) is highly customizable using command-line flags. Here are the available options for enhancing and tailoring your training runs:

| Flag | Description | Possible Values |
| :--- | :--- | :--- |
| `--wandb` | Enables/disables logging of training metrics to Weights & Biases (WandB). | \`0\` (Disable), \`1\` (Enable) |
| `--train-dataset` | Controls whether the raw dataset should be converted to a more efficient \`.bin\` format. This is a computationally expensive process and can take several minutes depending on the dataset size. | (Flag presence determines action) |
| `--train-vocab` | Determines whether a new vocabulary should be trained on the current dataset. Currently supports only \`<|endoftext|>\` as a special token. For custom tokens, please edit the \`bitty-tokenizer\` located in the project's source. | (Flag presence determines action) |

---

*Happy Coding and Training!*

