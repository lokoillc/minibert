# MiniBERT for Chinese Sentiment Classification

A lightweight BERT-based model for Chinese sentiment analysis, fine-tuned on the ChnSentiCorp dataset. The model supports easy inference via a simple Gradio demo.

## 📌 Project Overview

- **Task**: Chinese Sentiment Classification (Binary: Positive / Negative)
- **Dataset**: [ChnSentiCorp](https://huggingface.co/datasets/chnsenticorp)
- **Model**: Custom MiniBERT (Tiny BERT-like architecture)
- **Framework**: PyTorch
- **Demo**: Gradio-based web UI for text inference

## 🏗️ Model Architecture

- Transformer Encoder Blocks: 4 layers
- Hidden Size: 256
- Attention Heads: 4
- Modified Feed-Forward Network (with optional MoE)
- Dropout Regularization
- Classifier Head for binary sentiment prediction

## ⚙️ Training Details

- Optimizer: AdamW
- Learning Rate: 5e-5
- Batch Size: 32
- Epochs: 5
- Mixed Precision: Supported (AMP)
- Loss Function: CrossEntropyLoss
- Evaluation Metric: Accuracy, F1-score

## 📊 Results

| Metric     | Validation |
|------------|------------|
| Accuracy   | 92.4%     |
| F1-Score   | 92.1%     |

> Note: Results evaluated on ChnSentiCorp validation set.

## 🚀 Gradio Demo

Easily test the model via a simple Gradio web interface:

```bash
python gradio_demo.py
