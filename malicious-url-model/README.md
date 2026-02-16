---
language: en
tags:
  - cybersecurity
  - malicious-url-detection
  - bert
  - transformers
  - phishing-detection
license: apache-2.0
---

# Malicious URL Detection Model

> A fine-tuned **BERT-LoRA** model for detecting malicious URLs, including phishing, malware, and defacement threats.

## Model Description

This model is a **fine-tuned BERT-based classifier** designed to detect **malicious URLs** in real-time. It applies **Low-Rank Adaptation (LoRA)** for efficient fine-tuning, reducing computational costs while maintaining high accuracy.

The model classifies URLs into **four categories**:

- **Benign**
- **Defacement**
- **Phishing**
- **Malware**

It achieves **98% validation accuracy** and an **F1-score of 0.965**, ensuring robust detection capabilities.

---

## Intended Uses

### Use Cases

- Real-time URL classification for cybersecurity tools
- Phishing and malware detection for online safety
- Integration into browser extensions for instant threat alerts
- Security monitoring for SOC (Security Operations Centers)

---

## Model Details

- **Model Type:** BERT-based URL Classifier
- **Fine-Tuning Method:** LoRA (Low-Rank Adaptation)
- **Base Model:** `bert-base-uncased`
- **Number of Parameters:** 110M
- **Dataset:** Kaggle Malicious URLs Dataset (~651,191 samples)
- **Max Sequence Length:** `128`
- **Framework:** 🤗 `transformers`, `torch`, `peft`

---

## How to Use

You can use this model directly with 🤗 **Transformers**:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the model and tokenizer
model_name = "your-huggingface-model-name"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Example URL
url = "http://example.com/login"

# Tokenize and predict
inputs = tokenizer(url, return_tensors="pt", truncation=True, padding=True, max_length=128)
with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits).item()

# Mapping prediction to labels
label_map = {0: "Benign", 1: "Defacement", 2: "Phishing", 3: "Malware"}
print(f"Prediction: {label_map[prediction]}")
```

---

## Training Details

- **Batch Size:** `16`
- **Epochs:** `5`
- **Learning Rate:** `2e-5`
- **Optimizer:** AdamW with weight decay
- **Loss Function:** Weighted Cross-Entropy
- **Evaluation Strategy:** Epoch-based
- **Fine-Tuning Strategy:** LoRA applied to BERT layers

---

## Evaluation Results

| Metric       | Value     |
| ------------ | --------- |
| Accuracy     | **98%**   |
| Precision    | **0.96**  |
| Recall       | **0.97**  |
| **F1 Score** | **0.965** |

### Category-wise Performance

| Category       | Precision | Recall | F1-Score |
| -------------- | --------- | ------ | -------- |
| **Benign**     | 0.98      | 0.99   | 0.985    |
| **Defacement** | 0.98      | 0.99   | 0.985    |
| **Phishing**   | 0.93      | 0.94   | 0.935    |
| **Malware**    | 0.95      | 0.96   | 0.955    |

---

## Deployment Options

### Streamlit Web App

- Deployed on **Streamlit Cloud, AWS, or Google Cloud**.
- Provides **real-time URL analysis** with a user-friendly interface.

### Browser Extension (Planned)

- **Real-time scanning** of visited web pages.
- **Dynamic threat alerts** with confidence scores.

### API Integration

- REST API for bulk URL analysis.
- Supports **Security Operations Centers (SOC)**.

---

## Limitations & Bias

- **May misclassify complex phishing URLs** that mimic legitimate sites.
- **Needs regular updates** to counter evolving threats.
- **Potential bias** if future threats are not represented in training data.

---

## Training Data & Citation

### Data Source

Dataset sourced from **Kaggle Malicious URLs Dataset**:  
📌 [Dataset Link](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset)

### BibTeX Citation

```
@article{maliciousurl2025,
  author    = {Gleyzie Tongo, Dr. Farnaz Farid, Dr. Ala Al-Areqi, Dr. Farhad Ahamed},
  title     = {Fine-Tuned BERT for Malicious URL Detection},
  year      = {2025},
  institution = {Western Sydney University}
}
```

---

## Contact

For inquiries, collaborations, or feedback, feel free to reach out via LinkedIn:  
🔗 [Gleyzie Tongo](https://www.linkedin.com/in/gleyzie-tongo-83b454218/)
