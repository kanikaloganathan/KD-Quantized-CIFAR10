# 🔍 Knowledge Distillation + Quantization on CIFAR-10

This repository demonstrates the implementation of two powerful model compression techniques — **Knowledge Distillation (KD)** and **Post-Training Quantization (PTQ)** — applied on the CIFAR-10 dataset to reduce model size and inference cost, while preserving accuracy.

---

## 🧠 What’s Inside?

| Notebook | Description |
|----------|-------------|
| `01_Knowledge_Distillation_Training.ipynb` | Trains a student model using logits from a pretrained ResNet18 teacher on CIFAR-10. |
| `02_Quantization_and_Evaluation.ipynb` | Applies Post-Training Quantization to the distilled student and evaluates performance, size, and accuracy. |

---

## 🎯 Objective

- **Reduce Model Size** using Quantization (8-bit weights/activations)
- **Retain Accuracy** via Knowledge Distillation
- **Deploy Lightweight Models** on resource-constrained devices (e.g. mobile, edge, microcontrollers)

---

## 📊 Results Summary

| Stage                         | Accuracy | Model Size |
|------------------------------|----------|------------|
| Teacher (ResNet18)           | ~86%     | ~44 MB     |
| Distilled Student (Shallow CNN) | ~68%     | ~2 MB      |
| Quantized Student             | ~7.5%    | ~540 KB    |

> ⚠️ Note: Accuracy drop after naive PTQ suggests need for Quantization-Aware Training (QAT) or enhanced calibration.

---

## 📁 Dataset

- **CIFAR-10**, loaded using `torchvision.datasets`
- 60,000 32x32 color images in 10 classes

---

## 🚀 Run it on Colab (T4 GPU Recommended)

1. **Open notebooks** in Colab.
2. Set Runtime → Change runtime type → GPU (T4 or better).
3. Run each cell in order.

---

## 🧩 Dependencies

```bash
pip install torch torchvision matplotlib tqdm
