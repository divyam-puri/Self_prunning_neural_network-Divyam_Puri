# 🧠 Self-Pruning Neural Network (PyTorch)

A lightweight implementation of a **self-pruning feedforward neural network** that learns to identify and suppress less important connections during training.

Built as part of an AI Engineering case study focusing on **model efficiency, sparsity, and dynamic architecture adaptation**.

---

## 🚀 Overview

In real-world deployments, neural networks often face **memory and compute constraints**.
This project explores a technique where the network **learns to prune itself during training**, instead of relying on post-training pruning.

Each weight is associated with a learnable **gate parameter**, which determines whether that connection remains active or is suppressed.

---

## ⚙️ Key Idea

Instead of removing weights after training, we introduce:

* 🔹 **Learnable Gates** for each neuron

* 🔹 Gates ∈ (0, 1) using Sigmoid

* 🔹 Effective weight:

  ```
  pruned_weight = weight × gate
  ```

* 🔹 Sparsity encouraged via L1 regularization on gates

---

## 🏗️ Architecture

```
Input (3072)
   ↓
PrunableLinear (512)
   ↓
ReLU
   ↓
PrunableLinear (256)
   ↓
ReLU
   ↓
PrunableLinear (10)
   ↓
Output (CIFAR-10 classes)
```

---

## 📂 Project Structure

```
.
├── prunable_layer.py          # Core model + training loop
├── self_pruning_pipeline.py  # Full pipeline (experiments + plots)
├── data/                      # CIFAR-10 dataset (auto-downloaded)
├── gate_histogram.png         # Gate distribution plot
├── lambda_comparison.png      # Accuracy vs Sparsity plot
└── README.md
```

---

## 🧪 Experiments

The model was trained on **CIFAR-10** across multiple values of λ (sparsity strength):

| Lambda | Accuracy | Sparsity |
| ------ | -------- | -------- |
| 0.0001 | ~55%     | 0.00%    |
| 0.001  | ~55%     | 0.00%    |
| 0.01   | ~54%     | 0.00%    |

---

## 📊 Observations

* The model achieves **reasonable classification performance (~55%)**
* Gate values mostly lie in the range **0.7–0.8**
* **Minimal sparsity observed** across tested λ values

### 💡 Interpretation

* The sparsity regularization was **not strong enough** to override the classification objective
* The network prefers to **retain most connections** for better accuracy
* Demonstrates the **trade-off between sparsity and performance**

---

## 📈 Visualizations

### 🔹 Gate Distribution

* Most gates remain active
* No significant clustering near zero (pruned region)

### 🔹 Lambda vs Performance

* Accuracy remains stable across λ
* Sparsity does not increase significantly

---

## 🧠 Key Learnings

* Implementing **custom neural layers with learnable gates**
* Understanding **L1 regularization for sparsity**
* Observing **accuracy vs pruning trade-offs**
* Building **end-to-end ML pipelines (training → evaluation → visualization)**

---

## ⚡ How to Run

```bash
# Clone repo
git clone https://github.com/your-username/self-pruning-network.git
cd self-pruning-network

# Install dependencies
pip install torch torchvision matplotlib pandas

# Run full pipeline
python self_pruning_pipeline.py
```

---

## 🛠️ Tech Stack

* Python 🐍
* PyTorch 🔥
* NumPy & Pandas 📊
* Matplotlib 📈

---

## 📌 Future Improvements

* Stronger sparsity enforcement (higher λ or alternative penalties)
* Structured pruning (neuron/channel level)
* Extension to CNN architectures
* Dynamic threshold-based pruning

---

## 👨‍💻 Author

**Divyam Puri**
AI Engineering Enthusiast

---

## ⭐ Final Note

This project focuses on **understanding and implementing self-pruning mechanisms**, highlighting both their potential and practical challenges in real-world scenarios.
