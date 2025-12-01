

# 🚀 LLM Inference GPU Planner

**Physics-Based Production Resource Planner for Large Language Models**

**LLM Inference GPU Planner** is a rigorous, standalone Python CLI tool designed for AI Architects and DevOps engineers.

Unlike simple VRAM calculators, this tool (V4) uses a **Physics-Based Roofline Model** to accurately simulate inference performance. It corrects common misconceptions about batching and concurrency, providing realistic estimates for **NVIDIA H100/A100/H20**, **RTX 4090**, and emerging hardware like **Huawei Ascend** and **Apple M3**.

---

## ✨ Key Features 

*   **📐 Physics-Based Simulation**:
    *   Corrects the "linear degradation" fallacy.
    *   Models **Weight Sharing**: Simulates how GPUs read weights once to serve an entire batch, meaning **User Speed** remains high even as concurrency increases (up to the bandwidth limit).
*   **🧠 Comprehensive Bottleneck Diagnosis**:
    *   **VRAM**: Calculates Weights + Dynamic KV Cache + Fragmentation Buffer (15%).
    *   **Decode Latency**: Estimates token generation speed based on Memory Bandwidth Utilization (MBU) and KV Traffic Penalty.
    *   **Prefill Latency**: Estimates Time-To-First-Token (TTFT) using FP16/FP8 TFLOPS and FlashAttention efficiency.
    *   **Communication**: Accounts for NVLink vs. PCIe overhead in Tensor Parallel (TP) setups.
*   **💾 Multi-Precision & Architecture**:
    *   Supports FP16, INT8, INT4, AWQ/GPTQ, and H100 FP8.
    *   Built-in profiles for Llama-3, Qwen2, Yi, DeepSeek, and more.
*   **🖥️ Wide Hardware Support**:
    *   **Datacenter**: NVIDIA H100, A100 (80G/40G), H20, L40S.
    *   **Consumer**: RTX 4090, Apple M3 Max.
    *   **NPU**: Huawei Ascend 910C, Hygon DCU, Cambricon MLU.

---

## 🛠️ Quick Start

### 1. Prerequisites
**Zero dependencies.** No `torch`, `cuda`, or `pip install` required. Just Python 3.6+.

### 2. Installation
```bash
git clone https://github.com/your-username/llm-Inference-gpu-planner.git
cd llm-Inference-gpu-planner
```

### 3. Run
```bash
python llmInferenceGPUPlanner.py
```

### 4. Interactive Example
Follow the prompts to configure your scenario:

```text
--- [1] Model Configuration ---
Model Parameters (Billion) [Default 72]: 72
Weight Precision: [1] FP16 (Half Precision)

--- [2] Workload ---
Concurrency [Default 20]: 50
Avg Context Length (Input+Output) [Default 4096]: 8192

--- [3] Hardware ---
Target GPU: [2] NVIDIA A100 (80GB)
```

---

## 📊 Sample Report

The tool generates a detailed production report:

```text
[3] Performance Estimates (Key Metrics):

  👤 Single User Experience (User Speed):
     32.5 tokens/s  [🟢 Excellent]
     └─ Smooth interactive experience
     • Prefill Latency: 0.45 s (TTFT)
     • E2E Latency:     5.60 s

  📈 System Throughput:
     1625.0 tokens/s
     └─ Capacity: ~140M tokens per day

  ⚙️ Efficiency Analysis:
     • Base Step Speed: 42.0 steps/s (Bandwidth / Weights)
     • TP Overhead:     5%
     • KV Traffic Penalty: 18% speed reduction (due to KV loading)

[4] Bottleneck Diagnosis:
  ✅ System Balanced. No critical bottlenecks detected.
```

---

## 🧠 How It Works (The Math)

This planner is built on first principles of GPU architecture:

1.  **Memory Capacity**:
    $$ VRAM_{total} = Weight + KV_{cache} + Buffer(15\%) $$
    *Includes overhead for PyTorch context and fragmentation.*

2.  **User Generation Speed**:
    $$ Speed_{user} = \frac{Bandwidth \times MBU}{Weight + KV_{active}} \times Efficiency_{TP} $$
    *   **Concept**: In one decoding step, the GPU reads the entire model weights + the active KV cache for all concurrent users.
    *   **Correction**: Unlike simple calculators, we do **not** divide the speed by concurrency. The weight read cost is amortized across the batch.

3.  **System Throughput**:
    $$ Throughput = Speed_{user} \times Concurrency $$

---

## 📝 Supported Hardware

| Vendor | Model | VRAM | Bandwidth | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **NVIDIA** | H100 | 80GB | 3.3 TB/s | FP8 Powerhouse |
| **NVIDIA** | A100 | 80GB | 1.9 TB/s | Industry Standard |
| **NVIDIA** | H20 | 96GB | 4.0 TB/s | High Bandwidth (CN Market) |
| **NVIDIA** | RTX 4090 | 24GB | 1.0 TB/s | Best Consumer GPU |
| **Huawei** | Ascend 910C | 128GB | 3.2 TB/s | High Performance NPU |
| **Apple** | M3 Max | 128GB | 400 GB/s | Unified Memory |

*(See `GPU_SPECS` in the code for the full list)*

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request to:
*   Add new hardware specifications.
*   Refine MBU (Memory Bandwidth Utilization) coefficients based on real-world benchmarks.
*   Add support for new quantization methods.

## 📄 License

This project is licensed under the **MIT License**. Free for personal and commercial use.
