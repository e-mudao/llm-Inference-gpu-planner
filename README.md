
# 🚀 LLM Inference GPU Planner 

**基于物理仿真与 Roofline 模型的生产级大模型算力规划器**

[中文] | [English](README_EN.md)

`LLM Inference GPU Planner` 是一个轻量级但逻辑严密的 Python 命令行工具，专为 AI 架构师、运维工程师和开发者设计。

与市面上简单的“显存计算器”不同，本工具采用了**正确的物理仿真模型（Physics-Based Modeling）**，修正了传统计算中关于 Batching 和吞吐量的常见误区，能够精确预估 **NVIDIA H100/A100/H20**、**RTX 4090** 以及 **中国国产算力芯片（华为昇腾/海光/寒武纪）** 在生产环境下的真实表现。


## ✨ 核心特性

*   **📐 物理级性能仿真**：
    *   摒弃了“并发越高单用户越慢”的错误线性模型。
    *   引入 **“大巴车模型”**：正确模拟权重共享机制，单用户速度取决于带宽与流量膨胀，系统吞吐随并发线性增长。
*   **🧠 全链路瓶颈诊断**：
    *   **显存容量**：包含权重、动态 KV Cache、碎片化 Buffer (15%)。
    *   **Decode 延迟**：基于带宽利用率 (MBU) 和 KV 流量惩罚 (Traffic Penalty)。
    *   **Prefill 延迟**：基于算力 (TFLOPS) 和 FlashAttention 利用率计算首字时间。
    *   **通信损耗**：自动计算多卡 Tensor Parallel (TP) 下的 NVLink/PCIe 通信开销。
*   **💾 多精度与架构支持**：
    *   支持 FP16, INT8, FP8 (H100), INT4, AWQ/GPTQ。
    *   内置 Llama-3, Qwen2, Yi, DeepSeek 等主流模型架构参数。
*   **🖥️ 广泛的硬件库**：
    *   NVIDIA: H100, A100 (80G/40G), H20 (CN), RTX 4090, L40S。
    *   国产芯片: Huawei Ascend 910C, Hygon DCU K100, Cambricon MLU590。
    *   消费级: Apple M3 Max。

## 🛠️ 快速开始

### 1. 环境要求
本工具**零依赖**，无需安装 PyTorch 或 CUDA，仅需 Python 3.6+ 标准库即可运行。

### 2. 获取代码
```bash
git clone https://github.com/your-username/llm-Inference-gpu-planner.git
cd llm-Inference-gpu-planner
```

### 3. 运行规划器
```bash
python llmInferenceGPUPlanner.py
```

### 4. 交互示例
```text
🚀 GPU 生产级算力规划器 
============================================================

--- [1] 模型配置 ---
模型参数量 (Billion) [默认 72]: 72
权重精度: [1] FP16 (半精度)

--- [2] 业务负载 ---
并发用户数 (Concurrency) [默认 20]: 50
平均上下文长度 (Input+Output) [默认 4096]: 8192

--- [3] 硬件选择 ---
目标显卡: [2] NVIDIA A100 (80GB)
```

## 📊 报告解读示例

运行后，程序将生成详细的评估报告：

```text
[3] 性能表现 (关键指标):

  👤 单用户体验 (User Speed):
     32.5 tokens/s  [🟢 优秀]
     └─ 流畅交互体验
     • Prefill延迟: 0.45 s (首字等待)
     • 端到端延迟:  5.60 s

  📈 系统总吞吐 (System Throughput):
     1625.0 tokens/s
     └─ 每天可处理约 140M tokens

  ⚙️ 效率因子分析:
     • 基准步速: 42.0 steps/s (带宽/权重)
     • TP通信损耗: 5%
     • KV流量惩罚: 速度降低 18% (因搬运KV Cache)
```

## 🧠 核心算法原理 (Under the Hood)

本工具基于以下物理公式进行计算：

1.  **显存需求 (VRAM)**
    $$ VRAM_{total} = Weight + KV_{cache} + Buffer(15\%) $$
    *其中 KV Cache 考虑了 GQA 优化与并发数。*

2.  **单用户速度 (User Speed)**
    $$ Speed_{user} = \frac{Bandwidth \times MBU}{Weight + KV_{active}} \times Efficiency_{TP} $$
    *   **MBU**: 显存带宽利用率 (0.7)。
    *   **KV Active**: 每个生成步骤中必须传输的活跃 KV 数据量。
    *   **物理意义**: 权重读取一次服务所有用户，因此高并发下 User Speed 仅受 KV 流量膨胀影响，略微下降，而非大幅下降。

3.  **系统吞吐 (Throughput)**
    $$ Throughput = Speed_{user} \times Concurrency $$

## 📝 支持硬件列表

| 厂商       | 型号        | 显存  | 带宽     | 备注             |
| :--------- | :---------- | :---- | :------- | :--------------- |
| **NVIDIA** | H100        | 80GB  | 3.3 TB/s | FP8 算力怪兽     |
| **NVIDIA** | A100        | 80GB  | 1.9 TB/s | 工业界标准       |
| **NVIDIA** | H20         | 96GB  | 4.0 TB/s | 高带宽，适合推理 |
| **NVIDIA** | RTX 4090    | 24GB  | 1.0 TB/s | 性价比之王       |
| **Huawei** | Ascend 910C | 128GB | 3.2 TB/s | 国产旗舰         |
| **Apple**  | M3 Max      | 128GB | 400 GB/s | 统一内存架构     |

*(更多硬件请在代码 `GPU_SPECS` 字典中查看)*

## 🤝 贡献与反馈 (Contributing)

欢迎提交 Issue 或 Pull Request 来完善硬件数据库或修正计算系数。

*   如果你发现某款显卡的 MBU (Bandwidth Utilization) 与实测偏差较大，请反馈。
*   如果你有新的国产芯片实测数据，欢迎补充。



