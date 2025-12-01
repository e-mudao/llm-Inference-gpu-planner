import math
import sys

# ==========================================
# 1. 基础数据库
# ==========================================

GPU_SPECS = {
    "1": {"name": "NVIDIA H100 (80GB)", "vram": 80, "bw": 3350, "tflops": 989, "desc": "HBM3, 新一代旗舰, FP8算力强悍"},
    "2": {"name": "NVIDIA A100 (80GB)", "vram": 80, "bw": 1935, "tflops": 312, "desc": "HBM2e, 工业界标准训练/推理卡"},
    "3": {"name": "NVIDIA A100 (40GB)", "vram": 40, "bw": 1555, "tflops": 312, "desc": "HBM2, 显存较小"},
    "4": {"name": "NVIDIA H20 (96GB)",  "vram": 96, "bw": 4000, "tflops": 148, "desc": "HBM3, 中国特供, 高带宽低算力"},
    "5": {"name": "RTX 4090 (24GB)",    "vram": 24, "bw": 1008, "tflops": 330, "desc": "GDDR6X, 消费级最强"},
    "6": {"name": "Apple M3 Max (128G)", "vram": 128, "bw": 400, "tflops": 15, "desc": "统一内存, 适合本地推理"},
    "7": {"name": "华为 Ascend 910C",    "vram": 128, "bw": 3200, "tflops": 800, "desc": "HBM3, 国产算力旗舰"},
    "8": {"name": "海光 DCU K100",      "vram": 64,  "bw": 892,  "tflops": 196, "desc": "HBM3, 兼容 ROCm 生态"},
    "9": {"name": "寒武纪 MLU590",      "vram": 80,  "bw": 2000, "tflops": 314, "desc": "HBM2e, 类似 A100 性能"},
    "10": {"name": "L40S (48GB)",       "vram": 48,  "bw": 864,  "tflops": 366, "desc": "GDDR6, 推理专用, 无NVLink"}
}

# 权重精度 (Weight Precision)
WEIGHT_PRECISION = {
    "1": {"name": "FP16 (半精度)", "bytes": 2},
    "2": {"name": "INT8 (8-bit)",   "bytes": 1},
    "3": {"name": "INT4 (4-bit)",   "bytes": 0.5},
    "4": {"name": "AWQ/GPTQ",       "bytes": 0.55}
}

# KV Cache 精度 (KV Cache Precision)
KV_PRECISION = {
    "1": {"name": "FP16 (标准)",       "bytes": 2},
    "2": {"name": "INT8 (KV量化)",     "bytes": 1}, 
    "3": {"name": "FP8 (H100优化)",    "bytes": 1} 
}

# KV Cache 基准值 (MB per token, 包含 GQA 优化)
# 数据来源：vLLM/TRT-LLM 实测数据
MODEL_ARCH_BASE = {
    "7B":   0.18,  # Llama-3-8B / Qwen2-7B
    "14B":  0.28,  # Qwen2-14B
    "32B":  0.38,  # Yi-34B
    "72B":  0.50,  # Qwen2-72B / Llama-3-70B
    "110B": 0.65   # Qwen2-110B
}

# 系统效率参数 (基于 vLLM/FlashAttention-2)
MBU_EFFICIENCY = 0.70           # 显存带宽有效利用率
COMPUTE_EFFICIENCY = 0.32       # 算力有效利用率 (Prefill阶段)
EFFECTIVE_LOAD_FACTOR = 0.80    # 显存规划负载因子 (预留波动空间)

# ==========================================
# 2. 核心计算逻辑 (V4 物理修正版)
# ==========================================

def calculate_production_grade_v4(inputs):
    """
    生产级 GPU 规划计算核心 - V4
    基于 Roofline 模型与正确的 Batching 物理机制
    """
    
    # --- A. 显存容量计算 ---
    
    param_b = inputs['param_size']
    # 工程近似: 1B params * 2 bytes ≈ 2GB (误差被 Buffer 吸收)
    weight_gb = param_b * inputs['weight_bytes']
    
    # KV Cache 计算
    base_kv_mb = MODEL_ARCH_BASE.get(inputs['model_scale'], 0.50)
    kv_scale = inputs['kv_bytes'] / 2.0
    kv_mb_per_token = base_kv_mb * kv_scale
    
    concurrency = inputs['concurrency']
    avg_seq = inputs['avg_context']
    
    # 容量规划使用峰值负载
    total_tokens_capacity = avg_seq * concurrency * EFFECTIVE_LOAD_FACTOR
    kv_cache_gb = total_tokens_capacity * kv_mb_per_token / 1024
    
    # Buffer (15% Overhead: PyTorch context, fragmentation, activations)
    buffer_gb = 3.0 + (weight_gb + kv_cache_gb) * 0.15
    
    total_vram = weight_gb + kv_cache_gb + buffer_gb
    
    # --- B. 硬件需求计算 ---
    
    gpu_vram = inputs['gpu']['vram']
    num_gpus = math.ceil(total_vram / gpu_vram)
    vram_util = total_vram / (num_gpus * gpu_vram)
    
    # --- C. 性能计算 (修正后的物理模型) ---
    
    system_bw = inputs['gpu']['bw'] * num_gpus * MBU_EFFICIENCY
    
    # 1. 基础步速 (Base Steps/s)
    # 物理意义：不考虑 KV 和通信，仅读取权重能跑多快？
    # 这是单用户速度的理论上限。
    if weight_gb > 0:
        base_steps_per_sec = system_bw / weight_gb
    else:
        base_steps_per_sec = 0
        
    # 2. 效率修正因子
    
    # (a) Batch 效率: 并发过低无法喂饱 GPU
    if concurrency < 4:
        batch_eff = 0.5 + 0.125 * concurrency
    else:
        batch_eff = 1.0
        
    # (b) TP 通信损耗: 多卡互联的 overhead
    if num_gpus > 1:
        # NVLink 也会有损耗，假设每卡增加 5% 损耗，最多 30%
        tp_eff = max(0.70, 1.0 - 0.05 * (num_gpus - 1))
    else:
        tp_eff = 1.0
        
    # (c) KV Cache 流量惩罚 (Traffic Penalty)
    # 在 Decode 阶段，每一步传输的数据 = 权重 + 活跃KV
    # 活跃 KV = 并发数 * 平均历史长度 * KV大小
    avg_history_len = avg_seq / 2
    active_kv_gb = concurrency * avg_history_len * (kv_mb_per_token / 1024)
    
    # 流量膨胀系数
    traffic_ratio = (weight_gb + active_kv_gb) / weight_gb
    
    # 3. 最终速度计算
    
    # 单用户速度 (User Speed): 真实的生成体验
    user_speed = (base_steps_per_sec * batch_eff * tp_eff) / traffic_ratio
    
    # 系统总吞吐 (System Throughput): 服务承载能力
    system_throughput = user_speed * concurrency
    
    # --- D. 延迟估算 (Prefill & E2E) ---
    
    # Prefill (Compute Bound)
    avg_input_tokens = avg_seq * 0.75
    prefill_flops = 2 * param_b * 1e9 * avg_input_tokens
    system_tflops = inputs['gpu']['tflops'] * num_gpus * COMPUTE_EFFICIENCY
    if system_tflops > 0:
        prefill_latency = prefill_flops / (system_tflops * 1e12)
    else:
        prefill_latency = 999
        
    # Decode Time
    expected_output_tokens = avg_seq * 0.25
    if user_speed > 0:
        decode_time = expected_output_tokens / user_speed
    else:
        decode_time = 999
        
    e2e_latency = prefill_latency + decode_time
    
    # --- E. 评级 ---
    if user_speed >= 40: grade, comment = "🚀 极速", "超流畅 (人类阅读速度 3-4倍)"
    elif user_speed >= 20: grade, comment = "🟢 优秀", "流畅交互体验"
    elif user_speed >= 10: grade, comment = "🟡 良好", "可接受的阅读速度"
    else: grade, comment = "🔴 较差", "明显的逐字生成感"

    return {
        "capacity": {
            "total_vram": total_vram,
            "weight": weight_gb,
            "kv_cache": kv_cache_gb,
            "buffer": buffer_gb,
            "kv_mb_per_token": kv_mb_per_token
        },
        "hardware": {
            "num_gpus": num_gpus,
            "vram_util": vram_util,
            "total_vram_pool": num_gpus * gpu_vram
        },
        "performance": {
            "base_steps": base_steps_per_sec,
            "user_speed": user_speed,
            "system_throughput": system_throughput,
            "kv_traffic_ratio": traffic_ratio,
            "tp_efficiency": tp_eff,
            "prefill_latency": prefill_latency,
            "e2e_latency": e2e_latency,
            "grade": grade,
            "comment": comment
        },
        "bottleneck_analysis": {
            "memory_bound": active_kv_gb > weight_gb * 0.5,
            "compute_bound": prefill_latency > 2.0,
            "tp_constrained": tp_eff < 0.85
        }
    }

# ==========================================
# 3. 交互工具函数
# ==========================================

def get_choice(options, text, default_key="1"):
    """支持回车默认和错误重试的选择函数"""
    print(f"\n{text}")
    for k, v in options.items():
        name = v['name'] if 'name' in v else v
        suffix = " (默认)" if k == default_key else ""
        print(f"  [{k}] {name}{suffix}")
    
    while True:
        val = input(f"👉 选择 [默认 {default_key}]: ").strip()
        if not val:
            val = default_key
        
        if val in options:
            return options[val]
        else:
            print(f"❌ 输入无效，请从 {list(options.keys())} 中选择")

def get_number(prompt, default):
    val = input(f"{prompt} [默认 {default}]: ").strip()
    return float(val) if val else default

def get_closest_scale(param):
    scales = [7, 14, 32, 72, 110]
    closest = min(scales, key=lambda x: abs(x - param))
    return f"{closest}B"

# ==========================================
# 4. 主程序
# ==========================================

def main():
    print("\n" + "="*60)
    print("🚀 GPU 生产级算力规划器 (Final V4)")
    print("   Physical-Based Modeling | Corrected Batching Logic")
    print("="*60)
    
    # --- 输入 ---
    print("\n--- [1] 模型配置 ---")
    param = get_number("模型参数量 (Billion)", 72)
    model_scale = get_closest_scale(param)
    print(f"   → 匹配架构基准: {model_scale}")
    
    w_prec = get_choice(WEIGHT_PRECISION, "权重精度:")
    kv_prec = get_choice(KV_PRECISION, "KV Cache 精度:")
    
    print("\n--- [2] 业务负载 ---")
    conc = int(get_number("并发用户数 (Concurrency)", 20))
    seq = int(get_number("平均上下文长度 (Input+Output)", 4096))
    
    print("\n--- [3] 硬件选择 ---")
    gpu = get_choice(GPU_SPECS, "目标显卡:")
    
    # --- 计算 ---
    print("\n⏳ 正在进行物理仿真计算...")
    res = calculate_production_grade_v4({
        'param_size': param,
        'model_scale': model_scale,
        'weight_bytes': w_prec['bytes'],
        'kv_bytes': kv_prec['bytes'],
        'avg_context': seq,
        'concurrency': conc,
        'gpu': gpu
    })
    
    c = res['capacity']
    h = res['hardware']
    p = res['performance']
    b = res['bottleneck_analysis']
    
    # --- 报告 ---
    print("\n" + "="*60)
    print("📊 仿真评估报告")
    print("="*60)
    
    # 1. 容量
    print(f"\n[1] 显存容量规划:")
    print(f"  • 模型权重:     {c['weight']:>8.2f} GB ({w_prec['name']})")
    print(f"  • KV Cache:     {c['kv_cache']:>8.2f} GB ({kv_prec['name']})")
    print(f"    └─ 峰值估算:  {conc}并发 × {seq}长度 × {c['kv_mb_per_token']:.2f}MB/token")
    print(f"  • 系统Buffer:   {c['buffer']:>8.2f} GB (预留 15%)")
    print(f"  {'-'*40}")
    print(f"  ★ 总显存需求:   {c['total_vram']:>8.2f} GB")
    
    # 2. 硬件
    print(f"\n[2] 硬件配置建议:")
    print(f"  • 推荐配置:     {h['num_gpus']} × {gpu['name']}")
    print(f"  • 显存池:       {h['total_vram_pool']:.0f} GB")
    print(f"  • 利用率:       {h['vram_util']*100:.1f}%")
    
    if h['vram_util'] > 0.90:
        print("  ⚠️  警告: 显存极其紧张，建议增加 1 张卡防止 OOM")
    elif h['vram_util'] < 0.60:
        print("  💡 提示: 显存有大量富余，可尝试更大模型或更高并发")
        
    # 3. 性能
    print(f"\n[3] 性能表现 (关键指标):")
    
    print(f"\n  👤 单用户体验 (User Speed):")
    print(f"     {p['user_speed']:>6.1f} tokens/s  [{p['grade']}]")
    print(f"     └─ {p['comment']}")
    print(f"     • Prefill延迟: {p['prefill_latency']:.2f} s (首字等待)")
    print(f"     • 端到端延迟:  {p['e2e_latency']:.2f} s")
    
    print(f"\n  📈 系统总吞吐 (System Throughput):")
    print(f"     {p['system_throughput']:>6.1f} tokens/s")
    print(f"     └─ 每天可处理约 {int(p['system_throughput']*3600*24/1e6)}M tokens")
    
    print(f"\n  ⚙️  效率因子分析:")
    print(f"     • 基准步速: {p['base_steps']:.1f} steps/s (带宽/权重)")
    print(f"     • TP通信损耗: {(1-p['tp_efficiency'])*100:.0f}%")
    print(f"     • KV流量惩罚: 速度降低 {(p['kv_traffic_ratio']-1)*100:.0f}% (因搬运KV Cache)")

    # 4. 瓶颈与建议
    print(f"\n[4] 瓶颈诊断与建议:")
    
    has_issue = False
    
    if b['memory_bound']:
        print(f"  🔴 内存带宽瓶颈: KV Cache 传输量过大")
        print(f"     → 方案: 启用 {kv_prec['name']} -> INT8/FP8 KV Cache")
        print(f"     → 方案: 使用 GQA/MLA 架构模型 (如 DeepSeek/Llama3)")
        has_issue = True
        
    if b['compute_bound']:
        print(f"  🔴 算力瓶颈: Prefill 阶段过慢")
        print(f"     → 方案: 增加 GPU 数量利用 TP 聚合算力")
        has_issue = True
        
    if b['tp_constrained']:
        print(f"  🟡 通信瓶颈: 多卡通信损耗显著")
        print(f"     → 方案: 必须使用 NVLink/NVSwitch，避免 PCIe")
        has_issue = True
        
    if not has_issue:
        print(f"  ✅ 系统配置均衡，无明显硬件瓶颈")
        
    # 5. 架构推荐
    print(f"\n[5] 部署架构推荐:")
    if h['num_gpus'] == 1:
        print("  🏗️  单卡推理 (Single GPU)")
        print("     • 推荐引擎: vLLM, TensorRT-LLM")
    elif h['num_gpus'] <= 8:
        print(f"  🏗️  张量并行 (Tensor Parallel, TP={h['num_gpus']})")
        print("     • 必须拥有高带宽互联 (NVLink)")
        print(f"     • 启动命令参考: vllm serve ... --tensor-parallel-size {h['num_gpus']}")
    else:
        print(f"  🏗️  混合并行 (TP=8 + PP={math.ceil(h['num_gpus']/8)})")
        print("     • 适用于超大规模集群，需复杂编排")

    print("\n" + "="*60)
    print("✅ 计算完成")
    print("="*60)

if __name__ == "__main__":
    main()
