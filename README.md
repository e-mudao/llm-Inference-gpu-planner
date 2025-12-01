# llm-Inference-gpu-planner
这是一个基于 Python 的命令行工具，旨在帮助 AI 架构师、运维工程师和开发者在部署 LLM之前，精确计算所需的 GPU 显存资源 并预估 推理性能。  与简单的显存计算器不同，本工具结合了 Roofline 性能模型 与 真实工程经验（参考 vLLM/TRT-LLM），考虑了显存碎片、KV Cache 动态增长、通信开销以及算力利用率损耗。
