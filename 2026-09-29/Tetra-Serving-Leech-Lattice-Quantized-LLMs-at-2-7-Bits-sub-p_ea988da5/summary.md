---
title: "Tetra-Serving-Leech-Lattice-Quantized-LLMs-at-2-7-Bits-sub-p"
source: https://arxiv.org/pdf/2609.35465v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 11:07:33"
field: "大语言模型低比特量化与高效推理"
keywords: ["Leech lattice", "LLM quantization", "trellis decoding", "2-bit quantization", "low-bit inference", "Golay code", "GPU kernel"]
innovations: ["基于Golay码trellis的24维Leech格高效解码方案，每weight仅读2.148 bits", "混合精度配方：row scale微调+int4关键投影+4-bit嵌入表实现2.7 bits/parameter", "融合解码与GEMV的kernel，6次table load完成24权重解码"]
benchmarks: ["MMLU", "GSM8K"]
---

# 论文速读：Tetra-Serving-Leech-Lattice-Quantized-LLMs-at-2-7-Bits-sub-p

## 一句话总结
Tetra 提出了一种新的 Leech 格码本设计，利用 Golay 码的 trellis 结构实现高效 GPU 解码，将 2-bit 权重从加载时展开为 4.804 bits/weight 降至 2.148 bits/weight，使完整 Qwen3 模型实现约 2.7 bits/parameter 的压缩，显著减少显存占用。

## 研究问题与动机
1. **码本过大无法查表**：Leech 格（Λ₂₄）在 2 bits/weight 下的最优码本包含超过 10¹⁴ 个格点，无法直接存储为 GPU 查找表。
2. **先前方案带宽浪费严重**：作者的 Planes14 方案在加载时将 47-bit 码字展开为 112-bit 记录，kernel 实际读取 4.804 bits/weight，超过 4-bit AWQ 的 4.179 bits/weight，压缩收益被抵消。
3. **显存-质量权衡需求**：14B 模型 FP16 权重需 29.5 GB，超出消费级 GPU（如 24 GB L40S）容量，需在低比特压缩下保持可用质量。
4. **计算效率瓶颈**：即便码本可存储，高效解码（低指令数、无分支发散、寄存器优化）是实现 2-bit 格式实际部署的关键挑战。

## 核心贡献（创新点）
1. **Tetra 码本设计**：将 24 维格量化分解为 3 个独立的 8 维子问题，利用 Golay 码的 trellis 结构（64 状态节点）和单个 16 KiB 查找表实现解码，与之前 Planes14 展开方案本质不同。
2. **融合 kernel 实现 2.148 bits/weight 读取**：kernel 在一次 pass 中同时完成解码和矩阵向量乘法，仅需 6 次 table load + 2 次小查找，读取量仅为 Planes14 的 44.7% 和 4-bit AWQ 的 51.4%。
3. **完整模型构建配方**：不改写格码，通过 retrain row scales（零额外字节）、int4 处理关键投影层、4-bit embedding tables 三项改动，将 Qwen3-4B/8B/14B 压缩至约 2.7 bits/parameter。
4. **严格实验与预注册**：在 MMLU 和 GSM8K 上提供配对置信区间和 McNemar 检验，所有实验预注册，确保结果可复现性。

## 方法详解
1. **Leech 格数学表示**：向量 y ∈ ℤ²⁴ 属于 Λ₂₄ 当且仅当 yⱼ = p + 2cⱼ + 4kⱼ，其中 p 是共享奇偶位，c 是 Golay 码 G₂₄ 的 codeword，且满足 Σkⱼ ≡ p (mod 2)。两个全局约束将搜索空间从 2²⁴ 压缩至 2¹² 个合法 codeword。
2. **三 octad 分割与 trellis**：固定三个不重叠的 octad（各含 8 个 1），将 24 维块切分为三个 8 维 section。每个 section 的 pattern byte 通过 trellis 路径确定，trellis 每层有 64 个状态节点，共 2¹² = 4096 条合法路径。
3. **查找表结构（18,816 B）**：
   - 16,384 B：rank 向量表，2,048 条 class 0 + 2,048 条 class 1 的最优秩向量
   - 2,304 B：trellis 状态转移表
   - 32 B：32 个 float 值存储 1/√(16m)
4. **解码流程**：48-bit 字包含 8-bit state（6-bit 状态 s₈ + 1-bit 共享奇偶 p + 1-bit 第 1 section 奇偶 r）、3 个 11-bit 行索引（每个 section 一个）、edge choices 和 gain bit。kernel 执行 6 次 load 获取 24 个坐标值，仅 1 次等待（s₁₆ 状态传递）。
5. **坐标值寄存器编码技巧**：每个坐标值存为 biased byte（val + 128），放置在 0x4b0000__ 的低字节，读取时作为 float 2²³ + byte，一次减法还原，避免 int-to-float 转换。
6. **行尺度微调**：σ_row 从旋转后行范数初始化，使用 KL 散度损失在 DCLM-Edu 文本上微调，lattice codes 保持不变，零额外字节。
7. **混合精度策略**：将量化损失最大的投影层（v_proj、o_proj、部分 down_proj）转为 int4，节省的 bytes 通过 4-bit embedding tables 补偿，整体仍维持 ~2.7 bits/parameter。

## 实验与结果
1. **基座模型与评估基准**：Qwen3-4B、8B、14B；MMLU（14,042 题，5-shot，micro-averaged）、GSM8K（1,319 题，zero-shot）。
2. **主要 MMLU 结果（Table 3）**：
   - 4B：TETRA 63.37 vs FP16 70.14（−6.77）vs AWQ 68.14（−4.76），113.8 tok/s，1.38 GB
   - 8B：TETRA 69.58 vs FP16 75.05（−5.48）vs AWQ 73.79（−4.21），95.0 tok/s，2.76 GB
   - 14B：TETRA 75.66 vs FP16 78.88（−3.22）vs AWQ 77.8（−2.46），57.2 tok/s，5.04 GB
3. **GSM8K 结果（Table 4）**：相对 FP16 误差分别为 −9.63、−4.62、−3.26 分；相对 AWQ 分别为 −6.52、−4.32、−3.34 分。4B 在 GSM8K 上损失大于 MMLU。
4. **跨尺寸比较（Table 5）**：8B TETRA（2.76 GB）比 AWQ 4B（2.67 GB）高 1.44 MMLU 分；14B TETRA（5.04 GB）比 AWQ 8B（6.10 GB）高 1.87 MMLU 分且少用 1.05 GB。
5. **与 2-bit 格式对比（4B）**：IQ2_XXS（2.48 bits/param）仅 39.78 MMLU，TETRA 高出 23.6 分；引用的 LLVQ 62.8、QTIP 3INST 59.5、QuIP# 52.9。
6. **Kernel 性能（Table 2）**：TETRA 每 weight 读取 2.148 bits，带宽 278 GB/s；AWQ 读取 4.179 bits，带宽 583 GB/s。TETRA 解码时间 3.424 ms vs AWQ 3.261 ms（略慢）。
7. **各步骤增益（Table 6）**：row scale 微调带来 +3.15/4B、+3.29/8B、+1.67/14B MMLU 分；int4 + 4-bit tables 再增 +2.26/4B、+1.42/8B、+1.46/14B 分。

## 相关工作脉络
1. **van der Ouderaa et al. [24]（LLVQ）**：首次将 Leech 格用于 LLM 量化，48-bit 码本含 10¹⁴ 点，但 GPU kernel 仅支持单层 shell 解码，且码本过大无法部署——Tetra 保留相同格结构但重构码本。
2. **QuIP# [22]**：使用 E₈ 格，理论 MSE 比 Λ₂₄ 高 9.0%，Tetra 论证 24 维格在相同密度下更具优势。
3. **QTIP [23]**：同样使用 trellis 解码，但其 hybrid code 采用 2 KiB 表和 computed codes（如 3INST），Tetra 使用更大（16 KiB）但更结构化的 Golay trellis 表。
4. **AQLM [7] / VPTQ [18]**：学习式向量量化，码本 ≤ 2¹⁶，解码依赖 cache 命中率——Tetra 采用解析式格码而非学习码本。
5. **IQ2 formats [11]**：llama.cpp 中的 2-bit 格式，使用 256–1024 点查找表，TETRA 在 4B 上 MMLU 高出 23.6 分。
6. **AWQ [17]**：4-bit 激活感知量化，group size 128，是本文主要对比基线；Tetra 以约一半的 bits/parameter 实现接近质量。
7. **GPTQ [9]**：列级后训练量化，推误差至后续列——Tetra 的 encoder 沿用 GPTQ-style loop 进行逐块优化。

## 局限性与未来方向
1. **单卡实验**：所有测量仅在 NVIDIA L40S 上进行，未在 A100 等其他 GPU 验证；sm_120 上较早版本 TETRA 速度为 Planes14 的 0.82–1.03 倍。
2. **单次校准数据**：仅使用 131,072 tokens（DCLM-Edu）校准，未评估多轮校准的稳定性。
3. **MMLU 评估偏差**：MMLU 在 dense reconstruction 上评估（非 served kernel），且 int4 投影选择在测试集上完成，存在 selection bias。
4. **GSM8K 过于简单**：FP16 得分 92–95，题目自 2021 年公开；未测试 Qwen3 的 thinking mode 或更难的数学基准。
5. **缺失指标**：未报告 sealed files 的 perplexity。
6. **推理限制**：仅 batch=1、短上下文场景，未评估长序列或多请求吞吐。
7. **未来方向**：作者计划测试 Qwen3-32B（需新 kernel 处理 102,400 B shared memory 需求）、增加第二组校准数据、在多 GPU 上验证。

## 研究启发与可借鉴点
1. **Treliis 解码 + 分段查表设计**：将高维格分解为多个低维子问题，利用代数结构（如 Golay 码）降低查找表规模，该思路可迁移至其他高维格（如 E₈ 堆叠）的量化部署。
2. **混合精度资源分配策略**：将 int4 用于量化损失最大的投影层，节省的 bytes 通过 4-bit embedding tables 补偿——这种"损失感知 + 嵌入式补偿"的资源调度可应用于其他混合精度量化框架。
3. **Pairwise 统计检验的严格实验设计**：使用 paired MMLU intervals（bootstrap 重采样 + McNemar 检验）确保比较有效性，预注册实验流程，这种做法值得在量化论文中推广。
4. **寄存器级编码优化**：将 coordinate 值编码为 biased byte 并存入特殊 float 位模式，利用硬件浮点解析能力避免显式 int-to-float 转换，此类微优化在 custom kernel 设计中具有通用价值。
5. **与团队方向结合机会**：若团队关注低比特 LLM 部署，可将 Tetra 的 trellis 解码框架与 AQLM/VPTQ 的学习式码本结合，探索"解析+学习"混合量化方案；或在不同格结构（如 D₂₄、E₈ⁿ）上验证 trellis 方法的通用性。

## 关键术语表
**Leech lattice（Λ₂₄）**：24 维最密球堆积格，已知最优格量化器之一，Tetra 在其上构建 2-bit 码本。
**Golay code（G₂₄）**：长度 24 的二元完美线性码，含 2¹² 个 codeword，Tetra 利用其 trellis 结构实现高效解码。
**Trellis**：分层图结构，codeword 对应一条路径，每层节点表示"状态"，Tetra 用它将 24 维搜索分解为 3 个 8 维子问题。
**Rank vector**：8 维坐标在有序列表中的位置索引，4 bits/coordinate，存储于 16 KiB 查找表中。
**Row scale（σ_row）**：每输出行的 f64 缩放系数，初始化为旋转后行范数，微调时保持不变量级。
**Sealed file**：经过完整训练（row scale + int4 投影 + 4-bit embedding）的最终压缩模型文件，约 2.7 bits/parameter。
**Int4 projection**：对量化损失最大的投影层（v_proj、o_proj、部分 down_proj）使用 4-bit 整数量化而非 lattice 编码。
**Retained retention**：理论极限码率与实测码率的比值百分比，Tetra 达 88.80%，Planes14 达 91.98%。

## 可复现要素
- **数据集**：校准数据为 DCLM-Edu（HuggingFaceTB/dclm-edu），取前 131,072 tokens；评估数据为 MMLU 和 GSM8K 公开测试集。
- **代码开源**：是，GitHub https://github.com/pjmalandrino/llvq，MIT/Apache-2.0 许可。
- **权重开源**：是，三个 sealed 文件的 SHA-256 已公布（886391a8...、7bdb9a55...、61db37fe...），但未提供直接下载链接。
- **测量日志**：全部公开于 GitHub，每个数字可追溯来源。
- **关键超参**：block size = 24，bits/weight = 2，trellis 状态数 = 64，表行数 = 4096，int4 group size = 128，embedding group size = 64，tile T = 64（L40S）。
- **硬件**：单一 NVIDIA L40S GPU。
