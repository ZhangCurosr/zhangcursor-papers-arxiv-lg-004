---
title: "The-Hidden-Ratio-in-Adam-Stable-Structure-Compression-and-Si"
source: https://arxiv.org/pdf/2609.35392v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 11:07:21"
field: "深度学习优化器"
keywords: ["Adam优化器", "低精度量化", "tied-beta", "优化器分析", "Signum", "模型压缩"]
innovations: ["发现Adam在tied-beta下的变换比值y_t具有稳定重尾分布", "推导y_t标量递推实现4-bit固定码本压缩无需辅助缩放", "建立Adam与Signum间的解析学习率转移规则"]
benchmarks: ["FineWeb pre-training", "UltraChat SFT", "UltraFeedback RLHF"]
---

# 论文速读：The-Hidden-Ratio-in-Adam-Stable-Structure-Compression, and-Sign-Dynamics

## 一句话总结
本文在 tied-β（β₁=β₂）条件下将 Adam 的自适应行为重新参数化为一个变换比值 y_t，证明其具有稳定的重尾分布，并在此基础上实现了无需辅助缩放因子的 4-bit 固定码本压缩，同时揭示了 Adam 与 Signum 之间的学习率转移关系。

## 研究问题与动机
1. **Adam 自适应机制缺乏清晰的可解释性**：尽管 Adam 是深度学习默认优化器，但其一阶/二阶矩 EMA 之间的交互难以解释，导致跨规模、跨任务的分析框架缺失。
2. **原始矩量 m_t、v_t 的尺度敏感性强**：原始矩值随梯度尺度剧烈变化，缺乏可迁移的稳定表征。
3. **非 tied-β 下 v_t - m_t² 可为负**：当 β₁ ≠ β₂ 时，v_t - m_t² 可能为负，破坏 Adam 分母的统计解释；tied-β 是唯一保证该差值始终非负的 regime。
4. **优化器状态压缩需要稳定表征**：已有低精度优化器方法多直接量化原始状态，缺乏对不变结构的利用，导致压缩后性能损失较大。

## 核心贡献（创新点）
1. **发现变换比值 y_t 的稳定分布**：在 tied-β 下将 Adam 分解为 sign(m_t)/√(1+β/(1−β)·y_t)，y_t 跨任务/模型/训练阶段呈紧凑主体+持久重右尾的稳定分布。*与已有工作的区别在于首次提取出 Adam 分母中独立于确定性尺度的不变量。*
2. **推导 y_t 的标量递推公式**：得到 y_t = β x_t² y_{t−1} + (1−x_t)²（其中 x_t = m_{t−1}/m_t），使 (m_t, y_t) 成为 Adam 的等价参数化。*本质区别于直接在 (m_t, v_t) 上做量化的方法。*
3. **实现 4-bit 固定码本压缩且无需辅助缩放**：使用 16 级 FP4 风格码本存储 y_t，在 pre-training/SFT/RLHF 全流程上保持与全精度 Adam 相近性能。*这是首个无需 per-block/per-tensor 缩放因子即达到 4-bit 精度的 Adam 压缩结果。*
4. **建立 Adam 与 Signum 间的学习率转移规则**：将 y_t 替换为常数 y_const ≈ 3 可恢复 Signum-like 更新，并给出基于期望衰减匹配的学习率换算公式（式 11–13）。*将两者从经验调参关联提升到解析层面。*

## 方法详解
1. **变换比值定义**：令 r_t = (v_t − m_t²)/m_t² ≥ 0（tied-β 下保证非负），再定义 y_t = (1−β)/β · r_t，从而 Adam 更新方向写作 sign(m_t)/√(1 + β/(1−β)·y_t)。
2. **理论分解（Proposition 1）**：在局部高斯梯度假设下，y_t = Y_a·(1 + η_t/2)，其中 Y_a = 2/(Z+a)² 为平移逆平方参考（主导分布形状），η_t 为耦合+涨落修正项（小量）。固定参考 Y_c = 2/(Z+c)² 与混合参考在生存函数上二阶匹配。
3. **递推公式推导**：由 g_t = m_t(1−βx_t)/(1−β) 代入 v_t 递归，消去 g_t 后得到 y_t = β x_t² y_{t−1} + (1−x_t)²，实现仅用 (m_t, y_t) 追踪 Adam 状态。
4. **4-bit 固定码本量化**：截断 y_t 至 [0, 64]（cut64），再用固定非均匀码本 C = {0.25, 0.5, …, 64}（16 级）作最近邻投影，重建分母时直接代入 Eq.(4)，无需额外缩放因子。
5. **Signum-like 极限与学习率转移**：令 y_t → y_const，得 Δθ ≈ −η_adam·(1+β/(1−β)·y_const)^(−1/2)·sign(m_t)，通过期望衰减匹配（式 12）确定 y_const ≈ 3（跨 β∈[0.90, 0.97] 稳定）。

## 实验与结果
- **数据集与模型**：FineWeb（pre-training）、UltraChat（SFT）、UltraFeedback（RLHF/ReMax）；Llama-style 20M、Pythia-style 160M/1B、Llama-3.2-1B。
- **评估基线**：全精度 tied-β Adam（FP32）、TR-Adam(FP32-m_t, 4-bit y_t)、TR-Adam(FP8-m_t, 4-bit y_t)、Signum。
- **主要结果**：
  - **Pre-training**（Table 2）：Llama-20M β=0.95 时 TR-Adam(FP32) 与 Adam 完全一致（3.672 vs 3.672）；Pythia-1B β=0.92 时 TR-Adam 反而优于 Adam（2.977 vs 2.996，Δ=−0.019）。
  - **SFT**（Table 3）：Pythia-1B β=0.95 时 TR-Adam 优于 Adam（1.299 vs 1.309，Δ=−0.010）；Llama-3.2-1B 各 β 值均一致或略优。
  - **RLHF/ReMax**（Table 4）：TR-Adam 在全部 β 值下超越 Adam，β=0.90 时 Δ=+0.036（0.825 vs 0.788）。
  - **学习率敏感性**（Fig. 7）：4-bit TR-Adam 的 LR 最优区间与 Adam 高度对齐，跨 β∈{0.90, 0.92, 0.95, 0.97} 均成立。
- **最强结果**：RLHF 下 β=0.90，TR-Adam(FP32-m_t, 4-bit y_t) 达 0.825，较 Adam 提升 +0.036；pre-training 下 Llama-20M β=0.95 实现零误差匹配。

## 相关工作脉络
1. **Orvieto & Gower (2025) [31]**：提出 Adam 分母可解释为信噪比修正，本文在此基础上进一步提取出稳定分布的 y_t 变量。
2. **Fernández-Hernández et al. (2026) [16]**：证明 tied-β 是 Adam 梯度尺度不变性的充要条件，本文从结构分析出发将其用于压缩设计。
3. **Cattaneo & Shigida (2026) [6]**：研究 mini-batch 噪声下 β₁/β₂ 匹配对验证性能的影响，与本文 tied-β 选择动机互补。
4. **Signum / SignSGD with Momentum [3, 38]**：传统 sign-based 方法；本文揭示其为 Adam 的常数 y_t 极限情况，并提供学习率转移规则。
5. **Laprop [27]**：分离 Adam 的动量与适应性，本文同样分离但聚焦于分母的比值结构而非独立参数化。
6. **FP8-LM [35], Chitsaz et al. (2024) [8]**：低精度训练相关工作；本文创新在于无需 per-block 缩放即可对 Adam 状态做 4-bit 压缩。

## 局限性与未来方向
1. 实验局限于 Transformer 语言模型（≤1B 参数），未验证至更大规模或非 Transformer 架构（如 Vision、RL 策略网络）。
2. tied-β 假设下推导，标准 Adam（β₁=0.9, β₂=0.999）的 y_t 分布行为未分析。
3. 理论分布模型基于局部高斯窗口假设，真实训练中梯度分布更复杂，实际边界条件有待验证。
4. 论文未提供正式开源代码/数据（仅匿名压缩包提交），复现路径受限。
5. 未系统比较与其他低精度优化器（如 QAdam、AdaQuant）的优劣。

## 研究启发与可借鉴点
1. **"提取不变量→验证稳定性→再做压缩"** 的研究范式可迁移：在优化器设计/分析中，先寻找尺度无关的不变量（如 y_t），再针对其分布特性设计低精度表示，而非直接量化原始状态。
2. **固定非均匀码本替代动态缩放**：16 级 FP4 码本实现 4-bit 精度且无需 per-block scaling，为其他优化器状态压缩提供可参考方案。
3. **Adam↔Signum 学习率转移公式**（式 11–13）可直接用于已有 Signum 应用的场景快速调参，或作为两方法联合消融的实验设计模板。
4. **递推重参数化技巧**：将 v_t 替换为关于 y_t 的标量递推（Eq.10），可用于降低优化器状态内存占用，类似思路可探索于 AdaGrad/RMSProp 变体。
5. **分布稳定性诊断作为压缩前置检验**：本文 Fig.2 展示不同训练阶段/模型规模的分布一致性，这种"先画分布再定码本"的流程值得复用。

## 关键术语表
**Tied-β regime**：β₁=β₂ 的 Adam 配置，此时 v_t − m_t² 恒为非负，分母具有清晰的信噪比解释。
**Transformed ratio y_t**：y_t = (1−β)/β · (v_t − m_t²)/m_t²，提取自 Adam 分母的尺度无关变量，具稳定重尾分布。
**Shifted inverse-square reference Y_c**：Y_c = 2/(Z+c)²（Z~N(0,1)），作为 y_t 分布的理论参考，揭示其主导形状由逆平方奇点决定。
**TR-Adam**：本文提出的基于递推 y_t 的低精度 Adam 变体，支持 4-bit y_t 存储及 FP8 m_t 存储。
**Signum-like limit**：将 y_t 替换为常数 y_const 时的 Adam 退化形式，等价于带固定衰减的 Signum 更新。
**Cut64 截断**：将 y_t 上限截断至 64 的预处理，消除极少长尾值，对性能几乎无影响。
**FP4-like 固定码本**：16 级非均匀码本 C={0.25, 0.5, …, 64}，密度集中在 y_t 主体分布区。

## 可复现要素
- **数据集**：FineWeb、UltraChat、UltraFeedback（均已公开引用，可合法获取）
- **代码**：论文声明提供匿名压缩包（supplemental material），具体仓库链接论文未提及
- **关键超参**：β ∈ {0.90, 0.92, 0.95, 0.97}；LR（pre-training 需网格搜索；SFT 用 1e−5；RLHF 用 1e−6）；4-bit 码本 C 见正文；weight decay SFT/RLHF 分别为 0/0.01；batch size 见 Table 5–7
- **硬件**：1B-scale 实验用 8×H200 GPU；小规模实验用 16×RTX 4090 GPU
