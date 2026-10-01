---
title: "WHEN-TEXT-MATTERS-DESIGN-PRINCIPLES-FOR-VISUAL-TOKEN-PRUNING"
source: https://arxiv.org/pdf/2609.34861v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 04:36:09"
---

# 论文速读：WHEN-TEXT-MATTERS-DESIGN-PRINCIPLES-FOR-VISUAL-TOKEN-PRUNING

## 一句话总结
本文提出了一种免训练的视觉Token剪枝方法DeFT，通过“早期纯视觉粗筛 + 解码器中点文本精筛”的两阶段时序解耦设计，解决了现有VLM剪枝方法因过早依赖单模态信号而误删关键视觉证据的问题，在80%和90%剪枝率下平均Dense-relative性能恢复较最强基线分别提升11.10和16.84个百分点。

## 研究问题与动机
1. **计算开销瓶颈**：大视觉语言模型（VLM）的高分辨率图像输入会产生海量视觉Token，显著拉长LLM的序列长度并推高推理成本，视觉Token剪枝成为实用的加速手段。
2. **现有方法的固有缺陷**：基于图像的剪枝（如ZOO-Prune）仅依赖视觉显著性，容易丢弃问题要求的具体局部细节（如图表中的品牌名或数字）；基于文本的剪枝（如SparseVLM）在解码器早期应用text-to-visual attention时，尚未充分建模跨模态关系，导致相关支撑信息被保留而关键数值/标签被剔除。
3. **核心科学问题**：文本引导信号在VLM解码器的哪个深度最能有效识别与问题答案相关的视觉区域？如何在不重新训练模型的前提下，安全地利用该信号进行Token重选？
4. **动机来源**：通过region masking重要性实验发现，text-to-visual attention在解码器早期层对答案相关区域的排序能力较弱，但在中间层显著提升；这促使作者提出将文本引导推迟至中点，前期仅用视觉信号快速降维并保留候选池，以实现效率与精度的最优平衡。

## 核心贡献（创新点）
1. **揭示文本引导信号的深度依赖性**：首次系统量化text-to-visual attention随解码器深度的变化规律，证明其在中间层（midpoint）对答案相关视觉区域的识别能力最强，为剪枝时序设计提供了理论依据。
2. **提出DeFT两阶段免训练剪枝框架**：早期仅使用vision-encoder attention进行粗剪枝并保留扩展候选集；在解码器中点利用input text tokens的cross-attention从候选集中重选最终K个Token，严格分离两种模态信号的作用阶段。
3. **建立“视觉初筛+文本精筛”的设计原则**：与现有方法将文本信号前置或全程融合的本质不同，本文证明了延迟文本介入可大幅减少关键证据的误删，尤其在激进剪枝（80%-90%）下优势呈指数级放大。
4. **提供严谨的计算预算对齐与效率评估**：引入token-block计数 $B_{\mathrm{vis}} \approx DK + \alpha L(N-K)$ 作为公平对比基线，证明该方法在LLM-prefill延迟不增加甚至更低的情况下实现性能突破。

## 方法详解
- **整体架构**：免训练、两阶段、无需额外参数更新。
- **阶段一：视觉编码器引导的初始剪枝与候选保留**
  在进入LLM前，计算vision-encoder的head-averaged注意力分数 $s^{\mathrm{img}}$：若编码器含CLS token则使用CLS-to-visual注意力，否则使用visual self-attention。不直接截断至目标数$K$，而是保留额外候选：$M = \min\{N, K + \lceil \alpha(N-K) \rceil\}$，默认 $\alpha=0.2$。候选集 $\mathcal{C} = \mathrm{TopK}_M(s^{\mathrm{img}})$ 包含初始Top-K及额外的$\alpha$比例Token。
- **阶段二：中点文本引导重选**
  将$M$个候选送入解码器至中间层 $L = D/2$，此时计算input text tokens作为Query、候选Visual tokens作为Key的text-to-visual attention分数 $s^{\mathrm{text}}$。最终Token集由 $\mathcal{S} = \mathrm{TopK}_K(s^{\mathrm{text}}|_{\mathcal{C}})$ 确定，即**仅从候选池中按文本注意力分数重新挑选最终K个**，完全丢弃其余视觉Token。
- **关键原理**：避免在早期解码层引入文本信号（实验证明早期文本注意力会干扰视觉筛选）；通过 $\alpha$ 控制候选池大小，在极低成本下为后期重选保留“安全网”，使原本视觉不显著但问题高度相关的Token有机会进入最终集合。

## 实验与结果
- **模型与基线**：Qwen3-VL-4B、Qwen3-VL-8B、LLaVA-OneVision-1.5-8B；对比FastV、SparseVLM、VisPruner、ZOO-Prune、RESTORE，以及Progressive剪枝PyramidDrop、FlowCut。
- **基准测试**：TextVQA、ChartQA、InfoVQA、AI2D、MMMU、MMStar、NoCaps、TextCaps（共8项，覆盖场景文本、图表推理、多模态常识与开放描述）。
- **主要结果**：在80%和90%剪枝率下，本文方法在三个模型上平均Dense-relative performance recovery较最强基线分别提升**11.10**和**16.84**个百分点。以Qwen3-VL-8B为例，90%剪枝下得分为90.65%，而所有竞品均低于70%。
- **任务特异性突破**：在依赖局部精确证据的TextVQA、ChartQA、InfoVQA上提升最显著（80%剪枝下较基线分别提升8.16、11.59、15.97 pp；90%下进一步提升至19.26、19.91、18.88 pp）。
- **效率表现**：LLM-prefill延迟与多数基线相当或更低（如80%剪枝下约121.20ms，Dense为226.66ms），验证了实际部署可行性。

## 相关工作脉络
1. **FastV / VisionZip（早期视觉剪枝）**：仅依赖视觉编码器或解码器早期自我注意力剔除冗余Token，缺乏任务条件引导，在需要精细定位的任务（如TextVQA）上失效严重。
2. **SparseVLM / Pyramiddrop（渐进式文本引导）**：在解码器各层动态递减Token数，但早期层中文本引导信噪比低，且 progressive reduction 难以精准对齐最终答案所需的局部稀疏证据。
3. **ZOO-Prune / VisPruner（零阶梯度/多样性筛选）**：利用零阶估计或特征多样性排序，侧重视觉冗余去除而非语义相关性，对指令绑定的细微视觉线索敏感度不足。
4. **RESTORE / CRISPRune（内部校正机制）**：在语言模型层内引入位置或注意力校正来修复剪枝失真，计算开销较大，且未解决“何时引入文本信号最有效”的时序优化问题。
5. **本文定位差异**：不同于上述方法将文本信号前置、全程融合或依赖额外训练/校正模块，本文确立了“视觉初筛 → 中点文本精筛”的时序解耦原则，以极简的免训练设计实现了性能恢复的显著跃升。

## 局限性与
