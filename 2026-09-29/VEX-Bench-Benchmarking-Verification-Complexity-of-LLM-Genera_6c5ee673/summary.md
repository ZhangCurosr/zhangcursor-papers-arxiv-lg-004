---
title: "VEX-Bench-Benchmarking-Verification-Complexity-of-LLM-Genera"
source: https://arxiv.org/pdf/2609.35028v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 16:49:42"
---

# 论文速读：VEX-Bench: Benchmarking Verification-Complexity of LLM-Generated Misinformation

## 一句话总结
本文提出VEX-Bench，首个从筛选视角量化LLM生成虚假信息“验证复杂性”的统一基准。通过五维评分与VEX分数揭示：高收益生成与高验证负担之间存在根本解耦，且生成成本较核查成本低3～169倍，暴露出现有安全评估对下游核查系统资源挤占风险的盲区。

## 研究问题与动机
- **现有评估脱离下游真实影响**：主流基准仅在生成端度量越狱成功率、有害性评分或检测难度，无法刻画虚假信息流入新闻/社交平台后对现实核查资源的消耗。
- **核查本质是分诊（triage）过程**：媒体与事实核查机构受时间、人力与预算严格约束，高优先级但难验证的内容极易引发系统性资源错配，而现有框架缺乏对该风险的量化指标。
- **check-worthiness缺乏统一定义**：不同组织对“值得核查”的判定标准差异显著，亟需一个融合新闻学与实践经验、支持多维度加权且可跨场景复用的评估协议。
- **生成-验证成本严重不对称**：LLM每周可生成约30万亿token，虚假信息制造成本极低，而多步证据检索、跨源比对与专家咨询的验证成本持续攀升，需要实证刻画这一结构性风险。

## 核心贡献（创新点）
- **提出VEX-Bench统一基准**：将“验证复杂性”确立为独立评估轴，首次从筛选感知角度量化LLM生成虚假信息对核查容量的挤压，区别于传统成功导向或内容属性导向的评测。
- **形式化五维度量与VEX分数**：基于事实核查实践定义D1–D5，并通过遍历所有非空维度子集（31种组合）计算综合风险分，使评估可适配不同核查机构的优先级设定。
- **建立内容分析法支撑的裁判协议**：引入人类标注作为参考标准，使用序数Krippendorff α与Spearman ρ验证人机一致性，解决主观维度难以对齐传统客观标签的方法学难题。
- **揭示成功度与验证复杂性的解耦现象**：实证表明高SR/NR的方法未必产生高VEX内容，且最高风险输出并非始终来自最强生成方法或最高风险模型，证明多维度评估的必要性。
- **量化生成-核查成本不对称**：给出精确的单文章成本分解，证明高效生成方法能以 negligible marginal cost 产出高VEX内容，而核查成本高出3～169倍，凸显资源受限系统的脆弱性。

## 方法详解
- **基准设置**：以三元组 $(t, d, q)$（task, domain, topic）固定评估条件，确保模型与方法对比的公平性。涵盖2个任务类别（Fabrication：从零生成虚假文章；Rewrite：基于真实源文章改写为误导版本）与6个高风险领域（health, politics, public safety, finance, identity groups, environment），每领域10个基准中性主题，共60个主题。
- **结构化输出规范**：强制要求JSON格式输出，包含 `topic`, `headline`, `body`, `format`, `source`, `contact`, `date`, `claim` 字段；Rewrite任务额外输出 `changes_made`。保留来源/联系/日期等线索以支撑D3/D4评估。
- **五维评估协议（1–5分制）**：
  - **D1 Checkability**：衡量文章是否包含可证伪的具体主张，而非纯意见或修辞。
  - **D2 Harm Significance**：衡量被相信后可能触发的可行动危害路径及其规模/不可逆性。
  - **D3 Source Credibility**：衡量文章引用的机构、政府、报告或专家等制度权威信号密度（关注筛选阶段的感知可信度，非实际合法性）。
  - **D4 Imposter Legitimacy**：衡量文章在标题、结构、归属模式与排版上模仿正统新闻业的能力。
  - **D5 Verification Cost**：衡量验证所需的总体努力，从公开源快速核查至需专家咨询或受限数据访问。
- **VEX Score公式**：
  $$\mathbf{VEX}_{\mathcal{C}} = \mathrm{SR} \times \mathrm{NR} \times \frac{1}{|\mathcal{C}|} \sum_{i \in \mathcal{C}} D_i, \quad \varnothing \neq \mathcal{C} \subseteq \{D_1,\dots,D_5\}$$
  所有项归一化至$[0,1]$，$\mathcal{C}$取全部31种非空子集，默认聚合所有维度。等权假设便于风险可视化，实际可按机构优先级调整权重。
- **裁判与验证管线**：主裁判使用GPT-5.2，辅以Opus-4.7与Gemini-3.1进行鲁棒性校验。提示词开发遵循内容分析方法论，选取5篇困难样本作为few-shot锚点，通过100篇保留文章进行人类对齐与$\alpha$监控。部署Claude Sonnet 4.6 + Web Search事实核查Agent，输出Claim Support与Entity Integrity作为下游验证参照信号。

## 实验与结果
- **数据集与规模**：7个前沿LLM（Claude Sonnet 4.5, Gemini 3.1 Pro, GPT-5.4, Qwen3.5-Flash, Grok-4.1-Fast, Kimi-K2.5, DeepseekV4-Pro）× 7种生成方法（Direct Prompt, DisinfoCap, ISC, JailNewsBench, MisinfoQA, PoisonedRAG, PAP）× 2任务 × 60主题 = **5,880篇**文章。
- **评估基线**：JailNewsBench (JNB) judge、StrongREJECT；事实核查Agent（Claude Sonnet 4.6 + SerpAPI）。
- **裁判验证**：人类-人类、人类-LLM、LLM-LLM两两比较均达到tentative至reliable阈值（$\alpha \geq 0.667$），D3（来源可信度）人机一致性强于0.800。VEX维度与JNB呈弱-中度相关（Spearman），证明捕捉的是独立信号；D5与“证据不足主张比例”正相关，D3与Entity Integrity正相关，验证感知复杂度与下游验证难度的对齐。
- **核心发现**：
  - **任务诱导不同复杂度画像**：Rewrite继承真实源文章制度线索，D3/D4显著高于Fabrication；Fabrication缺少源锚点，D5成为主要风险轴。
  - **成功度与验证复杂性解耦**：Fabrication中ISC、DisinfoCap、MisinfoQA的SR/NR均≥95%，但VEX分布差异大；PAP成功率较低，但成功输出在D1/D5上最高，说明成功率导向评估会低估高风险方法。
  - **最高风险无单一主导**：方法级ISC整体风险最高；模型级DeepSeek-V4-Pro风险最高；交互层面DisinfoCap+Grok-4.1产出最高VEX内容（62种配置中占11次第一）。
  - **成本不对称量化**：按单篇有效输出计，Grok-4.1-Fast生成成本仅约0.13¢，核查成本约21.60¢，比例达**169.0×**；Qwen3.5-Flash为147.2×；整体平均核查成本是生成成本的12.4×。
- **最强结果**：ISC在Rewrite任务上VEX达46.4%，Claim Support仅下降17.4%（78.7%），Entity Integrity保持72.2%；DeepSeek-V4-Pro在各方法下均表现最高综合风险。

## 相关工作脉络
