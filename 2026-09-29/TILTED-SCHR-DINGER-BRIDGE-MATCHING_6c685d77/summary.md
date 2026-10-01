---
title: "TILTED-SCHR-DINGER-BRIDGE-MATCHING"
source: https://arxiv.org/pdf/2609.34642v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 04:33:46"
field: "生成模型与最优输运"
keywords: ["Schrödinger Bridge", "reward tilting", "stochastic optimal control", "diffusion bridge", "post-training alignment", "adjoint matching"]
innovations: ["提出相对SOC formulation将预训练SB的奖励倾斜转化为增量控制学习，无需重新访问原始目标分布", "设计Controller-Corrector交替优化的TSBM算法并给出严格收敛证明", "首次在SB上实现精确边际奖励倾斜，在CelebA/MNIST上优于DPS/CondSMC等近似方法且无reward hacking"]
benchmarks: ["Colored MNIST", "CelebA 128x128"]
---

# 论文速读：TILTED SCHRÖDINGER BRIDGE MATCHING

## 一句话总结
本文提出了 **Tilted Schrödinger Bridge Matching (TSBM)**，一种针对预训练 Schrödinger Bridge 的后训练微调方法，通过交替优化 Controller 和 Corrector 匹配，将目标分布沿可微奖励函数进行指数倾斜（$p_1^r \propto p_1 e^r$），同时保持源边际分布 $p_0$ 不变。在 Colored MNIST 和 CelebA 的非配对图像翻译任务上验证了方法的有效性。

## 研究问题与动机
- **核心问题**：Schrödinger Bridge（SB）已广泛用于非配对域迁移，但在部署后需要根据人类偏好或物理约束调整输出分布，即"奖励倾斜"（reward tilting）问题。
- **现有方法不足**：已有大量针对 Flow Matching 和 Diffusion Models 的奖励微调工作，但 SB 的奖励微调几乎未被探索。
- **Diffusion Bridge 的特殊难点**：对于无记忆的扩散模型，终端奖励可直接产生目标倾斜分布；但 SB 是有记忆的（source-target 存在依赖），直接应用终端奖励会引入偏差，导致实际边际分布偏离期望的 $p_1^r$，可能产生伪影或奖励作弊（reward hacking）。
- **已有近似方法的局限**：DPS 等推理时对齐方法为近似方案，会引入不同形式的偏差；finetuning 方法如 PalSB 针对的是可评估能量的 Boltzmann 采样场景，与隐式终端密度的预训练 SB 倾斜问题本质不同。

## 核心贡献（创新点）
1. **理论 reformulation**：证明了以预训练桥 $P$ 为参考的相对随机最优控制（SOC） formulation 与原问题等价（Proposition 1），从而将倾斜问题转化为增量控制的学习问题，无需重新访问原始目标分布 $p_1$。
2. **Terminal Potential 的显式分解**：Proposition 2 推导出相对 SOC 的终端代价项 $f_{\text{rel}}^r$ 可直接用预训练桥的旧 Schrödinger 势 $\log\widehat{\varphi}_1^{\text{old}}$、新势 $\log\widehat{\varphi}_1^r$ 和奖励 $r$ 表示，显式消去了 $p_1$ 因子。
3. **交替优化算法 TSBM**：基于 Adjoint Matching 设计了 Controller Matching 与 Corrector Matching 的交替迭代算法（Algorithm 1），并给出了严格收敛证明（Theorem 1），保证 KL 散度趋于零。
4. **实验验证**：首次在 Colored MNIST（数字颜色/笔画粗细属性操纵）和 CelebA（人脸属性翻译）上验证了 TSBM 在保持源内容保真度的同时实现奖励导向的准确分布转移，优于 DPS、CondSMC、CondSNIS 等基线。

## 方法详解
**问题设定**：给定预训练的 Schrödinger 桥 $P = \text{SB}_R(p_0, p_1)$，希望学习一个新的桥 $P^r = \text{SB}_R(p_0, p_1^r)$，其中 $p_1^r(x) \propto p_1(x)e^{r(x)}$，$r(x)$ 为可微奖励函数。

**关键理论（相对 SOC  formulation）**：
- 以预训练桥 $P$ 为参考过程，定义增量控制器 $u_t$，使得受控扩散漂移为 $b_t + \sigma_t u_t$（$b_t$ 为预训练桥的最优漂移）。
- Proposition 3 表明：求解以下相对 SOC 问题的最优控制 $u_r^*$ 即为 TiltedSBP 的解：
$$u_r^* = \arg\min_u \mathbb{E}_{P^u}\left[\frac{1}{2}\int_0^1 \|u_t(X_t)\|^2 dt + \log\widehat{\varphi}_1^r(X_1) - \log\widehat{\varphi}_1^{\text{old}}(X_1) - r(X_1)\right]$$
- 终端代价梯度可写为：$\nabla f_{\text{rel}}^r(x) = h^r(x) - h_{\text{old}}(x) - \nabla r(x)$，其中 $h = \nabla\log\widehat{\varphi}_1$ 为 corrector。

**TSBM 算法（Algorithm 1）**：
- **初始化**：$u^{(0)} = 0$，$h^{(0)} = h_{\text{old}}$（冻结的预训练 corrector）。
- **Controller Update**（固定 $h^{(k)}$）：使用 Adjoint Matching，构建"lean" adjoint $\tilde{a}_t$，终端条件为 $\tilde{a}_1 = -\nabla r(X_1) + h^{(k)}(X_1) - h_{\text{old}}(X_1)$，沿轨迹反向传播 ODE，最小化匹配损失 $\mathcal{L}_{\text{ctrl}}$ 更新控制器 $u$。
- **Corrector Update**（固定 $u^{(k)}$）：使用 Corrector Matching，以当前控制器生成的端点对 $(X_0, X_1)$ 为样本，回归目标为 $\nabla_{x_1}\log p^R(X_1|X_0)$，更新 corrector $h$。
- 两个步骤交替进行 $K$ 个阶段（stage），每一阶段内分别进行 $M_{\text{ctrl}}$ 次 controller 更新和 $M_{\text{corr}}$ 次 corrector 更新。
- **收敛性**（Theorem 1）：在 Assumptions 1-2 下，精确 population 更新满足 $\text{KL}(P^{u^*} \| P^{u^{(k)}}) \to 0$。

**工程实现要点**：
- 预训练的 forward drift $b_\psi$ 和 corrector $h_{\text{old},\psi}$ 冻结；新的 trainable 网络 $b_\theta$ 和 $h_\phi$ 从预训练权重初始化，增量控制器隐式表示为 $u_\theta = (b_\theta - b_\psi)/\sigma_t$。
- 使用 replay buffer 存储采样轨迹和对应的 adjoint，定期刷新。
- 参考过程设为 Wiener 过程 $W^\epsilon$（$\sigma_t = \epsilon$）。

## 实验与结果
**数据集**：
- **Colored MNIST**：数字 "2" → "3" 非配对翻译，奖励函数为 red chroma（偏红）、thin stroke（笔画细，目标厚度 3.65）、thick stroke（笔画粗，目标厚度 7）。
- **CelebA**：128×128 男性→女性翻译，使用 ImageReward 定义的三个属性奖励：Natural Toothy Smile、Heavy Makeup、Elderly Woman。

**评估基线**：Pretrained DSBM、DPS（$\gamma \in \{0.5, 1, 2, 4\}$）、CondSNIS、CondSMC。

**主要结果**：
- **Colored MNIST**：TSBM 在 reward-faithfulness Pareto 前沿上表现最优，成功生成符合奖励指定的细/粗/红色数字，同时保留源颜色/厚度等无关属性；DPS 引入可见伪影，CondSMC 对目标属性的改变较弱。
- **CelebA（Table 1）**：
  - TSBM 在所有三个属性的 ImageReward 上均优于 pretrained DSBM（Natural Toothyl Smile: -1.2348→0.1398；Heavy Makeup: -1.4054→0.9146；Elderly Woman: -1.4960→-0.0204）。
  - TSBM 在 eval PickScore 上均为最高（Smile: 18.2345；Makeup: 18.0390；Elderly: 18.8200），表明泛化性好，无明显 reward hacking。
  - TSBM 在 Smile 和 Elderly 提示上 CLIP-IQA 最高（0.6595, 0.6340），Heavy Makeup 略低于其他方法。
  - DPS 在高 $\gamma$ 时训练 ImageReward 更高，但 eval PickScore 下降、CLIP-IQA 显著降低，存在明显 reward hacking 和视觉伪影。
- **推理效率（Appendix G.2）**：TSBM 推理时间与预训练 DSBM 基本相同（单输出 1.460s vs 1.468s），仅为 DPS（7.116s，4.88×）的 1/4.88，工作内存仅 0.180 GiB（DPS 为 2.229 GiB）。

## 相关工作脉络
1. **Schrödinger Bridge 求解器**：DSBM（Shi et al., 2023）、Entropic Neural Optimal Transport（Gushchin et al., 2023）、Adjoint SB Sampler（Liu et al., 2025a）。本文工作在 DSBM 预训练桥基础上进行 post-training，而非从头训练。
2. **Diffusion 奖励微调**：Adjoint Matching（Domingo-Enrich et al., 2025）、Flow-GRPO（Liu et al., 2025b）、RL finetuning（Black et al., 2024; Clark et al., 2024）。这些方法主要针对无记忆的 noise-to-data 扩散模型，不处理 source-target 依赖的桥模型。
3. **Diffusion Bridge 奖励引导**：CDDB/DPS（Chung et al., 2023b）通过数据一致性引导实现推理时条件倾斜，为近似方法；PalSB（Li et al., 2025）针对可评估能量的物理约束场景；本文首次针对预训练 SB 的隐式终端密度进行精确的奖励倾斜微调。
4. **Particle-based 推理时方法**：CondSMC（Del Moral et al., 2006）和 CondSNIS 通过粒子重加权实现条件倾斜，计算开销随粒子数线性增长，且为近似方法。
5. **Boltzmann 采样 SB**：Liu et al. (2025a)、Havens et al. (2025)、Guo et al. (2026) 研究的是具有可评估目标能量函数的 SB 奖励对齐，与本文的预训练 SB 隐式密度倾斜问题设定不同。

## 局限性与未来方向
- **理论假设的实用性**：论文假设 exact optimization 和所有理论假设成立，实际中有限采样和网络近似可能导致偏差。
- **可微奖励的限制**：方法要求奖励函数 $r(x)$ 可微，限制了应用场景；非可微奖励需扩展（如 Bergmeister et al., 2026 的方向）。
- **计算成本**：TSBM 训练需要重复的桥模拟和 adjoint ODE 求解，微调计算密集；蒸馏变体（Gushchin et al., 2025）可能降低此成本。
- **离散状态空间**：当前方法面向连续空间，未来可探索扩展到离散状态空间（如 Wang et al., 2025; So et al., 2026）。
- **CelebA Heavy Makeup 的 CLIP-IQA 下降**：说明某些复杂属性奖励可能降低图像自然度，需要更好的 fidelity-reward 权衡。

## 研究启发与可借鉴点
1. **相对参考变换技巧**：Proposition 1 证明以预训练桥 $P$ 为参考与以原始参考 $R$ 为参考等价，这为任何预训练桥模型的 post-training 适配提供了一般性框架——只需学习增量控制，避免从头训练。
2. **Schrödinger 势的差分表达**：Proposition 2 将未知的新势 $\log\widehat{\varphi}_1^r$ 与已知旧势 $\log\widehat{\varphi}_1^{\text{old}}$ 做差，巧妙消去了原始目标密度 $p_1$，这一技巧可迁移到其他类似问题中需要隐式密度调整的场景。
3. **交替 Controller-Corrector 优化的收敛保证**：Theorem 1 的证明策略（基于 ASBS/IPF 的 endpoint KL 递减）严谨且通用，其 half-bridge 分解思想（Propositions 4-5）可作为设计其他交替优化算法的理论模板。
4. **实验设计亮点**：同时使用 train reward（ImageReward）和 eval reward（PickScore）双重指标来检测 reward hacking；TSBM 推理成本与预训练模型相同的效率优势在实际部署中极具价值。
5. **与团队方向的结合机会**：若团队研究非配对图像翻译、领域适应或生成模型对齐，TSBM 提供了一种无需重新训练即可适配新偏好/约束的高效微调方案；可探索将其应用于科学模拟（如分子构型调整）、counterfactual explanation 等场景。

## 关键术语表
- **Schrödinger Bridge (SB)**：在两个给定边际分布约束下，寻找相对于参考扩散过程 KL 散度最小的路径分布，连接 entropic optimal transport 与扩散生成模型。
- **Reward Tilting**：通过将目标分布乘以 $e^{r(x)}$ 并归一化，使生成模型输出向奖励函数 $r(x)$ 指示的属性偏移，$p_1^r(x) \propto p_1(x)e^{r(x)}$。
- **Stochastic Optimal Control (SOC)**：通过添加控制项 $u_t$ 到 SDE 漂移中，最小化控制能量加终端奖励的期望，从而调整扩散过程的输出分布。
- **Adjoint Matching**：一种基于"lean" adjoint state $\tilde{a}_t$（通过反向 ODE 传播）的 SOC 求解算法，通过匹配损失学习最优控制器，具有梯度截断的可扩展性。
- **Schrödinger Potential**：分为 forward potential $\varphi_t$ 和 backward potential $\widehat{\varphi}_t$，其乘积给出桥过程在时刻 $t$ 的边际密度 $p_t = \varphi_t \widehat{\varphi}_t$。
- **Controller Matching**：固定 corrector 时，通过 Adjoint Matching 更新控制器 $u$ 的步骤，最小化含终端 adjoint 目标的匹配损失。
- **Corrector Matching**：固定控制器时，回归终端正确项 $h = \nabla\log\widehat{\varphi}_1$ 的步骤，目标为参考过程的条件转移 score。
- **Diffusion Bridge**：连接给定源分布和目标分布的条件扩散过程，源和目标之间存在依赖关系（不同于无记忆扩散），Schrödinger Bridge 是其熵正则化版本。

## 可复现要素
- **数据集**：Colored MNIST（基于标准 MNIST，论文引用 Gushchin et al., 2024b 的设置）；CelebA（128×128，标准公开数据集）。
- **代码开源**：论文未明确声明代码开源，但提到使用 DSBM 官方 GitHub 仓库（Appendix E）。
- **权重开源**：预训练 DSBM 权重未声明开源；TSBM 权重以 checkpoint 形式在实验中训练。
- **关键超参**：
  - DSBM 预训练：20 stages，$\sigma_t = 1$，Adam lr=$10^{-4}$，EMA=0.999。
  - TSBM 微调：5 stages（MNIST/CelebA）或 20 stages（2D toy），$\lambda=1000$（MNIST）或 $\lambda=30$（CelebA），EMA=0.99，Adam lr=$10^{-5}$（CelebA）或 $3\times10^{-5}$（MNIST strokes）。
  - NFE：MNIST 30，CelebA 100。
  - 控制器/replay buffer：MNIST 每 50 步刷新，CelebA 每 100 步刷新。
