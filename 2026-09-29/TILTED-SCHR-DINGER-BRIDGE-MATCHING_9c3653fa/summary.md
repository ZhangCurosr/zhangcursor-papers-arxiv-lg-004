---
title: "TILTED-SCHR-DINGER-BRIDGE-MATCHING"
source: https://arxiv.org/pdf/2609.34642v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 04:33:43"
field: "生成模型与最优输运"
keywords: ["Schrödinger Bridge", "Reward Tilting", "Stochastic Optimal Control", "Adjoint Matching", "Image Translation", "Post-training Fine-tuning"]
innovations: ["提出以预训练桥P为参考的相对SOC形式化，消去对p1显式依赖", "交替Controller-Corrector Matching算法TSBM，保证exact marginal reward tilting", "证明TSBM在KL意义下收敛至TiltedSB解"]
benchmarks: ["Colored MNIST", "CelebA 128x128"]
---

# 论文速读：TILTED SCHRÖDINGER BRIDGE MATCHING

## 一句话总结
本文提出 TSBM（Tilted Schrödinger Bridge Matching），一种针对预训练 Schrödinger 桥的**后训练 reward-tilting 微调方法**，在保持源分布 $p_0$ 不变的前提下，将桥模型的目标分布调整为reward加权的 $p_1^r \propto p_1 e^{r}$；理论证明与 Colored MNIST、CelebA 实验验证了其在无配对图像翻译中对属性（笔画粗细、颜色、面部特征）的高效引导。

## 研究问题与动机
- **Schrödinger 桥的奖励微调尚未被系统研究。** 尽管 diffusion model 和 Flow Matching 的 reward finetuning 已有大量工作，但 Schrödinger Bridge 的 exact marginal reward tilting 仍属空白。
- **Bridge 模型的 $X_0$–$X_1$ 依赖性使终端 reward 产生偏差。** 对无记忆 diffusion 模型直接应用 SOC 可精确得到 $p_1^r$；但 SB 是 memoryless 的，终端 reward 会引入不可控的偏差，导致目标边际与期望不符。
- **现有近似方法存在 artifacts 或 reward hacking。** DPS、CondSMC 等 inference-time 方法通过条件指数加权近似的 terminal marginal，易在强引导下产生伪影；reward-tilting 精度不足。
- **应用场景需要"冻结预训练桥 + 增量微调"范式。** 用户常希望在预训练桥基础上满足人类偏好或物理约束，而非从头重新训练整条桥。

## 核心贡献（创新点）
1. **理论重述（Prop. 1–3）：** 证明以预训练桥 $P$ 为参考与以基础扩散 $R$ 为参考的 TiltedSBP 等价（$\mathrm{SB}_P(p_0,p_1^r)=\mathrm{SB}_R(p_0,p_1^r)$），并将相对 SOC 终端代价表达为 $f_{\text{rel}}^r = \log\widehat\varphi_1^r - \log\widehat\varphi_1^{\text{old}} - r + C$，**消除了对 $p_1$ 的显式依赖**。
2. **TSBM 算法：** 基于 Adjoint Matching 提出交替优化的 Controller Matching + Corrector Matching 框架（Algorithm 1），在已知 $b_t, h_{\text{old}}$ 和 reward $r$ 的条件下学习增量控制 $u_t$。
3. **收敛性定理（Thm. 1）：** 在 Assumptions 1–2 下，TSBM 的迭代更新满足 $\mathrm{KL}(P^{u^*}\|P^{u^{(k)}})\to 0$，即精确收敛到 TiltedSB 解。
4. **实验验证：** 在 Colored MNIST（笔画粗细/颜色）和 CelebA（面部属性）上超越 DPS、CondSNIS、CondSMC 等基线，在 reward-faithfulness  Pareto 前沿上表现最优；TSBM 推理速度与预训练桥相同（约 1×），远低于 DPS（~5×）和 CondSMC（~10×）。

## 方法详解
**问题定义（Def. 1）**：给定参考扩散 $R$、源 $p_0$、目标 $p_1$ 和 reward $r$，寻找 $u^*$ 使得受控过程 $X_0\sim p_0,\; X_1\sim p_1^r\propto p_1 e^{r}$，并最小化控制能量 $\frac12\int_0^1\|u_t\|^2 dt$。

**相对 SOC 形式化（Prop. 3，Eq. 18）**：
$$
u_r^*=\arg\min_u\; \mathbb{E}_{P^u}\!\left[\frac12\int_0^1\|u_t\|^2 dt+\log\widehat\varphi_1^r(X_1)-\log\widehat\varphi_1^{\text{old}}(X_1)-r(X_1)\right],
$$
其中 $P$ 的 drift 为 $b_t=f_t+\sigma_t u_t^{\star}$（预训练桥漂移），$u$ 是增量控制。

**关键恒等式（Prop. 2，Eq. 16）**：
$$
\nabla f_{\text{rel}}^r(x)=h^r(x)-h_{\text{old}}(x)-\nabla r(x),
$$
其中 $h^r=\nabla\log\widehat\varphi_1^r$ 为待学 corrector，$h_{\text{old}}$ 来自预训练桥。

**TSBM 算法（Algorithm 1）**：交替更新 Controller $u$ 和 Corrector $h$：
- **Controller Update（Eq. 19–20）**：以 Lean Adjoint $\tilde a_1=-\nabla r+h^{(k)}-h_{\text{old}}$ 为终端条件，沿轨迹反传 ODE（Eq. 5）后对 $u$ 做梯度步。
- **Corrector Update（Eq. 21）**：在固定 $u^{(k)}$ 下，对 $h$ 最小化 $\|h(X_1)-\nabla_{x_1}\log p^R(X_1|X_0)\|^2$，即回归参考过程的 endpoint transition score。

**实现细节（App. C）**：冻结预训练 $b_\psi$ 和 $h_{\text{old},\psi}$；可训练的完整 forward drift $b_\theta$ 和 corrector $h_\phi$ 初始化自预训练网络；通过 replay buffer 存储轨迹和 lean adjoint 用于 Controller 更新；参考过程设为 Wiener 过程 $W^\epsilon$（$\sigma_t=\epsilon$）。

## 实验与结果
**数据集**：
- **Colored MNIST**：digit "2"→"3" 无配对翻译；reward 设定为 thin-stroke（目标厚度 3.65）、thick-stroke（目标厚度 7）、red-chroma（红色通道增强）。
- **CelebA**：128×128 male→female 翻译；使用 ImageReward 定义三个 prompt-based reward："Natural Toothy Smile"、"Heavy Makeup"、"Elderly Woman"。

**基线方法**：Pretrained DSBM（零 reward 微调）、DPS（$\gamma\in\{0.5,1,2,4\}$）、CondSNIS（$K=4,8$）、CondSMC（$N=4,8$）。

**主要结果**：
- **Colored MNIST**（Fig. 3）：TSBM 在 reward↑ 与 faithfulness↓（色调误差/厚度偏差）上达到最优 Pareto 前沿；DPS 在高 $\gamma$ 下出现可见伪影；CondSMC 属性改变较弱。
- **CelebA**（Table 1）：TSBM 在三个 prompt 组均获得最高 eval PickScore；"Natural Toothy Smile" ImageReward=**0.1398**（对比 Pretrained=-1.2348，DPS γ=4=0.9783 但引入伪影）；"Heavy Makeup" ImageReward=**0.9146**（PickScore=18.04）；"Elderly Woman" CLIP-IQA=**0.634**（最高）。
- **推理效率**（Table 5）：TSBM 单样本推理时间 1.460s（相对 1.00×），DPS γ=1 为 7.116s（4.88×），CondSMC N=16 为 14.46s（9.91×）；TSBM working memory 仅 0.180 GiB，远低于 DPS 的 2.229 GiB。
- **消融（App. G.3–G.4）**：去掉 Corrector（静态 $h=h_{\text{old}}$）时 reward 提升明显变慢，验证 corrector 必要性；多数 reward 增益在第一阶段后趋于稳定。

## 相关工作脉络
1. **Schrödinger Bridge 求解器**：DSBM（Shi et al., 2023）、IMF、Adjoint Matching（Liu et al., 2025a）等迭代/匹配方法；本文首次在这些框架上实现 **exact marginal reward tilting**，区别于既往 Boltzmann sampler 等仅支持可评估能量函数的工作。
2. **Diffusion reward finetuning**：Domingo-Enrich et al. (2025) Adjoint Matching、Black et al. (2024) RLFT；本文将其推广至 **bridge（非 memoryless）场景**，解决 $X_0$–$X_1$ 依赖带来的终端偏差问题。
3. **Reward steering via guidance**：DPS（Chung et al., 2023a）、CondSMC（Del Moral et al., 2006）；本文指出这些方法的 **conditional tilt** $q_{1|0}\propto p_{1|0}^P e^{r}$ 无法精确保证 terminal marginal $p_1^r$，TSBM 通过 learning-based 微调绕过此限制。
4. **Diffusion bridge reward alignment**：CDDB（Chung et al., 2023b）、PalSB（Li et al., 2025）；本文强调这些方法不保证**源边际不变**且目标为精确 exponential tilt。
5. **Stochastic optimal control for generative models**：SOC-Matching 框架（Domingo-Enrich et al., 2024）；本文将其结合 SB 的双边际约束，构造出 **relative SOC** 形式。

## 局限性与未来方向
- **理论假设在实践中可能不成立**：要求 reward 可微、endpoint transition score 可访问、Schrödinger potential 的光滑性；对不可微 reward 无法直接适用。
- **计算成本较高**：每阶段需多次 SDE 模拟与 adjoint ODE 反传，fine-tuning 比 inference-time 方法更昂贵（但训练成本低）。
- **未覆盖离散状态空间**：当前方法在连续空间推导；扩展到 discrete bridge（如蛋白设计）是潜在方向。
- **未探索其他训练目标**：如 log-variance loss、其他 adjoint 算法等，可作为后续工作。

## 研究启发与可借鉴点
1. **"相对 SOC"思路可迁移**：将参考过程从基础 $R$ 切换至预训练桥 $P$，利用 $\widehat\varphi_1^{\text{old}}$ 消去对 $p_1$ 的依赖，是一种通用的"post-hoc adaptation without retraining"范式，可推广至 Flow Matching bridge 或其他 bridge 模型。
2. **交替 Controller-Corrector 优化框架**：解决 interdependent 学习的固定点迭代策略，可复用于其他需要同时优化 drift 与 potential 的桥模型任务。
3. **推理零开销的 reward steering**：TSBM 微调后 inference 时间与预训练桥相同，相比 inference-time 方法（DPS/SMC）避免了额外计算，对部署友好。
4. **实验设计值得借鉴**：同时报告 train reward（ImageReward）、eval reward（PickScore）和 fidelity（CLIP-IQA、MSE/LPIPS）三角指标，有效区分 genuine alignment 与 reward hacking。
5. **与团队方向结合机会**：若团队涉及无配对科学翻译（如分子构型、轨迹推断）或物理约束生成，TSBM 的 reward-tilting 框架可直接迁移，用物理/能量 reward 微调预训练 SB。

## 关键术语表
**Schrödinger Bridge（SB）**：在给定参考扩散过程下，以最小 KL 散度匹配两个边缘分布 $p_0$ 和 $p_1$ 的路径概率测度，与 entropic optimal transport 等价。
**Stochastic Optimal Control（SOC）**：在控制-affine SDE 下最小化控制能量（或含终端代价）的最优控制问题，其解与 HJB/adjoint 方程相关。
**Adjoint Matching**：基于 lean adjoint 状态 $\tilde a_t$ 的反传 ODE 来求解 SOC 的最优控制器，避免直接求值 terminal cost，可扩展性好。
**Controller Matching / Corrector Matching**：TSBM 的两个交替更新步骤；前者学习增量漂移 $u_t$，后者学习 terminal corrector $h=\nabla\log\widehat\varphi_1$。
**Reward Tilting**：将目标分布调整为 $p_1^r(x)\propto p_1(x)e^{r(x)}$，实现对 pretrained generative model 的 post-hoc 偏好对齐。
**Diffusion Bridge**：条件连接 $X_0$ 与 $X_1$ 的扩散过程（如 I2SB），与无记忆 diffusion 不同，其 $X_0$ 与 $X_1$ 存在依赖关系。
**Lean Adjoint**：不含控制项依赖的 adjoint 状态 $\tilde a_t$，由 $\tilde a_1=-\nabla r$ 反传得到，用于构建 scalable controller matching loss。
**Schrödinger Potential（$\widehat\varphi_1$）**：满足 $p_1=\varphi_1\widehat\varphi_1$ 的边缘因子，表征 pre-trained bridge 相对于 Wiener 参考的终端修正。

## 可复现要素
- **数据集**：Colored MNIST（基于标准 MNIST 构建，论文引用 Gushchin et al. 2024b 的设置）；CelebA 128×128（公开数据集）。
- **代码**：论文声明"official DSBM github repository"用于预训练；TSBM 代码未明确声明开源，需查阅附录及作者信息获取。
- **权重**：预训练 DSBM 权重未提供下载链接；TSBM 各 stage 检查点未公开。
- **关键超参**：$\lambda=1000$（Colored MNIST）、$\lambda=30$（CelebA）；$\sigma_t=1$；TSBM 训练 5 个 stage；NFE=30（MNIST）/100（CelebA）；Adam lr=$10^{-5}$；EMA rate=0.99；batch size=32。
