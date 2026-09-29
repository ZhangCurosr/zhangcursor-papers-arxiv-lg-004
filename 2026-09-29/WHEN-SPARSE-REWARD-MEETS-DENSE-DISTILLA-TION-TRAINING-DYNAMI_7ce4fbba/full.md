# WHEN SPARSE REWARD MEETS DENSE DISTILLA-TION: TRAINING DYNAMICS OF ON-POLICY DISTIL-LATION

Xinke Jiang<sup>1,2,3∗</sup>, Tao Feng<sup>∗</sup>, Zhibang Yang<sup>1,2,3∗</sup>, Zhixin Zhang<sup>1,2,3</sup>, Weixuan Xu<sup>1</sup>, Haoyu Zhang, Xu Chu<sup>2,3,4†</sup>

<sup>1</sup>National Engineering Research Center of Software Engineering, Peking University, Beijing, China <sup>2</sup>School of Computer Science, Peking University, Beijing, China

<sup>3</sup>Key Laboratory of High Confidence Software Technologies, Ministry of Education, Beijing, China <sup>4</sup>Center on Frontiers of Computing Studies, Peking University, Beijing, China

{xinkejiang, yangzb}@stu.pku.edu.cn

## ABSTRACT

Reinforcement learning with verifiable rewards provides a sparse post-training signal: a single binary outcome evaluates the entire rollout, and every token receives the same sequence-level advantage regardless of its individual contribution. To complement this sparse supervision, a growing family of methods adds a scalar-weighted teacher KL term to the policy-gradient objective, providing dense token-level guidance that may be unreliable at some positions. Despite the benefits of combining these signals, their interaction during optimization can destabilize joint training. To understand how this instability develops, we study the learning dynamics of hybrid reward–distillation training through a neural tangent kernel (NTK) analysis. We introduce the cross-signal NTK $K _ { D R } ( n )$ , a token-level statistic that measures the alignment between reward and distillation gradients at position n. Through this analysis, we identify two failure modes: ❶ Magnitude drowning, where the reward gradient exceeds the distillation gradient by orders of magnitude, so that even weak directional conflict can cause the distillation loss to rise despite its explicit inclusion in the training objective; and ❷ Localized directional conflict, where the sequence-level advantage and the teacher’s positionspecific distribution induce opposing updates at the same token $( K _ { D R } ( n ) < 0 )$ The severity of these effects depends on the optimization regime: the gradientnorm ratio $\dot { \kappa } = { \| \nabla \mathcal { L } _ { R } \| } / { \| \nabla \mathcal { L } _ { D } \| }$ varies by roughly an order of magnitude across tasks, and our experiments reveal an empirical threshold beyond which naive mixing can lead to persistent training collapse. Motivated by these findings, we introduce the M3 family, which combines magnitude normalization with three strategies for coordinating dense teacher supervision and sparse reward updates: a hard NTK-based mask that retains compatible teacher signals (M3-Select), a continuous relaxation of this mask (M3-Soft), and a fast–slow extragradient step that temporally separates teacher shaping from reward correction (M3-EG). Experiments across four model backbones and four benchmarks show that M3 maintains stable training dynamics and achieves superior performance in high-κ regimes where scalar-mixing baselines collapse.

## 1 INTRODUCTION

Reinforcement Learning with Verifiable Rewards (RLVR) has become a central approach to posttraining reasoning-capable large language models (DeepSeek-AI, 2025; OpenAI, 2024). However, its supervision is sparse: a single outcome reward evaluates the entire sequence, and every token receives the same sequence-level advantage regardless of its individual contribution. Process-reward methods partially address this limitation by evaluating intermediate reasoning steps, but step-level supervision does not directly distinguish the contributions of individual tokens. As the complementary, teacher distillation provides dense supervision with target distribution at each token position, although the teacher’s guidance may be unreliable at some positions. Therefore, a growing family of hybrid methods therefore augments the policy-gradient objective with a scalar-weighted teacher KL term (Zhao et al., 2026; Agarwal et al., 2024), aiming to provide token-level guidance.

Despite their empirical success (Zhao et al., 2026; Agarwal et al., 2024), these hybrid methods can exhibit unstable training dynamics and, in some cases, catastrophic collapse (Figure 1c). The underlying difficulty is that a reliable but sparse outcome signal and a dense but imperfect proxy signal do not necessarily complement each other. Under naive scalar mixing, they may instead compete: ❶ one signal can overwhelm the other, ❷ or their opposing updates can cancel, progressively homogenizing the policy, eliminating reward diversity, and ultimately inducing entropy collapse.

To understand how this instability develops, we study the learning dynamics of hybrid reward– distillation training through an NTK analysis. Building on the neural tangent kernel (NTK) framework (Ren et al., 2025), we examine how reward and distillation updates interact through the model’s shared parameters. We introduce the cross-signal NTK $K _ { D R } ( n )$ , the inner product between the parameter gradients contributed by the two objectives at token position $n ,$ which measures their local alignment while accounting for the mapping from token-level residuals to parameter updates through the model’s Jacobian. Our analysis identifies two failure modes: ❶ Magnitude drowning. The reward gradient can exceed the distillation gradient by orders of magnitude, as measured by the norm ratio $\kappa = \| g _ { R } \| / \| g _ { D } \|$ . In this regime, even weak negative alignment can make the increase in distillation loss caused by the reward update exceed the decrease produced by the distillation update itself. Consequently, the reward or distillation loss can rise despite its explicit inclusion in the training objective (Figure 1a). ❷ Localized directional conflict. The two objectives assign updates using different information: RL broadcasts a sequence-level advantage to every token, whereas distillation uses a position-specific teacher distribution. At positions where $K _ { D R } ( n ) < 0$ (Figure 1b), the resulting gradient contributions oppose each other. For a positive-advantage rollout, the teacher up date can decrease the probability of a sampled token that the reward update seeks to reinforce. For a negative-advantage rollout, it can instead reinforce a token that the reward update seeks to suppress. Their contributions can therefore cancel in aggregate diagnostics, obscuring local conflicts. Over longer training horizons, repeated conflicting updates can reduce diversity among rollouts and diminish within-group reward variation. When all rollouts in a group receive the conflict reward, their relative advantages and policy gradient vanish. Our experiments exhibit a corresponding progression from initial reward improvement to reduced reward diversity and abrupt collapse (Figure 1c).

The severity of these failure modes depends on the optimization regime. A hybrid run may ini tially appear stable, with reward increasing even as the distillation loss rises. Across architectures and tasks, κ varies by roughly an order of magnitude, and our experiments reveal an empirical threshold beyond which naive mixing becomes prone to persistent collapse. These findings motivate examining teacher supervision at two levels. At the run level, κ indicates the degree of magnitude imbalance and helps distinguish settings where standard scalar mixing remains effective from those where it collapses. At the token level, the sign of $K _ { D R } ( n )$ distinguishes locally compatible teacher updates from conflicting ones. Guided by this diagnosis, we introduce the M3 family, which combines magnitude normalization with three strategies for coordinating teacher supervision and reward updates: M3-Select applies a hard NTK-based mask, retaining teacher supervision only at positions where $K _ { D R } ( n ) \geq 0 ;$ ; M3-Soft replaces this mask with a continuous, temperature-controlled gate; and M3-EG temporally separates teacher shaping from reward correction through a fast–slow extragradient step. We make the following contributions:

• We develop an NTK-based framework that characterizes the interaction between RL and distillation through a single per-token statistic, the cross-signal NTK $K _ { D R } ( n )$ . This framework identifies two failure modes of linear mixing and explains why weight-space rebalancing methods.

• We show that κ remains stable throughout a run but varies by roughly an order of magnitude across tasks. A critical threshold separates a stable regime, in which naive mixing is effective, from a catastrophic regime where it collapses. This yields a simple probe-batch rule for selecting the mixing strategy before full training.

• In high-κ regimes, the M3 family remains stable over horizons at which scalar-mixing baselines collapse. M3-Select is the most robust variant in the highest-κ settings, while M3-Soft recovers catastrophic cases with order-of-magnitude gains. Combined with weight averaging, which removes the gate-variance pathology identified in our analysis, M3-Soft matches or exceeds the strongest baseline across all tested architecture–dataset pairs.

![](images/9bd9f91556688898101bc802186ad5d99dbfa089ef6b66993057e56c3b63f0ef.jpg)

![](images/78e9cf3ff34a4c3188abb1d1907f101e8eef069a78bc96fc706bb25901c80edb.jpg)

![](images/ac46b1b6ed58d458645190bc56b38c0a7081143508b959e872d86b1ce37116e0.jpg)  
Figure 1: Two failures of linear reward–distillation mixing (Qwen3-1.7B, GSM8K). (a) Distillation loss rises under OPSD, indicating magnitude drowning. (b) Positive and negative $K _ { D R } ( n )$ interleave in a correct rollout, revealing token-local conflict hidden by aggregation. (c) Baselines collapse by step 500, while M3-Select and M3-Soft remain stable.

## 2 RELATED WORK

❶ LLM Reasoning via RL and Distillation. DeepSeek-R1 (DeepSeek-AI, 2025) spurred RL reasoning (GRPO (Shao et al., 2024), DAPO (Yu et al., 2025), Dr. GRPO (Liu et al., 2025)); OPSD (Zhao et al., 2026) and MiniLLM (Gu et al., 2024) make distillation on-policy. Hybrids differ in coupling: SDPO (Hubotter et al., 2026) self-distills from a reprompt, RLSD (Yang et al., 2026) keeps¨ the teacher as magnitude-only reweighting, and HDPO (Ding, 2026) and DPKD (Li et al., 2024) interpolate losses. Unlike these methods, we study the dynamics of the coupling, identifying when dense teacher supervision conflicts with or is drowned by sparse reward updates.

❷ Multi-Objective Gradients and NTK. PCGrad (Yu et al., 2020), CAGrad (Liu et al., 2021), MGDA (Sener & Koltun, 2018), and NashMTL (Navon et al., 2022) operate on aggregate task gradients; GradNorm (Chen et al., 2018), UW (Kendall et al., 2018), and DWA (Liu et al., 2019) balance objectives in loss-weight space. Such global operations do not directly resolve conflicts that alternate across tokens, and we show that GradNorm becomes ineffective at $\kappa \gg 1$ (Proposition 5). Prior work applies NTK to gradient conflict and imbalance (Ren et al., 2025; Qin et al., 2025); our cross-signal NTK specializes this perspective to hybrid RL–distillation and links tokenlevel interaction to drowning-induced collapse, complementing known RL failures such as reward hacking (Skalse et al., 2022), entropy collapse (Yu et al., 2025), and length bias (Liu et al., 2025).

## 3 PRELIMINARIES AND THEORETICAL ANALYSIS

Notation. In this paper, scalars use italic symbols, vectors bold lowercase or Greek symbols $( \mathrm { e . g . }$ $g , \delta , \theta )$ , and matrices bold uppercase symbols $( \mathbf { e . g . , J , K } )$ . Calligraphic symbols denote sets and losses; R denotes real numbers. The model has $d _ { \mathrm { p a r } }$ parameters, collected in $\pmb { \theta } \in \mathbb { R } ^ { d _ { \mathrm { p a r } } }$ . At position $n , p _ { S } ^ { n }$ is the student’s token distribution and $p _ { S } ^ { n } ( v ) = [ p _ { S } ^ { n } ] _ { v }$ the probability of token v. We distinguish kernel matrix ${ \bf K } ( n , m )$ from the scalar interaction score $\dot { K } _ { D R } ( n )$ and use $\varphi _ { n }$ for alignment angles.

## 3.1 PROBLEM SETUP AND NTK PRELIMINARIES

We train student model $\pi _ { \theta }$ using teacher predicts and response-level rewards. For a prompt–answer pair $( x , y ^ { * } ) \sim \mathcal { D }$ , the student generates a response $\hat { y }$ of length $N ; { \hat { y } } _ { < n }$ is its prefix before position n. ❶ On-Policy Self-Distillation (OPSD). OPSD (Zhao et al., 2026) learns from responses sampled by the student itself. A frozen teacher receives the ground-truth answer as additional context, giving ${ \pmb p } _ { T } ^ { n } = \pi { \pmb \theta } _ { 0 } ( { \cdot } \ \vert \ x , y ^ { * } , { \hat { y } } _ { < n } )$ , while $\pmb { p } _ { S } ^ { n } = \pi \pmb { \theta } ( \cdot \ | \ x , \hat { y } _ { < n } )$ . Their distributions are compared at each position, with each contribution capped at τ:

$$
\mathcal { L } _ { D } ( \pmb { \theta } ) = \mathbb { E } _ { ( x , y ^ { * } ) \sim \mathcal { D } } \mathbb { E } _ { \hat { y } \sim \pi _ { \theta } ( \cdot \vert x ) } \left[ \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \operatorname* { m i n } \Bigl ( D _ { \lambda } \bigl ( p _ { T } ^ { n } \bigr \| p _ { S } ^ { n } \bigr ) , \tau \Bigr ) \right] ,\tag{1}
$$

Here, $D _ { \lambda } ( \pmb { p } \| \pmb { q } ) = \lambda \mathrm { K L } ( \pmb { p } \| \pmb { m } ) + ( 1 - \lambda ) \mathrm { K L } ( \pmb { q } \| \pmb { m } )$ is the generalized Jensen–Shannon divergence, with $m = \lambda p + ( 1 - \lambda ) q$ . We use $\lambda = 1 / 2 ,$ , giving both distributions equal weight. The teacher thus provides dense and position-specific supervision; KL limits are given in Appendix AG.1.

❷ Group Relative Policy Optimization (GRPO). GRPO (Shao et al., 2024), a critic-free variant of PPO (Schulman et al., 2017), samples G responses per prompt and assigns each a verifiable reward $r ^ { ( i ) } = r ( \hat { y } ^ { ( i ) } , y ^ { * } ) \in \{ 0 , 1 \}$ . Its advantage $A _ { i } = ( r ^ { ( i ) } - \bar { r } ) / ( \sigma _ { r } + \epsilon )$ compares that reward with the group mean r¯ and standard deviation $\sigma _ { r } .$ , where $\epsilon > 0$ prevents division by zero:

$$
\mathcal { L } _ { R } ( \pmb { \theta } ) = - \mathbb { E } _ { ( \pmb { x } , \pmb { y } ^ { * } ) \sim \mathcal { D } } \mathbb { E } _ { \{ \hat { \pmb { y } } ^ { ( i ) } \} _ { i = 1 } ^ { G } \sim \pi _ { \pmb { \theta } } ( \cdot \vert \pmb { x } ) } \left[ \frac { 1 } { G } \sum _ { i = 1 } ^ { G } A _ { i } \frac { 1 } { N _ { i } } \sum _ { n = 1 } ^ { N _ { i } } \log \pi _ { \pmb { \theta } } \left( \hat { y } _ { n } ^ { ( i ) } \left. x , \hat { y } _ { < n } ^ { ( i ) } \right. \right) \right] ,\tag{2}
$$

where $N _ { i } ~ = ~ | \hat { y } ^ { ( i ) } |$ Positive advantages encourage sampled responses and negative advantages discourage them. The same advantage weights every token, without identifying which tokens caused success or failure. Gradients hold the sampled responses and advantages fixed.

❸ NTK in Learning Dynamics. Following (Ren et al., 2025), let $z ^ { n } \in \mathbb { R } ^ { | \nu | }$ be the policy’s logits, the scores converted by softmax into probabilities over vocabulary V. The Jacobian $\mathbf { J } ^ { n } = \nabla _ { \pmb { \theta } } z ^ { n } \in$ $\mathbb { R } ^ { d _ { \mathrm { p a r } } \times | \nu | }$ describes their dependence on the parameters. For $\begin{array} { r } { \mathcal { L } = N ^ { - 1 } \sum _ { n } \ell ^ { n } } \end{array}$ , the chain rule gives

$$
\nabla _ { \theta } \mathcal { L } = \frac { 1 } { N } \sum _ { n } \mathbf { J } ^ { n } \delta ^ { n } , \qquad \delta ^ { n } = \nabla _ { z ^ { n } } \ell ^ { n } , \qquad \mathbf { K } ( n , m ) = ( \mathbf { J } ^ { n } ) ^ { \top } \mathbf { J } ^ { m } .\tag{3}
$$

The residual $\delta ^ { n }$ describes how the loss changes with each logit. The empirical NTK ${ \bf K } ( n , m )$ (Jacot et al., 2018) couples positions through their shared parameters: an update driven by position m can change predictions at n. Following Ren et al. (2025), we use the Action–Kernel–Gradient (AKG) decomposition to analyze these changes.

## 3.2 HYBRID UPDATE DYNAMICS AND LOSS INTERACTIONS

In OPSD, we assume the hybrid loss between policy update and distillation loss is ${ \mathcal { L } } _ { H } = ( 1 -$ $\alpha ) \mathcal { L } _ { R } + \alpha \mathcal { L } _ { D }$ , where $\alpha \in [ 0 , 1 ]$ weights distillation. A step of size η gives $\pmb { \theta } _ { t + 1 } = \pmb { \theta } _ { t } - \eta \nabla _ { \pmb { \theta } } \mathcal { L } _ { H }$ Expanding the logits to first order and summing over $T$ steps yields:

$$
z _ { T } ^ { n } = z _ { 0 } ^ { n } + \sum _ { t = 0 } ^ { T - 1 } \Delta z _ { t } ^ { n } , \qquad \Delta z _ { t } ^ { n } = - \frac { \eta } { N } \sum _ { m = 1 } ^ { N } \mathbf { K } _ { t } ( n , m ) \Big [ ( 1 - \alpha ) \delta _ { R } ^ { m } + \alpha \delta _ { D } ^ { m } \Big ] + O ( \eta ^ { 2 } ) ,\tag{4}
$$

where $z _ { t } ^ { n } = z ^ { n } ( \theta _ { t } )$ and $\Delta z _ { t } ^ { n } = z _ { t + 1 } ^ { n } - z _ { t } ^ { n }$ . At each step, the kernel maps the combined residual at every position m to a change in the logits at n. The residuals are evaluated at step t:

$$
\delta _ { R } ^ { m } = - A ( e _ { \hat { y } _ { m } } - { p } _ { S } ^ { m } ) , \qquad \delta _ { D } ^ { m } = { p } _ { S } ^ { m } - { p } _ { T } ^ { m } .\tag{5}
$$

The one-hot vector $e _ { \hat { y } _ { m } }$ selects the sampled token. The reward residual encourages or discourages this token according to $A ;$ the teacher residual compares the full distributions. We use a local forward-KL model of distillation: near agreement, the unclipped JSD gradient is $\lambda ( 1 - \lambda ) \delta _ { D } ^ { m } + O ( \| p _ { S } ^ { m } - { \pmb { p } } _ { T } ^ { m } \| ^ { 2 } )$ , with the leading constant absorbed into the teacher scale. Clipped tokens contribute zero gradient (Appendix AG.1).

First-Order Loss Changes. With $\begin{array} { r } { { \bf g } _ { R } = \nabla _ { \theta } \mathcal { L } _ { R } } \end{array}$ and $\mathbf { \sigma } _ { g D } = \nabla _ { \theta } \mathcal { L } _ { D }$ , the same update gives

$$
\Delta \mathcal { L } _ { R } \approx - \eta \Big [ ( 1 - \alpha ) \| g _ { R } \| ^ { 2 } + \alpha \langle g _ { R } , g _ { D } \rangle \Big ] , \qquad \Delta \mathcal { L } _ { D } \approx - \eta \Big [ \alpha \| g _ { D } \| ^ { 2 } + ( 1 - \alpha ) \langle g _ { D } , g _ { R } \rangle \Big ] ,\tag{6}
$$

Each squared-gradient term describes an objective’s own decrease; the inner product describes the other update’s effect. Positive alignment helps both objectives, while negative alignment opposes their progress. A loss increases only when this opposing contribution exceeds its own decrease within the first-order approximation.

## 3.3 TOKEN-LEVEL DECOMPOSITION OF GRADIENT INTERACTIONS

Cross-Position Interactions. The overall inner product can hide local conflicts. Each term pairs the teacher signal at n with the reward signal at m through the shared kernel. The sum includes same-position and cross-position interactions, whose positive and negative contributions can cancel.

Expanding both gradients gives that:

$$
\langle g _ { D } , g _ { R } \rangle = \frac { 1 } { N ^ { 2 } } \sum _ { n , m } \underbrace { ( \delta _ { D } ^ { n } ) ^ { \top } } _ { \mathrm { t e a c h e r r e s i d u a l } } \underbrace { \mathbf { K } ( n , m ) } _ { \mathrm { k e r n e l ~ c o u p l i n g } } \underbrace { \delta _ { R } ^ { m } } _ { \mathrm { r e w a r d ~ r e s i d u a l } } .\tag{7}
$$

Gradient Magnitude and Alignment. Expanding the squared norms in the same way gives the magnitude ratio when $\| \pmb { g } _ { D } \| > 0 \colon$

$$
\kappa ^ { 2 } = \frac { \| \pmb { g } _ { R } \| ^ { 2 } } { \| \pmb { g } _ { D } \| ^ { 2 } } = \frac { \sum _ { n , m } ( \pmb { \delta } _ { R } ^ { n } ) ^ { \top } \mathbf { K } ( n , m ) \pmb { \delta } _ { R } ^ { m } } { \sum _ { n , m } ( \pmb { \delta } _ { D } ^ { n } ) ^ { \top } \mathbf { K } ( n , m ) \pmb { \delta } _ { D } ^ { m } } ,\tag{8}
$$

the numerator measures reward-gradient strength and the denominator teacher-gradient strength; $\kappa \gg 1$ indicates strong imbalance. For nonzero gradients, the normalized inner product $\Phi _ { D R } =$ $\langle \pmb { g } _ { D } , \pmb { g } _ { R } \rangle / ( \lVert \pmb { g } _ { D } \rVert \lVert \pmb { g } _ { R } \rVert ) ^ { - } \in [ - 1 , 1 ]$ measures direction independently of magnitude. Positive values indicate alignment and negative values interference. These quantities arise from one gradient Gram matrix: its diagonal contains squared norms and its off-diagonal contains the cross inner product (Appendix AG.2).

## 3.4 TOKEN-LEVEL CONFLICT AND MAGNITUDE DROWNING

We first locate conflict at individual positions, then examine how gradient imbalance amplifies its effect on the teacher loss.

Definition 1 (Cross-Signal Token-Level NTK). Using the local residual model of Section 3.2, define

$$
K _ { D R } ( n ) = \left. { \bf J } ^ { n } \delta _ { D } ^ { n } , { \bf J } ^ { n } \delta _ { R } ^ { n } \right. , \qquad \delta _ { D } ^ { n } = p _ { S } ^ { n } - p _ { T } ^ { n } , \qquad \delta _ { R } ^ { n } = - A \left( e _ { \hat { y } _ { n } } - p _ { S } ^ { n } \right) .\tag{9}
$$

where $A = ( r - \bar { r } ) / ( \sigma _ { r } + \epsilon )$ . This scalar compares the teacher and reward parameter gradients contributed by the same position.

Positive, negative, and zero scores define the sets $\Omega _ { + } , \Omega _ { - } ,$ and $\Omega _ { 0 }$ . A zero score also includes vanishing gradients. $K _ { D R } ( n )$ is the $m = n$ summand in Eq. 7, before the common factor $N ^ { - 2 }$ ; it does not include cross-position interactions.

❶ Token-Level Conflict. On a positive-advantage response, a negative $K _ { D R } ( n )$ means that the teacher’s same-position contribution lowers the sampled token’s log-probability. The full update need not do so, since reward and other-position contributions also matter. Corollary 1 derives this local effect. The conflict rate $C _ { \mathrm { N T K } } ~ \overset { \circ } { = } ~ | \{ n : K _ { D R } ( n ) < 0 \} | / N$ is approximately 40% in our measurements, an empirical observation rather than a consequence of the definition.

❷ Magnitude Drowning. Write cos $\varphi = \Phi _ { D R }$ . Equation 6 gives $\Delta \mathcal { L } _ { D } \approx - \eta \| g _ { D } \| ^ { 2 } [ \alpha + ( 1 -$ α)κ cos φ]. For nonzero gradients and $0 < \alpha < 1$ , its first-order sign condition is $\Delta \mathcal { L } _ { D } > 0 \iff$ cos φ $\begin{array} { r } { < \dot { - } \frac { \alpha } { ( 1 - \alpha ) \kappa } } \end{array}$ . For fixed $\alpha ,$ , the threshold approaches zero from below as ${ \cal O } ( \kappa ^ { - 1 } )$ : when the reward gradient is large, even weak negative alignment can outweigh the teacher’s own descent. Imbalance alone is insufficient; negative alignment must also satisfy the threshold.

As the student approaches the teacher distribution, $\delta _ { D } ^ { n }$ shrinks, while the reward residual need not shrink at the same rate. The kernel maps both into parameter space, so their magnitudes and directions jointly determine κ. Our experiments identify magnitude imbalance as the dominant failure mode, with ratios reaching $1 . 3 \times 1 0 ^ { 4 }$ . Ratios are approximately stable across the tested LoRA ranks 8–64 but vary substantially across tasks (Proposition 7). Extended scaling analysis and projectionbaseline limitations appear in Appendices AG.2 and M.

## 3.5 REWARD-SIGNAL DEGENERATION AND TRAINING COLLAPSE

Repeated teacher contributions at conflicting positions may suppress successful response patterns. If the resulting responses become less diverse in reward, GRPO receives less information for distinguishing them. This is a possible training mechanism, not a direct consequence of the local gradient identity. The final step is exact: if all G sampled responses receive the same reward, then $r ^ { ( i ) } = { \bar { r } } .$ $\sigma _ { r } = 0$ , and every advantage $A _ { i } = 0$ . All reward residuals vanish, so that group contributes no reward gradient, $g _ { R } ^ { ( x ) } = 0$ . Only the teacher term can contribute to its hybrid update. However, one such group does not establish collapse: it may contain all successes or all failures, and later sampling or updates from other prompts may restore reward variation. Persistent failure requires poor responses to remain dominant without recovery of a useful reward signal. The abrupt drops in Figure 1c are consistent with this mechanism; Section 5.2 provides empirical tests.

![](images/9b386d9bb4d8f4b0ba8ed18cf365779f0b126c308d4871faf4ee33edea9383fa.jpg)  
Figure 2: From interaction analysis to hybrid updates. Shared parameters couple dense teacher and sparse reward supervision. M3 controls residual scale and allocates teacher influence by tokenlevel compatibility; its extragradient extension separates teacher shaping from reward correction.

Remark 1 (Empirical Gate-Selection Threshold). Our configurations show a transition near $\kappa ^ { * }$ ≈ $5 \times 1 0 ^ { 3 }$ : hard masking is most useful at high imbalance, while soft gating often retains more useful supervision at lower imbalance. This empirical guideline is not a universal collapse threshold; it depends on the compatible-token fraction, conflict strength, and gate-estimation error. Section 4 introduces the gating methods.

## 4 METHODOLOGY

M3 turns the preceding analysis into two coupled decisions: how strongly each signal enters the update, and where teacher guidance is admitted. We first normalize token residuals, then allocate teacher weight using local compatibility (Figure 2). M3-Select and M3-Soft implement this allocation with hard and continuous gates. M3-EG extends the same principle to update timing, letting the teacher shape where the reward direction is evaluated.

## 4.1 CONTROLLING SIGNAL SCALE

In a global mixture, comparable weighted gradient norms require $\alpha \approx \kappa / ( 1 { + } \kappa )$ ; the coefficient must compensate for scale before it can express a preference between the signals. M3 instead controls scale at the residual level, before conversion into a parameter update. Using the residual convention of Section 3.2, define

$$
\widehat { \pmb { \delta } } _ { q } ^ { n } = \frac { \pmb { \delta } _ { q } ^ { n } } { \lVert \pmb { \delta } _ { q } ^ { n } \rVert + \epsilon } , \qquad \pmb { h } _ { q } ^ { n } = \mathbf { J } ^ { n } \widehat { \pmb { \delta } } _ { q } ^ { n } , \qquad q \in \{ D , R \} .\tag{10}
$$

Here $\epsilon > 0$ stabilizes division and keeps zero residuals zero. Normalization reduces the dependence of mixing on raw residual magnitudes. Its positive scaling preserves the sign of $K _ { D R } ( n )$ , but does not equalize parameter-gradient norms after multiplication by J<sup>n</sup>. It therefore controls residual scale while leaving the directional coordination problem to the gate. The teacher budget acts on these rescaled contributions, while a common update scale controls their overall size. This separates the intended allocation of supervision from raw magnitude disparity that constrains naive mixing.

## 4.2 ALLOCATING TEACHER INFLUENCE ACROSS POSITIONS

With a token-dependent teacher weight $\alpha _ { n } \in [ 0 , \alpha _ { \mathrm { m a x } } ]$ and $\alpha _ { \mathrm { m a x } } < 1$ , the common update:

$$
{ \pmb h } _ { H } ^ { n } = \underbrace { \alpha _ { n } { \pmb h } _ { D } ^ { n } } _ { \mathrm { t e a c h e r c o n t r i b u t i o n } } + \underbrace { ( 1 - \alpha _ { n } ) { \pmb h } _ { R } ^ { n } } _ { \mathrm { r e w a r d c o n t r i b u t i o n } } ,
$$

$$
g _ { H } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } h _ { H } ^ { n } , \qquad \pmb { \theta } _ { t + 1 } = \pmb { \theta } _ { t } - \eta \bar { \pmb { s } } \pmb { g } _ { H } .\tag{11}
$$

The shared exponential-moving-average scale s¯ sets the update magnitude; $\alpha _ { n }$ controls its local composition. Teacher and reward weights vary together: rejecting the teacher restores reward weight to one. The remaining choice is how compatibility determines $\alpha _ { n }$

M3-Select: retain compatible teacher contributions. The hard gate directly implements the sign criterion:

$$
\alpha _ { n } ^ { * } = \alpha _ { \operatorname* { m a x } } \mathbb { I } \{ K _ { D R } ( n ) \geq 0 \} .\tag{12}
$$

This removes negative teacher projections onto the same-position reward direction. Since normalization preserves signs, the resulting contribution satisfies

$$
\langle h _ { H } ^ { n } , h _ { R } ^ { n } \rangle = ( 1 - \alpha _ { n } ^ { * } ) \| h _ { R } ^ { n } \| ^ { 2 } + \alpha _ { n } ^ { * } \langle h _ { D } ^ { n } , h _ { R } ^ { n } \rangle \geq ( 1 - \alpha _ { \operatorname* { m a x } } ) \| h _ { R } ^ { n } \| ^ { 2 } .\tag{13}
$$

Rejected positions retain the full local reward contribution. This is a same-position guarantee: crossposition interactions in the aggregate update remain. A zero score is admitted by convention and may simply reflect a vanishing signal. The gate ceiling controls how much teacher influence an admitted position receives; the score’s sign controls whether it receives that influence at all. These two choices need not be tied to one global loss weight.

M3-Soft: vary teacher influence continuously. Hard decisions can change abruptly near zero alignment. M3-Soft smooths this transition, retaining partial supervision when compatibility is weak or uncertain:

$$
\alpha _ { n } = \alpha _ { \operatorname* { m a x } } \sigma ( \beta \cos \varphi _ { n } ) , \quad \cos \varphi _ { n } = \frac { K _ { D R } ( n ) } { \| g _ { D } ^ { n } \| \| g _ { R } ^ { n } \| + \varepsilon } .\tag{14}
$$

Here $\sigma$ is the logistic sigmoid, $\beta \geq 0$ controls sharpness, and $\varepsilon > 0$ stabilizes the score. Larger $\beta$ approaches hard selection away from zero; at zero, the weight remains $\alpha _ { \mathrm { m a x } } / 2$ . Soft gating also retains some negatively aligned teacher contributions, trading strict local exclusion for smoother supervision. It therefore does not inherit Eq. 13 (Appendix AG.4). $\operatorname { A t } \beta = 0$ , all positions receive the same teacher weight $\alpha _ { \mathrm { m a x } } / 2 ;$ increasing sharpness progressively makes allocation depend on compatibility. Both variants thus share the same normalized update, with their distinction confined to the teacher-allocation rule.

## 4.3 COORDINATING THE SIGNALS IN TIME

The synchronous variants combine directions evaluated at the same parameters. M3-EG instead uses the gated teacher field $\begin{array} { r } { { m _ { D } } ( \pmb { \theta } ) = N ^ { - 1 } \sum _ { n } \alpha _ { n } ^ { * } \pmb { h } _ { D } ^ { n } } \end{array}$ to construct a temporary point, then evaluates the normalized reward field $\begin{array} { r } { \begin{array} { r } { h _ { R } ( \pmb { \theta } ) = N ^ { - 1 } \sum _ { n } \pmb { h } _ { R } ^ { n } } \end{array} } \end{array}$ there:

$$
\widetilde { \pmb { \theta } } = \pmb { \theta } - \eta _ { \mathrm { i n } } \pmb { m } _ { D } ( \pmb { \theta } ) , \quad \mathrm { t e a c h e r ~ l o o k - a h e a d } ,
$$

$$
\pmb { \theta } ^ { + } = \pmb { \theta } - \eta _ { \mathrm { o u t } } \pmb { h } _ { R } \widetilde { \pmb { \theta } } ) , \mathrm { r e w a r d c o r r e c t i o n } .\tag{15}
$$

The reward correction is applied from the original parameters, after restoring them; no gradient is propagated through the temporary step. Thus teacher guidance changes the reward evaluation point rather than entering the committed update additively. Appendix E.1 gives the procedure and the conditions on the inner displacement and outer step for local reward descent.

## 4.4 TRAINING RULE AND SCOPE

Each iteration samples student responses, obtains teacher distributions and verifier advantages, and forms the two residual fields. Their compatibility scores determine the gates; normalized, gated contributions then define either the synchronous direction or the extragradient step (Algorithms 2 and 1). Gates and normalization factors specify update coefficients and are held fixed when applying the direction. Clipped teacher terms and zero-advantage reward terms contribute zero; normalization does not recreate missing supervision.

The local projection bound directly supports the hard selection rule. Aggregate stationarity and conditional variance bounds require a more restrictive population model: smooth lower-bounded reward loss, orthogonal position subspaces, matched teacher and mean reward norms, deterministic conditional teacher directions, and gates fixed before fresh reward noise. These conditions are not enforced by residual normalization. Theorem 1 and Proposition 2 state the resulting guarantees; the experiments assess behavior beyond that model.

![](images/1eb1a194fb97611f4fa7385f1fa4d8fb495c001a270578d2c926f8c34febe92c.jpg)

![](images/a39704619cff9aecc0a505b6d7b4570f08ce1c9c98b04cb8b3fcc029fba7f0a3.jpg)

![](images/f801124f8208c3873f77c119e825e5a106b0330265b202099d20c4caefb68147.jpg)  
Figure 3: GradNorm challenge on InternLM2.5-1.8B GSM8K: training reward (left), gradient cosine (center), and learned distillation weight (right).

Table 1: Cell reports accuracy (top) and its percentage-point change from GRPO (bottom; ${ \uparrow \ / \ \downarrow \ } ;$ <sup>†</sup>Hard masking is vacuous for single-token ARC.
<table><tr><td rowspan="2">Method</td><td colspan="3">Qwen3-1.7B</td><td colspan="3">Qwen2.5-1.5B</td><td colspan="3">InternLM2.5-1.8B</td><td colspan="3">Llama-3.2-1B</td></tr><tr><td>GSM8K</td><td>SVAMP</td><td>ARC</td><td>GSM8K SVAMP</td><td></td><td>ARC</td><td>GSM8K</td><td>SVAMP</td><td>ARC</td><td></td><td>GSM8K SVAMP</td><td>ARC</td></tr><tr><td colspan="10">Reference: pure RL</td><td colspan="3"></td></tr><tr><td>GRPO</td><td>0.696</td><td>0.940</td><td>0.732</td><td>0.726</td><td>0.825</td><td>0.689</td><td>0.404</td><td>0.670</td><td>0.599</td><td>0.552</td><td>0.730</td><td>0.533</td></tr><tr><td>Teacher-augmented baselines</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td colspan="10"></td><td colspan="3"></td></tr><tr><td>Hybrid (OPSD)</td><td>0.786</td><td>0.932</td><td>0.796</td><td>0.717</td><td>0.857</td><td>0.693</td><td>0.492</td><td>0.608</td><td>0.604</td><td>0.563</td><td>0.722</td><td>0.544</td></tr><tr><td></td><td>↑+9.0</td><td>↓-0.8</td><td>↑+6.4</td><td>↓-0.9</td><td>↑+3.2</td><td>↑+0.4</td><td>↑+8.8</td><td>↓-6.2</td><td>↑+0.5</td><td>↑+1.1</td><td>↓-0.8</td><td>↑+1.1</td></tr><tr><td>OPSD+GradNorm</td><td>0.765</td><td>0.917</td><td>0.752</td><td>0.735</td><td>0.837</td><td>0.695</td><td>0.459</td><td>0.687</td><td>0.590</td><td>0.547</td><td>0.717 ↓-1.3</td><td>0.370 ↓-16.3</td></tr><tr><td></td><td>↑+6.9 0.8317</td><td>↓-2.3</td><td>↑+2.0</td><td>↑+0.9</td><td>↑+1.2</td><td>↑+0.6</td><td>↑+5.5</td><td>↑+1.7</td><td>↓-0.9</td><td>↓-0.5 0.568</td><td>0.648</td><td>0.545</td></tr><tr><td>RLSD</td><td></td><td>0.923</td><td>0.748</td><td>0.725</td><td>0.860</td><td>0.684</td><td>0.471</td><td>0.737</td><td>0.602</td><td>↑+1.6</td><td>↓-8.2</td><td>↑+1.2</td></tr><tr><td></td><td>↑+13.6 0.587</td><td>↓-1.7</td><td>↑+1.6 0.742</td><td>↓-0.1</td><td>↑+3.5</td><td>↓-0.5</td><td>↑+6.7</td><td>↑+6.7</td><td>↑+0.3 0.586</td><td>0.220</td><td>0.483</td><td>0.432</td></tr><tr><td>SDPO</td><td>↓-10.9</td><td>0.567 ↓-37.3</td><td>↑+1.0</td><td>0.085 ↓-64.1</td><td>0.612 ↓-21.3</td><td>0.424 ↓-26.5</td><td>0.316 ↓-8.8</td><td>0.432 ↓-23.8</td><td></td><td></td><td></td><td>↓-10.1</td></tr><tr><td>Ours: boundary-gated mixing (M3)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>↓-1.3</td><td>↓-33.2</td><td>↓-24.7</td><td></td></tr><tr><td colspan="10"></td><td colspan="3"></td></tr><tr><td>M3-Select†</td><td>0.803</td><td>0.943</td><td>0.275</td><td>0.459</td><td>0.585</td><td>0.571</td><td>0.394</td><td>0.538</td><td>0.599</td><td>0.516</td><td>0.673</td><td>0.459</td></tr><tr><td></td><td>↑+10.7</td><td>↑+0.3</td><td>↓-45.7</td><td>↓-26.7</td><td>↓-24.0</td><td>↓-11.8</td><td>↓-1.0</td><td>↓-13.2</td><td>0.0</td><td>↓-3.6</td><td>↓-5.7</td><td>↓-7.4</td></tr><tr><td>M3-Soft</td><td>0.8324</td><td>0.945</td><td>0.807</td><td>0.752</td><td>0.905</td><td>0.736</td><td>0.501</td><td>0.777</td><td>0.608</td><td>0.568</td><td>0.750</td><td>0.556</td></tr><tr><td></td><td>↑+13.6</td><td>↑+0.5</td><td>↑+7.5</td><td>↑+2.6</td><td>↑+8.0</td><td>↑+4.7</td><td>↑+9.7</td><td>↑+10.7</td><td>↑+0.9</td><td>↑+1.6</td><td>↑+2.0</td><td>↑+2.3</td></tr><tr><td>M3-EG</td><td>0.8446</td><td>0.952</td><td>0.731</td><td>0.749</td><td>0.868</td><td>0.692</td><td>0.440</td><td>0.468</td><td>0.6135</td><td>0.552</td><td>0.758</td><td>0.534</td></tr><tr><td></td><td>↑+14.9</td><td>↑+1.2</td><td>↓-0.1</td><td>↑+2.3</td><td>↑+4.3</td><td>↑+0.3</td><td>↑+3.6</td><td>↓-20.2</td><td>↑+1.5</td><td>0.0</td><td>↑+2.8</td><td>↑+0.1</td></tr></table>

M3-Soft ≥ best non-M3 baseline in 12/12 cells (11 strict wins, 1 exact tie).

## 5 EXPERIMENTS

## 5.1 SETUP

Our main evaluation covers Qwen3-1.7B, Qwen2.5-1.5B, InternLM2.5-1.8B, and Llama-3.2-1B on GSM8K, SVAMP, and ARC-Challenge, with Qwen3-0.6B and MATH added in supplementary longhorizon runs. We define κ¯ as the mean of $\kappa _ { t }$ over RL-active steps of the corresponding naive-Hybrid run. Full experimental details are provided in Appendix A.

## 5.2 DOES MAGNITUDE DROWNING ACTUALLY OCCUR?

Figure 1a shows loss inversion under scalar mixing. With κ¯ ranging from $1 . 4 \times 1 0 ^ { 3 } ~ \mathrm { t o } ~ 1 . 3 \times 1 0 ^ { 4 }$ , the reward update overwhelms distillation; panel c shows the resulting long-horizon collapse. We next test whether global norm balancing fixes it.

A controlled challenge. GradNorm trails both GRPO and naive Hybrid in training reward (Figure 3, left), despite a near-zero aggregate gradient cosine (center). Its distillation weight rapidly approaches $w _ { D } ^ { * } = \kappa / ( 1 + \kappa ) \approx 1$ (right), leaving the effective reward weight at only $1 - w _ { D } ^ { * } \approx 1 / \kappa .$ GradNorm therefore achieves global norm balance only by nearly turning off ${ \mathrm { R L } } ;$ because the same weight is applied to every token, it also cannot separate compatible from conflicting teacher updates. This motivates M3’s position-dependent gate; further diagnostics are reported in Appendix B.4.

## 5.3 CAN TOKEN-LEVEL GATING BEAT GLOBAL MIXING?

Accuracy across architectures. Across four architectures and three datasets (Table 1), M3-Soft matches or exceeds the strongest non-M3 baseline in all 12 cells under the per-cell best-observed protocol (11 strict wins, one tie). The largest gains occur on Qwen2.5-SVAMP (0.905, +4.5pp over RLSD), Qwen2.5-ARC (0.736, +4.1pp over OPSD+GradNorm), and InternLM-SVAMP (0.777, +4.0pp over RLSD). On the two Llama arithmetic cells, M3-Soft ties RLSD on GSM8K (0.568) and improves over GRPO on SVAMP $( 0 . 7 5 0 , + 2 . 0 9 9 )$ . The Qwen3 margins are tighter: 0.8324 on GSM8K (+0.07pp over RLSD) and 0.945 on SVAMP (+0.5pp over GRPO).

Table 2: Long-horizon reward on GSM8K after 500 steps (last-50 mean). ↓ denotes collapse.
<table><tr><td rowspan="2">Architecture</td><td colspan="3">Reference and adaptive baselines</td><td rowspan="2">Ours: boundary-gated mixing M3-Select M3-Soft</td></tr><tr><td>GRPO</td><td>Hybrid</td><td>GradNorm</td></tr><tr><td>Qwen3-1.7B</td><td>↓0.003</td><td>↓0.000</td><td>↓0.000</td><td>0.809</td></tr><tr><td>Qwen3-0.6B</td><td>↓0.003</td><td>↓0.000</td><td>↓0.053</td><td>0.866 0.644 0.673</td></tr><tr><td>Llama-1B</td><td>↓0.004</td><td>↓0.003</td><td>↓0.328</td><td>0.497 0.485</td></tr></table>

![](images/5cf3199ae06fafc8898b0d789e355bb52d8db418724467bf1bdfa33626d098ce.jpg)  
Figure 4: Mean gradient-magnitude ratio κ¯ for 12 architecture–dataset pairs under naive Hybrid; dashed line: empirical threshold $\kappa ^ { * } = 5 , 0 0 \mathrm { 0 }$

![](images/1ca5bf69a83c405defaad5025ad832983d2a778a9c29ab9eb5d16b68f25ddb3a.jpg)  
Figure 5: Runtime magnitude ratio. Hybrid and GradNorm cross $\kappa ^ { * } = \bar { 5 } , 0 0 0$ before collapsing at steps 221 and 358.

Long-horizon stability. Table 2 compares last-50 training reward after 500 steps on GSM8K. Across all three architectures, every reference or globally balanced baseline triggers the collapse criterion, whereas M3-Select and M3-Soft remain non-collapsed. M3-Soft attains the highest final reward on both Qwen3 scales, while M3-Select is marginally higher on Llama-3.2-1B. Detailed phase-wise trajectories and collapse times for Qwen3-1.7B are reported in Table 8 of Appendix B.2.

Regime Dependence of Token-Level Gating. Across the 12 architecture–dataset cells, κ¯ ranges from 1,423 to 12,705 (Figure 4). Figure 5 provides the within-run counterpart: on Qwen3-1.7B GSM8K, Hybrid crosses the empirical threshold $\kappa ^ { \ast } ~ = ~ 5 , 0 0 0$ near step 200 before collapsing at step 221, while GradNorm crosses later and collapses at step 358. In the two high-κ Llama arithmetic cells (7,690/12,705), Hybrid’s reference-rate training reward falls to 0.031/0.000 on GSM8K/SVAMP, whereas M3-Soft reaches 0.647/0.822 (Appendix B.1). At lower κ, hard masking can be overly restrictive: on Qwen3-ARC $( \bar { \kappa } = 2 \mathrm { , 4 6 5 } )$ , M3-Select reaches 0.275 accuracy, compared with 0.807 for M3-Soft (Table 1). Thus, κ¯ indicates the severity of magnitude imbalance and the appropriate gate strength, rather than directly predicting accuracy. Together, these results support κ as a regime indicator for instability and appropriate gate strength, rather than a predictor of absolute accuracy; exact per-cell κ¯ values and reference-rate rewards are in Appendix B.1.

Local Consequence of Token-Level Conflict. We test the sign-specific prediction on 30 correct Qwen3-1.7B GSM8K trajectories comprising 8,492 tokens. Tokens are partitioned by the pre-update cross-signal NTK, after which we apply one OPSD, GRPO, or hybrid update. On compatible positions, the hybrid update increases the sampled-token log-probability by ${ \bf \bar { 2 . 0 \times 1 0 ^ { - 3 } } } .$ ; on conflicting positions, it decreases it by $2 . 4 \times 1 0 ^ { - 4 }$ (Table 3). Over the same conflicting subset, the pure OPSD and GRPO updates yield positive changes. Because the partition is computed before the update, the result tests the local sign prediction rather than defining conflict from the observed logit change.

Table 3: One-step sampled-token log-probability change by pre-update conflict region (30 trajectories, 8,492 tokens). Negative values indicate suppression by the update.
<table><tr><td>Region</td><td>OPSD</td><td>GRPO</td><td>Hybrid</td></tr><tr><td>Compatible  $( K _ { D R } > \epsilon , n { = } 8 5 1 )$ </td><td> $+ 1 . 6 \times 1 0 ^ { - 3 }$ </td><td> $+ 2 . 7 \times 1 0 ^ { - 3 }$ </td><td> $+ 2 . 0 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Neutral  $( | K _ { D R } | \le \epsilon , n { = } 5 1 9 1 )$ </td><td> $\approx 0$ </td><td> $\approx 0$ </td><td> $\approx 0$ </td></tr><tr><td>Conflicting  $( K _ { D R } < - \epsilon , n { = } 2 4 5 0 )$ </td><td> $+ 0 . 3 \times 1 0 ^ { - 4 }$ </td><td> $+ 7 . 3 \times 1 0 ^ { - 4 }$ </td><td> $- 2 . 4 \times 1 0 ^ { - 4 }$ </td></tr></table>

## 6 CONCLUSION AND FUTURE WORK

We study when dense teacher supervision can complement sparse verifiable rewards in reasoningmodel post-training. Our NTK analysis separates their interaction into token-level compatibility, captured by the cross-signal NTK $K _ { D R } ( n )$ , and scale imbalance, captured by κ, exposing localized directional conflict and magnitude drowning. This diagnosis leads to the M3 family, which combines magnitude normalization with hard, soft, or temporally decoupled compatibility gating. Across four model families and three reasoning benchmarks, M3-Soft matches or exceeds the strongest non-M3 baseline, while M3 variants remain stable over 500-step GSM8K training where non-gated baselines collapse. In the future, we plan to scale this application to support larger and more complex agentic scenarios such as coding and deep research.

## AI USE STATEMENT

In this work, large language models (LLMs) were used for language polishing, figure design assistance, coding support, and mathematical proof assistance. Specifically, LLMs were used to improve the clarity, grammar, and readability of the manuscript, refine its stylistic quality, and suggest alternative phrasings to reduce redundancy. They also provided suggestions for figure design, visualization layouts, and graphical presentation; all final figures were created, verified, and curated by the authors using the authors’ experimental data. In addition, LLMs assisted with code development, intermediate mathematical derivations, proof construction, and consistency checking. All assumptions, formal statements, derivations, proofs, code, figures, and other LLM-assisted content were independently reviewed, verified, and revised by the authors before inclusion. The research questions, core methodology, scientific contributions, experimental design, critical analyses, and final decisions were independently determined by the authors.

## ETHICS STATEMENT

All experiments in this work were carried out using publicly available language models and standard reasoning benchmarks, including GSM8K, MATH, SVAMP, and ARC-Challenge, in accordance with their respective licenses and terms of use. The study does not involve human or animal subjects, and we did not collect, use, or disclose any personally identifiable information or private user data.

## REPRODUCIBILITY STATEMENT

All experiments use publicly available base models (Qwen3-0.6B, Qwen3-1.7B, Qwen2.5-1.5B, InternLM2.5-1.8B, and Llama-3.2-1B) and standard benchmarks (GSM8K, MATH, SVAMP, and ARC-Challenge). Full hyperparameters, training protocols, held-out split construction, and the rerun/best-of-candidates selection procedure are specified in Appendix A and Section 5.1. The cross-signal NTK diagnostic $K _ { D R } ( n )$ and the M3 gating rules are specified in closed form in Sections 3.4–4 and Algorithm 2; no proprietary data or infrastructure is required to reproduce the main results.

## REFERENCES

Rishabh Agarwal et al. On-policy distillation of language models: Learning from self-generated mistakes. ICLR, 2024.

Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. GradNorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In International Conference on Machine Learning, 2018.

DeepSeek-AI. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Ken Ding. HDPO: Hybrid distillation policy optimization via privileged self-distillation. arXiv preprint arXiv:2603.23871, 2026.

Yuxian Gu et al. MiniLLM: Knowledge distillation of large language models. In International Conference on Learning Representations, 2024.

Jonas Hubotter, Frederike L ¨ ubeck, Lejs Behric, Anton Baumann, et al. Reinforcement learning via¨ self-distillation. arXiv preprint arXiv:2601.20802, 2026.

Arthur Jacot, Franck Gabriel, and Clement Hongler. Neural tangent kernel: Convergence and gen- ´ eralization in neural networks. Advances in Neural Information Processing Systems, 31, 2018.

Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In IEEE Conference on Computer Vision and Pattern Recognition, 2018.

Yixing Li, Yuxian Gu, Li Dong, Dequan Wang, Yu Cheng, and Furu Wei. Direct preference knowledge distillation for large language models. arXiv preprint arXiv:2406.19774, 2024.

Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu. Conflict-averse gradient descent for multi-task learning. In Advances in Neural Information Processing Systems, volume 34, 2021.

Shikun Liu, Edward Johns, and Andrew J Davison. End-to-end multi-task learning with attention. In IEEE Conference on Computer Vision and Pattern Recognition, 2019.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding R1-Zero-Like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025.

Aviv Navon, Aviv Shamsian, Idan Achituve, Haggai Maron, Kenji Kawaguchi, Gal Chechik, and Ethan Fetaya. Multi-task learning as a bargaining game. In International Conference on Machine Learning, 2022.

OpenAI. Learning to reason with LLMs. 2024.

Xiaohan Qin, Xiaoxing Wang, Ning Liao, and Junchi Yan. NTKMTL: Mitigating task imbalance in multi-task learning from neural tangent kernel perspective. Advances in Neural Information Processing Systems, 2025. arXiv:2510.18258.

Yi Ren, Danica J Sutherland, et al. Learning dynamics of LLM finetuning. In International Conference on Learning Representations, 2025.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Ozan Sener and Vladlen Koltun. Multi-task learning as multi-objective optimization. In Advances in Neural Information Processing Systems, volume 31, 2018.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Joar Skalse, Nikolaus H R Howe, Dmitrii Krasheninnikov, and David Krueger. Defining and characterizing reward hacking. In Advances in Neural Information Processing Systems, volume 35, 2022.

Chenxu Yang, Chuanyu Qin, Qingyi Si, Minghui Chen, Naibin Gu, Dingyu Yao, Zheng Lin, Weiping Wang, Jiaqi Wang, and Nan Duan. Self-distilled RLVR. arXiv preprint arXiv:2604.03128, 2026.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. DAPO: An open-source LLM reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning. In Advances in Neural Information Processing Systems, volume 33, 2020.

Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-distilled reasoner: On-policy self-distillation for large language models. arXiv preprint arXiv:2601.18734, 2026.

Table 4: Default experimental hyperparameters.
<table><tr><td>Component</td><td>Hyperparameter</td><td>Value</td></tr><tr><td>LoRA</td><td>Rank r</td><td>64</td></tr><tr><td>LoRA</td><td>Scaling αLoRA</td><td>128</td></tr><tr><td>Distillation</td><td>Divergence</td><td>Generalized JSD</td></tr><tr><td>Distillation</td><td>JSD coefficient λ</td><td></td></tr><tr><td>Distillation</td><td>Token clipping τ</td><td></td></tr><tr><td>Reinforcement learning</td><td>Rollouts per group G</td><td>8</td></tr><tr><td>Hybrid baseline</td><td>Default mixing coefficient α</td><td>0.5</td></tr><tr><td>Optimization</td><td>Reference learning rate</td><td> $5 \times 1 0 ^ { - 5 }$ </td></tr></table>

## A EXPERIMENTAL DETAILS

Rollout and optimization protocol. We use the on-policy OPSD construction of Section 3.1. With LoRA disabled, the frozen teacher receives a privileged prompt containing the ground-truth solution, whereas the student receives only the problem and its sampled prefix. Unless otherwise stated, experiments use the defaults in Table 4.

Evaluation metrics and regime statistics. Training reward is the last-20-step mean unless a table states otherwise, and accuracy is measured by greedy decoding on GSM8K $( n = 1 3 1 9 )$ , SVAMP $( n = 6 0 0 )$ , and ARC $( n = 1 1 7 2 )$ . The SVAMP split excludes all prompts used by the 100- and 200-step training runs. For each architecture–dataset pair, κ¯ is the mean of $\kappa _ { t } = \| \dot { \pmb { g } _ { R } ( t ) } \| / \| \pmb { g } _ { D } ( t ) \|$ over RL-active steps of the corresponding naive-Hybrid run and is used as a regime indicator. A run is marked as collapsed when reward remains below 0.05 for ten consecutive steps.

Candidate selection and post-processing. For each method–cell pair, the evaluated candidates include the reference run and, where available, seed replicates, checkpoints every 20 steps for hori zons up to $T \in \{ 1 0 0 , 2 0 0 , 4 0 0 \}$ , and conservative-rate reruns at $1 { - } 2 \times 1 0 ^ { - 5 }$ . M3-Soft additionally includes the evaluated $( \beta , \alpha _ { \mathrm { m a x } } , T )$ sweep. SDPO uses its selected conservative-rate candidate; on InternLM-ARC, this is the 20-step pre-collapse checkpoint. Conservative-rate candidates are made available to all methods on Llama GSM8K and SVAMP. Collapsed runs are retained and reported at their measured reward and accuracy rather than excluded. SWA is applied to the Qwen3- and InternLM-GSM8K chains and to the Llama-GSM8K conservative-rate seed-42 chain using lasttwo-checkpoint averaging. Because candidates and checkpoints are chosen using accuracy and no independent validation split is defined, these entries are reported as best observed rather than validation-selected results. Seed-level robustness and SWA values are reported in Section B.1.

## B EXTENDED EXPERIMENTS

## B.1 DETAILED CROSS-ARCHITECTURE RESULTS

Accuracy. Table 5 directly compares M3-Soft with the strongest non-M3 baseline in each architecture–dataset cell. Under the per-cell best-observed protocol, M3-Soft matches or exceeds the strongest baseline in all 12 cells (11 strict wins, one tie). The largest gains occur on Qwen2.5- SVAMP (0.905, +4.5pp over RLSD), Qwen2.5-ARC (0.736, +4.1pp over OPSD+GradNorm), and InternLM-SVAMP (0.777, +4.0pp over RLSD); Llama-GSM8K is the exact tie with RLSD at 0.568.

Seed robustness and SWA. Seed replicates qualify the best-observed results in Table 5. Widemargin cells. The ranking is stable on Qwen2.5-SVAMP, where the M3-Soft seed mean remains +2.4pp above RLSD, and on Llama-SVAMP, where both replicates exceed GRPO by +7.7 and +10.2pp. Qwen2.5-GSM8K also wins for both available replicates, while two of three Qwen2.5- ARC seeds exceed the strongest baseline and the third ties it. Tight-margin cells. The Qwen3- GSM8K replicates remain within 1.2pp below RLSD before SWA, and the InternLM-GSM8K win is likewise obtained only after averaging. SWA raises the selected Qwen3-, InternLM-, and Llama-GSM8K chains by 0.013, 0.015, and 0.004, respectively, yielding accuracies of 0.8324, 0.501, and 0.568; it is not uniformly beneficial, decreasing the Llama seed-43 chain from 0.562 to 0.552. Llama-GSM8K is seed-fragile at the reference learning rate, where two of three replicates collapse, but both conservative-rate replicates survive (0.568/0.552). We therefore treat the tight cells as best-observed parity rather than seed-robust separation.

Table 5: Accuracy of M3-Soft versus the strongest non-M3 baseline.
<table><tr><td>Architecture</td><td colspan="3">Best non-M3 baseline</td><td colspan="3">M3-Soft</td><td rowspan="2">M3-Soft ≥ Baseline</td></tr><tr><td></td><td>GSM8K</td><td>SVAMP</td><td>ARC</td><td>GSM8K</td><td>SVAMP</td><td>ARC</td></tr><tr><td>Qwen3-1.7B</td><td>0.8317</td><td>0.940</td><td>0.796</td><td>0.8324</td><td>0.945</td><td>0.807</td><td>3/3</td></tr><tr><td>Qwen2.5-1.5B</td><td>0.735</td><td>0.860</td><td>0.695</td><td>0.752</td><td>0.905</td><td>0.736</td><td>3/3</td></tr><tr><td>InternLM2.5-1.8B</td><td>0.492</td><td>0.737</td><td>0.604</td><td>0.501</td><td>0.777</td><td>0.608</td><td>3/3</td></tr><tr><td>Llama-3.2-1B</td><td>0.568</td><td>0.730</td><td>0.545</td><td>0.568</td><td>0.750</td><td>0.556</td><td>3/3</td></tr><tr><td colspan="6">Total: 11 strict wins and 1 exact tie</td><td></td><td>12/12</td></tr></table>

Training reward and magnitude ratio. Table 6 reports per-cell training rewards and the exact κ¯ measured on the corresponding naive-Hybrid runs. The largest reference-rate separations occur on Llama-GSM8K (κ¯ = 7,690; M3-Soft 0.647, 20.9× over Hybrid 0.031) and Llama-SVAMP $( \bar { \kappa } = 1 2 , 7 0 5 ; \mathrm { M } 3 \mathrm { - } \mathrm { S o f t } 0 . 8 2 2$ , 11.4× over GRPO 0.072, while Hybrid reaches 0.000). At the low-κ end, Qwen3-ARC (κ¯ = 2,465) favors Hybrid in training reward (0.919 vs. 0.775 for the extendedtraining M3-Soft candidate). For Llama-GSM8K, this table reports the reference-rate M3-Soft run (0.647), whereas Table 1 uses the conservative-rate SWA chain selected by accuracy (reward 0.591, accuracy 0.568). Because the M3-Soft column includes tuned and, where marked, extended-training candidates, this table is an optimization diagnostic rather than a matched-budget comparison.

Table 6: Per-cell training reward (last-20 mean) and mean gradient-magnitude ratio κ¯. M3-Soft reports the selected evaluated configuration; <sup>∗</sup> denotes extended training, and κ¯ is measured on the corresponding naive-Hybrid run.
<table><tr><td>Architecture</td><td>Dataset</td><td>GRPO</td><td>OPSD</td><td>OPSD+GradNorm</td><td>M3-Select M3-Soft</td><td></td><td>κ</td></tr><tr><td rowspan="3">Qwen3-1.7B</td><td>GSM8K</td><td>0.838</td><td>0.825</td><td>0.731</td><td>0.844</td><td>0.881</td><td>3,404</td></tr><tr><td>SVAMP</td><td>0.922</td><td>0.894</td><td>0.894</td><td>0.947</td><td>0.969</td><td>5,080</td></tr><tr><td>ARC</td><td>0.741</td><td>0.919</td><td>0.769</td><td>0.263</td><td>0.775*</td><td>2,465</td></tr><tr><td rowspan="3">Qwen2.5-1.5B</td><td>GSM8K</td><td>0.806</td><td>0.747</td><td>0.653</td><td>0.153</td><td>0.828*</td><td>2,467</td></tr><tr><td>SVAMP</td><td>0.769</td><td>0.766</td><td>0.681</td><td>0.331</td><td>0.778*</td><td>2,234</td></tr><tr><td>ARC</td><td>0.659</td><td>0.650</td><td>0.644</td><td>0.319</td><td>0.700</td><td>2,184</td></tr><tr><td rowspan="3">InternLM2.5-1.8B</td><td>GSM8K</td><td>0.369</td><td>0.428</td><td>0.306</td><td>0.100</td><td>0.472*</td><td>2,745</td></tr><tr><td>SVAMP</td><td>0.594</td><td>0.563</td><td>0.594</td><td>0.366</td><td>0.688*</td><td>2,690</td></tr><tr><td>ARC</td><td>0.616</td><td>0.634</td><td>0.666</td><td>0.416</td><td>0.682*</td><td>1,423</td></tr><tr><td rowspan="3">Llama-3.2-1B</td><td>GSM8K</td><td>0.000</td><td>0.031</td><td>0.006</td><td>0.597</td><td>0.647</td><td>7,690</td></tr><tr><td>SVAMP</td><td>0.072</td><td>0.000</td><td>0.038</td><td>0.488</td><td>0.822</td><td>12,705</td></tr><tr><td>ARC</td><td>0.522</td><td>0.469</td><td>0.353</td><td>0.406</td><td>0.650*</td><td>1,562</td></tr></table>

## B.2 LONG-HORIZON STABILITY AND RUNTIME DIAGNOSTICS

Cross-dataset long-horizon results. Table 7 extends the 500-step evaluation to MATH, SVAMP, and ARC-Challenge using Qwen3-1.7B. Under the reference configurations, prolonged training can still trigger collapse beyond GSM8K: GRPO and OPSD+GradNorm collapse on MATH, while Hybrid collapses on SVAMP. In contrast, both M3 variants remain non-collapsed across all three datasets.

Phase-wise collapse dynamics. Table 8 resolves the Qwen3-1.7B GSM8K runs into 100-step phases. Collapse denotes reward below 0.05 for ten consecutive steps. Unless noted otherwise, the runs use the reference learning rate $5 \times 1 0 ^ { - 5 } ;$ all entries are training rewards rather than held-out accuracies.SDPO collapses first at step 50, followed by Hybrid at 221, OPSD+GradNorm at 358, RLSD at 362, and GRPO at 412. In contrast, M3-Select does not trigger the collapse criterion within 500 steps. Phase-wise M3-Soft results are unavailable, but its last-50 reward is reported in Table 2.

Post-collapse conflict diagnostic. After Hybrid collapses, its measured conflict rate falls to zero because the RL gradient vanishes, not because the two signals become compatible. M3-Select instead maintains an active conflict rate near $C _ { \mathrm { N T K } } = 0 . 3$ while preserving reward (Figure 6).

Table 7: Long-horizon reward across datasets on Qwen3-1.7B after 500 steps (last-50 mean). M3- Soft uses the best evaluated gate configuration per dataset; other methods use the reference configuration. Bold marks the column maximum, and ↓ denotes collapse.
<table><tr><td>Method</td><td>MATH</td><td>SVAMP</td><td>ARC-Challenge</td></tr><tr><td colspan="4">Reference and adaptive baselines</td></tr><tr><td>GRPO</td><td>↓0.018</td><td>0.900</td><td>0.773</td></tr><tr><td>Hybrid (OPSD)</td><td>0.448</td><td>↓0.170</td><td>0.765</td></tr><tr><td>OPSD+GradNorm</td><td>↓0.015</td><td>0.973</td><td>0.790</td></tr><tr><td colspan="4">Ours: boundary-gated mixing (M3)</td></tr><tr><td>M3-Select</td><td>0.367</td><td>0.943</td><td>0.282</td></tr><tr><td>M3-Soft</td><td>0.523</td><td>0.950</td><td>0.667</td></tr></table>

Table 8: Phase-wise training reward over 500 steps on Qwen3-1.7B GSM8K. “Full” is the all-step mean, “Trend” gives the first collapse step, and <sup>†</sup> denotes batch size 1.
<table><tr><td>Method</td><td>1-100</td><td>101-200</td><td>201-300</td><td>301-400</td><td>401-500</td><td>Full</td><td>Trend</td></tr><tr><td>Pure GRPO</td><td>0.840</td><td>0.801</td><td>0.809</td><td>0.802</td><td>0.075</td><td>0.665</td><td>collapse @412</td></tr><tr><td>Hybrid (OPSD, α = 0.5)</td><td>0.859</td><td>0.762</td><td>0.114</td><td>0.006</td><td>0.001</td><td>0.348</td><td>collapse @221</td></tr><tr><td>OPSD+GradNorm†</td><td>0.877</td><td>0.823</td><td>0.868</td><td>0.464</td><td>0.000</td><td>0.606</td><td>collapse @358</td></tr><tr><td>RLSD (Yang et al., 2026)</td><td>0.855</td><td>0.874</td><td>0.871</td><td>0.259</td><td>0.005</td><td>0.573</td><td>collapse @362</td></tr><tr><td>SDPO (Hübotter et al., 2026)</td><td>0.306</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.061</td><td>collapse @50</td></tr><tr><td>M3-Select  $( \alpha _ { \mathrm { m a x } } = 0 . 1 5 )$ </td><td>0.846</td><td>0.818</td><td>0.853</td><td>0.846</td><td>0.810</td><td>0.835</td><td>stable</td></tr></table>

## B.3 GATE SENSITIVITY ACROSS REGIMES

Controlled sharpness sweep. We first isolate gate sharpness on Qwen3-1.7B GSM8K at 100 steps. With $\alpha _ { \mathrm { m a x } } = 0 . 5$ , the gentle $\beta = 0 . 5$ gate attains the largest last-20 reward (0.881), compared with 0.856 for $\beta = 1$ , 0.850 for GRPO, and 0.844 for hard M3-Select (Table 9). The soft-gate configurations have similar values of the count-based budget proxy $\alpha _ { \mathrm { e f f } }$ (0.439–0.446), consistent with gate shape, rather than a large change in this proxy—driving the observed differences.

Table 9: M3-Soft sharpness sweep on Qwen3-1.7B GSM8K at 100 steps $( \alpha _ { \mathrm { m a x } } = 0 . 5 ) . ~ \alpha _ { \mathrm { e f f } }$ denotes the mean teacher weight. For hard selection, the measured conflict rate is C<sub>NTK</sub> ≈ 0.40, so approximately 60% of tokens retain weight $\alpha _ { \mathrm { m a x } } ,$ giving $\alpha _ { \mathrm { e f f } } \approx 0 . 5 \times 0 . 6 0 = 0 . 3 0$
<table><tr><td>Method</td><td> $\beta$  Last-10</td><td>Last-20</td><td></td><td> $\alpha _ { \mathrm { e f f } }$ </td></tr><tr><td colspan="2">GRPO OPSD (uniform)</td><td>0.858 0.833</td><td>0.850 0.825</td><td></td><td>0.500</td></tr><tr><td colspan="2">M3-Select (hard)</td><td>0.855</td><td></td><td>0.844</td><td>~0.30</td></tr><tr><td colspan="2">M3-Soft</td><td>∞ 10</td><td>0.875</td><td>0.847</td><td>0.446</td></tr><tr><td colspan="2">M3-Soft</td><td>5</td><td>0.813</td><td>0.838</td><td>0.439</td></tr><tr><td colspan="2">M3-Soft</td><td>1</td><td>0.906</td><td>0.856</td><td>0.444</td></tr><tr><td colspan="2">M3-Soft</td><td>0.5</td><td>0.913</td><td>0.881</td><td>0.446</td></tr></table>

Cross-architecture reward sensitivity. Table 10 fixes the horizon at 100 steps and compares $\beta \in$ {0.5, 1, 5} across all 12 architecture–dataset cells. A gentle gate $( \beta \in \{ 0 . 5 , 1 \} )$ ) is the best evaluated M3-Soft setting in 11/12 cells. Both high-κ Llama arithmetic cells favor β = 1, whereas InternLM-SVAMP is the sole $\beta = 5$ exception. This table diagnoses sensitivity within M3-Soft; it is not the source of the best-observed headline in Table 1.

Sensitivity. Training reward does not by itself select the best gate for accuracy. On Qwen2.5- SVAMP, the very-soft recipe $( \beta , \alpha _ { \mathrm { m a x } } ) \ : = \ : ( 0 . 1 , 0 . 0 0 1 )$ reaches 0.884 ± 0.026 across three seeds (0.855/0.892/0.905), exceeding RLSD’s 0.860 by 2.4pp in the seed mean. On InternLM-SVAMP, the same recipe gives 0.710/0.777/0.722: the mean (0.736) is at parity with RLSD (0.737), while the best observed seed reaches 0.777. Thus, the Qwen2.5 gain is seed-robust, whereas the InternLM headline is best-observed rather than a mean separation.

![](images/d983a3e4bd3b05963bab7cdbadfb462dbb3f2be532737581a9faa5635406093c.jpg)  
Figure $6 { : }$ Runtime token-level conflict on Qwen3-0.6B GSM8K. Hybrid’s post-collapse drop reflects a vanishing RL gradient; M3-Select remains active near $C _ { \mathrm { N T K } } = 0 . 3 .$

Table 10: M3-Soft sharpness sensitivity at 100 steps (last-20 mean reward). $\alpha _ { \mathrm { m a x } } = 0 . 0 5$ by default and 0.025 on Qwen3-SVAMP.
<table><tr><td>Model</td><td>Dataset</td><td> $\beta = \mathbf { 0 . 5 }$ </td><td> $\beta = { \bf 1 }$ </td><td> $\beta = 5$ </td><td>GRPO</td></tr><tr><td rowspan="3">Qwen3-1.7B</td><td>GSM8K</td><td>0.834</td><td>0.856</td><td>0.850</td><td>0.838</td></tr><tr><td>SVAMP</td><td>0.991</td><td>0.969</td><td>0.981</td><td>0.953</td></tr><tr><td>ARC</td><td>0.516</td><td>0.350</td><td>0.356</td><td>0.741</td></tr><tr><td rowspan="3">Llama-3.2-1B</td><td>GSM8K</td><td>0.569</td><td>0.647</td><td>0.353</td><td>↓0.000</td></tr><tr><td>SVAMP</td><td>0.456</td><td>0.822</td><td>0.609</td><td>0.072</td></tr><tr><td>ARC</td><td>0.559</td><td>0.500</td><td>0.388</td><td>0.522</td></tr><tr><td rowspan="3">Qwen2.5-1.5B</td><td>GSM8K</td><td>0.697</td><td>0.728</td><td>0.169</td><td>0.806</td></tr><tr><td>SVAMP</td><td>0.794</td><td>0.691</td><td>0.463</td><td>0.769</td></tr><tr><td>ARC</td><td>0.700</td><td>0.475</td><td>0.369</td><td>0.659</td></tr><tr><td rowspan="3">InternLM2.5-1.8B</td><td>GSM8K</td><td>0.234</td><td>0.269</td><td>0.138</td><td>0.369</td></tr><tr><td>SVAMP</td><td>0.594</td><td>0.366</td><td>0.744</td><td>0.594</td></tr><tr><td>ARC</td><td>0.469</td><td>0.575</td><td>0.472</td><td>0.616</td></tr></table>

## B.4 MECHANISTIC VALIDATION

The following diagnostics test the mechanism at progressively coarser levels. We first verify the predicted one-step effect after partitioning tokens by their pre-update cross-signal NTK, then compare response-level and token-level notions of conflict, and finally summarize the aggregate behavior across settings.

Table 11: Response-level semantic conflict (Qwen3-0.6B, 208 trajectories): on-policy = teacher evaluates the student’s own trajectory, off-policy = ground-truth contexts; true opposition = wrong trajectories with opposing RL and distillation gradients.
<table><tr><td>Mode</td><td>Conflict %</td><td> $\overline { { \cos } } ( { \pmb g } _ { R } , { \pmb g } _ { D } )$ </td><td>True opposition</td><td>Kmedian</td><td> $\bar { \mathcal { L } } _ { D }$ </td></tr><tr><td>On-policy OPSD</td><td>75%</td><td>+0.08</td><td>25/157 (16%)</td><td>388</td><td>0.098</td></tr><tr><td>Off-policy</td><td>75%</td><td>-0.01</td><td>90/157 (57%)</td><td>68</td><td>0.810</td></tr></table>

Response-Level Semantic Conflict. Response-level labels and token-level interactions answer different questions. For each trajectory in a GRPO group, we measure its reward, normalized advantage, trajectory-level gradient cosine cos $\left( g _ { R } , g _ { D } \right)$ , and magnitude ratio κ. Across 20 analysis steps with Qwen3-0.6B (batch size 2, group size 8), groups with zero reward variance are excluded because GRPO assigns zero advantage and hence no reward gradient, leaving 208 trajectories. Both modes label 75% of trajectories as semantically conflicting, but they differ sharply in gradient behavior. Among the 157 incorrect trajectories, true opposition occurs in 25 cases (16%) on-policy and 90 cases (57%) off-policy; the corresponding median magnitude ratios are 388 and 68. Within the on-policy sample, the mean cosine is +0.14 on incorrect trajectories and −0.11 on correct trajectories, matching the inversion predicted by Proposition 13. Thus, response-level rejection alone does not determine whether the two gradients oppose each other.

![](images/b5560fc59fa8274485a52ab3a5c772e25c66c8092654bbe837abe005a1412ce2.jpg)

![](images/8077ed1020e86e400eaa9a36fe654844cb2987898a6a6c5aa3f1957ac75990fb.jpg)

![](images/a6fc18f05cf675e79efda7a26e4b8420316f798db980683084cf67c2fae41f6a.jpg)  
Figure 7: On-policy versus off-policy semantic conflict. On-policy distillation has a higher median κ but fewer incorrect trajectories with genuinely opposed reward and distillation gradients; off-policy supervision reverses this pattern.

![](images/6a2c892f480aaadd21007ab2d4a18c2bda91570da512e641bde0c1d08431a18f.jpg)  
Figure 8: Per-position conflict rate C across model scales and datasets. The fraction of positions with negative cross-signal NTK remains between 0.37 and 0.46 (mean ≈ 0.41, dashed line).

Token-Level NTK Conflict. We next measure the same-position cross-signal NTK $K _ { D R } ( n )$ . Positions with $K _ { D R } ( n ) \geq 0$ are locally compatible, whereas positions with $K _ { D R } ( n ) < 0$ receive a teacher component that opposes reward progress. Across the measured model scales and datasets, the count-based conflict rate ranges from 0.37 to 0.46, with a mean of approximately 0.41 (Figure 8). The narrow range shows that the conflict observed in the main-text rollout is not isolated to one model or dataset, without implying that all settings have identical conflict structure.

Figure 9 restores token identities for one correct and one incorrect response to the same prompt. Across the four paired case studies, conflict covers 44.3% of positions on correct rollouts and 54.9% on incorrect rollouts, a difference of 10.6 percentage points. Every rollout nevertheless interleaves compatible and conflicting positions, so these examples support token-level localization rather than a response-level cutoff. The count difference is descriptive for the eight visualized rollouts and is not presented as a population-level estimate.

Aggregate Conflict Diagnostics. Figure 11 summarizes the reference Qwen diagnostics. Across the measured scales and datasets, the mean magnitude ratio remains in the 3,400–4,000 range, while the parameter-level cosine averages −0.009. The M<sup>3</sup>-Select trace also shows higher count-based conflict at lower-reward steps. This last relationship is descriptive: the plot does not establish that conflict rate alone causes or predicts method performance.

## B.5 ADDITIONAL ABLATIONS AND ILLUSTRATIVE EXAMPLES

LoRA Rank Ablation. We vary the LoRA rank from 8 to 128 on Qwen3-1.7B GSM8K while holding the remaining M<sup>3</sup>-Soft configuration fixed. The mean reward ranges from 0.824 to 0.844 across this 16× change in trainable rank, with the best value at $r = 6 4$ and a total spread of 0.020 (Table 12). Thus, performance in this setting is not sensitive to the tested LoRA rank. Because κ and the token-level conflict rate were not separately recorded for every rank, this ablation supports reward robustness rather than rank invariance of the underlying diagnostics.

Prompt: If left( r + frac{1}{r} right)^2 = 3, then find r^3 + frac{1}{r^3}.

Ground-truth answer: 0  
(a) correct response, reward=1, advantage=+0.72, conflict rate C<sub>NTK</sub>=42%
<table><tr><td rowspan=1 colspan=1>We</td><td rowspan=1 colspan=1>•are</td><td rowspan=1 colspan=1>‘given</td><td rowspan=1 colspan=1>:tt</td><td rowspan=1 colspan=1>$$</td><td rowspan=1 colspan=1>t</td><td rowspan=1 colspan=1>I</td><td rowspan=1 colspan=1>left</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>r</td><td rowspan=1 colspan=1>+</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>frac</td><td rowspan=1 colspan=1>{</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>}{</td></tr><tr><td rowspan=1 colspan=1>r</td><td rowspan=1 colspan=1>}</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>right</td><td rowspan=1 colspan=1>)^</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>=</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>t</td><td rowspan=1 colspan=1>$$</td><td rowspan=1 colspan=1>t t</td><td rowspan=1 colspan=1>We</td><td rowspan=1 colspan=1>are</td><td rowspan=1 colspan=1>-to</td><td rowspan=1 colspan=1>find</td></tr><tr><td rowspan=1 colspan=1>:t</td><td rowspan=1 colspan=1>$$</td><td rowspan=1 colspan=1>t</td><td rowspan=1 colspan=1>r</td><td rowspan=1 colspan=1>A</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>+</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>frac</td><td rowspan=1 colspan=1>{</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>}{</td><td rowspan=1 colspan=1>r</td><td rowspan=1 colspan=1>A</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>}</td></tr><tr><td rowspan=1 colspan=1>$$</td><td rowspan=1 colspan=1>tt</td><td rowspan=1 colspan=1>…tt</td><td rowspan=1 colspan=1>##＃</td><td rowspan=1 colspan=1>Step</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>• </td><td rowspan=1 colspan=1>·Expand</td><td rowspan=1 colspan=1>the</td><td rowspan=1 colspan=1>·given</td><td rowspan=1 colspan=1>equat..</td><td rowspan=1 colspan=1>t t</td><td rowspan=1 colspan=1>$$</td><td rowspan=1 colspan=1>t</td><td rowspan=1 colspan=1>I</td></tr></table>

(b) wrong response, reward=0, advantage=-1.21, conflict rate $C _ { \mathrm { N T K } } { = } { \bf 6 1 \% }$  
![](images/8a60210856921b4231bc6dedfa93ff739de936fa16b4cffe2ae3e49365af7fc5.jpg)

Figure 9: A readable token-level case study for a correct $( A > 0 )$ and an incorrect $( A < 0 )$ response to the same prompt. Darker red denotes stronger local conflict $( K _ { D R } ( n ) < 0 )$ , darker blue denotes stronger compatibility, and pale colors denote weak interaction. Both responses interleave the two signal types.  
Table 12: LoRA-rank ablation for M<sup>3</sup>-Soft on Qwen3-1.7B GSM8K over 100 training steps.
<table><tr><td>LoRA Rank</td><td> $r = 8$ </td><td> $r = 1 6$ </td><td> $r = 3 2$ </td><td> $r = 6 4$ </td><td> $r = 1 2 8$ </td></tr><tr><td>Mean Reward</td><td>0.826</td><td>0.824</td><td>0.828</td><td>0.844</td><td>0.841</td></tr></table>

Illustrative Failure Patterns. The following cases are stylized examples distilled from qualitative patterns observed in the GSM8K and SVAMP runs; they are not verbatim training trajectories or additional controlled experiments. Their purpose is to show how magnitude imbalance, group-level cancellation, and token-level sign conflict can appear in concrete reasoning traces. In the examples, + and − denote positive and negative group-relative advantages, respectively.

## Bad Case 1: length inflation under drowning (κ ≫ κ<sup>∗</sup>)

On Llama-3.2-1B SVAMP $( \bar { \kappa } \approx 1 . 3 \times 1 0 ^ { 4 } )$ , the RL gradient dominates the teacher gradient by roughly four orders of magnitude. Because the verifier rewards only the final numeric answer, the group-normalized advantage rewards any continuation that eventually reaches the correct digit and penalizes short-but-wrong ones. When distillation is drowned out, the model no longer receives the teacher’s “brevity + structure” prior, response length inflates steadily as training proceeds, and reward collapses to 0.072 (Table 6, row Llama/SVAMP/GRPO). A stylized postdrowning rollout of the kind we observe:

Let x be the answer. We are told that Melanie has 3 · 4   
apples...so x = 12. But wait, let us re-verify: 3 · 4 = 12,   
so indeed x = 12. Actually, checking once more, x = 12.   
...Answer: 12.

The re-verification loops are reward-neutral (verifier only checks $\mathbf { \dot { \mathbf { \ i } } } ^ { 4 6 } \mathbf { 1 2 } ^ { , 9 } )$ but teachernegative. Under M3 the magnitude normalization restores the teacher’s unit-scale voice—the brevity+structure prior is no longer drowned by the $1 0 ^ { 4 }$ norm ratio—and the boundary admits this anti-repetition pressure exactly where it is reward-compatible $( K _ { D R } ( n ) \geq 0 , { \mathrm { e . g . } }$ in negative-advantage rollouts whose RL residual also pushes against the loops), so length inflation is suppressed and reward reaches 0.822 (same table row, M3-Soft; the hard-gated M3- Select variant recovers 0.488). This is the token-level dual of the length-drift phenomenon

Prompt: Compute det( 2 & 0 & -1 ; 7 & 4 & -3 ; 2 & 2 & 5 ).  
Ground-truth answer: 46
<table><tr><td rowspan=1 colspan=16>(a) correct response, reward=1, advantage=+0.72, CNTk=49% (full response; first 48 tokens shown)</td></tr><tr><td rowspan=1 colspan=1>We</td><td rowspan=1 colspan=1>are</td><td rowspan=1 colspan=1>&#x27;given</td><td rowspan=1 colspan=1>•the</td><td rowspan=1 colspan=1>-deter...</td><td rowspan=1 colspan=1>•of</td><td rowspan=1 colspan=1>•à</td><td rowspan=1 colspan=1>.$</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>-I</td><td rowspan=1 colspan=1>times</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>.$</td><td rowspan=1 colspan=1>·matrix</td></tr><tr><td rowspan=1 colspan=1>:t t</td><td rowspan=1 colspan=1>$$</td><td rowspan=1 colspan=1>t</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>begin</td><td rowspan=1 colspan=1>{</td><td rowspan=1 colspan=1>vm</td><td rowspan=1 colspan=1>atrix</td><td rowspan=1 colspan=1>è</td><td rowspan=1 colspan=1>•t</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>·&amp;</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>·&amp;</td><td rowspan=1 colspan=1>•</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>-11</td><td rowspan=1 colspan=1>t</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>·&amp;</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>·&amp;</td><td rowspan=1 colspan=1>•-</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>t</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>·&amp;</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2</td></tr></table>

<table><tr><td rowspan=1 colspan=16>(b) wrong response, reward=0, advantage=-1.21, CNTk=52% (full response; first 48 tokens shown)</td></tr><tr><td rowspan=1 colspan=1>We</td><td rowspan=1 colspan=1>are</td><td rowspan=1 colspan=1>asked</td><td rowspan=1 colspan=1>·to</td><td rowspan=1 colspan=1>•compu..</td><td rowspan=1 colspan=1>·the</td><td rowspan=1 colspan=1>·deter...</td><td rowspan=1 colspan=1>•of</td><td rowspan=1 colspan=1>-the</td><td rowspan=1 colspan=1>•follo...</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>·matrix</td><td rowspan=1 colspan=1>:t t</td></tr><tr><td rowspan=1 colspan=1>$$</td><td rowspan=1 colspan=1>t</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>begin</td><td rowspan=1 colspan=1>{</td><td rowspan=1 colspan=1>vm</td><td rowspan=1 colspan=1>atrix</td><td rowspan=1 colspan=1>}</td><td rowspan=1 colspan=1>t</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>·&amp;</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>·&amp;</td><td rowspan=1 colspan=1>i=</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>-1I</td><td rowspan=1 colspan=1>t</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>·&amp;</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>·&amp;</td><td rowspan=1 colspan=1>i=</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>三</td><td rowspan=1 colspan=1>t</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>•&amp;</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>·&amp;</td></tr></table>

Prompt: On a balance scale, 3 green balls balance 6 blue balls, 2 yellow balls balance 5 blue balls, and 6 blue balls balance 4 white balls. How many blue balls are needed to balance 4 green, 2

## Ground-truth answer: 16

(c) correct response, reward=1, advantage=+0.35, C<sub>NTK</sub>=42% (full response; first 48 tokens shown)
<table><tr><td rowspan=1 colspan=1>We</td><td rowspan=1 colspan=1>are</td><td rowspan=1 colspan=1>&#x27;given</td><td rowspan=1 colspan=1>•sever...</td><td rowspan=1 colspan=1>-balan...</td><td rowspan=1 colspan=1>relat...</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-and</td><td rowspan=1 colspan=1>•we</td><td rowspan=1 colspan=1>need</td><td rowspan=1 colspan=1>·to</td><td rowspan=1 colspan=1>-deter...</td><td rowspan=1 colspan=1>-how</td><td rowspan=1 colspan=1>·many</td><td rowspan=1 colspan=1>.**</td><td rowspan=1 colspan=1>blue</td></tr><tr><td rowspan=1 colspan=1>balls</td><td rowspan=1 colspan=1>**</td><td rowspan=1 colspan=1>-are</td><td rowspan=1 colspan=1>·needed</td><td rowspan=1 colspan=1>·to</td><td rowspan=1 colspan=1>·balan...</td><td rowspan=1 colspan=1>, **</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>green</td><td rowspan=1 colspan=1>•balls</td><td rowspan=1 colspan=1>**,</td><td rowspan=1 colspan=1>.**</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>·yellow</td><td rowspan=1 colspan=1>·balls</td><td rowspan=1 colspan=1>*，</td></tr><tr><td rowspan=1 colspan=1>·and</td><td rowspan=1 colspan=1>.**</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>·white</td><td rowspan=1 colspan=1>·balls</td><td rowspan=1 colspan=1>**</td><td rowspan=1 colspan=1>tt</td><td rowspan=1 colspan=1>--tt</td><td rowspan=1 colspan=1>###</td><td rowspan=1 colspan=1>Step</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>·List</td><td rowspan=1 colspan=1>the</td><td rowspan=1 colspan=1>given</td></tr></table>

(d) wrong response, reward=0, advantage=-2.47, C<sub>NTK</sub>=62% (full response; first 48 tokens shown)
<table><tr><td rowspan=1 colspan=1>We</td><td rowspan=1 colspan=1>are</td><td rowspan=1 colspan=1>·given</td><td rowspan=1 colspan=1>•à</td><td rowspan=1 colspan=1>set</td><td rowspan=1 colspan=1>:of</td><td rowspan=1 colspan=1>·balan...</td><td rowspan=1 colspan=1>scale</td><td rowspan=1 colspan=1>-relat...</td><td rowspan=1 colspan=1>and</td><td rowspan=1 colspan=1>-need</td><td rowspan=1 colspan=1>·to</td><td rowspan=1 colspan=1>find</td><td rowspan=1 colspan=1>how</td><td rowspan=1 colspan=1>·many</td><td rowspan=1 colspan=1>**</td></tr><tr><td rowspan=1 colspan=1>blue</td><td rowspan=1 colspan=1>-balls</td><td rowspan=1 colspan=1>*</td><td rowspan=1 colspan=1>are</td><td rowspan=1 colspan=1>·needed</td><td rowspan=1 colspan=1>·to</td><td rowspan=1 colspan=1> *</td><td rowspan=1 colspan=1>balance</td><td rowspan=1 colspan=1>棠</td><td rowspan=1 colspan=1>ìt</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>*</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>green</td><td rowspan=1 colspan=1>•balls</td><td rowspan=1 colspan=1>菜</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>.**</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>·yellow</td><td rowspan=1 colspan=1>·balls</td><td rowspan=1 colspan=1>菜</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>.**</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>·white</td><td rowspan=1 colspan=1>·balls</td><td rowspan=1 colspan=1>菜仁t</td><td rowspan=1 colspan=1>…tt</td><td rowspan=1 colspan=1>###</td><td rowspan=1 colspan=1>-Step</td><td rowspan=1 colspan=1></td></tr></table>

Prompt: Tom has a red marble, a green marble, a blue marble, and three identical yellow marbles. How many different groups of two marbles can Tom choose?

Ground-truth answer: 7  
(e) correct response, reward=1, advantage=+1.62, C<sub>NTK</sub>=42% (full response; first 48 tokens shown)  
![](images/da8e0d03c45def1180f3c08cf3134a48ea74821705c8e8e978e66b015895b351.jpg)  
Figure 10: Three additional paired token-level case studies. Headers report $C _ { \mathrm { N T K } }$ over each full 200-token response; the first 48 tokens are displayed. The incorrect rollout has the higher conflict rate in each pair (49% → 52%, 42% → 62%, and 42% → 45%), while every rollout contains both compatible and conflicting positions.

![](images/829865add30f4d1aa25663d989ad68c6ee40a326e3077251cc17f47ffb1d3d6f.jpg)

![](images/951cb1d796f346f941a66f3e8499b5449fb5fbee122f3e7338dd8e474c2adac7.jpg)

![](images/49b77a582824868336cbef4604814832ec6d0db5456876382ac61eb6dbf2c437.jpg)  
Figure 11: Aggregate conflict diagnostics: (a) mean magnitude ratio across settings; (b) parameterlevel gradient cosine; and (c) token-level conflict rate versus reward for $\mathbf { M } ^ { 3 } .$ -Select.

reported in prior exploration-boundary work: the drowning threshold acts as an implicit “eossuppression” bias.

## Bad Case 2: shared-prefix contamination in the same rollout group

GRPO computes advantages per rollout, so a group of G trajectories that share a long prefix will apply opposite-sign updates to that prefix whenever their final answers disagree. Consider a GSM8K group of two rollouts from the same prompt:

Rollout 1 (+): Let x be the number of apples. x = 5 · 3 =   
15. Answer: 15.   
Rollout 2 (−): Let x be the number of apples. x = 5+3 =   
8. Answer: 8.

Denote the shared prefix as s. The naive RL contribution at any s-token $y _ { n }$ is

$$
\left. \mathcal { G } _ { R } ^ { n } \right| _ { y _ { n } \in s } = - \left[ \left( r _ { 1 } - b \right) + \left( r _ { 2 } - b \right) \right] \left( \pmb { e } _ { y _ { n } } - \pmb { p } _ { S } ^ { n } \right) = 0 ,
$$

since the group-mean baseline $\begin{array} { r } { b = { \frac { 1 } { 2 } } ( r _ { 1 } + r _ { 2 } ) } \end{array}$ centers the two advantages: the RL signal on the shared prefix cancels exactly in this two-rollout group—and nearly so for general $G ,$ where unbalanced +/− counts and length-normalization weights leave a small residual. But the distillation signal $\mathcal { G } _ { D } ^ { n } = { \pmb { p } } _ { S } ^ { n } - { \pmb { p } } _ { T } ^ { n }$ remains fully alive on the same tokens, and the teacher pushes $p _ { S }$ toward its preferred verbalization—while the small residual RL noise flips sign randomly across steps. The result is an unstable prefix whose gradient variance is dominated by cancellation noise (large $\gamma _ { n } ,$ Definition 3 extension in §F). M3-Select computes cos $\varphi _ { n }$ between $\mathcal { G } _ { D } ^ { n }$ and the per-rollout $\mathcal { G } _ { R } ^ { n }$ , detects $\bar { c } _ { n } = + 1$ on the shared prefix of the winning rollout (teacher and RL prefer the same continuation there), and admits the teacher at its full budget $\alpha _ { \mathrm { { m a x } } } { \mathrm { : } }$ after per-token normalization the distillation direction is the only persistent signal on the prefix, anchoring it against the sign-flipping residual noise. The RL term is never masked—its groupaveraged contribution self-cancels as shown above—so reward supervision effectively acts only on the divergence region $( ^ {  } 5 { \cdot } 3 = 1 5 ^ { , > } \mathrm { v s } ^ {  } 5 + 3 = 8 ^ { , 9 } )$ , where the two rollouts’ residuals no longer cancel. This is the on-policy hybrid analogue of the shared-prefix contamination reported by Ren et al. (2025) (“I-ate-lunch, happy/unhappy” example).

## Bad Case 3: correct intermediate step suppressed by wrong final answer

Even when the group contains only one (−) rollout, an internally-correct intermediate step is penalized because GRPO’s outcome reward propagates the trajectory-level sign to every token. Consider a two-step arithmetic rollout:

$$
\begin{array} { r l } & { \mathrm { R o 1 1 o u t } \quad A \quad ( - ) : \quad 5 \cdot 4 = { \bigg [ \ o { 2 0 } { \bigg ] } } , \mathrm { t h e n } 2 0 + 3 = { \bigg [ \ o { 2 4 } { \bigg ] } } . \quad \mathrm { \normalfont ~ \mathscr { A } n s w e r : } \quad 2 4 . } \\ & { \mathrm { R o 1 1 o u t } \quad B \quad ( + ) : \quad 5 \cdot 4 = { \bigg [ \ o { 2 0 } { \bigg ] } } , \mathrm { t h e n } 2 0 + 3 = { \bigg [ \ o { 2 3 } { \bigg ] } } . \quad \mathrm { \normalfont ~ \mathscr { A } n s w e r : } \quad 2 3 . } \end{array}
$$

The intermediate token $\mathbf { \bar { \theta } } ^ { 6 6 } 2 0 ^ { \mathbf { \eta } , 9 }$ is arithmetically correct in both rollouts, and the teacher’s distributional prior p<sub>T</sub> places > 0.9 mass on it. Under naive hybrid,

$$
\left. G _ { R } ^ { n } \right| _ { y _ { n } = " 2 0 ^ { \prime \prime } } ^ { \mathrm { R o l l o u t } A } \ = \ - ( r _ { A } - b ) ( e _ { 2 0 } - p _ { S } ^ { n } ) ,
$$

Table 13: Stylized failure patterns, observable signatures, and corresponding interventions.
<table><tr><td>Failure pattern</td><td>Observable signature</td><td>Intervention</td></tr><tr><td>Length inflation Magnitude drowning</td><td>Large  $\kappa ;$  repetitive reasoning not penalized Magnitude normalization by the final-answer verifier</td><td></td></tr><tr><td>Shared-prefix cancellation</td><td>Weak aggregate reward signal on tokens Per-token compatibility shared by opposite-advantage rollouts</td><td></td></tr><tr><td>Group-level cancellation Correct-intermediate sup- pression</td><td> $K _ { D R } ( n ) < 0$  on a locally correct token in- side a rejected response</td><td>gate  $\mathbf { M } ^ { 3 } .$  -Select or  $\mathbf { M } ^ { 3 }  – S \mathbf { o f t }$ </td></tr><tr><td>Token-level sign conflict Scaffold-token instability</td><td> $| K _ { D R } ( n ) | \approx 0$  on structural or low-content tokens</td><td> $\mathbf { M } ^ { 3 }  – S \mathrm { o f t }$ </td></tr></table>

with $r _ { A } - b < 0 .$ so the RL update lowers the probability of the correct intermediate. The per-position diagnostic (Proposition 11) yields $\langle \mathcal { G } _ { D } ^ { n } , \mathcal { G } _ { R } ^ { n } \rangle \propto - ( r _ { A } - b ) \sigma _ { n } < 0 , \mathrm { s o } \bar { c } _ { n } = - 1$ and M3-Select withholds teacher supervision from Rollout $\mathrm { \nabla } \cdot \mathrm { \nabla } _ { A } \mathrm { \nabla } _ { s }$ update at this token—the gate acts on the teacher, never on RL. The repair instead comes from the group: Rollout $B ^ { * } s$ equal-andopposite RL contribution cancels Rollout $A \ ' \mathrm { s }$ in the aggregate, while on Rollout B the diagnostic reads $\bar { c } _ { n } = + 1$ and the teacher is admitted, so the surviving net update pushes $p _ { S } ( { \breve { \cdots } } 2 { \breve { 0 } } ^ { \prime \prime } ) ^ { \not \prime }$ Correct-intermediate tokens under negative advantage arise whenever a group mixes right and wrong final answers built on shared sub-computations; protecting them through this admitted teacher vote is a driver of the 0.647 vs 0.000 reference-rate gap between M3-Soft and GRPO (Table 6). This is the dense–sparse counterpart ofthe $\cdots 1 + 1 { = } 2 ^ { , 3 }$ intermediate-step cancellation phenomenon documented in prior exploration-boundary analyses.

## Bad Case 4: directional cancellation on high-entropy scaffold tokens

Certain tokens are structurally common to almost every correct and wrong solution: the $^ { 6 6 } = ^ { 5 9 }$ sign, “Answer: $" , " \backslash n "$ , punctuation, and low-content connectives $( ^ {  } \mathrm { s o } ^ { , \cdot } , ^ {  } \mathrm { t h e n } ^ { , \cdot } )$ The teacher assigns $p _ { T } > 0 . 9$ on these, and the RL signal is noisy and near-zero in expectation (they appear roughly equally in + and − rollouts). Yet the per-step RL contribution is non-zero and has large variance, producing a gradient-cancellation rate

$$
\gamma _ { n } ~ = ~ 1 - \frac { \left. \frac { 1 } { G } \sum _ { g } \nabla _ { \pmb { \theta } } \ell _ { g } ( y _ { n } ) \right. ^ { 2 } } { \frac { 1 } { G } \sum _ { g } \left. \nabla _ { \pmb { \theta } } \ell _ { g } ( y _ { n } ) \right. ^ { 2 } }
$$

approaching 1 at these positions, in sharp contrast to content tokens, whose group updates are directionally aligned. Here the soft gate is the right tool: M3-Soft assigns these positions a stable intermediate budget $\alpha _ { n } = \alpha _ { \mathrm { m a x } } \sigma ( \beta \mathrm { c o s } \varphi _ { n } ) \approx \alpha _ { \mathrm { m a x } } / 2$ that, at small $\beta ,$ is insensitive to the noisy sign of $\widehat { \cos { \varphi _ { n } } }$ , whereas M3-Select’s hard indicator flips with that sign and evicts the teacher on roughly half the scaffold positions at random. Since the group-averaged RL residual is near zero-mean here while the normalized teacher direction is persistent, distillation dominates the expected update without any signal being masked—which is why M3-Soft with $\beta { = } 0 . 1$ recovers scaffold-token fluency on the low-κ Qwen2.5-SVAMP cell (§B.3).

Table 13 summarizes what each example is intended to illustrate. The first case concerns magnitude imbalance, the next two concern sign structure under outcome-level credit assignment, and the fourth concerns uncertainty near the compatibility boundary.

## C EXPLORATION BOUNDARY FRAMEWORK: FULL STATEMENTS AND BOUNDARY DEFINITION

This appendix states the intra-RL quantities underlying the cross-signal analysis in Section 3.4 and defines the reward–teacher compatibility boundary.

## C.1 EXPLORATION BOUNDARY AND GRADIENT FOLDING IN RL

Definition 2 (Gradient Diversity and Exploration Boundary). For G rollouts, let $\begin{array} { r l } { \mathbf { \boldsymbol { g } } ^ { ( i ) } } & { { } = \mathbf { \boldsymbol { \mathit { \Phi } } } } \end{array}$ $A _ { i } \nabla _ { \pmb \theta } \log \pi _ { \pmb \theta } ( y _ { i } \mid x )$ , with $A _ { i } = ( r ^ { ( i ) } - \bar { r } ) / { \sigma _ { r } } ,$ , and write $\begin{array} { r } { \bar { \pmb g } = G ^ { - 1 } \sum _ { i } \pmb g ^ { ( i ) } } \end{array}$ . For $\bar { \pmb { g } } \neq 0 ,$ , define

$$
\Delta ( \pmb \theta ) = \frac { G ^ { - 1 } \sum _ { i } \| \pmb g ^ { ( i ) } \| ^ { 2 } } { \| \pmb \bar { g } \| ^ { 2 } } \ge 1 , \qquad B _ { S } ( \pmb \theta ) = N _ { \mathrm { m a x } } \Delta ( \pmb \theta ) .\tag{16}
$$

The inequality is Jensen’s inequality; equality holds when all gradients coincide, while equal-norm orthogonal gradients give $\Delta \stackrel { \cdot } { = } G .$ . Large $\Delta$ measures cancellation relative to the individual gradient energy. Here $\bar { N _ { \mathrm { m a x } } }$ is the rollout budget per prompt, and $B _ { S }$ is the exploration-boundary index.

Definition 3 (Gradient Folding and Cancellation Rate). With binary rewards, positive- and negativeadvantage trajectories can oppose one another at shared token positions; we call this gradient folding. Its cancellation rate and surviving gradient magnitude satisfy

$$
\gamma = 1 - \Delta ^ { - 1 } = 1 - \frac { \| \sum _ { i } g ^ { ( i ) } \| ^ { 2 } } { G \sum _ { i } \| g ^ { ( i ) } \| ^ { 2 } } , \qquad \| \bar { g } \| = \sqrt { 1 - \gamma } \left( G ^ { - 1 } \sum _ { i } \| g ^ { ( i ) } \| ^ { 2 } \right) ^ { 1 / 2 } .\tag{17}
$$

Thus $\gamma \in [ 0 , 1 )$ measures the fraction of mean squared gradient magnitude canceled. Complete cancellation gives $\gamma = 1$ and $\Delta = + \infty$ , provided the individual gradients are not all zero.

Definition 4 (Token-Level NTK). Let $\mathbf { J } ^ { n } = \nabla _ { \pmb { \theta } } z ^ { n } \in \mathbb { R } ^ { d _ { \mathrm { p a r } } \times | \mathcal { V } | }$ . The vocabulary-space kernel is $\mathbf { K } ( s , t ) = ( \mathbf { J } ^ { s } ) ^ { \top } \mathbf { J } ^ { t }$ , and its sampled-token contraction is

$$
\begin{array} { r } { K _ { t } ( \tau , s , t ) = \langle \nabla _ { \theta } \log \pi _ { \theta } ( a _ { s } \mid \tau _ { < s } ) , \nabla _ { \theta } \log \pi _ { \theta } ( a _ { t } \mid \tau _ { < t } ) \rangle . } \end{array}\tag{18}
$$

For an update from position n alone, $\dot { z } ^ { n } = - \mathbf { K } ( n , n ) \delta ^ { n } $ ; a summed update gives $\begin{array} { r l } { { \dot { z } } ^ { n } } & { { } = } \end{array}$ $- \textstyle \sum _ { m } \mathbf { K } ( n , m ) \bar { \delta } ^ { m }$ . These kernels separate local effects from interactions through shared parameters and provide the geometryfor position-level masking.

Definition 5 (Reward–Teacher Compatibility Boundary). For $K _ { D R } ( n ) = \langle \mathbf { J } ^ { n } \delta _ { D } ^ { n } , \mathbf { J } ^ { n } \delta _ { R } ^ { n } \rangle$ , define

$$
\partial \mathcal { B } = \mathcal { B } ^ { 0 } = \{ n : K _ { D R } ( n ) = 0 \} , \qquad \mathcal { B } ^ { \pm } = \{ n : \pm K _ { D R } ( n ) > 0 \} .\tag{19}
$$

The compatible region $B ^ { + }$ contributes positively to the local reward projection, the orthogonal region $B ^ { \mathrm { { \acute { 0 } } } }$ contributes zero, and the conflicting region $B ^ { - }$ contributes negatively. M3-Select admits the teacher on $B ^ { + } \cup B ^ { 0 }$ and masks $B ^ { - }$

## D EXTENDED THEORETICAL ANALYSIS

This appendix contains extended theoretical results referenced in Sections 3.4 and 3.5.

Corollary 1 (One-Step Logits Degradation). For advantage A $\neq 0$ , write $\begin{array} { r } { \pmb { g } _ { R } ^ { n } = \mathbf { J } ^ { n } \delta _ { R } ^ { n } } \end{array}$ and $\begin{array} { r } { g _ { D } ^ { n } = } \end{array}$ $\mathbf { J } ^ { n } \delta _ { D } ^ { n }$ . Thefirst-order contribution ofposition n’s hybrid update to its sampled-token log-probability is

$$
\begin{array} { r l r } & { } & { \Delta _ { n } \log p _ { S } ( \hat { y } _ { n } \mid \hat { y } _ { < n } ) \approx \left. - \frac { g _ { R } ^ { n } } { A } , - \eta [ ( 1 - \alpha ) { \pmb g } _ { R } ^ { n } + \alpha { \pmb g } _ { D } ^ { n } ] \right. } \\ & { } & { = \displaystyle \frac { \eta } { A } \left[ ( 1 - \alpha ) K _ { R R } ( n ) + \alpha K _ { D R } ( n ) \right] , } \end{array}\tag{20}
$$

where $K _ { R R } ( n ) = \| \pmb { g } _ { R } ^ { n } \| ^ { 2 }$ and $\nabla _ { \pmb \theta } \log p _ { S } ( \hat { y } _ { n } \mid \hat { y } _ { < n } ) = - \pmb g _ { R } ^ { n } / A .$ . At a conflicting position, distillation lowers the sampled-token log-probability when $A > 0$ and raises it when $A \ < \ 0$ . For $A > 0$ the total diagonal contribution becomes negative exactly when $\alpha | K _ { D R } ( n ) | > ( 1 - \alpha ) K _ { R R } ( n )$ ; Section ?? measures this effect.

Exact κ-growth identity. For nonzero gradients, direct logarithmic differentiation gives

$$
\frac { d \log \kappa } { d t } = \frac { d \log \left. \pmb { g } _ { R } \right. } { d t } - \frac { d \log \left. \pmb { g } _ { D } \right. } { d t } .\tag{21}
$$

Thus κ grows whenever the reward norm decays more slowly than the teacher norm. If the difference of these logarithmic rates is the positive constant $\lambda ,$ then $\kappa ( t ) = \kappa ( 0 ) e ^ { \lambda t }$

Corollary 2 (Aggregate Synergy Condition). Let $\Phi _ { \tau } = \cos ( g _ { D } ( \tau ) , g _ { R } ( \tau ) )$ be the trajectory-level alignment, with fixed conditional means $\mu _ { 0 } = \mathbb { E } [ \Phi _ { \tau } \mid r = 0 ] > 0$ and $\mu _ { 1 } = \mathbb { E } [ \Phi _ { \tau } \mid r = 1 ] < 0 .$ . At accuracy p,

$$
\mathbb { E } [ \Phi _ { \tau } ] = ( 1 - p ) \mu _ { 0 } + p \mu _ { 1 } > 0 \quad \Longleftrightarrow \quad p < p ^ { * } : = \frac { \mu _ { 0 } } { \mu _ { 0 } - \mu _ { 1 } } .\tag{22}
$$

Corollary 3 (Diminishing Synergy Under Increasing Accuracy). Under thefixed-conditional-mean model ofCorollary $\ L _ { * } \| \mathbb { E } [ \Phi _ { \tau } ] / d p = \mu _ { 1 } - \mu _ { 0 } < 0$ , so expected trajectory alignment crosses zero at $p ^ { * }$ and approaches $\mu _ { 1 }$ as $p  1 .$ . The threshold is determined by the two conditional alignments.

Remark 2 (Why NTK, not just cosine?). The aggregate cosine summarizes the same parameterspace inner product represented by the NTK: $\begin{array} { r } { \langle \tilde { { \bf g } } _ { D } , \tilde { { \bf g } } _ { R } \rangle = \sum _ { n , m } ( \delta _ { D } ^ { n } ) ^ { \top } { \bf K } ( n , m ) \delta _ { R } ^ { m } } \end{array}$ for summed gradients. The kernel decomposition exposes which positions and cross-position interactions produce that scalar, enabling local gating that an aggregate cosine alone cannot specify.

The measured threshold separates cells in which hard masking helps from cells in which a softer gate preserves more teacher signal. The local score identifies the teacher’s immediate reward projection; gradient magnitude, estimation noise, and cross-position interactions determine how this local decision translates into training progress. The experiments in Sections 5.3 and 5.3 compare these regimes.

## D.1 CONFLICT AS INFORMATION DESTRUCTION

At conflicting positions of a positive-advantage trajectory, the teacher contribution lowers the sampled-token log-probability (Corollary 1). Repeated contributions of this sign can erode rewarded behavior. The 500-step experiments (Section 5.3) show collapse under uniform mixing in high-κ cells, while M3-Select removes these negative local teacher projections before the update. This links the local mechanism to the observed training trajectories.

## E EXTENDED METHOD THEORY

This appendix presents the full theoretical analysis of M3 and M3-Select. All formal statements and proof sketches summarized in Section 4 are collected here.

## E.1 M3-EG: FAST–SLOW EXTRAGRADIENT UPDATE

The extragradient variant of Section 4 keeps the boundary gate $\alpha _ { n } ^ { * }$ of Eq. (12) but replaces the synchronous mixture with a two-timescale schedule. From the anchor $\theta ,$ the inner fast step applies only the boundary-gated teacher to reach a look-ahead point

$$
\begin{array} { r } { \tilde { \pmb { \theta } } = \pmb { \theta } - \eta _ { \mathrm { i n } } \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \mathbf { J } ^ { n } \alpha _ { n } ^ { * } \hat { \delta } _ { D } ^ { n } . } \end{array}\tag{23}
$$

The on-policy reward group is then scored at ${ \tilde { \theta } } ,$ but its gradient is applied as an outer correction anchored at the original θ,

$$
\begin{array} { r } { \pmb { \theta } ^ { + } = \pmb { \theta } - \eta _ { \mathrm { o u t } } \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \mathbf { J } ^ { n } \hat { \pmb { \delta } } _ { R } ^ { n } \big | _ { \tilde { \pmb { \theta } } } , } \end{array}\tag{24}
$$

using a first-order approximation that does not backpropagate through the inner step: in practice one caches the slow-gradient direction $\begin{array} { r } { \frac { 1 } { N } \sum _ { n } \mathbf { J } ^ { n } \hat { \delta } _ { R } ^ { n } | _ { \tilde { \theta } } } \end{array}$ , restores $\theta ,$ and then applies Eq. (24). The step sizes $( \eta _ { \mathrm { i n } } , \eta _ { \mathrm { o u t } } )$ play the roles of the inner (look-ahead) and outer (correction) rates of a standard extragradient scheme. Algorithm 1 states the full procedure.

Algorithm 1 M3-EG: Fast–Slow Extragradient Update   
Input: gate ceiling $\alpha _ { \mathrm { m a x } } .$ , inner (look-ahead) rate $\eta _ { \mathrm { i n } } .$ , outer (correction) rate $\eta _ { \mathrm { o u t } }$ , smoothing ϵ   
1: For step $t = 1 , \dots , T$ do   
2: Compute per-position residuals $\delta _ { D } ^ { n } , \delta _ { R } ^ { n }$ , Jacobians $\mathbf { J } ^ { n }$ , and scores $K _ { D R } ( n ) \gets \langle \mathbf { J } ^ { n } \delta _ { D } ^ { n } , \mathbf { J } ^ { n } \delta _ { R } ^ { n } \rangle$   
3: Gate $\alpha _ { n } ^ { * }  \alpha _ { \mathrm { m a x } } \mathbf { 1 } \{ K _ { D R } ( n ) \geq 0 \}$ (hard) or $\alpha _ { \mathrm { m a x } } \sigma ( \beta \mathrm { c o s } \varphi _ { n } )$ (soft); normalize $\hat { \delta } _ { D } ^ { n } , \hat { \delta } _ { R } ^ { n }$   
4: Fast / look-ahead (teacher only): $\begin{array} { r } { \tilde { \pmb { \theta } }  \pmb { \theta } _ { t } - \eta _ { \mathrm { i n } } \frac { 1 } { N } \sum _ { n } \mathbf { J } ^ { n } \alpha _ { n } ^ { * } \hat { \delta } _ { D } ^ { n } } \end{array}$ ▷ hard gate masks Ω<sub>−</sub>   
5: Re-score at $\tilde { \theta }$ (reward only): $\begin{array} { r } { h _ { R } ( \tilde { \pmb { \theta } } )  \frac { 1 } { N } \sum _ { n } \mathbf { J } ^ { n } \hat { \pmb { \delta } } _ { R } ^ { n } | _ { \tilde { \pmb { \theta } } } } \end{array}$ (no backprop through $\tilde { \theta } ) .$   
6: Slow / correction (anchored at $\pmb \theta _ { t } )$ ): $\theta _ { t + 1 } \gets \theta _ { t } - \eta _ { \mathrm { o u t } } h _ { R } ( \tilde { \theta } )$   
End for

Conflict isolation. Write $\begin{array} { r } { { h } _ { R } ( \pmb { \theta } ) = N ^ { - 1 } \sum _ { n } \mathbf { J } ^ { n } \hat { \delta } _ { R } ^ { n } } \end{array}$ for the normalized reward field and $\mathbf { \nabla } m _ { D } = $ $\begin{array} { r } { N ^ { - 1 } \sum _ { n } \mathbf { J } ^ { n } \alpha _ { n } ^ { * } \hat { \pmb \delta } _ { D } ^ { n } } \end{array}$ for the gated teacher field. The executed move is $- \eta _ { \mathrm { o u t } } { h _ { R } } ( \pmb \theta - \eta _ { \mathrm { i n } } \pmb { m } _ { D } )$ , so teacher information enters through the look-ahead location. Positive rescaling of an unsmoothed teacher residual leaves $\mathbf { \nabla } m _ { D }$ unchanged.

Proposition 1 (Local Reward Descent of the Extragradient Field). Let ${ \pmb g } _ { R } = \nabla \mathcal L _ { R } ( { \pmb \theta } )$ , assume $a _ { R } \overset { - } { = } \langle h _ { R } ( \pmb \theta ) , g _ { R } \rangle > 0 ,$ , and let $h _ { R }$ be $L _ { h } – L$ ipschitz along the inner step. Then

$$
\begin{array} { r l } & { \langle \pmb { \theta } - \pmb { \theta } ^ { + } , \pmb { g } _ { R } \rangle = \eta _ { \mathrm { o u t } } \langle \pmb { h } _ { R } ( \pmb { \theta } - \eta _ { \mathrm { i n } } \pmb { m } _ { D } ) , \pmb { g } _ { R } \rangle } \\ & { \qquad \ge \eta _ { \mathrm { o u t } } \left( a _ { R } - \eta _ { \mathrm { i n } } L _ { h } \lVert \pmb { m } _ { D } \rVert \lVert \pmb { g } _ { R } \rVert \right) > 0 } \end{array}\tag{25}
$$

whenever $\eta _ { \mathrm { i n } } L _ { h } \| m _ { D } \| \| g _ { R } \| < a _ { R }$ . For sufficiently small outer step $\eta _ { \mathrm { o u t } }$ , this is a reward-descent step. When $h _ { R } = g _ { R } ,$ its first-order expansion is $\begin{array} { r } { \eta _ { \mathrm { o u t } } [ \| g _ { R } \| ^ { 2 } - \eta _ { \mathrm { i n } } \langle \dot { \mathbf { H } } _ { R } m _ { D } , \pmb { g } _ { R } \rangle ] + O ( \eta _ { \mathrm { o u t } } \eta _ { \mathrm { i n } } ^ { 2 } ) , } \end{array}$ with ${ \bf H } _ { R } = \nabla ^ { 2 } \mathcal { L } _ { R } ( \pmb { \theta } )$

The inner normalization controls the teacher field’s scale, while the hard gate removes its negative local reward projections. The outer descent condition above quantifies the additional effect of moving the reward evaluation point (empirical comparison: Table 1).

Algorithm 2 M3-Select: Boundary-Guided Token-Level Mixing   
Input: $\alpha _ { \mathrm { m a x } } .$ step size η, smoothing ϵ, EMA update scale s¯   
1: For step $t = 1 , \dots , T$ do   
2: Compute per-position residuals $\delta _ { D } ^ { n } , \delta _ { R } ^ { n }$ and Jacobians $\mathbf { J } ^ { n } .$   
3: Compute compatibility scores $K _ { D R } ( n ) \gets \langle \mathbf { J } ^ { n } \delta _ { D } ^ { n } , \mathbf { J } ^ { n } \delta _ { R } ^ { n } \rangle$   
4: Set $\alpha _ { n }  \alpha _ { \mathrm { m a x } } \mathbf { 1 } \{ K _ { D R } ( n ) \geq 0 \}$ and record $C _ { \mathrm { N T K } } .$   
5: Normalize residuals $\hat { \delta } _ { D } ^ { n }  \delta _ { D } ^ { n } / ( \lVert \delta _ { D } ^ { n } \rVert + \epsilon )$ and $\hat { \delta } _ { R } ^ { n }  \delta _ { R } ^ { n } / ( \| \delta _ { R } ^ { n } \| + \epsilon )$   
6: Update $\begin{array} { r } { \pmb { \theta } _ { t + 1 }  \pmb { \theta } _ { t } - \eta \bar { s } \frac { 1 } { N } \sum _ { n } \mathbf { J } ^ { n } [ \alpha _ { n } \hat { \pmb { \delta } } _ { D } ^ { n } + ( 1 - \alpha _ { n } ) \hat { \pmb { \delta } } _ { R } ^ { n } ] } \end{array}$   
End for

Quadratic sub-optimality of a fixed mixing coefficient. Let $\pmb { d } = \pmb { g } _ { D } - \pmb { g } _ { R }$ and let $\alpha ^ { * } \in [ 0 , 1 ]$ minimize $\| g _ { H } ( \alpha { \bar { ) } } \| ^ { 2 }$ . Expanding around the constrained minimizer gives

$$
\begin{array} { r l } & { \| g _ { H } ( \alpha ) \| ^ { 2 } - \| g _ { H } ( \alpha ^ { * } ) \| ^ { 2 } = \| d \| ^ { 2 } ( \alpha - \alpha ^ { * } ) ^ { 2 } + 2 ( \alpha - \alpha ^ { * } ) \langle g _ { H } ( \alpha ^ { * } ) , d \rangle } \\ & { \qquad \geq \| d \| ^ { 2 } ( \alpha - \alpha ^ { * } ) ^ { 2 } . } \end{array}\tag{26}
$$

The last term on the first line is nonnegative by constrained optimality and vanishes at an interior minimizer. The curvature is $\| \pmb { d } \| ^ { 2 } = r _ { D } ^ { - 2 } + { r _ { R } } ^ { 2 } - 2 \Phi _ { D R } r _ { D } r _ { R }$ . For a random vector g, we use the total variance

$$
\operatorname { V a r } ( { \pmb g } ) : = \mathbb { E } \| { \pmb g } - \mathbb { E } { \pmb g } \| ^ { 2 } = \operatorname { t r } \operatorname { C o v } ( { \pmb g } ) ,
$$

with $\operatorname { V a r } _ { t }$ denoting conditioning on the history available before the current gradient sample.

Theorem 1 (Boundary-Gated Reward Convergence). Assume $\mathcal { L } _ { R }$ is L-smooth and bounded $b e \mathrm { . }$ low. In the conditional model of Appendix L, each position has equal-norm reward and deterministic teacher means ${ \boldsymbol { \mathbf { } } } u _ { n } , d _ { n } ,$ , position subspaces are orthogonal, gates are fixed before fresh reward noise is sampled, and $\begin{array} { r } { { \dot { \nabla } } \mathcal { L } _ { R } ~ = ~ N ^ { - 1 } \sum _ { n } { \pmb { u } } _ { n } } \end{array}$ . Define $c _ { n } ~ = ~ \langle { \bf u } _ { n } , { \bf d } _ { n } \rangle / \| { \bf u } _ { n } \| ^ { 2 }$ and $\begin{array} { r } { { \bf \Delta } w _ { n } = \| { \bf u } _ { n } \| ^ { 2 } / \sum _ { m } \| { \bf u } _ { m } \| ^ { 2 } , } \end{array}$ , taking $c _ { n } = 0$ at zero-norm positions. The population-sign gate is $\alpha _ { n } = \alpha _ { \mathrm { m a x } } \mathbf { 1 } \{ c _ { n } \geq 0 \}$ , and $\begin{array} { r } { \rho _ { R , t } = \sum _ { n } w _ { n } [ ( 1 - \alpha _ { n } ) + \alpha _ { n } c _ { n } ] } \end{array}$ . For $\alpha _ { \mathrm { m a x } } < 1$ , choose deterministic bounds $0 < \underline { { \rho } } _ { R } \le \rho _ { R , t }$ and $\mathrm { V a r } _ { t } ( \hat { { \pmb g } } _ { H } ^ { \mathrm { s e l } } ) \le v _ { \alpha } \sigma _ { R } ^ { 2 }$ valid at every iteration and history; $\underline { { \rho } } _ { R } = 1 - \alpha _ { \mathrm { m a x } }$ is admissible. $\dot { I } f \eta \leq \underline { { \rho } } _ { R } / L$ , then

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| \nabla \mathcal { L } _ { R } ( \pmb { \theta } _ { t } ) \| ^ { 2 } \leq \frac { 1 } { \underline { { \rho } } _ { R } } \left[ \frac { 2 ( \mathcal { L } _ { R } ( \pmb { \theta } _ { 0 } ) - \mathcal { L } _ { R } ^ { * } ) } { \eta T } + L \eta v _ { \alpha } \sigma _ { R } ^ { 2 } \right] .\tag{27}
$$

Proposition 2 (Conditional Variance of Boundary-Gated Mixing). In the preceding conditional model, let $\nu _ { n } = \mathbb { E } _ { t } \Vert \pmb { \xi } _ { n } \Vert ^ { 2 }$ be the fresh reward-noise variance at position n. Then

$$
\mathrm { V a r } _ { t } ( g _ { H } ^ { \mathrm { s e l } } ) = \frac { 1 } { N ^ { 2 } } \sum _ { n } ( 1 - \alpha _ { n } ) ^ { 2 } \nu _ { n } = v _ { t } \sigma _ { R , t } ^ { 2 } ,\tag{28}
$$

where $\begin{array} { r } { \sigma _ { R , t } ^ { 2 } = N ^ { - 2 } \sum _ { n } \nu _ { n } } \end{array}$ and $\begin{array} { r } { v _ { t } = \sum _ { n } ( 1 - \alpha _ { n } ) ^ { 2 } \nu _ { n } / \sum _ { n } \nu _ { n } \leq 1 } \end{array}$ when noise is nonzero; take $\begin{array} { r } { v _ { t } \ = \ N ^ { - 1 } \sum _ { n } ( 1 - \alpha _ { n } ) ^ { 2 } } \end{array}$ when all $\nu _ { n } ~ = ~ 0 .$ . Choose a deterministic $v _ { \alpha } \ \geq \ v _ { t }$ valid uniformly over iterations and histories; $v _ { \alpha } = 1$ is always admissible when $\sigma _ { R , t } ^ { 2 } \leq \sigma _ { R } ^ { 2 }$ . For equal position variances and admitted fraction $\rho _ { \mathrm { a d m } } = | \Omega _ { + } \cup \Omega _ { 0 } | / N , v _ { t } = 1 - ( 2 \alpha _ { \mathrm { m a x } } - \alpha _ { \mathrm { m a x } } ^ { 2 } ) \rho _ { \mathrm { a d m } } = ( 1 -$ $\bar { \alpha } ) ^ { 2 } + \alpha _ { \mathrm { m a x } } ^ { 2 } \rho _ { \mathrm { a d m } } ( 1 - \bar { \rho _ { \mathrm { a d m } } } )$ , where $\bar { \alpha } = \alpha _ { \mathrm { m a x } } \rho _ { \mathrm { a d m } } .$

The local reward-projection guarantee underlying both statements is

$$
\Pi _ { R } ( n ) \geq ( 1 - \alpha _ { \operatorname* { m a x } } ) \| \mathbf { J } ^ { n } \hat { \pmb { \delta } } _ { R } ^ { n } \| ^ { 2 } \mathrm { ~ f o r ~ } n \in \Omega _ { + } \cup \Omega _ { 0 } , \qquad \Pi _ { R } ( n ) = \| \mathbf { J } ^ { n } \hat { \pmb { \delta } } _ { R } ^ { n } \| ^ { 2 } \mathrm { ~ f o r ~ } n \in \Omega _ { - } .\tag{29}
$$

Theorem 2 (M3-Norm Convergence Guarantee). Assume $\mathcal { L } _ { R }$ is L-smooth and bounded below, with an unbiased reward estimator of conditional variance at most $\sigma _ { R } ^ { 2 } .$ . Conditional on each iterate, let the teacher direction be deterministic, matched to the population reward-gradient norm, and have alignment at least ϕ<sub>∗</sub> (Appendix J). For fixed $\alpha \in ( 0 , 1 )$ , put $c = 1 - \alpha + \alpha \phi _ { * } > 0 . \ I f \eta \leq c / L ,$ , then

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| \nabla \mathcal { L } _ { R } ( \theta _ { t } ) \| ^ { 2 } \leq \frac { 1 } { c } \left[ \frac { 2 ( \mathcal { L } _ { R } ^ { 0 } - \mathcal { L } _ { R } ^ { * } ) } { \eta T } + L \eta ( 1 - \alpha ) ^ { 2 } \sigma _ { R } ^ { 2 } \right] .\tag{30}
$$

Proposition 3 (Variance Reduction via Distillation). Under the conditional deterministic-teacher model ofTheorem 2, Var $\begin{array} { r } { { \ u _ { \mathrm { : } } } ( \hat { { \bf g } } _ { H } ) = ( 1 - \alpha ) ^ { 2 } \mathrm { V a r } _ { t } ( \hat { { \bf g } } _ { R } ) \le ( 1 - \alpha ) ^ { 2 } \sigma _ { R } ^ { 2 } . } \end{array}$ Proposition 4 (Optimal Per-Position Mixing). For unit parameter-space directions with cosine $c _ { n } ,$ let Π $ _ { R } ( n ; \alpha ) ~ = ~ 1 - \alpha + \alpha c _ { n }$ . Maximizing the calibrated objective $\Pi _ { R } ( n ; \alpha ) + \alpha = 1 + \alpha c _ { n }$ over $[ 0 , \alpha _ { \mathrm { m a x } } ]$ gives the hard gate $\alpha _ { n } ^ { * } = \alpha _ { \mathrm { m a x } } \mathbf { 1 } \{ c _ { n } \geq 0 \}$ , with ties assigned to the teacher. Under additive logistic score noise ofscale $1 / \beta ,$ , its expected allocation is $\alpha _ { \mathrm { m a x } } \sigma ( \beta c _ { n } )$ , the M3-Soft gate.

Proposition 5 (GradNorm Degeneracy under $\kappa \gg 1 )$ . At equal target training rates, impose $w _ { D } +$ $w _ { R } = 1$ and norm balance $w _ { D } \| { \pmb g } _ { D } \| = w _ { R } \| { \pmb g } _ { R } \| .$ . Then $w _ { D } ^ { * } = \kappa / ( 1 + \kappa )$ and $w _ { R } ^ { * } = 1 / ( 1 + \kappa )$ giving $w _ { R } ^ { * } \approx 0 . 0 3 \% a t \kappa = 3 , 4 0 0$ . While ${ w _ { D } / w _ { R } = \Theta ( 1 ) }$ , the teacher’s share ofweighted gradient magnitude is ${ w _ { D } / ( w _ { D } + w _ { R } \kappa ) = \Theta ( \kappa ^ { - 1 } ) }$

A natural loss-level alternative to fixed mixing is to gate α on the positive part of the alignment functional, $\Phi _ { D R } { } ^ { + } = \operatorname* { m a x } ( \Phi _ { D R } , 0 )$ , yielding the adaptive schedule

$$
\alpha _ { \mathrm { a d a p t i v e } } = \alpha _ { \mathrm { m a x } } \cdot \Phi _ { D R } + / \operatorname* { m a x } ( \Phi _ { D R } { } ^ { + } , \epsilon ) .\tag{31}
$$

When the gate uses an exponentially smoothed alignment estimate, its response to a sign change has the following delay.

Proposition 6 (Phase Delay in Adaptive Mixing). Let $\widehat { \Phi } _ { t + 1 } = \rho \widehat { \Phi } _ { t } + ( 1 - \rho ) \Phi _ { t }$ , with $0 < \rho < 1$ Ifalignment changesfrom $\Phi _ { + } > 0$ to $\Phi _ { - } < 0$ at $\cdot _ { t _ { 1 } }$ and $\Phi _ { t _ { 1 } } = \Phi _ { + } ,$ , then

$$
\widehat { \Phi } _ { t _ { 1 } + k } = \Phi _ { - } + ( \Phi _ { + } - \Phi _ { - } ) \rho ^ { k } , \qquad k _ { * } = \left\lceil \frac { \log [ ( \Phi _ { + } - \Phi _ { - } ) / | \Phi _ { - } | ] } { \log ( 1 / \rho ) } \right\rceil .\tag{32}
$$

The estimate remains positive $f o r k < k _ { * }$ , so the teacher gate stays active during that interval.

Proposition 7 (Empirical Invariance of κ under LoRA Rank). Within the measured LoRA-rank range $r \in \{ 8 , 1 6 , 3 2 , 6 4 \} , \bar { \kappa } = 3 , 4 7 6 \pm 3 0 1 ( C V = 8 . 7 \% )$ . On Qwen3-GSM8K, the corresponding scale comparison is $\kappa \approx 3 { , } 3 9 7$ at 0.6B and ≈ 3,400 at 1.7B. These are within-task observations; at rank 128 the measured ratio rises to 6,296, and ratios vary substantially across tasks (Section B.1).

Remark 3 (κ Invariance Across LoRA Ranks). A shared rank factor cancels from κ when the reward and teacher gradients have the same rank-dependent norm scaling. Together with weak token correlations, this supplies the approximation developed in Appendix O. A common LoRA subspace alone does not enforce equal scaling for two different directions.

Proposition 8 (Distillation Mode Determines $\kappa ) .$ . Holding g fixed gives $\kappa _ { \mathbf { o n } } / \kappa _ { \mathbf { o f f } } = \| \pmb { g } _ { D } ^ { \mathrm { o f f } } \| / \| \pmb { g } _ { D } ^ { \mathrm { o n } } \|$ For mode $s \in \{ \mathrm { o n } , \mathrm { o f f } \}$ , let $E _ { s } = \mathcal { L } _ { D } ^ { s } -$ min $\mathcal { L } _ { D } ^ { s }$ and assume positive local curvature bounds $\lambda _ { s } ^ { - } \mathbf { I } \preceq \nabla ^ { 2 } \mathcal { L } _ { D } ^ { s } \preceq \dot { \lambda } _ { s } ^ { + } \mathbf { I }$ . Then

$$
\frac { \kappa _ { \mathbf { o n } } } { \kappa _ { \mathbf { o f f } } } \geq \sqrt { \frac { \lambda _ { \mathbf { o f f } } ^ { - } E _ { \mathrm { o f f } } } { \lambda _ { \mathbf { o n } } ^ { + } E _ { \mathrm { o n } } } } .\tag{33}
$$

A large off-policy excess KL relative to the on-policy excess therefore raises this lower bound, with the curvature ratio accountingfor the different prefix distributions.

Proposition 9 (Conflict-Free Guarantee for M3-Select). With $\widetilde { \pmb { g } } _ { D } ^ { n } = \mathbf { 1 } \{ K _ { D R } ( n ) \geq 0 \} \pmb { g } _ { D } ^ { n }$ , masking gives $\langle \widetilde { { \pmb g } } _ { D } ^ { n } , { \pmb g } _ { R } ^ { n } \rangle = \operatorname* { m a x } \{ K _ { D R } ( n ) , 0 \} \geq 0$ at every position.

Proposition 10 (Distillation as Implicit Regularization). In the matched-context isotropic quadratic model $\begin{array} { r } { \mathcal { L } _ { D } ( \pmb { \theta } _ { T } + \Delta \pmb { \theta } ) = \frac { 1 } { 2 } f _ { D } \| \Delta \pmb { \dot { \theta } } \| ^ { 2 } } \end{array}$ , with $f _ { D } > 0 ,$ , constant g , and $\Delta \theta ( 0 ) = 0 ,$ , hybrid gradient flow satisfies

$$
\Delta \theta _ { H } ( T ) = - \frac { ( 1 - \alpha ) g _ { R } } { \alpha f _ { D } } ( 1 - e ^ { - \eta \alpha f _ { D } T } ) , \qquad \| \Delta \theta _ { H } ( T ) \| \leq \frac { ( 1 - \alpha ) \| g _ { R } \| } { \alpha f _ { D } } .\tag{34}
$$

Ifdegradation is proportional to parameter deviation with a common coefficient, then $\delta _ { H } \leq \delta _ { R } / ( 1 +$ $\alpha \rho _ { \mathrm { r e g } } ) _ { \mathrm { : } }$ , where $\rho _ { \mathrm { r e g } } = \eta f _ { D } T / 2 .$ . A finite rank-dependent reversal follows in the model when the pure-RL degradation grows continuously without bound, the drowning penalty is bounded, $\rho _ { \mathrm { r e g } } \ i s$ bounded away from zero, and the initial reward gap is positive; Appendix AD gives this conditional argument and the observed reversal at rank 128.

## F GRADIENT FOLDING, FIVE BRIDGES, AND UNIFIED FRAMEWORK: FULL STATEMENTS

This section collects the formal statements of the propositions, theorems, and corollaries whose proofs appear in subsequent appendix sections and whose summaries appear in Section 3.5 and the method discussion.

Proposition 11 (Per-Position Conflict Decomposition). The per-position residual inner product has the exact decomposition

$$
( \delta _ { D } ^ { n } ) ^ { \top } \delta _ { R } ^ { n } = - A ( \sigma _ { n } + \eta _ { n } ) ,\tag{35}
$$

where $\sigma _ { n } = p _ { S } ^ { n } ( \hat { y } _ { n } ) - p _ { T } ^ { n } ( \hat { y } _ { n } )$ measures student-teacher disagreement on the sampled token, and $\eta _ { n } = ( \pmb { p } _ { T } ^ { n } ) ^ { \top } \pmb { p } _ { S } ^ { n } - \| \pmb { p } _ { S } ^ { n } \| ^ { 2 }$ captures cross-token probability redistribution. Under $\mathbf { K } ( n , n ) = \lambda _ { n } \mathbf { I }$ with $\lambda _ { n } > 0 ,$ , conflict $\mathsf { \Gamma } ( K _ { D R } ( n ) < 0 )$ on positive-advantage trajectories $( A > 0 )$ occurs exactly when $\sigma _ { n } + \eta _ { n } > 0$

Proposition 12 (Asymmetric Harm from Magnitude Drowning). For ${ \pmb g } _ { H } = ( 1 - \alpha ) { \pmb g } _ { R } + \alpha { \pmb g } _ { D }$ with $0 < \alpha < 1$ and $\Phi _ { D R } < 0 ,$ , dividing the harmful cross-terms by their respective self-progress terms gives

$$
H _ { R  D } = \frac { ( 1 - \alpha ) | \langle g _ { D } , g _ { R } \rangle | } { \alpha \| g _ { D } \| ^ { 2 } } = \frac { ( 1 - \alpha ) \kappa | \Phi _ { D R } | } { \alpha } , \quad H _ { D  R } = \frac { \alpha | \Phi _ { D R } | } { ( 1 - \alpha ) \kappa } .\tag{36}
$$

Hence $H _ { R  D } / H _ { D  R } = [ ( 1 - \alpha ) / \alpha ] ^ { 2 } \kappa ^ { 2 } . A t \alpha = 1 / 2$ and $\kappa = 3 { , } 4 0 0$ , the ratio is about $1 . 1 6 \times 1 0 ^ { 7 }$ $( E q . \ 6 )$

Proposition 13 (Cosine Inversion in On-Policy OPSD). For binary rewards with baseline $p \in ( 0 , 1 )$ write $\begin{array} { r } { { \pmb g } _ { R } = - ( r - p ) { \pmb s } , } \end{array}$ , where $\pmb { s } = \nabla _ { \pmb { \theta } } \log \pi _ { \pmb { \theta } } ( \hat { y } )$ , and let $q = \langle s , \pmb { g } _ { D } \rangle / ( \lVert s \rVert \lVert \pmb { g } _ { D } \rVert )$ $I f \operatorname { \mathbb { E } } [ q \mid r =$ $0 ] + \mathbb { E } [ q \mid r = 1 ] > 0 ,$ , then

$$
\begin{array} { r } { \mathbb { E } [ \cos ( g _ { R } , g _ { D } ) \mid r = 0 ] - \mathbb { E } [ \cos ( g _ { R } , g _ { D } ) \mid r = 1 ] = \mathbb { E } [ q \mid r = 0 ] + \mathbb { E } [ q \mid r = 1 ] > 0 . } \end{array}\tag{37}
$$

Thus the conditional score–teacher alignment determines whether the teacher aligns more strongly with reward updates on incorrect trajectories.

Proposition 14 (Folding–Drowning Coupling). Let γ be the cancellation rate and $\kappa _ { \mathrm { r a w } }$ the ratio formed from the root-mean-square trajectory gradient. Then $\kappa _ { \mathrm { e f f } } = \sqrt { 1 - \gamma } \kappa _ { \mathrm { r a w } }$ . Under the concentrated two-class model with equal class-mean score norms and overlap cosine c<sub>overlap</sub>,

$$
\gamma = 1 - 2 p ( 1 - p ) ( 1 - c _ { \mathrm { o v e r l a p } } ) .\tag{38}
$$

The overlap contribution $2 p ( 1 - p ) c _ { \mathrm { o v e r l a p } }$ peaks at balanced accuracy when $c _ { \mathrm { o v e r l a p } } > 0$ . Reduced effective magnitude and positive cross-signal alignment jointly favor hybridization whenever both conditions hold.

Proposition 15 (Distillation Boundary Bound). Let $\Delta _ { D R } = ( \| { \pmb g } _ { D } \| ^ { 2 } + \| { \pmb g } _ { R } \| ^ { 2 } ) / \| { \pmb g } _ { D } + { \pmb g } _ { R } \| ^ { 2 }$ and $\Delta _ { \tau } = ( 1 - \gamma ) ^ { - 1 } . \ : I f \kappa _ { \mathrm { r a w } } \gg \sqrt { \Delta _ { \tau } } ,$ , then

$$
\Delta _ { D R } \leq 1 + \frac { 2 | \Phi _ { D R } | \sqrt { \Delta _ { \tau } } } { \kappa _ { \mathrm { r a w } } } + O \biggl ( \frac { \Delta _ { \tau } } { \kappa _ { \mathrm { r a w } } ^ { 2 } } \biggr ) .\tag{39}
$$

The index $B _ { D } = \alpha _ { \mathrm { m a x } } \Delta _ { D R }$ is the cross-signal analogue of $B _ { S }$ . Its reference value $\Delta _ { D R } = 1$ separates negativefrom positive gradient cross-terms.

Proposition 16 (NTK-Guided Token Masking (Bridge 4)). For ${ \pmb u } _ { n } = { \bf J } ^ { n } \hat { \delta } _ { D } ^ { n } , { \pmb v } _ { n } = { \bf J } ^ { n } \hat { \delta } _ { R } ^ { n } ,$ and $S ^ { - } \overset { \textstyle - } { = } \{ n : \langle u _ { n } , v _ { n } \rangle < 0 \}$ }, the average local reward-projection gain over uniform M3-Norm is

$$
\Delta \Pi _ { R } = \frac { \alpha _ { \mathrm { m a x } } } { N } \sum _ { n \in S ^ { - } } \left( \lVert \pmb { v } _ { n } \rVert ^ { 2 } - \langle \pmb { u } _ { n } , \pmb { v } _ { n } \rangle \right) .\tag{40}
$$

It is positive when $S ^ { - } \ne \emptyset$ and $\alpha _ { \mathrm { m a x } } > 0$ . For unit parameter-space directions this reduces to $\Delta \Pi _ { R } = \alpha _ { \mathrm { m a x } } C _ { \mathrm { N T K } } ( 1 + \left| \bar { c } ^ { - } \right| )$ , where |c¯<sup>−</sup>| is the mean absolute cosine on $S ^ { - }$

Proposition 17 (RL Projection Under M3-Norm). For unit directions, bilinearity gives $\Pi _ { R } ( \alpha ) =$ $\langle \alpha \hat { g } _ { D } + ( 1 - \alpha ) \hat { g } _ { R } , \hat { g } _ { R } \rangle = 1 - \alpha + \alpha \Phi _ { D R }$ . With the same reward-norm reference scale, normbalanced GradNorm has projection $( 1 + \Phi _ { D R } ) / ( 1 + \kappa )$ . Hence their projection ratio is $\Theta ( \kappa )$ when both cosinefactors remain positive and bounded awayfrom zero.

Theorem 3 (Unified NTK Learning Efficiency). Define the composite efficiency index by

$$
\eta ^ { \mathrm { e f f } } = \underbrace { ( 1 - \gamma ) } _ { \eta _ { \mathrm { e x p l o r e } } = 1 / \Delta _ { \tau } } \underbrace { [ 1 - \alpha + \alpha \Phi _ { D R } ] } _ { \eta _ { \mathrm { h y b r i d } } } .\tag{41}
$$

Bothfactors have NTK decompositions: cross-trajectory interactions determine $\gamma ,$ and cross-signal interactions determine $\Phi _ { D R }$ . Under a normalized hybrid step, first-order reward progress relative to the root-mean-square reward-gradient scale is instead proportional to $\sqrt { 1 - \gamma } \eta _ { \mathrm { h y b r i d } }$ . For fixed α, the composite index decreases with accuracy only when $- \gamma ^ { \prime } ( p ) \eta _ { \mathrm { h y b r i d } } + ( 1 - \gamma ) \overline { { { \alpha } } } \Phi _ { D R } ^ { \prime } ( p ) \leq 0 .$

Corollary 4 $( C _ { \mathrm { { N T K } } }$ -Based α Scheduling). The hard gate allocates the mean coefficient $\alpha _ { \mathrm { e f f } } ~ =$ $\alpha _ { \mathrm { m a x } } ( 1 - C _ { \mathrm { N T K } } )$ and admits thefraction $1 - \widetilde { C } _ { \mathrm { N T K } }$ ofabsolute cross-signal NTK mass. Under the fixed-cohort population model of Proposition $2 l , \tilde { C } _ { \mathrm { N T K } }$ increases with accuracy, so this admitted mass fraction decreases automatically. The count-based coefficient budget follows the unweighted conflict rate.

Full proofs of the above statements appear in the dedicated appendix sections below, together with the PCGrad degeneracy analysis in Appendix M. The five bridges connecting exploration boundary theory to hybrid dynamics are: (1) shared NTK geometry (Theorem 3); (2) cancellation-modulated drowning (Proposition 14); (3) the distillation boundary (Proposition 15); (4) token masking (Proposition 16); and (5) conflict-rate–driven α scheduling (Corollary 4).

## G EXTENDED DISCUSSION

This appendix collects the optimality and benefit-condition results referenced in the Conclusion, together with the practitioner’s decision tree that operationalises them.

Corollary 5 (Local Projection and Teacher Allocation under Low Accuracy). Assume the unitdirection setting ofProposition 16 and thefixed conditional alignments ofCorollary 2. For $p < p ^ { * }$ trajectory alignment is positive in expectation, and M3-Select weakly improves the average local reward projection over M3-Norm. At the same ceiling $\alpha _ { \mathrm { m a x } } \in ( 0 , 1 )$ , its mean teacher allocation relative to naive mixing’s norm-based teacher share is

$$
\frac { \alpha _ { \mathrm { m a x } } ( 1 - C _ { \mathrm { N T K } } ) } { \alpha _ { \mathrm { m a x } } / [ \alpha _ { \mathrm { m a x } } + ( 1 - \alpha _ { \mathrm { m a x } } ) \kappa ] } = ( 1 - C _ { \mathrm { N T K } } ) [ \alpha _ { \mathrm { m a x } } + ( 1 - \alpha _ { \mathrm { m a x } } ) \kappa ] .\tag{42}
$$

Thus normalization restores teacher allocation at large $\kappa ,$ and the gate retains only positions with nonnegative local compatibility.

Proof. The two projection claims follow from $\mathbb { E } [ \Phi _ { \tau } ] = ( 1 - p ) \mu _ { 0 } + p \mu _ { 1 } > 0$ and $\Pi _ { R } ^ { \mathrm { S e l e c t } } - \Pi _ { R } ^ { \mathrm { N o r m } } =$ $\alpha _ { \mathrm { m a x } } \dot { C } _ { \mathrm { N T K } } ( 1 + \bar { | } \bar { c } ^ { - } \bar { | } ) \geq 0$ . The allocation identity follows by dividing the admitted mean coefficient by naive mixing’s norm-based teacher share. □

Corollary 6 (Hybrid Benefit Condition). At the same step size $\eta \leq c / L ,$ , the M3-Norm bound of Theorem 2, with $c = 1 - \alpha + \alpha \phi _ { * } > 0 ,$ is strictly smaller than its pure-GRPO instance exactly when

$$
L \eta \sigma _ { R } ^ { 2 } ( 1 - \alpha + \phi _ { * } ) > \frac { 2 ( 1 - \phi _ { * } ) ( \mathcal { L } _ { R } ^ { 0 } - \mathcal { L } _ { R } ^ { * } ) } { \eta T } .\tag{43}
$$

For fixed $\eta ,$ positive variance, and $\phi _ { * } > - ( 1 - \alpha )$ , this condition holds for sufficiently large T.

Proof. Let $D = 2 ( \mathcal { L } _ { R } ^ { 0 } - \mathcal { L } _ { R } ^ { * } ) / ( \eta T )$ and $V = L \eta \sigma _ { R } ^ { 2 }$ . Comparing the two bounds and using $\alpha > 0$ gives the single equivalence chain

$$
\begin{array} { r l r } & { } & { \frac { D + ( 1 - \alpha ) ^ { 2 } V } { c } < D + V \Longleftrightarrow V [ c - ( 1 - \alpha ) ^ { 2 } ] > D ( 1 - c ) } \\ & { } & { \Longleftrightarrow V ( 1 - \alpha + \phi _ { * } ) > D ( 1 - \phi _ { * } ) , } \end{array}\tag{44}
$$

which is the stated condition.

Remark 4 (Practitioner’s Decision Tree). Estimate κ¯ from a 10-step probe run on the target cell, then apply the following gate-selection rule (thresholds are the ones used throughout this work; $\kappa ^ { * } = 5 { , } 0 0 0$ is the catastrophic threshold ofRemark 1).

(1) Catastrophic regime, $\bar { \kappa } > \kappa ^ { * } = 5 , 0 0 0 :$ use M3-Select (hard gate). Empirically verified on Llama-GSM8K $( \bar { \kappa } = 7 , 6 9 0 )$ and Llama-SVAMP $( \bar { \kappa } \approx 1 . 3 \times 1 0 ^ { 4 } )$ , where every uniform-mixing baseline collapses to reward $< 0 . 0 5$ while M3-Select survives (§5.3, §5.3); at these κ the softgate rescue is seed-fragile (two ofthree Llama-GSM8K replicates collapse during training, §B.1), making the hard gate the robust long-horizon choice.

(2) Stable regime, $1 , 0 0 0 < \bar { \kappa } \le \kappa ^ { * } :$ use M3-Soft with the empirically effective sharpness $\beta \approx 1 ,$ Proposition ?? describes its local variance sensitivity. Empirically verified on Qwen3-GSM8K $( \bar { \kappa } = 3 . 4 0 4 ) _ { \mathrm { : } }$ , Qwen3-SVAMP $( \bar { \kappa } \approx 5 , 0 8 0$ , sitting essentially at the threshold), and every InternLM/Qwen2.5 cell in this range. Hard masking over-prunes here (Remark 1); Soft-gate preserves the beneficial synergy tokens the estimator misclassifies.

(3) Low-κ regime, $\bar { \kappa } \leq 1 , 0 0 0 \colon$ naive uniform mixing (Hybrid, $\alpha { \approx } 0 . 5 )$ is already sufficient; specialized gating offers diminishing marginal returns. This regime is not instantiated in our sweep—the nearest cells are the ARC cells across architectures $( \breve { \kappa } \in [ 1 . 4 , 2 . 5 ] \times 1 0 ^ { 3 } )$ ), where gentle M3-Soft $( \beta \in \{ 0 . 5 , 1 \}$ , small $\alpha _ { \mathrm { m a x } } )$ still helps but the gap over Hybrid is within the seed-level standard deviation (§B.1).

(4) On-policy vs. off-policy OPSD choice. Use the measured conditional alignments to locate the synergy threshold (Corollary 2); the observed on-policy crossover is near $p ^ { * } \approx 0 . 5 6$ . Compare on- and off-policy validation curves when choosing the sampling mode.

(5) Conflict-rate override. If the observed conflict rate $C _ { \mathrm { N T K } } > 5 0 \%$ persists into training, switch from M3-Soft to M3-Select regardless of κ¯: the mass of Ω<sub>−</sub> tokens is large enough that the Soft gate’s residual bias on Ω dominates its variance reduction on $\Omega _ { + }$

Two calibration remarks. (i) The Hybrid→Soft boundary at κ¯ ≈ 1,000 is set by an empirical gatecalibration heuristic, and is more forgiving than the Soft→Select boundary at $\kappa ^ { \ast } :$ below it the Soft gate is still safe, only unnecessary. (ii) In cells where dynamic-gating M3-Soft is atparity with staticmixing Hybrid at the peak-training regime (InternLM-GSM8K is the canonical example), apply the SWA post-step to reduce late-stage checkpointfluctuations (Proposition ??, §B.3) beforefalling back to Hybrid.

## H PROOF OF PROPOSITION 18: SPECTRAL CONFLICT BOUND

Proposition 18 (Spectral Conflict Bound). Let ${ \bf K } = [ { \bf K } ( n , m ) ] _ { n . m = 1 } ^ { N } \in \mathbb { R } ^ { N | \mathcal { V } | \times N | \mathcal { V } | }$ be the global token-level NTK, and let ${ \pmb u } _ { D }$ and ${ \pmb u } _ { R }$ stack the residuals $\delta _ { D } ^ { n }$ and $\delta _ { R } ^ { n } ,$ respectively. Then

$$
\langle g _ { D } , g _ { R } \rangle \geq \frac { \lambda _ { \operatorname* { m i n } } ( { \bf K } ) \| { \bf \it u } _ { D } + { \bf \it u } _ { R } \| ^ { 2 } - \lambda _ { \operatorname* { m a x } } ( { \bf K } ) \| { \bf \it u } _ { D } - { \bf \it u } _ { R } \| ^ { 2 } } { 4 N ^ { 2 } } ,\tag{45}
$$

$$
\langle g _ { D } , g _ { R } \rangle \leq \frac { \lambda _ { \operatorname* { m a x } } ( \mathbf { K } ) \| \pmb { u } _ { D } + \pmb { u } _ { R } \| ^ { 2 } - \lambda _ { \operatorname* { m i n } } ( \mathbf { K } ) \| \pmb { u } _ { D } - \pmb { u } _ { R } \| ^ { 2 } } { 4 N ^ { 2 } } .
$$

Thus local residual alignment alone does not determine the aggregate interaction.

Proof. Stacking the Jacobians gives $\mathbf { J } = [ \mathbf { J } ^ { 1 } , \ldots , \mathbf { J } ^ { N } ]$ and $\mathbf { K } = \mathbf { J } ^ { \top } \mathbf { J } \succeq 0$ . Polarization and the Rayleigh bounds yield the single chain

$$
\begin{array} { r l } & { \langle \pmb { g } _ { D } , \pmb { g } _ { R } \rangle = N ^ { - 2 } \pmb { u } _ { D } ^ { \top } \mathbf { K } \pmb { u } _ { R } } \\ & { \qquad = \frac { \left( \pmb { u } _ { D } + \pmb { u } _ { R } \right) ^ { \top } \mathbf { K } \left( \pmb { u } _ { D } + \pmb { u } _ { R } \right) - \left( \pmb { u } _ { D } - \pmb { u } _ { R } \right) ^ { \top } \mathbf { K } \left( \pmb { u } _ { D } - \pmb { u } _ { R } \right) } { 4 N ^ { 2 } } } \\ & { \qquad \geq \frac { \lambda _ { \operatorname* { m i n } } ( \mathbf { K } ) \| \pmb { u } _ { D } + \pmb { u } _ { R } \| ^ { 2 } - \lambda _ { \operatorname* { m a x } } ( \mathbf { K } ) \| \pmb { u } _ { D } - \pmb { u } _ { R } \| ^ { 2 } } { 4 N ^ { 2 } } . } \end{array}
$$

Interchanging the upper and lower Rayleigh bounds proves the second inequality.

Geometric reading. Conflict occurs exactly when $( { \pmb u } _ { D } \ - \ { \pmb u } _ { R } ) ^ { \top } { \bf K } ( { \pmb u } _ { D } \ - \ { \pmb u } _ { R } ) > ( { \pmb u } _ { D } \ + $ $\begin{array} { r } { { \pmb u } _ { R } ) ^ { \top } { \bf K } ( { \pmb u } _ { D } + { \pmb u } _ { R } ) \colon } \end{array}$ the difference field has more kernel-weighted energy than the sum field. This depends on the residuals’ projections onto the kernel eigenspaces. If the trainable parameter dimension is below $N | \nu |$ , then rank $( \mathbf { K } ) \leq \dim ( \pmb \theta )$ implies $\mathbf { \bar { \lambda } } _ { \operatorname* { m i n } } ( \mathbf { K } ) = 0$ , reducing the lower bound to $- \lambda _ { \mathrm { m a x } } ( \mathbf { K } ) \| \pmb { u } _ { D } - \pmb { u } _ { R } \| ^ { 2 } / ( 4 N ^ { 2 } )$

## I PROOF OF PROPOSITION 11: PER-POSITION CONFLICT DECOMPOSITION

Proof. Substitution of the two residuals directly gives

$$
\begin{array} { r l } & { ( \pmb { \delta } _ { D } ^ { n } ) ^ { \top } \pmb { \delta } _ { R } ^ { n } = - A ( \pmb { p } _ { S } ^ { n } - \pmb { p } _ { T } ^ { n } ) ^ { \top } ( \pmb { e } _ { \hat { y } _ { n } } - \pmb { p } _ { S } ^ { n } ) } \\ & { \qquad = - A \big [ p _ { S } ^ { n } ( \hat { y } _ { n } ) - p _ { T } ^ { n } ( \hat { y } _ { n } ) + ( \pmb { p } _ { T } ^ { n } ) ^ { \top } \pmb { p } _ { S } ^ { n } - \| \pmb { p } _ { S } ^ { n } \| ^ { 2 } \big ] } \\ & { \qquad = - A ( \sigma _ { n } + \eta _ { n } ) . } \end{array}
$$

The residual identity is exact. Under $\mathbf { K } ( n , n ) = \lambda _ { n } \mathbf { I }$ with $\lambda _ { n } > 0 , K _ { D R } ( n )$ has the same sign, so a positive-advantage trajectory conflicts precisely when $\sigma _ { n } + \eta _ { n } > 0$ □

Interpretation. $\sigma _ { n }$ isolates disagreement on the sampled token, whereas $\begin{array} { r } { \eta _ { n } ~ = ~ \sum _ { v } [ p _ { T } ^ { n } ( v ) ~ - } \end{array}$ $p _ { S } ^ { n } ( v ) ] p _ { S } ^ { n } ( v )$ aggregates vocabulary-wide redistribution weighted by student confidence. The measured ratio $| \eta _ { n } | / | \sigma _ { n } | \approx 8 . 6$ (Section 5) identifies redistribution as the larger contribution in our pilot experiments.

## J PROOF OF THEOREM 2: M3-NORM CONVERGENCE GUARANTEE

Proof. Let $\mathbb { E } _ { t }$ condition on the history before the fresh reward-gradient sample, and write $g _ { R } ^ { t } =$ $\nabla \mathcal { L } _ { R } ( \pmb \theta ^ { t } )$ . The population-norm-matched update is $\hat { { \pmb g } } _ { H } ^ { t } = ( 1 - \bar { \alpha } ) ( { \pmb g } _ { R } ^ { t } + { \pmb \xi } ^ { t } ) ^ { * } + \alpha \tilde { { \pmb g } } _ { D } ^ { t }$ , where $\hat { \pmb { g } } _ { D } ^ { t }$ is conditionally deterministic, $\lVert \tilde { { \boldsymbol { g } } } _ { D } ^ { t } \rVert = \lVert { \boldsymbol { g } } _ { R } ^ { t } \rVert , \mathbf { \dot { \mathbb { E } } } _ { t } \pmb { \xi } ^ { t } = \mathbf { \ddot { 0 } }$ , and $\begin{array} { r } { \dot { \mathbb { E } } _ { t } \| \pmb { \xi } ^ { t } \| ^ { 2 } \leq \sigma _ { R } ^ { 2 } } \end{array}$ . At a stationary point set $\tilde { \pmb g } _ { D } ^ { t } = 0$ . For the uniform alignment lower bound $\phi _ { * }$ , put $c = 1 - \alpha + \alpha \phi _ { * } > 0$ and $\bar { \pmb { g } } _ { H } ^ { t } = \mathbb { E } _ { t } \hat { \pmb { g } } _ { H } ^ { t }$ Then

$$
\langle { \pmb g } _ { R } ^ { t } , \bar { \pmb g } _ { H } ^ { t } \rangle = ( 1 - \alpha ) \| { \pmb g } _ { R } ^ { t } \| ^ { 2 } + \alpha \langle { \pmb g } _ { R } ^ { t } , \tilde { \pmb g } _ { D } ^ { t } \rangle \geq c \| { \pmb g } _ { R } ^ { t } \| ^ { 2 } ,\tag{46}
$$

$$
\begin{array} { r } { \mathbb { E } _ { t } \Vert \hat { { \pmb g } } _ { H } ^ { t } \Vert ^ { 2 } = \Vert \bar { { \pmb g } } _ { H } ^ { t } \Vert ^ { 2 } + ( 1 - \alpha ) ^ { 2 } \mathbb { E } _ { t } \Vert { \pmb \xi } ^ { t } \Vert ^ { 2 } \leq \Vert { \pmb g } _ { R } ^ { t } \Vert ^ { 2 } + ( 1 - \alpha ) ^ { 2 } \sigma _ { R } ^ { 2 } . } \end{array}\tag{47}
$$

The last inequality uses $\lVert \bar { \pmb { g } } _ { H } ^ { t } \rVert \leq ( 1 - \alpha ) \lVert \pmb { g } _ { R } ^ { t } \rVert + \alpha \lVert \tilde { \pmb { g } } _ { D } ^ { t } \rVert = \lVert \pmb { g } _ { R } ^ { t } \rVert$ . Smoothness, $\eta \le c / L$ , and telescoping now give

$$
\begin{array} { r l } & { \mathbb { E } _ { t } \mathcal { L } _ { R } ( \pmb { \theta } ^ { t + 1 } ) \leq \mathcal { L } _ { R } ( \pmb { \theta } ^ { t } ) - \eta \langle \pmb { g } _ { R } ^ { t } , \bar { \pmb { g } } _ { H } ^ { t } \rangle + \frac { L \eta ^ { 2 } } { 2 } \mathbb { E } _ { t } \| \hat { \pmb { g } } _ { H } ^ { t } \| ^ { 2 } } \\ & { \qquad \leq \mathcal { L } _ { R } ( \pmb { \theta } ^ { t } ) - \frac { \eta c } { 2 } \| \pmb { g } _ { R } ^ { t } \| ^ { 2 } + \frac { L \eta ^ { 2 } } { 2 } ( 1 - \alpha ) ^ { 2 } \sigma _ { R } ^ { 2 } , } \end{array}
$$

$$
\begin{array} { r l } & { \displaystyle \frac { \eta c } { 2 } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| { \pmb g } _ { R } ^ { t } \| ^ { 2 } \leq \mathcal { L } _ { R } ^ { 0 } - \mathbb { E } \mathcal { L } _ { R } ( { \pmb \theta } ^ { T } ) + \frac { T L \eta ^ { 2 } } { 2 } ( 1 - \alpha ) ^ { 2 } \sigma _ { R } ^ { 2 } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \leq \mathcal { L } _ { R } ^ { 0 } - \mathcal { L } _ { R } ^ { * } + \frac { T L \eta ^ { 2 } } { 2 } ( 1 - \alpha ) ^ { 2 } \sigma _ { R } ^ { 2 } . } \end{array}
$$

Dividing by $\eta c T / 2$ proves the rate. This argument applies to the stated population-normalized update; the EMA scale in Algorithm 2 estimates its scale. □

Remark 5 (Additional properties of the M3 update). At the minimum-norm coefficient $\alpha ^ { * }$ , the convex-hull optimality condition gives $\langle \pmb { g } _ { H } ( \alpha ^ { * } ) , \pmb { \mathscr { g } } _ { i } \rangle \geq \Vert \pmb { g } _ { H } ( \alpha ^ { * } ) \Vert ^ { 2 } f o r i \in \{ D , \tilde { R } \}$ , establishing first-order Pareto descent (Sener & Koltun, 2018). By comparison, naive mixing with $\begin{array} { r } { \alpha = \frac { 1 } { 2 } } \end{array}$ increases distillation loss to first order exactly when $\Phi _ { D R } < - 1 / \kappa ,$ , since $\langle g _ { H } , g _ { D } \rangle = \textstyle { \frac { 1 } { 2 } } r _ { D } { } ^ { 2 } \bar { ( } 1 +$ κΦ ). The normalized reward projection insteadfollows Eq. 46. For an interior minimum-norm coefficient, completing the square gives Eq. 26; at $\begin{array} { r } { \alpha = \frac { 1 } { 2 } } \end{array}$ its gap is $( r _ { R } { } ^ { 2 } - r _ { D } { } ^ { 2 } ) ^ { 2 } / ( 4 | | g _ { D } - g _ { R } | | ^ { 2 } )$ . A clipped boundary minimizer also contributes the corresponding one-sided linear term.

## K PROOF OF PROPOSITION 3: VARIANCE REDUCTION VIA DISTILLATION

Proof. In the conditional model above, $\hat { \pmb { g } } _ { H } ^ { t } - \mathbb { E } _ { t } \hat { \pmb { g } } _ { H } ^ { t } = ( 1 - \alpha ) \pmb { \xi } ^ { t }$ , hence

$$
\begin{array} { r } { \mathrm { V a r } _ { t } ( \hat { \pmb { g } } _ { H } ^ { t } ) = ( 1 - \alpha ) ^ { 2 } \mathbb { E } _ { t } \Vert \pmb { \xi } ^ { t } \Vert ^ { 2 } \leq ( 1 - \alpha ) ^ { 2 } \sigma _ { R } ^ { 2 } . } \end{array}
$$

Its contribution to the convergence bound is $L \eta ( 1 - \alpha ) ^ { 2 } \sigma _ { R } ^ { 2 } / c$ . Relative to the pure-RL bound at the same admissible step size, the noise-floor factor satisfies

$$
\frac { ( 1 - \alpha ) ^ { 2 } } { c } < 1 \quad \Longleftrightarrow \quad c - ( 1 - \alpha ) ^ { 2 } = \alpha ( 1 - \alpha + \phi _ { * } ) > 0 \quad \Longleftrightarrow \quad \phi _ { * } > - ( 1 - \alpha ) .
$$

For $\phi _ { * } \geq 0$ this factor is at most $1 - \alpha$ . The variance contraction follows from the reward weight; normalization additionally controls the mean projection through c. □

## L JOINT PROOF OF THEOREM 1 AND PROPOSITION 2

We use the conditional, orthogonal-position model of the two statements. At each iteration, let $\mathbf { \Delta } \mathbf { u } _ { n }$ be the mean reward contribution and $\scriptstyle { d _ { n } }$ its conditionally deterministic teacher counterpart, with $\left\| \pmb { d } _ { n } \right\| \ = \ \left\| \pmb { u } _ { n } \right\|$ . The reward noise $\xi _ { n }$ has $\mathbb { E } _ { t } \pmb { \xi } _ { n } = 0$ and $\nu _ { n } ~ = ~ \mathbb { E } _ { t } \| \pmb { \xi } _ { n } \| ^ { 2 }$ . Contributions from distinct positions lie in mutually orthogonal parameter subspaces, as in the exact block-diagonal NTK model. The gates are fixed before this fresh noise is sampled. Suppressing t locally, define

$$
\alpha _ { n } = \alpha _ { \mathrm { m a x } } \mathbf { 1 } [ \langle u _ { n } , d _ { n } \rangle \geq 0 ] , \qquad \rho _ { \mathrm { a d m } } = \frac { | \Omega _ { + } \cup \Omega _ { 0 } | } { N } , \qquad \bar { \alpha } = \alpha _ { \mathrm { m a x } } \rho _ { \mathrm { a d m } } = \alpha _ { \mathrm { m a x } } ( 1 - C _ { \mathrm { N T K } } ) .\tag{48}
$$

Here $\begin{array} { r } { \pmb { u } = N ^ { - 1 } \sum _ { n } \pmb { u } _ { n } = \nabla \mathcal { L } _ { R } ( \pmb { \theta } ^ { t } ) } \end{array}$ and $\begin{array} { r } { \hat { \pmb g } _ { H } = N ^ { - 1 } \sum _ { n } [ ( 1 - \alpha _ { n } ) ( \pmb u _ { n } + \pmb \xi _ { n } ) + \alpha _ { n } \pmb d _ { n } ] . } \end{array}$ . For a zero-norm position set $d _ { n } = 0$ and $c _ { n } = 0 ;$ ; it has zero reward-energy weight. $\mathrm { \bf A t } { \boldsymbol { \mathbf { \mathit { u } } } } = 0$ , all mean contributions vanish and the projection bound holds directly.

Proof. The same decomposition yields both projection and variance, so it suffices to establish their constants once. With $\begin{array} { r } { \boldsymbol { w _ { n } } = \| \boldsymbol { \mathsf { u } _ { n } } \| ^ { 2 } / \sum _ { m } \| \boldsymbol { \mathsf { u } _ { m } } \| ^ { 2 } , \boldsymbol { c _ { n } } = \langle \boldsymbol { \mathsf { u } _ { n } } , \boldsymbol { d _ { n } } \rangle / \| \boldsymbol { \mathsf { u } _ { n } } \| ^ { 2 } } \end{array}$ , and $\bar { \pmb g } _ { H } = \mathbb E _ { t } \hat { \pmb g } _ { H }$ , orthogonality gives

$$
\langle { \pmb u } , { \bar { \pmb g } } _ { H } \rangle = \frac { 1 } { N ^ { 2 } } \sum _ { n } [ ( 1 - \alpha _ { n } ) + \alpha _ { n } c _ { n } ] \| { \pmb u } _ { n } \| ^ { 2 } = \rho _ { R , t } \| { \pmb u } \| ^ { 2 } , \quad \rho _ { R , t } : = \sum _ { n } w _ { n } [ ( 1 - \alpha _ { n } ) + \alpha _ { n } c _ { n } ] ,\tag{49}
$$

$$
\rho _ { R , t } \geq 1 - \alpha _ { \operatorname* { m a x } } \sum _ { n \in \Omega _ { + } \cup \Omega _ { 0 } } w _ { n } \geq 1 - \alpha _ { \operatorname* { m a x } } > 0 ,\tag{50}
$$

$$
\mathrm { V a r } _ { t } ( \hat { g } _ { H } ) = \mathbb E _ { t } \Big | \Big | \frac { 1 } { N } \sum _ { n } ( 1 - \alpha _ { n } ) \pmb { \xi } _ { n } \Big | \Big | ^ { 2 } = \frac { 1 } { N ^ { 2 } } \sum _ { n } ( 1 - \alpha _ { n } ) ^ { 2 } \nu _ { n } = v _ { t } \sigma _ { R , t } ^ { 2 } ,\tag{51}
$$

where $\sigma _ { R , t } ^ { 2 } = N ^ { - 2 } \sum _ { n } \nu _ { n }$ and $\begin{array} { r } { v _ { t } = \sum _ { n } ( 1 - \alpha _ { n } ) ^ { 2 } \nu _ { n } / \sum _ { n } \nu _ { n } \leq 1 } \end{array}$ ; use $\begin{array} { r } { v _ { t } = N ^ { - 1 } \sum _ { n } ( 1 - \alpha _ { n } ) ^ { 2 } } \end{array}$ when all $\dot { \nu } _ { n } = 0$ . Cross-position noise terms vanish by the subspace orthogonality. These identities show why the aggregate projection uses reward-energy weights and the variance uses noise-energy weights.

For equal position variances, the variance factor has the closed form

$$
v _ { t } = ( 1 - \alpha _ { \operatorname* { m a x } } ) ^ { 2 } \rho _ { \mathrm { a d m } } + 1 - \rho _ { \mathrm { a d m } } = 1 - ( 2 \alpha _ { \operatorname* { m a x } } - \alpha _ { \operatorname* { m a x } } ^ { 2 } ) \rho _ { \mathrm { a d m } } = ( 1 - { \bar { \alpha } } ) ^ { 2 } + \alpha _ { \operatorname* { m a x } } ^ { 2 } \rho _ { \mathrm { a d m } } ( 1 - \rho _ { \mathrm { a d m } } ) .\tag{52}
$$

This proves Proposition 2, including the correction due to heterogeneity of the gate.

For convergence, take uniform bounds $\underline { { \rho } } _ { R } \leq \rho _ { R , t } , v _ { t } \leq v _ { \alpha } .$ , and $\sigma _ { R , t } ^ { 2 } \ \leq \ \sigma _ { R } ^ { 2 }$ . The local triangle inequality gives $\| ( 1 - \alpha _ { n } ) \pmb { u } _ { n } + \alpha _ { n } \pmb { d } _ { n } \| \tilde { \le } \| \pmb { u } _ { n } \|$ , so orthogonality and Eq. 51 imply

$$
\begin{array} { r } { \mathbb { E } _ { t } \| \hat { \pmb { g } } _ { H } ^ { t } \| ^ { 2 } = \| \bar { \pmb { g } } _ { H } ^ { t } \| ^ { 2 } + \mathrm { V a r } _ { t } ( \hat { \pmb { g } } _ { H } ^ { t } ) \leq \| \nabla { \mathcal { L } } _ { R } ( \pmb { \theta } ^ { t } ) \| ^ { 2 } + v _ { \alpha } \sigma _ { R } ^ { 2 } . } \end{array}\tag{53}
$$

Substituting Eqs. 49 and 53 into the smoothness-and-telescoping chain in Appendix J, with c replaced by $\underline { { \rho } } _ { R }$ and $( 1 - \alpha ) ^ { 2 }$ by $v _ { \alpha }$ , gives for $\eta \leq \underline { { \rho } } _ { R } / L$

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| \nabla \mathcal { L } _ { R } ( \pmb { \theta } ^ { t } ) \| ^ { 2 } \leq \frac { 1 } { \underline { { \rho } } _ { R } } \left[ \frac { 2 ( \mathcal { L } _ { R } ^ { 0 } - \mathcal { L } _ { R } ^ { * } ) } { \eta T } + L \eta v _ { \alpha } \sigma _ { R } ^ { 2 } \right] .\tag{54}
$$

The choice $\underline { { \rho } } _ { R } = 1 - \alpha _ { \mathrm { m a x } }$ is always valid in this model.

Comparison with uniform mixing. For the same matched local directions, masking increases the energy-weighted reward projection by α<sub>max</sub> $\begin{array} { r } { \sum _ { n \in \Omega _ { - } } w _ { n } ( 1 - c _ { n } ) \ge 0 } \end{array}$ . It also leaves more reward noise than uniform mixing: $v _ { t } \geq ( 1 - \alpha _ { \operatorname* { m a x } } ) ^ { 2 }$ . The convergence comparison therefore depends on the ratio of variance to projection. Against pure RL, the variance contracts strictly whenever an admitted position carries nonzero reward noise. These are reward-descent guarantees; at a rejected position the pure reward direction can still oppose the teacher direction.

## M PCGRAD DEGENERACY IN THE ASYMMETRIC REGIME

Proposition 19 (PCGrad Magnitude Preservation). Let ${ \bf { \mathit { g } } } _ { D } , { \bf { \mathit { g } } } _ { R }$ be nonzero gradients with cosine $- 1 < \Phi < 0$ . Their symmetric PCGrad projections preserve the magnitude ratio:

$$
{ \frac { \| { \pmb g } _ { D } ^ { \perp } \| } { \| { \pmb g } _ { R } ^ { \perp } \| } } = { \frac { \| { \pmb g } _ { D } \| } { \| { \pmb g } _ { R } \| } } = { \frac { r _ { D } } { r _ { R } } } = { \frac { 1 } { \kappa } } .\tag{55}
$$

Proof. For $( i , j ) \in \{ ( D , R ) , ( R , D ) \}$ , orthogonal projection gives

$$
\| g _ { i } ^ { \perp } \| ^ { 2 } = \left\| g _ { i } - \frac { \left. g _ { i } , g _ { j } \right. } { \| g _ { j } \| ^ { 2 } } g _ { j } \right\| ^ { 2 } = \| g _ { i } \| ^ { 2 } - \frac { \langle g _ { i } , g _ { j } \rangle ^ { 2 } } { \| g _ { j } \| ^ { 2 } } = { r _ { i } } ^ { 2 } ( 1 - \Phi ^ { 2 } ) .\tag{56}
$$

Canceling the common positive factor proves the ratio. Consequently, for $0 \textless \alpha \textless 1$ , the normbased teacher share in $\alpha { \pmb g } _ { D } ^ { \perp } + ( 1 - \alpha ) { \pmb g } _ { R } ^ { \perp }$ is $w _ { D } = { \alpha } / [ { \alpha } + ( 1 - { \alpha } ) { \kappa } ] ,$ , approximately 0.03% at $\alpha = 0 . 5$ and $\kappa = 3 { , } 4 0 0$ . At exact antiparallelity both projections vanish. □

## N PROOF OF PROPOSITION 6: PHASE DELAY IN ADAPTIVE MIXING

Proof. Consider an alignment estimate with exponential smoothing $\hat { \Phi } _ { t + 1 } = \rho \hat { \Phi } _ { t } + ( 1 - \rho ) \Phi _ { t }$ $0 < \rho < 1$ . Suppose the input changes from $\Phi _ { + } > 0 \mathrm { t o } \Phi _ { - } < 0$ at $t _ { 1 }$ , with $\hat { \Phi } _ { t _ { 1 } } = \Phi _ { + }$ . Solving the recurrence and its zero-crossing condition in one chain gives

$$
\hat { \Phi } _ { t _ { 1 } + k } = \rho ^ { k } \Phi _ { + } + ( 1 - \rho ) \Phi _ { - } \sum _ { j = 0 } ^ { k - 1 } \rho ^ { j } = \Phi _ { - } + ( \Phi _ { + } - \Phi _ { - } ) \rho ^ { k } ,\tag{57}
$$

$$
\widehat { \Phi } _ { t _ { 1 } + k } \leq 0 \iff \rho ^ { k } \leq \frac { | \Phi _ { - } | } { \Phi _ { + } - \Phi _ { - } } \iff k \geq \frac { \log [ ( \Phi _ { + } - \Phi _ { - } ) / | \Phi _ { - } | ] } { \log ( 1 / \rho ) } .\tag{58}
$$

Thus the first nonpositive estimate occurs after the ceiling of the last expression. A rule that retains the high teacher weight while $\hat { \Phi } > 0$ continues doing so during this delay, although the current alignment is negative. For a boxcar average of width $W .$ , replacing k old observations gives $\hat { \Phi } = \Phi _ { + } + k ( \Phi _ { - } - \Phi _ { + } ) / W$ , so the corresponding delay is $\lceil W \Phi _ { + } / ( \Phi _ { + } - \Phi _ { - } ) \rceil ;$ ; equal transition magnitudes give approximately $W / 2$ . The result applies to loss-based schedules when their smoothed control statistic follows this assumed alignment transition. □

## O HEURISTIC DERIVATION FOR PROPOSITION 7: RANK-INVARIANCE OF κ

Heuristic derivation. Write the distillation gradient as $\begin{array} { r } { { \pmb g } _ { D } = N ^ { - 1 } \sum _ { n } { \pmb d } _ { n } } \end{array}$ . To isolate the role of the LoRA subspace, suppose both signals share a rank-dependent second-moment factor $s _ { r } \ > \ 0$ with $\mathbb { E } \| \pmb { g } _ { R } \| ^ { 2 } \overset {  } { = } a _ { R } s _ { r } , \tilde { \mathbb { E } } \| \pmb { d } _ { n } \| ^ { 2 } = \overset {  } { a _ { D } } s _ { r } ,$ and $\mathbb { E } \langle d _ { n } , d _ { m } \rangle \dot { = } 0$ for n $\neq m$ . Here $a _ { R } , a _ { D } > 0$ are rank-independent signal constants. Then

$$
\mathbb { E } \| { \pmb { g } } _ { D } \| ^ { 2 } = \frac { 1 } { N ^ { 2 } } \sum _ { n , m } \mathbb { E } \langle { \pmb { d } } _ { n } , { \pmb { d } } _ { m } \rangle = \frac { a _ { D } s _ { r } } { N } , \qquad \sqrt { \frac { \mathbb { E } \| { \pmb { g } } _ { R } \| ^ { 2 } } { \mathbb { E } \| { \pmb { g } } _ { D } \| ^ { 2 } } } = \sqrt { \frac { a _ { R } } { a _ { D } } N } .\tag{59}
$$

The common factor cancels; concentration of the squared norms transfers this root-mean-square ratio to typical observed ratios. For example, an isotropic LoRA model may give $s _ { r } \propto L r / h$ for L adapted layers of width h. The cancellation depends on shared scaling and weak cross-token correlations, which are the modeling assumptions behind this heuristic. The measured rank sweep, $r \in \{ 8 , 1 6 , 3 2 , 6 4 \}$ and $\bar { \kappa } = 3 , 4 7 6 \bar { \pm } 3 0 1 \mathrm { ( C V } 8 . 7 \% )$ ), supplies the empirical evidence in Remark 3; the signal constants determine its absolute scale. □

## P PROOF OF PROPOSITION 8: DISTILLATION MODE DETERMINES κ

Proof. For mode $s \in \{ \mathrm { o n } , \mathrm { o f f } \}$ , let $E _ { s } = \mathcal { L } _ { D } ^ { s } ( \pmb { \theta } ) - \mathcal { L } _ { D } ^ { s } ( \pmb { \theta } _ { s } ^ { * } )$ be the excess distillation loss above its local minimum. In an exact local quadratic model, $\begin{array} { r } { E _ { s } ^ { - } = \frac { 1 } { 2 } \Delta \pmb { \theta } _ { s } ^ { \top } \mathbf { H } _ { s } \Delta \pmb { \theta } _ { s } } \end{array}$ and $\pmb { g } _ { D } ^ { s } = \mathbf { H } _ { s } \Delta \pmb { \theta } _ { s }$ , where $\Delta \theta _ { s } = \theta - \theta _ { s } ^ { * }$ and $\mathbf { H } _ { s }$ is positive definite on the active parameter subspace. Writing $\lambda _ { s } ^ { - } , \lambda _ { s } ^ { + }$ for its extremal eigenvalues yields

$$
\begin{array} { r } { 2 \lambda _ { s } ^ { - } E _ { s } = \lambda _ { s } ^ { - } \Delta \theta _ { s } ^ { \top } \mathbf { H } _ { s } \Delta \theta _ { s } \leq \Delta \theta _ { s } ^ { \top } \mathbf { H } _ { s } ^ { 2 } \Delta \theta _ { s } = \| g _ { D } ^ { s } \| ^ { 2 } \leq \lambda _ { s } ^ { + } \Delta \theta _ { s } ^ { \top } \mathbf { H } _ { s } \Delta \theta _ { s } = 2 \lambda _ { s } ^ { + } E _ { s } , } \end{array}\tag{60}
$$

$$
\frac { \kappa _ { \mathbf { o n } } } { \kappa _ { \mathbf { o f f } } } = \frac { \| g _ { D } ^ { \mathrm { o f f } } \| } { \| g _ { D } ^ { \mathrm { o n } } \| } \geq \sqrt { \frac { \lambda _ { \mathrm { o f f } } ^ { - } E _ { \mathrm { o f f } } } { \lambda _ { \mathrm { o n } } ^ { + } E _ { \mathrm { o n } } } } ,\tag{61}
$$

where the second line uses the same nonzero reward gradient in both modes. The same inequality follows from local strong-convexity and smoothness bounds with the corresponding constants. If both losses have zero minimum and share a curvature matrix F, this specializes to $\sqrt { D _ { \mathrm { K L } } ^ { \mathrm { o f f } } / D _ { \mathrm { K L } } ^ { \mathrm { o n } } } / \sqrt { \chi ( \mathbf { F } ) }$

A nearly deterministic teacher on ground-truth tokens gives $D _ { \mathrm { K L } } ^ { \mathrm { o f f } } ( n ) \approx - \log p _ { S } ( y _ { n } ^ { * } )$ . On-policy evaluation changes the prefix distribution, and its KL gap must be measured. Table 11 reports mean teacher losses 0.098 on-policy and 0.810 off-policy, with median magnitude ratios 388 and 68, respectively. These observations establish the cross-mode gap in the evaluated setting; the bound explains how excess loss and curvature jointly control gradient magnitude. □

## Q PROOF OF PROPOSITION 9: CONFLICT-FREE GUARANTEE

Proof. By Definition 1, ${ \pmb g } _ { D } ^ { n } = { \bf J } ^ { n } \delta _ { D } ^ { n } , { \pmb g } _ { R } ^ { n } = { \bf J } ^ { n } \delta _ { R } ^ { n }$ , and $\langle { \pmb g } _ { D } ^ { n } , { \pmb g } _ { R } ^ { n } \rangle = K _ { D R } ( n )$ . With the gate treated as a fixed coefficient during the update,

$$
\begin{array} { r } { \big \langle \mathbf { 1 } [ K _ { D R } ( n ) \geq 0 ] \pmb { g } _ { D } ^ { n } , \pmb { g } _ { R } ^ { n } \big \rangle = \mathbf { 1 } [ K _ { D R } ( n ) \geq 0 ] K _ { D R } ( n ) = \operatorname* { m a x } \{ K _ { D R } ( n ) , 0 \} \geq 0 . } \end{array}\tag{62}
$$

Positive norm rescaling preserves this sign, proving the token-level guarantee.

For comparison, if the aggregate normalized gradients have cosine $\Phi _ { D R } \ge 0$ and $0 < \alpha < 1$ , their mixture satisfies $\left. { { g } _ { H } } , \hat { { g } } _ { R } \right. = 1 - \alpha + \alpha \Phi _ { D R } > 0$ and $\langle { \pmb g } _ { H } , \hat { \pmb g } _ { D } \rangle = \alpha + ( 1 - \alpha ) \Phi _ { D R } > 0$ . The latter projection changes sign at $\Phi _ { D R } = - \alpha / ( 1 - \alpha )$ , compared with $- \alpha / [ ( 1 - \alpha ) \kappa ]$ for unnormalized mixing. Cross-position interactions enter the aggregate condition through Proposition 20.

## R PROOF OF PROPOSITION 5: GRADNORM DEGENERACY

Proof. At a norm-balanced GradNorm equilibrium with equal target training rates and $w _ { D } + w _ { R } =$ 1, the weighted norms coincide. Consequently,

$$
w _ { D } r _ { D } = w _ { R } r _ { R } \implies ( w _ { D } ^ { * } , w _ { R } ^ { * } ) = \frac { ( \kappa , 1 ) } { 1 + \kappa } ,\tag{63}
$$

$$
g _ { H } ^ { \mathrm { G N } } = \frac { r _ { R } } { 1 + \kappa } ( \hat { \pmb { g } } _ { D } + \hat { \pmb { g } } _ { R } ) , \qquad \| \pmb { g } _ { H } ^ { \mathrm { G N } } \| \leq \frac { 2 r _ { R } } { 1 + \kappa } ,\tag{64}
$$

$$
\langle { \pmb g } _ { R } , { \pmb g } _ { H } ^ { \mathrm { G N } } \rangle = { r _ { R } } ^ { 2 } \frac { 1 + \Phi _ { D R } } { 1 + \kappa } .\tag{65}
$$

For an L-smooth reward loss, $\mathcal { L } _ { R } ( \pmb { \theta } ) - \mathcal { L } _ { R } ( \pmb { \theta } - \eta \pmb { g } _ { H } ) = \eta \langle \pmb { g } _ { R } , \pmb { g } _ { H } \rangle + O ( L \eta ^ { 2 } \| \pmb { g } _ { H } \| ^ { 2 } )$ . The first-order reward decrease is therefore $( 1 + \Phi _ { D R } ) / ( 1 + \kappa )$ of the pure-RL decrease at the same learning rate. Before equilibrium, comparable positive weights give teacher share $w _ { D } / ( w _ { D } + w _ { R ^ { \acute { \kappa } } } ) = \Theta ( { \bar { 1 / \kappa } } )$

To compare with M3-Norm at a common reward scale, use $r _ { R } [ \alpha \hat { { \pmb g } } _ { D } + ( 1 - \alpha ) \hat { { \pmb g } } _ { R } ]$ . Its reward projection relative to GradNorm is

$$
\frac { \Pi _ { R } ^ { \mathrm { M 3 } } } { \Pi _ { R } ^ { \mathrm { G N } } } = \frac { ( 1 + \kappa ) [ 1 - \alpha + \alpha \Phi _ { D R } ] } { 1 + \Phi _ { D R } } ,\tag{66}
$$

which is $\Theta ( \kappa )$ when both cosine-dependent factors stay positive and bounded away from zero. At $\alpha = 0 . 1 5 , \kappa = 4 . 3 8 1$ , and near-zero cosine, this first-order ratio is about 3,725. It describes the specified update scaling; learning-rate rescaling or a different weight optimizer changes the comparison. □

## S PROOF OF PROPOSITION 14: FOLDING–DROWNING COUPLING

Proof. Let $\begin{array} { r } { M _ { R } ^ { 2 } = G ^ { - 1 } \sum _ { i } \Vert A _ { i } \nabla _ { \theta } \log \pi ( y _ { i } \mid x ) \Vert ^ { 2 } > 0 } \end{array}$ . The definitions of cancellation and raw magnitude immediately give

$$
\kappa _ { \mathrm { e f f } } = \frac { \| g _ { R } \| } { \| g _ { D } \| } = \frac { \sqrt { ( 1 - \gamma ) M _ { R } ^ { 2 } } } { \| g _ { D } \| } = \sqrt { 1 - \gamma } \kappa _ { \mathrm { r a w } } .\tag{67}
$$

For the binary-outcome model, let $p \in \mathsf { \Gamma } ( 0 , 1 )$ be the correct fraction, $\bar { \pmb { g } } _ { \pm }$ the class-mean score gradients, and $\sigma _ { r } = { \sqrt { p ( 1 - p ) } }$ . The standardized advantages imply

$$
g _ { R } = p \frac { 1 - p } { \sigma _ { r } } \bar { g } _ { + } - ( 1 - p ) \frac { p } { \sigma _ { r } } \bar { g } _ { - } = \sqrt { p ( 1 - p ) } ( \bar { g } _ { + } - \bar { g } _ { - } ) ,\tag{68}
$$

$$
\| g _ { R } \| ^ { 2 } = p ( 1 - p ) \left( \| \bar { g } _ { + } \| ^ { 2 } + \| \bar { g } _ { - } \| ^ { 2 } - 2 \langle \bar { g } _ { + } , \bar { g } _ { - } \rangle \right) .\tag{69}
$$

If score gradients concentrate at class means with common norm s, then $M _ { R } ^ { 2 } = s ^ { 2 }$ and

$$
\gamma = 1 - 2 p ( 1 - p ) ( 1 - c _ { \mathrm { o v e r l a p } } ) = \underbrace { 1 - 2 p ( 1 - p ) } _ { \mathrm { a v e r a g i n g ~ c o n t r i b u i o n } } + \underbrace { 2 p ( 1 - p ) c _ { \mathrm { o v e r l a p } } } _ { \mathrm { c r o s s - c l a s s ~ o v e r l a p ~ c o n t i b u t i o n } } .\tag{70}
$$

For nonnegative overlap this yields $\gamma \geq 2 p ( 1 - p ) c _ { \mathrm { o v e r l a p } } ;$ the overlap contribution peaks at $p = 1 / 2$ and decreases for $p > 1 / 2$ . It is this contribution, rather than the full cancellation rate, that vanishes as $p  1$ in the model.

Combining $\gamma > 0$ with the expected-alignment condition of Corollary 2 gives reduced effective magnitude imbalance and positive expected alignment whenever that corollary’s low-accuracy condition holds. As accuracy increases beyond $\bar { 1 / 2 } .$ , the overlap contribution decreases; Corollary 3 separately describes the decline in expected alignment. These are the two quantities tracked by the accuracy-adaptive interpretation of Eq. 31. □

## T PROOF OF PROPOSITION 20: AGGREGATE CONFLICT AS A CROSS-SIGNAL NTK SUM

Proposition 20 (Aggregate Conflict as Cross-Signal NTK Sum). For nonzero aggregate gradients, their cosine is the normalized sum of diagonal cross-signal NTK terms and cross-position interactions. Dropping the latter gives the diagonal NTK approximation.

Proof. Define $\begin{array} { r } { \mathcal { E } _ { \mathrm { c r o s s } } = N ^ { - 2 } \sum _ { n \neq m } ( \delta _ { D } ^ { n } ) ^ { \top } \mathbf { K } ( n , m ) \delta _ { R } ^ { m } } \end{array}$ . Using $\mathbf { K } ( n , m ) = ( \mathbf { J } ^ { n } ) ^ { \top } \mathbf { J } ^ { m }$ and Definition 1, the decomposition follows directly:

$$
\Phi _ { D R } = \frac { 1 } { N ^ { 2 } \| \pmb { g } _ { D } \| \| \pmb { g } _ { R } \| } \sum _ { n , m } ( \pmb { \delta } _ { D } ^ { n } ) ^ { \top } \mathbf { K } ( n , m ) \pmb { \delta } _ { R } ^ { m }\tag{71}
$$

$$
= \frac { N ^ { - 2 } \sum _ { n } \langle \mathbf { J } ^ { n } \delta _ { D } ^ { n } , \mathbf { J } ^ { n } \delta _ { R } ^ { n } \rangle + \mathcal { E } _ { \mathrm { c r o s s } } } { \| g _ { D } \| \| g _ { R } \| } = \frac { N ^ { - 2 } \sum _ { n } K _ { D R } ( n ) + \mathcal { E } _ { \mathrm { c r o s s } } } { \| g _ { D } \| \| g _ { R } \| } .\tag{72}
$$

The diagonal approximation is accurate to the extent that the normalized cross-position remainder is small. □

## U PROOF OF PROPOSITION 15: DISTILLATION BOUNDARY BOUND

Proof. Let $c = \Phi _ { D R }$ and $q = \sqrt { \Delta _ { \tau } } / \kappa _ { \mathrm { r a w } } = 1 / \kappa _ { \mathrm { e f f } }$ , using $1 - \gamma = 1 / \Delta$ <sub>τ</sub> and Proposition 14. For $q  0$ , the exact diversity identity and its expansion are

$$
\Delta _ { D R } = \frac { 1 + \kappa _ { \mathrm { e f f } } ^ { 2 } } { 1 + \kappa _ { \mathrm { e f f } } ^ { 2 } + 2 c \kappa _ { \mathrm { e f f } } } = \frac { 1 + q ^ { 2 } } { 1 + 2 c q + q ^ { 2 } } = 1 - 2 c q + O ( q ^ { 2 } )\tag{73}
$$

$$
\leq 1 + \frac { 2 \vert c \vert \sqrt { \Delta _ { \tau } } } { \kappa _ { \mathrm { r a w } } } + O \left( \frac { \Delta _ { \tau } } { \kappa _ { \mathrm { r a w } } ^ { 2 } } \right) .\tag{74}
$$

The expansion is uniform for $c \in \left[ - 1 , 1 \right]$ with $q$ sufficiently small; its regime is $\kappa _ { \mathrm { r a w } } \gg \sqrt { \Delta _ { \tau } }$ Hence $B _ { D } = \alpha _ { \mathrm { m a x } } \Delta _ { D R } = \alpha _ { \mathrm { m a x } } [ 1 + O ( q ) ]$

For nonzero gradients with nonzero sum, the exact denominator also gives $\Delta _ { D R } < 1 , = 1$ , or $> 1$ according as $c > 0 , = 0 , \mathrm { o r } < 0$ . Thus unity marks the sign change of the cross-signal contribution to $\| g _ { D } + g _ { R } \| ^ { 2 }$ . Relative to pure RL, the separate condition for a smaller squared update norm is $\| g _ { D } \| ^ { 2 } + 2 \langle g _ { D } , g _ { R } \rangle < 0$ □

## V PROOF OF PROPOSITION 16: NTK-GUIDED TOKEN MASKING

Proof. Write ${ \pmb u } _ { n } = { \bf J } ^ { n } \hat { \delta } _ { D } ^ { n } , { \pmb v } _ { n } = { \bf J } ^ { n } \hat { \delta } _ { R } ^ { n }$ , and $S ^ { - } = \{ n : K _ { D R } ( n ) < 0 \}$ . These are parameterspace directions obtained by positive residual rescaling, so $\left. { { \pmb u } _ { n } , { \pmb v } _ { n } } \right.$ has the sign of $K _ { D R } ( n )$ . The gate replaces $\alpha _ { \mathrm { m a x } } \pmb { u } _ { n } + ( 1 - \alpha _ { \mathrm { m a x } } ) \pmb { v } _ { n }$ by ${ \pmb v } _ { n }$ on $S ^ { - }$ and leaves the other positions unchanged. Consequently, the mean local reward-projection gain is

$$
\Delta \Pi _ { R } = \frac { 1 } { N } \sum _ { n } \langle { \pmb g } _ { H } ^ { \mathrm { s e l } , n } - { \pmb g } _ { H } ^ { \mathrm { M 3 } , n } , { \pmb v } _ { n } \rangle\tag{75}
$$

$$
= \frac { \alpha _ { \mathrm { m a x } } } { N } \sum _ { n \in S ^ { - } } \left. { \pmb v } _ { n } - { \pmb u } _ { n } , { \pmb v } _ { n } \right. = \frac { \alpha _ { \mathrm { m a x } } } { N } \sum _ { n \in S ^ { - } } \left( \| { \pmb v } _ { n } \| ^ { 2 } - \left. { \pmb u } _ { n } , { \pmb v } _ { n } \right. \right) > 0\tag{76}
$$

whenever $\alpha _ { \mathrm { m a x } } > 0$ and $S ^ { - }$ is nonempty. In the unit-direction model, $\| { \pmb u } _ { n } \| = \| { \pmb v } _ { n } \| = 1$ , this specializes to

$$
\Delta \Pi _ { R } = \frac { \alpha _ { \mathrm { m a x } } } { N } \sum _ { n \in S ^ { - } } ( 1 + | \cos \varphi _ { n } | ) = \alpha _ { \mathrm { m a x } } C _ { \mathrm { N T K } } ( 1 + | \bar { c } ^ { - } | ) ,\tag{77}
$$

where $C _ { \mathrm { N T K } } = | S ^ { - } | / N$ and $| \bar { c } ^ { - } |$ is the mean absolute parameter-space cosine on $S ^ { - }$ . The gain is zero when $S ^ { - }$ is empty. Under the diagonal NTK model, cross-position inner products vanish, so the aggregate projection onto $N ^ { - 1 } \sum _ { n } { \bf \bar { v } } _ { n }$ has the same sign, with gain $\Delta \Pi _ { R } / \dot { N }$ □

## W M3-SELECT PROJECTION AND CONVERGENCE GUARANTEES

Theorem 4 (M3-Select Projection Improvement over M3-Norm). Under the decoupled-position model ofProposition 16, with $0 < \alpha _ { \mathrm { m a x } } < 1$ , M3-Select satisfies:

(a) Reward projection: its aggregate update has at least the reward projection of uniform M3-Norm, strictly larger whenever $C _ { \mathrm { N T K } } > 0$

(b) Teacher allocation: the sum of its teacher coefficients is the fraction $1 - C _ { \mathrm { { N T K } } }$ of the uniform allocation, entirely on positions with $\tilde { K _ { D R } } ( n ) \geq 0$

(c) Reward convergence: under the additional smoothness, matched-norm, and noise assumptions ofTheorem 1, it satisfies the reward-stationarity bound in Eq. 54.

Proof. Part (a) is Proposition 16, using the positive rescaling from the mean local projection to the aggregate reward projection. For part (b), summing the gate gives $\begin{array} { r l } { ~ } & { { } \sum _ { n } \alpha _ { n } = \bar { \alpha } _ { \mathrm { m a x } } \bar { ( N - | S ^ { - } | ) } = } \end{array}$ $\bar { N } \bar { \alpha } _ { \mathrm { m a x } } ( 1 { - } C _ { \mathrm { N T K } } )$ , and every retained coefficient has nonnegative local compatibility. Part (c) is the descent-and-telescoping argument of Theorem 1, applied to the same gate and reward objective.

## X NTK CONFLICT RATE DYNAMICS

Proposition 21 (Accuracy Dependence of the Population Weighted Conflict Rate). Let $\bar { M } ^ { \pm } > 0$ be the expected per-trajectory absolute cross-signal NTK masses on correct and incorrect trajectories, and let ${ \widetilde { C } } _ { \pm }$ be the corresponding ratios of expected negative mass to expected absolute mass. If thesefour quantities arefixed as accuracy p varies, the pooled population conflict rate is

$$
\widetilde { C } _ { \mathrm { N T K } } ( p ) = \frac { p \bar { M } ^ { + } \widetilde { C } _ { + } + ( 1 - p ) \bar { M } ^ { - } \widetilde { C } _ { - } } { p \bar { M } ^ { + } + ( 1 - p ) \bar { M } ^ { - } } .\tag{78}
$$

It increases strictly with p $i f { \widetilde { C } } _ { + } > { \widetilde { C } } _ { - }$ , interpolating between $\widetilde { C } _ { - }$ at $p = 0$ and $\widetilde { C } _ { + } a t p = 1$

Proof. Conditioning the expected negative and absolute masses on trajectory correctness gives the displayed ratio. Differentiating and canceling the common terms yields

$$
\frac { d \widetilde { C } _ { \mathrm { N T K } } } { d p } = \frac { \bar { M } ^ { + } \bar { M } ^ { - } ( \widetilde { C } _ { + } - \widetilde { C } _ { - } ) } { [ p \bar { M } ^ { + } + ( 1 - p ) \bar { M } ^ { - } ] ^ { 2 } } > 0 .\tag{79}
$$

The endpoint values follow by substitution. Exact sign masking retains the fraction $1 - \widetilde { C } _ { \mathrm { N T K } }$ of absolute NTK mass, which therefore decreases under the same assumptions. This mass fraction differs from the position fraction $1 - C _ { \mathrm { { N T K } } }$ used in $\alpha _ { \mathrm { e f f } } \colon$ a change in mass allocation need not change the number of admitted positions. □

## Y PROOF OF PROPOSITION 13: COSINE INVERSION IN ON-POLICY OPSD

Proof. For a sampled trajectory, write $s ~ = ~ \nabla _ { \theta } \log \pi _ { \theta } ( \hat { y } ~ | ~ x ) , ~ g _ { R } ~ = ~ - A s$ , and $\begin{array} { r l } { g _ { D } } & { { } = } \end{array}$ $\begin{array} { r l } { {  { N ^ { - 1 } \sum _ { n } \nabla _ { \theta } \mathrm { K L } ( \dot { p _ { T } ^ { n } } \| p _ { S } ^ { n } ) } } } & { { } } \end{array}$ , where $A = ( r - b ) / \sigma _ { r } , 0 < b < 1$ , and $\sigma _ { r } > 0$ . Assume s and $\mathbf { \pmb { g } } _ { D }$ are nonzero, and define the normalized score–distillation alignment $q = \langle s , g _ { D } \rangle / ( \lVert s \rVert \lVert g _ { D } \rVert )$ . Then

$$
\cos ( g _ { R } , g _ { D } ) = - \operatorname { s i g n } ( A ) q ,\tag{80}
$$

$$
\begin{array} { r } { \mathbb { E } [ \cos ( g _ { R } , g _ { D } ) \mid r = 0 ] - \mathbb { E } [ \cos ( g _ { R } , g _ { D } ) \mid r = 1 ] = \mathbb { E } [ q \mid r = 0 ] + \mathbb { E } [ q \mid r = 1 ] > 0 , } \end{array}\tag{81}
$$

where the last inequality is the proposition’s conditional alignment assumption. In particular, positive conditional means of q give positive cosine on incorrect trajectories and negative cosine on correct ones.

The inner product underlying this condition includes all token pairings:

$$
\langle \pmb { s } , \pmb { g } _ { D } \rangle = \frac { 1 } { N } \sum _ { m , n } \langle \nabla _ { \pmb { \theta } } \log p _ { S } ^ { m } ( \hat { y } _ { m } ) , \nabla _ { \pmb { \theta } } \mathrm { K L } ( \pmb { p } _ { T } ^ { n } \lVert \pmb { p } _ { S } ^ { n } ) \rangle .\tag{82}
$$

Thus the assumption concerns gradient-weighted alignment, including cross-position interactions. For an on-policy teacher, positive q means its descent direction reduces the sampled trajectory’s log probability; changing the sign of the reward advantage reverses whether that change agrees with $\mathrm { R L }$ Table 11 reports the corresponding conditional cosines +0.14 and −0.11. □

## Z PROOF OF THEOREM 3: UNIFIED NTK LEARNING EFFICIENCY

Proof. Let $\begin{array} { r } { S _ { R } ^ { 2 } = G ^ { - 1 } \sum _ { i } \| \mathbf { g } _ { R } ^ { ( i ) } \| ^ { 2 } } \end{array}$ and $\begin{array} { r } { \pmb { g } _ { R } = G ^ { - 1 } \sum _ { i } \pmb { g } _ { R } ^ { ( i ) } } \end{array}$ . The definitions of diversity, folding, and the normalized hybrid direction give the complete factorization

$$
\eta _ { \mathrm { e x p l o r e } } = \frac { \| { \pmb g } _ { R } \| ^ { 2 } } { S _ { R } ^ { 2 } } = \Delta _ { \tau } ^ { - 1 } = 1 - \gamma ,\tag{83}
$$

$$
\eta _ { \mathrm { h y b r i d } } = \langle ( 1 - \alpha ) \hat { \pmb { g } } _ { R } + \alpha \hat { \pmb { g } } _ { D } , \hat { \pmb { g } } _ { R } \rangle = ( 1 - \alpha ) + \alpha \Phi _ { D R } ,\tag{84}
$$

$$
\eta ^ { \mathrm { e f f } } : = \eta _ { \mathrm { e x p l o r e } } \eta _ { \mathrm { h y b r i d } } = ( 1 - \gamma ) \big [ ( 1 - \alpha ) + \alpha \Phi _ { D R } \big ] .\tag{85}
$$

Both factors depend on NTK inner products. Writing $\pmb { \mathscr { s } } _ { i } = \nabla \log \pi ( \hat { y } ^ { ( i ) } \mid x )$ and $\pmb { g } _ { R } ^ { ( i ) } = - A _ { i } \pmb { s } _ { i }$ gives

$$
G ^ { 2 } \| g _ { R } \| ^ { 2 } = \sum _ { i , j } A _ { i } A _ { j } \langle s _ { i } , s _ { j } \rangle = \sum _ { i , j } A _ { i } A _ { j } \sum _ { n , m } K _ { t } ( \tau _ { i } , n ; \tau _ { j } , m ) ,\tag{86}
$$

while Proposition 20 expands $\Phi _ { D R }$ into within-position and cross-position signal interactions.   
These two expansions establish the shared geometry.

The composite index uses the squared cancellation factor. For a unit-scale hybrid update, the actual first-order RL decrease relative to the RMS single-trajectory scale is instead $\sqrt { 1 - \gamma } \eta _ { \mathrm { h y b r i d } }$ . At fixed $\alpha ,$ differentiation of the composite index yields

$$
\frac { d \eta ^ { \mathrm { e f f } } } { d p } = - \gamma ^ { \prime } ( p ) \left[ ( 1 - \alpha ) + \alpha \Phi _ { D R } ( p ) \right] + ( 1 - \gamma ( p ) ) \alpha { \Phi _ { D R } } ^ { \prime } ( p ) .\tag{87}
$$

Hence decreasing alignment lowers the hybrid factor; monotonicity of the product follows when the displayed derivative is nonpositive. Under fixed conditional trajectory cosines $\mu _ { 0 } > \mu _ { 1 }$ , their mixture has derivative $\mu _ { 1 } - \mu _ { 0 }$ , with zero crossing $p ^ { * } = \mu _ { 0 } / ( \mu _ { 0 } - \mu _ { 1 } )$ when $\mu _ { 0 } ~ > ~ 0 ~ > ~ \mu _ { 1 }$ This crossing characterizes the alignment factor, while the cancellation factor contributes separately through $\gamma ^ { \prime } ( p )$ . □

## AA PROOF OF PROPOSITION 4: OPTIMAL PER-POSITION MIXING

Proof. Write $c _ { n } = \cos \varphi _ { n }$ . Valuing admitted distillation by $\rho \alpha _ { n }$ in units of RL projection gives the separable allocation objective

$$
\mathcal { I } = \frac { 1 } { N } \sum _ { n } \left[ ( 1 - \alpha _ { n } ) + \alpha _ { n } c _ { n } + \rho \alpha _ { n } \right] = 1 + \frac { 1 } { N } \sum _ { n } \alpha _ { n } ( c _ { n } - \tau ) , \qquad \tau = 1 - \rho ,\tag{88}
$$

subject to $0 \leq \alpha _ { n } \leq \alpha _ { \mathrm { m a x } }$ . If the optional budget $\begin{array} { r } { N ^ { - 1 } \sum _ { n } \alpha _ { n } \le \bar { \alpha } } \end{array}$ is imposed, its multiplier $\lambda \geq 0$ shifts each coefficient to $c _ { n } - \tau - \lambda$ . Maximizing these linear terms gives

$$
\alpha _ { n } ^ { * } = \left\{ { 0 , \atop \mathrm { a n y ~ f e a s i b l e ~ v a l u e ~ i n ~ } [ 0 , \alpha _ { \mathrm { m a x } } ] , } \right. \sp { } c _ { n } > \tau + \lambda ,\tag{89}
$$

For $\rho = 1$ and a slack budget, choose the upper endpoint at ties to obtain $\alpha _ { n } ^ { * } = \alpha _ { \mathrm { m a x } } \mathbf { 1 } [ c _ { n } \geq 0 ]$ Under the isotropic within-position kernel assumption $\mathbf { K } ( n , n ) = \lambda _ { n } \mathbf { I }$ with $\lambda _ { n } > 0 , K _ { D R } ( n ) =$ $\lambda _ { n } \langle \delta _ { D } ^ { n } , \delta _ { R } ^ { n } \rangle$ has the same sign, yielding the M3-Select gate.

For a noisy score $\hat { c } _ { n } = c _ { n } + \varepsilon _ { n }$ with logistic error of scale $\beta ^ { - 1 }$ , the expected hard allocation is

$$
\begin{array} { r } { \mathbb { E } \big [ \alpha _ { \operatorname* { m a x } } \mathbf { 1 } \big [ \hat { c } _ { n } \ge 0 \big ] \big ] = \alpha _ { \operatorname* { m a x } } \operatorname* { P r } \big ( \varepsilon _ { n } \ge - c _ { n } \big ) = \alpha _ { \operatorname* { m a x } } \sigma \big ( \beta c _ { n } \big ) . } \end{array}\tag{90}
$$

This is M3-Soft. As $\beta \to \infty$ , it approaches the hard gate for $c _ { n } \neq 0$ and assigns half the budget at the indifferent point $c _ { n } = 0$ □

## AB CONVERGENCE RATE COMPARISON: M3-NORM VS GRADNORM

Theorem 5 (Descent-Bound Comparison at a Common Gradient Scale). Let $\mathcal { L } _ { R }$ be L-smooth and bounded below, with $\pmb { g } _ { R } = \nabla \mathcal { L } _ { R }$ . Compare the norm-restored M3 direction $h _ { \mathrm { M 3 } } = \| { \pmb g } _ { R } \| [ 1 -$ $\alpha ) \hat { { \pmb g } } _ { R } + \alpha \hat { { \pmb g } } _ { D } ]$ with the norm-balanced GradNorm direction $h _ { \mathrm { G N } } = \| { \pmb g } _ { R } \| \big ( { \hat { \pmb g } } _ { R } + { \hat { \pmb g } } _ { D } \big ) / ( 1 + \kappa )$ Assume $0 \dot { \leq } \dot { \alpha } \leq \alpha _ { 0 } < 1 / 2 , \Phi _ { D R } \geq - 1 + \delta f o r f u x e d \delta > 0 ,$ , and afixed magnitude ratio $\kappa > 0 .$ . For method $j ,$ let the stochastic update be $h _ { j } + \xi _ { j }$ with conditional mean $\mathbb { E } [ \pmb { \xi } _ { j } ] = 0$ and $\mathbb { E } \| \pmb { \xi } _ { j } \| ^ { 2 } \le s _ { j } ^ { 2 }$ Define

$$
( c _ { \mathrm { M 3 } } , b _ { \mathrm { M 3 } } ) = ( 1 - 2 \alpha _ { 0 } , 1 ) , \qquad ( c _ { \mathrm { G N } } , b _ { \mathrm { G N } } ) = \left( \frac { \delta } { 1 + \kappa } , \frac { 2 } { 1 + \kappa } \right) .\tag{91}
$$

For a common step size $\eta \leq \mathrm { m i n } _ { j } c _ { j } / ( L b _ { j } ^ { 2 } )$ and $\Delta _ { R } = \mathcal { L } _ { R } ( { \pmb \theta } ^ { 0 } ) - \mathrm { i n f } \mathcal { L } _ { R }$

$$
\frac { 1 } { T } \sum _ { t < T } \mathbb { E } \| \nabla \mathcal { L } _ { R } ( \pmb { \theta } ^ { t } ) \| ^ { 2 } \leq \frac { 2 \Delta _ { R } } { \eta c _ { j } T } + \frac { L \eta s _ { j } ^ { 2 } } { c _ { j } } .\tag{92}
$$

When $L \eta s _ { i } ^ { 2 } / c _ { j } \ \le \ \epsilon / 2$ , sufficient iteration budgets scale as $O ( \Delta _ { R } / ( \eta \epsilon ) )$ for M3 and $O ( ( 1 +$ $\kappa ) \Delta _ { R } / ( \eta \epsilon ) ) f o r G r a d N o r m .$

Proof. Bilinearity and the triangle inequality give $\langle g _ { R } , \pmb { h } _ { j } \rangle \geq c _ { j } \| \pmb { g } _ { R } \| ^ { 2 }$ and $\| h _ { j } \| \leq b _ { j } \| g _ { R } \|$ . Applying smoothness conditionally on the current iterate and using the step-size restriction yields the single descent chain

$$
\mathbb { E } _ { t } [ \mathcal { L } _ { R } ( \pmb { \theta } ^ { t + 1 } ) ] \le \mathcal { L } _ { R } ( \pmb { \theta } ^ { t } ) - \eta \langle \pmb { g } _ { R } , \pmb { h } _ { j } \rangle + \frac { L \eta ^ { 2 } } { 2 } ( \| \pmb { h } _ { j } \| ^ { 2 } + s _ { j } ^ { 2 } )\tag{93}
$$

$$
\leq \mathcal { L } _ { { R } } ( \pmb { \theta } ^ { t } ) - \eta \left( c _ { j } - \frac { L \eta b _ { j } ^ { 2 } } { 2 } \right) \| \pmb { g } _ { { R } } \| ^ { 2 } + \frac { L \eta ^ { 2 } s _ { j } ^ { 2 } } { 2 }\tag{94}
$$

$$
\leq \mathcal { L } _ { R } ( \pmb { \theta } ^ { t } ) - \frac { \eta c _ { j } } { 2 } \| \pmb { g } _ { R } \| ^ { 2 } + \frac { L \eta ^ { 2 } s _ { j } ^ { 2 } } { 2 } .\tag{95}
$$

Telescoping and dividing by $\eta c _ { j } T / 2$ proves the bound; taking $T \geq 4 \Delta _ { R } / ( \eta c _ { j } \epsilon )$ gives the stated sufficient budgets. Their optimization terms differ by a factor proportional to $1 + \kappa$ under the specified common scale and step size. □

## AC CONCENTRATION OF THE NTK CONFLICT RATE

Proposition 22 (Conflict-Rate Variance under Mixing). Let $\iota _ { n } = \mathbf { 1 } [ K _ { D R } ( n ) < 0 ]$ and $\begin{array} { r l } {  { C _ { \mathrm { N T K } } = } } \end{array}$ $N ^ { - \bar { 1 } } \sum _ { n } \iota _ { n } .$ . Suppose $| \operatorname { C o r r } ( \iota _ { n } , \iota _ { n + k } ) | \leq C _ { \rho } e ^ { - \beta _ { \operatorname* { m i x } } k }$ whenever both variances are nonzero. Define $D _ { \rho } = 1 + 2 C _ { \rho } / ( e ^ { \beta _ { \mathrm { m i x } } } - 1 )$ . Then

$$
\mathrm { V a r } ( C _ { \mathrm { N T K } } ) \leq \frac { D _ { \rho } } { 4 N } , \qquad \mathrm { P r } [ | C _ { \mathrm { N T K } } - \mathbb { E } C _ { \mathrm { N T K } } | > t ] \leq \operatorname* { m i n } \biggl \{ 1 , \frac { D _ { \rho } } { 4 N t ^ { 2 } } \biggr \} .\tag{96}
$$

For the mean over B independent length-N rollouts, with probability at least $1 - \delta ,$ , the deviation is at most $\sqrt { D _ { \rho } / ( 4 B N \delta ) }$

Proof. Since $\mathrm { V a r } ( \iota _ { n } ) \leq 1 / 4$ , the covariance expansion reduces to a geometric series:

$$
\mathrm { V a r } \left( { \frac { 1 } { N } } \sum _ { n } \iota _ { n } \right) = { \frac { 1 } { N ^ { 2 } } } \left[ \sum _ { n } \mathrm { V a r } ( \iota _ { n } ) + 2 \sum _ { k = 1 } ^ { N - 1 } \sum _ { n = 1 } ^ { N - k } \mathrm { C o v } ( \iota _ { n } , \iota _ { n + k } ) \right]\tag{97}
$$

$$
\leq \frac { 1 } { 4 N } \left[ 1 + 2 \sum _ { k = 1 } ^ { N - 1 } \left( 1 - \frac { k } { N } \right) C _ { \rho } e ^ { - \beta _ { \operatorname* { m i x } } k } \right] \leq \frac { D _ { \rho } } { 4 N } .\tag{98}
$$

Chebyshev’s inequality gives the tail bound; independence divides the variance by $B .$ . Using $D _ { \rho } \leq$ $1 + 2 \dot { C } _ { \rho } / \beta _ { \mathrm { m i x } }$ , the illustrative setting $N = 2 5 6 , \bar { C } _ { \rho } = 1 , \beta _ { \mathrm { m i x } } = 0 . 5$ , and $\delta = 0 . 0 5$ gives deviations at most 0.313 for one rollout and 0.079 for $B = 1 6$ □

## AD PROOF OF PROPOSITION 10: DISTILLATION AS IMPLICIT REGULARIZATION

Proof. Consider matched teacher and student distributions evaluated on the same contexts, with $\pi _ { \pmb { \theta } _ { T } } = \pi _ { T }$ . The local KL expansion is $\begin{array} { r } { \mathcal { L } _ { D } ( \pmb { \theta } _ { T } + \Delta \pmb { \theta } ) = \frac { 1 } { 2 } \Delta \pmb { \theta } ^ { \top } \mathbf { F } _ { D } \Delta \pmb { \theta } + o ( \| \Delta \pmb { \theta } \| ^ { 2 } ) } \end{array}$ , where $\mathbf { F } _ { D } \succeq 0$ is the Fisher matrix. In the proposition’s isotropic quadratic model, $\mathbf { F } _ { D } = f _ { D } \mathbf { I }$ with $f _ { D } > 0$ , so $\nabla { \mathcal { L } } _ { D } = f _ { D } \Delta \theta$ and $\langle \nabla \mathcal { L } _ { D } , \bar { \Delta } \pmb { \theta } \rangle = f _ { D } \| \Delta \pmb { \theta } \| ^ { 2 }$ : distillation provides a restoring direction.

With constant ${ \pmb g } _ { R }$ , initial condition $\Delta \theta ( 0 ) = 0$ , and $0 < \alpha < 1$ , solve the resulting linear flow in one step:

$$
\begin{array} { r } { \dot { \Delta { \pmb \theta } } _ { H } = - \eta [ ( 1 - \alpha ) { \pmb g } _ { R } + \alpha f _ { D } \Delta { \pmb \theta } _ { H } ] , } \end{array}\tag{99}
$$

$$
\Delta \theta _ { H } ( T ) = - \frac { ( 1 - \alpha ) g _ { R } } { \alpha f _ { D } } ( 1 - e ^ { - \eta \alpha f _ { D } T } ) , \qquad \| \Delta \theta _ { H } ( T ) \| \leq \frac { ( 1 - \alpha ) \| g _ { R } \| } { \alpha f _ { D } } .\tag{100}
$$

Pure RL has $\Delta \pmb { \theta } _ { R } ( T ) = - \eta T \pmb { g } _ { R }$ . If degradation is proportional to deviation with the same coeffi cient for both flows, setting $x = \eta \alpha f _ { D } T$ and $\rho _ { \mathrm { r e g } } = \eta f _ { D } T / 2$ gives

$$
\frac { \delta _ { H } } { \delta _ { R } } = ( 1 - \alpha ) \frac { 1 - e ^ { - x } } { x } \leq \frac { 1 - \alpha } { 1 + x / 2 } \leq \frac { 1 } { 1 + \alpha \rho _ { \mathrm { r e g } } } .\tag{101}
$$

The scalar inequality follows from $( 1 + x / 2 ) ( 1 - e ^ { - x } ) \leq x$ for $x \ge 0 ;$ ; the difference has derivative $[ 1 - ( 1 + x ) e ^ { - x } ] / 2 \geq 0$ and vanishes at zero.

For the rank-dependent model $G ( r ) = \epsilon _ { \mathrm { d r o w n } } - [ \delta _ { R } ( r ) - \delta _ { H } ( r ) ]$ , a finite crossing follows if $\epsilon _ { \mathrm { d r o w n } } >$ 0 is bounded, $\delta _ { R } ( r )$ is continuous and grows without bound, and $\rho _ { \mathrm { r e g } } ( r ) \geq \rho _ { 0 } > 0$ . Indeed,

$$
G ( r ) \leq \epsilon _ { \mathrm { d r o w n } } - \frac { \alpha \rho _ { 0 } } { 1 + \alpha \rho _ { 0 } } \delta _ { R } ( r ) \longrightarrow - \infty .\tag{102}
$$

Together with a positive initial gap, continuity gives a crossing of this model. The reported positive gap at $r = 6 4$ and negative gap at $r = 1 2 8$ locate the observed reversal between the tested ranks.

## AE PROOF OF PROPOSITION: LSGV VARIANCE AMPLIFICATION

Proof. Write ${ \pmb u } = { \pmb g } _ { R } , { \pmb v } = { \pmb g } _ { D } , { \pmb \mu } _ { u } = \mathbb E { \pmb u } , { \pmb \mu } _ { v } = \mathbb E { \pmb v }$ , and let the joint estimator covariance be $\Omega / { \dot { B } } .$ , including its cross-signal blocks. Assume $\lambda _ { - } \mathbf { I } \preceq \pmb { \Omega } \preceq \lambda _ { + } \mathbf { I }$ for fixed positive constants, $0 <$ $m \leq \| \pmb { \mu } _ { v } \| \leq \bar { M }$ , and $| c _ { 0 } | \leq 1 - \delta _ { c }$ , where $c _ { 0 } = \langle \pmb { \mu } _ { u } , \pmb { \mu } _ { v } \rangle / ( \| \pmb { \mu } _ { u } \| \| \pmb { \mu } _ { v } \| )$ . Set $\bar { \kappa } = \| \pmb { \mu _ { u } } \| / \| \pmb { \mu _ { v } } \| \leq 1$ In the small-noise regime, with delta-method remainders negligible relative to the leading variance, the cosine gradient, evaluated at $( \mu _ { u } , \mu _ { v } )$ with $\hat { \pmb { \mu } } _ { j } = \pmb { \mu } _ { j } / \lVert \pmb { \mu } _ { j } \rVert$ , and its squared norm are

$$
\nabla _ { u } f = \frac { \hat { \pmb \mu } _ { v } - c _ { 0 } \hat { \pmb \mu } _ { u } } { \lVert \pmb \mu _ { u } \rVert } ,
$$

$$
\nabla _ { v } f = \frac { \hat { \pmb \mu } _ { u } - c _ { 0 } \hat { \pmb \mu } _ { v } } { \lVert \pmb \mu _ { v } \rVert } ,\tag{103}
$$

$$
\| \nabla f \| ^ { 2 } = ( 1 - c _ { 0 } ^ { 2 } ) \left( { \frac { 1 } { \| \pmb { \mu } _ { u } \| ^ { 2 } } } + { \frac { 1 } { \| \pmb { \mu } _ { v } \| ^ { 2 } } } \right) , f ( \pmb { u } , \pmb { v } ) = { \frac { \langle \pmb { u } , \pmb { v } \rangle } { \| \pmb { u } \| \| \pmb { v } \| } } .\tag{104}
$$

Consequently, retaining the joint covariance throughout,

$$
\mathrm { V a r } ( c ) = { \frac { 1 } { B } } \nabla f ^ { \top } \pmb \Omega \nabla f + o \bigg ( { \frac { \| \nabla f \| ^ { 2 } } { B } } \bigg ) = \Theta \bigg ( { \frac { 1 - c _ { 0 } ^ { 2 } } { B } } \left[ { \frac { 1 } { \| \mu _ { u } \| ^ { 2 } } } + { \frac { 1 } { \| \mu _ { v } \| ^ { 2 } } } \right] \bigg ) = \Theta \bigg ( { \frac { 1 } { B \bar { \kappa } ^ { 2 } } } \bigg ) .\tag{105}
$$

This rate describes amplification while relative noise is small. The global bound $\mathrm { V a r } ( c ) \leq 1$ continues to hold when that approximation ceases to apply.

For $\alpha = \alpha _ { \mathrm { m a x } } \sigma ( \beta c )$ , a second delta expansion and the Lipschitz constant $\alpha _ { \mathrm { m a x } } \beta / 4$ give, respectively,

$$
\mathrm { V a r } ( \alpha ) = \alpha _ { \mathrm { m a x } } ^ { 2 } \beta ^ { 2 } \sigma ^ { \prime } ( \beta c _ { 0 } ) ^ { 2 } \mathrm { V a r } ( c ) + o ( \mathrm { V a r } ( c ) ) ,\tag{106}
$$

$$
\mathrm { V a r } ( \alpha ) \leq \operatorname* { m i n } \left\{ { \frac { \alpha _ { \mathrm { m a x } } ^ { 2 } } { 4 } } , { \frac { \alpha _ { \mathrm { m a x } } ^ { 2 } \beta ^ { 2 } } { 1 6 } } \mathrm { V a r } ( c ) \right\} .\tag{107}
$$

Static Hybrid has $\mathrm { V a r } ( \alpha ) = 0$ . To identify the corresponding contribution to update variance, let $a _ { 0 } ~ = ~ \alpha _ { \mathrm { m a x } } \sigma ( \beta c _ { 0 } )$ $\pmb { d } = \pmb { \mu _ { v } } - \pmb { \mu _ { u } }$ , and $\delta h _ { 0 } = ( 1 - a _ { 0 } ) ( { \pmb u } - { \pmb \mu } _ { u } ) + a _ { 0 } ( { \pmb v } - { \pmb \mu } _ { v } )$ . Linearizing $\pmb { h } = ( 1 - \alpha ) \pmb { u } +$ αv yields

$$
\mathrm { C o v } ( h ) \simeq \mathrm { C o v } ( \delta h _ { 0 } ) + \mathrm { V a r } ( \alpha ) d d ^ { \top } + \mathrm { C o v } ( \delta h _ { 0 } , \alpha ) d ^ { \top } + d \mathrm { C o v } ( \alpha , \delta h _ { 0 } ) .\tag{108}
$$

A constant gate removes the gate fluctuation terms. When the gate error is uncorrelated with $\delta h _ { 0 }$ the additional covariance is the positive semidefinite term $\mathrm { V a r } \bar { ( \alpha ) } d d ^ { \top }$ ; for a gate computed from the same gradients, the displayed cross-covariances determine its net effect. □

## AF PROOF OF PROPOSITION: CUMULATIVE CONFLICT BUDGET

Proof. At step s, let $\begin{array} { r } { q _ { s } \ = \ N _ { s } ^ { - 1 } \sum _ { n } { \alpha _ { n , s } \mathbf { 1 } } [ K _ { D R } ^ { s } ( n ) \ < \ 0 ] } \end{array}$ measure retained conflicting teacher weight, and define the scalar budget $\begin{array} { r } { { \bf \ddot { \boldsymbol E } } ( t ) = \sum _ { s < t } \kappa _ { s } \boldsymbol { q } _ { s } } \end{array}$ . For naive mixing $q _ { s } = \alpha C _ { \mathrm { N T K } } { } ^ { s } ;$ exact sign masking gives $q _ { s } = 0$ . If the time-averaged product stabilizes, the entire accumulation law is

$$
E ( t ) = t \left( \frac { 1 } { t } \sum _ { s \leq t } \kappa _ { s } q _ { s } \right) = t \bar { d } + o ( t ) , \qquad T _ { \Gamma } \approx \frac { \Gamma } { \bar { d } } ,\tag{109}
$$

where $T _ { \Gamma }$ is the first crossing of a fixed budget threshold Γ in this model. For approximately constant κ and conflict rate, $\bar { d } _ { \mathrm { n a i v e } } \bar { = } \alpha \bar { \kappa } \bar { C } _ { \mathrm { N T K } }$ . If imperfect masking retains weight $q _ { s } \approx \alpha _ { \mathrm { e f f } } \delta _ { \mathrm { m a s k } }$ , then under the same $\bar { \kappa } ,$

$$
\frac { E _ { \mathrm { M 3 } } ( t ) } { E _ { \mathrm { n a i v e } } ( t ) } \approx \frac { \alpha _ { \mathrm { e f f } } \delta _ { \mathrm { m a s k } } } { \alpha \bar { C } _ { \mathrm { N T K } } } .\tag{110}
$$

The illustrative values $\alpha _ { \mathrm { e f f } } = 0 . 0 9 , \delta _ { \mathrm { m a s k } } = 0 . 0 5 , \alpha = 0 . 5$ , and $\bar { C } _ { \mathrm { N T K } } = 0 . 4 0 \mathrm { g i v e } 0 . 0 2 2 5$ , or about 44 times slower budget accumulation.

This calculation concerns retained conflict. A parameter deviation formed from conflict updates obeys $\begin{array} { r } { \| \sum _ { s } \eta _ { s } { d _ { s } } \| \le \sum _ { s } \eta _ { s } \| { d _ { s } } \| } \end{array}$ , so connecting the scalar budget to reward collapse requires a model of update directions and a collapse threshold. The reported ordering—Hybrid at step 221, GradNorm at 358, GRPO at 412, and no M3 collapse through step 500—is an experimental observation. In particular, pure GRPO has $q _ { s } = 0$ and its collapse is governed by dynamics outside this conflict budget. □

## AG SUPPLEMENTARY DETAILS FOR THE MAIN ANALYSIS AND METHOD

## AG.1 DISTILLATION DIVERGENCE AND LOCAL RESIDUAL MODEL

For $D _ { \lambda } ( \pmb { p } \| \pmb { q } ) = \lambda \mathrm { K L } ( \pmb { p } \| \pmb { m } ) + ( 1 - \lambda ) \mathrm { K L } ( \pmb { q } \| \pmb { m } )$ , with $m = \lambda p + ( 1 - \lambda ) q$ , strictly positive distributions satisfy

$$
\frac { D _ { \lambda } ( p \| q ) } { \lambda } \to \mathrm { K L } ( p \| q ) \quad ( \lambda \to 0 ) , \qquad \frac { D _ { \lambda } ( p \| q ) } { 1 - \lambda } \to \mathrm { K L } ( q \| p ) \quad ( \lambda \to 1 ) .
$$

The experiments use the symmetric point $\lambda = 1 / 2$ . For an unclipped token near student–teacher agreement,

$$
\nabla _ { z ^ { n } } D _ { \lambda } ( \pmb { p } _ { T } ^ { n } \| \pmb { p } _ { S } ^ { n } ) = \lambda ( 1 - \lambda ) ( \pmb { p } _ { S } ^ { n } - \pmb { p } _ { T } ^ { n } ) + O ( \| \pmb { p } _ { S } ^ { n } - \pmb { p } _ { T } ^ { n } \| ^ { 2 } ) .
$$

The main analysis absorbs the leading constant into the teacher update scale and uses the local forward-KL residual $\pmb { \delta } _ { D } ^ { n } = \pmb { p } _ { S } ^ { n } - \pmb { p } _ { T } ^ { n }$ . Tokens above the clipping threshold have zero distillation gradient. This is a local approximation, not an identity for arbitrary distributions.

For completeness, the GRPO group statistics in Section 3.1 are $\begin{array} { r } { \bar { r } \ = \ G ^ { - 1 } \sum _ { i } r ^ { ( i ) } } \end{array}$ and $\sigma _ { r } ^ { 2 } \ =$ $\begin{array} { r l } { G ^ { - 1 } \sum _ { i } ( r ^ { ( i ) } - \bar { r } ) ^ { 2 } } \end{array}$ . Their values and the sampled responses are held fixed during differentiation.

## AG.2 GRADIENT GEOMETRY AND MAGNITUDE ASYMMETRY

The NTK sums in Eqs. 7–8 form the gradient Gram matrix

$$
\left( \begin{array} { c c } { { \| { \pmb g } _ { D } \| ^ { 2 } } } & { { \langle { \pmb g } _ { D } , { \pmb g } _ { R } \rangle } } \\ { { \langle { \pmb g } _ { R } , { \pmb g } _ { D } \rangle } } & { { \| { \pmb g } _ { R } \| ^ { 2 } } } \end{array} \right) .
$$

Its diagonal entries measure gradient strength; its off-diagonal entries measure interaction. The residual quadratic sums are kernel energies, equal to squared gradient norms up to the common factor $N ^ { \dot { - } 2 }$ . Together, the entries determine the first-order loss changes for given α and $\eta .$ The normalized score $\Phi _ { D R }$ retains direction but removes magnitude.

For $0 < \alpha < 1$ and $\Phi _ { D R } < 0 .$ , normalize each harmful cross-effect in Eq. 6 by the corresponding objective’s own descent term. The ratios are

$$
q _ { D } = \frac { 1 - \alpha } { \alpha } \kappa | \Phi _ { D R } | , \qquad q _ { R } = \frac { \alpha } { 1 - \alpha } \frac { | \Phi _ { D R } | } { \kappa } , \qquad \frac { q _ { D } } { q _ { R } } = \left( \frac { 1 - \alpha } { \alpha } \right) ^ { 2 } \kappa ^ { 2 } .
$$

Thus the relative asymmetry scales as $\Theta ( \kappa ^ { 2 } )$ for fixed interior mixing weights. A shrinking teacher residual can increase κ when the reward gradient remains substantial, but the reward residual can also vanish when $A = 0$ . Both residual directions and the kernel affect their parameter-gradient norms. Proposition 7 reports the observed LoRA-rank and model-scale comparisons, and $\mathsf { A p - }$ pendix F develops the extended connections.

## AG.3 MINIMUM-NORM GLOBAL MIXING

For ${ \pmb g } _ { H } ( \alpha ) = \alpha { \pmb g } _ { D } + ( 1 - \alpha ) { \pmb g } _ { R }$ , let $r _ { D } = \| g _ { D } \| , r _ { R } = \| g _ { R } \|$ , and $\Phi _ { D R } = \langle { \pmb g } _ { D } , { \pmb g } _ { R } \rangle / ( r _ { D } r _ { R } )$ assuming nonzero gradients. The minimum-norm mixture is the point closest to zero on the segment joining the two gradients. Minimizing $\| \pmb { g } _ { H } ( \alpha ) \| ^ { 2 }$ over [0, 1] gives, for $g _ { D } \neq g _ { R }$

$$
\alpha ^ { * } = \mathrm { c l i p } _ { [ 0 , 1 ] } \left( \frac { r _ { R } ^ { 2 } - \Phi _ { D R } r _ { D } r _ { R } } { r _ { D } ^ { 2 } + r _ { R } ^ { 2 } - 2 \Phi _ { D R } r _ { D } r _ { R } } \right) ,\tag{111}
$$

The clipping operation restricts the unconstrained minimizer to [0, 1]. If the gradients coincide, every coefficient produces the same update. For $\kappa = r _ { R } / r _ { D } \gg 1 , \stackrel {  } { 1 } - \alpha ^ { * } = \bar { O } ( \kappa ^ { - 1 } )$ . This directiondependent criterion need not equal the norm-balancing choice $\alpha = \kappa / ( 1 + \kappa )$ in Section 4.1. Neither global coefficient can remove teacher contributions only at conflicting positions. The quadratic gap around $\alpha ^ { * }$ is given in Eq. 26.

## AG.4 NORMALIZATION AND CONTINUOUS-GATE LIMITS

The full-gradient normalization principle is

$$
g _ { H } ^ { \mathrm { M A } } ( \alpha ) = \bar { s } [ \alpha \hat { g } _ { D } + ( 1 - \alpha ) \hat { g } _ { R } ] , \qquad \hat { g } _ { D } = g _ { D } / \| g _ { D } \| , \quad \hat { g } _ { R } = g _ { R } / \| g _ { R } \| ,
$$

for nonzero gradients, with s¯ an EMA-smoothed update scale. Algorithm 2 implements normalization on residuals before applying the Jacobians. This preserves compatibility signs but need not equalize parameter-gradient norms. The population-norm-matched convergence model is therefore distinct from the residual-normalized implementation.

For M3-Soft, $\beta $ ∞ recovers hard selection wherever $K _ { D R } ( n ) \neq 0$ . At an exact zero score the weight remains $\alpha _ { \mathrm { m a x } } / 2$ , whereas M3-Select admits it at weight $\alpha _ { \mathrm { m a x } }$ . As $\beta  0$ , all positions receive $\alpha _ { \mathrm { m a x } } / 2$ . The mean teacher weight is $\alpha _ { \mathrm { e f f } } = N ^ { - 1 } \textstyle \sum _ { n } \alpha _ { n }$ . Larger $\beta$ makes selection sharper but increases sensitivity near zero alignment; Appendix B.3 evaluates this trade-off.

## AG.5 CONDITIONAL VARIANCE AND GLOBAL REBALANCING

In Theorem 1, ρ lower-bounds the mean reward projection weighted by each position’s squared $\underline { { \rho } } _ { R }$ mean reward-gradient norm; $\underline { { \rho } } _ { R } = 1 - \alpha _ { \mathrm { m a x } }$ is admissible. The noise factor $v _ { \alpha }$ uniformly bounds the noise-energy-weighted mean of $( 1 - \alpha _ { n } ) ^ { 2 }$ . The effective step size includes the update scale, estimated in the algorithm by an EMA.

For equal position noise variances, admitted fraction $\rho _ { \mathrm { a d m } }$ , and $\bar { \alpha } = \alpha _ { \mathrm { m a x } } \rho _ { \mathrm { a d m } }$ , the exact factor is

$$
v _ { t } = ( 1 - \rho _ { \mathrm { a d m } } ) + \rho _ { \mathrm { a d m } } ( 1 - \alpha _ { \mathrm { m a x } } ) ^ { 2 } = ( 1 - { \bar { \alpha } } ) ^ { 2 } + \alpha _ { \mathrm { m a x } } ^ { 2 } \rho _ { \mathrm { a d m } } ( 1 - \rho _ { \mathrm { a d m } } ) .
$$

Rejected positions retain the full reward-noise contribution, while admitted positions reduce it. These statements condition on the history before fresh reward noise and treat the teacher and gates as fixed. Extra variability in estimated gates or teacher directions is not included.

Remark 6 (Limitations of Global Gradient Rebalancing). Global rebalancing does not select teacher contributions by position. For symmetric two-gradient projections with $- 1 < \Phi _ { D R } < 0 ,$ PCGrad (Yu et al., 2020) removes conflicting components but preserves the norm ratio $( A p \cdot$ pendix M). At κ = 3400 and equal mixing weights, the teacher accounts for approximately 0.03% of the sum of weighted gradient norms. Under the equal-target-rate, unit-sum weighting model of Proposition 5, GradNorm (Chen et al., 2018) balances norms with reward coefficient $\bar { 1 } / ( 1 + \kappa ) \bar { }$ reducing its step scale by order $1 / \kappa .$ M3 combines residual normalization with token-level control ofthe teacher signal.