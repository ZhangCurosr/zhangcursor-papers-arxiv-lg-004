# Teacher–Student Gaps Are Not Enough: Outcome-Guided On-Policy Distillation for Multi-Turn Autonomous Agents

Tong Zhang<sup>\*1,2</sup>, Zhou Liu<sup>2</sup>, Yihao Liu<sup>\*1</sup>, Jiahua Bao<sup>\*1</sup>, Xuchen Li<sup>3</sup>, Honglin Lin<sup>4</sup>, Tao Cheng<sup>\*1</sup>, Zhihan Yu<sup>\*1</sup>, Kai Tang<sup>†1</sup>, Xiaoxi Jiang<sup>1</sup> and Guanjun Jiang<sup>1</sup>

<sup>1</sup>Qwen Large Model Application Team, Alibaba, <sup>2</sup>Peking University, <sup>3</sup>University of Chinese Academy of Sciences, <sup>4</sup>Shanghai Jiao Tong University

Work done during an internship at Alibaba. <sup>†</sup>Corresponding author.

On-policy distillation (OPD) trains a student on its own trajectories with dense teacher supervision. Recent work on OPD for multi-turn autonomous agents often treats large teacher–student token-level distributional gaps as promising intervention points, linking larger gaps to a greater need for correction. Yet, our empirical analysis reveals a supervision–benefit mismatch: large gaps can be benign, while small gaps can be outcome-critical. Teacher–student gaps capture diferences at the current turn, whereas the benefit of teacher guidance depends on how the current student interacts with the environment afterward. The student may still succeed despite choosing an action that difers from the teacher’s, while a teacher-preferred action may lead to a state from which the student cannot complete the task. Local gaps alone are therefore not enough to determine whether teacher guidance benefits the current student. Efective supervision should instead emphasize guidance that the current student can translate into better final task outcomes. Accordingly, we propose Outcome-Guided On-Policy Distillation (OG-OPD), which applies trajectory-relative weighting to teacher supervision and calibrates these weights using final task outcomes from paired student continuations. This calibration selectively strengthens supervision on the student’s original trajectories at turns where teacher guidance benefits the current student. Across ALFWorld, ScienceWorld, and WebShop, OG-OPD consistently outperforms baselines under diverse settings. It improves task success rates by 3.6–17.7 percentage points over vanilla OPD and by up to 7.0 percentage points over the strongest baseline.

![](images/c0ac195d99a6cc420e63eda9ef291eda860222c7a898ddfea554a245cb37b14f.jpg)  
Figure 1 | Illustration of the supervision–benefit mismatch. From the same state and interaction history, one branch takes the student’s original action and the other a teacher-proposed action. Both branches then continue interacting under the same frozen student policy. Moving toward the teacher’s preferred action does not necessarily improve the current student’s final task outcome.

## 1. Introduction

On-policy distillation (OPD) trains students on their own trajectories (Agarwal et al., 2024; Lin et al., 2020) with dense teacher supervision (Lu & Thinking Machines Lab, 2025; Fu et al., 2026). Learning at student-visited states helps reduce the distribution mismatch between training and inference (Bengio et al., 2015; Ranzato et al., 2016). At each turn, the teacher provides token-level probability feedback on the student’s response (Jin et al., 2026; Ko et al., 2024; Jia et al., 2026), encouraging the student to narrow teacher–student token-level distributional gaps (Gu et al., 2024; Jang et al., 2026; Shao et al., 2026). However, local teacher–student gaps alone are not enough to determine whether teacher guidance benefits the current student. This limitation is especially important for multi-turn autonomous agents, where teacher supervision spans successive environment interactions (Wang et al., 2026; Liao et al., 2026). Actions at one turn shape subsequent observations (Yao et al., 2023) and the states in which later decisions are made (Ross et al., 2011). Errors can compound across successive turns (Ross & Bagnell, 2010; Li et al., 2026). Efective supervision must therefore consider both where to intervene and what the current student achieves afterward.

Against this background, existing agentic OPD methods organize teacher supervision across turns through rollout construction and teacher intervention. Some progressively extend student rollouts or shorten prefixes of successful teacher trajectories (Wang et al., 2026), giving the student control over more turns as training progresses. Others interleave teacher and student turns within a trajectory while gradually reducing the probability of teacher intervention (Li et al., 2026). For more targeted guidance, recent work uses teacher– student token-level distributional gaps to identify promising intervention points and steer subsequent student behavior toward the teacher (Chen et al., 2026). Yet, does teacher guidance at these turns actually improve the current student’s final task outcome?

To examine this relationship, we conduct an empirical analysis on ScienceWorld (Wang et al., 2022) and WebShop (Yao et al., 2022). Our findings reveal a supervision–benefit mismatch: large gaps can be benign, while small gaps can be outcome-critical. A teacher-proposed action can change subsequent states and decisions, but its benefit depends on the current student’s ability to complete the task from the resulting state. Assessing this benefit requires comparing the current student’s final task outcomes after its original action and after the teacher action. Local gaps remain useful for locating potential intervention points, while this comparison provides evidence for deciding which turns warrant stronger teacher supervision. Figure 1 shows a schematic, with details in Section 3.1.

Accordingly, we propose Outcome-Guided On-Policy Distillation (OG-OPD). First, OG-OPD uses trajectoryrelative weighting to allocate teacher supervision and identifies candidate turns based on the current gap and an abrupt increase from the preceding turn. It then calibrates these weights using final task outcomes from paired student continuations. When the teacher action improves the current student’s final task outcome, OG-OPD selectively strengthens supervision at the corresponding turn on the student’s own trajectory. Otherwise, it retains the pre-calibration weight. Teacher responses and their continuations are used only for calibration, not as additional distillation targets.

We evaluate OG-OPD on ALFWorld (Shridhar et al., 2021), ScienceWorld, and WebShop under diferent teacher–student configurations. OG-OPD achieves the highest mean task success rate among the compared methods in every evaluated setting. On WebShop, for example, the Qwen3-1.7B student achieves a 45.7% success rate, compared with 28.0% for vanilla OPD and 38.7% for the strongest competing baseline. These gains extend to higher task scores on both ScienceWorld and WebShop. OG-OPD also requires fewer interaction turns than vanilla OPD in most comparisons.

Our main contributions are as follows. (1) Supervision–benefit mismatch. Moving toward the teacher does not necessarily benefit the current student. Across the two benchmarks, teacher harm averages 10.8% of large-gap decisions, exceeding teacher rescue at 8.8%. (2) Outcome-Guided On-Policy Distillation. OG-OPD builds on trajectory-relative weighting and uses task outcomes to decide where to strengthen teacher supervision. (3) Broad validation. Experiments on three benchmarks with diferent teacher–student settings show consistent gains in task success over baselines.

## 2. Preliminaries

## 2.1. Multi-turn Interaction

At turn $k ,$ the frozen rollout student $p _ { \circ }$ generates response $\boldsymbol { z } _ { k }$ from history $c _ { k }$ . The action parser $\mathcal { A }$ maps $\boldsymbol { z } _ { k }$ to action $u _ { k } ,$ which changes environment state $s _ { k }$ according to transition law ${ \sf T } _ { \mathrm { e n v } }$ :

$$
\begin{array} { r } { z _ { k } \sim p _ { \circ } ( \cdot  { \mid } c _ { k } ) , \qquad u _ { k } = \mathcal { A } ( z _ { k } ) , \qquad } \\ { ( s _ { k + 1 } , o _ { k + 1 } ) \sim \mathsf { T } _ { \mathrm { e n v } } ( \cdot  { \mid } s _ { k } , u _ { k } ) , \qquad c _ { k + 1 } = c _ { k } \oplus ( z _ { k } , o _ { k + 1 } ) , } \end{array}\tag{1}
$$

Here $s _ { k }$ need not be fully observable and ⊕ appends the response and observation. Autoregressive generation gives $\begin{array} { r } { p _ { \circ } ( z _ { k } \mid c _ { k } ) = \prod _ { j } p _ { \circ } ( z _ { k , j } \mid \nu _ { k , j } ) } \end{array}$ , with $\upsilon _ { k , j } = ( c _ { k } , z _ { k , < j } )$ . Trajectory $\zeta$ ends at termination or the interaction limit. Its final task outcome ${ \sf S } ( \zeta ) \in \{ 0 , 1 \}$ indicates benchmark success. Turn sums cover $0 \leq k < K \zeta$ , where $K _ { \zeta }$ counts the interaction turns in trajectory �.

## 2.2. On-policy Distillation

OPD provides dense teacher supervision on the student’s own trajectories (Agarwal et al., 2024). For fixed teacher $q _ { \phi }$ and trainable student $p _ { \theta }$ , reverse KL distillation (Gu et al., 2024) minimizes

$$
\begin{array} { c } { { \mathcal { T } _ { \mathrm { { r K L } } } ( \theta ) = \mathbb { E } _ { \upsilon \sim d _ { p _ { \mathrm { o } } } } \left[ D _ { \mathrm { K L } } \bigl ( p _ { \theta } ( \cdot \mid \upsilon ) \big \| q _ { \phi } ( \cdot \mid \upsilon ) \bigr ) \right] , } } \\ { { D _ { \mathrm { K L } } \bigl ( p _ { \theta } ( \cdot \mid \upsilon ) \big \| q _ { \phi } ( \cdot \mid \upsilon ) \bigr ) = \displaystyle \sum _ { x \in \mathcal { X } } p _ { \theta } ( x \mid \upsilon ) \log \frac { p _ { \theta } ( x \mid \upsilon ) } { q _ { \phi } ( x \mid \upsilon ) } , } } \end{array}\tag{2}
$$

where $\chi$ is the vocabulary. The token context distribution $d _ { p _ { \circ } }$ is sampled under the frozen rollout student $p _ { \circ }$ and held fixed while the trainable student $p _ { \theta }$ is optimized within each update.

For each sampled token $\begin{array} { r } { z _ { k , j } , } \end{array}$ the teacher supervision and importance ratio are

$$
\psi _ { k , j } = \log { \frac { q _ { \phi } ( z _ { k , j } \mid \nu _ { k , j } ) } { p _ { \circ } ( z _ { k , j } \mid \nu _ { k , j } ) } } , \qquad \rho _ { k , j } ( \theta ) = { \frac { p _ { \theta } ( z _ { k , j } \mid \nu _ { k , j } ) } { p _ { \circ } ( z _ { k , j } \mid \nu _ { k , j } ) } } .\tag{3}
$$

At fixed $\nu ,$ the expected signed log probability ratio under $p _ { \circ }$ equals the negative reverse KL:

$$
\mathbb { E } _ { \boldsymbol { x } \sim p _ { \circ } ( \cdot \vert \boldsymbol { \nu } ) } \left[ \log \frac { q _ { \phi } ( \boldsymbol { x } \mid \boldsymbol { \nu } ) } { p _ { \circ } ( \boldsymbol { x } \mid \boldsymbol { \nu } ) } \right] = - D _ { \mathrm { K L } } \big ( p _ { \circ } ( \cdot \mid \boldsymbol { \nu } ) \| q _ { \phi } ( \cdot \mid \boldsymbol { \nu } ) \big ) .\tag{4}
$$

Positive $\psi _ { k , j }$ favors a higher sampled token probability, while negative values favor a lower one.

Let $M _ { k } ( \zeta )$ index generated tokens with valid teacher scores, excluding prompts, observations, and padding, and let $\begin{array} { r } { Z = \sum _ { \zeta \in \mathcal { D } } \sum _ { k } | \mathcal { M } _ { k } ( \zeta ) | } \end{array}$ count them in batch D. The clipped policy surrogate (Schulman et al., 2017), defined in Appendix $\mathsf { A } ,$ gives the turn-level and batch losses to minimize:

$$
\begin{array} { l } { { \displaystyle Q _ { k } ( \theta ; \zeta ) = \sum _ { j \in \mathcal { M } _ { k } ( \zeta ) } \ell _ { \mathrm { p o l } } \bigl ( \rho _ { k , j } ( \theta ) , s \mathrm { g } [ \psi _ { k , j } ] \bigr ) \ : , } } \\ { { \displaystyle \mathcal { L } _ { \mathrm { O P D } } ( \theta ) = \frac { 1 } { Z } \sum _ { \zeta \in \mathcal { D } } \sum _ { k } Q _ { k } ( \theta ; \zeta ) . } } \end{array}\tag{5}
$$

Here sg[·] holds $\psi _ { k , j }$ fixed during diferentiation. OG-OPD changes only the turn-level weights.

We quantify the magnitude of teacher–student token-level distributional gaps at each turn using

$$
\chi _ { k } = \frac { 1 } { | { \cal M } _ { k } ( \zeta ) | } \sum _ { j \in { \cal M } _ { k } ( \zeta ) } | \psi _ { k , j } | , \qquad k \in \mathcal { V } ( \zeta ) .\tag{6}
$$

Here $\mathcal { V } ( \zeta ) = \{ k : | { \cal M } _ { k } ( \zeta ) | > 0 \}$ retains original turn indices. The sampled gap magnitude $\chi _ { k }$ is distinct from signed token-level update directions and full-vocabulary KL divergence.

## 3. Outcome-Guided On-Policy Distillation

We first present an empirical analysis of the supervision–benefit mismatch. Motivated by these findings, we propose OG-OPD, building on trajectory-relative weighting and candidate selection. Its core component, outcome-based calibration, uses final task outcomes from paired student continuations to selectively strengthen teacher supervision on the student’s own trajectories.

## 3.1. Empirical Analysis

We analyze paired student continuations on ScienceWorld and WebShop using a fixed Qwen3-32B teacher and Qwen3-1.7B student. At sampled student-visited states, one branch takes the student’s original action and the other the teacher action, with both starting from the same state and history and continuing under the same frozen student policy. Within each benchmark, the highest and lowest of five quantile groups define the large-gap and small-gap groups. Each decision is classified from five matched trials. Appendix B describes state sampling and the full outcome classification rules.

Figure 2 summarizes final task outcomes across gap groups. In the large-gap group, benign gaps account for an average of 5.4% of decisions, where both the student’s original action and the teacher action lead to success. Teacher rescue and teacher harm occur in 8.8% and 10.8% of decisions, respectively. These findings indicate that a large gap can be benign, while the teacher action can either improve or worsen the final task outcome. In the small-gap group, 7.6% of decisions are outcome-critical, with the two branches leading to diferent final task outcomes. Together, these results reveal the supervision– benefit mismatch. Beyond these averages, the relative frequencies of teacher rescue and teacher harm vary across benchmarks. Within the large-gap group, teacher harm occurs more often than teacher rescue on WebShop, whereas the reverse holds on Science-World. Appendix B.1 further examines this pattern under gap magnitude ranking.

![](images/5bc30b801400e91253d4436dd0c3dd115088c06b9637a9f09f9d3563df424fc4.jpg)  
Figure 2 | Empirical analysis of final task outcomes. Bars show decision proportions by outcome category within each benchmark’s gap groups. Error bars indicate 95% decision-level bootstrap confidence intervals.

Beyond gap magnitude, we examine the role of the continuation policy by comparing teacher and student continuations after the same teacher action. In an average of 20.6% of large-gap decisions across the two benchmarks, the teacher frequently completes the task, while the current student rarely succeeds from the same resulting state. We refer to these cases as teacher-only completion. This suggests that the value of teacher guidance depends on the current student’s ability to continue from the resulting state and complete the remaining task successfully. These findings motivate OG-OPD to use paired student continuations when deciding where to strengthen teacher supervision.

## 3.2. Trajectory-Relative Weighting and Candidate Selection

Figure 3 illustrates trajectory-relative weighting and candidate selection. For the gap $\chi _ { k }$ in Equation $^ { 6 , }$ we define its logarithmic magnitude $\nu _ { k }$ and change from the preceding turn $v _ { k }$ as

$$
\nu _ { k } = \log ( \varepsilon _ { 0 } + \chi _ { k } ) , \qquad v _ { k } = \nu _ { k } - \nu _ { k - 1 } = \log \frac { \varepsilon _ { 0 } + \chi _ { k } } { \varepsilon _ { 0 } + \chi _ { k - 1 } } ,\tag{7}
$$

where $\varepsilon _ { 0 } > 0$ stabilizes the logarithm. The change $v _ { k }$ requires adjacent valid turns $k , k - 1 \in \mathcal { V } ( \zeta )$

Since $\chi _ { k }$ averages $| \psi _ { k , j } |$ , larger gaps imply larger supervision coeficients on average. Trajectory-relative weighting moderates this variation using the first valid turn � = min $\mathcal { V } ( \zeta )$ as a reference. Each valid turn in the student’s own trajectory receives the pre-calibration weight

$$
\beta _ { k } = \left\{ \begin{array} { l l } { 1 , } & { k = k _ { 0 } , } \\ { \operatorname* { m i n } \bigl \{ \beta _ { \mathrm { m a x } } , \exp ( \nu _ { k _ { 0 } } - \nu _ { k } ) \bigr \} , } & { k > k _ { 0 } , } \end{array} \right. \qquad \beta _ { \mathrm { m a x } } \geq 1 .\tag{8}
$$

Gaps above the reference reduce the weight, while gaps below it increase the weight up to $\beta _ { \mathrm { m a x } }$ . Using Equation 6, the average magnitude of the weighted token-level teacher supervision is

$$
{ \frac { 1 } { | \mathcal { M } _ { k } ( \zeta ) | } } \sum _ { j \in \mathcal { M } _ { k } ( \zeta ) } | \beta _ { k } \psi _ { k , j } | = \beta _ { k } \chi _ { k } = \operatorname* { m i n } \Biggl \{ \beta _ { \operatorname* { m a x } } \chi _ { k } , { \frac { ( \varepsilon _ { 0 } + \chi _ { k _ { 0 } } ) \chi _ { k } } { \varepsilon _ { 0 } + \chi _ { k } } } \Biggr \} .\tag{9}
$$

With $\varepsilon _ { 0 }$ negligible relative to the gaps and the cap inactive, $\beta _ { k } \chi _ { k } \approx \chi _ { k _ { 0 } }$ . This sets a common supervision scale across valid turns in the trajectory, but does not determine whether teacher guidance benefits the current student in terms of its final task outcome under its continuation policy.

![](images/3d0f05ceb5e60e779686245b617f05b47b604f0b09c23dcd01d8bba6d7c945a5.jpg)  
Figure 3 | Overview of Outcome-Guided On-Policy Distillation. Trajectory-relative weighting assigns supervision weights, while gap magnitude and abrupt increases identify candidate turns. Outcome-based calibration compares paired student continuations and strengthens supervision on the student’s original trajectory when teacher guidance improves the final task outcome.

To limit additional environment execution, we evaluate paired student continuations only at selected turns. Candidates require a suficiently large gap and, after the initial turn, a positive increase meeting its threshold. Final task outcomes determine whether to strengthen supervision. Let $\mathcal H ( \zeta )$ contain turns with valid scores and replay information. We define the candidate set as

$$
\mathcal { K } ( \boldsymbol { \zeta } ) = \left\{ \boldsymbol { k } \in \mathcal { H } ( \boldsymbol { \zeta } ) \big \vert \nu _ { k } \geq \vartheta _ { \nu } \wedge \big [ \boldsymbol { k } = 0 \vee \big ( \boldsymbol { v } _ { k } > 0 \wedge \boldsymbol { v } _ { k } \geq \vartheta _ { v } \big ) \big ] \right\} .\tag{10}
$$

Thresholds are batch quantiles with equal total weight per contributing trajectory. Only positive valid changes enter $\vartheta _ { v } .$ The magnitude-only exception applies to $k = 0 \in \mathcal { H } ( \zeta )$ , not a later first valid turn $k _ { 0 } > 0$ . Within a bounded proposal budget, we check candidates chronologically and select the first replay-valid turn $k ^ { \star }$ whose teacher response yields a legal action diferent from the student’s original action. Only this turn undergoes paired student continuations for outcome-based calibration in Section 3.3. Appendix C gives further details of this execution procedure.

## 3.3. Outcome-Based Calibration

Outcome-based calibration increases the selected candidate turn’s pre-calibration weight only when the current student fails with its original action but succeeds with the teacher action.

Let $z _ { k ^ { \star } } ^ { S } = z _ { k ^ { \star } }$ denote the student’s original response and $z _ { k ^ { \star } } ^ { T }$ the selected teacher response sampled from $q _ { \phi } ( \cdot \mid c _ { k ^ { \star } } )$ . We restore state $s _ { k ^ { \star } }$ and history $c _ { k ^ { \star } }$ from the student’s own trajectory. Define $\mathcal { U } ( s , c , z ; p _ { \circ } , \varsigma )$ to execute $\mathcal { A } ( z )$ , retain � in history, and continue under the frozen current student $p _ { \circ }$ with seed schedule $\varsigma .$ The paired student continuations are then generated as follows

$$
\begin{array} { r l } & { \tilde { \zeta } ^ { S } = \mathcal { U } ( s _ { k ^ { \star } } , c _ { k ^ { \star } } , z _ { k ^ { \star } } ^ { S } ; p _ { \circ } , \varsigma ) , } \\ & { \tilde { \zeta } ^ { T } = \mathcal { U } ( s _ { k ^ { \star } } , c _ { k ^ { \star } } , z _ { k ^ { \star } } ^ { T } ; p _ { \circ } , \varsigma ) . } \end{array}\tag{11}
$$

Superscript $T$ denotes the teacher response, not teacher continuation. Both use the same frozen current student, initial state, history, decoding settings, remaining interaction budget, and seed schedule. Responses may change subsequent states, so we test whether the current student can complete the task after taking the teacher action, rather than whether the teacher can. Training uses one pair at the selected turn, whereas the empirical analysis repeats matched trials per sampled decision.

Let ${ \mathcal { D } } _ { \operatorname { p a i r } } \subseteq { \mathcal { D } }$ contain trajectories with valid completed paired student continuations. Define

$$
\Delta _ { \zeta } = { \sf S } ( \tilde { \zeta } ^ { T } ) - { \sf S } ( \tilde { \zeta } ^ { S } ) \in \{ - 1 , 0 , 1 \} .\tag{12}
$$

Here $\Delta \boldsymbol { \zeta } = 1$ denotes failure under the original action and success under the teacher action. $\Delta \zeta = - 1$ denotes the reverse, and $\Delta \boldsymbol { \zeta } = 0$ denotes no change in final task outcome. Only the positive case permits additional distillation weight at the selected candidate turn.

$$
\begin{array} { r } { \mathsf { g } _ { \zeta } = \mathsf { S } ( \tilde { \zeta } ^ { T } ) \big ( 1 - \mathsf { S } ( \tilde { \zeta } ^ { S } ) \big ) = \mathbb { I } \big [ \Delta _ { \zeta } = 1 \big ] = \big [ \Delta _ { \zeta } \big ] _ { + } . } \end{array}\tag{13}
$$

Success with the teacher action provides positive execution evidence only when the original action fails, excluding cases where the current student succeeds under both responses.

The calibrated weight $\omega _ { k }$ adds the outcome-based increment $\gamma _ { \zeta }$ to the pre-calibration weight at the selected turn. For weight floor $\beta _ { \mathord { \uparrow } } > 0$ and $[ x ] _ { + } = \operatorname* { m a x } \{ x , 0 \}$ , the resulting weights are

$$
\gamma _ { \zeta } = \mathsf { g } _ { \zeta } \left[ \beta _ { \uparrow } - \beta _ { k ^ { \star } } \right] _ { + } , \qquad \mathsf { \Gamma } _ { \boldsymbol { \omega } _ { k } } = \left\{ \begin{array} { l l } { \beta _ { k } + \gamma _ { \zeta } , } & { k = k ^ { \star } , } \\ { \beta _ { k } , } & { k \neq k ^ { \star } . } \end{array} \right.\tag{14}
$$

When $\mathsf { g } _ { \boldsymbol { \zeta } } = 1$ , the calibrated weight is max $\{ \beta _ { k ^ { \star } } , \beta _ { \uparrow } \}$ . Paired outcomes determine whether to add supervision, while the floor and pre-calibration weight determine the increase. Positive execution evidence can thus strengthen supervision at a previously downweighted turn. Otherwise, or without valid paired student continuations, $\gamma _ { \zeta } = 0$ and pre-calibration weights remain unchanged. Since $\beta _ { \mathrm { m a x } }$ is not reapplied, a higher floor can raise the calibrated weight above the pre-calibration cap.

For a fixed batch D of the student’s own trajectories, calibrated weights scale the original response losses $\alpha _ { k } ( \theta ; \zeta )$ from Section 2.2. Teacher responses and paired student continuations are used only for outcome-based calibration, not as additional distillation targets. Keeping the original token normalization � unchanged, we write the resulting weighted training objective as

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { \mathrm { O G - O P D } } ( \boldsymbol { \theta } ) = \frac { 1 } { Z } \sum _ { \zeta \in \mathcal { D } } \displaystyle \sum _ { k \in \mathcal { V } ( \zeta ) } s g [ \omega _ { k } ] Q _ { k } ( \boldsymbol { \theta } ; \boldsymbol { \zeta } ) } \\ { \displaystyle = \frac { 1 } { Z } \sum _ { \zeta \in \mathcal { D } } \displaystyle \sum _ { k \in \mathcal { V } ( \zeta ) } s g [ \beta _ { k } ] Q _ { k } ( \boldsymbol { \theta } ; \boldsymbol { \zeta } ) + \frac { 1 } { Z } \sum _ { \zeta \in \mathcal { D } _ { \mathrm { p a i r } } } s g [ \gamma _ { \zeta } ] Q _ { k ^ { \star } } ( \boldsymbol { \theta } ; \boldsymbol { \zeta } ) . } \end{array}\tag{15}
$$

The first term retains dense teacher supervision across valid turns using their pre-calibration weights. The second supplements trajectory-relative weighting with supervision on the selected turn’s original student response only when $\gamma _ { \zeta } > 0$ , leaving other turns’ pre-calibration weights unchanged.

For this fixed batch, holding recorded $\psi _ { k , j }$ and weights fixed during diferentiation gives

$$
\ell _ { \mathrm { p o l } } \bigl ( \rho _ { k , j } ( \theta ) , s g [ \omega _ { k } \psi _ { k , j } ] \bigr ) = s g [ \omega _ { k } ] \ell _ { \mathrm { p o l } } \bigl ( \rho _ { k , j } ( \theta ) , s g [ \psi _ { k , j } ] \bigr ) , \qquad \omega _ { k } \geq 0 .\tag{16}
$$

Thus, turn weighting scales token-level teacher supervision while preserving the directions defined in Section 2.2. With the original token mask $M _ { k } ( \zeta )$ and normalization � unchanged, outcome-based calibration adds a gradient contribution only from the selected candidate turns

$$
\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { O G - O P D } } ( \boldsymbol { \theta } ) - \nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { b a s e } } ( \boldsymbol { \theta } ) = \frac { 1 } { Z } \sum _ { \boldsymbol { \zeta } \in \mathcal { D } _ { \mathrm { p a i r } } } s \mathrm { g } [ \gamma _ { \boldsymbol { \zeta } } ] \nabla _ { \boldsymbol { \theta } } Q _ { \boldsymbol { k } ^ { \star } } ( \boldsymbol { \theta } ; \boldsymbol { \zeta } ) .\tag{17}
$$

Paired outcomes provide evidence for strengthening supervision at the selected turn, while signed token-level teacher feedback determines the update directions on the original student response. Since � counts valid original tokens rather than weights, other turns’ loss coeficients remain unchanged. Appendix A provides the derivation and relates observed final task outcomes to training updates.

## 4. Experiments

## 4.1. Experimental Setup

Benchmarks and Models. We evaluate OG-OPD on ALFWorld, ScienceWorld, and WebShop, following TCOD (Wang et al., 2026) for environment configurations, prompt templates, and ALFWorld evaluation collections. All experiments use Qwen3 models (Yang et al., 2025). A Qwen3-32B teacher is paired with a Qwen3-1.7B student on all benchmarks, while ALFWorld and WebShop additionally use a Qwen3-8B-RL teacher with a Qwen3-4B student. Each RL teacher is trained with GiGPO (Feng et al., 2025) on its benchmark. We report

Table 1 | Main results with the Qwen3-32B teacher and Qwen3-1.7B student. Trained methods report mean ± standard deviation across three training seeds. SR (%) denotes success rate, with ALFWorld Overall pooling all splits. Score uses a 0–100 scale, and Rounds averages interactions per task. Dark/light green mark the best/second-best means among methods, with the best in bold.
<table><tr><td></td><td colspan="4">ALFWorld</td><td colspan="4">ScienceWorld</td><td colspan="3">WebShop</td></tr><tr><td>Method</td><td>Seen SR↑</td><td>Unseen SR ↑</td><td>Hard SR↑</td><td>Overall SR↑</td><td>Rounds↓</td><td>SR ↑</td><td></td><td>Score ↑ Rounds↓</td><td>SR↑</td><td></td><td>Score ↑ Rounds↓</td></tr><tr><td colspan="10">Qwen3-32B teacher → Qwen3-1.7B student</td></tr><tr><td>Student (zero-shot)</td><td>7.1</td><td>8.2</td><td>0.0</td><td>5.3</td><td>19.8</td><td>0.2</td><td>3.8</td><td>17.0</td><td>27.0</td><td>24.2</td><td>8.1</td></tr><tr><td>Teacher (zero-shot)</td><td>37.1</td><td>35.1</td><td>9.9</td><td>28.1</td><td>18.5</td><td>28.2</td><td>53.9</td><td>12.0</td><td>48.0</td><td>47.3</td><td>4.9</td></tr><tr><td>Vanilla OPD</td><td>26.7±3.0 27.6±3.3</td><td></td><td>4.1±1.4</td><td>20.1±2.1</td><td>17.7±0.4</td><td>13.0±1.4 48.1±1.2</td><td></td><td>12.6±0.3</td><td>28.0±2.6 27.5±2.6</td><td></td><td>6.8±0.2</td></tr><tr><td>TCOD-F2B</td><td>23.8±3.2</td><td>27.9±3.0</td><td>7.4±0.8</td><td>20.2±2.0</td><td>19.2±0.3</td><td>13.7±0.7</td><td>50.0±0.5</td><td>12.5±0.2</td><td>30.7±1.5</td><td>25.9±1.8</td><td>7.7±0.2</td></tr><tr><td>TCOD-B2F</td><td>31.0±3.6</td><td>33.6±3.3</td><td>5.8±1.4</td><td>24.1±2.3</td><td>18.5±0.3</td><td>14.3±0.5</td><td>48.3±0.6</td><td>11.9±0.2</td><td>33.7±3.1 33.5±3.4</td><td></td><td>6.1±0.3</td></tr><tr><td>Guided-OPD</td><td>26.7±5.1</td><td>28.4±4.9</td><td>5.8±1.7</td><td>20.8±3.3</td><td>18.4±0.5</td><td>15.8±0.8</td><td>45.2±2.0</td><td>12.4±0.4</td><td>34.3±9.6 30.8±8.9</td><td></td><td>5.9±0.4</td></tr><tr><td>FutureBridge-OPD</td><td>26.7±3.3</td><td>29.4±3.0</td><td>6.6±0.8</td><td>21.4±2.1</td><td>18.1±0.3</td><td>12.8±0.7</td><td>42.6±0.5</td><td>11.7±0.2</td><td>38.7±1.5</td><td>38.1±1.6</td><td>5.7±0.2</td></tr><tr><td>OG-OPD</td><td>31.7±2.1</td><td>32.3±2.3</td><td>6.6±0.8</td><td>24.2±1.5</td><td>18.0±0.2</td><td>16.6±0.6 52.0±0.7</td><td></td><td>11.6±0.2</td><td>45.7±1.2 44.3±1.5</td><td></td><td>5.6±0.2</td></tr></table>

Table 2 | Main results with the Qwen3-8B-RL teacher and Qwen3-4B student. Each teacher is trained with GiGPO on its benchmark. Results for trained methods are mean ± standard deviation across three training seeds. SR (%), Score (0–100), and Rounds follow Table 1. Dark/light green mark the best/second-best means among compared methods, with the best in bold.
<table><tr><td rowspan="2">Method</td><td colspan="5">ALFWorld</td><td colspan="3">WebShop</td></tr><tr><td>Seen SR↑</td><td>Unseen SR↑</td><td>Hard SR↑</td><td>Overall SR↑</td><td>Rounds↓</td><td>SR↑</td><td>Score ↑</td><td>Rounds↓</td></tr><tr><td colspan="9">Qwen3-8B-RL teacher → Qwen3-4B student</td></tr><tr><td>Student (zero-shot)</td><td>32.1</td><td>26.9</td><td>9.9</td><td>23.5</td><td>17.5</td><td>27.0</td><td>26.5</td><td>8.0</td></tr><tr><td>Teacher (zero-shot)</td><td>81.4</td><td>79.1</td><td>33.9</td><td>66.1</td><td>10.3</td><td>54.0</td><td>56.4</td><td>4.4</td></tr><tr><td>Vanilla OPD</td><td>71.4±3.7</td><td>67.4±3.5</td><td>20.9±2.4</td><td>54.6±2.7</td><td>11.3±0.3</td><td>51.7±1.5</td><td>55.2±2.2</td><td>4.2±0.1</td></tr><tr><td>TCOD-F2B</td><td>77.9±1.4</td><td>71.9±1.7</td><td>28.1±1.4</td><td>60.6±1.2</td><td>11.5±0.2</td><td>55.3±2.3</td><td>57.8±2.8</td><td>4.0±0.1</td></tr><tr><td>TCOD-B2F</td><td>77.1±2.1</td><td>75.6±1.9</td><td>31.4±1.4</td><td>62.6±1.6</td><td>11.3±0.2</td><td>54.7±1.2</td><td>56.6±1.5</td><td>4.1±0.1</td></tr><tr><td>Guided-OPD</td><td>74.3±3.8</td><td>73.4±3.5</td><td>19.3±2.5</td><td>57.1±2.8</td><td>11.2±0.3</td><td>52.3±4.6</td><td>53.7±3.6</td><td>4.3±0.2</td></tr><tr><td>FutureBridge-OPD</td><td>78.6±2.9</td><td>77.1±2.8</td><td>28.4±1.9</td><td>62.7±2.1</td><td>11.5±0.2</td><td>54.7±0.6</td><td>56.9±1.1</td><td>4.1±0.1</td></tr><tr><td>OG-OPD</td><td>82.1±1.9</td><td>76.4±1.9</td><td>30.0±1.3</td><td>64.2±1.4</td><td>11.1±0.2</td><td>56.3±1.2</td><td>57.9±1.3</td><td>3.9±0.1</td></tr></table>

success rate (SR), task score (Score), and mean interaction rounds (Rounds). Appendices D.1–D.2 detail teacher training and evaluation.

Training Setup. All methods are evaluated at training step 200, with results in the main, ablation, and eficiency tables averaged over three training seeds. Within each benchmark and model setting, all methods share the same training tasks, student initialization, teacher checkpoints, interaction protocol, and evaluation protocol, while retaining their original curriculum and prefix designs. Additional execution details, including action parsing and state replay, are provided in Appendix C.

Baselines. We compare OG-OPD with five agentic distillation baselines. Vanilla OPD provides dense teacher supervision on the student’s own trajectories (Agarwal et al., 2024). TCOD-F2B progressively increases student rollout depth, whereas TCOD-B2F shortens successful teacher prefixes (Wang et al., 2026). Guided-OPD interleaves teacher and student turns while gradually increasing student control (Li et al., 2026). FutureBridge-OPD identifies large-gap positions, evaluates teacher guidance using teacher preference over student continuations, and adds accepted teacher responses as distillation targets (Chen et al., 2026). Zero-shot teachers and students serve as references.

## 4.2. Main Results and Training Dynamics

OG-OPD achieves the highest SR among the compared methods in every evaluated setting, with Overall SR reported for ALFWorld, as shown in Tables 1 and 2. Relative to vanilla OPD, the gains range from 3.6 to 17.7

![](images/eaeb5e4ddfaccf5de7a946af1802f326bce484471c97fc6eea0e16d5ef37e9db.jpg)  
Figure 4 | Training dynamics on WebShop. The top row shows the Qwen3-32B teacher with the Qwen3-1.7B student. The bottom row shows the Qwen3-8B-RL teacher with the Qwen3-4B student.

Table 3 | Ablation results. The first group tests outcome-based calibration by removing it or directly upweighting candidate turns without verification. The second retains calibration and teacher response evaluation but replaces trajectory-relative weighting with uniform unit weights or gap-based candidate selection with random turn sampling. Trained methods report mean ± standard deviation across three training seeds. Bold marks each column’s best mean among trained methods.

<table><tr><td></td><td colspan="3">ScienceWorld</td><td colspan="3">WebShop</td></tr><tr><td>Method</td><td>SR ↑</td><td>Score ↑</td><td>Rounds↓</td><td>SR ↑</td><td>Score ↑</td><td>Rounds ↓</td></tr><tr><td colspan="7">Qwen3-32B teacher → Qwen3-1.7B student</td></tr><tr><td>Student (zero-shot) Teacher (zero-shot)</td><td>0.2</td><td>3.8</td><td>17.0</td><td>27.0</td><td>24.2</td><td>8.1</td></tr><tr><td>Vanilla OPD</td><td>28.2</td><td>53.9</td><td>12.0</td><td>48.0</td><td>47.3</td><td>4.9</td></tr><tr><td></td><td>13.0±1.4</td><td>48.1±1.2</td><td>12.6±0.3</td><td>28.0±2.6</td><td>27.5±2.6</td><td>6.8±0.2</td></tr><tr><td colspan="7">Outcome-Based Calibration</td></tr><tr><td>w/o outcome-based calibration</td><td>3.8±1.1</td><td>38.5±1.9</td><td>12.5±0.3</td><td>32.0±2.6</td><td>30.5±2.5</td><td>6.4±0.2</td></tr><tr><td>Upweighting without verification</td><td>14.9±0.9</td><td>49.9±1.1</td><td>12.3±0.2</td><td>35.0±2.0</td><td>34.5±2.1</td><td>6.1±0.2</td></tr><tr><td>Weighting and Candidate Selection</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="7"></td></tr><tr><td>w/o trajectory-relative weighting Random turn sampling</td><td>5.5±1.3 15.1±0.8</td><td>38.1±2.1 51.5±0.9</td><td>12.9±0.3 12.1±0.2</td><td>33.3±2.5 37.7±1.5</td><td>31.9±2.8 35.6±2.0</td><td>6.1±0.2</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>5.9±0.2</td></tr><tr><td>OG-OPD</td><td>16.6±0.6</td><td>52.0±0.7</td><td>11.6±0.2</td><td>45.7±1.2</td><td>44.3±1.5</td><td>5.6±0.2</td></tr></table>

percentage points. With the Qwen3-32B teacher and Qwen3-1.7B student, OG-OPD reaches 45.7% SR on WebShop, compared with 38.7% for FutureBridge-OPD, the strongest competing baseline. The advantage also holds with the task-trained Qwen3-8B-RL teacher and Qwen3-4B student. On ALFWorld, OG-OPD achieves 64.2% Overall SR, surpassing FutureBridge-OPD at 62.7%. Both students also outperform vanilla OPD on Seen, Unseen, and Hard tasks, showing consistent gains in unfamiliar environments and on dificult tasks.

With the Qwen3-32B teacher and Qwen3-1.7B student, OG-OPD improves SR over vanilla OPD by 17.7 percentage points on WebShop, compared with 3.6 points on ScienceWorld. This contrast is consistent with the empirical analysis in Section 3.1, which shows that teacher harm is more frequent than teacher rescue within WebShop’s large-gap group, while the reverse holds on ScienceWorld. The larger improvement on WebShop, together with this empirical finding, supports the motivation for checking whether teacher guidance benefits the current student before strengthening supervision.

The gains extend beyond the binary success criterion used for outcome-based calibration. Across all evaluated ScienceWorld and WebShop settings, OG-OPD achieves the highest Score and the lowest Rounds among the compared methods. For Qwen3-1.7B, ScienceWorld Score increases from 48.1 with vanilla OPD to 52.0, while Rounds decreases from 12.6 to 11.6. On WebShop, Score increases from 27.5 to 44.3, while Rounds decreases from 6.8 to 5.6. The Qwen3-4B student on WebShop shows the same pattern. Higher task success accompanies higher task scores and shorter interactions, even though neither Score nor Rounds is used to trigger additional upweighting.

On WebShop, OG-OPD shows earlier improvements in teacher likelihood and action parseability than vanilla OPD. With the Qwen3-32B teacher and Qwen3-1.7B student, Teacher NLL falls earlier and Parsed recovers sooner, as shown in Figure 4. These improvements accompany selective additional upweighting. Across both teacher–student configurations, candidate turns account for roughly one quarter of eligible turns, whereas fewer than 1% of recorded turns receive additional weight when pooled over training. Thus, earlier response-level improvements coexist with additional upweighting of only a small subset of turns, while dense teacher supervision is retained across the student’s own trajectories. These gains come at a per-step training cost of 1.42× vanilla OPD on ScienceWorld and 1.48× on WebShop with the Qwen3-1.7B student, representing a practical tradeof between training cost and task performance. Appendix E provides training time details.

## 4.3. Ablation Study

We evaluate outcome-based calibration, trajectory-relative weighting, and candidate selection using the Qwen3- 32B teacher and Qwen3-1.7B student, as shown in Table 3. We first test whether simpler weighting strategies can replace outcome-based calibration. The w/o outcome-based calibration variant retains the trajectory-relative weights without further adjustment, while Upweighting without verification directly increases the weight of the earliest gap-based candidate turn. Neither variant generates teacher responses or performs paired student continuations. Direct upweighting improves SR from 3.8% to 14.9% on ScienceWorld and from 32.0% to 35.0% on WebShop, but still falls short of OG-OPD at 16.6% and 45.7%, respectively. Increasing candidate turn weights improves SR on both benchmarks, but does not match the gains from the full outcome-based calibration procedure.

With outcome-based calibration retained, we next examine the role of trajectory-relative weighting. The w/o trajectory-relative weighting variant replaces the pre-calibration weights with uniform unit weights while keeping candidate selection, teacher response evaluation, and calibration unchanged. Relative to OG-OPD, SR drops by 11.1 percentage points on ScienceWorld and 12.4 points on WebShop. These results show that trajectory-relative weighting and outcome-based calibration are complementary, as removing either component substantially reduces performance.

Finally, we examine the efect of candidate turn selection. The Random turn sampling variant randomly selects eligible turns without replacement from trajectories containing at least one candidate turn, with teacher response evaluation and outcome-based calibration unchanged. Relative to OG-OPD, SR drops by 1.5 percentage points on ScienceWorld and 8.0 points on WebShop. Gap-based candidate selection thus helps outcome-based calibration focus on more informative turns.

## 5. Related Work

On-Policy Distillation for Multi-Turn Agents. Learning from student-generated responses addresses the training–inference distribution mismatch in autoregressive distillation (Agarwal et al., 2024). Subsequent research spans objective design and rollout construction. Objective-level studies investigate reverse KL distillation (Gu et al., 2024), adaptive intermediate distributions (Jang et al., 2026), and entropy-dependent objectives (Jin et al., 2026). Complementary work combines skew KL objectives with adaptive reuse of student-generated outputs (Ko et al., 2024). In multi-turn environments, temporal curricula regulate student rollout depth (Wang et al., 2026), while turn-level guidance schedules teacher participation within a rollout (Li et al., 2026). Prefix replay organizes student continuations using previously collected teacher trajectories (Liao et al., 2026). These approaches shape the states visited during training and the distributions used as learning targets.

Selective Supervision and Reweighting. In chain-of-thought distillation, learned token weights emphasize key reasoning tokens (Feng et al., 2024). For OPD, supervision strength is adjusted across tokens or steps using entropy and teacher–student token-level distributional gaps (Xu et al., 2026), or final task outcomes and perplexity (Zheng et al., 2026). A trajectory-relative weighting rule interprets these gaps as a reliability signal and attenuates supervision as their magnitude increases relative to a reference (Zhong et al., 2026). Although its overall objective includes trajectory rewards, the distillation weights depend on relative gap magnitudes rather than the task outcomes induced by teacher actions. The weighting rule therefore does not distinguish downweighted turns where teacher guidance would help the current student from those where it would not.

Downstream Evaluation and Teacher Guidance. In reasoning tasks, sampled continuations provide automatic step-level supervision (Wang et al., 2024) and Monte Carlo estimates for credit assignment (Kazemnejad et al., 2025). Policy optimization also uses discounted future KL to shape token-level advantages (Ma et al., 2026) and repeated-state action groups for step-level credit assignment (Feng et al., 2025). In agent distillation, Chen et al. (2026) evaluate teacher responses through increases in the proportion of teacher-preferred tokens in paired student continuations, retaining accepted teacher responses as additional distillation targets. OG-OPD likewise evaluates teacher guidance through paired student continuations, but difers in both the validation signal and how it enters training. It uses final task outcomes from paired student continuations to decide whether to strengthen teacher supervision at the selected turn of the student’s original trajectory, rather than adding accepted teacher responses as additional distillation targets for student policy optimization.

## 6. Conclusion

For multi-turn autonomous agents, teacher–student token-level distributional gaps are not enough to determine whether teacher guidance benefits the current student. Outcome-Guided On-Policy Distillation (OG-OPD) addresses this supervision–benefit mismatch by calibrating trajectory-relative weights with final task outcomes from paired student continuations while training on the student’s original trajectories. Across three benchmarks and two teacher–student pairings, OG-OPD achieves the highest mean task success rate among the compared methods in every evaluated setting.

## AI use statement

Generative AI tools were used to polish the paper and assist with experimental code development. They were not used for any other tasks requiring disclosure. All AI-assisted work was reviewed by the authors. AI-assisted code was reviewed and tested for correctness, and language edits were checked to ensure they preserved the intended technical content. We take responsibility for the final content of this work, including text, claims, or artifacts produced with the aid of generative AI.

## Ethics statement

We evaluate language agents in benchmark environments, without live purchases or physical deployments. Improved benchmark performance does not establish safety in real-world settings, where erroneous actions may cause consequences not captured by these environments. Deployment therefore requires separate safety evaluations and appropriate safeguards. All datasets, environments, and models remain subject to their respective licenses and terms of use.

## Reproducibility statement

We provide the source code in the supplementary material accompanying this submission. Section 3 presents the empirical analysis and the OG-OPD algorithm, with optimization details, diagnostic protocols, and implementation procedures in Appendices A–C. Appendix D.1 specifies the model configurations, and Appendix D.2 describes the evaluation metrics and aggregation across training seeds. The training cost measurement protocol is provided in Appendix E. Together, these materials document the method and experimental procedures needed to reproduce the reported results.

## References

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from self-generated mistakes. In International Conference on Learning Representations, 2024. URL https://proceedings.iclr.cc/paper\_ files/paper/2024/hash/5be69a584901a26c521c2b51e40a4c20-Abstract-Conference. html.

Samy Bengio, Oriol Vinyals, Navdeep Jaitly, and Noam Shazeer. Scheduled sampling for sequence prediction with recurrent neural networks. In Advances in Neural Information Process-

ing Systems, volume 28, 2015. URL https://proceedings.neurips.cc/paper/2015/hash/ e995f98d56967d946471af29d7bf99f1-Abstract.html.

Chishui Chen, Yaoyou Fan, Te Sun, Yi Yang, Chenghao Sun, Delin Mao, Hongbo Qiao, Zuowei Zhang, Junxi Wang, Chenxing Sun, Yangen Hu, Lu Pan, Xuyang Liu, and Linfeng Zhang. Look ahead before you distill: Future trajectory validation of teacher guidance for agentic on-policy distillation. arXiv preprint arXiv:2608.01953, 2026. URL https://arxiv.org/abs/2608.01953.

Kaituo Feng, Changsheng Li, Xiaolu Zhang, Jun Zhou, Ye Yuan, and Guoren Wang. Keypoint-based progressive chain-of-thought distillation for LLMs. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 13241–13255, 2024. URL https: //proceedings.mlr.press/v235/feng24e.html.

Lang Feng, Zhenghai Xue, Tingcong Liu, and Bo An. Group-in-group policy optimization for LLM agent training. In Advances in Neural Information Processing Systems, volume 38, pp. 46375–46408, 2025. doi: 10.52202/085713-1544. URL https://proceedings.neurips.cc/paper\_files/paper/2025/ hash/420c9f777c0b4f78d515e53cf74d58b2-Abstract-Conference.html.

Zixuan Fu, Bingxiang He, Yuxin Zuo, Haohuan Huang, Jinqian Zhang, Ruhang Xiao, Cheng Qian, Qinyu Luo, Huan-ang Gao, Yudong Wang, Zhiyuan Liu, Ning Ding, and Chaojun Xiao. Rethinking on-policy distillation of large language models II: One training example. arXiv preprint arXiv:2609.04172, 2026. URL https://arxiv.org/abs/2609.04172.

Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. MiniLLM: Knowledge distillation of large language models. In International Conference on Learning Representations, 2024. URL https://openreview.net/forum? id=5h0qf7IBZZ.

Ijun Jang, Jewon Yeom, Juan Yeo, Hyunggyu Lim, and Taesup Kim. Stable on-policy distillation through adaptive target reformulation. In Findings of the Association for Computational Linguistics: ACL 2026, pp. 42217–42227, 2026. doi: 10.18653/v1/2026.findings-acl.2094. URL https://aclanthology.org/ 2026.findings-acl.2094/.

Nan Jia, Haojin Yang, Xing Ma, Jiesong Lian, Shuailiang Zhang, Weipeng Zhang, Ke Zeng, Xunliang Cai, and Zequn Sun. Asymmetric on-policy distillation: Bridging exploitation and imitation at the token level. arXiv preprint arXiv:2605.06387, 2026. URL https://arxiv.org/abs/2605.06387.

Woogyeol Jin, Taywon Min, Yongjin Yang, Dennis Wei, Yi Zhou, Swanand Ravindra Kadhe, Nathalie Baracaldo, and Kimin Lee. Entropy-aware on-policy distillation of language models. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, 2026. URL https://arxiv.org/abs/2603.07079v3.

Amirhossein Kazemnejad, Milad Aghajohari, Eva Portelance, Alessandro Sordoni, Siva Reddy, Aaron Courville, and Nicolas Le Roux. VinePPO: Refining credit assignment in RL training of LLMs. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 29557–29590, 2025. URL https://proceedings.mlr.press/v267/kazemnejad25a.html.

Jongwoo Ko, Sungnyun Kim, Tianyi Chen, and Se-Young Yun. DistiLLM: Towards streamlined distillation for large language models. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 24872–24895, 2024. URL https://proceedings.mlr. press/v235/ko24c.html.

Gengsheng Li, Mao Zheng, Mingyang Song, Ruiqi Liu, Tianyu Yang, Jie Sun, Qiyong Zhong, Haiyun Guo, Junfeng Fang, Dan Zhang, and Jinqiao Wang. On-policy distillation with curriculum turn-level guidance for multi-turn agents. arXiv preprint arXiv:2606.15912, 2026. URL https://arxiv.org/abs/2606.15912.

Baohao Liao, Hanze Dong, Christof Monz, Xinxing Xu, Li Dong, and Furu Wei. Multi-turn on-policy distillation with prefix replay. arXiv preprint arXiv:2607.04763, 2026. URL https://arxiv.org/abs/2607.04763.

Alexander Lin, Jeremy Wohlwend, Howard Chen, and Tao Lei. Autoregressive knowledge distillation through imitation learning. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, pp. 6121–6133, 2020. doi: 10.18653/v1/2020.emnlp-main.494. URL https://aclanthology.org/ 2020.emnlp-main.494/.

Kevin Lu and Thinking Machines Lab. On-policy distillation. Thinking Machines Lab: Connectionism, 2025. doi: 10.64434/tml.20251026. URL https://thinkingmachines.ai/blog/ on-policy-distillation/.

Chiyu Ma, Shuo Yang, Kexin Huang, Jinda Lu, Haoming Meng, Shangshang Wang, Bolin Ding, Soroush Vosoughi, Guoyin Wang, and Jingren Zhou. FIPO: Eliciting deep reasoning with future-KL influenced policy optimization. arXiv preprint arXiv:2603.19835, 2026. URL https://arxiv.org/abs/2603.19835.

Marc’Aurelio Ranzato, Sumit Chopra, Michael Auli, and Wojciech Zaremba. Sequence level training with recurrent neural networks. In International Conference on Learning Representations, 2016. URL https: //arxiv.org/abs/1511.06732.

Stephane Ross and Drew Bagnell. Eficient reductions for imitation learning. In Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, volume 9 of Proceedings of Machine Learning Research, pp. 661–668, 2010. URL https://proceedings.mlr.press/v9/ross10a.html.

Stephane Ross, Geofrey Gordon, and Drew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, volume 15 of Proceedings of Machine Learning Research, pp. 627–635, 2011. URL https: //proceedings.mlr.press/v15/ross11a.html.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. URL https://arxiv.org/abs/1707.06347.

Bing Shao, Jiazheng Zhang, Long Ma, Yujiong Shen, Senjie Jin, Xin Guo, Yuming Yang, Mingxu Chai, Zhiheng Xi, Boyang Liu, Junlin Shang, Tao Gui, Qi Zhang, and Xuanjing Huang. A token-level analysis of sampled-token reverse-KL on-policy distillation. arXiv preprint arXiv:2608.25643, 2026. URL https://arxiv.org/abs/ 2608.25643.

Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. ALFWorld: Aligning text and embodied environments for interactive learning. In International Conference on Learning Representations, 2021. URL https://arxiv.org/abs/2010.03768.

Jiaqi Wang, Wenhao Zhang, Weijie Shi, Yaliang Li, and James Cheng. TCOD: Exploring temporal curriculum in on-policy distillation for multi-turn autonomous agents. arXiv preprint arXiv:2604.24005, 2026. URL https://arxiv.org/abs/2604.24005.

Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. Math-Shepherd: Verify and reinforce LLMs step-by-step without human annotations. In Proceedings of the 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 9426–9439, 2024. doi: 10.18653/v1/2024.acl-long.510. URL https://aclanthology.org/2024.acl-long.510/.

Ruoyao Wang, Peter Jansen, Marc-Alexandre Côté, and Prithviraj Ammanabrolu. ScienceWorld: Is your agent smarter than a 5th grader? In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.), Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 11279–11298, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022. emnlp-main.775. URL https://aclanthology.org/2022.emnlp-main.775/.

Yuanda Xu, Hejian Sang, Zhengze Zhou, Ran He, Zhipeng Wang, and Alborz Geramifard. TIP: Token importance in on-policy distillation. arXiv preprint arXiv:2604.14084, 2026. URL https://arxiv.org/abs/2604. 14084.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. URL https://arxiv.org/abs/2505.09388.

Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. WebShop: Towards scalable real-world web interaction with grounded language agents. In Advances in Neural Information Processing Systems, volume 35, pp. 20744–20757, 2022. doi: 10.52202/068431-1508. URL https://proceedings.neurips.cc/paper\_ files/paper/2022/hash/82ad13ec01f9fe44c01cb91814fd7b8c-Abstract-Conference. html.

Shunyu Yao, Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations, 2023. URL https://arxiv.org/abs/2210.03629.

Binbin Zheng, Xing Ma, Yiheng Liang, Jingqing Ruan, Xiaoliang Fu, Kepeng Lin, Benchang Zhu, Ke Zeng, and Xunliang Cai. SCOPE: Signal-calibrated on-policy distillation enhancement with dual-path adaptive weighting. arXiv preprint arXiv:2604.10688, 2026. URL https://arxiv.org/abs/2604.10688.

Qiyong Zhong, Mao Zheng, Mingyang Song, Xin Lin, Jie Sun, Yiqi Zhao, Xiaodi Wang, Houcheng Jiang, Xiang Wang, and Junfeng Fang. SOD: Step-wise on-policy distillation for small language model agents. arXiv preprint arXiv:2605.07725, 2026. URL https://arxiv.org/abs/2605.07725.

## A. Optimization details

The paired comparison in Equation 11 assumes successful replay to the same state and history before the selected response. With independent environment instances, a common continuation policy, and equal remaining horizons, the comparison conditions are

$$
\begin{array} { r l } & { ( s _ { k ^ { \star } } ^ { S } , c _ { k ^ { \star } } ^ { S } ) = ( s _ { k ^ { \star } } ^ { T } , c _ { k ^ { \star } } ^ { T } ) = ( s _ { k ^ { \star } } , c _ { k ^ { \star } } ) , } \\ & { ~ p _ { \mathrm { c o n t } } ^ { S } = p _ { \mathrm { c o n t } } ^ { T } = p _ { \circ } , ~ B _ { \mathrm { r e m } } ^ { S } = B _ { \mathrm { r e m } } ^ { T } . } \end{array}\tag{18}
$$

As in Section 3.3, � identifies the use of the teacher response, which replaces the student’s original response in the assistant history. Both continuations share decoding settings and seed schedules. Replay or runtime errors exclude a trajectory from ${ \mathcal { D } } _ { \operatorname { p a i r } }$ . The positive part of the final task outcome diference, $\left[ \Delta _ { \zeta } \right] _ { + }$ , equals the indicator $\mathsf { g } _ { \boldsymbol { \zeta } }$ for additional teacher supervision in Equation 13.

For a fixed batch $\mathcal { D }$ of the student’s own trajectories, vanilla OPD and OG-OPD apply the same clipped policy surrogate to the same valid original response tokens. Only the turn weights difer. The valid token set and normalization are

$$
\mathcal { M } _ { \mathrm { o r i g } } ( \zeta ) = \{ ( k , j ) : j \in \mathcal { M } _ { k } ( \zeta ) \} , \qquad Z = \sum _ { \zeta \in \mathcal { D } } | \mathcal { M } _ { \mathrm { o r i g } } ( \zeta ) | ,\tag{19}
$$

where $M _ { k } ( \zeta )$ excludes prompts, environment observations, padding, and positions without valid teacher scores. The following identities concern batches with $Z > 0$ . For importance ratio $\rho$ and fixed signed coeficient � representing token-level teacher supervision, the shared surrogate is

$$
\ell _ { \mathrm { r a t i o } } ( \rho , a ) = \operatorname* { m a x } \{ - \rho a , - \mathrm { c l i p } ( \rho , 1 - \varepsilon _ { - } , 1 + \varepsilon _ { + } ) a \} ,\tag{20}
$$

$$
\ell _ { \mathrm { p o l } } ( \rho , a ) = \left\{ \begin{array} { l l } { \mathrm { m i n } \{ \ell _ { \mathrm { r a t i o } } ( \rho , a ) , - \kappa a \} , } & { a < 0 , } \\ { \ell _ { \mathrm { r a t i o } } ( \rho , a ) , } & { a \geq 0 . } \end{array} \right.\tag{21}
$$

Here $\varepsilon _ { - } , \varepsilon _ { + } > 0$ are clipping widths and $\kappa > 1$ bounds the surrogate for $a < 0$ . Token-level teacher supervision retains the signed coeficients $\psi _ { k , \pmb { \imath } }$ defined in Section 2.2. Recorded probabilities, turn weights, and final task outcome indicators are held fixed during diferentiation. In implementation, the surrogate uses the numerically bounded ratio

$$
\widehat { \rho } _ { k , j } ( \theta ) = \exp \bigl ( \mathrm { c l i p } \bigl ( \log p _ { \theta } ( z _ { k , j } \mid \upsilon _ { k , j } ) - \log p _ { \circ } ( z _ { k , j } \mid \upsilon _ { k , j } ) , - C , C \bigr ) \bigr ) ,\tag{22}
$$

where $C = 2 0$ . The bounded ratio equals Equation 3 when clipping is inactive. Equation 24 holds for either ratio. The following identities apply to the fixed-coeficient policy surrogate, not the exact reverse-KL objective at arbitrary student parameters.

All valid original tokens at a turn share its weight, applied through the recorded advantage

$$
a _ { k , j } ^ { \mathrm { O G - O P D } } = \Im \big [ \omega _ { k } \psi _ { k , j } \big ] .\tag{23}
$$

The clipped loss is positively homogeneous in its signed argument. For fixed $\omega \ge 0$

$$
\begin{array} { r } { \ell _ { \mathrm { p o l } } ( \rho , \omega a ) = \omega \ell _ { \mathrm { p o l } } ( \rho , a ) . } \end{array}\tag{24}
$$

Multiplication by a positive fixed weight preserves the sign of � and the ordering of the arguments to each minimum and maximum; both sides vanish for $\omega = 0$ . On this fixed batch, weighting the recorded advantage is therefore equivalent to weighting the original turn loss $Q _ { k }$ . Substituting Equation 14 yields the decomposition in Equation 15.

Under Equation 14, $\gamma _ { \zeta } = 0$ when $\mathtt { g } _ { \zeta } = 0$ or $\beta _ { k ^ { \star } } \geq \beta _ { \uparrow }$ . With no additional weight, the objective reduces to $\mathcal { L } _ { \mathrm { b a s e } }$ . Unit pre-calibration weights further recover Equation $5 .$ For fixed recorded coeficients, positive turn weights rescale each token’s loss gradient without reversing its direction. Reweighting across turns can nevertheless change the direction of the batch gradient.

Gradient decomposition and weight bounds. For the fixed batch and recorded weights above, diferentiation can be taken inside the finite token sums:

$$
\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { O G - O P D } } = \frac { 1 } { Z } \sum _ { \zeta \in \mathcal { D } } \sum _ { k \in \mathcal { V } ( \zeta ) } \omega _ { k } \sum _ { \boldsymbol { j } \in M _ { k } ( \zeta ) } \nabla _ { \boldsymbol { \theta } } \ell _ { \mathrm { p o l } } \bigl ( \rho _ { k , \boldsymbol { j } } ( \boldsymbol { \theta } ) , \boldsymbol { s } \boldsymbol { g } [ \psi _ { k , \boldsymbol { j } } ] \bigr ) .\tag{25}
$$

Splitting $\omega _ { k }$ into its base and additional terms recovers Equation 17, with no gradient through the sampled final task outcomes. Wherever the derivatives exist, the triangle inequality gives

$$
\left. \nabla _ { \theta } \mathcal { L } _ { \mathrm { O G - O P D } } - \nabla _ { \theta } \mathcal { L } _ { \mathrm { b a s e } } \right. \leq \frac { 1 } { Z } \sum _ { \zeta \in \mathcal { D } _ { \mathrm { p a i r } } } \gamma _ { \zeta } \left. \nabla _ { \theta } Q _ { k ^ { \star } } ( \theta ; \zeta ) \right. .\tag{26}
$$

This expression bounds the additional loss gradient in terms of the selected turn gradients and their weights. A small number of upweighted turns alone does not imply a small gradient change, and the bound does not establish an improvement in task success after optimization.

The floor rule provides explicit bounds on the coeficients. Since $0 < \beta _ { k } \le \beta _ { \mathrm { m a x } }$ on valid turns and $\mathsf { g } _ { \boldsymbol { \zeta } } \in \{ 0 , 1 \}$

$$
0 \leq \gamma _ { \zeta } \leq \beta _ { \uparrow } , \qquad \beta _ { k } \leq \omega _ { k } \leq \operatorname* { m a x } \{ \beta _ { \operatorname* { m a x } } , \beta _ { \uparrow } \} .\tag{27}
$$

These are bounds on supervision coeficients, not on gradient norms. In particular, no additional normalization by the sum of weights is introduced, so the original valid token denominator � is preserved.

Interpretation of outcome-based calibration. For a restored state �, history �, response �, and frozen current student, define the probability of a successful final task outcome by

$$
V _ { p _ { \circ } } ( s , c , z ) = \mathbb { E } _ { \varsigma } [ \mathsf { S } ( \mathcal { U } ( s , c , z ; p _ { \circ } , \varsigma ) ) ] .\tag{28}
$$

For fixed $( s , c , z ^ { S } , z ^ { T } , p _ { \circ } )$ , correctly restored states and the intended sampling marginals give

$$
\begin{array} { r } { \mathbb { E } _ { \varsigma } \big [ \mathsf { S } ( \tilde { \zeta } ^ { T } ) - \mathsf { S } ( \tilde { \zeta } ^ { S } ) \big ] = V _ { p _ { \circ } } ( s , c , z ^ { T } ) - V _ { p _ { \circ } } ( s , c , z ^ { S } ) . } \end{array}\tag{29}
$$

The shared seed schedule couples the final task outcomes without changing this identity. Under these fixed conditions,

$$
\begin{array} { r l } & { { \mathbb E } _ { \varsigma } [ \Delta _ { \zeta } ] = \mathrm { P r } ( \Delta _ { \zeta } = 1 ) - \mathrm { P r } ( \Delta _ { \zeta } = - 1 ) , } \\ & { \ { \mathbb E } _ { \varsigma } [ \mathrm { \bf g } _ { \zeta } ] = \mathrm { P r } ( \Delta _ { \zeta } = 1 ) . } \end{array}\tag{30}
$$

The first expectation measures net execution benefit; the second measures teacher rescue probability under paired sampling. OG-OPD uses one observed teacher rescue to condition additional weight, rather than estimating net expected benefit from repeated trials. These execution statistics do not establish the efect of an individual training update.

For a selected turn with fixed pre-calibration weight, let $h _ { \zeta } = \left[ \beta _ { \uparrow } - \beta _ { k ^ { \star } } \right] _ { + }$ and $p _ { + } = \mathrm { P r } _ { \varsigma } ( \Delta _ { \zeta } = 1 )$ under the same fixed comparison conditions. Since $\gamma _ { \zeta } = h _ { \zeta } { \bf g } _ { \zeta }$ , its first two moments are

$$
\mathbb { E } _ { \varsigma } [ \gamma _ { \zeta } ] = h _ { \zeta } p _ { + } , \qquad \mathrm { V a r } _ { \varsigma } ( \gamma _ { \zeta } ) = h _ { \zeta } ^ { 2 } p _ { + } ( 1 - p _ { + } ) .\tag{31}
$$

The expected additional weight therefore depends on both the probability of an observed rescue and the distance to the weight floor. When $h _ { \zeta } = 0 _ { : }$ , even a positive paired outcome leaves the weight unchanged. For $h _ { \zeta } > 0 _ { : }$ , the probability of an actual increase equals $p _ { \cdot }$ <sub>+</sub> under these fixed conditions.

## B. Empirical analysis details

Section 3.1 introduces the empirical analysis with a fixed Qwen3-1.7B student and Qwen3-32B teacher on ScienceWorld and WebShop. Continuations � and �� use the current student after the student’s original action and the teacher action, respectively; �� uses the teacher after the same teacher action to compare what the teacher and current student can complete. Eligible positions within the student’s own trajectories are sampled uniformly without replacement, independently of OG-OPD’s candidate selection rule. Under a bounded proposal budget, we retain the first replay-valid sampled position with a teacher response whose parsed action is legal and difers from the student’s original action. The empirical analysis therefore includes only decisions with valid recorded scores, successful replay, and valid teacher responses.

Each retained decision is evaluated in five matched trials. Let $\tilde { \zeta } _ { m } ^ { b }$ be continuation � in trial �. With � valid trials, we compute each continuation’s success frequency and how often the paired student continuations difer in final task outcome:

$$
f _ { b } = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } S ( \tilde { \zeta } _ { m } ^ { b } ) , \quad b \in \{ S , T S , T T \} , \qquad q _ { \mathrm { d i f f } } = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } | S ( \tilde { \zeta } _ { m } ^ { T S } ) - S ( \tilde { \zeta } _ { m } ^ { S } ) | .\tag{32}
$$

We assign outcome categories using lower and upper success frequency thresholds $\alpha _ { - } < \alpha _ { + }$

$$
\mathrm { b e n i g n ~ g a p s : } \quad f _ { S } \geq \alpha _ { + } , ~ f _ { T S } \geq \alpha _ { + } ,
$$

teacher rescue : $f _ { S } \leq \alpha _ { - } , \ f _ { T S } \geq \alpha _ { + } ,$

teacher harm : $f _ { S } \geq \alpha _ { + } , \ f _ { T S } \leq \alpha _ { - } ,$

(33)

teacher-only completion : $f _ { T T } \geq \alpha _ { + } , \ f _ { T S } \leq \alpha _ { - }$

outcome-critical : $q _ { \mathrm { d i f f } } \ge \alpha _ { + } .$

Here $M = 5 , \alpha _ { - } = 1 / 5 ,$ , and $\alpha _ { + } = 3 / 5 ,$ corresponding to at most one and at least three successes. Benign gaps use each action’s success frequency, whereas outcome-critical decisions require diferent final task outcomes in at least three matched pairs. Categories can overlap, including teacher harm and teacher-only completion. Model parameters remain fixed, and all three continuations use matched seed schedules, decoding settings, and remaining interaction budgets. Teacher responses replace both the executable action and the assistant response retained in history.

The retained set contains 329 of 330 selected ScienceWorld decisions after excluding one replay failure, and all 274 selected WebShop decisions. We divide gap magnitudes into five groups using empirical quantiles of $\chi _ { k }$ in Equation 6; large and small gaps refer to the highest and lowest groups. Boundaries are computed separately over all eligible positions in each benchmark, including unselected positions. In ScienceWorld, the large- and small-gap groups contain 172 and 37 decisions, respectively; the corresponding counts for WebShop are 73 and 33. Empirical analysis quantiles give equal weight to each eligible position, whereas the training quantiles in Equation 36 balance trajectories. Decisions with completed trials are assigned to these existing groups, yielding unequal group sizes.

Counts and cross-benchmark aggregation. Table 4 gives the counts behind Figure 2. For outcome category $j ,$ let $I _ { j } ( d )$ equal one when decision � meets its criterion and zero otherwise. Let $\mathcal { D } _ { j , e }$ contain the decisions in the corresponding gap group of benchmark �. We divide the number of decisions in the category by the group size, then average the two benchmark percentages with equal weight:

$$
r _ { j , e } = \frac { 1 0 0 } { | \mathscr { D } _ { j , e } | } \sum _ { d \in \mathscr { D } _ { i , e } } I _ { j } ( d ) , \qquad \bar { r } _ { j } = \frac { r _ { j , \mathrm { S c i e n c e W o r l d } } + r _ { j , \mathrm { W e b S h o p } } } { 2 } .\tag{34}
$$

The calculation applies to all five categories, with percentages averaged before rounding. Repeated trials determine category membership; each decision contributes once to every category it meets. Rates use the number of retained decisions in the corresponding gap group as their denominator.

Confidence intervals. Within each benchmark and gap group, we resample retained decisions with replacement 2,000 times and take the 2.5th and 97.5th percentiles to obtain the 95% confidence intervals in Figure 2. Each decision is classified using its five matched trials before resampling. The zero observed count for small-gap outcome-critical decisions on ScienceWorld produces a degenerate bootstrap interval, not evidence of a zero population proportion.

Table 4 | Counts underlying the empirical analysis. Each benchmark entry gives the number of decisions in an outcome category divided by the corresponding gap group size, followed by the percentage in parentheses. Categories follow Equation 33. The average gives equal weight to the two benchmarks.
<table><tr><td>Outcome category</td><td>Gap group</td><td>ScienceWorld</td><td>WebShop</td><td>Average</td></tr><tr><td>Benign gaps</td><td>Large</td><td>2/172 (1.2%)</td><td>7/73 (9.6%)</td><td>5.4%</td></tr><tr><td>Teacher rescue</td><td>Large</td><td>9/172 (5.2%)</td><td>9/73 (12.3%)</td><td>8.8%</td></tr><tr><td>Teacher harm</td><td>Large</td><td>4/172 (2.3%)</td><td>14/73 (19.2%)</td><td>10.8%</td></tr><tr><td>Teacher-only completion</td><td>Large</td><td>26/172 (15.1%)</td><td>19/73 (26.0%)</td><td>20.6%</td></tr><tr><td>Outcome-critical</td><td>Small</td><td>0/37 (0.0%)</td><td>5/33 (15.2%)</td><td>7.6%</td></tr></table>

## B.1. Ranking by Gap Magnitude

We further examine whether larger gaps preferentially identify decisions where teacher guidance benefits the current student. For Figure $^ { 5 , }$ let $S _ { k }$ contain the top � of � retained decisions ranked by decreasing gap magnitude. With teacher rescue and teacher harm indicators $R _ { j }$ and $H _ { j }$ from Equation 33, the plotted measures are

$$
\mathrm { c o v e r a g e } ( k ) = \frac { k } { n } , \qquad \mathrm { p r e c i s i o n } ( k ) = \frac { \sum _ { j \in S _ { k } } R _ { j } } { k } , \qquad \mathrm { n e t } ( k ) = \frac { \sum _ { j \in S _ { k } } \bigl ( R _ { j } - H _ { j } \bigr ) } { k } .\tag{35}
$$

Ties are broken deterministically by task identity. WebShop’s net rescue is negative over much of the coverage range, while ScienceWorld’s stays near zero or slightly positive. These curves compare the frequencies of the decision categories in Equation 33, not the expected outcome diference in Equation 29. They describe the retained decisions under the frozen current student, with small selected sets at low coverage.

![](images/8694c0adf170ddebb13a9d72d75932a85ed8f8337162aab41bac9280ea8751c4.jpg)

![](images/8c3662a7ca5c88f642f840e0f519126041eba31100a3b2f32336c74301919086.jpg)  
Figure 5 | Final task outcomes under gap magnitude ranking. Decisions are ranked by decreasing gap magnitude. Rescue precision (left) is the teacher rescue rate; net rescue (right) subtracts the teacher harm rate.

## C. Execution contract

ALFWorld, ScienceWorld, and WebShop responses are parsed into their respective household commands, scientific actions, and search or click operations. The environment receives the parsed action, while the complete assistant response remains in the interaction history. For the table comparisons, methods within each benchmark share evaluation prompts, response token budgets, history retention, action parsing, interaction horizons, termination handling, and score extraction. Action parsing follows each benchmark’s shared protocol, without additional repair rules for individual methods.

Teacher and student log probabilities are evaluated on the same valid original response tokens, including generated reasoning and action text but excluding prompts, observations, and padding. Gap magnitudes from Equation 6 determine trajectory-relative weighting and candidate selection, while the signed coeficients in Equation 3 determine token-level teacher supervision. Missing or nonfinite scores are excluded. Turns with empty valid token masks have no gap or weight and are excluded from the loss and threshold populations; trajectories with no valid turns contribute neither supervision nor candidates. A first valid turn $k _ { 0 } > 0$ supplies the weighting reference but has no valid change from the preceding turn and cannot use the candidate exception reserved for $k = 0$

For the candidate thresholds, let $\mathcal { H } _ { \zeta } ^ { f }$ contain eligible positions for statistic $f \in \{ \nu , v \}$ and let $\mathcal { D } _ { f }$ contain trajectories with nonempty $\mathcal { H } _ { \zeta } ^ { f }$ . Only positive valid changes $v _ { k } > 0$ enter the population for �. The empirical distribution and quantile are

$$
\begin{array} { r l r } {  { \widehat { F } _ { f } ( \nu ) = \frac { 1 } { | \mathcal { D } _ { f } | } \sum _ { \zeta \in \mathcal { D } _ { f } } \frac { 1 } { | \mathcal { H } _ { \zeta } ^ { f } | } \sum _ { k \in \mathcal { H } _ { \zeta } ^ { f } } \mathbb { I } [ f _ { k } \leq \nu ] , } } \\ & { } & { \vartheta _ { f } = \operatorname* { i n f } \{ \nu : \widehat { F } _ { f } ( \nu ) \geq q _ { f } \} . } \end{array}\tag{36}
$$

Here $q _ { f } \in ( 0 , 1 )$ is the quantile level. Each contributing trajectory receives equal total weight in the threshold population. Since $\log ( \varepsilon _ { 0 } + \chi )$ is strictly increasing, the implementation equivalently transforms the inverse CDF quantile of the gap magnitude. If the threshold population is insuficient, pre-calibration weights are retained.

Teacher responses use the native prompt and parser for each benchmark with native thinking mode disabled. Each complete response, including any visible rationale, replaces the student’s original response in continuation history. Empty or unparseable responses and those yielding inadmissible or unchanged actions are excluded. Within a fixed proposal budget, candidates are checked chronologically. The first replay-valid turn with a legal teacher action diferent from the student’s original action is selected for paired student continuations. Independent environment instances replay the task prefix and check the intended observation and action context. Both continuations use the same frozen current student, remaining budget, decoding settings, and seed schedule at each step. Replay and runtime failures invalidate the comparison rather than count as task failures.

Only tokens from the student’s own trajectories enter the loss. The frozen rollout version supplies the importance ratio denominator, and asynchronous collection retains the existing staleness control. Both paired

Table 5 | OG-OPD weighting and calibration settings. Candidate thresholds follow Equation 36. Training uses one matched pair per selected position, whereas the empirical analysis uses five matched trials per decision.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Pre-calibration weight cap  $\beta _ { \mathrm { m a x } }$ </td><td>1.2</td></tr><tr><td>Positive-outcome weight floor  $\beta _ { \uparrow }$ </td><td>1.5</td></tr><tr><td>Gap-magnitude / positive-increase quantile</td><td> $q _ { \nu } = 0 . 5 ⁄ q _ { v } = 0 . 6$ </td></tr><tr><td>Candidate checks / teacher proposals per trajectory</td><td>At most  $2 / 2$ </td></tr><tr><td>Paired-evaluation positions per trajectory</td><td>At most 1</td></tr><tr><td>Matched trials per selected position</td><td>1 pair</td></tr></table>

student continuations use that same student version. Their final task outcomes determine the additional weight in Equation 14; the original token-level teacher supervision remains the signed coeficient in the loss.

## D. Experimental details

## D.1. Model configurations and comparison scope

The model pairings follow Section 4.1. Runs use one node with 8×A800 GPUs. The empirical analysis, gap magnitude ranking, ablations, and timing use the Qwen3-32B teacher and Qwen3-1.7B student. The WebShop training dynamics in Section 4.2 cover both configurations.

Table 5 summarizes the OG-OPD weighting and calibration settings. Complete training configurations are provided with the supplementary source code.

Figure 4 reports WebShop training curves separately from table results averaged over three seeds. Entropy and Teacher NLL are averaged over five steps, and the remaining rates are pooled over five batches.

The RL teachers are obtained by training Qwen3-8B separately on ALFWorld and WebShop with GiGPO (Feng et al., 2025), following the teacher training approach of Chen et al. (2026). Both use AdamW with learning rate $1 0 ^ { - 6 }$ and weight decay 0.01. The discount factor is 0.95, the rollout group size is 8, and the invalid action penalty coeficient is 0.1. Training batch sizes are 32 for ALFWorld and 16 for WebShop, with a PPO minibatch size of 64. Teacher training uses a prompt length limit of 10,240 tokens, with interaction limits of 30 steps for ALFWorld and 15 steps for WebShop. Responses are limited to 512 tokens, with thinking mode disabled. Training and validation temperatures are 1.0 and 0.4, respectively. We use the ALFWorld checkpoint at step 150 and the WebShop checkpoint at step 100, held fixed across all compared methods.

Performance and ablation comparisons evaluate complete training procedures at step 200. Ablations retain each procedure’s resulting number of additionally upweighted turns and total supervision weight, rather than matching these quantities across variants. OG-OPD includes additional environment execution and final task outcome feedback, so these comparisons assess the complete algorithm. They use a fixed number of optimization steps, with actual training costs reported separately in Appendix E.

## D.2. Evaluation tasks, metrics, and aggregation

Following Wang et al. (2026), ALFWorld evaluation uses 140 Seen tasks in environments encountered during training and 134 Unseen tasks with room layouts and object combinations not encountered during training. We also use TCOD’s fixed Hard collection of 121 tasks drawn from the training split, on which its teacher failed in all ten sampling attempts. We report SR for each collection and Overall SR over all 395 tasks pooled together. ScienceWorld evaluates 1,308 canonical tasks. WebShop uses 100 held-out sessions, as in Chen et al. (2026). Interaction limits are 30 steps for ALFWorld and ScienceWorld and 15 steps for WebShop. ScienceWorld success requires full completion. WebShop uses the shared TCOD success criterion of terminal reward above 0.5.

For episode �, let $s _ { j , t }$ be its ScienceWorld environment score on a 0–100 scale and $r _ { j , \mathrm { e n d } }$ its WebShop terminal reward on a 0–1 scale. Score is computed for each benchmark as

$$
\mathrm { S c o r e } _ { \mathrm { S W } } = \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \mathrm { c l i p } \Bigl ( \operatorname* { m a x } _ { t } s _ { j , t } , 0 , 1 0 0 \Bigr ) , \quad \mathrm { S c o r e } _ { \mathrm { W S } } = \frac { 1 0 0 } { n } \sum _ { j = 1 } ^ { n } r _ { j , \mathrm { e n d } } .\tag{37}
$$

Rounds averages the actual interaction lengths of all evaluated episodes, including failures. Score is extracted within each episode as defined above, using the maximum ScienceWorld score and the terminal WebShop reward.

For episode success label $Y _ { j }$ under the criterion for that benchmark and executed interaction length $L _ { j } ,$ the remaining aggregates are

$$
{ \mathrm { S R } } = { \frac { 1 0 0 } { n } } \sum _ { j = 1 } ^ { n } Y _ { j } , \qquad { \mathrm { R o u n d } } s = { \frac { 1 } { n } } \sum _ { j = 1 } ^ { n } L _ { j } , \qquad { \mathrm { O v e r a l l } } _ { { \mathrm { A L F } } } = { \frac { \sum _ { s \in S } n _ { s } { \mathrm { S R } } _ { s } } { \sum _ { s \in S } n _ { s } } } ,\tag{38}
$$

where $S = \{ S \mathrm { e } \mathrm { e } \mathrm { n }$ , Unseen, Hard} and $n _ { s }$ is the number of evaluated tasks in collection �. Overall SR gives each task equal weight.

Trained-model results in the main and ablation tables use the fixed endpoint at optimizer step 200 and report means and sample standard deviations across three seeds. For any evaluation metric $M ,$ we first compute the metric over the benchmark’s episodes for each seed, then report

$$
\overline { { { M } } } _ { 2 0 0 } = \frac { 1 } { 3 } \sum _ { r = 1 } ^ { 3 } M _ { r , 2 0 0 } , \qquad s _ { M , 2 0 0 } = \sqrt { \frac { 1 } { 3 - 1 } \sum _ { r = 1 } ^ { 3 } \bigl ( M _ { r , 2 0 0 } - \overline { { { M } } } _ { 2 0 0 } \bigr ) ^ { 2 } } .\tag{39}
$$

Each trained model is evaluated at the fixed endpoint, and zero-shot references are evaluated without training.   
Timing is measured per run and averaged across seeds as described in Appendix E.

## E. Training eficiency

Table 6 | Training cost of vanilla OPD and OG-OPD. Runs use 8×A800 GPUs with the Qwen3-32B teacher and Qwen3-1.7B student. OG-OPD timing includes outcome-based calibration, and Cost is normalized to vanilla OPD. Rounds reports mean evaluation interaction rounds. Boldface marks the best value per benchmark.

<table><tr><td>Benchmark</td><td>Method</td><td>Rounds↓</td><td>s/step ↓</td><td>Cost↓</td></tr><tr><td rowspan="2">ScienceWorld</td><td>Vanilla OPD</td><td>12.6</td><td>156.97</td><td>1.00×</td></tr><tr><td>OG-OPD</td><td>11.6</td><td>223.58</td><td>1.42×</td></tr><tr><td rowspan="2">WebShop</td><td>Vanilla OPD</td><td>6.8</td><td>67.21</td><td>1.00×</td></tr><tr><td>OG-OPD</td><td>5.6</td><td>99.70</td><td>1.48×</td></tr></table>

For each run, elapsed time from training start through step 200 is divided by 200, then averaged across three seeds. The measurement includes state replay, teacher response generation, and paired student continuations, and excludes final synchronization and evaluation. Deployment uses only the trained student.

Let $T _ { m , e , r }$ denote this measured elapsed time for method $m ,$ benchmark $e ,$ and training seed $r .$ With $K = 2 0 0$ optimization steps and $R = 3$ seeds, the reported seconds per step and normalized cost are

$$
\bar { \tau } _ { m , e } = \frac { 1 } { R } \sum _ { r = 1 } ^ { R } \frac { T _ { m , e , r } } { K } , \qquad \mathrm { C o s t } _ { m , e } = \frac { \bar { \tau } _ { m , e } } { \bar { \tau } _ { \mathrm { O P D } , e } } .\tag{40}
$$

Cost is the ratio of the two mean times per step, rather than a mean of ratios computed for individual seeds. The corresponding absolute and relative increases are

$$
\Delta \bar { \tau } _ { e } = \bar { \tau } _ { \mathrm { O G - O P D } , e } - \bar { \tau } _ { \mathrm { O P D } , e } , \qquad \eta _ { e } = 1 0 0 \left( \mathrm { C o s t } _ { \mathrm { O G - O P D } , e } - 1 \right) .\tag{41}
$$

Using the rounded times in Table $^ { 6 , }$ these increases are 66.61 seconds per step on ScienceWorld and 32.49 seconds on WebShop, or approximately 42% and 48%. They describe end-to-end training overhead under the reported hardware and execution settings, not a separate timing measurement of each calibration operation.