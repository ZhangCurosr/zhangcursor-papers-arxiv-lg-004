# Which the Eye Fears: WRITING WITH READ-BLINDNESS EXPLAINS MASSIVE ACTIVATIONS IN TRANSFORMERS

Swagatam Mukhopadhyay PsiDagger swag@vislesy.com

Vishal Vivek Saley   
IIT Delhi   
vishal.vivek.saley@cse.iitd.ac.in

Vraj Parikh Mausam IIT Delhi IIT Delhi

## ABSTRACT

Massive activation features (MAs) in Transformers are extreme-value residualstream features that persist across layers despite the model’s ability to suppress them. Why do they survive? Our investigation using an operator-level mechanistic analysis of attention and feed-forward (FFN) blocks reveals that these blocks systematically ignore MA coordinates while reading, but not while writing; creating a read-write asymmetry that blocks corrective feedback while allowing continued accumulation. We find that both attention and feed-forward layers have this read-blindness, and contribute to the emergence and persistence of MAs.

To validate prior work that hypothesized that FFN’s amplification ability is the primary reason for MAs (Sun et al., 2026), we analyze the model checkpoints during learning. Contrary to our expectation, read-blindness emerges before FFN amplification, suggesting that it acts upstream in the MA mechanism. We further contribute gradient analysis to link this behavior to surprising asymmetries in the loss landscape, concluding that the model actively maintains this read-blindness. Finally, we find that removing read-blocking at different locations induces compensatory shifts elsewhere, but MAs still persist.

## 1 INTRODUCTION

Massive activation features (MAs) are a small proportion of coordinates in the residual stream of a Transformer whose values exceed the rest of the model features by orders of magnitude (Bondarenko et al., 2023; Sun et al., 2024; 2026). They are central to how the model handles the first token of a sequence (Xiao et al., 2024), make low-bit quantization difficult (Yu et al., 2024), and shape the geometry of key-value cache compression (Ge et al., 2024). The most striking thing about them, however, is how stubbornly they persist. An MA at an early layer typically remains massive through nearly the entire network, even though every later attention head and feed-forward block has, in principle, the linear capacity to subtract it back to a normal range. None of them does. Why?

Several answers to this phenomenon have been proposed, ranging from an attention sink that dominates the BOS token (Xiao et al., 2024; Gu et al., 2025), to low-cost storage on delimiter tokens as compression valleys (Queipo-de-Llano et al., 2026), to more recently, early feed-forward blocks acting as a directional amplifier on a small set of channels (Sun et al., 2026). Though these attempt to identify the origins of massive activations, mechanistic explanations have been either elusive or partial. For example, none of these explain why no subsequent layer reduces the massive activations.

In this paper, we investigate progression of MAs through the key operators (attention, FFN) in the architecture. We first observe that an operator can fail to correct a residual-stream coordinate because either (1) the coordinate lies in its output null space: no choice of input produces a non-zero output for that coordinate; or (2) in its input null space; the coordinate is not read by the operator, and it produces the same output irrespective of the coordinate’s value. We find empirically that MAs are read-blocked (read-blindness) by attention’s $W _ { V } , W _ { Q }$ , and $W _ { K }$ matrices, and by the feed-forward (FFN) block’s gate and up-projections $( W _ { u p } )$ , but they are not write-blocked: every layer continue to add a non-zero contribution to MAs. We argue that this read-write asymmetry is the mechanism that makes MAs persistent through the network. Our observations adds several nuances and expands far beyond the simple hypothesis by Sun et al. (2026), which ascribes FFN amplification as the primary reason for MAs. To investigate this thoroughly, we analyze the model checkpoints during learning and find that read-blindness emerges before FFN amplification, suggesting that it acts upstream in the MA mechanism.

We next ask whether the observed asymmetry is actively maintained by training or is merely incidental. At a stable optimum, this question can be probed using local curvature via the Hessian of the loss. We examine the Hessian spectrum of the trained model, along parameters governing read and write pathways. We discover a pronounced asymmetry: directions associated with read operators are stiff, exhibiting higher-than-average curvature; while, those associated with write operators are comparatively sloppy, with substantially lower curvature. Simply stated, the model appears to actively maintain this read-blindness, and the phenomenon is likely not accidental.

We next attempt two structural interventions to constrain the value projection $( W _ { V } )$ so that no null space can develop. Massive activations attenuate, however, the trained model reconfigures and reinforces its read-blindness through other operators such as FFNs. This suggests that MAs are not tied to a single component, but arise from a collusion of mechanisms. Further analysis of FFNs reveals that the previously identified “amplifier direction” is generic; it can, in principle, amplify many features; but the down-projection $W _ { d o w n }$ selects which of these amplified features are actually written back to the residual stream. It selects several MA features to amplify.

In summary, these results collectively represent a substantial advance in our mechanistic under standing of massive activations in Transformers. Our key result is that massive activations lie in the read-null space of various operators (in attention and FFN blocks), but not in their write-null space, that this behavior is actively maintained by the model, and that blocking read-blindness in one component leads to compensatory behavior in other components.

## 2 RELATED WORK

The Impact and Persistence of Massive Activations: Massive activations (MAs); a small proportion of residual stream coordinates that exceed normal representation ranges by orders of magnitude; are a well-documented phenomenon in Transformers (Bondarenko et al., 2023; Sun et al., 2024; Kaul et al., 2025; An et al., 2025). They significantly constrain model deployment, acting as primary bottlenecks for low-bit quantization (Dettmers et al., 2022; Yu et al., 2024) and shaping the geometry of KV cache compression (Ge et al., 2024). One of the most striking characteristics of MAs is their stubborn persistence across layers. Analyses of their behavior across varied hyperparameters and configurations remain an active area of study (Gallego-Feliciano et al., 2025; Owen et al., 2025), yet the question of why subsequent layers fail to normalize these outliers remains a central mystery.

Hypotheses on Origins and the FFN Amplifier: Several hypotheses have been proposed to explain the origins and functional utility of MAs. They are frequently tied to how models process the beginning-of-sequence (BOS) token, acting as “attention sinks” (Xiao et al., 2024), or viewed as low-cost storage functioning as “compression valleys.” More recently, a prominent mechanis tic explanation by Sun et al. (2026) posits that early feed-forward (FFN) layers act as directional amplifiers, pushing signals into the massive regime when inputs align with a specific s<sup>∗</sup> direction. However, our investigation shows that these hypotheses are partial. They primarily address the initiation of MAs but fail to explain their structural persistence; specifically, why subsequent attention heads and FFN blocks do not utilize their linear capacity to subtract these outliers. Furthermore, our empirical findings regarding read-blindness in attention blocks offer a direct counterpoint to the hypothesis that FFN amplifiers are the sole or primary drivers of MAs.

Mechanistic Interpretability and Training Dynamics: To mechanistically understand why models stubbornly maintain these outliers, we build upon the spectral and null-space analysis pioneered by Cancedda (2024). We extend this to formally isolate the input (read) and output (write) null spaces of key operators across Transformer blocks. This formalized lens allows us to uncover the read-blindness that explains MA persistence. Finally, to determine if this structural asymmetry is merely incidental or actively maintained, we analyze the trained model through the lens of stiff and sloppy loss landscapes, measured via Hessian curvature (Transtrum et al., 2015). This landscape analysis contextualizes our findings that the model actively defends its read-blindness during training, framing MAs as a resilient, distributed mechanism rather than an isolated architectural quirk.

## 3 PRELIMINARIES & METHODOLOGY

We investigate the persistence of MAs through the interactions between a Transformer block and the residual stream, which include reading from the residual stream through its input projections and writing back to it through its output projections. We discover that the persistence of MAs follows from this read–write role asymmetry. As a result, our work creates a framework to systematically evaluate architectural modifications that aim to modify the statistical distribution of feature values, from massive to outlier and so on, by mechanistically tracking read and write operations in a Trans former block. Formally, given a weight matrix W, the read component of a linear operation ${ \vec { y } } = W { \vec { x } }$ concerns the vector space of ⃗x. Read blindness concerns a subspace of $\vec { x }$ not read by W, i.e., it i the (approximate) right null space of W. Similarly, the write operation concerns the vector space of $\vec { y } \colon$ write blindness implies the presence of an (approximate) left null-space of $W$ preventing it from modifying a subspace of ${ \vec { y } } .$ For a general vector space operator, the same concepts hold. The added complexity in Transformers is that the Attention and FFN operators require the residual stream itself as input, before we can study their write operations. We call this the operator view (see Sec 4), and general behavior of such operators are reported as average over sampled residual streams.

Transformer architecture. We consider a standard Pre-LN Transformer (Touvron et al., 2023) in which each layer applies multi-head attention (MHA) followed by a SwiGLU FFN, with RMSNorm preceding each block. Given residual input X, a layer computes $Y = X + \mathrm { M H A } ( \widetilde { X } )$ and $Z = Y +$ $\mathrm { F F N } ( \widetilde { Y } )$ , where tildes denote normalized inputs; full equations are provided in Appendix A. This architecture separates the parameters that read the residual stream from those that write to it. We focus on the MHA read operations coordinated through matrices $W _ { Q } , W _ { K }$ , and $W _ { V }$ corresponding to the query, key, and value projections because they read the residual stream directly. Likewise, for the the FFN we focus on the reads through the $W _ { \mathrm { g a t e } }$ and $W _ { \mathrm { u p } }$ matrices. The $W _ { \mathrm { d o w n } }$ matrix reads an intermediate vector, namely, the output of SiLU. The read-write separation allows us to quantitatively isolate and systematically measure the influence of every residual coordinate on the input and output side of every operation, within every block, of every layer of the Transformer.

Massive Activation Coordinates. We now formally identify the set $\mathcal { M }$ of massively activated coordinates. Let $Z _ { t , k } ^ { l } ( s )$ be the residual coordinate k in the output of layer l and token position t for an input sequence s from a held-out corpus. We define the set of massive coordinates as $\begin{array} { r c l } { \mathcal { M } } & { = } & { \left\{ \ k : \ \operatorname* { m a x } _ { s , l , t } \frac { | Z _ { t , k } ^ { l } ( s ) | } { \mathrm { m e d i a n } _ { j } | Z _ { t \ j } ^ { l } ( s ) | } > R \right\} } \end{array}$ . A coordinate belongs to M if it exceeds the threshold at any evaluated sequence, layer, or token position. Unless otherwise specified, we use $R = 5 ;$ our results are robust to changes in this threshold (Appendix H). We then compare the read and write behavior of Transformer blocks on coordinates in M and its complement set $\neg { M }$

Measuring Read-blindness. Let A be a given read-side matrix. A is blind to coordinate k when values in coordinate k cause a weak (near-zero) response. For instance, if the standard vector $e _ { k }$ corresponding to coordinate k has $A e _ { k }$ close to zero, we say that A is blind to coordinate k. Based on this intuition, we define the read-blindness (erasure) of a matrix A to coordinate k as follows:

$$
\operatorname { e r a } _ { k } ( G ) = 1 / ( 1 + d _ { m } G _ { k k } / \operatorname { t r } ( G ) ) \ \in \ ( 0 , 1 ] .\tag{1}
$$

where $G = A ^ { \top } A$ is the input-side Gram matrix of A. We use Gram matrix instead of the matrix itself as both $G$ and A have the same null space; G being positive semi-definite makes it a more stable measure for assessing read-blindness. The read-blindness $\mathrm { e r a } _ { k } ( G )$ is close to 1 when $G _ { k k }$ is small relative to the trace, and close to 0 when $G _ { k k }$ is large. Under uniform treatment across coordinates, the expected score is 0.5.

To assess whether $\mathcal { M }$ has systematically higher erasure scores than a control set $\neg { \mathcal { M } } ,$ we use a one-sided Mann-Whitney $\dot { U }$ test on the layer-wise maximum of $\mathrm { e r a } _ { k }$ for each feature k. We use either a size-matched sample or the full complement as specified in each analysis. We define $U$ to count pairwise wins of features in M over features in $\neg { M }$ , assigning half a win to ties, and report the rank-biserial correlation $r _ { b } = 2 U / ( | \mathcal { M } | | \neg \mathcal { M } | ) - 1 \in [ - 1 , \overline { { + 1 } } ]$ as an effect-size measure, together with the standardized z-score of the U statistic. A value $r _ { b } > 0 . 9 5$ with $p \ll 0 . 0 5$ indicates that virtually every feature in M has a higher maximum erasure score than virtually every feature in the control set on that Gram matrix. As a complementary spectral diagnostic, we also measure how strongly each coordinate aligns with the approximate null subspace of G. We define this null occupancy measure and report its results in Appendix C.

## READ BLINDNESS IN TRANSFORMERS

We now investigate whether massive coordinates exhibit the read–write asymmetry described above, and whether this behavior persists across model scale and architecture. We analyze three Llama models spanning 135M, 1.28B, and 2.56B parameters, together with a Qwen3 1.7B model. All four models use the Pre-LN residual architecture introduced in Section 3, with attention and SwiGLU FFN blocks that expose distinct read and write pathways. The Llama models allow us to test whether the phenomenon is stable over an order-of-magnitude change in scale, while Qwen3 tests whether it extends beyond a single model family.

For each model and operator role, Table 1 reports the median over $k \in \mathcal { M }$ of the layer-wise maximum erasure score, with the rank-biserial effect against the full complement $\neg { M }$ in parentheses. We focus the main-text analysis on this erasure statistic and report the complementary null-occupancy results in Appendix C. We analyze the learned projections (Weight View) and the block inputs and residual updates observed on held-out data (Operator View).

Weight View. On the read side, we apply the input-Gram construction from Section 3 to $W _ { V } , W _ { Q }$ $W _ { K }$ , and the FFN gate/up projections. The write side instead requires a Gram over residual output coordinates. For attention, let h denote the concatenated head output, so that the residual update is $W _ { O } h$ We use $G _ { O } ^ { \mathrm { w r i t e } } = W _ { O } W _ { O } ^ { \top }$ , whose kth diagonal entry is $( G _ { O } ^ { \mathrm { w r i t e } } ) _ { k k } = \| W _ { O } ^ { \top } e _ { k } \| _ { 2 } ^ { 2 } .$ , the total squared weight through which the attention output can modify residual coordinate k. High write-side erasure indicates weak structural access to that coordinate, while low erasure indicates an open write pathway. We apply the same construction to $W _ { \mathrm { d o w n } } ;$ precise definitions of all weightside Grams are given in Appendix B. Note that Weight View on the write side is an approximation: the actual write operations are dependent on the Attention and FFN operators which are nonlinear functions of the residual.

Operator View. As we mentioned above, the write-side weight Grams are an approximation of the actual write operations by the operators of Attention and FFN. The exact results are empirical second moments over a corpus. For attention, $G ^ { a , o i } = \mathbb { E } [ \widetilde { X } \widetilde { X } ^ { \top } ]$ measures mean squared value (we use the shorthand energy from here on) of each coordinate at the normalized input, while $G ^ { a , o o } =$ $\mathbb { E } [ ( Y - X ) ( Y - X ) ^ { \top } ]$ measures the energy of the attention residual update at each coordinate. We analogously use $G ^ { f , o i } = \mathbb { E } [ \widetilde { Y } \widetilde { Y } ^ { \top } ]$ and $G ^ { f , o o } = \mathbb { E } [ ( Z - Y ) ( Z - Y ) ^ { \top } ]$ for the FFN in the same way. High erasure on an operator-input Gram means a coordinate carries little magnitude at the input; this differs from erasure on a weight Gram, which means the following block cannot read the coordinate at all. High erasure on an operator-output Gram means the block writes little to that coordinate. The input Grams show whether massive coordinates are present to be read; the output Grams show whether the write pathways identified above are used in practice.

## 4.1 MAIN RESULTS

Massive coordinates are present in the normalized inputs to both blocks. On the operator-input Grams, the $\scriptstyle \mathcal { M } \scriptstyle \mathrm { - v e r s u s - } \mathcal { M }$ rank-biserial is negative in every model, for both $\widetilde { X }$ and ${ \widetilde { Y } } ;$ massive coordinates are erased less than the non-massive ones, so they carry a larger mean squared value (energy) at the input. Their downstream suppression therefore comes from the learned read projections. The signal is present at the input after normalization.

This suppression is strongest and most consistent in $W _ { V } , \ W _ { Q } ,$ , and the FFN gate/up projections. Across all four models, these projections show near-complete separation between massive coordinates and the full complement: $r _ { b } \ge 0 . 9 9$ for $W _ { V }$ and the FFN projections, and $r _ { b } ~ \ge ~ 0 . 9 6$ for $W _ { Q }$ . The coordinate-wise erasure evidence is weaker for $W _ { K }$ , where $r _ { b }$ ranges from 0.11 to 0.31. The spectral diagnostic in Appendix C, however, shows that massive coordinates consistently have greater occupancy in the approximate null subspace of $W _ { K }$ , with effects ranging from $r _ { b } = 0 . 8 6$ for Llama 135M to $r _ { b } = 1 . 0 0$ for Qwen3 1.7B (Table 4). Thus, $W _ { K }$ exhibits strong null-subspace alignment, although its total response to these coordinates is only modestly attenuated.

Both blocks can still write to massive coordinates. For the write matrices $W _ { O }$ and $W _ { \mathrm { d o w n } }$ , erasure stays near the uniform baseline and the $\scriptstyle \mathcal { M } \scriptstyle \mathrm { - v e r s u s - } \mathcal { M }$ rank-biserial is negative in every model, so massive coordinates sit inside the write range of attention and FFN even though the read side suppresses them. The operator-output Grams agree: the rank-biserial is negative in every model, so both blocks deposit a larger energy to massive coordinates than to the non-massive ones on held-out data. Note that because the second moment is not signed, we cannot tell from it if the deposits coordinate in being additive across layers. We investigate that next.

Table 1: Erasure results. Cells report the median era<sub>M</sub> with rank-biserial $r _ { b }$ against the full complement ¬M in parentheses. Bold cells have $r _ { b } > 0 . 9 5 $ stars denote one-sided Mann–Whitney tests of era<sub>M</sub> > era $\neg { \mathcal { M } }$ with $p < 0 . 0 5$
<table><tr><td></td><td>Llama 135M</td><td>Llama 1.28B</td><td>Llama 2.56B</td><td>Qwen3 1.7B</td></tr><tr><td>Massive coordinates |M|</td><td>10</td><td>10</td><td>5</td><td>7</td></tr><tr><td>Maximum activation magnitude</td><td>280</td><td> $2 . 2 7 \times 1 0 ^ { 3 }$ </td><td> $3 . 9 2 \times 1 0 ^ { 3 }$ </td><td> $2 . 2 3 \times 1 0 ^ { 3 }$ </td></tr><tr><td>Input availability:</td><td></td><td></td><td></td><td></td></tr><tr><td>Attention normalized input,</td><td>0.58 (-0.60)</td><td>0.68 (-0.73)</td><td>0.55 (-0.64)</td><td>0.58 (-0.37)</td></tr><tr><td>FFN normalized input,</td><td>0.56 (-0.65)</td><td>0.56 (-0.13)</td><td>0.50 (-0.63)</td><td>0.54 (-0.48)</td></tr><tr><td>Read side weight view:</td><td></td><td></td><td></td><td></td></tr><tr><td>Value projection  $W _ { V }$ </td><td> $\mathbf { 0 . 7 0 \left( + 0 . 9 9 \right) ^ { \ast } }$ </td><td> $\mathbf { 0 . 7 5 \left( + 0 . 9 9 \right) ^ { \ast } }$ </td><td> $\mathbf { 0 . 7 8 \ : ( + 1 . 0 0 ) } ^ { \ast }$ </td><td> ${ \bf 0 . 7 6 \left( + 1 . 0 0 \right) ^ { * } }$ </td></tr><tr><td>Query projection  $W _ { Q }$ </td><td> $\mathbf { 0 . 6 3 \left( + 0 . 9 9 \right) ^ { \ast } }$ </td><td> $\mathbf { 0 . 6 5 \left( + 0 . 9 9 \right) ^ { \ast } }$ </td><td> $\mathbf { 0 . 6 4 } \left( + \mathbf { 0 . 9 9 } \right) ^ { \ast }$ </td><td> ${ \bf 0 . 6 6 \left( + 0 . 9 6 \right) } ^ { \ast }$ </td></tr><tr><td>Key projection  $W _ { K }$ </td><td> $0 . 5 6 \left( + 0 . 2 9 \right)$ </td><td> $0 . 5 5 \left( + 0 . 3 1 \right) ^ { \ast }$ </td><td> $0 . 5 3 \ ( + 0 . 1 1 )$ </td><td> $0 . 5 9 \left( + 0 . 2 5 \right)$ </td></tr><tr><td>FFN gate/up  $W _ { \mathrm { g a t e } } , W _ { \mathrm { u p } }$ </td><td>0.66 (+0.99)</td><td> $\mathbf { 0 . 6 3 } \left( \mathbf { + 1 . 0 0 } \right) ^ { \ast }$ </td><td> $\mathbf { 0 . 5 9 } \left( \mathbf { + 0 . 9 9 } \right) ^ { \ast }$ </td><td>0.62 (+0.99)</td></tr><tr><td>Write side weight view:</td><td></td><td></td><td></td><td></td></tr><tr><td>Attention output  $W _ { O }$ </td><td>0.51 (-0.82)</td><td>0.51 (-0.61)</td><td>0.49 (-0.59)</td><td>0.51 (-0.15)</td></tr><tr><td>FFN down projection  $W _ { \mathrm { d o w n } }$ </td><td>0.49 (-0.86)</td><td>0.49 (-0.64)</td><td>0.49 (-0.46)</td><td>0.49 (-0.88)</td></tr><tr><td>Write side operator view:</td><td></td><td></td><td></td><td></td></tr><tr><td>Attention update  $\Delta ^ { \mathrm { a t t n } }$ </td><td>0.61 (-0.87)</td><td> $0 . 8 5 \ ( - 0 . 4 4 )$ </td><td>0.62 (-0.64)</td><td>0.64 (-0.36)</td></tr><tr><td>FFN residual update  $\Delta ^ { \mathrm { f f n } }$ </td><td>0.81 (-0.29)</td><td>0.69 (-0.77)</td><td>0.59 (-0.59)</td><td>0.56 (-0.60)</td></tr></table>

Table 2: Signed-alignment results. For each coordinate, S is median-aggregated across layers. Cells report the median $S _ { \mathcal { M } }$ with rank-biserial $r _ { b }$ comparing M and $\neg { M }$ in parentheses.
<table><tr><td>Signed alignment</td><td>Llama 135M</td><td>Llama 1.28B</td><td>Llama 2.56B</td><td>Qwen3 1.7B</td></tr><tr><td>Attention,  $X _ { k }$  with  $\Delta _ { k } ^ { \mathrm { a t t n } }$ </td><td>+0.03 (+0.45)</td><td>+0.09 (+0.29)</td><td>+0.02 (-0.27)</td><td>+0.06 (+0.16)</td></tr><tr><td>FFN,  $Y _ { k }$  with  $\Delta _ { k } ^ { \mathrm { f f n } }$ </td><td>+0.05 (+0.57)</td><td>+0.25 (+0.59)</td><td>-0.02 (-0.20)</td><td> $+ 0 . 1 0 \left( + 0 . 1 4 \right)$ </td></tr></table>

Signed deposits. To investigate signed direction of the deposits acorss layers, we compute $S _ { \ell , k } ^ { b } =$ $\mathbb { E } _ { s , t } [ \mathrm { s i g n } ( R _ { \ell , k } ^ { b } ) \Delta _ { \ell , k } ^ { b } ]$ , using $( R ^ { b } , \Delta ^ { b } ) = ( X , Y - X )$ for attention and $( Y , Z - Y )$ for the FFN. Positive values mean that the coordinate’s current sign is reinforced whereas negative values mean that they partially cancel. Table 2 reports the median over layers and then over $k \in \mathcal { M }$ , with the rank-biserial effect against the full complement $\neg { \mathcal { M } }$

## 4.2 PROBING ON READ-BLINDNESS

The previous section showed that several read pathways are blind to massive coordinates. Is blindness in any one of these pathways necessary, or can the model preserve the asymmetry when that pathway is forced open? We test this by preventing either the attention value projection or the FFN input projections from becoming read-blind during training. We then ask whether massive activations disappear and, if not, where the read-blindness reconfigures.

Forcing a read pathway open. We train three variants of the Llama 1.28B model using the same hyperparameters as the base model. In the $W _ { V }$ Frozen variant, the combined $W _ { V }$ is initialized as a random orthogonal matrix and held fixed throughout training. It therefore has no null space, but also no freedom to learn. The $W _ { V }$ Reparam variant instead uses a Cayley parametrization, which allows $W _ { V }$ to learn during training while keeping it orthogonal. In the third variant, FFN Frozen, we initialize $W _ { \mathrm { g a t e } }$ and $\bar { W _ { \mathrm { u p } } }$ orthogonally and freeze them. We do not use a Cayley parametrization for these large rectangular matrices because of its computational cost and large number of strategies available in such a construction with nothing to motivate a particular choice. Relative to the basemodel perplexity of 14.21, perplexity changes to 14.67 (+0.46) for $W _ { V }$ Frozen, 14.24 (+0.03) for $W _ { V }$ Reparam, and 13.67 (−0.54) for FFN Frozen. Table 5 reports the erasure results; the complementary null-occupancy results are reported in Table 6 in the appendix.

Read-blindness re-configures elsewhere. Both $W _ { V }$ interventions work as intended. The erasure and null-occupancy scores of $W _ { V }$ are 0.50, with no separation between massive and control coordinates. In the Frozen variant, number of massive coordinates is reduced to $^ { 7 , }$ and the maximum activation magnitude falls to $2 . 0 8 \times 1 0 ^ { 3 }$ . In the Reparam variant, the number of massive coordinates is 10, and the maximum magnitude is $2 . 1 2 \times 1 0 ^ { 3 }$ , slightly lower compared to the base model. Nevertheless, massive activations remain. With $W _ { V }$ forced open, read-blindness remains in the other read pathways. In both variants, $W _ { Q }$ and the FFN gate/up projections almost completely separate massive coordinates from their controls, with $r _ { b } \ge 0 . 9 9$ under both diagnostics. $\bar { W _ { K } }$ also remains strongly aligned with its approximate null subspace, with null-occupancy effects of $r _ { b } = 0 . 9 8$ and 0.99. Thus, removing read-blindness from $\hat { W _ { V } }$ re-distributes its role across the remaining read operators rather than eliminating it from the model.

The FFN intervention shows the same pattern in the opposite direction. Freezing $W _ { \mathrm { g a t e } }$ and $W _ { \mathrm { u p } }$ brings both of their blindness scores back to the uniform baseline of approximately 0.50. Readblindness remains strong in attention. The effects are $r _ { b } = 0 . 9 8$ for $W _ { V }$ and $r _ { b } = 0 . 9 9$ for $W _ { Q }$ while $W _ { K }$ has a null-occupancy effect of $r _ { b } = 0 . 9 7$ . The model still develops 11 massive coordinates, although their maximum magnitude falls to 681. Opening the FFN read pathway therefore weakens the largest activation, but does not remove massive activations. Because these FFN projections are frozen, part of this reduction may also come from their loss of learning capacity (reduced number of trainable parameters).

The re-distribution is confined to the read side. Across all three variants, $W _ { O } , W _ { \mathrm { d o w n } }$ , and the realized attention and FFN updates show no preferential suppression of massive coordinates. The write pathways remain open and continue to deposit energy into these coordinates.

These interventions demonstrate that no single read projection is responsible for the read–write asymmetry. When one pathway is forced open, read-blindness is re-distributed across the pathways that remain available, while the write side stays open. MAs are therefore maintained and reconfigured in a collaboration between Attention and FFN operators. This result was a surprise to us.

## 5 SUN’S AMPLIFIER AND THE GEOMETRY OF FFN READ-BLINDNESS

We now investigate the relationship of our observations so far to previous work on FFN in Sun et al. (2026). They blame the origins of massive activations exclusively on a directional FFN amplifier. They formulate FFN’s write to coordinate k as a quadratic form in the normalized residual ${ \widetilde { h } } .$

$$
y _ { k } ( \widetilde { h } ) \approx \widetilde { h } ^ { \top } S _ { k } \widetilde { h } , \qquad S _ { k } = \frac { 1 } { 2 } ( U _ { k } + U _ { k } ^ { \top } ) , \qquad U _ { k } = \sum _ { i } W _ { \mathrm { d o w n } } [ k , i ] W _ { \mathrm { g a t e } } [ i , : ] ^ { \top } W _ { \mathrm { u p } } [ i , : ] .
$$

Sun et al. (2026) show that under approximation (Appendix F), $\mathrm { F F N } ( \widetilde { h } ) _ { k } \approx \lambda _ { \star } ^ { ( k ) } ( s _ { \star } ^ { ( k ) \top } \widetilde { h } ) ^ { 2 }$ , where $s _ { \star } ^ { ( k ) }$ is the unit input direction (eigen-vector of $S _ { k } )$ that drives output coordinate k and ${ \lambda } _ { \star } ^ { ( k ) }$ is its signed gain (corresponding eigen-value). Note that since $s _ { \star } ^ { ( k ) }$ need not align with the residual coordinate direction $e _ { k }$ , the FFN can be blind to the value stored in k while still reading $s _ { \star } ^ { ( k ) }$ and writing a large update to k proportional to the signed-gain ${ \lambda } _ { \star } ^ { ( k ) }$ . This will become important in our analysis.

Amplifier specificity lies in gain and spectral concentration Each output coordinate k has an amplifier direction $\bar { s } _ { \star } ^ { ( k ) }$ and a corresponding gain ${ \lambda } _ { \star } ^ { ( k ) }$ . We ask which of these is distinctive for massive coordinates: do they share a special amplifier direction, or do they instead have unusually strong amplification along their respective directions? We test direction using the within-set absolute cosine, which measures how strongly the amplifier directions for different coordinates align. We test amplifier strength using $\| U _ { k } \| _ { F }$ , the overall scale of the quadratic construction, and $| \lambda _ { \star } ^ { ( k ) } |$ , the gain along $s _ { \star } ^ { ( k ) }$ . Finally, $| \lambda _ { \star } ^ { ( k ) } | / \| S _ { k } \| _ { F }$ measures how strongly the quadratic form is concentrated in that dominant direction. We report an additional subspace-alignment diagnostic in Appendix F.

Table 3 shows that direction is not the main distinction. The within-set absolute cosine is only moderately higher for M than for the size-matched non-massive set $c { \mathcal { M } } \subset \neg { \mathcal { M } }$ (0.53 versus 0.46), so massive coordinates do not share a clearly distinctive amplifier direction. Gain provides a much sharper distinction: $\| U _ { k } \| _ { F }$ and $| \lambda _ { \star } ^ { ( k ) } |$ | have rank-biserial effects of 0.78 and 0.98, respectively. The strongest discriminator is the concentration $| \lambda _ { \star } ^ { ( k ) } | / \| S _ { k } \| _ { F }$ , which separates the two groups perfectly in this sample $( r _ { b } = 1 . 0 0 ) $ .

We observe that massive coordinates are strongly distinguished by having higher quadratic gain concentrated in a dominant direction, indicating a shift toward approximately rank-one behavior of the amplifier. However, the specific direction in the FFN’s internal space is coordinate-specific and is not shared among the massive coordinates. Intuitively, this lack of collusion is perhaps unsurprising because the FFN has a separate quadratic form $S _ { k }$ for each

Table 3: s amplifier diagnostics for Llama 1.28 Base. Per-coordinate entries are medians over M and a deterministic, size-matched $\neg { \mathcal { M } } ;$ the final row is the layermean of the within-set median pairwise absolute cosine. $r _ { b } > 0$ means larger values on ${ \bar { \mathcal { M } } } .$
<table><tr><td>Diagnostic</td><td>value M</td><td> $r _ { b }$ </td></tr><tr><td> $\| U _ { k } \| _ { F }$ </td><td>7.74</td><td>+0.78</td></tr><tr><td> $| \lambda _ { \star } ^ { ( k ) } |$ </td><td>2.19</td><td> $+ 0 . 9 8$ </td></tr><tr><td> $| \lambda _ { \star } ^ { ( k ) } | / \| S _ { k } \| _ { F }$ </td><td>0.39</td><td> $+ 1 . 0 0$ </td></tr><tr><td> $\scriptstyle \operatorname { W i t h i n - s e t } | \cos ( s _ { \star } ) |$ </td><td>0.53/0.46</td><td></td></tr></table>

output coordinate. What is surprising is the collapse of these forms toward a single direction with high gain. What occurs first: read-blindness or this collapse? We answer that next.

Read-blindness is detected before amplifier specialization. We track FFN read-blindness and amplifier gain through training. For each checkpoint and coordinate, we take the maximum across layers of the FFN erasure $\mathrm { e r a } _ { k }$ derived from FFN’s read-side matrices and the amplifier gain $| \lambda _ { \star } ^ { ( k ) } |$ We assign $\mathcal { M }$ and the non-massive control set using the final checkpoint, and then trace these fixed sets backward through training. This lets us ask which FFN signature first distinguishes the coordinates that eventually become massive. We concede that this analysis tests temporal ordering, not causal dependence.

Figure 1(a–b) summarizes the comparison. We define the separation time $t _ { \mathrm { s e p } }$ as the first checkpoint at which the rank-biserial effect between eventual massive and control coordinates reaches $r _ { b } \ge 0 . 9 5$ . FFN read-blindness separates the groups at 2.3k steps (Figure 1(a)), whereas amplifier gain separates them later, at 4.5k steps (Figure 1(b)). Thus, FFN read-blindness is already detectable before the amplifier specializes. This temporal precedence does not establish that read-blindness causes amplifier specialization; it shows that read-blindness is the earlier structural signature of the coordinates that eventually become massive. Together with Section 4, the results support complementary roles: the amplifier explains how selected coordinates receive exceptional writes, while FFN read-blindness explains why subsequent FFN blocks do not condition corrective updates on the massive values already present.

$W _ { \mathbf { d o w n } }$ concentration and the read-blindness leak. The FFN amplifier poses a puzzle. The FFN is read-blind to massive coordinates, so how does it nevertheless produce large writes to them? The blindness is not total. Attention erases massive coordinates more strongly than the FFN: in the 1.28B model, $W _ { V }$ erasure is 0.75, compared with 0.63 for $W _ { \mathrm { g a t e } } , W _ { \mathrm { u p } }$ (Table 1); the same gap appears through training in Figure 3. This difference is plausible since, unlike attention, the FFN acts independently at each token. The weaker FFN erasure therefore leaves a partial read pathway through which some FFN intermediate channels can still respond to massive residual coordinates.

Under the quadratic approximation, the FFN writes strongly to residual coordinate k when the input $\widetilde { h } \mathrm { a l i g n s } \mathrm { w i t h } s _ { \star } ^ { ( k ) }$ , the top eigenvector of $S _ { k }$ . From the sum defining $U _ { k }$ , this direction is determined by the read directions $W _ { \mathrm { g a t e } } [ \bar { i } , : ]$ and $W _ { \mathrm { u p } } [ i , : ]$ of the intermediate channels i weighted by $W _ { \mathrm { d o w n } } [ k , i ]$ For the partial read pathway to produce a strong amplifier, two conditions are needed. First, some intermediate channels that contribute to output coordinate k must retain read access to the massive residual coordinates. Second, the write row $W _ { \mathrm { d o w n } } [ k , : ]$ must concentrate on those channels. Then $S _ { k }$ is dominated by a few read directions and can develop a high-gain dominant mode; if $W _ { \mathrm { d o w n } } [ k , : ]$ is spread across many unrelated units, their contributions are diluted.

![](images/9638aacb39218cf8ffa65edc671ece20cf4093de8b1ab96dd7305e45af742002.jpg)  
(a) FFN read-blindness

![](images/5681ad510b2384df044e806adc5cfa387506f565262da3663dae3f26a9c8e591.jpg)  
(b) Amplifier gain

![](images/700b85d90405643c9aca1080ba6e02277bf283665e4a6a75a9a9bc4831a8dbde.jpg)  
(c) Write-row concentration vs. amplifier norm  
Figure 1: FFN read-blindness precedes amplifier specialization. We assign massive and control coordinates at the final checkpoint and trace these fixed sets through 21 checkpoints from 0 to 20k training steps. (a) Maximum-over-layer FFN erasure for eventual massive coordinates (orange) and non-massive controls (gray). (b) Maximum-over-layer leading amplifier gain for the same sets. Lines and shaded regions show medians and interquartile ranges; vertical dotted lines mark $t _ { \mathrm { s e p } } .$ the first checkpoint at which the rank-biserial effect reaches $r _ { b } \ge 0 . 9 5$ FFN erasure separates at 2.3k steps, before amplifier gain at 4.5k steps. (c) Final-checkpoint amplifier norm $\| U _ { k } \| _ { F }$ versus the inverse participation ratio of the corresponding FFN write row $W _ { \mathrm { d o w n } } [ k , : ]$ . Colored points are massive coordinates, with size denoting massiveness and color denoting attention-input erasure; gray points are non-massive coordinates.

The hypothesis predicts that FFN gain on coordinate k grows with the concentration of $W _ { \mathrm { d o w n } } [ k , : ]$ We measure concentration using the inverse participation ratio (IPR) metric defined below.

$$
\mathrm { I P R } \big ( W _ { \mathrm { d o w n } } [ k , : ] \big ) = \frac { \sum _ { i } W _ { \mathrm { d o w n } } [ k , i ] ^ { 4 } } { \big ( \sum _ { i } W _ { \mathrm { d o w n } } [ k , i ] ^ { 2 } \big ) ^ { 2 } } \in \left[ 1 / d _ { \mathrm { f f n } } , 1 \right]
$$

A higher IPR means that the write row for residual coordinate k is concentrated on fewer FFN intermediate units. Figure 1(c) reports the result. Within $\mathcal { M }$ , residual coordinates with more concentrated $W _ { \mathrm { d o w n } }$ rows have larger amplifier norm $\| U _ { k } \| _ { F } ,$ , consistent with concentration turning the partial FFN read pathway into a strong amplifier. Why does $W _ { \mathrm { d o w n } } [ k , : ]$ concentrate? We suspect a token-independent coordinate is served by a few dedicated FFN intermediate channels, with weight decay and small-initialization gradient descent favoring the low-norm row that uses only those channels.

## 6 TRAINING ACTIVELY REINFORCES MASSIVE ACTIVATIONS

The intervention experiments show that the model can reorganize its read pathways to preserve read-blindness. We now ask whether this configuration is also visible in the local loss geometry, and whether the next optimizer step would preserve or reduce the resulting massive activations.

The loss geometry follows the read–write asymmetry. We borrow the language of stiff and sloppy directions from Transtrum et al. (2015). A direction is stiff when moving away from trained parameters increases the loss, and sloppy when the parameters can move without incurring much penalty. We measure this sensitivity using the directional Hessian curvature $v ^ { \top } H v$ . For each parameter slice associated with a residual coordinate, we sample multiple random unit directions v and average their curvature. Positive curvature indicates a locally stable stiff direction, curvature near zero suggests a flat direction, and negative curvature indicates that the parameters do not lie in a locally convex basin along that direction. Appendix G.1 gives the full definition and sampling procedure.

Figure 2 shows that this geometry separates the two operator roles. Curvature is higher on M than on ¬M for the read-side matrices $W _ { V } , W _ { \mathrm { g a t e } }$ , and $W _ { \mathrm { u p } } ^ { \mathrm { - } } .$ . Moving these parameters is therefore more costly. The pattern reverses for the write-side matrices $\mathbf { \bar { \boldsymbol { W } } } _ { O }$ and $\bar { W } _ { \mathrm { d o w n } }$ , whose curvature is lower on M. Thus, the loss constrains how the model reads MAs more strongly than how it writes to them. This geometric asymmetry matches the read-blind, write-open structure found in Section 4.

For deep strong-amplifier M rows of $W _ { \mathrm { d o w n } } .$ , the cross-entropy gradient locally favors increasing the row norm, while weight decay opposes this tendency (Appendix G.1 and Figure 5). This partial force balance does not determine the net effect of AdamW optimizer update.

The full AdamW step increases massive-activation magnitude. We therefore measure the end effect directly. At the trained checkpoint, we apply a synthetic next AdamW step to the block that produces each target activation and compute its first-order effect on the activation’s $L _ { 2 }$ norm. We call this gradient pressure; its definition and the exact AdamW decomposition are given in Appendix G.3. As shown in figure $^ { 6 , }$ the full update has positive pressure on $\mathcal { M }$ at every layer, with a larger effect in later layers, while its pressure on matched non-massive coordinates stays near zero. Decomposing the step shows that the positive pressure comes mainly from AdamW’s saved moment state. The new batch gradient has a much smaller effect, and weight decay reduces rathe than increases the massive activations.

Together, the two analyses answer different parts of the question. The curvature analysis shows that massive coordinates occupy a distinct read–write geometry in parameter space. The optimizer analysis shows the next AdamW step is poised to reinforce their activation magnitude, with the accumulated optimizer state overcoming weight decay. This is local evidence that training actively maintains massive activations rather than merely tolerating them.

## 7 CONCLUSION

![](images/fad4f8280da642eb5ed92862b37b41f8bc08e239593564bfb9afe633169f3d20.jpg)

Massive activations persist because the transformer architecture systematically learns to ignore them while reading, yet continues to amplify them while writing. Our findings reveal that this read-write asymmetry is not an architectural quirk, but a systems-level configuration strictly enforced by the training dynamics and loss landscape. By constraining the $W _ { V } , W _ { \mathrm { u p } }$ and $W _ { \mathrm { g a t e } }$ matrices, we demonstrated that MAs are not tied to a single component; rather, the

Figure 2: Directional curvature on $\mathcal { M }$ versus $c \mathcal { M }$ in the 1.28B base model (30 batches, 64 directions). Read-side slices are stiffer on $\mathcal { M } ;$ writeside slices are sloppier. Error bars are bootstrap 95% confidence intervals.

network actively reorganizes read-blindness to other operators like the FFN. We speculate that readblindness is a cheap route for the model to maintain token-independent features, which are perhaps necessary for modeling language. However, the model architecture, especially the quadratic amplifier in FFN and lack of feedback control in write operations for all operators, can amplify a subset of these features to massive values as an unintended consequence.

Specifically, our analysis refines the directional-amplifier hypothesis. Massive coordinates do not share a distinctive amplifier direction; instead, they are distinguished by unusually high quadratic gain concentrated in their respective dominant directions. FFN read-blindness becomes detectable before this amplifier specialization, suggesting complementary rather than competing roles: the amplifier explains how selected coordinates receive exceptional FFN writes, while read-blindness explains why subsequent blocks do not condition corrective updates on the massive values already present. Within M, stronger amplification is associated with greater concentration of the corresponding $W _ { \mathrm { d o w n } }$ row, consistent with a partial FFN read pathway being concentrated into a dominant mode.

Our work provides strong evidence that the MA state is a fixed point actively maintained by the optimizer. Read-side rows exhibit high curvature, heavily penalizing deviations from the read-blind state, while write-side rows sit at a saddle point stabilized by a delicate force balance: the crossentropy loss pulls the weights outward, while weight decay pushes them inward. AdamW optimizer actively maintains the MA via its momentum state. By identifying read-blindness as the core mechanism, we provide a compelling mechanistic explanation for why subsequent layers in a transformer fail to normalize these extreme outliers.

## REFERENCES

Yongqi An, Xu Zhao, Tao Yu, Ming Tang, and Jinqiao Wang. Systematic outliers in large language models. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025. URL https://openreview.net/ forum?id=rLX7Vyyzus.

Yelysei Bondarenko, Markus Nagel, and Tijmen Blankevoort. Quantizable transformers: Removing outliers by helping attention heads do nothing. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (eds.), Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper\_files/paper/2023/hash/ edbcb7583fd8921dad78adecfe06a99b-Abstract-Conference.html.

Nicola Cancedda. Spectral filters, dark signals, and attention sinks. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pp. 4792–4808. Association for Computational Linguistics, 2024. doi: 10.18653/V1/2024. ACL-LONG.263. URL https://doi.org/10.18653/v1/2024.acl-long.263.

Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. Gpt3.int8(): 8-bit matrix multiplication for transformers at scale. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems, volume 35, pp. 30318–30332. Curran Associates, Inc., 2022. doi: 10.52202/ 068431-2198. URL https://proceedings.neurips.cc/paper\_files/paper/ 2022/file/c3ba4962c05c49636d4c6206a97e9c8a-Paper-Conference.pdf.

Jorge Gallego-Feliciano, S. Aaron McClendon, Juan Morinelli, Stavros Zervoudakis, and Antonios Saravanos. Hidden dynamics of massive activations in transformer training. CoRR, abs/2508.03616, 2025.

Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang, Jiawei Han, and Jianfeng Gao. Model tells you what to discard: Adaptive KV cache compression for llms. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id=uNrFpDPMyo.

Xiangming Gu, Tianyu Pang, Chao Du, Qian Liu, Fengzhuo Zhang, Cunxiao Du, Ye Wang, and Min Lin. When attention sink emerges in language models: An empirical view. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025. URL https://openreview.net/forum?id=78Nn4QJTEN.

Prannay Kaul, Chengcheng Ma, Ismail Elezi, and Jiankang Deng. From attention to activation: Unraveling the enigmas of large language models. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025. URL https://openreview.net/forum?id=IjduZQK8gM.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net, 2019. URL https://openreview.net/forum?id=Bkg6RiCqY7.

Louis Owen, Nilabhra Roy Chowdhury, Abhay Kumar, and Fabian Gura. A refined analysis of¨ massive activations in llms. CoRR, abs/2503.22329, 2025. doi: 10.48550/ARXIV.2503.22329. URL https://doi.org/10.48550/arXiv.2503.22329.

Enrique Queipo-de-Llano, Alvaro Arroyo, Federico Barbero, Xiaowen Dong, Michael M. Bronstein, Yann LeCun, and Ravid Shwartz-Ziv. Attention sinks and compression valleys in llms are two sides of the same coin. In Carl Vondrick, Bharath Hariharan, Colin Raffel, Lerrel Pinto, Diyi Yang, and Aleksandra Faust (eds.), The Fourteenth International Conference on Learning Representations, ICLR 2026, Rio de Janeiro, Brazil, April 23-27, 2026. proceedings.iclr.cc, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/hash/ 1734b19d9afe7d2c7f1154954eaf0d5a-Abstract-Conference.html.

Mingjie Sun, Xinlei Chen, J. Zico Kolter, and Zhuang Liu. Massive activations in large language models. CoRR, abs/2402.17762, 2024. doi: 10.48550/ARXIV.2402.17762. URL https:// doi.org/10.48550/arXiv.2402.17762.

Shangwen Sun, Alfredo Canziani, Yann LeCun, and Jiachen Zhu. The spike, the sparse and the sink: Anatomy of massive activations and attention sinks. CoRR, abs/2603.05498, 2026. doi: 10. 48550/ARXIV.2603.05498. URL https://doi.org/10.48550/arXiv.2603.05498.

Qwen Team. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton-Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabe Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez,´ Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and finetuned chat models. CoRR, abs/2307.09288, 2023. doi: 10.48550/ARXIV.2307.09288. URL https://doi.org/10.48550/arXiv.2307.09288.

Mark K Transtrum, Benjamin B Machta, Kevin S Brown, Bryan C Daniels, Christopher R Myers, and James P Sethna. Perspective: Sloppiness and emergent theories in physics, biology, and beyond. The Journal ofchemical physics, 143(1), 2015.

Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id=NG7sS51zVF.

Mengxia Yu, De Wang, Qi Shan, Colorado Reed, and Alvin Wan. The super weight in large language models. CoRR, abs/2411.07191, 2024.

## A DEFINITIONS AND TRANSFORMER ARCHITECTURE

Let $d _ { m }$ be the residual dimension, H the number of attention heads, $d _ { h } = d _ { m } / H$ the dimension of each head, and $d _ { f f }$ the FFN hidden dimension. We represent the residual stream after layer l as $Z ^ { l } = [ z _ { 1 } ^ { l } , \dots , z _ { T } ^ { l } ] \ { \stackrel { \cdot \cdot } { \in } } \mathbb { R } ^ { d _ { m } \times T }$ , with one token per column. For a single Pre-LN layer with input X, we write

$$
Y = X + \mathrm { M H A } ( \widetilde { X } ) , \qquad Z = Y + \mathrm { F F N } ( \widetilde { Y } ) ,
$$

where X is the layer input and $\widetilde { X } \ = \ \mathrm { R M S N o r m } ( X )$ and $\widetilde { Y } \ = \ \mathrm { R M S N o r m } ( Y )$ are computed independently for each token. Below, we also write ${ \dot { Z } } _ { 0 } = X , Z _ { 1 } = Y$ , and $Z _ { 2 } = Z .$

We use column-vector linear maps of the form $y = A x$ . Each attention head h therefore has read matrices ${ \boldsymbol { W } _ { Q } ^ { ( h ) } , \boldsymbol { W } _ { K } ^ { ( h ) } , \boldsymbol { W } _ { V } ^ { ( h ) } \in \dot { \mathbb { R } } ^ { d _ { h } \times d _ { m } } }$ and an output matrix $W _ { O } ^ { ( h ) } \in \mathbb { R } ^ { d _ { m } \times d _ { h } }$ . Defining the rownormalized attention matrix

$$
A ^ { ( h ) } ( X ) = \mathrm { s o f t m a x } \left( \frac { ( W _ { Q } ^ { ( h ) } X ) ^ { \top } ( W _ { K } ^ { ( h ) } X ) } { \sqrt { d _ { h } } } \right) ,
$$

the full multi-head output is

$$
\mathrm { M H A } ( X ) = \sum _ { h = 1 } ^ { H } W _ { O } ^ { ( h ) } W _ { V } ^ { ( h ) } X A ^ { ( h ) } ( X ) ^ { \top } .
$$

We use $W _ { Q } , W _ { K } , W _ { V } , W _ { O } \in \mathbb { R } ^ { d _ { m } \times d _ { m } }$ to denote the corresponding stacked or concatenated MHA matrices.

The SwiGLU feed-forward block uses read matrices $W _ { \mathrm { g a t e } } , W _ { \mathrm { u p } } \in \mathbb { R } ^ { d _ { f f } \times d _ { m } }$ and an output matrix $W _ { \mathrm { d o w n } } \in \mathbb { R } ^ { d _ { m } \times d _ { f f } } ;$

$$
\mathrm { F F N } ( X ) = W _ { \mathrm { d o w n } } \big ( \mathrm { S i L U } ( W _ { \mathrm { g a t e } } X ) \odot ( W _ { \mathrm { u p } } X ) \big ) .
$$

Finally, we denote the additive contributions of each module as $\Delta ^ { \mathrm { a t t n } } = Y - X$ and $\Delta ^ { \mathrm { f f n } } = Z - Y$

## B GRAM MATRICES FOR READ AND WRITE ANALYSIS

To apply the read-blindness scores $\eta _ { k }$ and $\mathrm { e r a } _ { k }$ to a transformer, we form one Gram matrix per operator role. We use two complementary views.

Weight View. The Weight View asks what the trained weights, taken alone, permit each block to read or write. On the read side, a feature in the null space of a weight matrix is provably invisible to that operator regardless of input. On the write side, the Gram measures structural access to each residual output coordinate; what is actually deposited also depends on the operator’s internal activations. The six Weight-View Grams, summing over heads h, are:

$$
\begin{array} { r l r } { G ^ { a , w i } = \sum _ { h } W _ { V } ^ { ( h ) \top } W _ { V } ^ { ( h ) } } & { \qquad } & { ( W _ { V } ~ \mathrm { r e a d - v a l u e ~ p r o j e c t i o n } ) } \\ { G ^ { a , w o } = \sum _ { h } W _ { O } ^ { ( h ) } W _ { O } ^ { ( h ) \top } } & { \qquad } & { ( W _ { O } ~ \mathrm { w r i t e - o u t p u t ~ p r o j e c t i o n } ) } \\ { G ^ { f , w i } = W _ { \mathrm { g a t e } } ^ { \top } W _ { \mathrm { g a t e } } + W _ { \mathrm { u p } } ^ { \top } W _ { \mathrm { u p } } } & { \qquad } & { ( \mathrm { F F N ~ g a t e / u p ~ r e a d } ) } \\ { G ^ { f , w o } = W _ { \mathrm { d o w n } } W _ { \mathrm { d o w n } } ^ { \top } } & { \qquad } & { ( \mathrm { F F N ~ d o w n ~ w r i t e } ) } \\ { G ^ { Q } = \sum _ { h } W _ { Q } ^ { ( h ) \top } W _ { Q } ^ { ( h ) } } & { \qquad } & { ( W _ { Q } ~ \mathrm { r e a d - q u e r y ~ p r o j e c t i o n } ) } \\ { G ^ { K } = \sum _ { h } W _ { K } ^ { ( h ) \top } W _ { K } ^ { ( h ) } . } & { \qquad } & { ( W _ { K } ~ \mathrm { r e a d - k e y ~ p r o j e c t i o n } ) } \end{array}
$$

We list $G ^ { Q }$ and $G ^ { K }$ separately from $G ^ { a , w i }$ because, while all three read from the residual stream, they play different computational roles: $W _ { V }$ determines which information is routed into the attention output, whereas $W _ { Q }$ and $W _ { K }$ determine the attention pattern itself. Read-blindness in $W _ { V }$ blocks information flow directly; read-blindness in $W _ { Q } / W _ { K }$ distorts where attention looks.

Operator View. The Operator View uses empirical second-moment matrices computed on a heldout corpus. Its input Grams measure which residual coordinates are available at the input to each block, while its output Grams measure which coordinates receive energy from the block’s realized residual update. The four Operator-View Grams are:

$$
\begin{array} { r l } & { G ^ { a , o i } = \mathbb { E } \big [ \tilde { Z } _ { 0 } \tilde { Z } _ { 0 } ^ { \top } \big ] } \\ & { G ^ { a , o o } = \mathbb { E } \big [ ( Z _ { 1 } - Z _ { 0 } ) ( Z _ { 1 } - Z _ { 0 } ) ^ { \top } \big ] } \\ & { G ^ { f , o i } = \mathbb { E } \big [ \tilde { Z } _ { 1 } \tilde { Z } _ { 1 } ^ { \top } \big ] } \\ & { G ^ { f , o o } = \mathbb { E } \big [ ( Z _ { 2 } - Z _ { 1 } ) ( Z _ { 2 } - Z _ { 1 } ) ^ { \top } \big ] . } \end{array}
$$

(Attention input second moment)

(Attention residual deposit)

(FFN input second moment)

(FFN residual deposit)

Expectations are taken over tokens and sequences, with each token represented as a column vector. $G ^ { a , o i }$ and $G ^ { f , o i }$ measure which features the normalized residual stream emphasizes at the input to each block; they do not by themselves measure whether the following operator reads those features. $G ^ { a , o o }$ and $G ^ { f , \bar { o } o }$ measure which residual coordinates actually receive energy from each block’s output.

## C NULL OCCUPANCY

A component of the input in the null space of A has no effect on its output, so approximate null directions reveal features that the matrix effectively ignores. We use the input-side Gram matrix $G = A ^ { \top } A$ , which has the same null space as A and is symmetric positive semidefinite. Because

Table 4: Null occupancy results. Cells report the median $\eta _ { \mathcal { M } }$ with rank-biserial $r _ { b }$ in parentheses. Bold cells have $r _ { b } > 0 . 9 5 ;$ stars denote one-sided Mann–Whitney tests of $\eta _ { \mathcal { M } } > \eta _ { \neg { \mathcal M } }$ with $p <$ 0.05.
<table><tr><td></td><td>Llama 135M</td><td>Llama 1.28B</td><td>Llama 2.56B</td><td>Qwen3 1.7B</td></tr><tr><td>Massive coordinates |M|</td><td>10</td><td>10</td><td>5</td><td>7</td></tr><tr><td>Maximum activation magnitude</td><td>280</td><td> $2 . 2 7 \times 1 0 ^ { 3 }$ </td><td> $3 . 9 2 \times 1 0 ^ { 3 }$ </td><td> $2 . 2 3 \times 1 0 ^ { 3 }$ </td></tr><tr><td>Input availability:</td><td></td><td></td><td></td><td></td></tr><tr><td>Attention normalized input,  $\tilde { X }$ </td><td>0.52 (-0.40)</td><td>0.52 (-0.24)</td><td>0.52 (-0.56)</td><td>0.76 (+0.19)</td></tr><tr><td>FFN normalized input,  $\tilde { Y }$ </td><td>0.51 (-0.41)</td><td> $\mathbf { 0 . 8 0 \left( + 1 . 0 0 \right) ^ { \ast } }$ </td><td>0.65 (+0.93) 水</td><td> $0 . 8 0 \left( + 0 . 6 8 \right) ^ { \ast }$ </td></tr><tr><td>Read side weight view:</td><td></td><td></td><td></td><td></td></tr><tr><td>Value projection  $W _ { V }$ </td><td> $\mathbf { 0 . 6 9 \left( + 0 . 9 9 \right) ^ { \ast } }$ </td><td> $\mathbf { 0 . 7 2 \left( + 1 . 0 0 \right) ^ { \ast } }$ </td><td> $\mathbf { 0 . 7 3 \ : ( + 1 . 0 0 ) } ^ { \ast }$ </td><td> $\mathbf { 0 . 7 4 \ ( + 1 . 0 0 ) } ^ { \ast }$ </td></tr><tr><td>Query projection  $W _ { Q }$ </td><td> $\mathbf { 0 . 6 9 } \left( \mathbf { + 0 . 9 8 } \right) ^ { \ast }$ </td><td> $\mathbf { 0 . 7 0 \left( + 0 . 9 9 \right) ^ { \ast } }$ </td><td> $\mathbf { 0 . 7 1 \ ( + 1 . 0 0 ) } ^ { \ast }$ </td><td> $\mathbf { 0 . 7 2 \left( + 1 . 0 0 \right) ^ { \ast } }$ </td></tr><tr><td>Key projection  $W _ { K }$ </td><td> $0 . 6 0 \left( + 0 . 8 4 \right) ^ { \ast }$ </td><td> ${ \bf 0 . 6 3 \left( + 0 . 9 6 \right) } ^ { \ast }$ </td><td> $\mathbf { 0 . 6 1 } \left( \mathbf { + 0 . 9 8 } \right) ^ { \ast }$ </td><td> $\mathbf { 0 . 6 7 \ ( + 1 . 0 0 ) } ^ { \ast }$ </td></tr><tr><td>FFN gate/up  $W _ { \mathrm { g a t e } } , W _ { \mathrm { u p } }$ </td><td> $\mathbf { 0 . 7 9 \left( + 0 . 9 9 \right) ^ { \ast } }$ </td><td> ${ \bf 0 . 8 7 \left( + 1 . 0 0 \right) } ^ { \ast }$ </td><td> $\mathbf { 0 . 8 9 } \left( \mathbf { + 1 . 0 0 } \right) ^ { \ast }$ </td><td> ${ \bf 0 . 8 5 } \left( { \bf + 1 . 0 0 } \right) ^ { \ast }$ </td></tr><tr><td>Write side weight view:</td><td></td><td></td><td></td><td></td></tr><tr><td>Attention output  $W _ { O }$ </td><td>0.51 (-0.71)</td><td>0.50 (-0.58)</td><td>0.49 (-0.50)</td><td>0.51 (-0.11)</td></tr><tr><td>FFN down projection  $W _ { \mathrm { d o w n } }$ </td><td>0.49 (-0.84)</td><td>0.50 (-0.26)</td><td>0.50 (-0.49)</td><td>0.50 (-0.16)</td></tr><tr><td>Write side operator view:</td><td></td><td></td><td></td><td></td></tr><tr><td>Attention update  $\Delta ^ { \mathrm { a t t n } }$ </td><td>0.51 (-0.74)</td><td>0.50 (-0.61)</td><td>0.49 (-0.57)</td><td>0.51 (-0.13)</td></tr><tr><td>FFN residual update  $\Delta ^ { \mathrm { f f n } }$ </td><td>0.49 (-0.83)</td><td>0.49 (-0.63)</td><td>0.49 (-0.49)</td><td>0.50 (-0.41)</td></tr></table>

the strict algebraic null space is brittle for learned matrices, we instead use an energy-thresholded approximate null subspace.

Let $\begin{array} { r } { G = \sum _ { i } \sigma _ { i } v _ { i } v _ { i } } \end{array}$ be the eigendecomposition of G, with eigenvalues in descending order. We define

$$
\mathcal { N } _ { \tau } ( G ) = \mathrm { s p a n } \left\{ v _ { i } : \sigma _ { i } \leq \varepsilon \sigma _ { \operatorname* { m a x } } \mathrm { ~ a n d ~ } \sum _ { j \geq i } \sigma _ { j } \leq ( 1 - \tau ) \sum _ { j } \sigma _ { j } \right\} ,
$$

where $\tau \in ( 0 , 1 )$ controls the retained energy and ε prevents spurious null detection under nearly uniform spectra. We use $\tau = 0 . 9 9$ and $\varepsilon = 0 . 0 1$ . Let $P _ { \mathcal { N } _ { \tau } ( G ) }$ denote the orthogonal projector onto this subspace.

The null occupancy of feature k is the fraction of its coordinate direction $e _ { k }$ that lies in the approximate null subspace:

$$
\eta _ { k } ( G ) = \frac { \| P _ { \mathrm { \mathcal { N } _ { \tau } } ( G ) } e _ { k } \| ^ { 2 } } { \| e _ { k } \| ^ { 2 } } \ \in \ [ 0 , 1 ] .
$$

When $\eta _ { k } ( G ) \approx 1$ , coordinate k lies predominantly in directions that A effectively ignores; when $\eta _ { k } ( G ) \approx \mathrm { 0 }$ , it lies outside the approximate null subspace and is actively read.

## D PROBING RESULTS ON READ-BLINDNESS

## E IMPLEMENTATION AND MODEL DETAILS

We follow the data and hyperparameter setup of Gu et al. (2025) for all models. Each model is trained on the 5-billion-token RegMix dataset for 20,000 optimization steps, using a context length of 2,048 tokens. We use the GPT-NeoX tokenizer adopted by Gu et al. (2025), whose vocabulary contains 50,257 tokens. Table 7 summarizes the architectural configurations evaluated in our experiments. Training is done under a fixed seed.

We optimize every model with AdamW (Loshchilov & Hutter, 2019), using a peak learning rate of $1 0 ^ { - 4 }$ , a weight decay of 0.01, and momentum parameters $\beta _ { 1 } = 0 . 9$ and $\beta _ { 2 } = 0 . 9 9 9$ . The learning rate is warmed up linearly for the first 100 steps and then decayed according to a cosine schedule for the remainder of training. Models are trained on 8x8 Nvidia A100 80GB accelerator hardward.

Table 5: Llama 1.28B Erasure results for intervention settings. Bold cells have $r _ { b } > 0 . 9 5 ;$ stars denote one-sided Mann–Whitney tests of era $\mathcal { M } \mathcal { \mathrm { ~ > ~ } }$ era with $\neg { \mathcal { M } }$ $p < 0 . 0 5$
<table><tr><td></td><td> $W _ { V }$  Frozen</td><td> $W _ { V }$  Reparam</td><td> $W _ { \mathrm { u p } } , W _ { \mathrm { g a t e } }$  Frozen</td></tr><tr><td>Massive coordinates  $| { \mathcal { M } } |$ </td><td>7</td><td>10</td><td>11</td></tr><tr><td>Maximum activation magnitude</td><td> $2 . 0 8 \times 1 0 ^ { 3 }$ </td><td> $2 . 1 2 \times 1 0 ^ { 3 }$ </td><td>681</td></tr><tr><td>Input availability:</td><td></td><td></td><td></td></tr><tr><td>Attention normalized input,  $\tilde { X }$ </td><td>0.51 (-0.52)</td><td>0.56 (-0.57)</td><td>0.61 (-0.17)</td></tr><tr><td>FFN normalized input,  $\tilde { Y }$ </td><td>0.58 (+0.70)</td><td>0.52 (-0.28)</td><td>0.57 (-0.17)</td></tr><tr><td>Read side weight view:</td><td></td><td></td><td></td></tr><tr><td>Value projection  $W _ { V }$ </td><td>0.50 (+0.22)</td><td>0.50 (+0.09)</td><td> $\mathbf { 0 . 6 9 \ ( + 0 . 9 8 ) } ^ { * }$ </td></tr><tr><td>Query projection  $W _ { Q }$ </td><td>0.61 (+1.00) 茶</td><td>0.64 (+0.99) *</td><td>0.66 (+0.99) 2</td></tr><tr><td>Key projection  $W _ { K }$ </td><td>0.55 (+0.42) 号</td><td>0.55 (+0.24)</td><td> $0 . 6 3 \left( + 0 . 9 3 \right) ^ { * }$ </td></tr><tr><td>FFN gate/up  $W _ { \mathrm { g a t e } } , W _ { \mathrm { u p } }$ </td><td>0.64 (+1.00) d</td><td>0.64 (+0.99)</td><td>0.50 (-0.21)</td></tr><tr><td>Write side weight view:</td><td></td><td></td><td></td></tr><tr><td>Attention output  $W _ { O }$ </td><td>0.52 (-0.16)</td><td>0.51 (-0.41)</td><td>0.51 (-0.37)</td></tr><tr><td>FFN down projection  $W _ { \mathrm { d o w n } }$ </td><td>0.49 (-1.00)</td><td>0.50 (-0.95)</td><td>0.50 (-0.77)</td></tr><tr><td>Write side operator view:</td><td></td><td></td><td></td></tr><tr><td>Attention update  $\Delta ^ { \mathrm { a t t n } }$ </td><td>0.84 (-0.68)</td><td>0.83 (-0.43)</td><td>0.60 (-0.75)</td></tr><tr><td>FFN residual update  $\Delta ^ { \mathrm { { f f n } } }$ </td><td>0.59 (-0.96)</td><td>0.61 (-0.48)</td><td>0.60 (-0.70)</td></tr></table>

Table 6: Llama 1.28B Null occupancy results for intervention settings. Bold cells have $r _ { b } > 0 . 9 5 ;$ stars denote one-sided Mann–Whitney tests of era ${ \mathcal { M } } > \mathrm { e r a } _ { \lnot { M } }$ with $p < 0 . 0 5$
<table><tr><td></td><td>Wv Frozen</td><td> $W _ { V }$  Reparam</td><td> $W _ { \mathrm { u p } } , W _ { \mathrm { g a t e } }$  Frozen</td></tr><tr><td>Massive coordinates  $| { \mathcal { M } } |$ </td><td>7</td><td>10</td><td>11</td></tr><tr><td>Maximum activation magnitude</td><td> $2 . 0 8 \times 1 0 ^ { 3 }$ </td><td> $2 . 1 2 \times 1 0 ^ { 3 }$ </td><td>681</td></tr><tr><td>Input availability:</td><td></td><td></td><td></td></tr><tr><td>Attention normalized input,  $\tilde { X }$ </td><td> $0 . 5 7 \left( + 0 . 3 0 \right)$ </td><td> $0 . 5 4 \left( - 0 . 1 3 \right)$ </td><td> $0 . 5 5 \left( + 0 . 0 2 \right)$ </td></tr><tr><td>FFN normalized input,  $\tilde { Y }$ </td><td> $\mathbf { 0 . 8 9 } \left( \mathbf { + 1 . 0 0 } \right) ^ { \ast }$ </td><td>0.80 (+1.00)</td><td>0.68 (+0.77) 州</td></tr><tr><td>Read side weight view:</td><td></td><td></td><td></td></tr><tr><td>Value projection  $W _ { V }$ </td><td> $0 . 5 0 \left( + 0 . 0 0 \right)$ </td><td> $0 . 5 0 \left( + 0 . 0 0 \right)$ </td><td> ${ \bf 0 . 6 7 \left( + 0 . 9 9 \right) ^ { * } }$ </td></tr><tr><td>Query projection  $W _ { Q }$ </td><td> ${ \bf 0 . 6 6 \left( + 0 . 9 9 \right) } ^ { \ast }$ </td><td> $\mathbf { 0 . 6 9 \left( + 1 . 0 0 \right) ^ { \ast } }$ </td><td> $\mathbf { 0 . 6 6 \left( + 0 . 9 9 \right) ^ { \ast } }$ </td></tr><tr><td>Key projection  $W _ { K }$ </td><td> $\mathbf { 0 . 5 9 \left( + 0 . 9 8 \right) ^ { \ast } }$ </td><td> $\mathbf { 0 . 6 1 } \left( \mathbf { + 0 . 9 9 } \right) ^ { \ast }$ </td><td> ${ \bf 0 . 6 5 } \left( { \bf + 0 . 9 7 } \right) ^ { \ast }$ </td></tr><tr><td>FFN gate/up  $W _ { \mathrm { g a t e } } , W _ { \mathrm { u p } }$ </td><td> ${ \bf 0 . 8 7 \left( + 1 . 0 0 \right) ^ { \ast } }$ </td><td> $\mathbf { 0 . 9 0 \ : ( + 1 . 0 0 ) } ^ { \ast }$ </td><td>0.50 (+0.00)</td></tr><tr><td>Write side weight view:</td><td></td><td></td><td></td></tr><tr><td>Attention output  $W _ { O }$ </td><td>0.52 (+0.00)</td><td>0.51 (-0.12)</td><td>0.51 (-0.38)</td></tr><tr><td>FFN down projection  $W _ { \mathrm { d o w n } }$ </td><td>0.50 (-0.42)</td><td>0.51 (-0.43)</td><td>0.50 (-0.76)</td></tr><tr><td>Write side operator view:</td><td></td><td></td><td></td></tr><tr><td>Attention update  $\Delta ^ { \mathrm { a t t n } }$ </td><td>0.52 (-0.07)</td><td>0.52 (-0.10)</td><td>0.51 (-0.48)</td></tr><tr><td>FFN residual update  $\Delta ^ { \mathrm { { f f n } } }$ </td><td>0.49 (-0.99)</td><td>0.50 (-0.54)</td><td>0.49 (-0.78)</td></tr></table>

Huggingface Transformers<sup>1</sup> framework is used for the model training. To speed up the training process, liger-kernel<sup>2</sup> library is used. A typical training run concludes in 8 hours.

## F DIRECTIONAL FFN AMPLIFIER DIAGNOSTICS

$$
\begin{array} { r } { \dot { g } _ { i } = ( W _ { \mathrm { g a t e } } ) _ { i , } ^ { \mp } } \end{array}
$$

Table 7: Llama (Touvron et al., 2023) and Qwen3 (Team, 2025) Architectures of the models used in our experiments.
<table><tr><td colspan="5">Property Llama 135M Llama 1.28B Llama 2.56B Qwen 1.7B</td></tr><tr><td>Number of Transformer layers (L)</td><td>10</td><td>16</td><td>27</td><td>28</td></tr><tr><td>Residual-stream dimension  $( d _ { \mathrm { m o d e l } } )$ </td><td>768</td><td>2,048</td><td>2,560</td><td>2,048</td></tr><tr><td>Feed-forward dimension  $( d _ { \mathrm { f f } } )$ </td><td>1,536</td><td>8,192</td><td>7,680</td><td>6,144</td></tr><tr><td>Number of query heads  $( n _ { h } )$ </td><td>8</td><td>32</td><td>40</td><td>16</td></tr><tr><td>Number of key-value heads  $( n _ { k v } )$ </td><td>8</td><td>32</td><td>40</td><td>16</td></tr><tr><td>Attention-head dimension  $( d _ { h } )$ </td><td>96</td><td>64</td><td>64</td><td>128</td></tr></table>

$u _ { i } = ( W _ { \mathrm { u p } } ) _ { i , : } ^ { \top }$ denote the input directions of intermediate unit i. For output coordinate k, define

$$
U _ { k } = \sum _ { i } W _ { \mathrm { d o w n } } [ k , i ] g _ { i } u _ { i } ^ { \top } = W _ { \mathrm { g a t e } } ^ { \top } \mathrm { d i a g } \big ( W _ { \mathrm { d o w n } } [ k , : ] \big ) W _ { \mathrm { u p } } , \qquad S _ { k } = \textstyle { \frac { 1 } { 2 } } \big ( U _ { k } + U _ { k } ^ { \top } \big ) .\tag{2}
$$

The FFN contribution to coordinate k is then approximately

$$
\begin{array} { r } { \mathcal { F } _ { \mathrm { f f n } } ( \widetilde { h } ) _ { k } \approx \widetilde { h } ^ { \top } S _ { k } \widetilde { h } . } \end{array}
$$

Let $( \lambda _ { \star } ^ { ( k ) } , s _ { \star } ^ { ( k ) } )$ be the eigenpair of $S _ { k }$ with largest eigenvalue magnitude, with $\| s _ { \star } ^ { ( k ) } \| _ { 2 } = 1$ . When this eigenpair dominates the spectrum,

$$
\mathcal { F } _ { \mathrm { f f n } } ( \widetilde { h } ) _ { k } \approx \lambda _ { \star } ^ { ( k ) } \big ( s _ { \star } ^ { ( k ) \top } \widetilde { h } \big ) ^ { 2 } .
$$

Thus, $s _ { \star } ^ { ( k ) }$ is the residual-stream direction with the largest quadratic gain into output coordinate $k ,$ and ${ \lambda } _ { \star } ^ { ( k ) }$ is its signed gain.

Per-coordinate diagnostics. We derive four diagnostics from this construction. Except where the layer aggregation is shown explicitly, the quantities below are computed separately at each layer.

• The overall quadratic magnitude is $\| U _ { k } \| _ { F } ,$ computed without explicitly assembling $U _ { k }$ as

$$
\begin{array} { r } { \| U _ { k } \| _ { F } ^ { 2 } = W _ { \mathrm { d o w n } } [ k , : ] ^ { \top } \left( W _ { \mathrm { g a t e } } W _ { \mathrm { g a t e } } ^ { \top } \odot W _ { \mathrm { u p } } W _ { \mathrm { u p } } ^ { \top } \right) W _ { \mathrm { d o w n } } [ k , : ] , } \end{array}
$$

where ⊙ is the element-wise (Hadamard) product.

• The massive-subspace mass of the amplifier direction is

$$
\rho _ { \mathcal { M } } ^ { ( k ) } = \underset { L } { \mathrm { m e a n } } \sum _ { j \in \mathcal { M } } \big ( s _ { \star } ^ { ( k , L ) } [ j ] \big ) ^ { 2 } .\tag{3}
$$

Because $s _ { \star } ^ { ( k , L ) }$ is a unit vector, this is the fraction of its squared norm lying in the coordinate subspace span $\{ e _ { j } : j \in \mathcal { M } \}$ , averaged across layers. It is bounded between zero and one. An isotropically oriented unit vector has expected mass $| { \mathcal { M } } | / d _ { m } ;$ larger $\rho _ { \mathcal { M } } ^ { ( k ) }$ therefore indicates stronger alignment with the massive-coordinate subspace.

In the base 1.28B model, massive output coordinates have median $\rho _ { \mathcal { M } } ^ { ( k ) } = 0 . 0 3 1$ , about six times the isotropic baseline $1 0 / 2 0 \hat { 4 } 8 \approx 0 . 0 0 4 9$ . This quantity is larger than in the deterministic, size-matched non-massive set with rank-biserial effect $r _ { b } ~ = ~ 0 . 6 0$ . Thus, their amplifier directions are preferentially aligned with the massive-coordinate subspace, but are not confined to it: only 3.1% of their squared norm lies in that subspace.

• The leading-gain magnitude is $| \lambda _ { \star } ^ { ( k ) } |$ . Given the previously identified unit eigenvector, it can be computed as

$$
\lambda _ { \star } ^ { ( k ) } = s _ { \star } ^ { ( k ) \top } U _ { k } s _ { \star } ^ { ( k ) } = W _ { \mathrm { d o w n } } [ k , : ] ^ { \top } \left( ( W _ { \mathrm { g a t e } } s _ { \star } ^ { ( k ) } ) \odot ( W _ { \mathrm { u p } } s _ { \star } ^ { ( k ) } ) \right) .
$$

We report max $_ { \cdot } \ln \vert \lambda _ { \star } ^ { ( k , L ) } \vert$ for each coordinate.

![](images/59fd3696748ca3be8b8fde82147f88213c7af1129243ed8d34e1dc5004c4d7f0.jpg)

![](images/1bb5bd9d20712ff94ead8c6afca37684209150bb071a6fb43c6ebd4210f343f2.jpg)

![](images/a9517043ddf99bd2345620e8f8b8586027e3dcb75c003bcd2328cab5dd2532cf.jpg)

Figure 3: Emergence of M discrimination during pre-training of the 1.28B model (21 checkpoints, 0–20k steps; per-coordinate maximum across 16 layers). Solid colored and dashed gray lines show medians and interquartile ranges for $\mathcal { M } \left( n = 1 0 \right)$ and sampled $c { \mathcal { M } }$ controls $( n = 1 0 0 )$ , respectively. The dotted line marks $t _ { \mathrm { s e p } } .$ , the first checkpoint with rank-biserial effect $> 0 . 9 5$ . FFN read-blindness preceeds its amplifier.  
![](images/978d1f4f80415c506f54f259536ea0d45255c49dc89a2a2fb1ca4b9eaf52be72.jpg)

![](images/380ef3980887455dd2b6f0f0003c3b5a187030215d91ccd84b50c40b29837e88.jpg)  
Figure 4: FFN write-row concentration vs. amplifier gain (left) and $s _ { \star }$ mass on $\mathcal { M }$ (right). Color denotes attention-input erasure, size denotes massiveness $R ,$ and grey points are non-MA coordinates.

• The rank-one purity is

$$
r ^ { ( k ) } = \frac { \vert \lambda _ { \star } ^ { ( k ) } \vert } { \vert \vert S _ { k } \vert \vert _ { F } } \in [ 0 , 1 ] ,
$$

where

$$
\begin{array} { r } { \| S _ { k } \| _ { F } ^ { 2 } = \frac { 1 } { 2 } \big ( \| U _ { k } \| _ { F } ^ { 2 } + \mathrm { t r } ( U _ { k } ^ { 2 } ) \big ) , } \end{array}
$$

and

$$
\mathrm { t r } ( U _ { k } ^ { 2 } ) = W _ { \mathrm { d o w n } } [ k , : ] ^ { \top } \big ( W _ { \mathrm { g a t e } } W _ { \mathrm { u p } } ^ { \top } \odot W _ { \mathrm { u p } } W _ { \mathrm { g a t e } } ^ { \top } \big ) W _ { \mathrm { d o w n } } [ k , : ] .
$$

Following Sun et al. (2026), $r ^ { ( k ) } \to 1$ indicates rank-one dominance. The randomsymmetric baseline scales as $\mathbb { E } [ r ^ { ( k ) } ] \sim 2 / \sqrt { d _ { m } }$ (approximately 0.044 for $d _ { m } = 2 0 4 8 )$ We report max ${ _ L r } ^ { ( k , L ) }$ for each coordinate.

We additionally measure direction sharing within a coordinate set S using the sign-invariant statistic

$$
\overline { { | \cos | } } _ { \mathcal { S } } = \operatorname* { m e a n } _ { L } \operatorname* { m e a n } _ { \stackrel { i , j \in \mathcal { S } } { i \not = j } } \left| \left. s _ { \star } ^ { ( i , L ) } , s _ { \star } ^ { ( j , L ) } \right. \right| .
$$

For two independent isotropic directions in $d _ { m }$ dimensions, the large- $\cdot d _ { m }$ baseline is approximately $\sqrt { 2 / ( \pi d _ { m } ) }$

## G LOSS GEOMETRY AND LOCAL ADAMW PRESSURE

This appendix gives the definitions behind the two analyses in Section 6. The curvature analysis describes the local geometry of parameter space. The AdamW-pressure analysis instead asks how a complete next optimizer step changes activation magnitude. These are different measurements and we do not identify one with the other.

![](images/dcf565eec823d22de5bba68c856840b83bde86688d88f9d35ac076b0000118a8.jpg)  
Figure 5: Curvature and radial loss-gradient projection for $W _ { \mathrm { d o w n } } [ k , : ]$ in deep layers $L \in$ {9, 12, 15}. Negative $\boldsymbol { \nabla } \mathcal { L } _ { \mathrm { C E } }$ · wˆ means that the gradient vector points inward, so the loss-descent direction points outward. Strong-amplifier M coordinates concentrate in the negative-curvature, negative-gradient quadrant, unlike the matched non-massive coordinates.

## G.1 DIRECTIONAL CURVATURE AND RADIAL GRADIENT

For each parameter slice associated with residual coordinate k, we sample unit directions v within that slice and measure the signed directional curvature

$$
\kappa ( v ) = v ^ { \top } H v , \qquad H = \nabla ^ { 2 } \mathcal { L } _ { \mathrm { C E } } .\tag{4}
$$

We average over 64 sampled directions and compare the resulting per-coordinate values for M and $\neg { M }$ over 30 batches. Positive rank-biserial effects mean that the signed curvature tends to be higher on $\mathcal { M } ;$ negative effects mean that it tends to be lower.

The read-side matrices $W _ { V } , W _ { \mathrm { g a t e } }$ , and $W _ { \mathrm { u p } }$ have higher curvature on $\mathcal { M }$ , whereas the write-side matrices $W _ { O }$ and $W _ { \mathrm { d o w n } }$ have lower curvature (Figure 2). For the strong-amplifier M rows of $W _ { \mathrm { d o w n } }$ in deep layers, the average signed curvature is negative. These rows therefore do not lie in an ordinary locally convex basin along the measured directions.

For the same $W _ { \mathrm { d o w n } }$ rows, we also report the radial projection

$$
r _ { k } = \nabla _ { w _ { k } } \mathcal { L } _ { \mathrm { C E } } \cdot \hat { w } _ { k } , \qquad \hat { w } _ { k } = \frac { w _ { k } } { \| w _ { k } \| } .\tag{5}
$$

The strong-amplifier coordinates largely have $r _ { k } < 0$ . This sign requires care: the gradient vector itself points inward, but a gradient-descent update uses the opposite direction and therefore points outward. By contrast, AdamW’s decoupled weight-decay update $- \eta \lambda w _ { k }$ always points inward in parameter space. The two tendencies oppose one another. Because Figure 5 uses the raw loss gradient rather than the complete preconditioned update, it establishes this local geometry but not an exact AdamW force balance.

## G.2 ADAMW UPDATE BREAKDOWN

We next measure the end effect of the optimizer on the activations. Let $( m _ { t } , v _ { t } )$ be the saved AdamW state, let g be the new local loss gradient, and let $s = t + 1$ be the next optimizer step. With $c _ { 1 } = 1 - \beta _ { 1 } ^ { s }$ and $c _ { 2 } = 1 - \beta _ { 2 } ^ { s }$ , define the adaptive update without weight decay as

$$
F ( g ; m _ { t } , v _ { t } ) = - \eta \frac { [ \beta _ { 1 } m _ { t } + ( 1 - \beta _ { 1 } ) g ] / c _ { 1 } } { \sqrt { [ \beta _ { 2 } v _ { t } + ( 1 - \beta _ { 2 } ) g ^ { 2 } ] / c _ { 2 } } + \epsilon } .\tag{6}
$$

We decompose the next step into

$$
\Delta \theta _ { \mathrm { s t a t e } } = F ( 0 ; m _ { t } , v _ { t } ) ,\tag{7}
$$

$$
\Delta \theta _ { g | \mathrm { s t a t e } } = F ( g ; m _ { t } , v _ { t } ) - F ( 0 ; m _ { t } , v _ { t } ) ,\tag{8}
$$

$$
\Delta \theta _ { \mathrm { W D } } = - \eta \lambda \theta ,\tag{9}
$$

$$
\Delta \theta _ { \mathrm { f u l l } } = F ( g ; m _ { t } , v _ { t } ) - \eta \lambda \theta .\tag{10}
$$

The components add exactly:

$$
\Delta \theta _ { \mathrm { f u l l } } = \Delta \theta _ { \mathrm { s t a t e } } + \Delta \theta _ { g | \mathrm { s t a t e } } + \Delta \theta _ { \mathrm { W D } } .\tag{11}
$$

This decomposition separates the contribution already stored in the optimizer state from the additional contribution of the current gradient and from decoupled weight decay.

## G.3 ACTIVATION PRESSURE

For coordinate $k$ at the output of block $l ,$ define its activation magnitude on the selected token positions $V _ { i }$ of batch i as

$$
A _ { k } ^ { ( l , i ) } = \left( \sum _ { u \in V _ { i } } h _ { u , k } ^ { ( l , i ) 2 } \right) ^ { 1 / 2 } .\tag{12}
$$

For update component $c ,$ its local first-order pressure is

$$
P _ { k , c } ^ { ( l , i ) } = \left. \nabla _ { \theta ^ { ( l ) } } A _ { k } ^ { ( l , i ) } , \Delta \theta _ { c } ^ { ( l , i ) } \right. .\tag{13}
$$

Positive pressure predicts an increase in activation magnitude under that component; negative pressure predicts a decrease. By linearity,

$$
P _ { \mathrm { f u l l } } = P _ { \mathrm { s t a t e } } + P _ { g | \mathrm { s t a t e } } + P _ { \mathrm { W D } } .\tag{14}
$$

The implementation computes the Jacobian–vector product for each additive direction once and then evaluates all selected coordinates together.

Figure 6 shows the layerwise result. The full AdamW step has positive mean pressure on M at every layer, and the pressure generally grows in the deeper half of the model. The matched non-massive control remains near zero. The saved optimizer state explains most of the positive pressure; the additional contribution of the current gradient is much smaller. Weight decay has negative pressure on M, especially in the final layer. Thus, weight decay acts as a brake on massive activations, while the accumulated adaptive state more than compensates for it.

Scope. This is a local, first-order, one-step synthetic move at a fixed checkpoint. We update only the decoder block that produces the target activation and do not include prefix parameters. The result therefore shows the direction in which the next local AdamW step is poised to move the activations; it does not claim that every step throughout training increases them.

## H SENSITIVITY TO MA SELECTION THRESHOLD

To evaluate sensitivity to the MA Selection threshold R discussed in section 3, we additionally repeat erasure analysis on the Llama 1.28B model using progressively stricter thresholds. Results are reported in table 8.

The main observations remain consistent across thresholds. The value, query, and FFN gate/up read projections continue to show strong positive separation between M and cM coordinates. At the more permissive threshold (R=2), the separation is slightly weaker. This is expected since M set here includes less-extreme coordinates. As the threshold increases, the read-side separation becomes stronger. Thus, the reported read-blind, write-open asymmetry is not specific to the choice and becomes more pronounced for more extreme massive activations.

![](images/bb977553b1550358a13d899fa4285c341e82fa3cd5d9ad8ca8c91ddaa4fee81a.jpg)  
Figure 6: Local first-order AdamW pressure on activation magnitude in the Llama 1.28B model. Lines show the batch mean and shaded regions show the batch 10th–90th percentile spread. Orange denotes M and dashed blue denotes the matched non-massive control. The full AdamW step increases the magnitude of M while leaving the control near zero. This effect is dominated by the saved optimizer state. The current gradient conditioned on that state is smaller, and weight decay acts in the opposite direction.

Table 8: Erasure results at different MA selection thresholds for Llama 1.28B base model. Cells report the median $\mathrm { e r a } _ { \mathcal { M } }$ with rank-biserial $r _ { b }$ in parentheses. Bold cells have $r _ { b } ~ > ~ 0 . 9 5 ;$ stars denote one-sided Mann–Whitney tests of era $\mathbf { \mathcal { M } } > \mathrm { e r a } _ { \lnot \mathcal { M } }$ with $p < 0 . 0 5$
<table><tr><td></td><td>R=2</td><td>R=5</td><td>R=10</td><td>R=20</td></tr><tr><td>Massive coordinates  $| { \mathcal { M } } |$ </td><td>15</td><td>10</td><td>6</td><td>4</td></tr><tr><td>Maximum activation magnitude</td><td> $2 . 2 7 \times 1 0 ^ { 3 }$ </td><td> $2 . 2 7 \times 1 0 ^ { 3 }$ </td><td> $2 . 2 7 \times 1 0 ^ { 3 }$ </td><td> $2 . 2 7 \times 1 0 ^ { 3 }$ </td></tr><tr><td>Input availability:</td><td></td><td></td><td></td><td></td></tr><tr><td>Attention normalized input,</td><td>0.68 (-0.66)</td><td>0.68 (-0.73)</td><td>0.62 (-0.74)</td><td>0.61 (-0.77)</td></tr><tr><td>FFN normalized input,  $\tilde { Y }$ </td><td>0.56 (-0.14)</td><td>0.56 (-0.13)</td><td>0.56 (-0.04)</td><td>0.56 (-0.12)</td></tr><tr><td>Read side weight view:</td><td></td><td></td><td></td><td></td></tr><tr><td>Value projection  $W _ { V }$ </td><td> $0 . 7 2 \left( + 0 . 8 1 \right) ^ { * }$ </td><td> $\mathbf { 0 . 7 5 \left( + 0 . 9 9 \right) ^ { \ast } }$ </td><td> $\mathbf { 0 . 8 3 } \left( \mathbf { + 1 . 0 0 } \right) ^ { \ast }$ </td><td>0.86 (+1.00)</td></tr><tr><td>Query projection  $W _ { Q }$ </td><td> $0 . 6 3 \left( + 0 . 8 4 \right) ^ { \ast }$ </td><td> $\mathbf { 0 . 6 5 \left( + 0 . 9 9 \right) ^ { \ast } }$ </td><td> $\mathbf { 0 . 6 9 \ ( + 1 . 0 0 ) } ^ { * }$ </td><td>0.71 (+1.00)</td></tr><tr><td>Key projection  $W _ { K }$ </td><td> $0 . 5 4 \left( + 0 . 3 2 \right) ^ { \ast }$ </td><td> $0 . 5 5 \left( + 0 . 3 1 \right) ^ { \ast }$ </td><td>0.53 (-0.04)</td><td>0.52 (-0.08)</td></tr><tr><td>FFN gate/up  $W _ { \mathrm { g a t e } } , W _ { \mathrm { u p } }$ </td><td>0.61 (+0.96)</td><td>0.63 (+1.00)</td><td>0.65 (+1.00) 2</td><td>0.69 (+1.00)</td></tr><tr><td>Write side weight view:</td><td></td><td></td><td></td><td></td></tr><tr><td>Attention output  $W _ { O }$ </td><td>0.52 (-0.31)</td><td>0.51 (-0.61)</td><td>0.50 (-0.98)</td><td>0.42 (-0.99)</td></tr><tr><td>FFN down projection  $W _ { \mathrm { d o w n } }$ </td><td>0.50 (-0.57)</td><td>0.49 (-0.64)</td><td>0.49 (-0.96)</td><td>0.46 (-1.00)</td></tr><tr><td>Write side operator view:</td><td></td><td></td><td></td><td></td></tr><tr><td> $\Delta ^ { \mathrm { a t t n } }$ </td><td></td><td></td><td></td><td></td></tr><tr><td>Attention update</td><td>0.86 (-0.34)</td><td>0.85 (-0.44)</td><td>0.86 (-0.34)</td><td>0.67 (-0.50)</td></tr><tr><td>FFN residual update  $\Delta ^ { \mathrm { f f n } }$ </td><td>0.69 (-0.80)</td><td>0.69 (-0.77)</td><td>0.67 (-0.90)</td><td>0.66 (-0.97)</td></tr></table>