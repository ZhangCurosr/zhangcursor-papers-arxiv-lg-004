# UNIFORM RACE: PARAMETER-FREE APPROXIMATE REJECTION SAMPLING

Seiyun Shin   
Graduate School of Artificial Intelligence   
Pohang University of Science and Technology   
Pohang, 37673, South Korea   
seiyun923@gmail.com   
Kwang-Sung Jun   
Graduate School of Artificial Intelligence   
Department of Computer Science and Engineering   
Pohang University of Science and Technology   
Pohang, 37673, South Korea   
kwangsungjun@postech.ac.kr

Juhyeong Pang Department of Computer Science University of Wisconsin–Madison Madison, WI 53706, USA pang35@wisc.edu

## ABSTRACT

We study approximate sampling: given N independent samples from a proposal distribution $\mu ,$ the goal is to select one whose distribution is close to a target π specified only up to a normalizing constant. The seminal work of Block & Polyanskiy (2023) provides finite budget error bounds for approximate rejection sampling (RS) as a function of the algorithm’s acceptance threshold M. The threshold M giving the smallest bound, however, depends on properties of $( \pi , \mu )$ that are typically unavailable from the observed sample. This raises a natural question: Can one attain the best RS guarantee without taking M as input? We answer this question affirmatively by proposing a parameter-free sampling algorithm called uniform race (UR), based on importance weights, which are ratios of target to proposal probabilities (or densities). It divides each observed weight by an independent uniform random variable to form a score and returns the candidate with the largest score. For every budget N, its total variation error satisfies the RS upper bound for every fixed threshold M simultaneously, thereby achieving the best such bound in hindsight. We also characterize its output distribution conditional on the largest score, identifying when it is exactly the target π. Uniform race has no larger total variation error than a natural budget-calibrated RS derived from Rohatgi et al. (2025) and sampling importance resampling (SIR). In particular, we exhibit instances where UR’s error is exponentially smaller in N than that of either baseline. Furthermore, we establish conditions under which attaining this RS guarantee for every $( \pi , \mu )$ uniquely determines the selection probabilities as those of UR. Finally, test-time scaling experiments on LLM math-reasoning tasks corroborate the theoretical comparisons and demonstrate that UR remains competitive in ground-truth accuracy without requiring threshold selection.

## 1 INTRODUCTION

Inference-time computation often involves generating a finite number of samples from a proposal distribution and selecting one to approximate a target distribution. Inspired by this, we study the following approximate sampling problem: given N independent and identically distributed (i.i.d.) samples $Y _ { 1 } , \dots , Y _ { N }$ drawn from a proposal distribution $\mu ,$ select one whose marginal distribution is close to a target π that is specified only up to a normalizing constant. To describe how the target differs from the proposal, define the importance weight as $\begin{array} { r } { w : = \frac { \mathrm { d } \pi } { \mathrm { d } \mu } } \end{array}$ . For discrete distributions, this is simply $w ( y ) = \pi ( y ) / \mu ( y )$ , so $w ( y )$ measures how much the proposal probability at y must be reweighted to obtain the target. We assume that the sampler does not evaluate this weight directly.

Instead, at each sampled point $Y _ { i }$ for $i = 1 , \ldots , N$ , the sampler only has access to the unnormalized weight $w _ { u } ( Y _ { i } ) : = Z w ( Y _ { i } )$ ), where $Z ~ > ~ 0$ is an unknown normalizing constant. Equivalently, $w ( \bar { Y _ { i } } ) = { w _ { u } ( \dot { Y _ { i } } ) / Z }$ , but neither $Z$ nor $w ( Y _ { i } )$ is available to the sampler. We assume that the proposal covers the target, written $\pi \ll \mu \colon$ every set with zero proposal probability also has zero target probability. Throughout the main text, $0 < w ( Y ) < \infty$ almost surely under $\mu ;$ we consider zero weights separately in Appendix B.5.

We refer to the sampled points as candidates. Let $\widehat { Y }$ be the returned candidate and $P _ { N } : = \mathcal { L } ( \widehat { Y } )$ its marginal distribution, where ${ \mathcal { L } } ( X )$ denotes the law of $X$ . Our goal is to make $P _ { N }$ close to $\pi ,$ measured by total variation (TV) distance,

$$
D _ { \mathrm { T V } } ( P _ { N } , \pi ) : = \operatorname* { s u p } _ { D } | P _ { N } ( D ) - \pi ( D ) | ,\tag{1}
$$

where $D$ ranges over measurable sets. We note that the distribution $P _ { N }$ averages over both the candidates and the sampler’s additional randomness.

A useful baseline for this objective is classical rejection sampling (RS), which produces an exact sample from π if a valid acceptance threshold M is available. For the threshold to be valid, it must satisfy $w ( y ) \leq M$ over all possible proposals. Given such a threshold, RS repeatedly proposes $Y \sim \mu$ and accepts with probability $w \bar { ( \boldsymbol { Y } ) / M }$ until the first acceptance. The returned sample follows π since the probability of proposing y and accepting it is proportional to $\mu ( y ) w ( y ) = \pi ( y )$ in the discrete case; the same calculation holds for densities. Notably, this does not require prior knowledge of the normalizing constant $Z .$ If only the unnormalized weights $w _ { u } = Z w$ are available, the same acceptance test can be implemented using the corresponding upper bound $Z M \colon w _ { u } ( Y ) / Z M =$ w $( \dot { Y } ) / M$ . Hence RS need not compute $Z ,$ , but it still relies on a valid upper bound, called an envelope, on the importance weight, and may continue sampling until an acceptance occurs.

This exposes two desiderata in our setting: the sampler should respect a prescribed proposal budget $N _ { \ast }$ and it should not require prior knowledge of a problem-dependent envelope. One way to accommodate both is to choose an arbitrary threshold $M > 0$ , accept each proposal with probability (in normalized units) min $\{ w ( Y ) / M , 1 \}$ , and return an observed candidate if all N proposals are rejected. With the observed unnormalized weights, this still requires supplying the corresponding threshold ZM on the unnormalized scale.

The threshold now controls two sources of error. First, if M falls below some importance weights, the cap at one changes the distribution of accepted samples: weights above M are effectively replaced by M. This is the clipping effect. To quantify $\mathbf { i t } ,$ for $M > 0$ , define

$$
\begin{array} { r } { \mathcal { E } _ { M } ( \pi , \mu ) : = {  { \mathbb E } } _ { \mu } [ ( w ( Y ) - M ) _ { + } ] , \quad A _ { M } ( \pi , \mu ) : = 1 - \mathcal { E } _ { M } ( \pi , \mu ) = {  { \mathbb E } } _ { \mu } [ \operatorname* { m i n } \{ w ( Y ) , M \} ] . } \end{array}\tag{2}
$$

Here $( x ) _ { + } : = \operatorname* { m a x } \{ x , 0 \} ; \mathcal { E } _ { M }$ is the excess weight removed by clipping at $M ,$ , while $A _ { M }$ is the retained clipped mass. Denote $\mathcal { E } _ { M } ( \pi , \mu )$ by $\mathcal { E } _ { M }$ when $( \pi , \mu )$ is fixed. For $\bar { M } \ge 1 , \mathcal { E } _ { M }$ is the standard E<sub>M</sub>-divergence (Liu et al., 2017); see also Block & Polyanskiy (2023, Example 4). We use the same clipping definition for $0 < M < 1$ . Notice that ${ \mathcal { E } } _ { M }$ bounds the TV error of the accepted distribution; it is nonincreasing in M and tends to zero as $M \to \infty$ for every fixed $( \pi , \mu )$ . Second, increasing M lowers the acceptance probability and makes it more likely that all $N$ proposals are rejected.

This tradeoff is central to the approximate RS using finite budget. Building on the analysis from Block & Polyanskiy (2023), Huang et al. (2025, Lemma D.4) analyze the clipped-threshold rule above and show that, for a fixed threshold M,

$$
D _ { \mathrm { T V } } ( P _ { N , M } ^ { \mathrm { R S } } , \pi ) \lesssim \underbrace { 2 \mathcal { E } _ { M } ( \pi , \mu ) } _ { \mathrm { c l i p p i n g } } + \underbrace { \exp ( - N / M ) } _ { \mathrm { b u d g e t e x h a u s t i o n } } ,\tag{3}
$$

where $P _ { N , M } ^ { \mathrm { R S } }$ denotes the marginal output distribution of this finite budget procedure. The two terms favor opposite choices of M: a larger threshold reduces clipping, but makes exhausting the proposal budget more likely. Consequently, the threshold giving the best guarantee hinges on the unknown target–proposal pair, including importance weights that may never appear among the N observed candidates. This motivates the central question of this paper:

## Can one attain the bestfixed threshold RS guarantee without taking M as input?

Uniform race. To answer the question, we propose a parameter-free sampling algorithm, called uniform race (UR). It divides each observed weight by an independent uniform random variable on

<table><tr><td>Sampling rule</td><td>Sharp RS guarantee</td><td>Extra input</td><td>TV error as N grows</td></tr><tr><td>SIR</td><td>X</td><td></td><td> $\Theta ( N ^ { - 1 } )$ </td></tr><tr><td>RS with  $M = C _ { \infty } ^ { \pi }$ </td><td>√</td><td>Exact envelope</td><td> $\Theta ( 3 ^ { - N } )$ </td></tr><tr><td>Budget-calibrated RS</td><td>X</td><td>Calibration parameter δ</td><td> $\Omega ( \dot { 2 } ^ { - N / \dot { 2 } } )$ </td></tr><tr><td>UR</td><td> $\checkmark$ </td><td></td><td> $\Theta ( 3 ^ { - N } )$ </td></tr></table>

Table 1: Comparison of sampling rules. TV orders are for $\mu = ( 1 / 2 , 1 / 2 ) , \pi = ( 1 / 4 , 3 / 4 )$ and odd total budgets $\dot { N } \geq 3 .$ . The calibrated RS entry is a lower bound after optimizing δ before sampling.

(0, 1) and returns the candidate with the largest resulting score:

$$
\widehat { I } _ { U } : = \operatorname * { a r g m a x } _ { i \in [ N ] } \frac { w _ { u } ( Y _ { i } ) } { U _ { i } } , \qquad \widehat { Y } _ { U } : = Y _ { \widehat { I } _ { U } } , \qquad U _ { i } \overset { \mathrm { { i . i . d . } } } { \sim } \mathrm { U n i f } ( 0 , 1 ) ,\tag{4}
$$

where $[ N ] : = \{ 1 , \dots , N \}$ and the uniforms are independent of the proposals. Since $Z$ is common to all candidates, it cancels from the ranking. By parameter-free, we mean that the sampler does not take an acceptance threshold as input, yet its guarantee adapts automatically to the best threshold in hindsight. It also does not require an estimate of the normalization constant $Z ;$ the proposal budget and any parameters that define the target remain inputs. Specifically, our main contributions are:

Main result (informal; Theorem 1). Let $P _ { N } ^ { U } : = \mathcal { L } ( \widehat { Y } _ { U } )$ be UR’s marginal output distribution. For every fixed $N \geq 1$

$$
D _ { \mathrm { T V } } ( P _ { N } ^ { U } , \pi ) \leq \operatorname * { i n f } _ { M > 0 } \left\{ 2 \mathcal { E } _ { M } ( \pi , \mu ) + \exp ( - N / M ) \right\} .\tag{5}
$$

The key difference from equation 3 is the infimum over M: UR satisfies the bound for every threshold simultaneously, so choosing a threshold is unnecessary. Theorem 1 gives a sharper version of this guarantee. Beyond this guarantee, we establish the following structural and comparative results.

• Sharp guarantees under bounded weights. Let $C _ { \infty } ^ { \pi } : = \mathrm { e s s } \operatorname* { s u p } _ { \mu }$ w denote the smallest almostsure upper bound on the importance weight. When it is finite, we show that UR’s TV error is at most $( \stackrel { \cdot } { 1 } - 1 / C _ { \infty } ^ { \pi } ) ^ { N }$ , even though the sampler is not given this bound. This guarantee matches the smallest possible worst-case error over each bounded weight class; see Section 2 and Corollary 1.

• Rationale behind the guarantee. Using the normalized scores $S _ { i } : = w ( Y _ { i } ) / U _ { i }$ only for analysis, we characterize the output conditional on their maximum. In particular, when the weights are bounded, conditioning on max<sub>i</sub> $S _ { i } > C _ { \infty } ^ { \pi }$ gives exactly the target distribution $\pi ,$ although UR need not know this bound. Section 2 establishes the general identity and uses it to prove Theorem 1.

• Comparison with budget-calibrated RS. We next ask whether or not estimating the normalizer and choosing a threshold from the available budget can recover UR’s accuracy. In particular, a direct budget-calibrated version of Rohatgi et al. (2025, Algorithm 5) uses pilot proposals for this estimate, sets a calibration parameter $\delta \in ( 0 , 1 )$ , and then applies RS to fresh candidates. Theorem 2 shows that UR has no greater TV distance from π at the same budget, and even when UR uses only the fresh sampling portion of that budget. Moreover, on a fixed binary pair, the baseline-to-UR TV error ratio grows exponentially with N, even after optimizing prior to sampling. See Section 3.

• Comparison with SIR. Rather than setting an acceptance threshold, sampling importance resampling (SIR) (Rubin, 1987; Smith & Gelfand, 1992) selects candidate i with probability $w _ { u } ( \mathrm { \bar { Y } } _ { i } ) / \mathrm { \sum } _ { j } w _ { u } ( Y _ { j } )$ . We show that UR has no greater TV error for every $( \pi , \mu )$ and fixed N, and this also extends to every convex $f .$ divergence. The gap can be substantial: on a fixed binary pair, UR’s error decreases exponentially in $N _ { : }$ , whereas SIR’s decreases as $1 / N$ . See Section 4.

• Uniqueness of selection probabilities. We also ask whether the universal RS guarantee allows other selection probabilities. Among rules that exploit only relative observed weights, apply the same rule to every target–proposal pair, and treat candidate order symmetrically, the bounded weight guarantee forces each candidate’s selection probability to equal UR’s. See Section 5.

Table 1 highlights the main distinction: UR alone combines the sharp RS guarantee with no envelope or additional calibration input, while matching the TV error of RS supplied with the exact envelope on the displayed binary pair. Overall, UR removes threshold calibration while ensuring the best fixed threshold RS guarantee in hindsight. Numerical experiments on LLM tasks corroborate these theoretical comparisons.

## 2 RS GUARANTEES WITHOUT A THRESHOLD

We now state the sharper guarantee underlying equation 5. With $\mathcal { E } _ { M }$ and $A _ { M }$ defined in equation 2, the quantity $A _ { M } / M$ is the acceptance probability of one proposal under the fixed-threshold RS rule. Thus $( 1 - \dot { A } _ { M } / \dot { M } ) ^ { N }$ is the probability that all $\dot { N }$ proposals are rejected.

Theorem 1 (RS without threshold selection). Under the assumptions ofSection 1,for every fixed $N \geq 1$

$$
D _ { \mathrm { T V } } ( P _ { N } ^ { U } , \pi ) \leq \operatorname* { i n f } _ { M > 0 } \left\{ \mathcal { E } _ { M } + \left( 1 - \frac { A _ { M } } { M } \right) ^ { N } \right\} .\tag{6}
$$

To prove Theorem 1, we compare uniform race (UR) with rejection sampling (RS) at a fixed threshold and then optimize that threshold only in the analysis. To this end, fix $\bar { N } \geq 1$ under the assumptions of Section 1, and use the same candidates and uniforms for both procedures. Recall the normalized scores $\begin{array} { r } { S _ { i } : = \frac { w ( Y _ { i } ) } { U _ { i } } } \end{array}$ and their maximum $S _ { ( N ) } : = \operatorname* { m a x } _ { i \in [ N ] } S _ { i }$ , and define the threshold-crossing event $H _ { M } : = \{ \dot { S } _ { ( N ) } \geq M \}$ . By construction, $S _ { i } \geq M$ exactly when $U _ { i } \ \leq$ min $\{ w ( Y _ { i } ) / M , 1 \}$ Hence $H _ { M }$ is exactly the event that RS accepts at least one proposal when the same uniforms are used; on $H _ { M }$ , RS returns the first accepted candidate. UR, by contrast, always returns the candidate with the largest score. We denote their conditional distributions on $H _ { M }$ by

$$
Q _ { M } ^ { \mathrm { R S } } ( \mathrm { d } y ) : = \frac { \operatorname * { m i n } \{ w ( y ) , M \} } { A _ { M } } \mu ( \mathrm { d } y ) , \qquad Q _ { N , M } ^ { U } : = \mathcal { L } ( \widehat { Y } _ { U } \mid H _ { M } ) .\tag{7}
$$

ProofofTheorem 1. The proof has two key ingredients. First, fix $M > 0$ and let $\delta _ { M } : = \operatorname* { P r } ( H _ { M } ^ { c } )$ Lemma 1 yields $\delta _ { M } = ( 1 - A _ { M } / M ) ^ { N }$ and $D _ { \mathrm { T V } } ( Q _ { M } ^ { \mathrm { R S } } , \pi ) \leq \mathcal { E } _ { M }$ . Second, Lemma 2 then shows that selecting the largest accepted score has no greater conditional error than returning the first accepted candidate. When $\delta _ { M } > 0$ , write $F _ { N , M } ^ { U } : = \mathcal { L } ( \widehat { Y } _ { U } \mid H _ { M } ^ { c } )$ . By the law of total probability, $P _ { N } ^ { U } = ( 1 - \delta _ { M } ) Q _ { N , M } ^ { U } + \delta _ { M } F _ { N , M } ^ { U }$ . With this decomposition, we obtain:

$$
\begin{array} { r l } & { D _ { \mathrm { T V } } ( P _ { N } ^ { U } , \pi ) \overset { \mathrm { ( i ) } } { \leq } ( 1 - \delta _ { M } ) D _ { \mathrm { T V } } ( Q _ { N , M } ^ { U } , \pi ) + \delta _ { M } D _ { \mathrm { T V } } ( F _ { N , M } ^ { U } , \pi ) } \\ & { \qquad \mathrm { ( i i ) } } \\ & { \qquad \leq ( 1 - \delta _ { M } ) D _ { \mathrm { T V } } ( Q _ { M } ^ { \mathrm { R S } } , \pi ) + \delta _ { M } } \\ & { \qquad \mathrm { ( i i i ) } } \\ & { \qquad \leq { \mathcal E } _ { M } + \left( 1 - \frac { A _ { M } } { M } \right) ^ { N } , } \end{array}\tag{8}
$$

where (i) follows from the convexity of total variation; (ii) follows from Lemma 2 and $D _ { \mathrm { T V } } \leq 1 ;$ ; and (iii) follows from Lemma 1 and the definition of $\delta _ { M } . \mathrm { H } \delta _ { M } = 0$ , then $P _ { N } ^ { U } = Q _ { N , M } ^ { U }$ , and the same bound follows directly from Lemmas 1 and 2. Since $P _ { N } ^ { U }$ does not depend on the analysis threshold M, taking the infimum over $M > 0$ completes the proof. We note that the lemmas are stated in Section 2.2, with full proofs in Appendix B. □

## 2.1 OUTPUT DISTRIBUTION AT A FIXED LARGEST SCORE

We first characterize the output conditional on the largest score.

Proposition 1 (Output distribution at a fixed largest score). For every fixed $N \geq 1$ , except on a set of score values having probability zero under the distribution of ${ \bf \dot { S } } _ { ( N ) }$

$$
{ \mathcal { L } } ( \widehat { Y } _ { U } \mid S _ { ( N ) } = s ) = \pi ( \cdot \mid w < s ) ,\tag{9}
$$

whenever $\pi ( w < s ) > 0$ . Here $\{ w < s \} : = \{ y : w ( y ) < s \}$ . In particular, $i f C _ { \infty } ^ { \pi } < \infty$ , then

$$
{ \mathcal { L } } ( \widehat { Y } _ { U } \mid S _ { ( N ) } = s ) = \pi\tag{10}
$$

for every $s > C _ { \infty } ^ { \pi }$ , except possibly on the same probability zero set.

Proof. See Appendix B.2. The proposition has a simple interpretation. For a fixed score $s ,$ states with $w ( y ) \geq :$ s are excluded, while the remaining states follow π after renormalization. If $C _ { \infty } ^ { \pi } < \infty ,$ then conditional on $S _ { ( N ) } > C _ { \infty } ^ { \pi }$ , the returned candidate has distribution π. Finally, we note that the budget affects only the distribution of the largest score, not this conditional target rule. □

## 2.2 ACCEPTANCE PROBABILITY AND CONDITIONAL ERROR

We now return to the two lemmas used to prove Theorem 1. The first describes RS at a fixed threshold, and the second compares its conditional output distribution with that of UR.

Lemma 1. For every $M > 0 , 0 < A _ { M } / M \leq 1$ and

$$
\operatorname* { P r } ( H _ { M } ^ { c } ) = \left( 1 - \frac { A _ { M } } { M } \right) ^ { N } , \qquad D _ { \mathrm { T V } } ( Q _ { M } ^ { \mathrm { R S } } , \pi ) \leq \mathcal { E } _ { M } .\tag{11}
$$

Proof. One proposal is accepted with probability ${ \mathbb E } _ { \mu } [ \operatorname* { m i n } \{ w ( Y ) / M , 1 \} ] = A _ { M } / M$ , so independence gives the probability of no acceptance. Conditioning a proposal on acceptance yields $Q _ { M } ^ { \mathrm { R S } }$ Appendix B verifies that the first accepted candidate has this same distribution. To bound its error, observe that $A _ { M } Q _ { M } ^ { \mathrm { R S } } ( \mathrm { d } y ) = \operatorname* { m i n } \{ w ( y ) , M \} \mu ( \mathrm { d } y ) \leq \pi ( \mathrm { d } y )$ . The residual measure has mass $1 - A _ { M }$ so π is a mixture containing $Q _ { M } ^ { \mathrm { R S } }$ with weight $A _ { M }$ . Consequently $D _ { \mathrm { T V } } ( Q _ { M } ^ { \mathrm { R S } } , \pi ) \leq 1 - A _ { M } = \mathcal { E } _ { M } ;$ when $A _ { M } = 1$ , the two distributions agree. □

Lemma 2. For every $N \geq 1$ and M > 0, D<sub>TV</sub>(Q<sup>U</sup><sub>N,M</sub>, π) ≤ D<sub>TV</sub>(Q<sup>RS</sup><sub>M</sub> , π).

Proofsketch. Using the joint distribution from Proposition 1, we obtain two monotonicity properties. First, the density of $Q _ { N , M } ^ { U ^ { \scriptstyle * } }$ relative $\mathbf { t o } \ \pi$ is nonincreasing in the weight. Second, the density ratio $d Q _ { N , M } ^ { U } / d Q _ { M } ^ { \mathrm { R S } }$ is nondecreasing in the weight. Together, these monotonicity properties show that selecting the largest accepted score does not increase the total variation error relative to the target. The formal proof is deferred to Appendix B, where Lemma 3 converts these two properties into the desired result. □

## 2.3 BOUNDED WEIGHTS AND MINIMAX ERROR

At $M = C _ { \infty } ^ { \pi } < \infty$ , both conditional distributions are exactly π, so the inequality in Lemma 2 is tight. Note that there is then no clipping, so $Q _ { C _ { \infty } ^ { \pi } } ^ { \mathrm { R S } } = \pi$ . Averaging equation 10 over $H _ { C _ { \infty } ^ { \pi } }$ and applying Lemma 1 yield:

$$
{ \mathcal L } ( \widehat { Y } _ { U } \mid H _ { C _ { \infty } ^ { \pi } } ) = \pi , \qquad \operatorname* { P r } ( H _ { C _ { \infty } ^ { \pi } } ^ { c } ) = \left( 1 - \frac { 1 } { C _ { \infty } ^ { \pi } } \right) ^ { N } .\tag{12}
$$

Hence RS with $M = C _ { \infty } ^ { \pi }$ has the same conditional output distribution π on $H _ { C _ { \infty } ^ { \pi } }$ . Note that this does not imply equality of the full output distributions, since their conditional laws on $H _ { C _ { \infty } ^ { \pi } } ^ { c }$ may differ. Let $\begin{array} { r } { \delta _ { N } : = \mathrm { P r } ( H _ { C _ { \infty } ^ { \pi } } ^ { c } ) = ( 1 - \frac { 1 } { C _ { \infty } ^ { \pi } } ) ^ { N } } \end{array}$ . When $\delta _ { N } > 0 .$ , let $F _ { N , C _ { \infty } ^ { \pi } } ^ { U } : = \mathcal { L } ( \widehat { Y } _ { U } \mid H _ { C _ { \infty } ^ { \pi } } ^ { c } )$ . It follows that

$$
P _ { N } ^ { U } = ( 1 - \delta _ { N } ) \pi + \delta _ { N } \stackrel { \sim } { F } _ { N , C _ { \infty } ^ { \pi } } ^ { U } ,
$$

$$
D _ { \mathrm { T V } } ( P _ { N } ^ { U } , \pi ) = \delta _ { N } D _ { \mathrm { T V } } ( F _ { N , C _ { \infty } ^ { \pi } } ^ { U } , \pi ) \le \delta _ { N } .\tag{13}
$$

Here $F _ { N , C _ { \infty } ^ { \pi } } ^ { U }$ is simply UR’s output conditioned on $H _ { C _ { \infty } ^ { \pi } } ^ { c }$ ; it is not an additional fallback step. If $\delta _ { N } = 0$ , then $P _ { N } ^ { U } = \pi$ directly. In particular, Section 4 gives an example where UR and RS with $M = C _ { \infty } ^ { \pi }$ also agree on the event of no acceptance, so their full laws coincide. Equation (13) gives exponential convergence whenever $C _ { \infty } ^ { \pi } < \infty$ . The next corollary places this bound alongside a second-moment bound and identifies the exact minimax error over the class $C _ { \infty } ^ { \pi } \leq C$

Corollary 1 (Error bounds and minimax optimality). Let $C ^ { \pi } : = \mathbb { E } _ { \mu } [ w ( Y ) ^ { 2 } ]$ . For every $N \geq 1$

$$
D _ { \mathrm { T V } } ( P _ { N } ^ { U } , \pi ) \leq \operatorname* { m i n } \left\{ 1 , \frac { 2 C ^ { \pi } } { e N } , \left( 1 - \frac { 1 } { C _ { \infty } ^ { \pi } } \right) ^ { N } \right\} .\tag{14}
$$

If either $C ^ { \pi }$ or $C _ { \infty } ^ { \pi }$ is infinite, its corresponding term is interpreted as the trivial bound 1. For every $\dot { C } \geq 1$

$$
\operatorname* { i n f } _ { \mathsf { A } } \operatorname* { s u p } _ { ( \pi , \mu ) : C _ { \infty } ^ { \pi } \leq C } D _ { \mathrm { T V } } ( P _ { N } ^ { \mathrm { A } } , \pi ) = \left( 1 - { \frac { 1 } { C } } \right) ^ { N } .\tag{15}
$$

Here the infimum is over procedures A that return one ofthe N observed proposals. The supremum is over pairs satisfying the stated weight bound.

Appendix D gives both bounds: the lower bound follows from the fact that a selector cannot return a target state that does not appear among the N proposals.

## 3 COMPARISON WITH BUDGET-CALIBRATED RS

Section 2 compares UR with fixed threshold RS guarantees. We now ask whether an RS procedure that estimates the unknown normalizer from pilot samples and calibrates its threshold to the available proposal budget can match UR’s actual sampling accuracy. A natural construction comes from the budget relation in Rohatgi et al. (2025, Algorithm 5); we call this specific reparameterization budget-calibrated rejection sampling.

The algorithm starts by estimating Z by averaging the unnormalized weights of n pilot proposals and then applies RS to fresh candidates. For a maximum proposal budget $N = 2 n + 1 \geq 3$ and calibration parameter $\delta \in ( 0 , 1 )$ , invert the relation $n = 4 \bar { M } \log ( 4 / \delta )$ to set

$$
M _ { N , \delta } : = \frac { n } { 4 \log ( 4 / \delta ) } , \qquad \widehat { Z } : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } w _ { u } ( Y _ { i } ) .\tag{16}
$$

Here $\delta$ is a calibration input; Appendix C.1 records the sufficient condition under which it upperbounds the TV error. The procedure then tests $Y _ { n + 1 } , \dots , Y _ { 2 n }$ in order, accepting $Y _ { i }$ with probability min $\{ w _ { u } ( Y _ { i } ) / ( M _ { N , \delta } \widehat { Z } ) , 1 \}$ . Budget-calibrated RS then returns the first accepted candidate, or the fresh proposal $Y _ { 2 n + 1 }$ if all n proposals are rejected. Let $P _ { N , \delta } ^ { R }$ denote its output distribution. For a general integer budget $N \geq 3$ , take $n = \lfloor ( N - 1 ) / 2 \rfloor$ ⌋ and leave at most one proposal unused.

## 3.1 EXPONENTIAL SEPARATION FOR FIXED TARGET–PROPOSAL PAIR

We first show that the gap can grow exponentially with the budget on a single $( \pi , \mu )$ , even after optimizing the calibration.

Proposition 2. Let $\mu = ( 1 / 2 , 1 / 2 )$ and $\pi = ( 1 / 4 , 3 / 4 )$ . For every odd total proposal budget $N \geq 3$

$$
\operatorname* { i n f } _ { \delta \in ( 0 , 1 ) } D _ { \mathrm { T V } } ( P _ { N , \delta } ^ { R } , \pi ) \geq \frac { 1 } { 1 2 } 2 ^ { - ( N - 1 ) / 2 } , \qquad D _ { \mathrm { T V } } ( P _ { N } ^ { U } , \pi ) = \frac { 3 } { 4 } 3 ^ { - N } .\tag{17}
$$

The calibration may depend on $( \pi , \mu )$ and budget, but is chosen before observing the pilot.

Why calibration leaves error. Write $n = ( N - 1 ) / 2 , M = M _ { N , \delta }$ , and $\begin{array} { r } { \overline { { W } } _ { n } : = n ^ { - 1 } \sum _ { i = 1 } ^ { n } w ( Y _ { i } ) } \end{array}$ so the normalized rejection threshold is ${ \cal T } = M \overline { { { W } } } _ { n } .$ . The normalized importance weights are $w ( 0 ) = 1 / 2$ and $w ( 1 ) = 3 / 2$ . The lower bound reflects an unavoidable tradeoff: If $M \le 2$ , then on a rare pilot event the threshold falls below the larger weight $3 / 2$ , so clipping compresses the preference for state 1. If $M > 2 \AA$ , then clipping disappears on a constant-probability pilot event, but the acceptance probability becomes too small, so complete rejection exposes the fresh proposal. Therefore, either choice underweights state 1 and leaves error of order $2 ^ { - n }$

We first make the clipping mechanism explicit. At threshold T, the acceptance probabilities of states 0 and 1 are $a _ { 0 } ( \bar { T } ) = \mathrm { { m i n } } \{ 1 / ( 2 T ) , 1 \}$ , and $a _ { 1 } ( T ) = \mathrm { m i n } \{ 3 / ( 2 T ) , 1 \}$ . Since clipping can only reduce the ratio between the larger and smaller acceptance probabilities, $\dot { a _ { 1 } } ( T ) / a _ { 0 } ( T ) \le 3 . \mathrm { A s } \mu$ is uniform, the accepted proposal therefore satisfies $\begin{array} { r } { Q _ { T } ^ { \mathrm { R S } } ( 1 ) = \frac { a _ { 1 } ( T ) } { a _ { 0 } ( T ) + a _ { 1 } ( T ) } \le \frac { 3 } { 4 } = \pi ( 1 ) } \end{array}$ . Also, note that the fresh proposal used after complete rejection assigns state 1 probability $1 / 2$ . Hence every pilot-conditioned output underweights state 1, so averaging over the pilot cannot cancel the error.

When $M \ \leq \ 2$ , consider the event that all n pilot weights equal $1 / 2$ , which has probability $2 ^ { - n }$ This gives $T = M / 2 \leq 1$ . State 1 is always accepted, whereas state 0 is accepted with probability $a _ { 0 } ( T ) = \mathrm { m i n } \{ 1 , 1 / ( 2 T ) \} \geq 1 / 2$ . The probability of state 1 conditional on acceptance is therefore $1 / ( 1 + a _ { 0 } ( T ) ) \leq 2 / 3$ . The fallback assigns state 1 probability $1 / 2 .$ , so the entire conditional output assigns state 1 probability at most $2 / 3$ . Hence $\begin{array} { r } { D _ { \mathrm { T V } } ( P _ { N , \delta } ^ { R } , \pi ) \ge 2 ^ { - n } ( \frac { 3 } { 4 } - \frac { 2 } { 3 } ) = \frac { 2 ^ { - n } } { 1 2 } } \end{array}$ . For $M > 2$ , the opposite failure occurs: on a constant probability pilot event there is no clipping, but each proposal is rejected with probability at least $1 / 2$ , so complete rejection leaves an $\Omega ( 2 ^ { - \bar { n } } )$ error through the fresh proposal. See Appendix C.3 for the full proof.

For UR, on the other hand, a score stays below $3 / 2$ exactly when $Y _ { i } = 0$ and $U _ { i } > 1 / 3$ , with probability $( 1 / 2 ) ( 2 / 3 ) = 1 / 3$ . All N scores stay below with probability $3 ^ { - N }$ , on which UR returns 0; on the complement its law is π by equation 12. Its exact $\bar { \mathrm { T V } }$ error is therefore $( 3 / 4 ) 3 ^ { - N }$

Beyond a difference in exponential rates. Despite an error ratio of at least $\scriptstyle { \frac { 1 } { 3 } } ( 9 / 2 ) ^ { ( N - 1 ) / 2 }$ , both methods have $\Theta ( \log ( 1 / \xi ) )$ sample complexity for TV tolerance $\xi \downarrow 0$ with optimal tuning; see Appendix C.3. We now fix N and ask whether calibration can recover the bounded-weight guarantee of Corollary 1 within a pair-independent factor.

Corollary 2 (No uniform constant-factor guarantee). Fix an odd $N \geq 1 3 .$ . There is no finite constant $K _ { N }$ , independent $o f ( \pi , \mu )$ , satisfying

$$
\operatorname* { i n f } _ { \delta \in ( 0 , 1 ) } D _ { \mathrm { T V } } ( P _ { N , \delta } ^ { R } , \pi ) \leq K _ { N } \left( 1 - \frac { 1 } { C _ { \infty } ^ { \pi } } \right) ^ { N }
$$

for all strictly positive target–proposal pairs on finite state spaces.

Proof. Proposition 4 in Appendix C.4 gives a binary family $( \pi _ { \eta } , \mu _ { \eta } )$ with lower weight proposal probability η and $C _ { \infty } ^ { \pi _ { \eta } } = ( 1 - \eta / 2 ) ^ { - 1 }$ . Therefore, $1 - 1 / C _ { \infty } ^ { \pi _ { \eta } } = \eta / 2$ , and the optimized error $\Theta _ { N } ( \eta ^ { 2 } )$ divided by the benchmark $( \eta / 2 ) ^ { N }$ is $\Theta _ { N } \big ( \eta ^ { 2 - N } \big ) \to \infty$ as $\eta \downarrow 0$ at fixed N. □

## 3.2 GENERAL COMPARISON

The comparison is not limited to these binary examples: UR has no greater TV error than budgetcalibrated RS for every $( \pi , \mu )$

Theorem 2 (Marginal error relative to budget-calibrated RS). For every pair satisfying the positiveweight assumptions ofSection 1, every $n \geq 1$ , and every $\delta \in ( 0 , 1 )$

$$
D _ { \mathrm { T V } } ( P _ { 2 n + 1 } ^ { U } , \pi ) \leq D _ { \mathrm { T V } } ( P _ { n + 1 } ^ { U } , \pi ) \leq D _ { \mathrm { T V } } ( P _ { 2 n + 1 , \delta } ^ { R } , \pi ) .
$$

More generally,

$$
D _ { f } ( P _ { 2 n + 1 } ^ { U } \| \pi ) \leq D _ { f } ( P _ { n + 1 } ^ { U } \| \pi ) \leq D _ { f } ( P _ { 2 n + 1 , \delta } ^ { R } \| \pi )\tag{18}
$$

for every convex $f : ( 0 , \infty ) \to \mathbb { R }$ with $f ( 1 ) = 0$ . Here $\begin{array} { r } { D _ { f } ( P \| \pi ) : = \int f ( \mathrm { d } P / \mathrm { d } \pi ) } \end{array}$ dπ denotes f-divergence (Sason & Verdú, 2016).

Proof. See Appendix C.2.

Consequently, UR needs only $n + 1$ proposals, no more than the baseline uses in total on any run.   
Giving UR the full $2 n + 1$ budget cannot increase its divergence from π.

## 4 COMPARISON WITH SAMPLING IMPORTANCE RESAMPLING (SIR)

A standard threshold-free alternative is SIR, which selects candidate i with probability $w _ { u } ( Y _ { i } ) / \sum _ { i } w _ { u } ( Y _ { j } )$ . Let $\widehat { Y } _ { \mathrm { S I R } }$ denote the returned candidate and $P _ { N } ^ { \mathrm { S I R } } : = \mathcal { L } ( \widehat { Y } _ { \mathrm { S I R } } )$ its marginal output distribution. We first illustrate the difference on a binary example with $N = 2$ , and then establish the general comparison at the same proposal budget.

## 4.1 EXAMPLE WHERE UR AND SIR DIFFER

Consider the binary family where $\begin{array} { r } { \mu = \left( \frac { 1 } { 2 } , \frac { 1 } { 2 } \right) } \end{array}$ and $\pi = ( \varepsilon , 1 - \varepsilon )$ . Here $0 < \varepsilon < \textstyle { \frac { 1 } { 2 } } . \mathrm { A t } \varepsilon = 1 / 4 .$ , this is the fixed pair in Proposition 2. State 1 has the larger target probability, and the two importance weights are $\bar { ( } w _ { L } , w _ { H } ) \bar { = } ( 2 \varepsilon , 2 ( 1 - \varepsilon ) )$ , where $C _ { \infty } ^ { \pi } = 2 ( 1 - \varepsilon )$ . Fix $N = \bar { 2 }$ and write $B : = { \overline { { ( Y _ { 1 } , Y _ { 2 } ) } } }$ . Each of 00, 01, 10, 11 has probability <sup>1</sup> . Every rule returning an observed proposal must return state 0 on 00 and state 1 on 11. The methods can therefore differ only when both states are present.

On either mixed batch, SIR chooses state 1 with probability $w _ { H } / ( w _ { L } + w _ { H } ) = 1 - \varepsilon$ . For UR, let $U _ { L } , U _ { H }$ be the independent uniform variables attached to the two candidates. The candidate in state 0 is selected exactly when $\begin{array} { r } { \frac { w _ { L } } { U _ { L } } > \frac { w _ { H } } { U _ { H } } } \end{array}$ , equivalent to $\begin{array} { r } { U _ { L } < \frac { \varepsilon } { 1 - \varepsilon } U _ { H } } \end{array}$ . Conditioning on $U _ { H } = u$ yields:

$$
\operatorname* { P r } ( { \widehat { Y } } _ { U } = 0 \mid B = 0 1 ) = \int _ { 0 } ^ { 1 } \operatorname* { P r } \left( U _ { L } < { \frac { \varepsilon } { 1 - \varepsilon } } u \right) \mathrm { d } u = { \frac { \varepsilon } { 2 ( 1 - \varepsilon ) } } .\tag{19}
$$

Hence UR chooses state 1 with probability $1 - \frac { \varepsilon } { 2 ( 1 - \varepsilon ) }$ on either mixed batch. RS with $M = w _ { H } = C _ { \infty } ^ { \pi }$ accepts state 0 with probability $\frac { \varepsilon } { 1 - \varepsilon }$ and state 1 with probability one. It returns the first accepted

proposal, or an observed proposal if both are rejected. We note that this reference exploits the true unnormalized envelope $Z w _ { H }$ . Averaging over the four equally likely batches in turn gives

$$
\operatorname* { P r } ( \widehat { Y } _ { U } = 1 ) = \frac { 3 } { 4 } - \frac { \varepsilon } { 4 ( 1 - \varepsilon ) } , \qquad \operatorname* { P r } ( \widehat { Y } _ { \mathrm { { S I R } } } = 1 ) = \frac { 3 } { 4 } - \frac { \varepsilon } { 2 } .\tag{20}
$$

For RS with $M = w _ { H }$ , averaging the probabilities of returning state 1 over the two mixed batches gives $1 - \frac { \varepsilon } { 2 ( 1 - \varepsilon ) }$ , matching UR’s probability. Hence, the two output distributions agree after averaging.

Where the improvement comes from. On a mixed batch, SIR chooses state 1 with its target probability $1 - \varepsilon .$ . This does not make the marginal output exact, because the homogeneous batches 00 and 11 contribute unequal errors. By contrast, UR partially compensates for this imbalance by choosing state 1 more often when both states are available. Consequently,

$$
\underbrace { \frac { 1 - 2 \varepsilon } { 4 } } _ { \mathrm { S I R e r r o r } } = \underbrace { \frac { \varepsilon ( 1 - 2 \varepsilon ) } { 4 ( 1 - \varepsilon ) } } _ { \mathrm { r e d u c t i o n u n d e r \ : U R } } + \underbrace { \frac { ( 1 - 2 \varepsilon ) ^ { 2 } } { 4 ( 1 - \varepsilon ) } } _ { \mathrm { U R e r r o r } } .\tag{21}
$$

When $\textstyle \varepsilon = { \frac { 1 } { 4 } }$ , the identity gives $\textstyle { \frac { 1 } { 8 } } = { \frac { 1 } { 2 4 } } + { \frac { 1 } { 1 2 } }$

Why RS with $M = w _ { H } = C _ { \infty } ^ { \pi }$ agrees with UR. The calculation above shows that UR and RS with the exact envelope have the same output distribution. From the threshold view, on $H _ { w _ { H } }$ both conditional output distributions are π. Moreover, since any state 1 proposal has score above $w _ { H }$ $H _ { w _ { H } } ^ { c }$ can occur only on batch 00, where both rules return state 0. Appendix F.2 extends this equality to general two-level weights.

## 4.2 COMPARISON FOR GENERAL DISTRIBUTIONS

Theorem 3 (Marginal error relative to SIR). Under the positive weight assumptions in Section 1, for every fixed $N \geq \bar { 1 }$

$$
D _ { \mathrm { T V } } ( P _ { N } ^ { U } , \pi ) \leq D _ { \mathrm { T V } } ( P _ { N } ^ { \mathrm { S I R } } , \pi ) ,\tag{22}
$$

More generally,

$$
D _ { f } ( P _ { N } ^ { U } \| \pi ) \le D _ { f } ( P _ { N } ^ { \mathrm { { S I R } } } \| \pi )\tag{23}
$$

for every convex $f : ( 0 , \infty ) \to \mathbb { R }$ with $f ( 1 ) = 0 ,$ , allowing infinite divergences.

Why the comparison holds. Appendix E proves that, for every $t > 0 , \mathrm { P r } ( w ( \widehat { Y } _ { \mathrm { S I R } } ) > t ) \leq$ $\operatorname* { P r } ( w ( \widehat { Y } _ { U } ) > t ) \leq \pi ( w > t )$ . Hence, relative to SIR, UR shifts probability toward larger importance weights without overshooting the target tail probabilities. We highlight that this alone would not imply a TV comparison. A second property is that the density of the UR output relative to π is nonincreasing in the weight. Its excess probability is therefore at smaller weights and its shortfall at larger weights, so the threshold comparison controls the set attaining its TV error. Appendix E gives the formal argument and extends it to every convex f-divergence.

Beyond two proposals. We now return to the binary example and let the proposal budget grow.

Proposition 3. Fix $\varepsilon \in ( 0 , 1 / 2 )$ , and let $\mu = ( 1 / 2 , 1 / 2 )$ and $\pi = ( \varepsilon , 1 - \varepsilon )$ . For every $N \geq 1$

$$
D _ { \mathrm { T V } } ( P _ { N } ^ { U } , \pi ) = ( 1 - \varepsilon ) \left( \frac { 1 - 2 \varepsilon } { 2 ( 1 - \varepsilon ) } \right) ^ { N } ,\tag{24}
$$

$$
D _ { \mathrm { T V } } ( P _ { N } ^ { \mathrm { S I R } } , \pi ) = \frac { 2 \varepsilon ( 1 - \varepsilon ) ( 1 - 2 \varepsilon ) } { N } + O _ { \varepsilon } ( N ^ { - 3 / 2 } ) .\tag{25}
$$

Proof. See Appendix F.1.

The separation persists even when both states appear in nearly every batch. SIR retains the leading bias of $O ( 1 / N )$ caused by random candidate counts and normalization, whereas $\mathrm { U R } ' _ { \mathrm { s } }$ remaining error is exponentially small. Appendix F.3 shows that this phenomenon extends to every fixed nontrivial pair with bounded positive weights.

## 5 UNIQUENESS OF THE SELECTION PROBABILITIES

Sections 3 and 4 compare UR’s output law with budget-calibrated RS and SIR, respectively. These comparisons show advantages over particular alternatives, but leave a structural question: can different selection probabilities satisfy the $R S$ guarantee for every target–proposal pair? Within the class of sampling algorithms restricted as below, we prove that the answer is no: the guarantee determines the probabilities on every fixed positive weight vector, and they must coincide with UR’s.

What can the sampler use? Fix $N \geq 2$ and let the algorithm A return one of the N candidates. We restrict our attention to a class of selection rules satisfying two structural properties: First, its selection probabilities depend only on the observed weights up to a common positive factor, not on candidate states or other information about $( \pi , \mu )$ . The same rule must be used for every pair. Second, changing the order of the candidates must only reorder their selection probabilities. For instance, weights (1, 2) and (10, 20) must give the same probabilities, while (2, 1) must exchange them. We note that both UR and SIR satisfy these restrictions. For a fixed weight vector $w _ { 1 : N } : = ( w _ { 1 } , \dots , w _ { N } )$ , let $p _ { i } ^ { \mathsf { A } } ( w _ { 1 : N } )$ be the probability of selecting candidate i over the sampler’s random choices. The restrictions require, for every $c > 0$ and permutation σ,

$$
\begin{array} { r } { p _ { i } ^ { \mathsf { A } } ( c w _ { 1 : N } ) = p _ { i } ^ { \mathsf { A } } ( w _ { 1 : N } ) , \qquad p _ { \sigma ( i ) } ^ { \mathsf { A } } ( \sigma w _ { 1 : N } ) = p _ { i } ^ { \mathsf { A } } ( w _ { 1 : N } ) , } \end{array}\tag{26}
$$

where $( \sigma w _ { 1 : N } ) _ { \sigma ( i ) } = w _ { i }$ . Write $p _ { i } ^ { U } ( w _ { 1 : N } )$ for UR’s probability on the same vector and $P _ { N } ^ { \mathsf { A } }$ for the output distribution after averaging over the proposal draws. The following theorem requires the bounded weight guarantee from Corollary 1.

Theorem 4. Fix $N \geq 2$ and suppose A satisfies the information restriction above and equation 26. If a finite constant $L _ { N }$ , independent of $( \pi , \mu )$ , satisfies

$$
D _ { \mathrm { T V } } \bigl ( P _ { N } ^ { \mathsf { A } } , \pi \bigr ) \leq L _ { N } \left( 1 - \frac { 1 } { C _ { \infty } ^ { \pi } } \right) ^ { N }\tag{27}
$$

for every target and proposal on a finite state space with strictly positive probabilities, then

$$
p _ { i } ^ { \mathsf { A } } \bigl ( w _ { 1 : N } \bigr ) = p _ { i } ^ { U } \bigl ( w _ { 1 : N } \bigr ) \qquad f o r e \nu e r y w _ { 1 : N } \in ( 0 , \infty ) ^ { N } a n d i \in [ N ] .
$$

Conversely, UR satisfies the bound with $L _ { N } = 1$

Proof. See Appendix G. We note that the requirement ranges over all $( \pi , \mu )$ at fixed N, so a different rule may have smaller error on a particular pair without satisfying this requirement. □

## 6 TEST-TIME SCALING EXPERIMENTS

We complement our theoretical findings with controlled LLM experiments on math reasoning tasks where the samplers approximate a reward-tilted target policy, also known as soft BoN (Aminian et al., 2026; Verdun et al., 2025). We evaluate both sampling fidelity, measured by TV distance, and ground-truth answer accuracy.

Experimental setup and evaluation. For each prompt x, we pre-generate a fixed response pool $\mathcal { A } _ { x } ^ { \star }$ of $N ^ { \star }$ responses from Llama-3.2-3B-Instruct (Grattafiori et al., 2024). Let ${ \widehat { \mu } } ( y \mid x )$ denote the empirical distribution on $\mathcal { A } _ { x } ^ { \star }$ . For each sampling budget N, we draw N candidates i.i.d. from ${ \widehat { \mu } } ( \cdot \ { \bar { \vert } } \ x )$ and apply each sampling procedure. Each response y is scored by the OASST reward model (Köpf et al., 2023), yielding $\widehat { r } ( x , y )$ . Motivated by the standard KL-regularized RLHF objective (Jaques et al., 2017; 2020; Rafailov et al., 2023), we define the reward-tilted target policy $\begin{array} { r } { \pi _ { \beta } ( y \mid x ) \propto \widehat { \mu } ( y \mid x ) \exp ( \frac { \widehat { r } ( x , y ) } { \beta } ) , \ y \in \mathcal { A } _ { x } ^ { \star } } \end{array}$ , so that the unnormalized importance weight for each sampler is $\begin{array} { r } { w _ { u } ( x , y ) = \exp ( \frac { \widehat { r } ( x , y ) } { \beta } ) } \end{array}$ . We compare four methods: UR (OURS), ENVELOPE RS with threshold equal to the maximum weight in $\mathcal { A } _ { x } ^ { \star }$ , BUDGET-CALIBRATED RS, adapted from Rohatgi et al. (2025, Algorithm 5), and SIR. We evaluate on GSM8K (Cobbe et al., 2021), varying the sampling budget N and temperature parameter $\beta .$ Estimates of the TV error $D _ { \mathrm { T V } } ( P _ { N } , \pi _ { \beta } )$ and ground-truth accuracy are based on $1 0 ^ { 6 }$ Monte Carlo trials, with ground-truth accuracy defined as the probability that the selected response has the correct final answer. We also evaluate on MATH500 (Hendrycks et al., 2021; Lightman et al., 2024), where we observe the same trends; additional results and implementation details are given in Appendix I.

![](images/6866fb917fd0bc1b6f97ddfef3db7946a2508aac38ad5d2e635a07b896e15a90.jpg)  
Figure 1: TV error (top) and ground-truth accuracy (bottom) w.r.t. sampling budget $N$ on GSM8K Problem #258, for $\beta = 0 . 1 5$ (left) and $\beta = 0 . 8 5$ (right). The bands indicate 95% confidence intervals.

Results and analysis. Figure 1 reports TV error (top) and ground-truth accuracy (bottom) for the same GSM8K problem under two values of $\beta .$ Across both settings, UR achieves lower TV error than budget-calibrated RS and SIR, consistent with Theorems 2 and $^ { 3 , }$ while remaining comparable to envelope RS without requiring threshold tuning. Before reaching the Monte Carlo floor, UR exhibits substantially faster decay in TV error than budget-calibrated RS and SIR. The accuracy curves show how these distributional differences are reflected in ground-truth accuracy. For $\beta = 0 . 1 \dot { 5 }$ , the methods exhibit visible differences in TV error at larger sampling budgets, while their accuracies are already close to the common target accuracy. This indicates that the TV upper bound on the accuracy gap is loose in this regime. Specifically, accuracy probes only the probability of the correct answer event, so $\bigl | \mathrm { A c c } ( P _ { N } ; x ) - \bar { \mathrm { A c c } } ( \bar { \pi _ { \beta } } ; x ) \bigr | \le \bar { D } _ { \mathrm { T V } } ( P _ { N } , \bar { \pi } _ { \beta } )$ . For $\bar { \beta } = 0 . \bar { 8 } 5$ , the accuracy gaps track the TV errors more closely, with UR approaching the common target accuracy at a smaller budget. Additional results across prompts and values of $\beta ,$ including MATH500, are given in Appendix I.3.

## REFERENCES

Sergios Agapiou, Omiros Papaspiliopoulos, Daniel Sanz-Alonso, and Andrew M. Stuart. Importance sampling: Intrinsic dimension and computational cost. Statistical Science, 32(3):405–431, 2017. doi: 10.1214/17-STS611.

Gholamali Aminian, Idan Shenfeld, Amir R. Asadi, Ahmad Beirami, and Youssef Mroueh. Best-of-n through the smoothing lens: KL divergence and regret analysis. In The Fourteenth International Conference on Learning Representations, 2026.

Adam Block and Yury Polyanskiy. The sample complexity of approximate rejection sampling with applications to smoothed online learning. In Proceedings ofthe Thirty Sixth Conference on Learning Theory, volume 195 of Proceedings ofMachine Learning Research, pp. 228–273, 2023. URL https://proceedings.mlr.press/v195/block23a.html.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

George Deligiannidis, Pierre E. Jacob, El Mahdi Khribch, and Guanyang Wang. On importance sampling and independent Metropolis–Hastings with an unbounded weight function. arXiv preprint arXiv:2411.09514v3, 2026. URL https://arxiv.org/abs/2411.09514. Version 3, July 2026. Originally posted in 2024.

Nick Duffield, Carsten Lund, and Mikkel Thorup. Priority sampling for estimation of arbitrary subset sums. Journal ofthe ACM, 54(6):32, 2007. doi: 10.1145/1314690.1314696.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

Audrey Huang, Adam Block, Qinghua Liu, Nan Jiang, Akshay Krishnamurthy, and Dylan J. Foster. Is best-of-N the best of them? coverage, scaling, and optimality in inference-time alignment. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 25075–25126, 2025. URL https://procee dings.mlr.press/v267/huang25c.html.

Pierre E. Jacob, Christian P. Robert, and Murray H. Smith. Using parallel computation to improve independent Metropolis–Hastings based estimation. Journal of Computational and Graphical Statistics, 20(3):616–635, 2011. doi: 10.1198/jcgs.2011.10167.

Natasha Jaques, Shixiang Gu, Dzmitry Bahdanau, José Miguel Hernández-Lobato, Richard E Turner, and Douglas Eck. Sequence tutor: Conservative fine-tuning of sequence generation models with kl-control. In International Conference on Machine Learning, pp. 1645–1654. PMLR, 2017.

Natasha Jaques, Judy Hanwen Shen, Asma Ghandeharioun, Craig Ferguson, Agata Lapedriza, Noah Jones, Shixiang Gu, and Rosalind Picard. Human-centric dialog training via offline reinforcement learning. In Proceedings of the 2020 conference on empirical methods in natural language processing (EMNLP), pp. 3985–4003, 2020.

Andreas Köpf, Yannic Kilcher, Dimitri Von Rütte, Sotiris Anagnostidis, Zhi Rui Tam, Keith Stevens, Abdullah Barhoum, Duc Nguyen, Oliver Stanley, Richárd Nagyfi, et al. Openassistant conversations-democratizing large language model alignment. Advances in neural information processing systems, 36:47669–47681, 2023.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In International Conference on Learning Representations, volume 2024, pp. 39578–39601, 2024.

Jingbo Liu, Paul Cuff, and Sergio Verdú. E -resolvability. IEEE Transactions on Information Theory, 63(5):2629–2658, 2017.

Jun S. Liu. Metropolized independent sampling with comparisons to rejection sampling and importance sampling. Statistics and Computing, 6(2):113–119, 1996. doi: 10.1007/BF00162521.

Kerrie L. Mengersen and Richard L. Tweedie. Rates of convergence of the Hastings and Metropolis algorithms. The Annals ofStatistics, 24(1):101–121, 1996.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.

Dhruv Rohatgi, Abhishek Shetty, Donya Saless, Yuchen Li, Ankur Moitra, Andrej Risteski, and Dylan J. Foster. Taming imperfect process verifiers: A sampling perspective on backtracking. arXiv preprint arXiv:2510.03149, 2025. URL https://arxiv.org/abs/2510.03149.

Donald B Rubin. Comment: A noniterative sampling/importance resampling alternative to the data augmentation algorithm for creating a few imputations when fractions of missing information are modest: The sir algorithm. Journal of the American Statistical Association, 82(398):542–543, 1987.

Igal Sason and Sergio Verdú. f-divergence inequalities. IEEE Transactions on Information Theory, 62(11):5973–6006, 2016.

Øivind Skare, Erik Bølviken, and Lars Holden. Improved sampling-importance resampling and reduced bias importance sampling. Scandinavian Journal ofStatistics, 30(4):719–737, 2003.

Adrian FM Smith and Alan E Gelfand. Bayesian statistics without tears: a sampling–resampling perspective. The American Statistician, 46(2):84–88, 1992.

Claudio Mayrink Verdun, Alex Oesterling, Himabindu Lakkaraju, and Flavio P. Calmon. Soft best-of-n sampling for model alignment. In 2025 IEEE International Symposium on Information Theory (ISIT), volume 00, pp. 1–6. Institute of Electrical and Electronics Engineers (IEEE), June 2025. doi: 10.1109/isit63088.2025.11195541.

Guanyang Wang. Exact convergence analysis of the independent Metropolis–Hastings algorithms. Bernoulli, 28(3):2012–2033, 2022. URL https://arxiv.org/abs/2008.02455.

## A RELATED WORK

Approximate rejection sampling. Our problem is closely related to the approximate rejection sampling (RS) framework of Block & Polyanskiy (2023): given N independent samples from a proposal distribution $\mu ,$ the goal is to select one whose marginal distribution is close in total variation to a target distribution π. They characterize the minimax sample complexity of this problem under fdivergence constraints between target and proposal. Their upper bounds are obtained by truncating the target according to a likelihood ratio threshold and applying RS to the resulting truncated distribution. Our focus is complementary. Rather than optimizing sample complexity over a divergence class, we study a single parameter-free selection rule that uses only the observed unnormalized importance weights. In particular, UR requires neither the normalizing constant of the target nor an envelope bound or clipping threshold as an algorithmic input. The clipped RS formulation has also been used in recent inference-time sampling work (Huang et al., 2025).

Priority sampling. UR uses the same selection rule as a special case of classical priority sampling (Duffield et al., 2007). Given a fixed collection of weighted items, priority sampling assigns each item an independent priority ${ w _ { i } / U _ { i } }$ and retains the items with the largest priorities. When only one item is retained, this reduces exactly to the UR selection rule, $\begin{array} { r } { I = \arg \operatorname* { m a x } _ { i } \bar { \frac { w _ { i } } { U _ { i } } } } \end{array}$ . Classical priority sampling, however, is designed for a different statistical task: retain the top-k items with the largest priorities. Together with the (k + 1)st largest priority as a threshold, the retained items are assigned adjusted weights that yield unbiased estimators of arbitrary subset sums $( \mathrm { i } . \mathrm { e } . , \sum _ { \substack { i \in A } } w _ { i }$ for subsets A of the population). Consequently, the classical analysis is concerned with inclusion, unbiased weight estimation, and the variance of subset-sum estimators. In our setting, the items themselves are random, $Y _ { i } \stackrel { \mathrm { i i d } } { \sim } \mu ,$ , and we instead study the marginal distribution of the selected candidate $Y _ { I }$ as an approximation to the target distribution π. Therefore, while UR is a special case of the classical priority selection rule, our objective and analysis concern the induced sampling distribution rather than subset-sum estimation.

Sampling importance resampling and importance sampling. Sampling importance resampling (SIR) also generates a candidate pool from $\mu$ and selects a candidate using importance weights. Conditional on $Y _ { 1 : N } ,$ , SIR selects index i with probability $\frac { w _ { u } ( Y _ { i } ) } { \sum _ { j = 1 } ^ { N } w _ { u } ( Y _ { j } ) }$ , and therefore coincides with sampling from the self-normalized empirical importance distribution. For finite N, the resulting marginal distribution is generally different from π, although it converges to the target as the candidate pool grows. Skare et al. (2003) propose adjusted resampling probabilities to reduce this finite sample bias and compare the resulting method with independent Metropolis–Hastings. Related work on self-normalized importance sampling studies the bias and stability of estimators based on the same normalized importance weights; see, $\mathrm { e . g . }$ , Agapiou et al. (2017) and Deligiannidis et al. (2026). These works primarily concern estimation of expectations under $\pi ,$ whereas our primary object is the marginal distribution of a single selected candidate and its total variation distance from π.

Independent Metropolis–Hastings. Independent Metropolis–Hastings (IMH) also combines proposals from a fixed distribution $\mu$ with the importance ratio between π and $\mu .$ Unlike the one-shot selection rules considered here, IMH constructs a Markov chain with stationary distribution π. Its convergence properties and relationship with rejection and importance sampling have been studied extensively. Liu (1996) compares metropolized independent sampling with rejection sampling and importance sampling, while Mengersen & Tweedie (1996) establish classical convergence results for independence samplers. More recent work gives sharp convergence analyses (Wang, 2022) and studies IMH under unbounded weight functions (Deligiannidis et al., 2026). The independence of its proposals also permits parallel computation; Jacob et al. (2011) exploit batches of independent proposals to improve IMH-based estimation. Our setting differs in that no Markov chain is run: given a fixed budget of N independently generated and scored candidates, the algorithm returns one of them directly. This one-shot formulation is particularly natural for inference-time sampling, where candidate generation and scoring can be parallelized and the desired output is a single response rather than a sequence from a stationary Markov chain.

## B PROOF OF THE RS BOUND

Section 2 assembles Theorem 1 from the acceptance calculation and the comparison of conditional errors. We first prove the comparison in Lemma 2. The remaining subsections justify the conditioning, give the exact clipping error, and extend the results to zero weights. Write $\overset { \cdot } { W } : = \overset { \cdot } { w } ( Y )$ for $Y \sim \mu$ with $0 < W <$ ∞ almost surely until Appendix B.5.

## B.1 PROOF OF LEMMA 2

Fix $N \geq 1$ and $M > 0$ . Let $r _ { M }$ and $q _ { M }$ be functions of the scalar weight such that

$$
\frac { \mathrm { d } Q _ { N , M } ^ { U } } { \mathrm { d } \pi } ( y ) = r _ { M } ( w ( y ) ) , \qquad \frac { \mathrm { d } Q _ { M } ^ { \mathrm { R S } } } { \mathrm { d } \pi } ( y ) = q _ { M } ( w ( y ) ) .
$$

We suppress the fixed $N$ in $r _ { M }$ . Integrating the joint law equation 30 over $s \geq M$ gives, for $x > 0$

$$
r _ { M } ( x ) = \frac { N } { \mathrm { P r } ( H _ { M } ) } \int _ { \mathrm { m a x } \{ x , M \} } ^ { \infty } \frac { G ( s ) ^ { N - 1 } } { s ^ { 2 } } \mathrm { d } s , \quad q _ { M } ( x ) = \frac { \mathrm { m i n } \{ x , M \} } { A _ { M } x } ,
$$

where $G ( s ) : = \operatorname* { P r } ( W / U \leq s )$ for an independent $U \sim \mathrm { U n i f } ( 0 , 1 )$ . Both functions are positive and finite, and their compositions with w integrate to one under π. The comparison lemma below requires two properties: $r _ { M }$ is nonincreasing, and $r _ { M } / q _ { M }$ is nondecreasing. The first follows because the lower integration limit cannot decrease with $x .$ . For the second, both functions are constant on $x \leq M$ For $x > M$ , their ratio is a positive constant times

$$
x \int _ { x } ^ { \infty } \frac { G ( s ) ^ { N - 1 } } { s ^ { 2 } } \mathrm { d } s = \int _ { 0 } ^ { 1 } G ( x / u ) ^ { N - 1 } \mathrm { d } u ,\tag{28}
$$

using $u = x / s$ . The right side is nondecreasing in x because G is nondecreasing. The two expressions for the ratio agree at $x = M$ , proving the second property. Lemma 3, with $\check { P } = \pi , Q = \dot { Q } _ { M } ^ { \mathrm { R S } }$ , and $R = Q _ { N , M } ^ { U }$ , now proves Lemma 2.

Lemma 3 (A comparison of densities). Let $R \ll Q \ll P$ be probability measures whose densities have theform $\mathrm { d } Q / \mathrm { d } P = q ( W ) > 0$ and $\mathrm { d } R / \mathrm { d } P = { \ ' } r ( W )$ for a measurable realfunction W. Ifr is nonincreasing and r/q is nondecreasing on the range ofW, then

$$
D _ { \mathrm { T V } } ( P , R ) \leq D _ { \mathrm { T V } } ( P , Q ) .
$$

Proof. Let $D : = \{ y : r ( W ( y ) ) < 1 \}$ , the set where the density ratio of R to P is below one. Hence $D _ { \mathrm { T V } } ( P , R ) \ = P ( D ) - R ( D )$ , and it is enough to show $\hat { R ( D ) } \geq Q ( D )$ . Put $v : = r / q$ so $\mathrm { d } R / \mathrm { d } Q = v ( W )$ and $\textstyle \int { \dot { v } } ( W ) \mathrm { d } Q { \dot { = } } { \dot { 1 } }$ . If $y \in D$ and $\bar { z } \notin D$ , then $r ( W ( y ) ) < 1 \leq r ( W ( z ) )$ . Since r is nonincreasing, $W ^ { ' } ( y ) { \ ' } > W ( z )$ , and the monotonicity of v gives $v ( W ( y ) ) \geq v ( W ( z ) )$ . Thus, when $Q$ is multiplied by the density ratio $v ( W )$ to obtain $R ,$ the factors on $D$ are at least as large as those outside D. Because both measures have total mass one, this implies

$$
\begin{array} { r l } & { R ( D ) - Q ( D ) \overset { \mathrm { ( i ) } } { = } R ( D ) Q ( D ^ { c } ) - Q ( D ) R ( D ^ { c } ) } \\ & { \overset { \mathrm { ( i i ) } } { = } \displaystyle \int _ { D } \int _ { D ^ { c } } [ v ( W ( y ) ) - v ( W ( z ) ) ] Q ( \mathrm { d } z ) Q ( \mathrm { d } y ) } \\ & { \overset { \mathrm { ( i i i ) } } { \geq } 0 . } \end{array}
$$

Here (i) uses total mass one, (ii) substitutes d $R = v ( W ) { \mathrm { d } } Q $ , and (iii) uses the pointwise comparison above. The absolute integrand is bounded by $v ( W ( y ) ) + v ( W ( z ) )$ , so it is integrable. It follows that

$$
D _ { \mathrm { T V } } ( P , R ) = P ( D ) - R ( D ) \leq P ( D ) - Q ( D ) \leq D _ { \mathrm { T V } } ( P , Q ) ,
$$

where the last inequality is the definition of TV.

## B.2 PROOF OF PROPOSITION 1

We first consider the state and score of a single candidate, then account for the other candidates. Given $Y _ { i } = y$ , the change of variables $u = w ( y ) / s$ has absolute derivative $w ( y ) / s ^ { 2 }$ and requires $s > w ( y )$ . Since $w ( y ) \mu ( \mathrm { d } y ) = \pi ( \mathrm { d } y )$ ), the joint distribution is

$$
\operatorname* { P r } ( Y _ { i } \in \mathrm { d } y , S _ { i } \in \mathrm { d } s ) = { \frac { \mathbb { 1 } \left\{ w ( y ) < s \right\} } { s ^ { 2 } } } \pi ( \mathrm { d } y ) \mathrm { d } s .\tag{29}
$$

To be selected, this candidate must also have a larger score than the other $N - 1$ candidates. Let $G ( s ) : = \operatorname* { P r } ( S _ { i } \leq s )$ be the distribution function of one score. Independence yields the factor $G \dot { ( } s \dot { ) } ^ { N - 1 }$ . Crucially, this factor depends on s but not on y. The scores are atomless, so ties have probability zero, and summing over the $N$ possible selected indices yields:

$$
\operatorname* { P r } ( \widehat { Y } _ { U } \in \mathrm { d } y , S _ { ( N ) } \in \mathrm { d } s ) = \frac { N G ( s ) ^ { N - 1 } } { s ^ { 2 } } \mathbb { 1 } \{ w ( y ) < s \} \pi ( \mathrm { d } y ) \mathrm { d } s .\tag{30}
$$

Integrating over y gives the density of $S _ { ( N ) }$ as $\frac { N G ( s ) ^ { N - 1 } } { s ^ { 2 } } \pi ( w < s )$ . Therefore, for every measurable $D$ and almost every such s,

$$
\operatorname* { P r } ( \widehat { Y } _ { U } \in D \mid S _ { ( N ) } = s ) = \frac { \pi ( D \cap \{ w < s \} ) } { \pi ( w < s ) } = \pi ( D \mid w < s ) ,
$$

proving equation $9 ; { \mathrm { i f ~ } } s > C _ { \infty } ^ { \pi } , \pi ( w < s ) = 1$ , so equation 10 holds. This completes the proof.

## B.3 CONDITIONING ON THE LARGEST SCORE

The following identities are consequences of the joint law in Proposition 1. They also make precise the conditioning on a continuously distributed score.

Lemma 4 (Integrating the joint distribution). Integrating equation 30 over $s \geq M$ and dividing by $\mathrm { P r } ( H _ { M } ) \ g i \nu e s \ \bar { r } _ { M }$ above. Integrating over all positive scores gives

$$
\frac { \mathrm { d } P _ { N } ^ { U } } { \mathrm { d } \pi } ( y ) = N \int _ { w ( y ) } ^ { \infty } \frac { G ( s ) ^ { N - 1 } } { s ^ { 2 } } \mathrm { d } s .
$$

$I f C _ { \infty } ^ { \pi } < \infty$ , thenfor every measurable state set D and Borel score set $B \subset ( C _ { \infty } ^ { \pi } , \infty )$

$$
\operatorname* { P r } ( \widehat { Y } _ { U } \in D , S _ { ( N ) } \in B ) = \pi ( D ) \operatorname* { P r } ( S _ { ( N ) } \in B ) .\tag{31}
$$

Proof. The first two identities follow by integrating the nonnegative joint density. For $s > C _ { \infty } ^ { \pi } ,$ , the indicator $\mathbb { 1 } \{ w ( y ) < s \}$ is one for π-almost every y. The joint density therefore factors into $\pi ( \mathrm { d } y )$ and the density of $S _ { ( N ) }$ , which gives equation 31 after integration over $D \times B$ 口

Conditioning on $S _ { ( N ) } = s$ is not division by the probability of this event, which is zero. Instead, the joint density verifies the conditional probability formula

$$
\operatorname* { P r } ( \widehat { Y } _ { U } \in D \mid S _ { ( N ) } = s ) = \frac { \pi ( D \cap \{ w < s \} ) } { \pi ( w < s ) }
$$

where $\pi ( w < s ) > 0$ . On the remaining score values we may define it to be $\pi ( D )$ . Those values carry no score probability, since the score density contains the factor $\pi ( w < s )$ . Equation (31) also shows that, conditional on $S _ { ( N ) } > C _ { \infty } ^ { \pi }$ , the output state has law π independently of the largest score.

## B.4 ACCEPTED SAMPLES AND THE EVENT OF NO ACCEPTANCE

Put $a _ { M } : = A _ { M } / M$ and let $J _ { M }$ be the first accepted index on $H _ { M }$ . For $J _ { M } = i$ , the first $i - 1$ candidates must be rejected and candidate i must be accepted. Thus

$$
\begin{array} { r } { \mathrm { P r } ( Y _ { J _ { M } } \in { \cal D } , H _ { M } ) \overset { \mathrm { ( i ) } } { = } \displaystyle \sum _ { i = 1 } ^ { N } ( 1 - a _ { M } ) ^ { i - 1 } a _ { M } Q _ { M } ^ { \mathrm { R S } } ( { \cal D } ) } \\ { \overset { \mathrm { ( i i ) } } { = } [ 1 - ( 1 - a _ { M } ) ^ { N } ] Q _ { M } ^ { \mathrm { R S } } ( { \cal D } ) , } \end{array}
$$

where (i) follows from the independence and the accepted law of one candidate; and (ii) follows from evaluating the geometric sum. Dividing by $\mathrm { P r } ( \dot { H _ { M } } ) = 1 - ( 1 - a _ { M } ) ^ { N }$ proves the claimed conditional law. This calculation also explains why that law does not depend on $N _ { \cdot }$ . The clipping bound in Lemma 1 can be sharpened to an exact identity:

$$
\begin{array} { r l } & { D _ { \mathrm { T V } } ( \pi , Q _ { M } ^ { \mathrm { R S } } ) \stackrel { \mathrm { ( i ) } } { = } \displaystyle \int _ { \bf \Pi } \left( w ( y ) - \frac { \operatorname* { m i n } \{ w ( y ) , M \} } { A _ { M } } \right) _ { + } \mu ( { \mathrm { d } } y ) } \\ & { \qquad \stackrel { \mathrm { ( i i ) } } { = } \displaystyle \int _ { \bf \Pi } \left( w ( y ) - \frac { M } { A _ { M } } \right) _ { + } \mu ( { \mathrm { d } } y ) } \\ & { \qquad \stackrel { \mathrm { ( i i i ) } } { = } { \mathcal E } _ { M / A _ { M } } \leq { \mathcal E } _ { M } , } \end{array}\tag{32}
$$

where (i) holds since TV is the integrated positive difference of the densities. For (ii), both positive parts vanish when $w ( y ) \leq M$ , since $A _ { M } \leq 1$ , and their contents agree when $w ( y ) > M$ . Step (iii) follows due to the definition of $\mathcal { E } _ { M }$ evaluated at the threshold $\mathbf { \bar { \cal M } } / A _ { M }$ . The last inequality uses $M / A _ { M } \geq M$ and monotonicity in the threshold. Consequently, ensuring the factor $1 - \delta _ { M }$ in equation 8 gives $( 1 - \delta _ { M } ) \mathcal { E } _ { M / A _ { M } } + \delta _ { M }$ , where $\delta _ { M } = ( 1 - \dot { A } _ { M } / \dot { M } ) ^ { N }$

A direct acceptance interpretation for bounded weights. Suppose $C _ { \infty } ^ { \pi } < \infty$ . For candidate $i ,$ define

$$
H _ { i } : = \left\{ U _ { i } \leq \frac { w ( Y _ { i } ) } { C _ { \infty } ^ { \pi } } \right\} , \quad V _ { i } : = \frac { C _ { \infty } ^ { \pi } U _ { i } } { w ( Y _ { i } ) } \quad \mathrm { o n ~ } H _ { i } .
$$

For every measurable D and $0 \leq v \leq 1$

$$
\operatorname* { P r } ( Y _ { i } \in D , H _ { i } , V _ { i } \leq v ) = \int _ { D } { \frac { v w ( y ) } { C _ { \infty } ^ { \pi } } } \mu ( \mathrm { d } y ) = { \frac { v \pi ( D ) } { C _ { \infty } ^ { \pi } } } .\tag{33}
$$

After division by $\begin{array} { r } { \mathrm { P r } ( H _ { i } ) = 1 / C _ { \infty } ^ { \pi } } \end{array}$ , this factors as π $( D ) v$ . Hence, conditional on acceptance, $Y _ { i }$ has law $\pi$ and is independent of $\tilde { V _ { i } } \sim \mathrm { U n i f } ( 0 , 1 )$ . Conditioning on which candidates are accepted preserves independence of these pairs, because each acceptance uses only its own candidate and uniform. Among the accepted candidates, UR selects the smallest $V _ { i } ,$ equivalently the largest score $C _ { \infty } ^ { \pi } / V _ { i }$ . This choice uses variables independent of the accepted states, so the selected state still has law π. Together with the independent probability $1 / C _ { \infty } ^ { \pi }$ of each acceptance, this proves equation 12 directly.

## B.5 ZERO IMPORTANCE WEIGHTS

Now allow $w \geq 0$ with $\mathbb { E } _ { \mu } w = 1$ . If all observed weights are zero, let both UR and SIR choose each of the N candidate indices with probability $1 / N$ . Otherwise neither rule selects a zero weight. Write $p _ { 0 } : = \mu ( w = 0 ) < 1$ . The all zero event has probability $p _ { 0 } ^ { N }$ , and, $\mathrm { i f } \ p _ { 0 } > 0$ , both conditional output laws on that event are $\mu ( \cdot \mid w = 0 )$ . The joint density formulas still hold for positive scores, with an additional mass $p _ { 0 } ^ { N }$ at $S _ { ( N ) } = 0$ . For every $M > 0$ , the conditional laws $Q _ { N , M } ^ { \hat { U } }$ and $Q _ { M } ^ { \mathrm { R S } }$ put no mass on $\{ w = 0 \}$ . Their density formulas and monotonicity properties on $\{ w > 0 \}$ are unchanged, so Lemma 2 still applies. The mixture argument in Theorem 1 then applies with the same bound on $H _ { M } ^ { c }$ . The bounded weight conclusion is unchanged, and Appendix D includes the extra zero score event in its moment argument. For the SIR comparison, the proof on a fixed batch in Appendix E applies after zero entries are removed from a batch with a positive weight. On a batch where all weights are zero, the two rules agree. On $\{ w > 0 \}$ , the density of the UR law relative to π is still $h _ { N } ^ { U } ( w ( y ) )$ from equation 53, but its integral is now $\dot { \mathrm { ~ 1 ~ - ~ } } p _ { 0 } ^ { N }$ . All remaining mass is on a set of target probability zero. Therefore the full probability gap is attained on

$$
D : = \{ y : w ( y ) > 0 , h _ { N } ^ { U } ( w ( y ) ) < 1 \} .
$$

This is an upper weight set within $\{ w > 0 \}$ , possibly the whole positive weight set. The selectedweight comparison, including the endpoint 0 where the positive masses agree, gives $P _ { N } ^ { U } ( D ) \geq$ $P _ { N } ^ { \mathrm { S I R } } ( D )$ . Consequently,

$$
D _ { \mathrm { T V } } ( P _ { N } ^ { U } , \pi ) = \pi ( D ) - P _ { N } ^ { U } ( D ) \leq \pi ( D ) - P _ { N } ^ { \mathrm { S I R } } ( D ) \leq D _ { \mathrm { T V } } ( P _ { N } ^ { \mathrm { S I R } } , \pi ) .
$$

We retain the positive weight assumption for the general f-divergence statement, so no convention for singular mass is needed there.

## B.6 EXPONENTIAL RELAXATION: DERIVATION OF EQUATION 5

The output law $P _ { N , M } ^ { \mathrm { R S } }$ uses the clipped acceptance rule and returns an observed candidate if all N proposals are rejected. Conditional on at least one acceptance, its law is $Q _ { M } ^ { \mathrm { R S } }$ ; on the complementary event, its TV error is at most one regardless of the specified fallback. Thus the same mixture argument as in equation 8 gives

$$
D _ { \mathrm { T V } } ( P _ { N , M } ^ { \mathrm { R S } } , \pi ) \leq { \mathcal E } _ { M } + \left( 1 - \frac { A _ { M } } { M } \right) ^ { N } .
$$

To obtain the simpler form used in the Introduction, note that $0 < A _ { M } \leq 1 , 0 \leq \mathcal { E } _ { M } < 1$ , and $A _ { M } = 1 - \mathcal { E } _ { M }$ . Then

$$
\begin{array} { r l } & { { \displaystyle \mathcal E } _ { M } + \left( 1 - \frac { A _ { M } } { M } \right) ^ { N } \stackrel { \mathrm { ( i ) } } { \le } { \mathcal E } _ { M } + \exp \left( - \frac { N A _ { M } } { M } \right) } \\ & { \stackrel { \mathrm { ( i i ) } } { \le } 2 { \mathcal E } _ { M } + A _ { M } \exp ( - N / M ) } \\ & { \stackrel { \mathrm { ( i i i ) } } { \le } 2 { \mathcal E } _ { M } + \exp ( - N / M ) . } \end{array}
$$

Here (i) uses $1 - x \leq \exp ( - x )$ . For (ii), convexity of the exponential gives

$$
\exp ( - A _ { M } x ) = \exp ( \mathcal { E } _ { M } \cdot 0 + A _ { M } ( - x ) ) \leq \mathcal { E } _ { M } + A _ { M } \exp ( - x ) , \qquad x : = N / M .
$$

(iii) uses $A _ { M } \leq 1$ . This proves equation $_ { 3 ; }$ applying it inside Theorem 1 proves equation 5. The term $\exp ( - N / M )$ is not asserted to bound the rejection probability by itself when clipping occurs. The relaxation reallocates part of that probability bound to the clipping term.

## C COMPARISON WITH BUDGET-CALIBRATED RS

We analyze the budget reparameterization in equation 16. Throughout this appendix, weights are positive and finite almost surely, and $N = 2 n + 1 \geq 3$ is the maximum budget counting all pilot, rejection, and fallback proposals. For other integer budgets $N \geq 3 .$ , use $n = \bar { \lfloor ( N - 1 ) / 2 \rfloor }$ , leaving at most one proposal unused. Appendix C.1 records the conditional output law, and Appendix C.2 proves Theorem 2. Appendix C.3 proves Proposition 2 and verifies the N-dependent entries of Table 1. Appendix C.4 states and proves the complementary fixed budget local separation.

## C.1 BASELINE AND ITS CONDITIONAL LAW

Algorithm 5 of Rohatgi et al. (2025) exploits $n = 4 M \log ( 4 / \delta )$ pilot proposals, at most n fresh rejection proposals, and one fresh fallback upon complete rejection. Solving for $M$ gives equation 16. The effective threshold is therefore $M _ { N , \delta } \widehat { Z } .$ . Proposition F.2 of Rohatgi et al. (2025) applies when

$$
M _ { N , \delta } \geq 4 C _ { \infty } ^ { \pi } \quad \Longleftrightarrow \quad n \geq 1 6 C _ { \infty } ^ { \pi } \log ( 4 / \delta ) .\tag{34}
$$

Under this sufficient condition, $D _ { \mathrm { T V } } ( P _ { N , \delta } ^ { R } , \pi ) ~ \leq ~ \delta .$ Therefore, the source provides an $O ( C _ { \infty } ^ { \pi } \log ( 1 / \varepsilon ) )$ sufficient budget when $\delta = \varepsilon$ is an accuracy input. Notice that our comparisons concern the actual output law and do not assume this sufficient condition.

Condition on the pilot first. Write $M : = M _ { N , \delta }$ and define

$$
\overline { { W } } _ { n } : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } w ( Y _ { i } ) , \qquad T : = M \overline { { W } } _ { n } = \frac { M \widehat { Z } } { Z } , \qquad r _ { t } : = 1 - \frac { A _ { t } } { t } .
$$

The normalized threshold $T$ is an analysis variable; the algorithm does not know $Z .$ . Conditional on the pilot, a fresh trial is accepted with probability $A _ { T } / T$ and its accepted law is $Q _ { T } ^ { \mathrm { R S } }$ . All n trials fail with probability $r _ { T } ^ { n }$ , independently of the fresh fallback, whose law is $\mu .$ Therefore

$$
P _ { N , \delta } ^ { R } = \mathbb { E } \left[ ( 1 - r _ { T } ^ { n } ) Q _ { T } ^ { \mathrm { R S } } + r _ { T } ^ { n } \mu \right] .\tag{35}
$$

We note that the expectation is over the pilot. $\mathrm { I f } \ T \geq C _ { \infty } ^ { \pi }$ , the accepted law is exactly $\pi ;$ hence normalizer estimation affects the error through clipping and complete rejection.

## C.2 PROOF OF THEOREM 2

We first prove the TV comparison using only $k : = n + 1 \mathrm { U R }$ proposals, then extend it to convex f-divergences. A final comparison between two UR budgets gives the same-budget assertion in Theorem 2. None of these arguments uses the source’s sufficient condition equation 34.

Conditioning and randomizing the fresh-candidate order. Condition on the pilot and any independent threshold randomization, and denote the resulting fixed normalized threshold by $\dot { T }$ Generate all $k = n + 1$ fresh candidates in advance, including the fallback, and uniformly permute them. Uniformly permuting these i.i.d. candidates preserves their joint law. Assign each candidate an independent acceptance indicator with probability min $\{ w _ { i } / T , 1 \}$ , including a virtual indicator for the last candidate. That indicator does not change the output: the last candidate is returned whenever all preceding candidates are rejected, regardless of its indicator. If an accepted index exists, the output is consequently the first accepted index in the permuted order; if no index is accepted, it is the last index. Conditional on the candidate list and indicators, the rule therefore selects uniformly among accepted indices, or uniformly among all indices when none is accepted. Conditional on $T .$ , the following lemma applies at the fixed threshold $\tau = T$

Lemma ${ \textbf { 5 } } ( { \mathrm { A } } $ fixed batch comparison). Fix positive weights $w _ { 1 : k }$ and $\tau > 0$ . Retain index i independently with probability min $\{ w _ { i } / \tau , 1 \}$ , select uniformly among retained indices, and use a uniform index ifnone is retained. Let $q _ { i } ( \tau )$ be its selection probabilities. For every $t > 0$

$$
\sum _ { i : w _ { i } > t } q _ { i } ( \tau ) \leq \sum _ { i : w _ { i } > t } p _ { i } ^ { U } ( w _ { 1 : k } ) .
$$

The same inequality holds with $w _ { i } \geq t .$

Proof. For $k = 1$ , the rules coincide. Otherwise, put $w _ { \operatorname* { m a x } } : = \operatorname* { m a x } _ { i } w _ { i }$ and $I _ { t } : = \{ i : w _ { i } > t \}$ . We show that moving the threshold to $w _ { \mathrm { m a x } }$ cannot decrease the probability of selecting an index in $I _ { t }$ At this threshold, the retained set is nonempty and the rule is UR’s Bernoulli representation from Appendix E.3.

For $k \geq 2 .$ , suppose $\tau \leq w _ { \mathrm { m a x } }$ and put $v _ { i } : = \operatorname* { m i n } \{ w _ { i } , \tau \}$ . The rule is UR on the clipped vector $v _ { 1 : k }$ Couple this race with UR on $w _ { 1 : k }$ using the same uniforms. The factor $w _ { i } / v _ { i } = \operatorname* { m a x } \{ 1 , w _ { i } / \tau \}$ is nondecreasing in $w _ { i }$ . If the clipped race selects $i \in I _ { t } .$ , then for every $j \notin I _ { t }$

$$
\frac { w _ { i } } { U _ { i } } = \frac { w _ { i } } { v _ { i } } \frac { v _ { i } } { U _ { i } } > \frac { w _ { j } } { v _ { j } } \frac { v _ { j } } { U _ { j } } = \frac { w _ { j } } { U _ { j } } .
$$

Ties have probability zero. Hence UR also selects within $I _ { t }$ , proving the comparison in this case.

Now consider the case where $\tau \geq w _ { \mathrm { m a x } }$ . Set $z _ { i } : = w _ { i } / w _ { \mathrm { m a x } }$ , and write $\alpha : = w _ { \mathrm { m a x } } / \tau \in ( 0 , 1 ]$ Integrating the reciprocal size of the retained set as in Appendix E.3 yields

$$
q _ { i } ( w _ { \operatorname* { m a x } } / \alpha ) = z _ { i } \int _ { 0 } ^ { \alpha } \prod _ { j \neq i } ( 1 - z _ { j } u ) \mathrm { d } u + \frac { 1 } { k } \prod _ { j = 1 } ^ { k } ( 1 - \alpha z _ { j } ) .
$$

The first term accounts for a nonempty retained set; and the second accounts for the fallback when the retained set is empty. Consequently,

$$
\begin{array} { r l } & { \frac { \displaystyle \mathrm { d } } { \mathrm { d } \alpha } \sum _ { i \in I _ { t } } q _ { i } ( w _ { \operatorname* { m a x } } / \alpha ) \stackrel { ( \mathrm { i } ) } { = } \frac { 1 } { k } \sum _ { i \in I _ { t } } \sum _ { j \notin I _ { t } } \left[ z _ { i } \prod _ { \ell \neq i } ( 1 - \alpha z _ { \ell } ) - z _ { j } \prod _ { \ell \neq j } ( 1 - \alpha z _ { \ell } ) \right] } \\ & { \quad \quad \stackrel { \mathrm { ( i i ) } } { = } \frac { 1 } { k } \sum _ { i \in I _ { t } } \displaystyle \sum _ { j \notin I _ { t } } ( z _ { i } - z _ { j } ) \prod _ { \ell \notin \{ i , j \} } ( 1 - \alpha z _ { \ell } ) } \\ & { \quad \quad > 0 . } \end{array}
$$

Here (i) differentiates and groups terms across $I _ { t }$ and its complement. (ii) cancels the two terms containing $\alpha z _ { i } z _ { j }$ . The last inequality uses $z _ { i } \geq z _ { j }$ for the indicated indices. Hence the probability increases up to $\chi = 1$ , where the rule is UR. Empty and full sets cause no difficulty, and the argument is unchanged for $I _ { t } : = \{ i : w _ { i } \geq t \}$ □

From weight ordering to divergence ordering. Average Lemma 5 over the fresh candidates and the pilot. Writing $\widehat { Y } _ { R }$ for the baseline output and using k candidates for $\widehat { Y } _ { U }$ gives

$$
\mathrm { P r } ( w ( \widehat { Y } _ { R } ) > t ) \leq \mathrm { P r } ( w ( \widehat { Y } _ { U } ) > t ) ,
$$

with the same inequality for weights at least t. This establishes the required selected weight stochastic ordering. For the extension, notice first that both output laws have positive finite densities relative to π almost surely. For the baseline, positivity follows from the possibility of accepting its first fresh candidate, and finiteness follows from $P _ { N , \delta } ^ { \check { R } } ( D ) \leq k \mu ( D )$ . Write $h ^ { R } \check { ( } y ) : = ( \dot { \mathrm { d } } P _ { N , \delta } ^ { R ^ { < } } / \mathrm { d } \pi ) ( y )$ . By Lemma 8, $\mathrm { d } P _ { k } ^ { U } / \mathrm { d } \pi = h _ { k } ^ { U }$ ◦w, where $h _ { k } ^ { U }$ is nonincreasing. For $c > 0$ , let $D _ { c } : = \{ y : h _ { k } ^ { U } ( w ( y ) ) > c \}$ This is a lower weight set, so $P _ { k } ^ { U } ( D _ { c } ) \overset { \cdot } { \leq } P _ { N , \delta } ^ { R } ( D _ { c } )$ . For $X \sim \pi$

$$
\begin{array} { r l } & { \mathbb { E } [ ( h _ { k } ^ { U } ( w ( X ) ) - c ) _ { + } ] \stackrel { \mathrm { ( i ) } } { = } P _ { k } ^ { U } ( D _ { c } ) - c \pi ( D _ { c } ) } \\ & { \qquad \stackrel { \mathrm { ( i i ) } } { \leq } P _ { N , \delta } ^ { R } ( D _ { c } ) - c \pi ( D _ { c } ) } \\ & { \qquad \stackrel { \mathrm { ( i i i ) } } { \leq } \mathbb { E } [ ( h ^ { R } ( X ) - c ) _ { + } ] . } \end{array}
$$

Here (i) integrates over the positive set; (ii) uses the probability ordering; and (iii) integrates the positive part over the whole space. $\mathbf { A } \mathbf { t } c = 1$ , this also recovers the TV comparison. Both density ratios have mean one, so affine terms have equal expectations. Positive sums of these hinge inequalities therefore compare every convex piecewise-linear function with finitely many breakpoints. For an arbitrary allowed convex $f ,$ , subtract a supporting affine function at one, approximate the resulting nonnegative function increasingly by maxima of finitely many supporting affine functions and zero, and apply monotone convergence as in Appendix E. This proves the selection-stage inequality in equation 18, including infinite divergences. Nothing here required a sample-mean pilot: any positive finite threshold determined from an independent pilot and held fixed during the fresh stage is covered. The fresh-proposal fallback is essential to this argument; arbitrary other fallbacks and thresholds adapted to the fresh candidates are not covered.

Comparison under the same budget. It remains to compare two UR budgets. A decreasing upper bound would not establish that the actual error decreases.

Lemma 6 (UR’s marginal divergences decrease with the budget). For every positive weight pair and integers $1 \leq k \leq m$

$$
D _ { \mathrm { T V } } ( P _ { m } ^ { U } , \pi ) \leq D _ { \mathrm { T V } } ( P _ { k } ^ { U } , \pi ) .
$$

More generally, $D _ { f } ( P _ { m } ^ { U } \| \pi ) \le D _ { f } ( P _ { k } ^ { U } \| \pi )$ for every convex f covered by Theorem 2.

Proof. We first compare selected weights, then apply the same positive part argument. For a score value s with π $( w < s ) > 0$

$$
\pi ( w > t \mid w < s ) = { \left\{ \begin{array} { l l } { 0 , } & { s \leq t , } \\ { 1 - { \frac { \pi ( w \leq t ) } { \pi ( w < s ) } } , } & { s > t . } \end{array} \right. }
$$

This probability is nondecreasing in s on the score values that can occur. Using the first k candidates among the same m candidate–uniform pairs couples their maxima so that $S _ { ( k ) } \overset { \cdot } { \le } S _ { ( m ) }$ . Proposition 1 then gives $P _ { k } ^ { U } ( w > t ) \leq P _ { m } ^ { U } ( w > t )$ . The same reasoning applies to w $\geq t ,$ replacing $\pi ( w \leq t )$ by $\pi ( w < t )$ in the second branch. For $c > 0$ , the set $D _ { c } : = \{ y : h _ { m } ^ { U } ( w ( y ) ) > c \}$ is a lower weight set, and hence $P _ { m } ^ { U } ( D _ { c } ) \leq P _ { k } ^ { U } ( D _ { c } )$ . Therefore, for $X \sim \pi$

$$
\begin{array} { r l } & { \mathbb { E } [ ( h _ { m } ^ { U } ( w ( X ) ) - c ) _ { + } ] = P _ { m } ^ { U } ( D _ { c } ) - c \pi ( D _ { c } ) } \\ & { \qquad \leq P _ { k } ^ { U } ( D _ { c } ) - c \pi ( D _ { c } ) } \\ & { \qquad \leq \mathbb { E } [ ( h _ { k } ^ { U } ( w ( X ) ) - c ) _ { + } ] . } \end{array}
$$

Setting $c = 1$ proves the TV assertion. Both density ratios have mean one, so the convex approximation argument above proves the general assertion, including infinite divergences. □

Combining the selection stage comparison with Lemma 6 therefore proves Theorem 2. For an even maximum budget, the baseline leaves one proposal unused and the same budget monotonicity applies.

Lastly, notice that the proof does not use the specific form of $M _ { N , \delta } \widehat { Z } \colon$ the same argument applies to any positive finite threshold determined from an independent pilot and held fixed during the rejection stage.

## C.3 PROOF OF PROPOSITION 2: EXPONENTIAL SEPARATION ON A FIXED PAIR

We prove Proposition 2 for the fixed pair $\mu = ( 1 / 2 , 1 / 2 )$ and $\pi = ( 1 / 4 , 3 / 4 )$ . We then compare the budgets needed for a prescribed accuracy and verify the last column of Table 1. Throughout the fixed pair proof, $N = 2 n + 1$ with $n \geq 1$

Proof. First, notice that the normalized weights are $\begin{array} { r } { w ( 0 ) = \frac { \pi ( 0 ) } { \mu ( 0 ) } = 1 / 2 } \end{array}$ and $\begin{array} { r } { w ( 1 ) = \frac { \pi ( 1 ) } { \mu ( 1 ) } = 3 / 2 } \end{array}$ Fix a calibration and set $M : = M _ { N , \delta }$ . It suffices to show that every multiplier $M > 0$ leaves TV error at least $2 ^ { - n } / 1 2$ . We first record the output law of one accepted proposal at a fixed normalized threshold $\tau .$ . The acceptance probabilities of states 0 and 1 are

$$
a _ { 0 } ( \tau ) = \mathrm { m i n } \left\{ 1 , \frac { 1 } { 2 \tau } \right\} , \qquad a _ { 1 } ( \tau ) = \mathrm { m i n } \left\{ 1 , \frac { 3 } { 2 \tau } \right\} .
$$

Since $\mu$ is uniform, conditional on acceptance, the probability of state 1 is

$$
h ( \tau ) : = Q _ { \tau } ^ { \scriptscriptstyle \mathrm { R S } } ( \{ 1 \} ) = \frac { a _ { 1 } ( \tau ) } { a _ { 0 } ( \tau ) + a _ { 1 } ( \tau ) } = \left\{ \begin{array} { l l } { 1 / 2 , } & { 0 < \tau \leq 1 / 2 , } \\ { \tau / ( \tau + 1 / 2 ) , } & { 1 / 2 < \tau < 3 / 2 , } \\ { 3 / 4 , } & { \tau \geq 3 / 2 . } \end{array} \right.
$$

Hence,

$$
{ \frac { 1 } { 2 } } \leq h ( \tau ) \leq { \frac { 3 } { 4 } } = \pi ( 1 ) , \quad { \mathrm { f o r ~ e v e r y ~ } } \tau > 0 .
$$

The fresh fallback is drawn from $\mu ,$ so it assigns state 1 probability $1 / 2$ as well. Hence, for every realized pilot threshold, both the accepted law and the fallback underweight state 1 relative to its target probability $3 / 4$ . In particular, the errors arising from different pilot outcomes all have the same sign and therefore cannot cancel when we average over the pilot.

Let $r _ { T }$ denote the rejection probability of one proposal at the random threshold T. Conditional on $T ,$ at least one of the n rejection stage proposals is accepted with probability $1 - r _ { T } ^ { n }$ , while all n proposals are rejected with probability $r _ { T } ^ { n }$ , in which case the fresh fallback is used. Therefore

$$
\operatorname* { P r } ( \widehat { Y } _ { R } = 1 \mid T ) = ( 1 - r _ { T } ^ { n } ) h ( T ) + \frac { 1 } { 2 } r _ { T } ^ { n } .
$$

Since this quantity is always at most $3 / 4 .$ , the binary TV error has no absolute value cancellation, and equation 35 gives

$$
\begin{array} { r l } & { D _ { \mathrm { T V } } ( P _ { N , \delta } ^ { R } , \pi ) = \mathbb { E } \left[ \cfrac { 3 } { 4 } - ( 1 - r _ { T } ^ { n } ) h ( T ) - \cfrac { 1 } { 2 } r _ { T } ^ { n } \right] } \\ & { \quad \quad \quad = \mathbb { E } \left[ ( 1 - r _ { T } ^ { n } ) \left( \cfrac { 3 } { 4 } - h ( T ) \right) + \cfrac { 1 } { 4 } r _ { T } ^ { n } \right] . } \end{array}\tag{36}
$$

We now lower-bound this expression for every $M > 0 . { \mathrm { I f } } M \leq 2 ,$ , consider the event that all n pilot weights equal the lower weight $1 / 2$ . This event has probability $2 ^ { - n }$ and gives

$$
\overline { { { W } } } _ { n } = \frac { 1 } { 2 } , \qquad T = M \overline { { { W } } } _ { n } = \frac { M } { 2 } \leq 1 .
$$

Since $h ( \tau )$ is nondecreasing,

$$
h ( T ) \leq h ( 1 ) = \frac { 1 } { 1 + 1 / 2 } = \frac { 2 } { 3 } .
$$

On this event, both the accepted law and the fallback assign state 1 probability at most $2 / 3 .$ , so the conditional TV error is at least

$$
{ \frac { 3 } { 4 } } - { \frac { 2 } { 3 } } = { \frac { 1 } { 1 2 } } .
$$

Since all conditional errors have the same sign, we obtain

$$
D _ { \mathrm { T V } } ( P _ { N , \delta } ^ { R } , \pi ) \geq 2 ^ { - n } \cdot \frac { 1 } { 1 2 } .
$$

Now consider the case where $M \ > \ 2$ . Since each pilot weight equals $1 / 2$ or $3 / 2$ with equal probability, $\overline { { W } } _ { n } - 1$ is symmetric about zero. Hence $\operatorname* { P r } ( { \overline { { W } } } _ { n } \geq 1 ) \geq { \frac { 1 } { 2 } }$ . On this event,

$$
T = M \overline { { { W } } } _ { n } \geq M > 2 > \frac { 3 } { 2 } ,
$$

so there is no clipping and hence $h ( T ) = 3 / 4$ . Moreover, since $\mathbb { E } _ { \mu } [ w ( Y ) ] = 1$ , the acceptance probability of one proposal is $1 / T$ , and therefore

$$
r _ { T } = 1 - \frac { 1 } { T } \geq \frac 1 2 , \qquad r _ { T } ^ { n } \geq 2 ^ { - n } .
$$

The clipping term in equation 36 vanishes, while the fallback term contributes at least $\textstyle { \frac { 1 } { 4 } } r _ { T } ^ { n } \geq { \frac { 1 } { 4 } } 2 ^ { - n }$ Since the event $\{ \overline { { W } } _ { n } \geq 1 \}$ has probability at least $1 / 2$

$$
D _ { \mathrm { T V } } ( P _ { N , \delta } ^ { R } , \pi ) \geq \frac { 1 } { 2 } \cdot \frac { 1 } { 4 } 2 ^ { - n } = \frac { 1 } { 8 } 2 ^ { - n } \geq \frac { 1 } { 1 2 } 2 ^ { - n } .
$$

Hence the same lower bound holds after taking the infimum over $\delta \in ( 0 , 1 )$

The UR formula in Proposition 3 gives $\begin{array} { r } { D _ { \mathrm { T V } } ( P _ { k } ^ { U } , \pi ) = \frac { 3 } { 4 } 3 ^ { - k } } \end{array}$ , which yields the claimed equality at $k = N = 2 n + 1$ . This completes the proof. □

The gap persists with fewer UR proposals. Every baseline run uses at least $n + 1$ proposals in total, including its pilot. Even with only this many proposals, UR satisfies

$$
{ \frac { \operatorname* { i n f } _ { \delta \in ( 0 , 1 ) } D _ { \mathrm { T V } } ( P _ { 2 n + 1 , \delta } ^ { R } , \pi ) } { D _ { \mathrm { T V } } ( P _ { n + 1 } ^ { U } , \pi ) } } \geq { \frac { 1 } { 3 } } \left( { \frac { 3 } { 2 } } \right) ^ { n } .\tag{37}
$$

With the same total budget $N = 2 n + 1$ for both methods, the lower bound on the ratio is ${ \frac { 1 } { 3 } } ( 9 / 2 ) ^ { n }$ as stated in Section 3. The different bases correspond to different UR budgets, not different target– proposal pairs.

Error ratio versus sample complexity. An exponentially growing error ratio does not imply different orders of optimized sample complexity. Let $\xi \in ( \bar { 0 } , 1 )$ be a TV error tolerance, with $\xi \downarrow 0$ . For UR, the exact formula gives a required budget of order log(1/ξ). For calibrated RS, the lower bound requires n to be at least of this order. Conversely, $C _ { \infty } ^ { \pi } = 3 / 2$ , and choosing $\delta = \xi$ in equation 34 gives TV error at most ξ whenever the integer n satisfies

$$
n \geq 2 4 \log ( 4 / \xi ) .
$$

Hence both methods require $\Theta ( \log ( 1 / \xi ) )$ proposals when tuned for the prescribed accuracy. This statement allows the calibration to vary with the requested accuracy; it does not assert convergence for every fixed δ. Appendix C.4 addresses the separate question of matching the sharp guarantee uniformly over pairs at a fixed budget.

The budget-dependent entries of Table 1. The last column uses the fixed pair of Proposition 2. UR has exact error $( 3 / 4 ) 3 ^ { - N }$ , and known-envelope RS has the same law by Proposition 6, with a uniform observed-index fallback. Substituting $\varepsilon = 1 / 4$ into equation 25 gives

$$
D _ { \mathrm { T V } } ( P _ { N } ^ { \mathrm { S I R } } , \pi ) = \frac { 3 } { 1 6 N } + { \cal O } ( N ^ { - 3 / 2 } ) .
$$

For calibrated RS, equation 17 gives

$$
\operatorname* { i n f } _ { \delta \in ( 0 , 1 ) } D _ { \mathrm { T V } } ( P _ { N , \delta } ^ { R } , \pi ) \geq \frac { 1 } { 1 2 } 2 ^ { - ( N - 1 ) / 2 } = \frac { \sqrt { 2 } } { 1 2 } 2 ^ { - N / 2 }
$$

for every odd $N \geq 3$ . This proves the claimed $\Omega ( 2 ^ { - N / 2 } )$ lower bound; no matching $\Theta ( 2 ^ { - N / 2 } )$ rate is asserted. Appendix C.4 justifies the sharp guarantee column.

## C.4 LOCAL SEPARATION UNDER OPTIMAL CALIBRATION

The fixed pair comparison in Section 3 distinguishes exponential rates as the budget grows. Here we fix the budget and vary the pair to ask whether calibrated RS can match the sharp bounded-weight guarantee within a factor independent of the pair. Keep the unnormalized weights of states 0 and 1 equal to $1 / 2$ and 1, respectively, and set

$$
\mu _ { \eta } : = ( \eta , 1 - \eta ) , \qquad \pi _ { \eta } : = \left( \frac { \eta } { 2 - \eta } , \frac { 2 ( 1 - \eta ) } { 2 - \eta } \right) , \qquad 0 < \eta < 1 .\tag{38}
$$

Hence η is the proposal probability of the lower weight state. The normalizer is $Z _ { \eta } = 1 - \eta / 2$ , and $C _ { \infty } ^ { \pi _ { \eta } } = 1 / Z _ { \eta }$ . In particular, the bounded-weight benchmark is $( 1 - 1 / C _ { \infty } ^ { \pi _ { \eta } } ) ^ { N } = ( \dot { \eta } / 2 ) ^ { N }$

Proposition 4 (Local separation under optimal calibration). Fix an odd $N \geq 1 3$ . For the family in equation 38, as $\eta \downarrow 0 ,$

$$
\operatorname* { i n f } _ { \delta \in ( 0 , 1 ) } D _ { \mathrm { T V } } ( P _ { N , \delta } ^ { R } , \pi _ { \eta } ) = \frac { N - 1 } { 4 ( N - 2 ) } \eta ^ { 2 } + o _ { N } ( \eta ^ { 2 } ) ,\tag{39}
$$

whereas

$$
D _ { \mathrm { T V } } ( P _ { N } ^ { U } , \pi _ { \eta } ) = \frac { 1 - \eta } { 1 - \eta / 2 } \left( \frac { \eta } { 2 } \right) ^ { N } .\tag{40}
$$

The infimum allows pair- and budget-dependent tuning before observing the pilot. The UR identity holdsfor every $N \geq 1$ and $0 < \eta < 1$

Proof. Fix $n \geq 6$ and $N = 2 n + 1$ , and let $\eta \downarrow 0$ . For the lower bound, allow any deterministic multiplier $M > 0$ chosen before observing the pilot. Let J count the state 0 candidates among the n pilot proposals. Then $J \sim \mathrm { B i n } ( n , \eta )$ , and

$$
\widehat Z = { \frac { J / 2 + n - J } { n } } = 1 - { \frac { J } { 2 n } } , \qquad \tau _ { J } : = M \widehat Z = M \left( 1 - { \frac { J } { 2 n } } \right) .
$$

Here $\tau _ { J }$ is the raw threshold; its normalized counterpart is $T = \tau _ { J } / Z _ { \eta }$ . Allowing all $M > 0$ enlarges the calibrated family. The upper-bound choice $\dot { M } = 1$ is feasible in the original family because $\delta = 4 \exp ( - n / 4 ) \stackrel { \cdot } { \in } ( 0 , 1 )$ gives $M _ { N , \delta } = 1$

The error conditional on the pilot. The accepted probability of state 0 is

$$
q _ { \eta } ( \tau ) = \left\{ \begin{array} { l l } { \eta , } & { 0 < \tau \leq 1 / 2 , } \\ { \frac { \eta / 2 } { \eta / 2 + ( 1 - \eta ) \tau } , } & { 1 / 2 < \tau < 1 , } \\ { \frac { \eta } { 2 - \eta } , } & { \tau \geq 1 . } \end{array} \right.
$$

It is nonincreasing in τ and lies between $\pi _ { \eta } ( 0 )$ and $\mu _ { \eta } ( 0 ) = \eta$ . Write

$$
d _ { \eta } ( \tau ) : = q _ { \eta } ( \tau ) - \pi _ { \eta } ( 0 ) , \qquad g _ { \eta } : = \eta - \pi _ { \eta } ( 0 ) = \frac { \eta ( 1 - \eta ) } { 2 - \eta } .
$$

Let $r _ { \eta } ( \tau ) : = r _ { \tau / Z _ { \eta } }$ be the rejection probability of one fresh trial; the division by $Z _ { \eta }$ converts the raw threshold to the normalized units of Appendix C.1.

Both acceptance and fallback assign state 0 at least its target probability, so averaging over the pilot cannot cancel this excess. Consequently, equation 35 gives the exact $\mathrm { \bar { T V } }$ error of the relaxed rule:

$$
\begin{array} { r } { e _ { n } ( M , \eta ) = \mathbb { E } \left[ d _ { \eta } ( \tau _ { J } ) + r _ { \eta } ( \tau _ { J } ) ^ { n } \left( g _ { \eta } - d _ { \eta } ( \tau _ { J } ) \right) \right] = \mathbb { E } \left[ ( 1 - r _ { \eta } ( \tau _ { J } ) ^ { n } ) d _ { \eta } ( \tau _ { J } ) + r _ { \eta } ( \tau _ { J } ) ^ { n } g _ { \eta } \right] . } \end{array}\tag{41}
$$

The expectation is over J. The integrand lies in $[ d _ { \eta } ( \tau _ { J } ) , g _ { \eta } ]$ , a fact used for both bounds below.

Step 1: Upper bound from $M = 1 .$ . For $J ~ = ~ 0$ , the threshold is one, the accepted law is exact, and $r _ { \eta } ( 1 ) = \eta / 2$ . The contribution is at most $g _ { \eta } ( \eta / 2 ) ^ { n } = O _ { n } ( \eta ^ { n + 1 } )$ . For $J = 1$ , put $v _ { n } : = 1 - 1 / ( 2 n ) \in ( \mathrm { 1 } / 2 , 1 )$ , where we use $n \geq 2$ . Direct expansion gives

$$
d _ { \eta } ( v _ { n } ) = { \frac { 1 } { 2 } } \left( { \frac { 1 } { v _ { n } } } - 1 \right) \eta + { \cal O } _ { n } ( \eta ^ { 2 } ) = { \frac { \eta } { 2 ( 2 n - 1 ) } } + { \cal O } _ { n } ( \eta ^ { 2 } ) .
$$

Here $r _ { \eta } ( v _ { n } ) = \eta ( 1 - 1 / ( 2 v _ { n } ) ) = O _ { n } ( \eta )$ and $\operatorname* { P r } ( J = 1 ) = n \eta + O _ { n } ( \eta ^ { 2 } )$ . The contribution is consequently $n \eta ^ { 2 } / [ 2 ( 2 n - 1 ) ] + O _ { n } ( \eta ^ { 3 } )$ . Finally, $\operatorname* { P r } ( J \geq 2 ) = O _ { n } ( \eta ^ { 2 } )$ and each conditional error is at most $g _ { \eta } = O ( \eta )$ . Hence

$$
e _ { n } ( 1 , \eta ) = \frac { n } { 2 ( 2 n - 1 ) } \eta ^ { 2 } + O _ { n } ( \eta ^ { 3 } ) .\tag{42}
$$

Step 2: Typical pilot forces $M _ { \eta } \to 1$ . Suppose $e _ { n } ( M _ { \eta } , \eta ) = O _ { n } ( \eta ^ { 2 } )$ , and fix $\gamma \in ( 0 , 1 / 2 )$ . If $M _ { \eta } \leq 1 - \gamma$ along a subsequence, then the event $J = 0$ and the monotonicity of $d _ { \eta }$ give

$$
e _ { n } ( M _ { \eta } , \eta ) \geq ( 1 - \eta ) ^ { n } d _ { \eta } ( 1 - \gamma ) , \qquad \frac { d _ { \eta } ( 1 - \gamma ) } { \eta } \to \frac { \gamma } { 2 ( 1 - \gamma ) } > 0 .
$$

This contradicts the assumed second-order error. If instead $M _ { \eta } \ge 1 + \gamma$ along a subsequence, then on $J = 0$ there is no clipping, but $r _ { \eta } ( M _ { \eta } ) = 1 - Z _ { \eta } / M _ { \eta } \geq \gamma \dot { / } ( 1 + \gamma )$ . Hence

$$
e _ { n } ( M _ { \eta } , \eta ) \geq ( 1 - \eta ) ^ { n } \left( \frac { \gamma } { 1 + \gamma } \right) ^ { n } g _ { \eta } .
$$

Since $g _ { \eta } / \eta \to 1 / 2$ , this again leaves first-order error. Therefore $M _ { \eta } \to 1$

Step 3: The event $J = 1$ forces the leading coefficient. For each η, choose $M _ { \eta } > 0$ such that

$$
e _ { n } ( M _ { \eta } , \eta ) \le \operatorname* { i n f } _ { M > 0 } e _ { n } ( M , \eta ) + \eta ^ { 3 } .
$$

Step 1 makes this sequence second-order accurate, so Step 2 gives $M _ { \eta } \to 1$ . Hence $\tau _ { 1 } = M _ { \eta } v _ { n }  v _ { n }$ and

$$
\frac { d _ { \eta } ( M _ { \eta } v _ { n } ) } { \eta }  \frac { 1 } { 2 ( 2 n - 1 ) } .
$$

The nonnegative rejection term in equation 41 can only increase the error. Keeping just the $J = 1$ contribution therefore gives

$$
\operatorname* { i n f } _ { M > 0 } e _ { n } ( M , \eta ) \ge \operatorname* { P r } ( J = 1 ) d _ { \eta } ( M _ { \eta } v _ { n } ) - \eta ^ { 3 } .
$$

Dividing by $\eta ^ { 2 }$ and using $\operatorname* { P r } ( J = 1 ) / \eta $ n proves

$$
\operatorname* { l i m } _ { \eta \downarrow 0 } \operatorname* { i n f } _ { } \frac { \operatorname* { i n f } _ { M > 0 } e _ { n } ( M , \eta ) } { \eta ^ { 2 } } \geq \frac { n } { 2 ( 2 n - 1 ) } .
$$

Together with the feasible upper bound in equation 42, this proves equation 39, since $n / [ 2 ( 2 n - 1 ) ] =$ $( N - 1 ) / [ 4 ( N - 2 ) $ ]. In particular, the multiplier may depend on the pair and budget, but not on the realized pilot. □

UR’s exact error. The largest normalized weight is $C _ { \infty } ^ { \pi _ { \eta } } = 1 / Z _ { \eta } . \mathrm { \bf ~ A }$ state 1 candidate crosses this score level almost surely. A state 0 candidate stays below it exactly when its uniform exceeds $1 / 2$ , so a single candidate fails to cross with probability $\eta / 2$ . Independence gives $\mathrm { P r } ( H _ { C _ { \infty } ^ { \pi _ { \eta } } } ^ { c } ) = ( \eta / 2 ) ^ { \dot { N } }$ . On this event all candidates are in state 0, so UR returns $0 ;$ on the complement its output law is $\pi _ { \eta }$ by equation 12. The mixture identity in equation 13 therefore gives

$$
D _ { \mathrm { T V } } ( P _ { N } ^ { U } , \pi _ { \eta } ) = \pi _ { \eta } ( 1 ) \left( \frac { \eta } { 2 } \right) ^ { N } = \frac { 1 - \eta } { 1 - \eta / 2 } \left( \frac { \eta } { 2 } \right) ^ { N } ,
$$

which proves equation 40. This identity holds for every $N \geq 1$ and $0 < \eta < 1$ , not only in the local limit.

Comparison to others. At fixed N, exactly one state 0 candidate appears with probability $N \eta +$ $O _ { N } ( \bar { \eta } ^ { 2 } )$ , and SIR selects it with probability $1 / ( 2 N - 1 )$ . Batches containing at least two such candidates have total probability $\hat { \ b { O } } _ { N } ( \eta ^ { 2 } )$ . Hence $P _ { N } ^ { \mathrm { S I R } } \big ( \{ 0 \} \big ) = N \eta / ( 2 N - \mathbf { \Big { 1 } } ) + O _ { N } ( \eta ^ { 2 } )$ , and subtracting $\pi _ { \eta } ( 0 ) = \eta \dot { / 2 } + O ( \eta ^ { \bar { 2 } } )$ yields

$$
D _ { \mathrm { T V } } ( P _ { N } ^ { \mathrm { S I R } } , \pi _ { \eta } ) = \frac { \eta } { 2 ( 2 N - 1 ) } + O _ { N } ( \eta ^ { 2 } ) .
$$

Known-envelope RS uses the exact raw envelope $Z _ { \eta } C _ { \infty } ^ { \pi _ { \eta } } = 1$ and a uniform observed-index fallback. Its output law equals UR’s on this two-weight family by Proposition 6. These calculations establish the local-error column. For the sharp guarantee column, UR satisfies the universal bounded-weight bound by Corollary 1, and known-envelope RS does so by its exact accepted law and completerejection probability. On this family the bound is $( 1 - 1 / C _ { \infty } ^ { \pi _ { \eta } ^ { \bullet } } ) ^ { N } = ( \eta / 2 ) ^ { N }$ , so the positive first- and second-order coefficients above rule it out for SIR and calibrated RS, even up to a pair-independent multiplicative constant at fixed odd $N \geq 1 3$ . Even UR with only $n + 1$ proposals has error $\bar { \Theta _ { n } } \big ( \eta ^ { n + 1 } \big )$ on this family. All local limits fix the budget; no remainder uniform in N is asserted.

## D MOMENT BOUNDS AND MINIMAX RATES

The bounded weight upper bound in Corollary 1 follows from equation 12. We prove the second moment bound next, followed by the matching lower bound under a uniform weight bound. The final subsection gives a lower bound under a second moment constraint. These lower bounds use the same restriction: a sampler cannot return a state that never appears among its proposals.

## D.1 DIRECT SECOND MOMENT UPPER BOUND

Assume $C ^ { \pi } = \mathbb { E } _ { \mu } [ w ( Y ) ^ { 2 } ] < \infty$ . By Proposition 1, conditioning on $S _ { ( N ) } = s$ excludes exactly the target mass $\pi ( w \geq s )$ . We first average this conditional error, then bound the chance that the largest score is too small. Let $X \sim \pi$ be independent of the entire race, used only for analysis. Recall $G ( { \bar { t } } ) : = \operatorname* { P r } ( w ( Y ) / U \leq t )$ for independent $\bar { Y } \sim \mu$ and $U \sim \mathrm { U n i f } ( 0 , 1 )$ . Then

$$
\begin{array} { r l } & { D _ { \mathrm { T V } } ( P _ { N } ^ { U } , \pi ) \overset { \mathrm { ( i ) } } { \leq } \mathbb { E } [ \pi ( w \geq S _ { ( N ) } ) ] } \\ & { \qquad \overset { \mathrm { ( i i ) } } { = } \operatorname* { P r } ( S _ { ( N ) } \leq w ( X ) ) } \\ & { \qquad \overset { \mathrm { ( i i i ) } } { = } \mathbb { E } [ G ( w ( X ) ) ^ { N } ] , } \end{array}\tag{43}
$$

where (i) follows from convexity of TV and the conditional law; (ii) follows from using the independent target point $X \sim \pi ;$ and (iii) conditions on X and uses independence of the N scores. Under the zero weight convention of Appendix B.5, the conditional error at score zero is one, equal to $\pi ( w \geq 0 )$ ), so the same calculation applies. We now bound the chance that a single score exceeds $t > 0 \colon$

$$
\begin{array} { r l } & { 1 - G ( t ) = \mathbb { E } _ { \mu } \left[ \operatorname* { m i n } \left\{ \frac { w ( Y ) } { t } , 1 \right\} \right] } \\ & { \qquad \overset { \mathrm { ( i ) } } { \geq } \mathbb { E } _ { \mu } \left[ \frac { w ( Y ) } { t + w ( Y ) } \right] } \\ & { \qquad \overset { \mathrm { ( i i ) } } { = } \mathbb { E } \left[ \frac { 1 } { t + w ( X ) } \right] } \\ & { \qquad \overset { \mathrm { ( i i i ) } } { \geq } \frac { 1 } { t + \mathbb { E } [ w ( X ) ] } = \frac { 1 } { t + C ^ { \pi } } , } \end{array}
$$

where (i) follows from the fact that min $\textstyle \left\{ { \frac { a } { t } } , 1 \right\} \geq { \frac { a } { t + a } }$ for $a \geq 0 ;$ (ii) follows since $w \mathrm { d } \mu = \mathrm { d } \pi ;$ and (iii) follows from Jensen’s inequality for the convex function $\textstyle a \mapsto { \frac { 1 } { t + a } } .$ . This is where the second moment enters: $\mathbb { E } [ w ( X ) ] = \mathbb { E } _ { \mu } [ w ( Y ) ^ { 2 } ] = C ^ { \pi }$ . Independence and $1 - a \leq e ^ { - a }$ give $G ( t ) ^ { N } \leq e ^ { - N / ( t + C ^ { \pi } ) }$ . Substituting this into equation 43,

$$
D _ { \mathrm { T V } } ( P _ { N } ^ { U } , \pi ) \leq \mathbb { E } \left[ \exp \left( - \frac { N } { w ( X ) + C ^ { \pi } } \right) \right]\tag{44}
$$

$$
\leq \frac { \mathbb { E } [ w ( X ) ] + C ^ { \pi } } { e N } = \frac { 2 C ^ { \pi } } { e N } ,\tag{45}
$$

where the last inequality follows from the fact that $e ^ { - x } \leq 1 / ( e x )$ for $x > 0$ . Together with the trivial bound 1, this proves the second moment term in Corollary 1.

Relation to the known SIR bound. The order $C ^ { \pi } / N$ is not specific to UR. Agapiou et al. (2017, Theorem 2.1) bound the absolute bias of self-normalized importance sampling by $1 2 C ^ { \pi } / N$ for bounded measurable functions (bounded by one in absolute value). For an event $D _ { : }$ , apply that bound to $2 \Im _ { D } - 1$ . The bias is then $2 ( P _ { N } ^ { \mathrm { S I R } } ( \dot { D } ) - \pi ( D ) )$ , giving $D _ { \mathrm { T V } } ( P _ { N } ^ { \mathrm { S I R } } , \pi ) \le \operatorname* { m i n } \{ 1 , \dot { 6 } \dot { C } ^ { \tilde { \pi } } / N \}$ Theorem 3 also transfers that rate to UR. The direct proof above gives the constant $2 / e$ without using the SIR comparison, but does not establish optimality of that constant.

## D.2 EXACT BOUNDED RATIO MINIMAX ERROR

UR already supplies the upper bound in equation 15. For a lower bound, consider a binary pair with target mass one on state 1 and proposal probability $1 / C$ on that state, where $C > 1$ . Every observed-candidate sampler must return state 0 when all N proposals miss state 1. This is the same basic obstruction used in approximate sampling lower bounds (Block & Polyanskiy, 2023). To obtain the bound using strictly positive distributions, fix $C > 1$ and $0 < \eta < 1$ and take

$$
\pi _ { \eta } ( 1 ) = 1 - \eta , \quad \mu _ { \eta } ( 1 ) = \frac { 1 - \eta } { C } .
$$

Then $w _ { \eta } ( 1 ) = C$ and $w _ { \eta } ( 0 ) = \eta / [ 1 - ( 1 - \eta ) / C ] < C$ . For any sampler A returning an observed candidate,

$$
\begin{array} { r l } & { D _ { \mathrm { T V } } ( P _ { N } ^ { \mathrm { A } } , \pi _ { \eta } ) \overset { \mathrm { ( i ) } } { \geq } P _ { N } ^ { \mathrm { A } } ( \{ 0 \} ) - \eta } \\ & { \qquad \overset { \mathrm { ( i i ) } } { \geq } \left( 1 - \displaystyle \frac { 1 - \eta } { C } \right) ^ { N } - \eta . } \end{array}
$$

Here (i) evaluates the TV supremum on $\{ 0 \}$ ; and (ii) uses the event that every proposal is state $0 ,$ on which the output must also be state 0. Letting $\eta \downarrow 0$ gives $( 1 - 1 / C ) ^ { N }$ , uniformly over the choice of sampler, and proves the minimax lower bound. The argument applies even to samplers with additional information about the pair. For $C = 1 , w \leq 1$ and $\mathbb { E } _ { \mu } w = 1$ imply $w = 1$ almost surely. Thus $\pi = \mu$ and returning the first proposal is exact.

## D.3 SECOND MOMENT MINIMAX LOWER BOUND

A second moment constraint still allows a state to be much rarer under the proposal than under the target. We choose its proposal probability so that the chance of observing it is at most half its target probability.

Proposition 5 (A lower bound under a second moment constraint). Fix $R > 1$ and $N \geq 1$ . For every procedure returning one ofthe N observed proposals, there is a finite strictly positive pair with $C ^ { \pi } \leq R$ and TV error at least

$$
\frac { 1 } { 1 6 } \operatorname* { m i n } \left\{ 1 , \frac { R - 1 } { N } \right\} .
$$

For $R \geq 2 ,$ , this is at least $\textstyle { \frac { 1 } { 3 2 } } \operatorname* { m i n } \{ 1 , R / N \}$

Proof. Set

$$
p : = \operatorname* { m i n } \left\{ \frac { R - 1 } { 8 N } , \frac { 1 } { 4 } \right\} , \quad q : = \frac { p } { 2 N } , \quad \pi ( 1 ) : = p , \quad \mu ( 1 ) : = q .
$$

Both binary distributions are strictly positive. Since the output must be observed,

$$
\operatorname* { P r } ( \widehat { Y } = 1 ) \le \operatorname* { P r } ( \exists i : Y _ { i } = 1 ) \le N q = p / 2 ,
$$

where the second inequality is the union bound. The TV error is therefore at least

$$
p - \operatorname* { P r } ( \widehat { Y } = 1 ) \geq p / 2 \geq \frac { 1 } { 1 6 } \operatorname* { m i n } \left\{ 1 , \frac { R - 1 } { N } \right\} .
$$

It remains to check the moment constraint. For a binary pair, direct substitution gives

$$
C ^ { \pi } - 1 = \frac { ( p - q ) ^ { 2 } } { q ( 1 - q ) } \overset { \mathrm { ( i ) } } { \leq } \frac { 2 N p } { 1 - q } \overset { \mathrm { ( i i ) } } { \leq } \frac { 1 6 } { 7 } N p \overset { \mathrm { ( i i i ) } } { \leq } \frac { 2 } { 7 } ( R - 1 ) \leq R - 1 .
$$

Here (i) uses $( p - q ) ^ { 2 } \leq p ^ { 2 }$ and $q = p / ( 2 N ) ;$ ; (ii) and (iii) use $q \leq 1 / 8$ and $N p \leq ( R - 1 ) / 8$ . Hence $C ^ { \pi } \leq R$ . For $R \ge 2 , R - 1 \ge R / 2$ gives the second lower bound. □

For $R \geq 2 .$ , the upper and lower bounds have the same dependence on R and N, up to constants. This is a minimax statement over a moment-constrained class, not a lower bound for every fixed pair.

## E PROOFS FOR THE COMPARISON TO SIR

The proof of Theorem 3 uses two facts. UR returns weights above any threshold at least as often as SIR (Lemma 7), and UR’s density relative to π is nonincreasing in the weight (Lemma 8). We first combine these facts to prove the theorem, then establish them directly from the selection rule.

## E.1 PROOF FOR CONVEX DIVERGENCES

Let $h _ { N } ^ { U }$ and $h _ { N } ^ { \mathrm { S I R } }$ be the scalar functions in Lemma $^ { 8 , }$ so that

$$
\frac { \mathrm { d } P _ { N } ^ { U } } { \mathrm { d } \pi } ( y ) = h _ { N } ^ { U } ( w ( y ) ) , \quad \frac { \mathrm { d } P _ { N } ^ { \mathrm { S I R } } } { \mathrm { d } \pi } ( y ) = h _ { N } ^ { \mathrm { S I R } } ( w ( y ) ) .
$$

For $X \sim \pi$ , both $h _ { N } ^ { U } ( w ( X ) )$ and $h _ { N } ^ { \mathrm { S I R } } ( w ( X ) )$ are positive, finite, and have mean one. We compare the amount by which each exceeds a level $c > 0$ . Define $D _ { c } : = \{ y : h _ { N } ^ { U } ( w ( y ) ) > c \}$ . Since $h _ { N } ^ { U }$ is nonincreasing, $D _ { c }$ has the form $\{ w < t \}$ or $\{ w \leq t \}$ for some threshold t, or is an empty or full set. The threshold comparison in Lemma 7, with complements when necessary, gives $P _ { N } ^ { U } ( D _ { c } ) \leq P _ { N } ^ { \mathrm { S I R } } ( D _ { c } )$ . Consequently,

$$
\begin{array} { r } { \mathbb { E } [ ( h _ { N } ^ { U } ( w ( X ) ) - c ) _ { + } ] \stackrel { \mathrm { ( i ) } } { = } P _ { N } ^ { U } ( D _ { c } ) - c \pi ( D _ { c } ) } \end{array}\tag{46}
$$

$$
\stackrel { \mathrm { ( i i ) } } { \leq } P _ { N } ^ { \mathrm { S I R } } ( D _ { c } ) - c \pi ( D _ { c } )\tag{47}
$$

$$
\begin{array} { l } { \displaystyle \stackrel { \mathrm { ( i i i ) } } { \leq } \int ( h _ { N } ^ { \mathrm { S I R } } ( w ( y ) ) - c ) _ { + } \pi ( \mathrm { d } y ) } \end{array}
$$

$$
= \mathbb { E } [ ( h _ { N } ^ { \mathrm { S I R } } ( w ( X ) ) - c ) _ { + } ] .\tag{48}
$$

Here (i) integrates the positive part over its positive set $D _ { c }$ . (ii) is the probability comparison on $D _ { c }$ For (iii), the integral of $h _ { N } ^ { \mathrm { S I R } } ( \dot { w } ( y ) ) - c \mathrm { o v e r } D _ { c }$ is at most the integral of its positive part over the whole space. $\mathbf { A } \mathbf { t } c = 1$ , the two expectations are exactly the TV errors, proving the desired result in TV distance. To pass to a convex f-divergence, first consider a convex piecewise linear function with finitely many breakpoints. It can be written as

$$
g ( t ) = a + b t + \sum _ { k = 1 } ^ { m } c _ { k } ( t - t _ { k } ) _ { + } , \quad c _ { k } \geq 0 , \quad t _ { k } > 0 .
$$

Each $c _ { k }$ is the increase in slope at a breakpoint. Because both density ratios have mean one, the affine terms have equal expectations. Applying equation 48 to each remaining term gives

$$
\begin{array} { r } { \mathbb { E } [ g ( h _ { N } ^ { U } ( w ( X ) ) ) ] \leq \mathbb { E } [ g ( h _ { N } ^ { \mathrm { S I R } } ( w ( X ) ) ) ] . } \end{array}
$$

Now let $f : ( 0 , \infty ) \to \mathbb { R }$ be convex with $f ( 1 ) = 0$ . Choose the slope b of a supporting line at 1 and put ${ \widetilde { f } } ( t ) : = f ( t ) - b ( t - 1 ) \geq 0$ . Choose a countable dense set of points in $( 0 , \infty )$ and a supporting affine function of $\widetilde { f }$ at each point. Let $g _ { m }$ be the maximum of zero and the first m such functions. Then $g _ { m }$ is nonnegative and convex piecewise linear, and $g _ { m } \uparrow \stackrel { \sim } { f }$ pointwise. For completeness, supporting slopes are bounded on every compact subinterval of $( 0 , \infty )$ by secant slopes to nearby endpoints. At points of the dense set approaching t, continuity and this bound make the supporting lines approach $\widetilde f ( t )$ , proving the claimed limit. Monotone convergence yields

$$
\mathbb { E } [ \widetilde { f } ( h _ { N } ^ { U } ( w ( X ) ) ) ] \le \mathbb { E } [ \widetilde { f } ( h _ { N } ^ { \mathrm { S I R } } ( w ( X ) ) ) ] ,
$$

including infinite values. Adding back the affine part changes neither side because both means are one. This proves equation 23. The choices $f ( t ) \stackrel { \cdot } { = } | t - 1 | \bar { / } 2$ , t log $t , - \log t ,$ and $( t - 1 ) ^ { 2 }$ give TV, forward KL, reverse KL, and $\chi ^ { 2 }$ divergence, respectively.

The target also bounds the selected weight probabilities. We additionally obtain the interpretation used in Section 4:

$$
\operatorname* { P r } ( w ( \widehat { Y } _ { \mathrm { S I R } } ) > t ) \leq \operatorname* { P r } ( w ( \widehat { Y } _ { U } ) > t ) \leq \pi ( w > t ) , \quad t > 0 .\tag{49}
$$

Lemma 7 gives the first inequality. For the second, take $A : = \{ y : w ( y ) > t \}$ and use that the density integrates to one:

$$
\begin{array} { r l } & { P _ { N } ^ { U } ( A ) - \pi ( A ) = P _ { N } ^ { U } ( A ) \pi ( A ^ { c } ) - \pi ( A ) P _ { N } ^ { U } ( A ^ { c } ) } \\ & { \qquad = \displaystyle \int _ { A } \int _ { A ^ { c } } [ h _ { N } ^ { U } ( w ( y ) ) - h _ { N } ^ { U } ( w ( z ) ) ] \pi ( \mathrm { d } z ) \pi ( \mathrm { d } y ) } \\ & { \qquad \leq 0 . } \end{array}
$$

The last inequality holds because $v ( y ) > w ( z )$ on $A \times A ^ { c }$ and $h _ { N } ^ { U }$ is nonincreasing. The same proof works for $\{ \bar { w } \geq \dot { t } \}$ . No moment assumption beyond $\mathbb { E } _ { \mu } w = 1$ is used.

## E.2 TWO SUPPORTING LEMMAS

Lemma 7 (Probability of returning a weight above a threshold). For every $t > 0$

$$
\operatorname* { P r } ( w ( \widehat { Y } _ { \mathrm { S I R } } ) > t ) \leq \operatorname* { P r } ( w ( \widehat { Y } _ { U } ) > t ) .\tag{50}
$$

The same inequality holdsfor weights at least t. Taking complements reverses the inequalitiesfor weights below a threshold.

Proof. For $N = 1$ , the claim is immediate. For $N \geq 2 .$ , fix a positive vector $w _ { 1 : N }$ and write $z _ { i } : = w _ { i } / w _ { \mathrm { m a x } }$ , where $w _ { \mathrm { m a x } } : = \operatorname* { m a x } _ { j } w _ { j }$ . Let $p _ { i } ^ { U }$ and $p _ { i } ^ { \operatorname { S I R } }$ be the selection probabilities on this fixed vector. We first compare candidates, then sum over those above t. A maximal weight candidate has score strictly above $w _ { \mathrm { m a x } }$ . Candidate i can therefore be selected only if $U _ { i } < z _ { i }$ . Write $U _ { i } = z _ { i } u$ For $u \in ( 0 , 1 )$ , its score is larger than candidate $j ^ { \prime } { \bf s }$ exactly when $U _ { j } > z _ { j } u$ . Independence gives

$$
p _ { i } ^ { U } = z _ { i } \int _ { 0 } ^ { 1 } \prod _ { j \neq i } ( 1 - z _ { j } u ) \mathrm { d } u .\tag{51}
$$

For distinct $i , k$ with $w _ { i } \geq w _ { k }$ , put $\begin{array} { r } { A _ { i k } ( u ) : = \prod _ { j \not \in \{ i , k \} } ( 1 - z _ { j } u ) } \end{array}$ . Subtracting the two formulas gives

$$
\frac { p _ { i } ^ { U } } { z _ { i } } - \frac { p _ { k } ^ { U } } { z _ { k } } = ( z _ { i } - z _ { k } ) \int _ { 0 } ^ { 1 } u A _ { i k } ( u ) \mathrm { d } u \geq 0 .
$$

Since $p _ { i } ^ { \mathrm { S I R } } = z _ { i } / \sum _ { j } z _ { j }$ , this is equivalent to

$$
p _ { i } ^ { U } p _ { k } ^ { \mathrm { S I R } } - p _ { i } ^ { \mathrm { S I R } } p _ { k } ^ { U } \geq 0 \quad \mathrm { w h e n e v e r } w _ { i } \geq w _ { k } .\tag{52}
$$

Thus the change from SIR to UR favors larger weights in pairwise comparisons. Let $I _ { t } : = \{ i : w _ { i } >$ $t \}$ . Using that both probability vectors sum to one,

$$
\begin{array} { r l } & { \displaystyle \sum _ { i \in I _ { t } } ( p _ { i } ^ { U } - p _ { i } ^ { \mathrm { S I R } } ) \overset { \mathrm { ( i ) } } { = } \displaystyle \sum _ { i \in I _ { t } } \sum _ { k \notin I _ { t } } ( p _ { i } ^ { U } p _ { k } ^ { \mathrm { S I R } } - p _ { i } ^ { \mathrm { S I R } } p _ { k } ^ { U } ) } \\ & { \qquad \quad \overset { \mathrm { ( i i ) } } { \geq } 0 . } \end{array}
$$

In (i), the terms with both indices in $I _ { t }$ cancel. (ii) uses equation $5 2 ,$ since every weight in $I _ { t }$ is larger than every weight outside it. This includes empty and full sets $I _ { t } .$ Averaging over the proposals proves equation 50. Replacing $I _ { t }$ by $\{ i : w _ { i } \geq t \}$ proves the other endpoint convention. □

Lemma 8 (Density of the returned sample). Let $G ( s ) : = \operatorname* { P r } ( w ( Y ) / U \leq s )$ for independent $Y \sim \mu$ and $U \sim \mathrm { U n i f } ( 0 , \dot { 1 } )$ . Then

$$
\frac { \mathrm { d } P _ { N } ^ { U } } { \mathrm { d } \pi } ( y ) = h _ { N } ^ { U } ( w ( y ) ) , \quad h _ { N } ^ { U } ( x ) : = N \int _ { x } ^ { \infty } \frac { G ( s ) ^ { N - 1 } } { s ^ { 2 } } \mathrm { d } s , \quad x > 0 .\tag{53}
$$

The function $h _ { N } ^ { U }$ is nonincreasing.

Proof. The density formula is the integral in Lemma 4. Its integrand is nonnegative, so increasing the lower limit x cannot increase the integral. For comparison, the SIR density is

$$
\frac { \mathrm { d } P _ { N } ^ { \mathrm { S I R } } } { \mathrm { d } \pi } ( y ) = h _ { N } ^ { \mathrm { S I R } } ( w ( y ) ) , \quad h _ { N } ^ { \mathrm { S I R } } ( x ) : = N \mathbb { E } \left[ \frac { 1 } { x + \sum _ { j = 2 } ^ { N } W _ { j } } \right] ,\tag{54}
$$

where $x > 0$ and $W _ { j } \ : = \ : w ( Y _ { j } )$ for independent proposals. Indeed, a specified candidate at $y$ contributes its SIR selection probability $\begin{array} { r } { w ( y ) / ( w ( y ) + \sum _ { j = 2 } ^ { N } W _ { j } ) } \end{array}$ times $\mu ( \mathrm { d } y )$ . There are N exchangeable candidate positions, and $w ( y ) \mu ( \mathrm { d } y ) = \pi ( \mathrm { d } y )$ , giving the formula. Both scalar functions are positive and at most $\dot { N } / x$ for $x > 0 ,$ , and their compositions with w integrate to one under π. At $N = 1$ , both functions are $1 / x$ □

## E.3 EQUIVALENT BERNOULLI SELECTION RULE

UR has an equivalent rule on each fixed positive weight vector: retain candidates independently and select uniformly among those retained. Write $z _ { j } : = w _ { j } /$ max w , draw independent $B _ { i } \sim$ Bern $( z _ { j } )$ , and retain j when $B _ { j } = 1$ . If there are k retained indices, each is selected with probability $1 / k$ . At least one $z _ { j } = 1$ , so the retained set is nonempty. The denominator in $z _ { j }$ is the largest observed weight, not a bound on unobserved weights. Put $\begin{array} { r } { \dot { K } _ { - i } : = \sum _ { j \neq i } B _ { j } } \end{array}$ . By independence, the probability of selecting i is $z _ { i } \mathbb { E } [ ( 1 + K _ { - i } ) ^ { - 1 } ]$ . Using $\textstyle 1 / ( k + 1 ) = \int _ { 0 } ^ { 1 } x ^ { k } \mathrm { d } x$

$$
\begin{array} { r l } & { \mathbb { E } \left[ \displaystyle \frac { 1 } { 1 + K _ { - i } } \right] \stackrel { ( \mathrm { i } ) } { = } \displaystyle \int _ { 0 } ^ { 1 } \mathbb { E } [ x ^ { K _ { - i } } ] \mathrm { d } x } \\ & { \stackrel { \mathrm { ( i i ) } } { = } \displaystyle \int _ { 0 } ^ { 1 } \prod _ { j \ne i } ( 1 - z _ { j } + z _ { j } x ) \mathrm { d } x } \\ & { \stackrel { \mathrm { ( i i i ) } } { = } \displaystyle \int _ { 0 } ^ { 1 } \prod _ { j \ne i } ( 1 - z _ { j } t ) \mathrm { d } t . } \end{array}
$$

Here (i) interchanges a nonnegative integral and expectation. (ii) uses independence of the Bernoulli variables, and (iii) substitutes $t = 1 - x$ . After multiplication by $z _ { i }$ , this is exactly equation 51. The two implementations therefore have the same selection probabilities on every fixed batch, though they need not select the same index when coupled using the same uniforms.

## F BINARY EXAMPLES AND BIAS CALCULATIONS

Section 4 gives the direct comparison for two candidates. We first extend that binary calculation to every budget and identify the source of SIR’s remaining error. We then prove the equality with RS with $M \stackrel { \textstyle - } { = } C _ { \infty } ^ { \pi }$ for two weight values, derive the general SIR bias and UR’s corresponding adjustment, and conclude with a continuous example. Throughout the binary calculation,

$$
\mu = \left( \frac 1 2 , \frac 1 2 \right) , \qquad \pi = ( \varepsilon , 1 - \varepsilon ) , \qquad 0 < \varepsilon < \frac 1 2 ,
$$

with $w _ { L } = 2 \varepsilon$ and $w _ { H } = 2 ( 1 - \varepsilon )$

## F.1 PROOF OF PROPOSITION 3: BINARY EXAMPLE FOR EVERY BUDGET

Let $K \sim \mathrm { B i n } \left( N , \frac { 1 } { 2 } \right)$ count the candidates in state 1, and put $X : = K / N$ . Thus X is the observed fraction of candidates in state 1. Define, for $x \in [ 0 , 1 ]$

$$
d ( x ) : = \varepsilon + ( 1 - 2 \varepsilon ) x , \qquad g ( x ) : = \frac { ( 1 - \varepsilon ) x } { d ( x ) } .
$$

When $X = x$ , the total observed weight is $2 N d ( x )$ , and SIR returns state 1 with probability $g ( x )$ The exact output probabilities are

$$
\operatorname* { P r } ( \widehat { Y } _ { \mathrm { S I R } } = 1 ) = \mathbb { E } [ g ( X ) ] ,\tag{55}
$$

$$
\mathrm { P r } ( \widehat { Y } _ { U } = 1 ) = ( 1 - \varepsilon ) \left[ 1 - \left( \frac { 1 - 2 \varepsilon } { 2 ( 1 - \varepsilon ) } \right) ^ { N } \right] .\tag{56}
$$

Although $g ( \mathbb { E } [ X ] ) = 1 - \varepsilon$ , averaging $g ( X )$ does not give this target probability. More precisely,

$$
\pi ( 1 ) - \operatorname* { P r } ( \widehat { Y } _ { \mathrm { S I R } } = 1 ) = 4 \varepsilon ( 1 - \varepsilon ) ( 1 - 2 \varepsilon ) \mathbb { E } \left[ \frac { \left( X - \frac { 1 } { 2 } \right) ^ { 2 } } { d ( X ) } \right] .\tag{57}
$$

On a binary state space, TV is the absolute error in the probability of returning state 1. Thus these formulae will give the claimed TV rates once we verify that both probabilities are below $1 - \varepsilon .$ . We first establish the output formulae by conditioning on $\dot { K }$ , then obtain the sign and size of SIR’s error from the gap between g and its tangent at $\mathbb { E } [ X ] { \stackrel { - } { = } } 1 / 2$

The two output distributions. It suffices to compute each selection probability conditional on K, since averaging over this binomial count gives the marginal output probability. For SIR, given $K = k ,$ , the probability is

$$
\frac { ( 1 - \varepsilon ) k } { ( 1 - \varepsilon ) k + \varepsilon ( N - k ) } = g \left( \frac { k } { N } \right) .
$$

Averaging over $K$ proves equation 55. For UR and $k \geq 1$ , fix a state 1 candidate and condition on its uniform variable being u. It beats another state 1 candidate when that candidate’s uniform exceeds $u ,$ and a state 0 candidate when its uniform exceeds $\varepsilon u / ( 1 - \varepsilon )$ ). The uniforms are independent, and ties have probability zero. Summing over the k possible selected state 1 candidates gives

$$
\operatorname* { P r } ( \widehat { Y } _ { U } = 1 \mid K = k ) = k \int _ { 0 } ^ { 1 } ( 1 - u ) ^ { k - 1 } \left( 1 - \frac { \varepsilon } { 1 - \varepsilon } u \right) ^ { N - k } \mathrm { d } u .
$$

The probability is zero when $k = 0$ . Therefore,

$$
\begin{array} { l } { \displaystyle \mathrm { P r } ( \widehat { Y } _ { U } = 1 ) = \displaystyle \frac { 1 } { 2 ^ { N } } \int _ { 0 } ^ { 1 } \sum _ { k = 1 } ^ { N } \binom { N } { k } k ( 1 - u ) ^ { k - 1 } \left( 1 - \displaystyle \frac { \varepsilon } { 1 - \varepsilon } u \right) ^ { N - k } \mathrm { d } u } \\ { \displaystyle \stackrel { \mathrm { ( i ) } } { = } \displaystyle \frac { N } { 2 } \int _ { 0 } ^ { 1 } \left( 1 - \displaystyle \frac { u } { 2 ( 1 - \varepsilon ) } \right) ^ { N - 1 } \mathrm { d } u } \\ { \displaystyle \stackrel { \mathrm { ( i i ) } } { = } ( 1 - \varepsilon ) \left[ 1 - \left( \displaystyle \frac { 1 - 2 \varepsilon } { 2 ( 1 - \varepsilon ) } \right) ^ { N } \right] , } \end{array}\tag{58}
$$

(59)

where (i) uses $\begin{array} { r } { k \binom { N } { k } = N \binom { N - 1 } { k - 1 } } \end{array}$ and the binomial theorem; and (ii) evaluates the integral, proving equation 56 directly from the selection rule.

The gap and its rate. To prove equation $^ { 5 7 , }$ , it is enough to express $g ( m ) - g ( X )$ as a centered linear term plus a nonnegative remainder, where $m : = 1 / 2$ . The linear term will disappear on taking expectations because $\mathbb { E } [ { \bf \bar { X } } - m ] = 0$ . Since $g ( m ) = 1 - \varepsilon = \pi ( 1 )$ and

$$
g ^ { \prime \prime } ( x ) = - \frac { 2 \varepsilon ( 1 - \varepsilon ) ( 1 - 2 \varepsilon ) } { d ( x ) ^ { 3 } } < 0 ,
$$

$g$ lies below its tangent at m. The exact distance from that tangent is

$$
g ( m ) + g ^ { \prime } ( m ) ( x - m ) - g ( x ) = \frac { 4 \varepsilon ( 1 - \varepsilon ) ( 1 - 2 \varepsilon ) ( x - m ) ^ { 2 } } { d ( x ) } .
$$

Taking expectations therefore proves equation 57. The formula is positive for every finite $N _ { \ast }$ , and attributes the gap to fluctuations in the candidate counts, not only to batches that miss a state. The bounds $\varepsilon \le { \bar { d ( X ) } } \le 1 - \varepsilon$ and $\mathbb { E } [ ( X - m ) ^ { 2 } ] = 1 / ( 4 N )$ give

$$
\frac { \varepsilon ( 1 - 2 \varepsilon ) } { N } \leq \pi ( 1 ) - \operatorname* { P r } ( \widehat { Y } _ { \mathrm { S I R } } = 1 ) \leq \frac { ( 1 - \varepsilon ) ( 1 - 2 \varepsilon ) } { N } .
$$

For the leading coefficient, it suffices to keep the quadratic term in Taylor’s formula and bound its expected remainder by $O _ { \varepsilon } ( N ^ { - 3 / 2 } )$ . Here $g ^ { \prime \prime } ( m ) = - 1 6 \varepsilon ( 1 - \varepsilon ) ( 1 - 2 \varepsilon )$ . For fixed $\varepsilon \in ( 0 , 1 / 2 )$ the third derivative of $g$ is bounded on $[ 0 , 1 ]$ , because $d ( x ) \geq \varepsilon$ . Taylor’s formula gives

$$
\begin{array} { l } { \displaystyle \mathbb { E } [ g ( X ) ] = g ( m ) + g ^ { \prime } ( m ) \mathbb { E } [ X - m ] + \frac { g ^ { \prime \prime } ( m ) } { 2 } \mathbb { E } [ ( X - m ) ^ { 2 } ] + O _ { \varepsilon } \left( \mathbb { E } [ | X - m | ^ { 3 } ] \right) } \\ { \displaystyle \qquad = g ( m ) + \frac { g ^ { \prime \prime } ( m ) } { 8 N } + O _ { \varepsilon } ( N ^ { - 3 / 2 } ) . } \end{array}
$$

Here the remainder follows from

$$
\mathbb { E } [ | X - m | ^ { 3 } ] \le \Big ( \mathbb { E } [ ( X - m ) ^ { 4 } ] \Big ) ^ { 3 / 4 } = O ( N ^ { - 3 / 2 } ) , \qquad \mathbb { E } [ ( X - m ) ^ { 4 } ] = \frac { 3 } { 1 6 N ^ { 2 } } - \frac { 1 } { 8 N ^ { 3 } } .
$$

Consequently,

$$
\pi ( 1 ) - \mathrm { P r } ( \widehat { Y } _ { \mathrm { S I R } } = 1 ) = \frac { 2 \varepsilon ( 1 - \varepsilon ) ( 1 - 2 \varepsilon ) } { N } + O _ { \varepsilon } ( N ^ { - 3 / 2 } ) .
$$

Both samplers assign state 1 less than its target probability. On a binary space, each TV error equals this gap, proving Proposition 3. The leading SIR coefficient is strictly positive for every fixed $0 < \varepsilon < 1 / 2$ . The remainder constants may depend on $\varepsilon ;$ no uniform limit as ε approaches an endpoint is claimed.

## F.2 EQUALITY WITH RS AT $M = C _ { \infty } ^ { \pi }$ FOR TWO WEIGHT VALUES

The equality with RS at $M = C _ { \infty } ^ { \pi }$ extends beyond two states: the relevant restriction is that the weights take at most two values. We retain a specific fallback rule for RS: if all candidates are rejected, each of the N observed candidate indices is selected with probability $1 / N$ . This is uniform selection among indices, not among distinct states.

Proposition 6 (Two possible weight values). Suppose $C _ { \infty } ^ { \pi } <$ ∞ and $w ( Y ) \in \{ a , C _ { \infty } ^ { \pi } \}$ almost surely under $\mu ,$ where $0 \leq a < C _ { \infty } ^ { \pi }$ . Let RS use threshold $M = C _ { \infty } ^ { \pi }$ and the uniform fallback above. If all sampled weights vanish, let UR also select each candidate index with probability $1 / N$ . Then, for everyfixed $N \geq 1$

$$
\begin{array} { r } { P _ { N } ^ { U } = P _ { N , C _ { \infty } ^ { \pi } } ^ { \mathrm { R S } } . } \end{array}
$$

Proof. It suffices to compare the probability of selecting each candidate on a fixed list, after uniformly randomizing the order in which RS processes it. The randomization preserves the RS marginal law because the proposals are independent and identically distributed; UR already treats candidate order symmetrically. There are two cases to check: whether the list contains a weight equal to the global maximum $C _ { \infty } ^ { \dot { \pi } }$

First express the order-averaged RS rule in a useful form. Conditional on the list, assign independent acceptance indicators with probabilities $w _ { i } / C _ { \infty } ^ { \pi } ,$ , independently of the random order. Given a nonempty accepted set, its first index in that order is uniform over the set. Thus this RS rule selects uniformly among accepted indices, or uses the specified fallback if none is accepted.

If a weight $C _ { \infty } ^ { \pi }$ is present, it is also the largest observed weight. The acceptance probabilities are exactly those in UR’s Bernoulli representation from Appendix E.3. At least one index is accepted with probability one. The retained set has the same distribution in the two Bernoulli representations, and both select uniformly from it, so their conditional selection probabilities agree. Zero-weight indices, if present, are never accepted or selected and can be removed when applying that representation.

If no weight $C _ { \infty } ^ { \pi }$ is present, every observed weight is a. UR is uniform by symmetry, including the specified convention when $a = 0$ . Order-averaged RS is also symmetric over these indices, both on acceptance and on its fallback, so every index has probability $1 / N$ . These cases cover every list under the two-value assumption. Averaging the equal conditional probabilities over the candidates proves the claimed equality of marginal laws. □

The two-value restriction makes a batch without a maximal-weight candidate have equal weights throughout. With more weight values, such a batch can still contain unequal weights, so this second case of the proof no longer applies.

## F.3 LEADING SIR BIAS

The binary calculation attributes SIR’s error to random counts and normalization. To examine the same effect beyond state probabilities, let $\psi$ be a bounded measurable function of the returned sample, and write $\begin{array} { r } { P ( \dot { \psi } ) : = \int \psi \dot { \mathrm { d } } \dot { P } } \end{array}$ for its expectation under $P .$ For example, choosing $\psi = \mathbb { 1 } _ { D }$ makes $\bar { P ( \psi ) }$ the probability of an event $D ;$ taking all bounded $\psi$ will later recover the TV error. Throughout this subsection and the next subsection, assume $0 < w ( Y ) \leq C < \infty$ µ-almost surely for a fixed C. No positive lower bound on w is required. Put $a : = \pi ( \psi ) , b : = \| \psi \| _ { \infty } .$ , and

$$
A _ { \psi } : = \mathbb { E } _ { \pi } [ ( w - C ^ { \pi } ) \psi ] .
$$

No monotonicity of $\psi$ is assumed. We prove

$$
P _ { N } ^ { \mathrm { S I R } } ( \psi ) - \pi ( \psi ) = - \frac { A _ { \psi } } { N } + O _ { C } ( b N ^ { - 3 / 2 } ) .\tag{60}
$$

The coefficient is signed and may vanish for a particular ψ. The TV expansion below instead considers all bounded measurable functions.

Reduction to two estimates. We will separate the bias into an exactly computable leading term and a remainder uniform over bounded ψ. Write $W _ { i } : = w ( Y _ { i } )$ and define

$$
\overline { { { W } } } _ { N } : = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } W _ { i } , \qquad Z _ { N } : = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } W _ { i } ( \psi ( Y _ { i } ) - a ) , \qquad D _ { N } : = \overline { { { W } } } _ { N } - 1 .
$$

Here $Z _ { N }$ is a centered weighted fluctuation, not an estimate of the normalizing constant $Z .$ Conditional on the candidates, SIR’s expected value of ψ is their weighted average. Subtracting a gives

$$
{ \frac { \sum _ { i = 1 } ^ { N } W _ { i } \psi ( Y _ { i } ) } { \sum _ { i = 1 } ^ { N } W _ { i } } } - a = { \frac { Z _ { N } } { \overline { { W } } _ { N } } } .
$$

Although $\mathbb { E } [ Z _ { N } ] = 0$ , dividing by the dependent random denominator need not preserve that zero mean. The identity

$$
\frac { 1 } { 1 + D _ { N } } = 1 - D _ { N } + \frac { D _ { N } ^ { 2 } } { 1 + D _ { N } }
$$

separates the leading interaction with the denominator from a remainder:

$$
P _ { N } ^ { \mathrm { S I R } } ( \psi ) - a = - \mathbb { E } [ Z _ { N } D _ { N } ] + \mathbb { E } \left[ \frac { Z _ { N } D _ { N } ^ { 2 } } { \overline { { W } } _ { N } } \right] .\tag{61}
$$

Thus it suffices to establish

$$
\mathbb { E } [ Z _ { N } D _ { N } ] = \frac { A _ { \psi } } { N } , \qquad \mathbb { E } [  [ \frac { Z _ { N } D _ { N } ^ { 2 } } { \overline { { W } } _ { N } } | ] = O _ { C } ( b N ^ { - 3 / 2 } ) .
$$

The second estimate controls the signed remainder as well, and its uniform dependence on b will justify taking the TV supremum.

The leading coefficient. Let $Y \sim \mu$ and $W : = w ( Y )$ . Both $W ( \psi ( Y ) - a )$ and $W - 1$ have mean zero, so cross terms from distinct samples vanish. Consequently,

$$
\begin{array} { l } { \displaystyle \mathbb { E } [ Z _ { N } D _ { N } ] \stackrel { \mathrm { ( i ) } } { = } \frac { 1 } { N } \mathbb { E } _ { \mu } [ W ( \psi ( Y ) - a ) ( W - 1 ) ] } \\ { \displaystyle \stackrel { \mathrm { ( i i ) } } { = } \frac { 1 } { N } \mathbb { E } _ { \mu } [ W ^ { 2 } ( \psi ( Y ) - a ) ] } \\ { \displaystyle \stackrel { \mathrm { ( i i i ) } } { = } \frac { 1 } { N } \mathbb { E } _ { \pi } [ ( w - C ^ { \pi } ) \psi ] } \\ { \displaystyle = \frac { A _ { \psi } } { N } . } \end{array}\tag{62}
$$

(63)

Here (i) uses independence and centering; (ii) uses $\mathbb { E } _ { \mu } [ W ( \psi ( Y ) - a ) ] = 0 \mathrm { . }$ ; and (iii) uses $w \mathrm { d } \mu = \mathrm { d } \pi$ $\mathbb { E } _ { \pi } [ w ] = C ^ { \pi }$ , and $a = \pi ( \psi )$ . Consequently, the leading bias in equation 61 comes from the dependence between the centered weighted sum and the total observed weight.

The remainder without a lower weight bound. We cannot bound $1 / \overline { { W } } _ { N }$ uniformly, because weights may approach zero. The required estimate will instead follow from two centered moments and a lower-tail probability. Set $B _ { N } \mathrm { ' } { : = } \lbrace \overline { { W } } _ { N } \geq 1 / 2 \rbrace$ . We first reduce to those quantities:

$$
\begin{array} { r l } & { \mathbb { E } \left[ \left| \frac { Z _ { N } D _ { N } ^ { 2 } } { \overline { { W } } _ { N } } \right| \right] \overset { \mathrm { ( i ) } } { \leq } 2 \mathbb { E } [ | Z _ { N } | D _ { N } ^ { 2 } ] + 2 b \operatorname* { P r } ( B _ { N } ^ { c } ) } \\ & { \qquad \overset { \mathrm { ( i i ) } } { \leq } 2 \left( \mathbb { E } [ Z _ { N } ^ { 2 } ] \right) ^ { 1 / 2 } \left( \mathbb { E } [ D _ { N } ^ { 4 } ] \right) ^ { 1 / 2 } + 2 b \operatorname* { P r } ( B _ { N } ^ { c } ) . } \end{array}
$$

Here (i) follows since $1 / \overline { { W } } _ { N }  \leq 2$ on $B _ { N }$ . On $B _ { N } ^ { c }$ , use the whole ratio instead: $Z _ { N } / \overline { { W } } _ { N }$ is a weighted average of ψ minus $^ { a , }$ so its absolute value is at most $2 b ;$ also $| D _ { N } | \leq 1$ there. (ii) follows from Cauchy–Schwarz inequality.

It remains to bound the three quantities on the right. Independence and centering give

$$
\mathbb { E } [ Z _ { N } ^ { 2 } ] = \frac { 1 } { N } \mathbb { E } _ { \mu } [ W ^ { 2 } ( \psi ( Y ) - a ) ^ { 2 } ] \leq \frac { 4 C b ^ { 2 } } { N } .
$$

In the fourth moment of the centered sum, only four equal indices or two equal pairs survive, so

$$
\begin{array} { l } { { \displaystyle { \mathbb E } [ D _ { N } ^ { 4 } ] = \frac { N { \mathbb E } _ { \mu } [ ( W - 1 ) ^ { 4 } ] + 3 N ( N - 1 ) \left( { \mathbb E } _ { \mu } [ ( W - 1 ) ^ { 2 } ] \right) ^ { 2 } } { N ^ { 4 } } } } \\ { { \displaystyle \quad = O _ { C } ( N ^ { - 2 } ) . } } \end{array}
$$

Finally, since $0 < W \leq C$ and $\mathbb { E } _ { \mu } [ W ] = 1$ , Hoeffding’s inequality yields

$$
\mathrm { P r } ( B _ { N } ^ { c } ) \leq \exp \left( - \frac { N } { 2 C ^ { 2 } } \right) .
$$

Substituting these estimates into the displayed bound gives $O _ { C } ( b N ^ { - 3 / 2 } )$ . Together with the leading coefficient, this proves equation 60 without an inverse moment assumption.

From expectation bias to TV error. The remainder is uniform over $\| \psi \| _ { \infty } \leq 1$ , which is essential when taking the supremum that defines TV. Hence

$$
\begin{array} { r l } & { D _ { \mathrm { T V } } ( P _ { N } ^ { \mathrm { S I R } } , \pi ) = \displaystyle \frac { 1 } { 2 } \operatorname* { s u p } _ { \| \psi \| _ { \infty } \le 1 } \vert P _ { N } ^ { \mathrm { S I R } } ( \psi ) - \pi ( \psi ) \vert } \\ & { \qquad \stackrel { \mathrm { ( i ) } } { = } \displaystyle \frac { 1 } { 2 N } \operatorname* { s u p } _ { \| \psi \| _ { \infty } \le 1 } \vert A _ { \psi } \vert + O _ { C } ( N ^ { - 3 / 2 } ) } \\ & { \qquad \stackrel { \mathrm { ( i i ) } } { = } \displaystyle \frac { 1 } { 2 N } \mathbb { E } _ { \pi } [ \vert w - C ^ { \pi } \vert ] + O _ { C } ( N ^ { - 3 / 2 } ) . } \end{array}\tag{64}
$$

Here (i) uses the uniform remainder in equation 60. For (ii), choose $\psi ( y ) = \mathrm { s i g n } ( w ( y ) - C ^ { \pi } )$ to attain the supremum. The coefficient vanishes exactly when $w = C ^ { \pi }$ π-almost surely. Because the weights are positive, π and $\mu$ have the same null sets; hence w is then constant $\mu \cdot$ -almost surely. The identity $\mathbb { E } _ { \mu } [ w ] = 1$ forces that constant to be one, so the coefficient vanishes exactly when $\pi = \mu$ The leading term in equation 60 is the classical asymptotic bias term for self-normalized importance sampling; see, for example, Deligiannidis et al. $( 2 0 2 6 $ , Theorem 2.1). The proof here additionally supplies the displayed remainder uniformly over bounded measurable functions $\psi$ under a finite upper weight bound, without requiring a positive lower weight bound or an inverse moment assumption.

Combining equation 64 with Corollary 1, every fixed nontrivial pair with bounded positive weights therefore has SIR error of order $1 / N$ , whereas UR’s TV error decays exponentially.

## F.4 HOW UR CANCELS THE LEADING SIR BIAS

With the definitions and assumptions above, we prove

$$
P _ { N } ^ { U } ( \psi ) - P _ { N } ^ { \mathrm { S I R } } ( \psi ) = \frac { A _ { \psi } } { N } + o ( N ^ { - 1 } ) .\tag{65}
$$

The marginal coefficient also follows from equation 60 and the exponential UR error bound. Here we derive it from the choices within a fixed batch, to explain how UR changes the SIR probabilities before averaging over proposals. Let $p _ { i } ^ { U }$ and $p _ { i } ^ { \mathrm { S I R } }$ denote the selection probabilities conditional on the realized candidates, and define

$$
\Delta _ { N } ( \psi ) : = \sum _ { i = 1 } ^ { N } ( p _ { i } ^ { U } - p _ { i } ^ { \mathrm { S I R } } ) \psi ( Y _ { i } ) .
$$

Its expectation is the left side of equation 65. Write $\begin{array} { r } { W _ { \Sigma } : = \sum _ { i = 1 } ^ { N } W _ { i } } \end{array}$ and $\begin{array} { r } { Q _ { \Sigma } : = \sum _ { i = 1 } ^ { N } W _ { i } ^ { 2 } } \end{array}$ The event $B _ { N }$ from the preceding subsection is equivalently $\{ W _ { \Sigma } \ \tilde { \geq } \ N / 2 \}$ . We first show how a batchwise expansion implies equation 65, then reduce that expansion to a candidate-wise estimate and prove the remaining integral approximation.

Proof from a batchwise expansion. We will establish, on $B _ { N }$ and for sufficiently large N depending only on C,

$$
\Delta _ { N } ( \psi ) = \frac { \sum _ { i } W _ { i } ^ { 2 } \psi ( Y _ { i } ) } { W _ { \Sigma } ^ { 2 } } - \frac { Q _ { \Sigma } \sum _ { i } W _ { i } \psi ( Y _ { i } ) } { W _ { \Sigma } ^ { 3 } } + O _ { C } ( b N ^ { - 2 } ) .\tag{66}
$$

To see why this is sufficient, split the expected correction according to $B _ { N }$

$$
N \mathbb { E } [ \Delta _ { N } ( \psi ) ] = \mathbb { E } [ N \Delta _ { N } ( \psi ) \mathbb { 1 } _ { B _ { N } } ] + \mathbb { E } [ N \Delta _ { N } ( \psi ) \mathbb { 1 } _ { B _ { N } ^ { c } } ] .
$$

It is enough for the first term to converge to $A _ { \psi }$ and the second to vanish. Assuming equation 66, the strong law gives

$$
\begin{array} { r l } & { N \Delta _ { N } ( \psi ) \mathbb { 1 } _ { B _ { N } } = \left[ \frac { N ^ { - 1 } \sum _ { i } W _ { i } ^ { 2 } \psi ( Y _ { i } ) } { ( W _ { \Sigma } / N ) ^ { 2 } } - \frac { ( Q _ { \Sigma } / N ) ( N ^ { - 1 } \sum _ { i } W _ { i } \psi ( Y _ { i } ) ) } { ( W _ { \Sigma } / N ) ^ { 3 } } \right] \mathbb { 1 } _ { B _ { N } } + O _ { C } ( b / N ) } \\ & { \qquad \to \mathbb { E } _ { \pi } [ w \psi ] - C ^ { \pi } \pi ( \psi ) } \\ & { \qquad = A _ { \psi } \qquad \mathrm { a l m o s t ~ s u r e l y } . } \end{array}
$$

Here $W _ { \Sigma } / N \to 1 , Q _ { \Sigma } / N \to C ^ { \pi }$ , and the two weighted sample averages converge to $\mathbb { E } _ { \pi } [ w \psi ]$ and $\pi ( \psi ) ;$ ; in particular, $B _ { N }$ occurs eventually almost surely. Each of the two leading terms, restricted to $B _ { N }$ , is at most 2Cb in absolute value. In particular, ${ \cal Q } _ { \Sigma } \leq C W _ { \Sigma } , | \sum _ { i } \bar { W _ { i } } \psi ( Y _ { i } ) | \leq b W _ { \Sigma }$ and $W _ { \Sigma } \geq N / 2$ . The remainder is uniform over batches in $B _ { N }$ , so dominated convergence gives $\mathbb { E } [ N \Delta _ { N } ( \psi ) \mathbb { 1 } _ { B _ { N } } ]  A _ { \psi }$ . On the complement, $| \Delta _ { N } ( \psi ) | \leq 2 b$ and the preceding Hoeffding bound give

$$
N \mathbb { E } [ | \Delta _ { N } ( \psi ) | \mathbb { 1 } _ { B _ { N } ^ { c } } ] \leq 2 b N \exp ( - \frac { N } { 2 C ^ { 2 } } )  0 .
$$

Hence the batchwise expansion implies equation 65. We now prove that expansion.

A sufficient candidate-wise estimate. Because $\Delta _ { N } ( \psi )$ is a sum over $N$ candidates, it suffices to prove the following expansion uniformly in i on $B _ { N }$

$$
p _ { i } ^ { U } - p _ { i } ^ { \mathrm { S I R } } = \frac { W _ { i } } { W _ { \Sigma } ^ { 2 } } \left( W _ { i } - \frac { Q _ { \Sigma } } { W _ { \Sigma } } \right) + O _ { C } ( N ^ { - 3 } ) .\tag{67}
$$

Multiplying by $\psi ( Y _ { i } )$ and summing gives the two leading terms in equation $^ { 6 6 , }$ with total remainder $O _ { C } ( b \overrightharpoon { N } ^ { - 2 } )$ . The reference value $\begin{array} { r } { Q _ { \Sigma } ^ { \bf { \breve { \Phi } } } / W _ { \Sigma } = \sum _ { i } p _ { i } ^ { \mathrm { S I R } } W _ { i } } \end{array}$ is the SIR-weighted average of the observed weights. The leading correction therefore increases the probabilities of candidates above this average and decreases those below it. This describes the leading term, not the exact sign of every finite-budget difference. Its total mass is zero:

$$
\sum _ { i } { \frac { W _ { i } } { W _ { \Sigma } ^ { 2 } } } \left( W _ { i } - { \frac { Q _ { \Sigma } } { W _ { \Sigma } } } \right) = { \frac { Q _ { \Sigma } } { W _ { \Sigma } ^ { 2 } } } - { \frac { Q _ { \Sigma } } { W _ { \Sigma } ^ { 2 } } } = 0 .
$$

We next derive this estimate from the exact selection probabilities.

Reduction to the selection integral. Fix a positive batch and set

$$
z _ { j } : = \frac { W _ { j } } { \operatorname* { m a x } _ { k } W _ { k } } , \qquad \lambda _ { i } : = \sum _ { j \neq i } z _ { j } , \qquad s _ { 2 , i } : = \sum _ { j \neq i } z _ { j } ^ { 2 } .
$$

The exact selection law equation 51 gives

$$
\frac { p _ { i } ^ { U } } { z _ { i } } = \int _ { 0 } ^ { 1 } \prod _ { j \neq i } ( 1 - z _ { j } t ) \mathrm { d } t .
$$

We will prove, uniformly for $\lambda _ { i } \geq 1$

$$
\frac { p _ { i } ^ { U } } { z _ { i } } = \frac { 1 } { \lambda _ { i } } - \frac { s _ { 2 , i } } { \lambda _ { i } ^ { 3 } } + { \cal O } ( \lambda _ { i } ^ { - 3 } ) .\tag{68}
$$

The remainder constant is independent of the batch. The second term cannot be absorbed into the remainder: its numerator $s _ { 2 , i }$ can grow with $\lambda _ { i } ,$ although $s _ { 2 , i } \leq \lambda _ { i }$

First use this integral approximation to obtain equation $^ { 6 7 . }$ . Since $0 < z _ { i } \le 1$

$$
p _ { i } ^ { \mathrm { S I R } } = \frac { z _ { i } } { \lambda _ { i } + z _ { i } } = \frac { z _ { i } } { \lambda _ { i } } - \frac { z _ { i } ^ { 2 } } { \lambda _ { i } ^ { 2 } } + O \left( \frac { z _ { i } } { \lambda _ { i } ^ { 3 } } \right) .
$$

Multiplying equation 68 by $z _ { i }$ and subtracting this expansion yields

$$
p _ { i } ^ { U } - p _ { i } ^ { \mathrm { S I R } } = \frac { z _ { i } } { \lambda _ { i } ^ { 2 } } \left( z _ { i } - \frac { s _ { 2 , i } } { \lambda _ { i } } \right) + O \left( \frac { z _ { i } } { \lambda _ { i } ^ { 3 } } \right) .\tag{69}
$$

On $B _ { N } , W _ { i } \leq C$ and $W _ { \Sigma } \geq N / 2$ imply

$$
\lambda _ { i } \geq \frac { N / 2 - C } { C } \geq \frac { N } { 4 C } \qquad \mathrm { w h e n ~ } N \geq 4 C .
$$

Thus $\lambda _ { i } \geq 1$ and the remainder is uniformly $O _ { C } ( N ^ { - 3 } )$ . In the original weights, the leading expression is

$$
\frac { W _ { i } ^ { 2 } } { ( W _ { \Sigma } - W _ { i } ) ^ { 2 } } - \frac { W _ { i } ( Q _ { \Sigma } - W _ { i } ^ { 2 } ) } { ( W _ { \Sigma } - W _ { i } ) ^ { 3 } } .
$$

Because $W _ { i } \leq C , Q _ { \Sigma } \leq C W _ { \Sigma } .$ , and $W _ { \Sigma } \mathrm { ~ - ~ } W _ { i } \geq \mathrm { ~ } N / 4$ , replacing $W _ { \Sigma } \mathrm { ~ - ~ } W _ { i }$ $W _ { \Sigma }$ in the denominators changes this expression by $O _ { C } ( N ^ { - 3 } )$ , uniformly in i. The term with numerator $W _ { i } ^ { 3 }$ is also $O _ { C } ( N ^ { - 3 } )$ . This proves equation $^ { 6 7 }$ , and hence equation 65, once the integral approximation is verified.

Proof of the integral approximation. The product in the selection integral is bounded by $e ^ { - \lambda _ { i } t }$ so its mass concentrates near $t = 0$ as $\lambda _ { i }$ grows. The approximation in equation 68 will follow by replacing that product with $e ^ { - \lambda _ { i } t } ( 1 - s _ { 2 , i } \overline { { t } } ^ { 2 } / 2 )$ , since

$$
\int _ { 0 } ^ { \infty } e ^ { - \lambda _ { i } t } \left( 1 - { \frac { s _ { 2 , i } t ^ { 2 } } { 2 } } \right) \mathrm { d } t = { \frac { 1 } { \lambda _ { i } } } - { \frac { s _ { 2 , i } } { \lambda _ { i } ^ { 3 } } } .
$$

To justify the replacement, it suffices to bound the difference on $[ 0 , 1 / 2 ]$ and both tails starting at $1 / 2$ by ${ \bar { O } } ( \lambda _ { i } ^ { - 3 } )$ .

For the local difference, we claim that, uniformly on $0 \leq t \leq 1 / 2$

$$
\prod _ { j \neq i } ( 1 - z _ { j } t ) = e ^ { - \lambda _ { i } t } \left[ 1 - \frac { s _ { 2 , i } t ^ { 2 } } { 2 } + O ( \lambda _ { i } t ^ { 3 } + \lambda _ { i } ^ { 2 } t ^ { 4 } ) \right] .
$$

Its integrated error is at most a constant times

$$
\int _ { 0 } ^ { 1 / 2 } e ^ { - \lambda _ { i } t } ( \lambda _ { i } t ^ { 3 } + \lambda _ { i } ^ { 2 } t ^ { 4 } ) \mathrm { d } t \leq \lambda _ { i } \frac { 3 ! } { \lambda _ { i } ^ { 4 } } + \lambda _ { i } ^ { 2 } \frac { 4 ! } { \lambda _ { i } ^ { 5 } } = { \cal O } ( \lambda _ { i } ^ { - 3 } ) ,
$$

using $\textstyle \int _ { 0 } ^ { \infty } t ^ { k } e ^ { - \lambda _ { i } t } \mathrm { d } t = k ! / \lambda _ { i } ^ { k + 1 }$ . The original product’s tail is at most $e ^ { - \lambda _ { i } / 2 } / \lambda _ { i }$ . The absolute tail of the approximating function is at most

$$
\int _ { 1 / 2 } ^ { \infty } e ^ { - \lambda _ { i } t } \left( 1 + \frac { \lambda _ { i } t ^ { 2 } } { 2 } \right) \mathrm { d } t ,
$$

since $s _ { 2 , i } ~ \leq ~ \lambda _ { i }$ . Both tails are uniformly $O ( \lambda _ { i } ^ { - 3 } )$ for $\lambda _ { i } ~ \geq ~ 1$ . Thus only the claimed local approximation remains to be checked.

The logarithmic series gives

$$
\prod _ { j \neq i } ( 1 - z _ { j } t ) = e ^ { - \lambda _ { i } t } \exp \left( - { \frac { s _ { 2 , i } t ^ { 2 } } { 2 } } - R _ { i } ( t ) \right) , \qquad 0 \leq R _ { i } ( t ) \leq { \frac { 2 } { 3 } } \lambda _ { i } t ^ { 3 } .
$$

Specifically, $\textstyle \sum _ { k > 3 } u ^ { k } / k \leq 2 u ^ { 3 } / 3$ for $0 \leq u \leq 1 / 2$ , and $\textstyle \sum _ { j \neq i } z _ { j } ^ { 3 } \leq \lambda _ { i }$ . For $u , v \geq 0 .$

$$
\left| e ^ { - u - v } - 1 + u \right| \leq \left| e ^ { - u } - 1 + u \right| + 1 - e ^ { - v } \leq \frac { u ^ { 2 } } { 2 } + v .
$$

Apply this with $u = s _ { 2 , i } t ^ { 2 } / 2$ and $v = R _ { i } ( t )$ . Since $s _ { 2 , i } \leq \lambda _ { i }$ , this gives the claimed local remainder. This verifies equation 68 and completes the candidate-wise, batchwise, and marginal arguments above. The exponential UR target error follows from equation 12, not merely from cancellation of the displayed $N ^ { - 1 }$ coefficients.

## F.5 CONTINUOUS EXAMPLE

Consider $\mu = \mathrm { U n i f } ( 0 , 1 )$ and $\pi ( \mathrm { d } y ) = 2 y \mathrm { d } y$ on (0, 1), so $w ( y ) = 2 y$ and the target mean is $2 / 3$ The two output means are

$$
\mathbb { E } [ \widehat { Y } _ { U } ] = \frac { 2 } { 3 } - \frac { 2 ^ { 1 - N } } { 3 ( N + 1 ) } , \qquad \mathbb { E } [ \widehat { Y } _ { \mathrm { S I R } } ] = \frac { 2 } { 3 } - \frac { 1 } { 9 N } + O ( N ^ { - 3 / 2 } ) .\tag{70}
$$

These formulae concern means, not TV distances. For UR, we first express the mean error through the largest score; this shows which part of its distribution must be computed. For SIR, it suffices to evaluate $A _ { \psi }$ from Appendix F.3 at $\psi ( y ) = y$

UR: Reduce the mean error to scores below 2. The conditional law equation 9 has mean $s / 3$ for $0 < s < 2$ and $2 / 3$ for $s \geq 2$ . The first value is the mean of the target restricted to $( 0 , s / 2 )$ ; in the second case, no target mass is excluded. Consequently, the law of total expectation gives

$$
\frac 2 3 - \mathbb { E } [ \widehat { Y } _ { U } ] = \mathbb { E } \left[ \left( \frac 2 3 - \frac { S _ { ( N ) } } 3 \right) \mathbb { 1 } \{ S _ { ( N ) } < 2 \} \right] .
$$

It remains to compute the largest-score density on $( 0 , 2 )$ and evaluate this expectation. For independent $Y \sim \mu$ and $\bar { U } \sim \mathrm { U n i f } ( \bar { 0 } , 1 )$ , set $S : = 2 Y / U$ . Since $\operatorname* { P r } ( S \le s \mid Y = \overset { \cdot } { y } ) = ( 1 - 2 y / s )$ <sub>+</sub> for $s > 0 ,$

$$
G ( s ) = \int _ { 0 } ^ { \operatorname* { m i n } \{ 1 , s / 2 \} } ( 1 - 2 y / s ) \mathrm { d } y = { \left\{ \begin{array} { l l } { s / 4 , } & { 0 < s < 2 , } \\ { 1 - 1 / s , } & { s \geq 2 . } \end{array} \right. }
$$

Independence makes the largest-score CDF equal to $G ( s ) ^ { N }$ . Its derivative on (0, 2) is $N s ^ { N - 1 } / 4 ^ { N }$ Substituting into the preceding expectation gives

$$
\begin{array} { r } { \frac { 2 } { 3 } - \mathbb { E } [ \widehat { Y } _ { U } ] \stackrel { \mathrm { ( i ) } } { = } \displaystyle \int _ { 0 } ^ { 2 } \left( \frac { 2 } { 3 } - \frac { s } { 3 } \right) \frac { N s ^ { N - 1 } } { 4 ^ { N } } \mathrm { d } s } \\ { \stackrel { \mathrm { ( i i ) } } { = } \displaystyle \frac { 2 ^ { 1 - N } } { 3 ( N + 1 ) } . } \end{array}\tag{71}
$$

Step (i) uses the score density in the preceding reduction, and (ii) evaluates the polynomial integral.

SIR: Evaluate the general bias coefficient. For $\psi ( y ) = y$

$$
C ^ { \pi } = \frac { 4 } { 3 } , \qquad A _ { \psi } = \int _ { 0 } ^ { 1 } ( 2 y - 4 / 3 ) y ( 2 y ) \mathrm { d } y = 1 - \frac { 8 } { 9 } = \frac { 1 } { 9 } .
$$

Substituting into equation 60 proves the second formula in equation 70. Although $w ( y )$ approaches zero near the origin, it is positive almost surely and bounded above, so the assumptions of that bias calculation apply.

## G PROOF OF THEOREM 4

Fix $N \geq 2$ and maintain the structural restrictions on A mentioned in Section 5. The proof keeps relative weights fixed while varying their proposal probabilities. The resulting marginal output

probabilities are polynomials in these probabilities; the required error rate forces their low-degree coefficients to agree with $\mathrm { U R } ^ { \prime } \mathbf { s } .$ , thereby determining the selection probabilities on each batch. We first illustrate the argument for two and three candidates, then provide the general proof.

Why accuracy determines a choice. Take $N = 2$ and keep the relative weights of states 0 and 1 fixed at $1 / 2$ and 1. We change only how often state 0 is proposed: let $\mu _ { \eta } ( 0 ) = \eta$ and $\mu _ { \eta } ( 1 ) = 1 - \eta$ where $0 < \eta < 1$ . Since $\pi ( y ) \propto w _ { u } ( y ) \mu ( y )$

$$
\pi _ { \eta } ( 0 ) = \frac { \eta / 2 } { \eta / 2 + 1 - \eta } = \frac { \eta } { 2 - \eta } , \qquad C _ { \infty } ^ { \pi _ { \eta } } = \frac { 1 } { 1 - \eta / 2 } = \frac { 2 } { 2 - \eta } .
$$

Let $q$ denote the probability that A selects state 0 when the batch contains both states. The sampler sees the same relative weights $( 1 / 2 , 1 )$ on every such batch, regardless of $\eta ,$ and symmetry makes this probability independent of the order of the two candidates. Hence the same q must be used for every η. Decomposing over the three batch types,

$$
P _ { 2 } ^ { \mathrm { A } } ( \{ 0 \} ) = \operatorname* { P r } ( 0 0 ) + \operatorname* { P r } ( \{ 0 1 , 1 0 \} ) \operatorname* { P r } ( \mathrm { r e t u r n } \ 0 \ | \ \{ 0 1 , 1 0 \} ) = \eta ^ { 2 } + 2 \eta ( 1 - \eta ) q .
$$

For small $\eta ,$ one can readily see that $P _ { 2 } ^ { \mathsf { A } } ( \{ 0 \} ) = 2 q \eta + O ( \eta ^ { 2 } )$ and $\begin{array} { r } { \pi _ { \eta } ( 0 ) = \frac { \eta } { 2 } + { O } ( \eta ^ { 2 } ) } \end{array}$ . Hence any mismatch between $2 q$ and $\mathrm { i } / 2$ creates a first-order error in η. However, the assumed guarantee allows only second-order error, since $\begin{array} { r } { 1 - \frac { 1 } { C _ { \infty } ^ { \pi _ { \eta } } } = \frac { \eta } { 2 } } \end{array}$ . Since the state space is binary, $| P _ { 2 } ^ { \mathsf { A } } ( \bar { \{ 0 \} } ) - \pi _ { \eta } ( 0 ) | =$ $D _ { \mathrm { T V } } ( P _ { 2 } ^ { \mathsf { A } } , \pi _ { \eta } )$ and equation 27 becomes:

$$
\left| \eta ^ { 2 } + 2 \eta ( 1 - \eta ) q - \frac { \eta } { 2 - \eta } \right| = | P _ { 2 } ^ { \Delta } ( \{ 0 \} ) - \pi _ { \eta } ( 0 ) | \leq \frac { L _ { 2 } \eta ^ { 2 } } { 4 } \Longleftrightarrow \left| \eta + 2 ( 1 - \eta ) q - \frac { 1 } { 2 - \eta } \right| \leq \frac { L _ { 2 } \eta } { 4 } .
$$

Letting $\eta \downarrow 0 ,$ , this forces $2 q = 1 / 2 , \mathrm { { s o } } q = 1 / 4$ . This is exactly $\mathrm { U R } ' _ { \mathrm { s } }$ probability, since $\mathrm { P r } ( U _ { L } <$ $U _ { H } / 2 \bar { ) } \stackrel { \cdot } { = } 1 / 4$ for independent uniforms on $( 0 , 1 )$ . The general proof below applies a similar argument with several rare states, determining the probabilities on every possible batch.

## G.1 THREE CANDIDATES SHOW HOW ACCURACY DETERMINES A BATCHWISE CHOICE

The preceding binary example determines a choice from a first-order term. With three candidates, a batch containing two rare states first appears in a second-order term. This is why one must examine more than the probability of seeing a single rare candidate.

Take distinct relative weights $a , b \in ( 0 , 1 )$ and a reference weight 1. Give these three states proposal probabilities $\eta _ { 1 } , \eta _ { 2 } , 1 - \eta _ { 1 } - \eta _ { 2 }$ , respectively, where $\eta _ { 1 } , \eta _ { 2 } > 0$ and $\eta _ { 1 } + \eta _ { 2 } < 1$ . Write $\eta : = ( \eta _ { 1 } , \eta _ { 2 } )$ The common normalizer is $Z _ { \eta } : = 1 - ( 1 - a ) \eta _ { 1 } - ( 1 - b ) \eta _ { 2 }$ . The largest normalized importance weight is therefore

$$
C _ { \infty } ^ { \pi _ { \eta } } = \frac { 1 } { Z _ { \eta } } ,
$$

and hence

$$
1 - \frac { 1 } { C _ { \infty } ^ { \pi _ { \eta } } } = ( 1 - a ) \eta _ { 1 } + ( 1 - b ) \eta _ { 2 } .
$$

The assumed bound equation 27, specialized to $N = 3$ , gives

$$
D _ { \mathrm { T V } } ( P _ { 3 } ^ { \mathrm { A } } , \pi _ { \eta } ) \leq L _ { 3 } \left( ( 1 - a ) \eta _ { 1 } + ( 1 - b ) \eta _ { 2 } \right) ^ { 3 } .
$$

Therefore, any first- or second-order mismatch between the sampler and the target would violate the cubic error bound.

First, consider the separate binary family containing only relative weights a and 1. Let ℓ be the probability that A selects the sole a candidate from $( a , 1 , 1 )$ . The sampler returns the lower weight state with probability $3 \eta _ { 1 } ( 1 - \eta _ { 1 } ) ^ { 2 } \ell + O ( \eta _ { 1 } ^ { 2 } ) = 3 \eta _ { 1 } \ell + O ( \eta _ { 1 } ^ { 2 } )$ , while the corresponding target probability is

$$
\frac { a \eta _ { 1 } } { 1 - ( 1 - a ) \eta _ { 1 } } = a \eta _ { 1 } + O ( \eta _ { 1 } ^ { 2 } ) .
$$

The assumed error is $O ( \eta _ { 1 } ^ { 3 } )$ , so their linear coefficients agree and $\ell = a / 3$

Now return to the ternary state family and let x be the probability of selecting the a candidate from $( a , b , 1 )$ . The target probability of the state with relative weight a expands as:

$$
\frac { a \eta _ { 1 } } { Z _ { \eta } } = a \eta _ { 1 } + a ( 1 - a ) \eta _ { 1 } ^ { 2 } + a ( 1 - b ) \eta _ { 1 } \eta _ { 2 } + O ( ( \eta _ { 1 } + \eta _ { 2 } ) ^ { 3 } ) .
$$

Hence its mixed $\eta _ { 1 } \eta _ { 2 }$ coefficient is $a ( 1 - b )$ . We now compute the same coefficient for the sampler. The contribution of the batch composition $( a , 1 , 1 )$ is

$$
3 \eta _ { 1 } ( 1 - \eta _ { 1 } - \eta _ { 2 } ) ^ { 2 } \frac { a } { 3 } = a \eta _ { 1 } - 2 a \eta _ { 1 } ^ { 2 } - 2 a \eta _ { 1 } \eta _ { 2 } + O ( ( \eta _ { 1 } + \eta _ { 2 } ) ^ { 3 } ) ,
$$

so its mixed coefficient is $- 2 a$ . The contribution of $( a , b , 1 )$ is

$$
6 \eta _ { 1 } \eta _ { 2 } \bigl ( 1 - \eta _ { 1 } - \eta _ { 2 } \bigr ) x = 6 x \eta _ { 1 } \eta _ { 2 } + O \bigl ( ( \eta _ { 1 } + \eta _ { 2 } ) ^ { 3 } \bigr ) ,
$$

so its mixed coefficient is 6x. Batches without an a candidate contribute zero. Every remaining composition with an a candidate contains either at least two a candidates or three lower weight candidates, so none contributes to the $\eta _ { 1 } \eta _ { 2 }$ coefficient. Therefore the sampler’s mixed coefficient is $- 2 a + 6 x$

The cubic error bound forces this coefficient to match the target’s, giving

$$
- 2 a + 6 x = a ( 1 - b ) .
$$

Solving for x gives

$$
x = \frac { a ( 3 - b ) } { 6 } = a \int _ { 0 } ^ { 1 } ( 1 - b u ) ( 1 - u ) \mathrm { d } u ,\tag{72}
$$

which is precisely UR’s probability from equation 51. The contribution from $( a , 1 , 1 )$ was determined first, leaving the mixed coefficient to determine the choice on $( a , b , 1 )$ . The general proof repeats this procedure in increasing order of the number of lower weight candidates.

## G.2 GENERAL PROOF IN THREE STEPS

Step 1. Construct a family with fixed relative weights. We construct a family of target–proposal pairs that keeps the relative weights fixed while making the lower weight states rare.

To this end, we choose $1 \leq d \leq N - 1$ distinct relative weights $a _ { 1 } , \dotsc , a _ { d } \in ( 0 , 1 )$ and let $a _ { 0 } : = 1$ State 0 is the reference state with the largest relative weight, while states $1 , \ldots , d$ have smaller fixed weights. For $j = 1 , \ldots , d ,$ choose $\eta _ { j } > 0$ with $\textstyle \sum _ { j = 1 } ^ { d } \eta _ { j } < 1$ , and set

$$
\eta _ { 0 } : = 1 - \sum _ { j = 1 } ^ { d } \eta _ { j } .
$$

We consider the setting where the $\boldsymbol { a } _ { j } ^ { \cdot } \mathbf { \dot { s } }$ remain fixed throughout the argument, whereas the $\eta _ { j } \mathrm { ^ { * } s }$ tend to zero.

Now write $\pmb { \eta } : = ( \eta _ { 1 } , \dots , \eta _ { d } )$ and define the normalizing constant

$$
Z _ { \eta } : = \sum _ { j = 0 } ^ { d } a _ { j } \eta _ { j } = 1 - \sum _ { j = 1 } ^ { d } ( 1 - a _ { j } ) \eta _ { j } .
$$

The proposal, target, and normalized weights are, for $j = 0 , \ldots , d ,$

$$
\mu _ { \eta } ( j ) : = \eta _ { j } , \qquad \pi _ { \eta } ( j ) : = \frac { a _ { j } \eta _ { j } } { Z _ { \eta } } , \qquad w _ { \eta } ( j ) : = \frac { a _ { j } } { Z _ { \eta } } .\tag{73}
$$

Hence the fixed values $a _ { j }$ are unnormalized importance weights, in the same sense as $w _ { u } = Z w$ in Section 1. Note that varying η changes their frequencies and their common normalization, but not their ratios. The largest normalized weight is $C _ { \infty } ^ { \pi _ { \eta } ^ { \star } } = 1 / Z _ { \eta }$ , so

$$
1 - \frac { 1 } { C _ { \infty } ^ { \pi _ { \eta } } } = \sum _ { j = 1 } ^ { d } ( 1 - a _ { j } ) \eta _ { j } .\tag{74}
$$

As the smaller weights become rare, this quantity tends to zero and the required error becomes small.

Step 2: Express marginal output differences as a polynomial. We compare A with UR rather than directly with the target. The target probabilities involve the changing normalizer $Z _ { \eta } ^ { - 1 }$ <sup>1</sup>, whereas both samplers’ output probabilities are polynomials in $\eta .$ To this end, let $\pmb { n } : = ( n _ { 1 } , \ldots , n _ { d } )$ be a count vector which records how many candidates from each lower weight state appear in the batch. Here $\begin{array} { r } { | { \pmb n } | : = \sum _ { i = 1 } ^ { d } n _ { j } \le N } \end{array}$ , so the remaining $N - | n |$ candidates are in state 0. For example, if $d = 2$ , then ${ \pmb n } = ( 1 , 2 )$ means one candidate from state 1, two from state $2 ,$ , and the remaining $N - 3$ candidates from state 0. Let $q _ { j } ^ { \mathsf { A } } ( { \pmb n } )$ be the probability of returning state $j$ from this batch, and define $q _ { j } ^ { U } ( { \pmb n } )$ similarly. Unlike $p _ { i } ^ { \mathsf { A } }$ , which refers to one candidate, $q _ { j } ^ { \mathsf { A } }$ sums the probabilities of all candidates in state $j .$ Neither the ordering nor the common normalization affects these probabilities, so they are independent of η. Write $\Delta _ { j } ( \mathbf { \breve { n } } ) : = q _ { j } ^ { \mathsf { A } } ( { \pmb n } ) - q _ { j } ^ { U } ( { \pmb n } )$

Lemma 9 (Output probability differences are polynomials). For the pair in equation 73 and $j =$ $1 , \ldots , d ,$ let $D _ { j } \mathbf { \bar { ( } } \pmb { \eta } ) \overset { \cdot } { : } = P _ { N } ^ { \mathsf { A } } ( \{ j \} ) - P _ { N } ^ { U } ( \{ j \} )$ . Then

$$
D _ { j } ( \pmb { \eta } ) = \sum _ { | \pmb { n } | \le N } \frac { N ! } { ( N - | \pmb { n } | ) ! \prod _ { h = 1 } ^ { d } n _ { h } ! } \eta _ { 0 } ^ { N - | \pmb { n } | } \prod _ { h = 1 } ^ { d } \eta _ { h } ^ { n _ { h } } \Delta _ { j } ( \pmb { n } ) .\tag{75}
$$

This is a polynomial in $\eta _ { 1 } , \ldots , \eta _ { d }$ of total degree at most $N .$

Proof. For each count vector, the multinomial factor times the powers of $\eta _ { h }$ is the probability of that batch composition. Multiplying by its conditional output difference and summing is the law of total probability. Since $\begin{array} { r } { \eta _ { 0 } = 1 - \sum _ { h = 1 } ^ { d } \eta _ { h } } \end{array}$ , each term has degree at most N. No continuity of the selection functions is needed: their values on each fixed relative weight vector remain unchanged while the probabilities of those vectors vary. □

Step 3: Use the error guarantee to recover the batchwise probabilities. We now use the assumed bound equation 27 to determine the choices within each batch. Fix $v \in ( 0 , \infty ) ^ { d }$ and let $\pmb { \eta } = \theta \pmb { v }$ for sufficiently small $\theta > 0$ . By equation $7 4 .$ , the triangle inequality and the two error bounds $\mathrm { g i v e }$

$$
\begin{array} { l } { \displaystyle | D _ { j } ( \theta v ) | \overset { \mathrm { ( i ) } } { \leq } D _ { \mathrm { T V } } ( P _ { N } ^ { \mathrm { A } } , \pi _ { \theta v } ) + D _ { \mathrm { T V } } ( P _ { N } ^ { U } , \pi _ { \theta v } ) } \\ { \displaystyle \qquad \mathrm { ( i i ) } } \\ { \displaystyle \qquad \leq ( L _ { N } + 1 ) \left( \theta \sum _ { h = 1 } ^ { d } ( 1 - a _ { h } ) v _ { h } \right) ^ { N } } \\ { \displaystyle \qquad = O ( \theta ^ { N } ) } \\ { \displaystyle \qquad = o ( \theta ^ { N - 1 } ) , } \end{array}
$$

where (i) compares both samplers with the same target on the event $\{ j \}$ ; and (ii) uses equation 27 and Corollary 1. The constant $L _ { N }$ cannot change with θ, since it is independent of the target and proposal. The following lemma explains why this small error removes every lower-degree term.

Lemma 10 (Small marginal differences remove the lower-degree terms). Suppose that, for every fixed $v \in ( 0 , \infty ) ^ { d }$

$$
D _ { j } ( \theta v ) = o ( \theta ^ { N - 1 } ) \qquad ( \theta \downarrow 0 ) .
$$

Then every coefficient of $D _ { j }$ of total degree below N is zero.

Proof. Setting $\mathbf { \boldsymbol { \eta } } = \theta \mathbf { \boldsymbol { v } }$ scales all rare-state probabilities down together while keeping their proportions fixed. For this fixed $^ { v , }$ write

$$
D _ { j } ( \theta \pmb { v } ) = c _ { 0 } ( \pmb { v } ) + c _ { 1 } ( \pmb { v } ) \theta + \cdot \cdot \cdot + c _ { N } ( \pmb { v } ) \theta ^ { N } .
$$

If the first nonzero term had degree $m < N$ , dividing by $\theta ^ { m }$ would give a nonzero limit. The assumed bound instead gives a limit of zero, including when $m = N - 1$ . Hence $c _ { m } ( \pmb { v } ) = 0$ for every $m < N$

This conclusion holds for every positive v, not just one choice of proportions. For each $m , c _ { m } ( \pmb { v } )$ is exactly the degree-m part of $D _ { j }$ evaluated at v. It is a polynomial that vanishes throughout the positive orthant, and therefore all its coefficients vanish. In particular, fix all but one coordinate in an open box and use that a univariate polynomial vanishing on an interval is zero; repeating this for the other coordinates proves the claim. Lastly, notice that varying the proportions is essential: a polynomial such as $\eta _ { 1 } - \eta _ { 2 }$ vanishes when $\eta _ { 1 } = \eta _ { 2 }$ without having zero coefficients. □

Lemma 11 (The coefficients determine the selection probabilities). Ifall coefficients of $D _ { j }$ oftotal degree below N vanishfor $j = 1 , \ldots , d ,$ then $\Delta _ { j } ( { \pmb n } ) = 0$ whenever $| n | < N$ . Allowing the relative weights to vary identifies each candidate’s selection probability on every positive input vector.

Proof. Fix $j \geq 1$ and proceed in increasing order of $m : = | { \boldsymbol { n } } |$ . For $m = 0$ , state $j$ is absent, so both rules assign it probability zero. Now fix $1 \overset { \cdot } { \le } m \le N - 1$ and suppose all compositions with fewer than m lower weight candidates already have $\Delta _ { j } = 0$ . Their terms in equation 75 vanish entirely. Compositions with more than m lower weight candidates cannot contribute to degree m, since they contain more than m factors of the rare-state probabilities. For a composition with exactly m such candidates,

$$
\eta _ { 0 } ^ { N - m } = \left( 1 - \sum _ { h = 1 } ^ { d } \eta _ { h } \right) ^ { N - m }
$$

has constant term one, and all its other terms raise the total degree above $m$ . Therefore the degree-m part is exactly

$$
\sum _ { | { \pmb { n } } | = m } { \frac { N ! } { ( N - m ) ! \prod _ { h = 1 } ^ { d } n _ { h } ! } } \Delta _ { j } ( { \pmb { n } } ) \prod _ { h = 1 } ^ { d } \eta _ { h } ^ { n _ { h } } .
$$

Each composition produces a distinct monomial with a positive multinomial coefficient. Since the displayed polynomial is identically zero, every $\Delta _ { j } ( n )$ with $| { \boldsymbol n } | = m$ is zero. Induction therefore proves the assertion for $j \geq 1$ ; the probabilities of state 0 then agree since each rule’s probabilities sum to one.

Finally, divide an arbitrary positive weight vector by its maximum. If all entries become one, symmetry gives probability $1 / N$ to each candidate. Otherwise, its distinct smaller values are some $a _ { 1 } , \ldots , a _ { d } ,$ with multiplicities $n _ { 1 } , \ldots , n _ { d }$ . At least one maximal entry remains, so $| { \pmb n } | \le N - 1$ . The preceding induction identifies the total selection probability of each equal-weight group. Within a group, exchanging two candidates leaves the weight vector unchanged, so symmetry assigns them equal probabilities. In conclusion, the group probability determines each individual probability.

Lemma 10 removes the lower-degree coefficients, and Lemma 11 determines every batchwise probability. Conversely, Corollary 1 supplies the UR bound with $L _ { N } = 1$ . This completes the proof of Theorem 4.

A weaker local condition. For fixed $N ,$ , the same conclusion holds under the weaker requirement

$$
\operatorname* { s u p } _ { ( \pi , \mu ) : C _ { \infty } ^ { \pi } \leq 1 + \delta } D _ { \mathrm { T V } } ( P _ { N } ^ { \mathrm { A } } , \pi ) = o ( \delta ^ { N - 1 } ) \qquad ( \delta \downarrow 0 ) ,
$$

where the supremum ranges over finite target–proposal pairs with strictly positive probabilities. For each fixed $v \in ( 0 , \infty ) ^ { d }$ , equation 74 shows that $C _ { \infty } ^ { \pi _ { \theta v } } - 1$ is proportional to θ to first order. The local requirement and UR’s bound therefore give $D _ { j } ( \tilde { \theta } v ) = o ( \tilde { \theta ^ { N - 1 } } )$ ). Lemmas 10 and 11 then apply unchanged.

## H NONUNIFORM DIVISORS CAN ALSO CONVERGE EXPONENTIALLY

Uniformity of the auxiliary variables is not necessary for exponential convergence. Consider the more general score

$$
S _ { i } ^ { F } : = \frac { w ( Y _ { i } ) } { T _ { i } } ,
$$

where the $T _ { i }$ are i.i.d. positive random variables independent of the proposals. As a simple example, let

$$
T \sim { \frac { 1 } { 2 } } \operatorname { U n i f } ( 0 , 1 ) + { \frac { 1 } { 2 } } \operatorname { U n i f } ( 2 , 3 ) .
$$

Its CDF satisfies

$$
F ( t ) = \frac { t } { 2 } , \qquad 0 \leq t \leq 1 .
$$

Suppose $C _ { \infty } ^ { \pi } < \infty$ . A candidate with state y has score at least $C _ { \infty } ^ { \pi }$ exactly when

$$
T _ { i } \leq \frac { w ( y ) } { C _ { \infty } ^ { \pi } } .
$$

Since $w ( y ) / C _ { \infty } ^ { \pi } \leq 1$

$$
\mathrm { P r } \left( T _ { i } \leq \frac { w ( y ) } { C _ { \infty } ^ { \pi } } \Big | Y _ { i } = y \right) = \frac { w ( y ) } { 2 C _ { \infty } ^ { \pi } } .
$$

Let $\widehat { I } _ { F } : = \arg \operatorname* { m a x } _ { i \in [ N ] } S _ { i } ^ { F }$ and $P _ { N } ^ { F } : = \mathcal { L } ( Y _ { \widehat { I } _ { F } } )$ . Suppose $C _ { \infty } ^ { \pi } < \infty$ and define

$$
H _ { i } : = \left\{ T _ { i } \leq { \frac { w ( Y _ { i } ) } { C _ { \infty } ^ { \pi } } } \right\} , \quad \quad V _ { i } : = { \frac { C _ { \infty } ^ { \pi } T _ { i } } { w ( Y _ { i } ) } } \quad { \mathrm { o n ~ } } H _ { i } .
$$

For every measurable set $D$ and $0 \leq v \leq 1$

$$
\operatorname* { P r } ( Y _ { i } \in D , H _ { i } , V _ { i } \leq v ) = \int _ { D } F \left( { \frac { v w ( y ) } { C _ { \infty } ^ { \pi } } } \right) \mu ( d y ) = { \frac { v } { 2 C _ { \infty } ^ { \pi } } } \pi ( D ) .
$$

Hence, one can observe that conditional on $H _ { i } , Y _ { i } \sim \pi$ and $V _ { i } \sim \mathrm { U n i f } ( 0 , 1 )$ are independent. Conditioning on the set of crossing indices preserves independence across candidates, since each crossing event depends only on its own candidate and divisor. Among the candidates satisfying $H _ { i } ,$ the rule selects the smallest $V _ { i } ,$ , since $S _ { i } ^ { F } = C _ { \infty } ^ { \pi } / V _ { i }$ on $H _ { i }$ . Therefore, conditional on at least one threshold crossing, the selected state has law π.

Each candidate crosses independently with probability $1 / ( 2 C _ { \infty } ^ { \pi } )$ . It follows that

$$
D _ { \mathrm { T V } } ( P _ { N } ^ { F } , \pi ) \leq \left( 1 - \frac { 1 } { 2 C _ { \infty } ^ { \pi } } \right) ^ { N } .
$$

In conclusion, we observe that uniform divisors are not necessary for exponential convergence. Notice that this bound has a slower exponential rate than UR’s bounded weight guarantee though, since only half of the divisor mass lies in the linear part of the CDF.

## I EXPERIMENTAL DETAILS

This section provides additional details on the experimental setup, evaluation procedure, and results.

Datasets. Throughout the experiments, we evaluate on two mathematical reasoning benchmarks.

GSM8K (Cobbe et al., 2021) is a dataset created by OpenAI, consisting of 8.5K grade-level math word problems. For each evaluated prompt, we pre-generated a response pool $\mathcal { A } _ { x } ^ { \star }$ of $N ^ { \star } : = 4 , 0 9 6$ responses from the reference policy, with a maximum generation length of 500 tokens.

MATH500 (Hendrycks et al., 2021; Lightman et al., 2024) is a 500-question evaluation subset of the MATH benchmark, composed of competition-level mathematics problems. Compared to GSM8K, MATH500 requires more complex multi-step mathematical reasoning, providing a setting where we can evaluate our method with more challenging problems. For each problem (i.e., evaluated prompt), we pre-generated a response pool $\mathcal { A } _ { x } ^ { \star }$ of 4,096 responses, with a maximum generation length of 1,024 tokens.

Reference policy and response generation. We use meta-llama/Llama-3.2-3B-Instr uct (Grattafiori et al., 2024) as the reference policy. Specifically, for each evaluated prompt x, we pre-generate a fixed response pool

$$
\mathcal { A } _ { x } ^ { \star } : = \{ a _ { 1 } , \ldots , a _ { N ^ { \star } } \} , \qquad N ^ { \star } = 4 , 0 9 6 .
$$

We use a generation temperature of 0.3 and maximum generation lengths of 500 tokens for GSM8K and 1,024 tokens for MATH500 respectively. The response generation and reward model settings are summarized in Table 2. All response generation and reward model evaluation were run on two NVIDIA RTX 6000 Pro Blackwell GPUs.

Finite empirical proposal. The fixed response pool $\mathcal { A } _ { x } ^ { \star }$ defines the empirical distribution $\widehat { \mu } ( y \mid$ $\begin{array} { r } { x ) : = \frac { 1 } { N ^ { \star } } \sum _ { i = 1 } ^ { N ^ { \star } } \mathbb { 1 } \{ a _ { i } = y \} } \end{array}$ . Since $\mathcal { A } _ { x } ^ { \star }$ may contain repeated responses, let s $\operatorname { 1 p p } ( { \widehat { \mu } } ( \cdot \mid x ) )$ denote the set of distinct responses represented in $\mathcal { A } _ { x } ^ { \star }$ . For brevity, we write $\operatorname { s u p p } ( \widehat { \mu } )$ when the prompt x is fixed. This fixed empirical proposal allows us to draw repeated independent candidate sets without additional LLM generations. The pool size $N ^ { \star }$ is fixed throughout the experiment, whereas N denotes the sampling budget available to each sampling procedure and is varied to produce the curves.

<table><tr><td>Component</td><td>Setting</td><td>Value</td></tr><tr><td></td><td>Model</td><td>meta-11ama/Llama-3.2-3B-Instruct</td></tr><tr><td></td><td>Max tokens</td><td>500 (GSM8K); 1,024 (MATH500)</td></tr><tr><td>Response generation</td><td>Generation temperature</td><td>0.3</td></tr><tr><td></td><td>Pre-generated responses/prompt 4,096</td><td></td></tr><tr><td>Reward model</td><td>Model</td><td>OpenAssistant/reward-model-deber  $\mathtt { t a \mathrm { - v } 3 - l a r g e - v } 2$ </td></tr></table>

Table 2: Response generation and reward model settings.

Reward model and target policy. Each distinct response $y \in \operatorname { s u p p } ( \widehat { \mu } )$ is scored by OpenAssist ant/reward-model-deber $\mathtt { : a - v 3 - l a r g e - v 2 }$ (Köpf et al., 2023), yielding a reward $\widehat { r } ( x , y )$ Motivated by the standard KL-regularized RLHF objective (Jaques et al., 2017; 2020; Rafailov et al., 2023), we define the reward-tilted target policy

$$
\pi _ { \beta } ( y \mid x ) = \frac { \widehat \mu ( y \mid x ) \exp ( \widehat \ r ( x , y ) / \beta ) } { \sum _ { y ^ { \prime } \in \operatorname { s u p p } ( \widehat \mu ) } \widehat \mu ( y ^ { \prime } \mid x ) \exp ( \widehat { r } ( x , y ^ { \prime } ) / \beta ) } , \quad y \in \operatorname { s u p p } ( \widehat \mu ) .
$$

Since $\operatorname { s u p p } ( \widehat { \mu } )$ is finite, the normalizing constant can be computed exactly for evaluation. The sampling procedures themselves receive only the unnormalized importance weights

$$
w _ { u } ( x , y ) = \exp \left( \widehat { r } ( x , y ) / \beta \right) .
$$

Hence, $( \pi _ { \beta } , \widehat { \mu } )$ forms a finite target–proposal pair to which the theoretical results in Sections 2–5 apply directly.

## I.1 SAMPLING ALGORITHMS

We consider the four sampling algorithms mentioned in Section 6. In particular, all four samplers receive the same N i.i.d. candidates $Y _ { 1 } , \dots , Y _ { N } \overset { \mathrm { i i d } } { \sim } \widehat \mu ( \cdot \mid x )$ and their unnormalized weights $w _ { u } ( x , Y _ { i } )$ Each sampler returns exactly one of the observed candidates. If $N = 1$ , all four methods return the sole candidate and hence coincide with ${ \widehat { \mu } } .$ As the implementations of UR and SIR are described clearly in previous sections, we provide the detailed implementations of the remaining two algorithms.

Envelope RS. This baseline implements the approximate rejection sampler of Section 1. For convenience, we parameterize its threshold directly on the unnormalized weight scale. Given a threshold $\tau > 0$ , candidates are examined in order and candidate i is accepted with probability

$$
\operatorname* { m i n } \left\{ \frac { w _ { u } ( x , Y _ { i } ) } { \tau } , 1 \right\} .
$$

The first accepted candidate is returned; if all N candidates are rejected, one of the observed candidates is returned uniformly at random. Equivalently, $\tau = Z M$ for the normalized threshold M used in the theoretical formulation.

For each $( x , \beta )$ , we set the threshold using the fixed response pool $\mathcal { A } _ { x } ^ { \star }$ and use the same threshold for every N. Specifically, we set $\tau : = \mathrm { m a x } _ { y \in \mathrm { s u p p } ( \widehat { \mu } ) } w _ { u } ( x , y )$ . Hence, envelope RS serves as a proxy for the best fixed-threshold RS in hindsight.

Budget-calibrated RS. This baseline is adapted from Rohatgi et al. (2025, Algorithm 5) as described in Section 3. In particular, given a sampling budget $N _ { \ast }$ , set $\begin{array} { r } { n = { \left| \begin{array} { l } { \frac { N - 1 } { 2 } } \end{array} \right| } } \end{array}$ . For $N \in \{ 1 , 2 \}$ for which $n = 0 ,$ , we define the baseline to return a single fresh candidate $Y _ { 1 } \stackrel { - } { \sim } \widehat { \mu } ( \cdot \overline { { | } } x )$ . The algorithm first uses n candidates as pilot samples to estimate the normalizing constant, $\begin{array} { r } { \widehat { Z } : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } w _ { u } ( x , Y _ { i } ) } \end{array}$ and the threshold multiplier is set to $\begin{array} { r } { M _ { N , \delta } : = \frac { n } { 4 \log ( 4 / \delta ) } , \delta \in ( 0 , \mathbf { \bar { 1 } } ) } \end{array}$ . The algorithm then runs standard RS on the next n candidates with acceptance probability

$$
\operatorname* { m i n } \left\{ \frac { w _ { u } ( x , Y _ { i } ) } { M _ { N , \delta } \widehat { Z } } , 1 \right\} ,
$$

and the fresh fallback candidate $Y _ { 2 n + 1 }$ is returned if all are rejected. Hence, at most $2 n + 1 \leq N$ candidates are used. In addition, we select δ from $\{ 0 . 0 0 1 , 0 . 0 1 , 0 . 0 5 , 0 . 1 , 0 . 2 5 , 0 . 5 , 0 . 9 \}$ separately for each $( x , \beta , N )$ . For the TV plots, we choose the value minimizing the estimated TV error; for the accuracy plots, we choose the value maximizing ground-truth accuracy. All candidate values of δ are evaluated on the same Monte Carlo runs.

## I.2 EVALUATION METRICS AND ESTIMATION

We report two complementary metrics: (1) the TV error $D _ { \mathrm { T V } } ( P _ { N } , \pi _ { \beta } )$ and (2) ground-truth accuracy.

TV error. Let $P _ { N } ( \cdot \mid x )$ denote the marginal distribution of the response returned by a sampler with budget N. Since all four algorithms are designed to approximate the same target distribution $\pi _ { \beta }$ , we measure sampling fidelity by $D _ { \mathrm { T V } } ( P _ { N } , \pi _ { \beta } )$ . We note that this is also the quantity controlled directly by our theoretical results.

Ground-truth accuracy. Ground-truth accuracy is not the objective optimized by the samplers. Rather, it measures whether the sampled response gives the correct final answer to the underlying math problem. For a fixed prompt x, define the target accuracy

$$
\operatorname { A c c } ( \pi _ { \beta } ; x ) : = \sum _ { y } \pi _ { \beta } ( y \mid x ) \mathbb { 1 } \left\{ y { \mathrm { ~ h a s ~ t h e ~ c o r r e c t ~ f i n a l ~ a n s w e r } } \right\} .
$$

Since correctness corresponds to an event, $\left| \operatorname { A c c } ( P _ { N } ; x ) - \operatorname { A c c } ( \pi _ { \beta } ; x ) \right| \leq D _ { \operatorname { T V } } ( P _ { N } , \pi _ { \beta } )$ . Hence, as the output distribution approaches $\pi _ { \beta } .$ , its ground-truth accuracy approaches the corresponding target accuracy. We report accuracy to assess whether improved sampling fidelity also appears in downstream task performance. Importantly, the target accuracy may be either higher or lower than the reference policy accuracy, so accuracy need not increase monotonically with $\bar { N } .$

## I.2.1 ESTIMATING THE TV ERROR ON A FINITE POOL

Computing the TV error requires knowledge of the target distribution $\pi _ { \beta }$ and the output law $P _ { N }$ Note that the target distribution $\pi _ { \beta }$ can be computed exactly on the finite empirical support supp(µb), but the output law $P _ { N }$ of a sampling procedure should be estimated through repeated trials.

To estimate $P _ { N }$ , for each prompt $x ,$ temperature parameter $\beta ,$ and sampling budget $N ,$ each trial independently draws $Y _ { 1 } , \dots , Y _ { N } \overset { \mathrm { i i d } } { \sim } \widehat \mu ( \cdot \mid x )$ and applies the corresponding sampling procedure. We repeat this experiment m times, using the trial counts specified below and record the returned responses $\widehat { Y } ^ { ( 1 ) } , \ldots , \widehat { Y } ^ { ( m ) }$ . A fresh candidate set is drawn in every trial because $P _ { N }$ is the marginal law of the returned response, averaging over both candidate generation and the sampler’s internal randomness. Reusing a single candidate set would instead estimate the output law conditional on that set. With this, we estimate $P _ { N }$ by its empirical frequencies,

$$
\widehat { P } _ { N } ( y \mid x ) : = \frac { 1 } { m } \sum _ { t = 1 } ^ { m } \mathbb { 1 } \{ \widehat { Y } ^ { ( t ) } = y \} , \quad y \in \operatorname { s u p p } ( \widehat { \mu } ) ,
$$

and report the plug-in TV estimate

$$
\widehat { D } _ { \mathrm { T V } } ( P _ { N } , \pi _ { \beta } ) : = \frac { 1 } { 2 } \sum _ { y \in \mathrm { s u p p } ( \widehat { \mu } ) } \left| \widehat { P } _ { N } ( y \mid x ) - \pi _ { \beta } ( y \mid x ) \right| .
$$

Monte Carlo floor. We highlight that even when a sampler is exact, the plug-in TV estimate above is generally positive because $\widehat { P } _ { N }$ is computed from finitely many Monte Carlo trials. Consequently, once the true sampling error becomes sufficiently small, the observed TV error is limited by the statistical resolution of the estimator.

To quantify this effect without relying on an asymptotic approximation, we compute an exact sampler baseline. For each setting, we draw m independent samples directly from $\pi _ { \beta } .$ , form their empirical distribution using the same estimator, and compute its TV distance from $\pi _ { \beta }$ . This provides the Monte Carlo floor at the same number of trials used for the sampling algorithms. We show this exact sampler floor as a dotted line in the corresponding appendix figures. Once a sampler reaches this level, differences of comparable magnitude cannot be reliably distinguished from Monte Carlo estimation error.

## I.3 FURTHER EXPERIMENTAL RESULTS

This section provides additional empirical results supporting Section 6. We first summarize the theoretical comparisons relevant to the experiments and then examine their empirical behavior across prompts and values of β.

Theoretical comparisons and empirical reference points. Since $( \pi _ { \beta } , \widehat { \mu } )$ forms a valid finite target–proposal pair, the theoretical results apply directly to the corresponding population output distributions:

• UR vs. envelope RS. (Theorem 1) UR attains the best fixed-threshold RS guarantee in hindsight without requiring threshold information. Envelope RS serves as an empirical proxy for this benchmark, and we therefore expect the two methods to exhibit comparable empirical performance.

• UR vs. budget-calibrated RS (Theorem 2): for every prompt, $\beta ,$ sampling budget N, and δ, $D _ { \mathrm { T V } } ( P _ { N } ^ { U } , \bar { \pi _ { \beta } } ) \leq D _ { \mathrm { T V } } ( P _ { N , \delta } ^ { R } , \pi _ { \beta } )$

• UR vs. SIR (Theorem 3): for every prompt, $\beta ,$ and sampling budget N, $D _ { \mathrm { T V } } ( P _ { N } ^ { U } , \pi _ { \beta } ) \ \leq$ $D _ { \mathrm { T V } } ( P _ { N } ^ { \mathrm { S I R } } , \pi _ { \beta } )$

• Monte Carlo resolution. Once the true TV error becomes sufficiently small, the estimated curves approach the scale of the exact-sampler Monte Carlo baseline. Differences at this scale cannot be reliably distinguished from Monte Carlo estimation error.

Number of trials. All appendix results in this section use $m = 1 0 ^ { 5 }$ independent runs. For Figure 1, we use $m = 1 0 ^ { 6 }$ to reduce Monte Carlo estimation error.

Confidence intervals for TV error. Since $\widehat { D } _ { \mathrm { T V } }$ is computed from the empirical output distribution $\widehat { P } _ { N }$ based on m independent runs, we estimate its Monte Carlo variability using a bootstrap. For each $b = 1 , \dots , B$ with $B = 2 0 0$ , we draw m responses independently from $\hat { P } _ { N }$ , then form the corresponding empirical distribution $\widehat { P } _ { N } ^ { \star ( b ) }$ , and compute

$$
\widehat { D } _ { b } ^ { \star } : = \frac { 1 } { 2 } \sum _ { y \in \mathrm { s u p p } ( \widehat { \mu } ) } \left| \widehat { P } _ { N } ^ { \star ( b ) } ( y \mid x ) - \pi _ { \beta } ( y \mid x ) \right| .
$$

We take se to be the sample standard deviation of $\{ \widehat { D } _ { b } ^ { \star } \} _ { b = 1 } ^ { B }$ and report the approximate 95% interval $\widehat { D } _ { \mathrm { T V } } \pm z _ { 0 . 9 7 5 } \widehat { \mathrm { s e } }$ . The interval reflects Monte Carlo variability of the estimator. The exact-sampler baseline separately shows the nonzero plug-in error induced by finite m when the true TV error is zero.

Confidence intervals for accuracy. For accuracy, each of the m independent runs yields a Bernoulli outcome. We report $\widehat { a } \pm t _ { m - 1 , 0 . 9 7 5 } \widehat { \mathrm { s e } }$ , and $\begin{array} { r } { \widehat { \mathrm { s e } } = \frac { s } { \sqrt { m } } } \end{array}$ , where ba and s denote the sample mean and sample standard deviation of the outcomes, respectively. At $m = 1 0 ^ { 5 } , t _ { m - 1 , 0 . 9 7 5 } \approx 1 . 9 6$

GSM8K: variation across prompts. Figure 2a reports the TV error at $\beta = 0 . 1$ for eight randomly selected GSM8K problems. The difficulty of approximating the target varies substantially across prompts. For some prompts, the methods reach the Monte Carlo floor within a small sampling budget, leaving little resolution for distinguishing them at larger N. For others, substantial separation remains throughout the tested range. Across the displayed prompts, UR achieves estimated TV error below or comparable to SIR and budget-calibrated RS, up to Monte Carlo resolution, consistent with Theorems 2 and 3. UR remains competitive with envelope RS without requiring threshold information.

GSM8K: variation across β. Figure 2b fixes Problem #201 and varies $\beta \in \mathsf { \Omega }$ {0.01, 0.05, 0.1, 0.125, 0.25, 0.5, 1, 2}. Changing $\beta$ changes the concentration of the rewardtilted target and therefore the finite-budget difficulty of the sampling problem. At smaller $\beta ,$ , the target is more strongly tilted toward high-reward responses, whereas larger $\beta$ makes $\pi _ { \beta }$ closer to the reference proposal ${ \widehat { \mu } } .$ Accordingly, the sampling difficulty varies substantially with $\beta .$ . The same qualitative comparison persists across the displayed values: UR remains competitive with other baselines without requiring threshold information.

![](images/c51d221975739a1099ff1db78b2585edeae4fc200f7abcebb67cc1ca5b2a8cac.jpg)

![](images/0a2995654c109d2763a2a6f61d5b4d944996e451040af8021c53e44382718c5a.jpg)

![](images/a120302cea0a3361fb341edac724e76811883445a80d3dd4612de83acbe9052c.jpg)

![](images/a733957abe208604e61a2441003cf56649e61be3101bfad4f034fa18c5d12e3b.jpg)

![](images/ae938305095710f5a1754416aae8c6fdb74928901ea88ff0051a3e680bd4377d.jpg)

![](images/07083cf0ab35f1422a57d0fa9152ae9f362c4fe940591464c85302110eefbaa4.jpg)

![](images/61efcbc7ef33da8d16af79a70299e2f474e61fea4bb041e522a3af5f7f3d84fc.jpg)

![](images/ce4e379326c6d9b6b25e5144e1dcef7842a085fcfb616c089bb4d8b1a8f309f1.jpg)  
SIR envelope RS budget-calibrated RS UR exact-sampler floor

(a) β = 0.1 for eight randomly selected problems.  
![](images/3d515fd7e4ab052b6406aa991a91f57a59642b6d34677687b49e473a1bf0aeb3.jpg)

![](images/8da708b5e8f1c609808b1c8ff142afb926ea00bdb5080a2bd7d58d3fd6402da6.jpg)

![](images/15d88f71ee9e6da4e8c12ce9fc904cf892abdfcb028d402236e64576c6dc2181.jpg)

![](images/4e1713af354b28aedba51a0136fcc7497785d45ab4e9c8e43153ff8b28b2f6cb.jpg)

![](images/937018edba32f7e604a816ca29b40c7ac982d67b9ff1d57a82f054c0c3e2a0ed.jpg)

![](images/cb8a2afcf48c43e41788714b2f5346962f16ca00e82b26f2953d9d3223e62720.jpg)

![](images/71fe903da4422ad78b10650bb22516f90c4c7d7d6f1e0069a4b6f992a616b282.jpg)  
SIR envelope RS budget-calibrated RS UR exact-sampler floo

![](images/c83eddba19cac72b2981743e8488c1e43e370cdfd4854974dba2b09513add187.jpg)  
(b) Problem #201 across eight values of β from 0.01 to 2.

Figure 2: TV error $\widehat { D } _ { \mathrm { T V } } ( P _ { N } , \pi _ { \beta } )$ w.r.t sampling budget N on GSM8K. The dotted line indicates the exact-sampler Monte Carlo floor.

![](images/bfe7a2f90d50b554c92159e0bc09dd865a119ea420320b5ea264d27789998d7c.jpg)

![](images/83f19e4887dcd9519238766f69787ca96c80bd26cdc055ee2a6d2589e9b079da.jpg)

![](images/db680b71a8879a1fc9952d61d6f6e10be388fe857c107dd23d87d9fc806daa6c.jpg)

![](images/07ea31a6731f69b3fdc4885719c9007e76e04564f04bda6e705b5335519b89bb.jpg)

![](images/08a8b0456d7cd8942c0f69fba5a75cd01e6fa0867332edb6f54c8449288d49df.jpg)

![](images/e35d8c3e09e31e0383f7f9a8f58f18218659a229c550b183281433c03d1d28fb.jpg)

![](images/f9887c1a682cd97a106b745505587d04aa9fed2ba5c33b44588c45cad7b9869b.jpg)

![](images/837e0d7a480a001f47f7111ba6107bb79eef3f3bc80441ec729919381fbddf43.jpg)

(a) β = 0.1 for eight randomly selected problems.  
![](images/989b4189693c2ecf19396035e3444a023d889f0542e8ed862c67ab5a2f64a723.jpg)

![](images/ee2758e888c19efcb9058e3620db691e28e8ddbfaf352bc2ae5ad095aa8bae76.jpg)

![](images/79c9fbf36a258202ccc1119dbb123d1a641be88e1e5b2b71a4eebf6e215b1f34.jpg)

![](images/07d58675bbf9c45c85978c63281e4a37221ba9011813f2e22a827b1d78892100.jpg)

![](images/f3d05adf76f9af88902420d4c8fae6ac86ac9610dc5937a9cf93cbb974848d6b.jpg)

![](images/0f4434d65f7577a50e908b409ea2c6e7b5f626d87382062b8dea844833e55249.jpg)

![](images/6af90dabbea9c75cd4f396a7f402c9950cddf938afd358c5fb9e6c6b51a9ebf7.jpg)

![](images/80459ee4f1051f71c37cf1b1caa5624881dd4171a600c24d4bb28b6ec2544f23.jpg)  
(b) Problem #258 for β from 0.01 to 2.  
Figure 3: Ground-truth accuracy w.r.t. sampling budget N on GSM8K.

![](images/149befa37869bd90ec8581445169a34ed8318f445ecbcc084a1698b9ddc3b514.jpg)

![](images/f13d174eeb352429393ea24c7721c27b63fa41122ea9ce04d9ac755458fceaa8.jpg)

![](images/7006bb5861e3bd210bfbf5a8ab36624dc866c20827daa85b4c2e9144e2f91f90.jpg)

![](images/45daba81922ab8468822c8527c2d9ce11a9e924cf6a68066ec19e645cd717913.jpg)

![](images/b7c958bab578cd7d2e0c32d3a7dce0a6348ccb5fd2ef5dd1b2bec4aa63344ff8.jpg)

![](images/8dec0766c15d32b76ce9fddf751bf9a7fc72563feba92e04b1f3ed60b7d27695.jpg)

![](images/880ffdbf614ce011ebb1517974a71b924fb6a9f2cfaa7c6cf80e620dc325e67c.jpg)

![](images/4e8129e35a9d515edc67c6d492bc7c3654c411ff0811aae647705f24fc7c4853.jpg)  
SIR envelope RS budget-calibrated RS UR exact-sampler floor

(a) β = 0.1 for eight randomly selected problems.  
![](images/05b65418ccb48de32c7e7d6bbe807735f3e5c4814327537e5fc875534a07be21.jpg)

![](images/fa693c5aa4d903db137369fae70f5331daa20e8e751b29754ae2aab122c6eb7e.jpg)

![](images/71af23e5fbb2f25ed2b09914d056bf7a622ba6349f12c4f3338758917c911d8c.jpg)

![](images/56049b4ab323d4e84a39f80f3e410d54d561e66affd9ef3bfe213e4dca7939fb.jpg)

![](images/efb30787e9a842996a4c56cc361b9af0d03d16e17f1809aaa8901d85d2df9370.jpg)

![](images/6a6fa3f74ebec4b9e2f05d3d1b18ef6fffd0a1c52f913171dd793d3e2ced78d6.jpg)

![](images/8efd831348d685e8a425bb9963b7c4dee9c157db388ad8524ec77abf864b0226.jpg)  
SIR envelope RS budget-calibrated RS UR exact-sampler floo

![](images/9ebf94627a27f614546bf8299c3582a0619859048073db0530f15406355b639a.jpg)  
(b) Problem #143 for β from 0.01 to 2.

Figure 4: TV error $\widehat { D } _ { \mathrm { T V } } ( P _ { N } , \pi _ { \beta } )$ w.r.t sampling budget N on MATH500. The dotted line indicates the exact-sampler Monte Carlo floor.

![](images/c59742f2ea53f361ff6e76593a1e522fbc0e7ee6b30a75c250a5b038a2c89405.jpg)

![](images/a6f599ecd2345fc4b9c6d37d7008350165d15cb8fbc640b6946a31cafb79bde2.jpg)

![](images/86f15e95afca377564ae190f3b0fb1f8fcaa179820fda4e7c62532a1348f5451.jpg)

![](images/5656b4c7656bca307359ec3830598e04cafed21bcb896e1f70b311d64f5733ef.jpg)

![](images/7665e120acd318819ad105af198c45ff067c1b44362a03c7a25bace169dbd7f4.jpg)

![](images/14fa81da95da546bb2123899b7bd8859a66fcdf9a0541787c733a0963d341efc.jpg)

![](images/c5c08336e1d733c87a801c99c457169b15e05bb3c81dbef62a7ec5557f581c34.jpg)

![](images/2d06198e423dad9df3f97e8a40d16d269aff3ed33c337dd881fef9a22dcf9f13.jpg)

(a) β = 0.1 for eight randomly selected problems.  
![](images/a40779cde26edfee12aaeaf1f983fbb57b74c09acc5cf82489181af378db137d.jpg)

![](images/b502e5ee83c6518a654824dcf8c3c6c7b25780cbfa2645dc8faad94c74661f29.jpg)

![](images/0b4503716ae04b666183006799956b736fd9cb4f894cf53f6a9ba3e211dac6c4.jpg)

![](images/1a1b0b6dd43ca43905e8b68080691fd29caefedee218f69ac922abe1f88a7686.jpg)

![](images/3cfeb95044a47b845dc42a7f572e5f11bc08c2427d550c81284bc7058ed316aa.jpg)

![](images/1f7e558d29bed791c0f07c0a68849f84afcbc92d573fa3f97cac756234c3922c.jpg)

![](images/4e6c881f9c3b98f38b7ba22f2bdbe31a19d7e23be7a3c8a94d45010ae9b53794.jpg)

![](images/1f83b14c8be66a60a544b844d5c96faa17c7a63799cf6e01c4762748d707174b.jpg)  
(b) Problem #128 for β from 0.01 to 2.  
Figure 5: Accuracy w.r.t sampling budget N on MATH500.

GSM8K: Ground-truth accuracy. Figure 3 reports ground-truth accuracy across eight prompts at $\beta = 0 . 1$ and, for Problem #258, across $\beta \in \{ 0 . 0 1 , 0 . 0 5 , 0 . 1 , 0 . 1 2 5 , 0 . 2 5 , 0 . 5 , 1 , 2 \}$ . Distributional closeness to $\pi _ { \beta }$ translates directly into closeness in ground-truth accuracy, since correctness corresponds to an event. However, the target accuracy may be either above or below the reference policy accuracy. Accordingly, as a sampler approaches $\pi _ { \beta } .$ , its accuracy may either increase or decrease toward the corresponding target accuracy. For this reason, we use TV error as the primary metric for comparing the sampling procedures and accuracy as a complementary metric.

MATH500. Figures 4 and 5 report the MATH500 results across eight problems at $\beta = 0 . 1$ and across values of β for selected problems. Across the displayed settings, the qualitative behavior observed on GSM8K also appears on MATH500, indicating that the empirical comparisons carry over to these more challenging mathematical reasoning problems.