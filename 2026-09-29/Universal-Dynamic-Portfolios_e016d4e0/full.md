# Universal Dynamic Portfolios

Yu-Jie Zhang University of Washington

Yu-Xiang Wang University of California, San Diego

yujiez7@cs.washington.edu

yuxiangw@ucsd.edu

Peng Zhao zhaop@la State Key Laboratory for Novel Software Technology, Nanjing University School of Artificial Intelligence, Nanjing University

Kevin Jamieson University of Washington

mda.nju.edu.cn

jamieson@cs.washington.edu

## Abstract

Cover’s Universal Portfolio (Cover, 1991) matches the performance of the best constant rebalanced portfolio in hindsight. We generalize this framework to compete with an arbitrary comparator sequence $\mathbf { u } _ { 1 } , \ldots , \mathbf { u } _ { T }$ , leading to a dynamic regret minimization problem for the log loss where existing methods break down due to potentially unbounded gradients. The log loss is exp-concave, a curvature property that classically yields fast rates for static regret, yet we show that this advantage generally disappears in the dynamic setting. In particular, a linear-loss-type $\sqrt { T P _ { T } }$ dependence is unavoidable, where $\begin{array} { r } { P _ { T } = \sum _ { t = 2 } ^ { T } \lVert \mathbf { u } _ { t } - \mathbf { u } _ { t - 1 } \rVert } \end{array}$ <sub>1</sub> is the standard path length. This limitation stems from the coarse nature of $P _ { T }$ , which obscures finer spatial and temporal structure of the comparator sequence. We therefore introduce two structure-aware measures—the Jensen–Shannon distance for spatial structure and the JS<sup>q</sup>-path length for temporal structure—under which faster rates are attainable when the comparator sequence has favorable structure. To achieve sharp bounds for both measures simultaneously, we develop Universal Dynamic Portfolio, a parameter-free method that combines a new Dirichlet Hedge algorithm with a fixed-share update, while retaining a near-optimal $P _ { T }$ guarantee in the worst case. Finally, under an additional bounded-gradient assumption, we show that OPS admits the faster $T ^ { 1 / 3 } P _ { T } ^ { 2 / 3 }$ dynamic regret rate over all comparator sequences. We attain this rate with a tractable proper algorithm that applies more broadly to general online exp-concave optimization over arbitrary compact convex domains.

## 1. Introduction

Online portfolio selection (OPS), which studies how to sequentially allocate wealth among a set of assets to maximize cumulative returns, is a textbook motivating example in online learning (Hazan, 2016). Cover (1991) introduced the problem as a distribution-free model for sequential investment and proposed the seminal

Universal Portfolio algorithm. Since then, OPS has attracted substantial interest from the online learning community because it can be formulated as an online convex optimization problem with Cover’s logarithmic loss. The rich curvature of this loss can be exploited to obtain fast learning guarantees, making OPS one of the central testbeds for understanding the role of loss curvature in both algorithm design and regret analysis (Agarwal et al., 2006; Van Erven et al., 2020; Luo et al., 2018; Zimmert et al., 2022; Mhammedi and Rakhlin, 2022; J´ez´equel et al., 2025).

Most existing OPS studies focus on static regret, which compares the learner’s cumulative loss with that of the best single portfolio in hindsight. In OPS, this benchmark is known as the best constant rebalanced portfolio (CRP), which allocates wealth according to fixed proportions. Many methods, including the classical Universal Portfolio algorithm, achieve the minimax-optimal $\mathcal { O } ( d \log T )$ static regret (Cover, 1991; Ordentlich and Cover, 1998). In its dependence on $T$ , this logarithmic rate improves upon the $\sqrt { T }$ rate typical of general convex losses with bounded gradients. However, in a continuously evolving and possibly adversarial market, such a constant comparator may be restrictive because it cannot adapt to market changes. This motivates us to extend the OPS framework to allow time-varying comparators.

This non-stationary extension is naturally captured by the notion of dynamic regret in online convex optimization. To formalize this objective, let ${ \ell _ { t } } ( { \mathbf w } ) = - \ln ( { \mathbf w } ^ { { \top } } { \mathbf x } _ { t } )$ denote Cover’s loss, where $\mathbf { w } \in \Delta _ { d }$ is the learner’s portfolio and $\mathbf { x } _ { t } \in \mathbb { R } _ { + } ^ { d }$ is the market return. The dynamic regret (Herbster and Warmuth, 2001; Zinkevich, 2003; Zhang et al., 2018) measures the gap between the learner’s cumulative loss and that of a time-varying comparator sequence $\{ { \mathbf { u } } _ { t } \} _ { t = 1 } ^ { T }$ by

$$
\mathrm { D - R e g } _ { T } ( \{ \mathbf { u } _ { t } \} _ { t = 1 } ^ { T } ) = \sum _ { t = 1 } ^ { T } \ell _ { t } ( \mathbf { w } _ { t } ) - \sum _ { t = 1 } ^ { T } \ell _ { t } ( \mathbf { u } _ { t } ) .\tag{1}
$$

The above measure is often referred to as universal dynamic regret because it seeks guarantees that hold uniformly over all comparator sequences and adapt to their complexity. Dynamic regret reduces to the classical notion of (static) regret by setting $\begin{array} { r } { \mathbf { u } _ { t } = \mathbf { w } _ { * } = \arg \operatorname* { m i n } _ { \mathbf { w } \in \Delta _ { d } } \sum _ { t = 1 } ^ { T } \ell _ { t } ( \mathbf { w } ) } \end{array}$ . Meanwhile, it ofers greater flexibility by allowing the comparator sequence $\mathbf { u } _ { t }$ to adapt to the underlying environment, rather than being tied to a single realized return. For instance, when the market return $\mathbf { x } _ { t }$ is sampled from a time-varying distribution $\mathcal { D } _ { t }$ , a natural choice is $\begin{array} { r } { \mathbf { u } _ { t } = \arg \operatorname* { m i n } _ { \mathbf { w } \in \Delta _ { d } } \mathbb { E } _ { \mathbf { x } _ { t } \sim \mathcal { D } _ { t } } [ \ell _ { t } ( \mathbf { w } ) ] } \end{array}$ Compared with the minimizer $\mathbf { w } _ { t } ^ { \star } = \arg \operatorname* { m i n } _ { \mathbf { w } \in \Delta _ { d } } \ell _ { t } ( \mathbf { w } )$ at each round, this avoids chasing noise from a single observation.

## 1.1. Related Work and Research Question

Although non-stationary online learning has been extensively studied over the past decades (Herbster and Warmuth, 1998; Hazan and Seshadhri, 2009; Cesa-Bianchi et al., 2012; Gy¨orgy and Szepesv´ari, 2016; Zhang et al., 2018; Zhao et al., 2020, 2021; Wei and Luo, 2021; Zhang et al., 2023; Qian et al., 2024; Zhao et al., 2024, 2025;

Jacobsen et al., 2025), dynamic regret for OPS remains surprisingly underexplored. The only closely related result is due to Singer (1997), who proposed a method that competes with comparators switching among N fixed portfolios. This guarantee covers only a restricted form of dynamic regret because $\mathbf { u } _ { t }$ is confined to a finite set and cannot evolve continuously over time.

The most well-developed results on dynamic regret minimization in the online convex optimization literature measure nonstationarity through the path length

$$
P _ { T } = \sum _ { t = 2 } ^ { T } \lVert \mathbf { u } _ { t } - \mathbf { u } _ { t - 1 } \rVert _ { 1 } ,
$$

which quantifies the cumulative variation of the comparator sequence. For general convex losses with bounded gradients, Ader (Zhang et al., 2018) achieves the $\mathcal { O } ( G \sqrt { T ( 1 + P _ { T } ) } )$ dynamic-regret guarantee, where G is the upper bound on the gradient norm. Since Cover’s logarithmic loss can have unbounded gradients, this result does not apply directly to OPS, leaving unresolved even the attainability of the canonical $\sqrt { T P _ { T } } \mathrm { - t y p e }$ dynamic-regret guarantee without a gradient bound.

More importantly, Cover’s logarithmic loss is known to be exp-concave, a curvature property that yields logarithmic regret against a static benchmark. This rate is substantially faster than the $\sqrt { T } \mathrm { - t y p e }$ static regret typical of general convex losses, motivating us to seek an analogous acceleration for dynamic comparators. The literature ofers a partial clue. Under bounded gradients, Baby and Wang (2021) and Zhang et al. (2025) establish that exp-concavity is also beneficial in the dynamic setting, improving the dynamic regret to $T ^ { 1 / 3 } P _ { T } ^ { 2 / 3 }$ . Neither result, however, resolves the case of OPS. The bounded-gradient condition is restrictive for Cover’s loss, and even under this condition, the algorithms are not compatible with the geometry of portfolio selection: the method of Baby and Wang (2021) may predict outside the simplex, whereas that of Zhang et al. (2025) requires a projection over distributions that is computationally prohibitive. Taken together, these gaps lead us to ask:

## What is the achievable dynamic regret rate for OPS?

In fact, we believe resolving this question would also clarify the role of loss curvature in dynamic regret minimization for non-stationary online learning.

## 1.2. Our Results

In this paper, we develop algorithms and matching lower bounds that characterize the dynamic regret achievable for OPS against arbitrary comparator sequences. We first settle the minimax rate under the classical $\boldsymbol { L } _ { \mathrm { 1 ^ { - } } \mathrm { p a t h } }$ length, showing that the fast rates typical of exp-concave losses are unattainable. This limitation arises because $L _ { 1 }$ -path length ignores where the movement occurs and how it evolves over time, even though both can substantially afect tracking dificulty. This motivates refined guarantees that adapt to the spatial and temporal structure of the comparator sequence. Our main results are summarized in Table 1 and detailed below.

Table 1: Summary of dynamic regret bounds under diferent complexity measures. Here, $J _ { t } = \sqrt { ( \mathrm { K L } ( \mathbf { u } _ { t } \| \bar { \mathbf { u } } _ { t } ) + \mathrm { K L } ( \mathbf { u } _ { t - 1 } \| \bar { \mathbf { u } } _ { t } ) ) / 2 }$ denotes the Jensen–Shannon (JS) distance between consecutive comparators, where $\bar { \mathbf { u } } _ { t } = ( \mathbf { u } _ { t - 1 } + \mathbf { u } _ { t } ) / 2$ is their midpoint. The notation $\widetilde { \mathcal { O } } ( \cdot )$ hides logarithmic factors.
<table><tr><td>Measure</td><td>Upper bounds</td><td>Lower bounds</td></tr><tr><td> $L _ { \mathrm { 1 ^ { - p a t h } } } \ \mathrm { l e n g t h }$   $\begin{array} { r } { P _ { T } = \sum _ { t = 2 } ^ { T } \| \mathbf { u } _ { t } - \mathbf { u } _ { t - 1 } \| _ { 1 } } \end{array}$ </td><td> $\widetilde { \mathcal { O } } \big ( d + \sqrt { d T P _ { T } } \big )$  (Section 3.1)</td><td> $\Omega \big ( \operatorname* { m a x } \{ d \log T , \operatorname* { m i n } \{ T , \sqrt { d T P _ { T } } \} \} \big )$  (Section 3.1)</td></tr><tr><td> $\mathrm { J S - p a t h \ l e n g t h }$   $\begin{array} { r } { P _ { T } ^ { \mathrm { J S } } = \sum _ { t = 2 } ^ { T } J _ { t } } \end{array}$ </td><td> $\mathcal { \widetilde { O } } \big ( d ( 1 + T ^ { \frac { 1 } { 3 } } \big ( P _ { T } ^ { \mathrm { J S } } \big ) ^ { \frac { 2 } { 3 } } \big ) \big )$  (Section 3.2)</td><td> $\Omega \big ( \operatorname* { m a x } \{ d \log T , \operatorname* { m i n } \{ T , d ^ { \frac { 2 } { 3 } } T ^ { \frac { 1 } { 3 } } ( P _ { T } ^ { \mathrm { J S } } ) ^ { \frac { 2 } { 3 } } \} \} \big )$  (Section 3.2)</td></tr><tr><td>JSª-path length  $\begin{array} { r } { P _ { T , q } ^ { \mathrm { J S } } = \sum _ { t = 2 } ^ { T } J _ { t } ^ { q } , q \in [ 0 , 1 ] } \end{array}$ </td><td> $\mathcal { \widetilde { O } } \big ( d ( 1 + T ^ { \frac { q } { q + 2 } } ( P _ { T , q } ^ { \mathrm { J S } } ) ^ { \frac { 2 } { q + 2 } } ) \big )$  (Section 3.3)</td><td></td></tr></table>

• Minimax Rate under the Standard Path Length. We establish an Ω  max $\left\{ d \log T , \sqrt { d T P _ { T } } \right\}$ lower bound for OPS. This rules out the favorable $T ^ { 1 / 3 } P _ { T } ^ { 2 / 3 }$ dependence uniformly over arbitrary comparator sequences, despite the exp-concavity of Cover’s loss. We complement this lower bound with a black-box reduction from interval regret to dynamic regret, showing that any algorithm with an $\mathcal { O } ( d \log T )$ interval regret guarantee achieves O  max{d log $T , \sqrt { d T P _ { T } \log T } \} ,$  dynamic regret,thereby matching the lower bound up to logarithmic factors.

• Spatial Adaptivity through the Jensen-Shannon Distance. We provide an algorithm that adapts to the spatial structure of the comparator sequence and achieves $\mathcal { \tilde { O } } ( d ( 1 + \tilde { T } ^ { 1 / 3 } ( P _ { T } ^ { \mathrm { J S } } ) ^ { 2 / 3 } ) )$ dynamic regret. Here, $\begin{array} { r } { P _ { T } ^ { \mathrm { J S } } = \bar { \sum } _ { t = 2 } ^ { T } J _ { t } } \end{array}$ is the path length based on the Jensen–Shannon (JS) distance, where $J _ { t } =$ $\sqrt { ( \mathrm { K L } ( \mathbf { u } _ { t } \| \bar { \mathbf { u } } _ { t } ) + \mathrm { K L } ( \mathbf { u } _ { t - 1 } \| \bar { \mathbf { u } } _ { t } ) ) / 2 }$ and $\bar { \mathbf { u } } _ { t } = ( \mathbf { u } _ { t - 1 } + \mathbf { u } _ { t } ) / 2$ . The resulting bound yields a faster rate for interior comparators while recovering the worst-case $T ^ { 1 / 2 } P _ { T } ^ { 1 / 2 }$ dependence for arbitrary comparator sequences. A corresponding lower bound matches its dependence on $T$ and $P _ { T } ^ { \mathrm { J S } }$ , up to logarithmic factors.

• Temporal Adaptivity through JS<sup>q</sup>-Path Length. We further show that comparator sequences with the same JS-path length can difer in tracking dificulty because of their temporal structure. We capture this structure using the JS<sup>q</sup>-path length $\begin{array} { r } { P _ { T , q } ^ { \mathrm { J S } } = \sum _ { t = 2 } ^ { \hat { T } } J _ { t } ^ { q } } \end{array}$ for $q \in [ 0 , 1 ]$ , with the endpoint convention $J _ { t } ^ { 0 } = \mathbb { 1 } \{ J _ { t } > 0 \}$ , and establish a $\mid \widetilde { \mathcal { O } } ( d + d T ^ { q / ( q + 2 ) } ( P _ { T , q } ^ { \mathrm { J S } } ) ^ { 2 / ( q + 2 ) } )$ dynamic regret bound that holds simultaneously for all $q \in [ 0 , 1 ]$ . The endpoint $q = 1$ recovers the $T ^ { 1 / 3 } ( P _ { T } ^ { \mathrm { J S } } ) ^ { 2 / 3 }$ rate, whereas $q = 0$ yields $\mathscr { O } ( d ( 1 + \mathsf { S } _ { T } ) \log ( d T ) )$ , where $\mathsf { S } _ { T }$ is the number of comparator switches. Intermediate values of q interpolate between these endpoints, allowing the regret bound to adapt more finely to the temporal structure of comparator variation.

We achieve all of the above upper bounds with a single algorithm by equipping Cover’s Universal Portfolio algorithm with fixed-share updates. Despite the simplicity of this modification, proving these guarantees requires a novel mixability-based analysis that uses a Dirichlet comparator to accommodate both the unbounded log loss and the simplex constraint. Section 3.4 outlines the main technical ideas.

Toward General OXO. Under an additional bounded-gradient assumption, we show that a fast rate of $\widetilde { \mathcal { O } } \big ( d G ( 1 + T ^ { 1 / 3 } P _ { T } ^ { 2 / 3 } ) \big )$ is attainable for all comparator sequences. Existing methods achieving this rate either require improper learning or lack a computationally tractable implementation (Baby and Wang, 2021; Zhang et al., 2025), whereas our method is both proper and computationally tractable. Beyond OPS, our approach extends to general online exp-concave optimization over arbitrary compact convex domains, providing a tractable afirmative answer to the question raised by Baby and Wang (2021) of whether strongly adaptive methods can achieve optimal dynamic regret in the proper learning setting. We establish this guarantee via a new two-layer mixability argument, which is detailed in Section 4.2.

Organization. The rest of the paper is organized as follows. Section 2 introduces the problem setup and additional related work. Section 3 presents minimax-optimal regret bounds and spatially and temporally adaptive guarantees for OPS without a gradient bound. Section 4 develops a computationally tractable proper method for general OXO under bounded gradients. Finally, Section 5 concludes the paper.

## 2. Problem Setup and Related Work

## 2.1. Notation and Setup

For a positive integer n, let $[ n ] = \{ 1 , \dots , n \}$ . We write $\mathbb { R } _ { + } ^ { d }$ for the nonnegative orthant and $\begin{array} { r } { \Delta _ { d } = \left\{ \mathbf { w } \in \mathbb { R } _ { + } ^ { d } : \sum _ { i = 1 } ^ { d } w _ { i } = 1 \right\} } \end{array}$ for the $( d - 1 )$ -dimensional probability simplex.

Online portfolio selection proceeds over T rounds of interaction between the learner and the market. At each round $t \in [ T ]$ , the learner starts with wealth $S _ { t - 1 }$ and distributes it across d assets according to a probability vector $\mathbf { w } _ { t } \in \Delta _ { d }$ . The market then reveals the non-negative price relative vector $\mathbf { x } _ { t } \in \mathbb { R } _ { + } ^ { d }$ , where each component $x _ { t , i } \geq 0$ represents the relative return of asset i. The learner’s wealth is updated as $\begin{array} { r } { S _ { t } = S _ { t - 1 } \sum _ { i = 1 } ^ { d } w _ { t , i } \cdot x _ { t , i } } \end{array}$ . After T rounds, the learner’s wealth is $\begin{array} { r } { S _ { T } = S _ { 0 } \cdot \prod _ { t = 1 } ^ { T } ( \mathbf { w } _ { t } ^ { \top } \mathbf { x } _ { t } ) } \end{array}$ For OPS in non-stationary environments, our goal is to minimize the dynamic regret (1) with the log loss ${ \ell _ { t } } ( { \mathbf w } ) = - \ln ( { \mathbf w } ^ { { \top } } { \mathbf x } _ { t } )$

$$
\sum _ { t = 1 } ^ { T } \ell _ { t } ( \mathbf { w } _ { t } ) - \sum _ { t = 1 } ^ { T } \ell _ { t } ( \mathbf { u } _ { t } ) = - \ln \frac { \prod _ { t = 1 } ^ { T } \mathbf { w } _ { t } ^ { \top } \mathbf { x } _ { t } } { \prod _ { t = 1 } ^ { T } \mathbf { u } _ { t } ^ { \top } \mathbf { x } _ { t } } ,
$$

which is equivalent to maximizing the logarithmic ratio of the learner’s cumulative wealth to that of the time-varying investment strategy $\mathbf { u } _ { 1 } , \ldots , \mathbf { u } _ { T } \in \Delta _ { d }$

For the OPS results in Section 3, we only assume $\mathrm { m a x } _ { i \in [ d ] } x _ { t , i } > 0$ for every $t \in [ T ]$ which merely excludes degenerate rounds where every portfolio incurs infinite loss. This minimal assumption is what makes the problem technically challenging: the log-loss gradients $\nabla \ell _ { t } ( \mathbf { w } ) = - \mathbf { x } _ { t } / ( \mathbf { w } ^ { \top } \mathbf { x } _ { t } )$ can be unbounded, since the denominator $\mathbf { w } ^ { \top } \mathbf { x } _ { t }$ may approach zero near the boundary. Standard online convex optimization techniques rely on a uniform gradient bound, which is available only under further restrictions such as bounded return ratios or a domain clipped away from the simplex boundary (Helmbold et al., 1998; Agarwal et al., 2006). Neither restriction is imposed for these results; Section 4 separately considers bounded gradients.

## 2.2. Related Work

Static Regret for OPS. Under the nonzero-return condition stated above, Universal Portfolio (Cover and Ordentlich, 1996) achieves the minimax-optimal $\mathcal { O } ( d \log T )$ static regret. The method requires integrating over the simplex to generate predictions, which can be implemented in polynomial time using log-concave sampling techniques (Kalai and Vempala, 2002). More recent work develops more eficient algorithms under the same condition (Orseau et al., 2017; Luo et al., 2018; Zimmert et al., 2022; Mhammedi and Rakhlin, 2022; J´ez´equel et al., 2025). Two Pareto-optimal results in terms of regret and computational eficiency are VB-FTRL (J´ez´equel et al., 2025), which achieves $\mathcal { O } ( d \log T )$ regret with $\mathcal { O } ( d ^ { 2 } T )$ computational cost per round, and AdaMix+DONS (Mhammedi and Rakhlin, 2022), which attains an $\mathcal { O } ( d ^ { 2 } \log ^ { 5 } T )$ regret bound with $\mathcal { O } ( d ^ { 3 } \log ^ { 2 } T )$ per-round complexity. With bounded gradients, Exponential Gradient (Helmbold et al., 1998) achieves $\mathcal { O } ( G \sqrt { T } )$ regret, while Online Newton Step (Agarwal et al., 2006) attains O(dG log T) regret with $\mathcal { O } ( d ^ { 3 } )$ cost per round.

Dynamic Regret for Curved Losses. There are two lines of research that achieve an $\mathcal { O } ( G \operatorname* { m a x } \{ \log T , T ^ { 1 / 3 } P _ { T } ^ { 2 / 3 } \} )$ dynamic regret for non-stationary OXO (Baby and Wang, 2021; Zhang et al., 2025). Since these methods are primarily designed for general OXO purposes, their regret bounds typically scale with a bound on the gradient norm. However, setting aside the gradient-bound issue, these results still do not directly apply to OPS due to the restrictions imposed by domain constraints.

• Reduction-based analysis. An important research line for non-stationary OXO starts from Baby and Wang (2021) and is followed by Baby and Wang (2022a,b). Under certain domain conditions, they provide a reduction from the interval regret bound (Hazan and Seshadhri, 2009), which guarantees a static regret bound on each interval, to a fast-rate dynamic regret bound. A key component of their analysis is a precise characterization of the optimal time-varying sequence $\{ \mathbf { u } _ { t } ^ { * } \} _ { t = 1 } ^ { T }$ via KKT conditions and shows that $\{ \mathbf { u } _ { t } ^ { * } \} _ { t = 1 } ^ { T }$ can be tracked by a piecewisestationary sequence with $M = \mathcal { O } ( T ^ { 1 / 3 } P _ { T } ^ { 2 / 3 } )$ switches. The initial work (Baby and Wang, 2021) requires improper learning, allowing the algorithm to predict in an extended box-constrained domain in order to obtain a suficiently strong piecewise-stationary approximation. Later, Baby and Wang (2022a) show that proper learning can be achieved when the domain is exactly a box. This restriction to box constraints is intrinsic to the KKT-based analysis, as it only imposes coordinate-wise constraints on the optimal sequence. It remains unclear how to extend these analyses to the simplex or more general domains, which would introduce additional coupling constraints and complicate the analysis.

• Mixability-based analysis. Recently, Zhang et al. (2025) showed that continuous exponential weights with a fixed-share update achieve fast-rate dynamic regret via mixability (Vovk, 1998), which lifts the analysis from pointwise predictors to distributional comparators. While this framework provides additional flexibility, its analysis relies on Gaussian comparators, which are incompatible with the simplex constraint. To enforce the domain constraint, the method requires an information projection onto a set of Gaussian mixture models with potentially infinitely many components, with bounded component means and variances, which makes the procedure computationally intractable. Our work instead uses Dirichlet comparators, which naturally respect the simplex and allow the analysis to accommodate unbounded log-loss gradients. Further details are provided in Section 3.2.

## 3. Dynamic Regret for Online Portfolio Selection

This section characterizes the achievable dynamic regret rates for OPS. We first establish the minimax-optimal rate in terms of the commonly used norm-based path length. Our minimax analysis reveals that the standard path length can obscure fine-grained spatial and temporal diferences among comparator sequences. Building on this insight, we establish guarantees in terms of the Jensen–Shannon distance that adapt to the spatial structure of comparator movements, together with a family of q-order guarantees that further adapt to their temporal distribution.

## 3.1. Minimax Rate under the Standard Path Length

We begin by establishing a lower bound that captures the worst-case dificulty of OPS over the full range of the standard path-length budget.

Theorem 1 Consider the $O P S$ problem with $d \geq 2$ assets and $T > 2 d$ . For any online algorithm and any $C \in [ 0 , T ]$ , there exists a comparator sequence $\mathbf { u } _ { 1 } , \ldots , \mathbf { u } _ { T } \in \Delta _ { d }$ and ${ \bf x } _ { 1 } , \dots , { \bf x } _ { T } \in \mathbb { R } _ { + } ^ { d }$ such that

$$
P _ { T } \leq C \qquad a n d \qquad \mathrm { D } \mathrm { - } \mathrm { R e g } _ { T } ( \{ \mathbf { u } _ { t } \} _ { t = 1 } ^ { T } ) \geq \Omega \Bigl ( \operatorname* { m a x } \Bigl \{ d \log T , \operatorname* { m i n } \Bigl \{ T , \sqrt { d T C } \Bigr \} \Bigr \} \Bigr ) .
$$

Theorem 1 reveals a sharp contrast between stationary and genuinely non-stationary OPS. When $C = 0$ , our result recovers the standard $\Theta ( d \log T )$ minimax rate for competing with a static comparator. Once the comparator variation becomes nontrivial, the dynamic term scales as $\sqrt { d T C }$ for $C \lesssim T / d _ { \colon }$ , recovering the same $\sqrt { T C }$ dependence as in general convex dynamic regret. Thus, the exp-concavity of Cover’s loss alone does not guarantee the favorable $T ^ { 1 / 3 } C ^ { 2 / 3 }$ dependence uniformly over arbitrary comparator sequences, although such a fast rate is achievable for other curved losses, such as squared loss on bounded domains.

The proof of Theorem 1 is provided in Appendix B.1 via a reduction to the sequential probability assignment (SPA) problem. The static term follows from the classical minimax lower bound for competing with the best constant rebalanced portfolio (Ordentlich and Cover, 1998). For the dynamic term, we restrict attention to the Kelly market with return vectors $\mathbf { x } _ { t } \in \{ \mathbf { e } _ { i } \} _ { i = 1 } ^ { d }$ , where $\mathbf { e } _ { i }$ denotes the i-th standard basis vector in $\mathbb { R } ^ { d }$ . Under this restriction, the OPS problem reduces to multi-class SPA under logarithmic loss (Cesa-Bianchi and Lugosi, 2006, Chapter 9.1). To establish the lower bound, we partition the horizon into K blocks and construct a piecewise-stationary environment. In each block, the optimal comparator lies near the boundary of the simplex: most of its mass is placed on the d-th asset, while each of the first $d - 1$ assets independently receives either a small probability mass or zero. The learner must identify a new set of rare active assets in each block, incurring $\Omega ( d )$ regret per block and hence $\Omega ( d K )$ regret in total. Meanwhile, the near-boundary construction ensures that adjacent blockwise comparators difer by only $\mathcal { O } ( d K / T )$ yielding $P _ { T } = \mathcal { O } ( d K ^ { 2 } / T )$ Choosing $K = \Theta ( \operatorname* { m i n } \{ T / d , \sqrt { T C / d } \} )$ therefore gives $\Omega ( d K ) = \Omega ( \operatorname* { m i n } \{ T , \sqrt { d T C } \} )$ while ensuring $P _ { T } \leq C$

Matching Upper Bound via a Black-Box Reduction. Classical online learning methods, such as online gradient descent, typically assume that gradients are bounded by $G > 0$ . Since gradients in OPS need not be bounded, it was previously unknown whether even the $\sqrt { T P _ { T } }$ dependence could be achieved. We close this gap through a black-box reduction from interval regret, obtaining a G-free dynamic regret guarantee that matches the preceding lower bound up to logarithmic factors.

Lemma 2 For the OPS problem, assume there exists an online algorithm A that, for any interval $\mathcal { T } \subseteq [ T ]$ , attains the interval-regret guarantee

$$
\sum _ { t \in \mathbb { Z } } \ell _ { t } ( \mathbf { w } _ { t } ) - \operatorname* { m i n } _ { \mathbf { w } \in \Delta _ { d } } \sum _ { t \in \mathbb { Z } } \ell _ { t } ( \mathbf { w } ) \leq B ( T ) ,\tag{2}
$$

where $B \colon \mathbb { N } \to ( 0 , \infty )$ is a function. Then, for every comparator sequence $\mathbf { u } _ { 1 } , \dotsc , \mathbf { u } _ { T } \in$ $\Delta _ { d . }$ , the dynamic regret of A satisfies

$$
\mathrm { D - R e g } _ { T } ( \{ \mathbf { u } _ { t } \} _ { t = 1 } ^ { T } ) = \mathcal { O } \Big ( B ( T ) + \sqrt { B ( T ) T P _ { T } } \Big ) ,
$$

where $\begin{array} { r } { P _ { T } = \sum _ { t = 2 } ^ { T } \| \mathbf { u } _ { t } - \mathbf { u } _ { t - 1 } \| _ { 1 } } \end{array}$ is the path length defined in terms of the $L _ { 1 }$ norm.

Lemma 2 provides a black-box reduction from interval regret (Hazan and Seshadhri, 2009) to dynamic regret for the OPS problem. A key advantage of this guarantee is that it does not require a bounded gradient norm, provided that the algorithms are chosen appropriately. For OPS, such algorithms can be constructed with $B ( T ) = \mathcal { O } ( d \log T )$ For example, one may run the FLH (Hazan and Seshadhri, 2009) algorithm with Universal Portfolio (Cover, 1991) or VB-FTRL (J´ez´equel et al., 2025) as the base learner. In this case, the resulting dynamic regret matches the optimal rate up to logarithmic factors. The proof is provided in Appendix B.2.

We note that reduction-based arguments are widely used in the dynamic regret minimization literature (Cutkosky, 2020; Baby and Wang, 2021). The main distinction in our setting is that the loss functions in OPS do not admit a uniform Lipschitz constant. In contrast, existing analyses in OCO typically rely on a bounded gradient norm to relate the instantaneous loss diference to the path length, e.g., $\ell _ { t } ( { \mathbf { u } } ) - \ell _ { t } ( { \mathbf { v } } ) \leq$ $G \| \mathbf { u } - \mathbf { v } \|$ . Nevertheless, we show this issue can be overcome by a direct treatment of the log loss.

## 3.2. Spatial Adaptivity through the Jensen–Shannon Distance

The standard $\boldsymbol { L } _ { \mathrm { 1 ^ { - } } \mathrm { p a t h } }$ lengths or $\scriptstyle L _ { \mathrm { 2 } ^ { - } } { \mathrm { p a t h } }$ lengths quantify the variation of the comparator sequence but are insensitive to their locations inside the simplex. This matters in OPS because the logarithmic loss has nonuniform geometry. Specifically, the hard instance constructed for the lower bound relies on a comparator sequence that stays near the boundary of the simplex, while an analogous sequence in the interior does not exhibit the same hardness. A guarantee based on $P _ { T }$ can only reflect the worst-case dificulty and cannot distinguish easier comparator sequences in the interior. To this end, we introduce a path length based on the Jensen–Shannon distance

$$
P _ { T } ^ { \mathrm { J S } } : = \sum _ { t = 2 } ^ { T } \mathrm { J S } ( \mathbf u _ { t } , \mathbf u _ { t - 1 } ) = \sum _ { t = 2 } ^ { T } \sqrt { \frac { 1 } { 2 } \mathrm { K L } \left( \mathbf { u } _ { t } \parallel \bar { \mathbf { u } } _ { t } \right) + \frac { 1 } { 2 } \mathrm { K L } \left( \mathbf { u } _ { t - 1 } \parallel \bar { \mathbf { u } } _ { t } \right) } ,\tag{3}
$$

where $\bar { \mathbf { u } } _ { t } : = ( \mathbf { u } _ { t } + \mathbf { u } _ { t - 1 } ) / 2$ and $\begin{array} { r } { \mathrm { K L } ( \mathbf { p } \| \mathbf { q } ) : = \sum _ { i = 1 } ^ { d } p _ { i } \log ( p _ { i } / q _ { i } ) } \end{array}$ denotes the Kullback– Leibler divergence. Its coordinate-wise logarithmic ratios align $P _ { T } ^ { \mathrm { J S } }$ with the geometry of the log loss, making it sensitive to the local geometry of each comparator transition.

A Simple and Nearly Optimal Algorithm. Using the JS distance to measure comparator variation, we develop Algorithm 1, whose regret bound adapts to the local geometry of comparator transitions and, in particular, implies a rate faster than $\sqrt { T P _ { T } }$ for interior comparator sequences.

Algorithm 1 is a fixed-share variant of Cover’s Universal Portfolio (Cover, 1991). It maintains a distribution over the simplex and updates it with exponential weights. After each update, the algorithm mixes in a small fraction of the uniform Dirichlet distribution, replenishing probability mass across the simplex and allowing the learner to shift toward newly favorable portfolios as the environment changes. Algorithm 1 enjoys the following dynamic regret guarantee.

Algorithm 1 Universal Dynamic Portfolio   
Input: Fixed-share parameter $\mu _ { t } = 1 / t .$   
1: Initialize $\tilde { P } _ { 1 } = \bar { P _ { 1 } } = \operatorname { D i r } ( \alpha _ { 1 } )$ as a Dirichlet distribution with parameters $\pmb { \alpha } _ { 1 } = \pmb { 1 }$   
2: for $t = 1 , 2 , \dots , T$ do   
3: The learner submits the prediction $\mathbf { w } _ { t } = \mathbb { E } _ { \mathbf { u } \sim P _ { t } } [ \mathbf { u } ]$ and then observes $\mathbf { x } _ { t } \in \mathbb { R } _ { + } ^ { d }$   
4: The learner updates the distributions by   
$\tilde { P } _ { t + 1 } ( \mathbf { u } ) \propto P _ { t } ( \mathbf { u } ) \exp ( - \ell _ { t } ( \mathbf { u } ) ) , \quad \forall \mathbf { u } \in \Delta _ { d } ,$ (4)   
$P _ { t + 1 } ( \mathbf { u } ) = ( 1 - \mu _ { t + 1 } ) \tilde { P } _ { t + 1 } ( \mathbf { u } ) + \mu _ { t + 1 } \mathrm { D i r } ( \pmb { \alpha } _ { 1 } ) .$ (5)   
5: end for

Theorem 3 Algorithm 1 with $\mu _ { t } = 1 / t$ ensures

$$
\begin{array} { r } { \mathrm { D - R e g } _ { T } ( \{ \mathbf { u } _ { t } \} _ { t = 1 } ^ { T } ) \leq \ O ( d \Big ( T ^ { \frac { 1 } { 3 } } \big ( P _ { T } ^ { \mathrm { J S } } \big ) ^ { \frac { 2 } { 3 } } \big ( \ln ( d T ) \big ) ^ { \frac { 2 } { 3 } } + \ln ( d T ) ) ) , } \end{array}
$$

for any comparator sequence $\mathbf { u } _ { 1 } , \ldots , \mathbf { u } _ { T } \in \Delta _ { d }$

Notably, Algorithm 1 achieves this guarantee without parameter tuning or prior knowledge of $P _ { T } ^ { \mathrm { J S } }$ or $T$ . The following lower bound, proved in Appendix C.3, shows that the dependence on $T$ and $P _ { T } ^ { \mathrm { J S } }$ is nearly optimal.

Theorem 4 Consider the OPS problem with $d \geq 2$ assets and $T > 4 d$ . For any online algorithm and any $C \in [ 0 , T ]$ , there exist a comparator sequence $\mathbf { u } _ { 1 } , \ldots , \mathbf { u } _ { T } \in \Delta _ { d }$ satisfying $P _ { T } ^ { \mathrm { J S } } \leq C$ and market vectors ${ \bf x } _ { 1 } , \dots , { \bf x } _ { T } \in \mathbb { R } _ { + } ^ { d }$ such that

$$
\mathrm { D } \cdot \mathrm { R e g } _ { T } ( \left\{ \mathbf { u } _ { t } \right\} _ { t = 1 } ^ { T } ) \geq \Omega \left( \operatorname* { m a x } \left\{ d \log \left( 1 + \frac { T } { d } \right) , \operatorname* { m i n } \left\{ T , d ^ { \frac { 2 } { 3 } } T ^ { \frac { 1 } { 3 } } C ^ { \frac { 2 } { 3 } } \right\} \right\} \right) .
$$

Theorem 3 also yields a refined guarantee in terms of the standard $\boldsymbol { L } _ { \mathrm { 1 ^ { - } p a t h } }$ length $P _ { T }$ , with an explicit dependence on the margin of the comparator sequence from the boundary. Specifically, let $\begin{array} { r } { \alpha ( \mathbf { u } _ { 1 : T } ) : = \operatorname* { m i n } _ { t \in [ T ] , i \in [ d ] } u _ { t , i } > 0 } \end{array}$ denote the minimum margin to the boundary. Since $P _ { T } ^ { \mathrm { J S } } \leq \left( \alpha ( \mathbf { u } _ { 1 : T } ) \right) ^ { - 1 / 2 } P _ { T }$ , Theorem 3 implies

$$
\mathrm { D } \mathbf { - } \mathrm { R e g } _ { T } ( \{ \mathbf { u } _ { t } \} _ { t = 1 } ^ { T } ) \leq \mathcal { O } \left( d \left( \left( \alpha ( \mathbf { u } _ { 1 : T } ) \right) ^ { - \frac { 1 } { 3 } } T ^ { \frac { 1 } { 3 } } P _ { T } ^ { \frac { 2 } { 3 } } ( \ln ( d T ) ) ^ { \frac { 2 } { 3 } } + \ln ( d T ) \right) \right) .
$$

This reveals a form of spatial adaptivity that is invisible to the standard path length alone: for the same $P _ { T }$ , the regret guarantee improves as the comparator sequence moves farther into the interior of the simplex. The resulting $T ^ { 1 / 3 } P _ { T } ^ { 2 / 3 }$ -type dependence for the interior comparators improves over the worst-case $\sqrt { T P _ { T } }$ dependence. The direct $P _ { T } ^ { \mathrm { J S } }$ -based guarantee is sharper still, since it accounts for each comparator variation according to its local position in the simplex rather than through the minimum margin of the entire sequence.

Recovering the Minimax-Optimal Dependence on $T$ and $P _ { T }$ . The guarantee above also recovers the minimax-optimal $\sqrt { T P _ { T } }$ dependence for comparator sequences on the boundary. For each ${ \mathbf { u } } _ { t } \in \Delta _ { d } .$ , consider its smoothed counterpart $\tilde { \mathbf { u } } _ { t } = ( 1 -$ $\beta ) { \bf u } _ { t } + ( \beta / d ) { \bf 1 }$ . As shown in Appendix C.4, D- $\cdot \mathrm { R e g } _ { T } ( \{ \mathbf { u } _ { t } \} _ { t = 1 } ^ { T } ) \leq \mathrm { D } { \cdot } \mathrm { R e g } _ { T } ( \{ \tilde { \mathbf { u } } _ { t } \} _ { t = 1 } ^ { T } ) +$ $T \ln ( 1 / ( 1 - \beta ) )$ . Applying the bound above to the interior sequence $\{ \tilde { \mathbf { u } } _ { t } \} _ { t = 1 } ^ { T }$ and balancing β yields the claimed $\sqrt { T P _ { T } }$ dependence up to logarithmic factors.

Corollary 5 Suppose d, $T \ \geq \ 2$ . Let $B \geq 1$ and let A be an online algorithm guaranteeing, for every comparator sequence $\mathbf { u } _ { 1 } , \ldots , \mathbf { u } _ { T } \in \mathrm { r i } ( \Delta _ { d } )$

$$
\mathrm { D } { \cdot } \mathrm { R e g } _ { T } ( \{ \mathbf { u } _ { t } \} _ { t = 1 } ^ { T } ) \leq \mathcal { O } \left( d B T ^ { \frac { 1 } { 3 } } \big ( P _ { T } ^ { \mathrm { J S } } \big ) ^ { \frac { 2 } { 3 } } + d \ln ( d T ) \right) ,
$$

where $\mathrm { r i } ( \Delta _ { d } )$ denotes the relative interior of $\Delta _ { d }$ . Then A also guarantees, for any comparator sequence $\mathbf { u } _ { 1 } , \ldots , \mathbf { u } _ { T } \in \Delta _ { d }$

$$
\mathrm { D - R e g } _ { T } ( \{ \mathbf { u } _ { t } \} _ { t = 1 } ^ { T } ) \leq \mathcal { O } \left( d B ^ { \frac { 3 } { 4 } } \sqrt { T P _ { T } } + d \ln ( d T ) \right) ,
$$

Applying Corollary 5 to Theorem 3 with $B = ( \ln ( d T ) ) ^ { 2 / 3 }$ yields an ${ \tilde { \mathcal { O } } } \big ( d ( \sqrt { T P _ { T } } { + } 1 ) \big )$ regret bound for arbitrary comparator sequences, matching the lower bound in terms of T and $P _ { T }$

Equivalent Implementation and Interval-Regret Guarantee. Following the same arguments in Adamskiy et al. (2016); Zhang et al. (2025), one can show Algorithm 1 is equivalent to running the FLH algorithm with Universal Portfolio as the base learner. Therefore, our method naturally enjoys the interval regret guarantee, which also implies a bound on the switching regret.

Proposition 6 For any interval $\mathcal { T } = [ r , s ] \subseteq [ T ]$ and any comparator $\mathbf { u _ { \lambda } } \in \ \Delta _ { d }$ Algorithm 1 with $\mu _ { t } = 1 / t$ ensures

$$
\sum _ { t \in \mathbb { Z } } \ell _ { t } ( \mathbf { w } _ { t } ) - \sum _ { t \in \mathbb { Z } } \ell _ { t } ( \mathbf { u } ) \leq \mathcal { O } \big ( d \ln ( d T ) \big ) .
$$

Furthermore, let $\begin{array} { r } { \mathsf { S } _ { T } = \sum _ { t = 2 } ^ { T } \mathbb { 1 } \{ \mathbf { u } _ { t } \neq \mathbf { u } _ { t - 1 } \} } \end{array}$ . Then the dynamic regret of Algorithm 1 is bounded by $\mathcal { O } \big ( d ( \mathsf { S } _ { T } + 1 ) \bar { \ln ( d T ) } \big )$

## 3.3. Temporal Adaptivity through JS<sup>q</sup>-Path Length

The JS-path length captures the spatial geometry of comparator movements, but sequences with the same total JS variation can admit diferent regret guarantees depending on how that variation is distributed over time. For example, comparators with $\mathsf { S } _ { T }$ switches admit $\mathcal { O } ( d ( \mathsf { S } _ { T } + 1 ) \ln ( d T ) )$ dynamic regret (Proposition 6) even when $P _ { T } ^ { \mathrm { J S } } = \Theta ( \mathsf { S } _ { T } )$ , whereas the worst-case regret under the same path-length budget is $\Omega ( T ^ { 1 / 3 } \mathsf { S } _ { T } ^ { 2 / 3 } )$ (Theorem 4). We therefore introduce a family of measures that interpolates between the JS-path length and the switching number. Specifically, for $q \in [ 0 , 1 ]$ and a comparator sequence in $\Delta _ { d }$ , we define the JS<sup>q</sup>-path length as

$$
P _ { T , q } ^ { \mathrm { J S } } : = \sum _ { t = 2 } ^ { T } \mathrm { J S } ( \mathbf u _ { t } , \mathbf u _ { t - 1 } ) ^ { q } .\tag{6}
$$

The ${ \mathrm { J S } } ^ { q } .$ -path length interpolates between two familiar quantities. At $q = 0$ , it reduces to the switching number $\mathsf { S } _ { T }$ under the convention $0 ^ { 0 } = 0$ , and at $q = 1$ it recovers the JS-path length $P _ { T } ^ { \mathrm { J S } }$ of the preceding subsection. The following theorem shows that Algorithm 1 simultaneously achieves the corresponding dynamic regret guarantee for every order $q \in [ 0 , 1 ]$

Theorem 7 Let $d , T \ge 2$ . Algorithm 1 with $\mu _ { t } = 1 / t$ simultaneously guarantees, for every $q \in [ 0 , 1 ]$ ，

$$
\mathrm { D } \mathbf { - } \mathrm { R e g } _ { T } ( \{ \mathbf { u } _ { t } \} _ { t = 1 } ^ { T } ) \leq \mathcal { O } \left( d \left( T ^ { \frac { q } { q + 2 } } \left( P _ { T , q } ^ { \mathrm { J S } } \right) ^ { \frac { 2 } { q + 2 } } \left( \ln ( d T ) \right) ^ { \frac { 2 } { q + 2 } } + \ln ( d T ) \right) \right)
$$

for every comparator sequence $\mathbf { u } _ { 1 } , \ldots , \mathbf { u } _ { T } \in \Delta _ { d }$

The guarantee holds simultaneously for all $q \in [ 0 , 1 ]$ because neither q nor $P _ { T , q } ^ { \mathrm { J S } }$ is an input to the algorithm. Consequently, one may take the best of these bounds in hindsight. The choices $q = 1$ and $q = 0$ recover the $T ^ { 1 / 3 } ( P _ { T } ^ { \mathrm { J S } } ) ^ { 2 / 3 }$ dependence of Theorem 3 and the $\mathcal { O } \big ( d ( \mathsf { S } _ { T } + 1 ) \ln ( d T ) \big )$ switching guarantee of Proposition 6. Interestingly, for every $q > 1$ , the $q = 1$ guarantee already yields the $T ^ { 1 - 2 / ( 3 q ) } ( P _ { T , q } ^ { \mathrm { J S } } ) ^ { 2 / ( 3 q ) }$ dependence via H¨older’s inequality $P _ { T , 1 } ^ { \mathrm { J S } } \leq T ^ { 1 - 1 / q } ( P _ { T , q } ^ { \mathrm { J S } } ) ^ { 1 / q }$ , matching the leading dependence on the horizon and variation budget in minimax online forecasting (Baby and Wang, 2019).

An Intermediate-Order Example. The two endpoints do not exhaust the benefits of Theorem 7. The following example shows how an intermediate order can exploit comparator movements whose magnitudes decay over time. We consider a two-asset market and define $\mathbf { u } _ { t } = ( 1 / 2 + z _ { t } , 1 / 2 - z _ { t } )$

![](images/301f8d4f784da5e597b0379680e0e0a39861fd5491a79c018d517eed3a0e70f1.jpg)

where $\begin{array} { r } { z _ { t } = \frac { 1 } { 4 } ( 1 - 1 / t ) } \end{array}$ . For this path, $P _ { T , 1 } ^ { \mathrm { J S } } = \Theta ( 1 )$ , so choosing $q = 1$ gives $\widetilde { \mathcal { O } } ( T ^ { 1 / 3 } )$ . At the other endpoint, $P _ { T , 0 } ^ { \mathrm { J S } } = \mathsf { S } _ { T } = T - 1$ , so choosing $q = 0$ gives $\widetilde { \mathcal { O } } ( T )$ . When $q = 1 / 2$ we have $P _ { T , 1 / 2 } ^ { \mathrm { J S } } = \Theta ( \ln T )$ , yielding a bound of $\widetilde { \mathcal { O } } ( T ^ { 1 / 5 } )$ , which is substantially better than those at the two endpoints. Appendix D.2 provides the calculations.

## 3.4. Proof Sketch of Theorem 7

Our analysis is based on the notion of mixability, which was first used to analyze the prediction with expert advice problem (Vovk, 1998) and has proven useful for achieving fast rates in both stochastic learning and online learning (Vovk, 2001; van Erven et al., 2015; Foster et al., 2018). Recently, Zhang et al. (2025) used this notion to obtain fast-rate dynamic regret bounds. However, their analysis does not directly extend to OPS when the log-loss gradients are unbounded. We provide a more detailed discussion after briefly sketching the proof of Theorem 7 below. The complete proof is provided in Appendix D.1.

Proof Sketch of Theorem 7. The starting point of our analysis is that Cover’s loss for OPS is 1-mixable over the simplex $\Delta _ { d }$ for any $\mathbf { x } _ { t } \in \mathbb { R } _ { + } ^ { d }$ , in the sense that for any distribution $P _ { t }$ over $\Delta _ { d }$ and $\mathbf { w } _ { t } = \mathbb { E } _ { \mathbf { u } \sim P _ { t } } [ \mathbf { u } ]$ , we have $\ell _ { t } ( \mathbf { w } _ { t } ) = - \ln \left( \mathbb { E } _ { \mathbf { u } \sim P _ { t } } [ \exp ( - \ell _ { t } ( \mathbf { u } ) ) ] \right)$ which implies the following variational identity.

Lemma 8 For any distribution $Q _ { t }$ with $\operatorname { s u p p } ( Q _ { t } ) \subseteq \operatorname { s u p p } ( P _ { t } )$ , it holds that

$$
\ell _ { t } ( \mathbf { w } _ { t } ) = \mathbb { E } _ { \mathbf { u } \sim Q _ { t } } [ \ell _ { t } ( \mathbf { u } ) ] + \mathrm { K L } ( Q _ { t } \parallel P _ { t } ) - \mathrm { K L } ( Q _ { t } \parallel \tilde { P } _ { t + 1 } ) ,
$$

where $\tilde { P } _ { t + 1 } ( \mathbf { u } ) \propto P _ { t } ( \mathbf { u } ) \exp ( - \ell _ { t } ( \mathbf { u } ) )$ for all $\mathbf { u } \in \mathrm { s u p p } ( P _ { t } )$

By the update rules (4) and (5), we have $P _ { t + 1 } = ( 1 - \mu _ { t + 1 } ) \tilde { P } _ { t + 1 } + \mu _ { t + 1 } \mathrm { D i r } ( \mathbf { 1 } )$ Telescoping over T iterations and upper-bounding the discrepancy between $P _ { t + 1 }$ and $\tilde { P } _ { t + 1 }$ due to the fixed-share update yield

$$
\begin{array} { r l } & { \displaystyle \sum _ { t = 1 } ^ { T } \ell _ { t } ( \mathbf w _ { t } ) \leq \sum _ { t = 1 } ^ { T } \mathbb { E } _ { \mathbf { u } \sim Q _ { t } } [ \ell _ { t } ( \mathbf { u } ) ] + \sum _ { t = 2 } ^ { T } \int _ { \mathbf { u } \in \Delta _ { d } } \left( Q _ { t } ( \mathbf { u } ) - Q _ { t - 1 } ( \mathbf { u } ) \right) \ln \frac { 1 } { P _ { t } ( \mathbf { u } ) } \mathrm { d } \mathbf { u } } \\ & { \qquad + \operatorname { K L } ( Q _ { T } | | P _ { 1 } ) + 1 + \ln T . } \end{array}
$$

To accommodate the simplex constraint, we choose the comparator $Q _ { t } = \operatorname * { D i r } ( { \bf 1 } + \gamma { \bf u } _ { t } )$ as a Dirichlet distribution, where $\gamma > 0$ is a free parameter in the analysis and $\mathbf { u } _ { t }$ is the comparator sequence.

The Dirichlet comparator naturally respects the simplex constraint and allows us to control the expected log loss without any bounded-gradient assumption. By a careful analysis exploiting the structure of the Dirichlet distribution, we can show that

$$
\left\{ \begin{array} { l l } { \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { E } _ { Q _ { t } } [ \ell _ { t } ( \mathbf { u } ) ] \leq \sum _ { t = 1 } ^ { T } \ell _ { t } ( \mathbf { u } _ { t } ) + T \ln \left( 1 + \frac { d } { \gamma } \right) , } \\ { \displaystyle \sum _ { t = 2 } ^ { T } \int _ { \mathbf { u } \in \Delta _ { d } } \left( Q _ { t } ( \mathbf { u } ) - Q _ { t - 1 } ( \mathbf { u } ) \right) \ln \frac { 1 } { P _ { t } ( \mathbf { u } ) } \mathrm { d } \mathbf { u } \lesssim d \ln ( d T ) \gamma ^ { q / 2 } P _ { T , q } ^ { \mathrm { J S } } . } \end{array} \right.
$$

The first inequality follows from reparameterizing the Dirichlet distribution by several Gamma distributions. For the second inequality, Lemma 17 gives

$$
\begin{array} { r l r } {  { \int _ { \Delta _ { d } }  Q _ { t } ( \mathbf { u } ) - Q _ { t - 1 } ( \mathbf { u } )  \mathrm { d } \mathbf { u } \leq \operatorname* { m i n } \Big \{ 2 , 2 \sqrt { 2 \gamma } \mathrm { J S } ( \mathbf { u } _ { t } , \mathbf { u } _ { t - 1 } ) \Big \} } } \\ & { } & { \leq 2 ( 2 \gamma ) ^ { q / 2 } \mathrm { J S } ( \mathbf { u } _ { t } , \mathbf { u } _ { t - 1 } ) ^ { q } , } \end{array}
$$

where the last inequality follows from min $\{ 1 , a \} \le a ^ { q }$ for $a > 0$ and $q \in [ 0 , 1 ]$ , with the endpoint convention above when $\mathbf { u } _ { t } = \mathbf { u } _ { t - 1 }$

Combining these bounds and accounting for the endpoint term yields

$$
\mathrm { D - } \mathrm { R e g } _ { T } ( \{ \mathbf { u } _ { t } \} _ { t = 1 } ^ { T } ) \lesssim T \ln \left( 1 + \frac { d } { \gamma } \right) + d \ln ( d T ) \gamma ^ { q / 2 } P _ { T , q } ^ { \mathrm { J S } } + d \ln ( 1 + \gamma ) + \ln T .
$$

Choosing $\gamma = \operatorname* { m i n } \big \{ \Theta \big ( ( T / ( { P _ { T , q } ^ { \mathrm { J S } } \ln ( d T ) } ) ) ^ { 2 / ( q + 2 ) } \big )$ , T	 yields the claimed bound. Since γ only specifies the comparator distributions used in the analysis, Algorithm 1 requires no prior knowledge of $q$ or $P _ { T , q } ^ { \mathrm { J S } }$ □

Remark 9 (Comparison with Zhang et al. (2025)) Related mixability-based arguments were also employed by Zhang et al. (2025), where a Gaussian comparator distribution was adopted for analytical convenience. However, since Gaussian distributions have full support on $\mathbb { R } ^ { d }$ , this choice is not directly applicable to constrained domains. To handle general constraints, Zhang et al. (2025) proposed learning with a quadratic surrogate loss that extends the domain to $\mathbb { R } ^ { d }$ , which requires a bounded gradient. Moreover, when updating with the surrogate loss, the mean of the learned distribution $P _ { t + 1 }$ may lie outside the constraint set. To address this issue, they project $P _ { t }$ onto a family of Gaussian mixture models with possibly infinitely many components and bounded component means and variances, which is generally computationally intractable. By contrast, we adopt a Dirichlet comparator to naturally accommodate the simplex constraint and the unbounded loss.

## 4. Tractable Fast Rates for General OXO

This section studies OPS under an additional bounded-gradient assumption and establishes fast-rate dynamic regret bounds for all comparator sequences. In fact, we consider the more general setting of online exp-concave optimization over an arbitrary compact convex domain W under the following assumptions:

Assumption 1 For any $t \in [ T ]$ , the loss $\ell _ { t } : \mathcal { W } \to \mathbb { R }$ is κ-exp-concave over $\mathcal { W }$

Assumption 2 The domain W is compact and convex with $\begin{array} { r } { \operatorname* { s u p } _ { { \bf u } , { \bf w } \in \mathcal { W } } \| { \bf u } - { \bf w } \| _ { 1 } \leq 2 . ^ { 1 } \qquad } \end{array}$

Assumption 3 For any $t \in [ T ]$ and some $G > 1$ , we have $\begin{array} { r } { \operatorname* { s u p } _ { \mathbf { w } \in \mathcal { W } } \| \nabla \ell _ { t } ( \mathbf { w } ) \| _ { \infty } \leq G } \end{array}$

OPS satisfies Assumptions 1 and 2 since Cover’s loss is 1-exp-concave and the simplex has $\ell _ { 1 }$ diameter at most 2. The gradient bound can be satisfied under several natural conditions (Helmbold et al., 1998; Agarwal et al., 2006). For example, one may assume that the ratio of returns is bounded by G, that is, ma $\begin{array} { r } { \mathrm { { c } } _ { i \in [ d ] } \mathscr { x } _ { t , i } \big / \operatorname* { m i n } _ { i \in [ d ] } \mathscr { x } _ { t , i } \leq G } \end{array}$ for every $t \in [ T ]$ , or restrict the portfolio domain to $\mathcal { W } = \left\{ \mathbf { w } \in \Delta _ { d } : \operatorname* { m i n } _ { i \in [ d ] } w _ { i } \geq 1 / G \right\}$ (with $G \geq d )$ . Beyond OPS, other examples of online exp-concave optimization include logistic regression and least-squares regression, to which our method can be applied.

```latex
Algorithm 2 Follow-the-Leading-History
Input: Loss parameter $\begin{array} { r } { \eta = \frac { 1 } { 5 } \operatorname* { m i n } \{ 1 / ( 2 G ) , \kappa \} } \end{array}$ and fixed-share parameter $\mu _ { t } = 1 / t$
1: Initialize $P _ { 1 } = \mathcal { N } ( \mathbf { u } _ { 0 } , I _ { d } )$ as a Gaussian distribution with mean $\mathbf { u } _ { 0 } \in \mathcal { W }$
2: Initialize a pool with base-learners $\mathcal { H } _ { 1 } = \{ B _ { 1 } \}$ , where $\boldsymbol { B } _ { 1 }$ is the initial base-learner
with the distribution $P _ { 1 , 1 } = P _ { 1 }$ and weight $p _ { 1 , 1 } = 1$
3: for $t = 1 , 2 , \dots , T$ do
4: The learner submits the prediction $\begin{array} { r } { \mathbf { w } _ { t } = \mathbb { E } _ { \mathbf { u } \sim P _ { t } } [ \mathbf { u } ] = \sum _ { B _ { i } \in \mathcal { H } _ { t } } p _ { t , i } \mathbf { w } _ { t , i } } \end{array}$ , observes
the loss $\ell _ { t } ,$ and constructs the surrogate loss in (10).
5: Update the distribution of each base learner $\boldsymbol { B } _ { i } \in \mathcal { H } _ { t }$ by
( $P _ { t + 1 , i } ^ { \prime } ( \mathbf { u } ) \propto P _ { t , i } ( \mathbf { u } ) \cdot e ^ { - \eta \tilde { \ell } _ { t } ( \mathbf { u } ) } , \forall \mathbf { u } \in \mathbb { R } ^ { d }$ (7)
$\begin{array} { r } { P _ { t + 1 , i } = \arg \operatorname* { m i n } _ { Q \in \mathcal { W } } \mathrm { K L } \big ( Q \| P _ { t + 1 , i } ^ { \prime } \big ) } \end{array}$
where ${ \mathcal { W } } = \{ Q : \mathbb { E } _ { \mathbf { u } \sim Q } [ \mathbf { u } ] \in \mathcal { W } \}$ is the set of distributions with means in W.
6: Update the weight for each base learner $\boldsymbol { B } _ { i } \in \mathcal { H } _ { t }$ by
$\tilde { p } _ { t + 1 , i } \propto p _ { t , i } \cdot \mathbb { E } _ { \mathbf { u } \sim P _ { t , i } } [ \exp ( - \eta \tilde { \ell } _ { t } ( \mathbf { u } ) ) ] .$ (8)
7: Initialize a new base-learner $\boldsymbol { B } _ { t + 1 }$ with the distribution $P _ { t + 1 , t + 1 } = \mathcal { N } ( \mathbf { u } _ { 0 } , I _ { d } )$ and
update the weight for existing base-learner by
$p _ { t + 1 , i } = \left\{ \begin{array} { l l } { \ ( 1 - \mu _ { t + 1 } ) \cdot \tilde { p } _ { t + 1 , i } \ \mathrm { f } } \\ { \ \mu _ { t + 1 } \mathrm { f o r } \ B _ { i } = B _ { t + 1 } } \end{array} \right.$ for $\boldsymbol { B } _ { i } \in \mathcal { H } _ { t }$
(9)
8: Update the pool $\mathcal { H } _ { t + 1 } = \mathcal { H } _ { t } \cup \{ B _ { t + 1 } \}$ and obtain $\begin{array} { r } { P _ { t + 1 } ( \mathbf { u } ) = \sum _ { B _ { i } \in \mathcal { H } _ { t + 1 } } p _ { t + 1 , i } . } \end{array}$
$P _ { t + 1 , i } ( \mathbf { u } )$
9: end for
```

## 4.1. Proposed Method

Our algorithm is summarized in Algorithm 2. Instead of learning directly with the original loss, we employ the following surrogate loss:

$$
\begin{array} { r } { \tilde { \ell } _ { t } ( \mathbf { w } ) = \mathbf { g } _ { t } ^ { \top } ( \mathbf { w } - \mathbf { w } _ { t } ) + \eta \left. \mathbf { w } - \mathbf { w } _ { t } \right. _ { \mathbf { g } _ { t } \mathbf { g } _ { t } ^ { \top } } ^ { 2 } . } \end{array}\tag{10}
$$

Here, $\mathbf { g } _ { t } = \nabla \ell _ { t } ( \mathbf { w } _ { t } )$ denotes the gradient of the loss function. For a κ-exp-concave loss over the domain W, Hazan (2016, Lemma 4.3) shows that the regret under the original loss can be upper bounded by that under the surrogate loss: $\ell _ { t } ( \mathbf { w } _ { t } ) - \ell _ { t } ( \mathbf { u } _ { t } ) \leq$ $\widetilde { \ell } _ { t } ( \mathbf { w } _ { t } ) - \widetilde { \ell } _ { t } ( \mathbf { u } _ { t } )$ for any ${ \mathbf { u } } _ { t } \in \mathcal { W }$ , provided that $\eta \leq \textstyle { \frac { 1 } { 4 } }$ min $\{ ( \mathrm { m a x } _ { \mathbf { w } \in \mathcal { W } } | \mathbf { g } _ { t } ^ { \top } ( \mathbf { w } - \mathbf { w } _ { t } ) | ) ^ { - 1 } , \kappa \}$ This choice of a quadratic surrogate loss is standard in online learning for obtaining eficient updates. In our setting, its quadratic form also allows us to work with Gaussian distributions, which simplify the regret analysis.

Our method follows the FLH framework (Hazan and Seshadhri, 2009), with multiple base learners started at diferent times and a meta-learner that aggregates their predictions.

• Base-learners: At each iteration $t = i .$ , we initialize a new base learner $B _ { i }$ and add it to the expert pool $\mathcal { H } _ { t }$ . The base-learner is initialized with a Gaussian distribution $P _ { t , i } = \mathcal { N } ( \mathbf { u } _ { 0 } , I _ { d } )$ , where $\mathbf { u } _ { 0 } \in \mathcal { W }$ can be any point in the feasible domain. The distribution of each base learner $B _ { i }$ is updated using exponential weights with respect to the surrogate loss, followed by the projection in line 5 of Algorithm 2. Since the surrogate loss $\tilde { \ell } _ { t }$ is quadratic, van der Hoeven et al. (2018, Theorem 5) show that the resulting distribution $P _ { t , i } = \mathcal { N } ( \mathbf { w } _ { t , i } , H _ { t , i } ^ { - 1 } )$ remains Gaussian, with its mean and covariance updated via an ONS-type rule. An explicit update formula is provided in (53) in Appendix E.

• Meta-learner: We also maintain a meta-learner that assigns a weight $p _ { t , i }$ to each base learner $\boldsymbol { B } _ { i } \in \mathcal { H } _ { t }$ to aggregate their predictions. Specifically, the weights are updated based on their historical performance (line 6), with a fixed-share step that incorporates the new base learner (line 7). The final prediction is obtained by taking the weighted average of the base learners’ predictions (line 4). One slight diference between Algorithm 2 and standard FLH is that line 6 updates the weights using the expectation term $\tilde { p } _ { t + 1 , i } \propto p _ { t , i } \mathbb { E } _ { \mathbf { u } \sim P _ { t , i } } \big [ e ^ { - \eta \tilde { \ell } _ { t } ( \mathbf { u } ) } \big ]$ , whereas the classical update is $\tilde { p } _ { t + 1 , i } \propto p _ { t , i } e ^ { - \eta \tilde { \ell } _ { t } ( \mathbf { w } _ { t , i } ) }$ . This diference is important for our analysis, as it allows us to align Algorithm 2 with exponential-weights updates over distributions. We also note that the update in line 6 admits a closed-form expression, since $P _ { t , i }$ is Gaussian and $\tilde { \ell } _ { t }$ is a quadratic function.

We have the following guarantee, whose proof is provided in Appendix E.

Theorem 10 Under Assumptions $\begin{array} { r } { I , \ 2 , } \end{array}$ and 3 and $T \geq 2$ , Algorithm 2 with $\mu _ { t } = 1 / t$ and $\begin{array} { r } { \eta = \frac { 1 } { 5 } \operatorname* { m i n } \{ 1 / ( 2 G ) , \kappa \} } \end{array}$ ensures

$$
\mathrm { D - R e g } _ { T } ( \{ \mathbf { u } _ { t } \} _ { t = 1 } ^ { T } ) \leq \mathcal { O } \left( \frac { d } { \eta } \Big ( \ln ( d T ) + T ^ { \frac { 1 } { 3 } } P _ { T } ^ { \frac { 2 } { 3 } } ( \ln ( T d ) ) ^ { \frac { 2 } { 3 } } \Big ) \right) ,
$$

for any sequence ${ \mathbf u } _ { 1 } , \dots , { \mathbf u } _ { T } \in { \mathcal W }$ , where $\begin{array} { r } { P _ { T } = \sum _ { t = 2 } ^ { T } \| \mathbf { u } _ { t } - \mathbf { u } _ { t - 1 } \| _ { 1 } } \end{array}$ is the path length and $\eta ^ { - 1 } = 5 \operatorname* { m a x } \{ 2 G , \kappa ^ { - 1 } \}$

Remark 11 (Relation to FLH-ONS) Our algorithm is a variant of FLH (Hazan and Seshadhri, 2009) with ONS (Hazan et al., 2007) as its base learner. Although FLH-ONS has well-established guarantees on interval regret, previous reductions to nearly optimal dynamic regret for exp-concave losses either require improper learning, which is infeasible in OPS, or are restricted to box-constrained domains (Baby and Wang, 2021, 2022a). Our dynamic regret guarantee holds for arbitrary compact convex domains while remaining proper and computationally tractable.

## 4.2. Two-layer Mixability-based Analysis

This section sketches the proof of Theorem 10 using a mixability-based argument. Zhang et al. (2025) also used mixability to obtain nearly optimal dynamic regret for OXO, but their method is computationally intractable, as it requires projecting the full Gaussian mixture. We first explain why their analysis does not directly apply to our algorithm and then present the key ideas behind our two-layer analysis.

Limitations of Previous Attempts. Zhang et al. (2025) showed that although the surrogate loss is not mixable in general, mixability-based analysis still applies when the mean of each component $P _ { t , i }$ in the Gaussian mixture $P _ { t }$ lies in the feasible domain, yielding the variational formulation:

$$
\widetilde { \ell } _ { t } ( \mathbf { w } _ { t } ) \leq \mathbb { E } _ { \mathbf { u } \sim Q _ { t } } [ \widetilde { \ell } _ { t } ( \mathbf { u } ) ] + \frac { 1 } { \eta } \mathrm { K L } ( Q _ { t } \Vert P _ { t } ) - \frac { 1 } { \eta } \mathrm { K L } ( Q _ { t } \Vert P _ { t + 1 } ^ { \prime } ) ,\tag{11}
$$

where $P _ { t + 1 } ^ { \prime } ( \mathbf { u } ) \propto P _ { t } ( \mathbf { u } ) \exp ( - \eta \tilde { \ell } _ { t } ( \mathbf { u } ) )$ , which remains a Gaussian mixture model. However, the exponential weights update does not guarantee that the component means of $P _ { t + 1 } ^ { \prime }$ stay within the decision domain, and thus one cannot set $P _ { t + 1 } = P _ { t + 1 } ^ { \prime }$ to telescope the KL terms over $T$ rounds. To overcome this issue, Zhang et al. (2025) project $P _ { t + 1 } ^ { \prime }$ onto a set M of Gaussian mixtures with component means in W and bounded covariances, which ensures a KL–Pythagorean inequality such that KL $\left( Q \parallel P _ { t + 1 } \right) \leq \mathrm { K L } \left( Q \parallel P _ { t + 1 } ^ { \prime } \right)$ for any $Q \in { \mathcal { M } }$ with $\mathbb { E } _ { Q } [ \mathbf { u } ] \in \mathcal { W }$ . This allows telescoping and yields

$$
\sum _ { t = 1 } ^ { T } \widetilde { \ell } _ { t } ( \mathbf { w } _ { t } ) \lesssim \sum _ { t = 1 } ^ { T } \mathbb { E } _ { Q _ { t } } [ \widetilde { \ell } _ { t } ( \mathbf { u } ) ] + \frac { 1 } { \eta } \sum _ { t = 2 } ^ { T } \bigl ( \mathrm { K L } ( Q _ { t } \| P _ { t } ) - \mathrm { K L } ( Q _ { t - 1 } \| P _ { t } ) \bigr ) .\tag{12}
$$

Here, $\lesssim$ suppresses additive initialization and logarithmic terms. By choosing $Q _ { t } =$ $\mathcal { N } ( \mathbf { u } _ { t } , \sigma ^ { 2 } I _ { d } )$ and selecting $\sigma$ properly, the above inequality leads to the desired fast rate bound. This analysis does not apply to Algorithm 2, since our method projects each component $P _ { t + 1 , i } ^ { \prime }$ into the domain separately to gain computational eficiency, and such componentwise projections do not in general satisfy the required KL–Pythagorean guarantee for $P _ { t + 1 }$ and $P _ { t + 1 } ^ { \prime }$

Our Analysis. We overcome the projection issue by exploiting the two-layer structure, rather than applying mixability at the level of the aggregated distribution. Specifically, for the comparator distributions $Q _ { t }$ with $\mathbb { E } _ { Q _ { t } } [ \mathbf { u } ] \in \mathcal { W }$ and $\mathbf { q } _ { t } \in \Delta _ { | \mathcal { H } _ { t } | }$ , we have

$$
\begin{array} { r } { \widetilde { \ell } _ { t } ( \mathbf { w } _ { t } ) \leq \mathbb { E } _ { \mathbf { u } \sim Q _ { t } } [ \widetilde { \ell } _ { t } ( \mathbf { u } ) ] + \frac { 1 } { \eta } \Bigg ( \displaystyle \sum _ { B _ { i } \in \mathcal { H } _ { t } } q _ { t , i } \mathrm { K L } ( Q _ { t } \| P _ { t , i } ) + \mathrm { K L } ( \mathbf { q } _ { t } \| \mathbf { p } _ { t } ) ) } \\ { - \frac { 1 } { \eta } \Bigg ( \displaystyle \sum _ { B _ { i } \in \mathcal { H } _ { t } } q _ { t , i } \mathrm { K L } ( Q _ { t } \| P _ { t + 1 , i } ) + \mathrm { K L } ( \mathbf { q } _ { t } \| \widetilde { \mathbf { p } } _ { t + 1 } ) ) . } \end{array}\tag{13}
$$

The above variational-form bound can be viewed as a two-layer counterpart of (11), providing the flexibility to choose the comparators $\mathbf { q } _ { t }$ and $Q _ { t }$ for the meta-learner and the base-learner, respectively. We note that componentwise projections can be performed safely under (13), since the KL divergence is defined in terms of the individual distributions rather than the aggregated one.

The next question is how to choose $\mathbf { q } _ { t }$ and $Q _ { t }$ . To make the bound as tight as possible, we choose $\begin{array} { r } { \mathbf { q } _ { t } = \arg \operatorname* { m i n } _ { \mathbf { q } \in \Delta _ { | \mathcal { H } _ { \epsilon } | } } \sum _ { B _ { i } \in \mathcal { H } _ { t } } q _ { i } \mathrm { K L } \left( Q _ { t } \parallel P _ { t , i } \right) + \mathrm { K L } \left( \mathbf { q } \parallel \mathbf { p } _ { t } \right) } \end{array}$ , whose optimal value attains a closed-form formula $V _ { t } ( Q _ { t } ) = - \ln \left( \mathbb { E } _ { \mathbf { p } _ { t } } [ \exp ( - \mathrm { K L } \left( Q _ { t } \parallel P _ { t , i } \right) ) ] \right)$ After a sequence of algebraic manipulations, the bound admits a telescoping structure as in (12).

$$
\sum _ { t = 1 } ^ { T } \tilde { \ell } _ { t } ( \mathbf { w } _ { t } ) \lesssim \sum _ { t = 1 } ^ { T } \mathbb { E } _ { Q _ { t } } [ \tilde { \ell } _ { t } ( \mathbf { u } ) ] + \frac { 1 } { \eta } \sum _ { t = 2 } ^ { T } \left( V _ { t } ( Q _ { t } ) - V _ { t } ( Q _ { t - 1 } ) \right) .\tag{14}
$$

By specifying $Q _ { t } = \mathcal { N } ( \mathbf { u } _ { t } , \sigma ^ { 2 } I _ { d } )$ as a Gaussian distribution, we can further show that

$$
\begin{array} { r } { \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { E } _ { Q _ { t } } [ \tilde { \ell } _ { t } ( \mathbf { u } ) ] \leq \displaystyle \sum _ { t = 1 } ^ { T } \tilde { \ell } _ { t } ( \mathbf { u } _ { t } ) + \eta d G ^ { 2 } T \sigma ^ { 2 } , } \\ { \displaystyle \sum _ { t = 2 } ^ { T } ( V _ { t } ( Q _ { t } ) - V _ { t } ( Q _ { t - 1 } ) ) \lesssim \frac { d P _ { T } \log ( d T ) } { \sigma } + d P _ { T } \sigma ^ { 2 } . } \end{array}
$$

The first inequality follows from the quadratic formulation of the surrogate loss function, and the second inequality is obtained by showing that the gradient of $V _ { t } ( Q _ { t } )$ with respect to $\mathbf { u } _ { t }$ can be upper bounded by $\mathcal { O } ( d \log ( d T ) / \sigma + d \sigma ^ { 2 } )$ . We can obtain the desired bound by setting $\sigma = \widetilde { \Theta } \big ( P _ { T } ^ { 1 / 3 } T ^ { - 1 / 3 } \big )$ since $P _ { T } \leq 2 T$

## 5. Conclusion

We establish nearly matching upper and lower bounds for dynamic regret in nonstationary OPS under the standard $\boldsymbol { L } _ { \mathrm { 1 } } \mathrm { - p a t h }$ length. Universal Dynamic Portfolio achieves the upper bound and finer guarantees based on the JS-path length and the JS<sup>q</sup>-path length, adapting to the spatial and temporal structure of comparator sequences without parameter tuning or bounded-gradient assumptions. Under bounded gradients, we also develop an eficient proper method with fast dynamic regret for OPS and general online exp-concave optimization over compact convex domains with tractable metric projections. Future work includes developing more computationally eficient algorithms that retain the refined OPS guarantees, and reducing the number of active base learners.

## Acknowledgments and AI-use Statement

KJ and YZ were supported in part by a Singapore National Research Foundation AI Visiting Professorship award and NSF TRIPODS II DMS-2023166.

The authors developed the main research ideas, technical results, and theoretical developments in this work before January 2026. During subsequent extensions of the work from July to September 2026, GPT-6 Astra was used to assist in exploring proof strategies for the lower bounds, particularly the multidimensional constructions, and in developing the illustrative example for the JS<sup>q</sup>-path length results. GPT-5.6 and GPT-6 were also used during manuscript preparation for language editing, grammar checking, and polishing. The authors have carefully checked all AI-assisted mathematical arguments and take full responsibility for the content of the paper.

## References

Dmitry Adamskiy, Wouter M. Koolen, Alexey V. Chernov, and Vladimir Vovk. A closer look at adaptive regret. Journal of Machine Learning Research, 17:23:1–23:21, 2016.

Amit Agarwal, Elad Hazan, Satyen Kale, and Robert E. Schapire. Algorithms for portfolio management based on the newton method. In Proceedings of the Twenty-Third International Conference (ICML), volume 148, pages 9–16, 2006.

Dheeraj Baby and Yu-Xiang Wang. Online forecasting of total-variation-bounded sequences. In Advances in Neural Information Processing Systems 32 (NeurIPS), 2019.

Dheeraj Baby and Yu-Xiang Wang. Optimal dynamic regret in exp-concave online learning. In Proceedings of the 34th Conference on Learning Theory (COLT), pages 359–409, 2021.

Dheeraj Baby and Yu-Xiang Wang. Optimal dynamic regret in proper online learning with strongly convex losses and beyond. In Proceedings of the 25th International Conference on Artificial Intelligence and Statistics (AISTATS), pages 1805–1845, 2022a.

Dheeraj Baby and Yu-Xiang Wang. Optimal dynamic regret in LQR control. In Advances in Neural Information Processing Systems 35 (NeurIPS), pages 24879– 24892, 2022b.

Nicol‘o Cesa-Bianchi and G´abor Lugosi. Prediction, Learning, and Games. Cambridge University Press, 2006.

Nicol\`o Cesa-Bianchi, Pierre Gaillard, G´abor Lugosi, and Gilles Stoltz. Mirror descent meets fixed share (and feels no regret). In Advances in Neural Information Processing Systems 25 (NIPS), pages 989–997, 2012.

Thomas M. Cover. Universal portfolios. Mathematical Finance, 1(1):1–29, 1991.

Thomas M Cover and Erik Ordentlich. Universal portfolios with side information. IEEE Transactions on Information Theory, 42(2):348–363, 1996.

Ashok Cutkosky. Parameter-free, dynamic, and strongly-adaptive online learning. In Proceedings of the 37th International Conference on Machine Learning (ICML), pages 2250–2259, 2020.

Dylan J. Foster, Satyen Kale, Haipeng Luo, Mehryar Mohri, and Karthik Sridharan. Logistic regression: The importance of being improper. In Proceedings of the 31st Conference on Learning Theory (COLT), pages 167–208, 2018.

Andr´as Gy¨orgy and Csaba Szepesv´ari. Shifting regret, mirror descent, and matrices. In Proceedings of the 33nd International Conference on Machine Learning (ICML), pages 2943–2951, 2016.

Elad Hazan. Introduction to Online Convex Optimization. Foundations and Trends in Optimization, 2(3–4):157–325, 2016.

Elad Hazan and C. Seshadhri. Eficient learning algorithms for changing environments. In Proceedings of the 26th International Conference on Machine Learning (ICML), pages 393–400, 2009.

Elad Hazan, Amit Agarwal, and Satyen Kale. Logarithmic regret algorithms for online convex optimization. Machine Learning, 69(2–3):169–192, 2007.

David P Helmbold, Robert E Schapire, Yoram Singer, and Manfred K Warmuth. On-line portfolio selection using multiplicative updates. Mathematical Finance, 8 (4):325–347, 1998.

Mark Herbster and Manfred K. Warmuth. Tracking the best expert. Machine Learning, 32(2):151–178, 1998.

Mark Herbster and Manfred K. Warmuth. Tracking the best linear predictor. Journal of Machine Learning Research, 1:281–309, 2001.

Andrew Jacobsen, Alessandro Rudi, Francesco Orabona, and Nicol\`o Cesa-Bianchi. Dynamic regret reduces to kernelized static regret. In Advances in Neural Information Processing Systems 38 (NeurIPS), 2025.

R´emi J´ez´equel, Dmitrii Ostrovskii, and Pierre Gaillard. Eficient and near-optimal online portfolio selection. Mathematics of Operations Research, 0(0), 2025.

Adam Kalai and Santosh S. Vempala. Eficient algorithms for universal portfolios. Journal of Machine Learning Research, 3:423–440, 2002.

Haipeng Luo, Chen-Yu Wei, and Kai Zheng. Eficient online portfolio with logarithmic regret. In Advances in Neural Information Processing Systems 31 (NeurIPS), pages 8245–8255, 2018.

Zakaria Mhammedi and Alexander Rakhlin. Damped online Newton step for portfolio selection. In Proceedings of the Conference on Learning Theory (COLT), pages 5561–5595, 2022.

Erik Ordentlich and Thomas M. Cover. The cost of achieving the best portfolio in hindsight. Mathematics of Operations Research, 23(4):960–982, 1998.

Laurent Orseau, Tor Lattimore, and Shane Legg. Soft-bayes: Prod for mixtures of experts with log-loss. In Proceedings of the International Conference on Algorithmic Learning Theory (ALT), pages 372–399, 2017.

Yu-Yang Qian, Peng Zhao, Yu-Jie Zhang Masashi Sugiyama, and Zhi-Hua Zhou. Eficient non-stationary online learning by wavelets with applications to online distribution shift adaptation. In Proceedings of the 41st International Conference on Machine Learning (ICML), pages 41383–41415, 2024.

Yoram Singer. Switching portfolios. International Journal of Neural Systems, 08(04): 445–455, 1997.

Dirk van der Hoeven, Tim van Erven, and Wojciech Kot lowski. The many faces of exponential weights in online learning. In Proceedings of the 31st Conference on Learning Theory (COLT), pages 2067–2092, 2018.

Tim van Erven and Wouter M. Koolen. Metagrad: Multiple learning rates in online learning. In Advances in Neural Information Processing Systems 29 (NIPS), pages 3666–3674, 2016.

Tim van Erven, Peter D. Gr¨unwald, Nishant A. Mehta, Mark D. Reid, and Robert C. Williamson. Fast rates in statistical and online learning. Journal of Machine Learning Research, 16:1793–1861, 2015.

Tim Van Erven, Dirk Van der Hoeven, Wojciech Kot lowski, and Wouter M. Koolen. Open problem: Fast and optimal online portfolio selection. In Proceedings of the 33rd Conference on Learning Theory (COLT), pages 3864–3869, 2020.

Vladimir Vovk. A game of prediction with expert advice. Journal of Computer and System Sciences, 56(2):153–173, 1998.

Vladimir Vovk. Competitive on-line statistics. International Statistical Review, 69(2): 213–248, 2001.

Chen-Yu Wei and Haipeng Luo. Non-stationary reinforcement learning without prior knowledge: an optimal black-box approach. In Proceedings of the 34th Conference on Learning Theory (COLT), pages 4300–4354, 2021.

Lijun Zhang, Shiyin Lu, and Zhi-Hua Zhou. Adaptive online learning in dynamic environments. In Advances in Neural Information Processing Systems 31 (NeurIPS), pages 1330–1340, 2018.

Yu-Jie Zhang, Zhen-Yu Zhang, Peng Zhao, and Masashi Sugiyama. Adapting to continuous covariate shift via online density ratio estimation. In Advances in Neural Information Processing Systems 36 (NeurIPS), pages 29074–29113, 2023.

Yu-Jie Zhang, Peng Zhao, and Masashi Sugiyama. Non-stationary online learning for curved losses: Improved dynamic regret via mixability. In Proceedings of the 42nd International Conference on Machine Learning (ICML), 2025.

Peng Zhao, Yu-Jie Zhang, Lijun Zhang, and Zhi-Hua Zhou. Dynamic regret of convex and smooth functions. In Advances in Neural Information Processing Systems 33 (NeurIPS), pages 12510–12520, 2020.

Peng Zhao, Guanghui Wang, Lijun Zhang, and Zhi-Hua Zhou. Bandit convex optimization in non-stationary environments. Journal of Machine Learning Research, 22(125):1–45, 2021.

Peng Zhao, Yu-Jie Zhang, Lijun Zhang, and Zhi-Hua Zhou. Adaptivity and nonstationarity: Problem-dependent dynamic regret for online convex optimization. Journal of Machine Learning Research, 25(98):1–52, 2024.

Peng Zhao, Yan-Feng Xie, Lijun Zhang, and Zhi-Hua Zhou. Eficient methods for non-stationary online learning. Journal of Machine Learning Research, 26(208): 1–66, 2025.

Julian Zimmert, Naman Agarwal, and Satyen Kale. Pushing the eficiency-regret pareto frontier for online learning of portfolios and quantum states. In Proceedings of the Conference on Learning Theory (COLT), pages 182–226, 2022.

Martin Zinkevich. Online convex programming and generalized infinitesimal gradient ascent. In Proceedings of the 20th International Conference on Machine Learning (ICML), pages 928–936, 2003.

## Appendix A. Properties of the Dirichlet Distribution

In this section, we present some useful properties of the Dirichlet distribution that will be used in our analysis.

Definition 12 (Dirichlet Distribution) A random vector w $\in \Delta _ { d }$ is said to follow a Dirichlet distribution with parameter vector ${ \pmb { \alpha } } = ( \alpha _ { 1 } , . . . , \alpha _ { d } ) \in \mathbb { R } _ { + } ^ { d }$ , denoted $b y$ $\mathbf { w } \sim \mathrm { D i r } ( \pmb { \alpha } )$ , if its probability density function is

$$
p ( \mathbf { w } \mid \alpha ) = \frac { \Gamma \mathopen { } \mathclose \bgroup \left( \sum _ { i = 1 } ^ { d } \alpha _ { i } \aftergroup \egroup \right) } { \prod _ { i = 1 } ^ { d } \Gamma \mathopen { } \mathclose \bgroup \left( \alpha _ { i } \aftergroup \egroup \right) } \prod _ { i = 1 } ^ { d } w _ { i } ^ { \alpha _ { i } - 1 } , \qquad \mathbf { w } \in \Delta _ { d } .
$$

where $\textstyle \Gamma ( \alpha ) = \int _ { 0 } ^ { \infty } t ^ { \alpha - 1 } e ^ { - t } \mathrm { d } t$ is the Gamma function.

Property 13 (Properties of Dirichlet Distribution) Let $\mathbf { w } \sim \mathrm { D i r } ( \pmb { \alpha } )$ and $\alpha _ { 0 } =$ $\textstyle \sum _ { i = 1 } ^ { d } \alpha _ { i }$ . Then,

• The mean of the random vector is given by $\mathbb { E } _ { \mathbf { w } \sim \mathrm { D i r } ( \pmb { \alpha } ) } [ \mathbf { w } ] = \pmb { \alpha } / \alpha _ { 0 }$

• The covariance matrix is $\begin{array} { r } { \mathrm { C o v } [ \mathbf { w } ] = \frac { 1 } { \alpha _ { 0 } ^ { 2 } ( \alpha _ { 0 } + 1 ) } ( \alpha _ { 0 } \mathrm { d i a g } ( \pmb { \alpha } ) - \alpha \pmb { \alpha } ^ { \top } ) } \end{array}$

• Dirichlet distribution belongs to the exponential family with natural parameter $\pmb { \eta } = \pmb { \alpha } - \mathbf { 1 }$ and log-partition function $\begin{array} { r } { A ( \pmb { \eta } ) = \sum _ { i = 1 } ^ { d } \ln \Gamma ( \alpha _ { i } ) - \ln \Gamma ( \alpha _ { 0 } ) } \end{array}$

• The KL divergence between two Dirichlet distributions $\operatorname { D i r } ( \alpha )$ and $\operatorname { D i r } ( \beta )$ is given by

$$
\mathrm { K L } ( \mathrm { D i r } ( \alpha ) | | \mathrm { D i r } ( \beta ) ) = \ln \frac { \Gamma ( \alpha _ { 0 } ) } { \Gamma ( \beta _ { 0 } ) } - \sum _ { i = 1 } ^ { d } \ln \frac { \Gamma ( \alpha _ { i } ) } { \Gamma ( \beta _ { i } ) } + \sum _ { i = 1 } ^ { d } ( \alpha _ { i } - \beta _ { i } ) \left( \psi ( \alpha _ { i } ) - \psi ( \alpha _ { 0 } ) \right) ,
$$

where $\textstyle \psi ( \alpha ) = { \frac { d } { d \alpha } }$ ln Γ(α) is the digamma function.

• The diferential Shannon entropy of $\operatorname { D i r } ( \alpha )$ is

$$
H ( \operatorname { D i r } ( \alpha ) ) = \ln \left( { \frac { \prod _ { i = 1 } ^ { d } \Gamma ( \alpha _ { i } ) } { \Gamma ( \alpha _ { 0 } ) } } \right) + ( \alpha _ { 0 } - d ) \psi ( \alpha _ { 0 } ) - \sum _ { i = 1 } ^ { d } ( \alpha _ { i } - 1 ) \psi ( \alpha _ { i } ) .
$$

Property 14 (Properties of Gamma Function) Let $\textstyle \Gamma ( \alpha ) ~ = ~ \int _ { 0 } ^ { \infty } t ^ { \alpha - 1 } e ^ { - t } \mathrm { d } t$ be the Gamma function and $\begin{array} { r } { \psi ( \alpha ) = \frac { \mathrm { d } } { \mathrm { d } \alpha } \ln \Gamma ( \alpha ) \quad } \end{array}$ be the digamma function. Then, for any $\alpha > 0$ , we have

$$
\bullet \ \Gamma ( \alpha + 1 ) = \alpha \Gamma ( \alpha ) \ a n d \ \psi ( \alpha + 1 ) = \psi ( \alpha ) + 1 / \alpha .
$$

$$
\begin{array} { r } { \bullet \ln ( \alpha ) - \frac { 1 } { \alpha } \leq \psi ( \alpha ) \leq \ln ( \alpha ) - \frac { 1 } { 2 \alpha } . } \end{array}
$$

Lemma 15 Let $P = \mathrm { D i r } ( \mathbf { 1 } + \gamma \mathbf { u } ) , Q = \mathrm { D i r } ( \mathbf { 1 } + \gamma \mathbf { e } _ { i } )$ and $P _ { 1 } = \mathrm { D i r } ( \mathbf { 1 } )$ Then, KL $( P \parallel P _ { 1 } ) \le \mathrm { K L } \left( Q \parallel P _ { 1 } \right)$ for any $\mathbf { u } \in \Delta _ { d } , i \in [ d ]$ and $\gamma > 0$

Proof of Lemma 15 By definition of the KL divergence between two Dirichlet distributions, we have

$$
\mathrm { K L } \left( P \parallel P _ { 1 } \right) = \ln \frac { \Gamma ( d + \gamma ) } { \Gamma ( d ) } - \gamma \psi ( d + \gamma ) + \sum _ { i = 1 } ^ { d } \left( \gamma u _ { i } \psi ( 1 + \gamma u _ { i } ) - \ln \Gamma ( 1 + \gamma u _ { i } ) \right) .
$$

Let $\begin{array} { r } { G ( \mathbf { u } ) = \sum _ { i = 1 } ^ { d } g ( u _ { i } ) } \end{array}$ where $g ( u ) = \gamma u \psi ( 1 + \gamma u ) - \ln \Gamma ( 1 + \gamma u )$ . According to Lemma 18, one can show that $G ( \mathbf { u } )$ is a convex function over the simplex. Then, we have $\begin{array} { r } { G ( \mathbf { u } ) = G ( \sum _ { j = 1 } ^ { d } u _ { j } \mathbf { e } _ { j } ) \leq \sum _ { j = 1 } ^ { d } u _ { j } G ( \mathbf { e } _ { j } ) \leq G ( \mathbf { e } _ { i } ) } \end{array}$ for any $i \in [ d ]$ , where the last inequality holds because $G ( \mathbf { u } )$ is invariant under permutations of the coordinates. Then, we complete the proof by showing

$$
\begin{array} { r l r } {  { \mathrm { K L } ( P \parallel P _ { 1 } ) = \ln \frac { \Gamma ( d + \gamma ) } { \Gamma ( d ) } - \gamma \psi ( d + \gamma ) + G ( \mathbf { u } ) } } \\ & { } & { \leq \ln \frac { \Gamma ( d + \gamma ) } { \Gamma ( d ) } - \gamma \psi ( d + \gamma ) + G ( \mathbf { e } _ { i } ) = \mathrm { K L } ( Q \parallel P _ { 1 } ) . } \end{array}
$$

## Appendix B. Omitted Proofs for Section 3.1

## B.1. Proof of Theorem 1

Proof of Theorem 1 We focus on the minimax regret for the d-asset OPS problem:

$$
\mathcal { W } _ { T } ( \mathcal { U } _ { C } ) = \operatorname* { i n f } _ { f _ { 1 : T } } \operatorname* { s u p } _ { \substack { { \bf x } _ { 1 } , \ldots , { \bf x } _ { T } \in \mathbb { R } _ { + } ^ { d } } } \operatorname* { s u p } _ { { \bf u } _ { 1 : T } \in \mathcal { U } _ { C } } \left( \sum _ { t = 1 } ^ { T } \ell _ { t } ( { \bf w } _ { t } ) - \sum _ { t = 1 } ^ { T } \ell _ { t } ( { \bf u } _ { t } ) \right) ,
$$

where the infimum is taken over the algorithm’s online prediction rules $f _ { 1 : T }$ , and $\mathbf { w } _ { t }$ is the algorithm’s prediction based only on past observations $\mathbf { X } _ { 1 } , \ldots , \mathbf { X } _ { t - 1 }$ . To make the dependence explicit, we will also write $\mathbf { w } _ { t } = f _ { t } ( \mathbf { x } _ { 1 : t - 1 } )$ , where $\mathbf { x } _ { 1 : t - 1 } = ( \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { t - 1 } )$ and $f _ { t } : \mathbb { R } _ { + } ^ { d \times ( t - 1 ) }  \Delta _ { d }$ is a measurable online prediction rule determined by the algorithm. For any sequence $\mathbf { u } _ { 1 : T } = ( \mathbf { u } _ { 1 } , \dots , \mathbf { u } _ { T } )$ , we define

$$
\mathcal { U } _ { C } = \left\{ \mathbf { u } _ { 1 : T } \in \Delta _ { d } ^ { T } \left| \sum _ { t = 2 } ^ { T } \| \mathbf { u } _ { t } - \mathbf { u } _ { t - 1 } \| _ { 1 } \leq C \right. \right\}
$$

as the set of all comparator sequences whose $\boldsymbol { L } _ { \mathrm { 1 ^ { - } p a t h } }$ length is at most $C .$

Reduction to SPA. For any $C \in [ 0 , T ]$ , we derive a lower bound on the value $\mathscr { W } _ { T } ( \mathscr { U } _ { C } )$ by reducing the OPS problem to the sequential probability assignment (SPA) problem (Cesa-Bianchi and Lugosi, 2006, Chapter 9.1). Specifically, we consider the Kelly market vector setting, where $\mathbf { x } _ { t } \in \{ \mathbf { e } _ { 1 } , \ldots \mathbf { \mu } , \mathbf { e } _ { d } \}$ for all $t \in [ T ]$ , and $\mathbf { e } _ { i }$ denotes the i-th standard basis vector. We encode the market outcomes by defining $y _ { t } = i$ when $\mathbf { x } _ { t } = \mathbf { e } _ { i }$ , and introduce the multi-class log loss

$$
\ell _ { \log } ( \mathbf { w } , y ) = - \log [ \mathbf { w } ] _ { y } , \qquad \mathbf { w } \in \Delta _ { d } , \quad y \in [ d ] ,
$$

where $[ \mathbf { w } ] _ { y }$ denotes the y-th coordinate of w. It is straightforward to verify that $\ell _ { t } ( \mathbf { w } ) = \ell _ { \log } ( \mathbf { w } , y _ { t } )$ when $\mathbf { x } _ { t } = \mathbf { e } _ { y _ { t } }$ . Consequently, letting $\mathcal { V } = [ d ]$ , we obtain

$$
\mathcal { W } _ { T } ( \mathcal { U } _ { C } ) \geq \mathcal { V } _ { T } ( \mathcal { U } _ { C } ) : = \operatorname* { i n f } _ { f _ { 1 : T } } \operatorname* { s u p } _ { y _ { 1 } , \ldots , y _ { T } \in \mathcal { V } } \operatorname* { s u p } _ { \mathbf { u } _ { 1 : T } \in \mathcal { U } _ { C } } \left( \sum _ { t = 1 } ^ { T } \ell _ { \log } ( \mathbf { w } _ { t } , y _ { t } ) - \sum _ { t = 1 } ^ { T } \ell _ { \log } ( \mathbf { u } _ { t } , y _ { t } ) \right) .
$$

Throughout the proof, the dimension d is fixed and the asymptotic notation is with respect to $T .$

Hard Example Construction. To lower bound the dynamic term in the minimax regret $\nu _ { T } ( \mathcal { U } _ { C } )$ , we first restrict attention to the main regime $C \in [ 2 ( d - 1 ) / T , T / ( 2 ( d -$ $1 ) ) ]$ . In this regime, we construct an environment in which the time horizon is partitioned into consecutive intervals of length

$$
L = \left\lceil { \frac { d - 1 } { \epsilon } } \right\rceil { \mathrm { ~ w i t h ~ } } \epsilon = { \sqrt { \frac { ( d - 1 ) C } { 2 T } } } .
$$

This choice ensures $( d - 1 ) / T \le \epsilon \le 1 / 2$ and hence $L \leq T$ . The corner cases will be handled at the end of the proof. We further let $K = \lfloor T / L \rfloor$ denote the number of intervals. The first $K - 1$ intervals each have length L, while the final interval has length $T - ( K - 1 ) L \in [ L , 2 L - 1 ]$ . We will also use $\mathcal { T } _ { k } = \left[ s _ { k } , e _ { k } \right]$ to denote the k-th interval for $k \in \left\lceil K \right\rceil$ , with start time $s _ { k }$ and end time $e _ { k }$

For each block $k \in [ K ]$ , we consider a collection of environments indexed by a binary vector

$$
\mathbf { I } _ { k } = [ I _ { k , 1 } , \ldots , I _ { k , d - 1 } ] \in \{ 0 , 1 \} ^ { d - 1 } .
$$

The coordinates of $\mathbf { I } _ { k }$ are drawn independently and uniformly at random, i.e., $I _ { k , j } \sim$ $\operatorname { B e r n } ( 1 / 2 )$ independently for every $j \in [ d - 1 ]$ and $k \in [ K ]$ . There are in total $2 ^ { d - 1 }$ possible realizations of the environment index $\mathbf { I } _ { k }$ for each block k. On each interval $\mathcal { T } _ { k }$ , the labels are generated according to a static probability distribution $\widetilde { \mathbf { u } } _ { t } \in \Delta _ { d }$ that depends on the index $\mathbf { I } _ { k }$ . Specifically, for any $t \in \mathcal { Z } _ { k }$ , we set $\widetilde { \mathbf { u } } _ { t }$ as

$$
[ \widetilde { \mathbf { u } } _ { t } ] _ { j } = \frac { \epsilon } { d - 1 } I _ { k , j } ~ \mathrm { f o r ~ a l l } ~ j \in [ d - 1 ] ~ \mathrm { a n d } ~ [ \widetilde { \mathbf { u } } _ { t } ] _ { d } = 1 - \frac { \epsilon } { d - 1 } \sum _ { j = 1 } ^ { d - 1 } I _ { k , j } .
$$

In the above, the first $d - 1$ dimensions are associated with the environment index coordinate-wise, while the d-th dimension is a common asset. Since $\epsilon \leq 1 / 2$ , we have $[ \widetilde { \mathbf { u } } _ { t } ] _ { d } \geq 1 - \epsilon \geq 1 / 2$ , and hence the above vector is a valid portfolio in $\Delta _ { d }$ . The label is then generated according to $y _ { t } \sim \mathrm { C a t } ( \widetilde { \mathbf { u } } _ { t } )$ , where Cat(uet) denotes the categorical distribution with probability mass function $\widetilde { \mathbf { u } } _ { t }$

For any realization of the environment indices $\{ \mathbf { I } _ { k } \} _ { k = 1 } ^ { K }$ , the cumulative path length of the comparator sequence $\widetilde { \mathbf { u } } _ { 1 : T }$ is bounded by

$$
\begin{array} { l } { \displaystyle \sum _ { t = 2 } ^ { T } \| \widetilde { \mathbf { u } } _ { t } - \widetilde { \mathbf { u } } _ { t - 1 } \| _ { 1 } = \displaystyle \sum _ { k = 2 } ^ { K } \left( \frac { \epsilon } { d - 1 } \sum _ { j = 1 } ^ { d - 1 } | I _ { k , j } - I _ { k - 1 , j } | + \frac { \epsilon } { d - 1 } \left| \sum _ { j = 1 } ^ { d - 1 } I _ { k , j } - \sum _ { j = 1 } ^ { d - 1 } I _ { k - 1 , j } \right| \right) } \\ { \displaystyle \leq \sum _ { k = 2 } ^ { K } \frac { 2 \epsilon } { d - 1 } \sum _ { j = 1 } ^ { d - 1 } | I _ { k , j } - I _ { k - 1 , j } | \leq 2 \epsilon ( K - 1 ) \leq \frac { 2 T \epsilon ^ { 2 } } { d - 1 } = C , } \end{array}
$$

where the last inequality follows from $K \leq T / L \leq T \epsilon / ( d - 1 )$ . The above displayed inequality indicates that $\widetilde { \mathbf { u } } _ { 1 : T } \in \mathcal { U } _ { C }$ . Then, the minimax regret $\nu _ { T } ( \mathcal { U } _ { C } )$ can be further lower bounded by

$$
\begin{array} { r l } & { \mathcal { V } _ { T } ( \mathcal { U } _ { C } ) \geq \underset { f _ { 1 : T } } { \operatorname* { i n f } } \mathbb { E } _ { \mathbb { I } _ { 1 : \kappa } } [ \mathbb { E } _ { y _ { 1 : T } } [ \underset { t = 1 } { T } [ \underset { \ell = 1 } { \overset { T } { \prod } } \ell _ { \log } ( \mathbf { w } _ { t } , y _ { t } ) - \underset { t = 1 } { T } \ell _ { \log } ( \widetilde { \mathbf { u } } _ { t } , y _ { t } ) ] \mathbf { I } _ { 1 : \kappa } ] ] } \\ & { \quad \quad = \underset { f _ { 1 : T } } { \operatorname* { i n f } } \mathbb { E } _ { \mathbb { I } _ { 1 : \kappa } } [ \mathbb { E } _ { y _ { 1 : T } } [ \log ( \underset { t = 1 } { T } [ \frac { \widetilde { \mathbf { u } } _ { t } } { \log } | y _ { t } ) _ { \mathcal { U } } ) \Bigg | \mathbf { I } _ { 1 : \kappa } ] ] } \\ & { \quad \quad = \underset { f _ { 1 : T } } { \operatorname* { i n f } } \mathbb { E } _ { \mathbf { I } _ { 1 : \kappa } } [ \mathbb { E } _ { y _ { 1 : T } } [ \log ( \underset { t = 1 } { T } \frac { [ \widetilde { \mathbf { u } } _ { t } ] _ { y _ { t } } } { [ f _ { t } ( y _ { 1 : t - 1 } ) ] _ { \mathcal { U } _ { t } } } ) \Bigg | \mathbf { I } _ { 1 : \kappa } ] ] } \\ & { \quad \quad = \underset { f _ { 1 : T } } { \operatorname* { i n f } } \mathbb { E } _ { \mathbf { I } _ { 1 : \kappa } } [ \mathbb { E } _ { y _ { 1 : T } } [ \log ( \frac { \widetilde { \mathbf { q } } } { \log } ( \frac { \mathbb { I } _ { 1 : \kappa } } { \log ( y _ { 1 : T } ) } ) \Bigg | \mathbf { I } _ { 1 : \kappa } ] ] } \\ &  \quad \end{array}
$$

The first inequality follows from the fact that $\widetilde { \mathbf { u } } _ { 1 : T } \in \mathcal { U } _ { C }$ for every realization of the environment indices $\mathbf { I } _ { 1 : K }$ . The first equality follows from the definition of the log loss. Under the restriction $\mathbf { x } _ { t } = \mathbf { e } _ { y _ { t } }$ , we use $f _ { t } ( y _ { 1 : t - 1 } )$ as shorthand for $f _ { t } ( \mathbf { e } _ { y _ { 1 } } , \ldots , \mathbf { e } _ { y _ { t - 1 } } )$ . In the last equality, we define the joint mass function

$$
\mathbf { p } ( y _ { 1 : T } ) = \prod _ { t = 1 } ^ { T } [ f _ { t } ( y _ { 1 : t - 1 } ) ] _ { y _ { t } } ,
$$

where $f _ { t } : \mathcal { V } ^ { t - 1 } \to \Delta _ { d }$ is the online prediction rule at time t. We note that the online prediction rule $f _ { t }$ is deterministic given $y _ { 1 } , \ldots , y _ { t - 1 }$ , and hence p is fully determined by the online algorithm and is independent of the randomness in $\mathbf { I } _ { 1 : K }$ . Similarly, conditional on the environment indices $\mathbf { I } _ { 1 : K }$ , we define the joint probability mass function of the label sequence $y _ { 1 : T }$ as

$$
\widetilde { \mathbf { q } } ( y _ { 1 : T } \mid \mathbf { I } _ { 1 : K } ) = \operatorname* { P r } \left( Y _ { 1 : T } = y _ { 1 : T } \mid \mathbf { I } _ { 1 : K } \right) = \prod _ { t = 1 } ^ { T } [ \widetilde { \mathbf { u } } _ { t } ] _ { y _ { t } } .
$$

With a slight abuse of notation, we use $\widetilde { \mathbf { q } } _ { Y _ { 1 : T } | \mathbf { I } _ { 1 : K } }$ to denote the corresponding conditional distribution and $\scriptstyle \mathbf { p } _ { Y _ { 1 : T } }$ to denote the distribution induced by $\mathbf p ( y _ { 1 : T } )$ . One can check that $\mathbf p ( y _ { 1 : T } )$ is a valid probability mass function and, for every realization of $\mathbf { I } _ { 1 : K }$ $\widetilde { \mathbf { q } } \left( y _ { 1 : T } \mid \mathbf { I } _ { 1 : K } \right)$ is a valid conditional probability mass function. In particular,

$$
\sum _ { y _ { 1 : T } \in \mathcal { y } ^ { T } } \mathbf { p } ( y _ { 1 : T } ) = \sum _ { y _ { 1 : T - 1 } \in \mathcal { y } ^ { T - 1 } } \mathbf { p } ( y _ { 1 : T - 1 } ) = \cdot \cdot \cdot = \sum _ { y _ { 1 } \in \mathcal { y } } \mathbf { p } ( y _ { 1 } ) = 1 .
$$

Furthermore, since $y _ { t } \sim \mathrm { C a t } ( \widetilde { \mathbf { u } } _ { t } )$ conditional on $\mathbf { I } _ { 1 } , \ldots , \mathbf { I } _ { K }$ , the conditional distribution of the sequence $y _ { 1 : T }$ given $\mathbf { I } _ { 1 : K }$ has probability mass function $\widetilde { \mathbf { q } } ( y _ { 1 : T } \mid \mathbf { I } _ { 1 : K } )$ . Then, we have

$$
\mathbb { E } _ { \mathbf { I } _ { 1 : K } } \left[ \mathbb { E } _ { y _ { 1 : T } } \left[ \log \left( \frac { \widetilde { \mathbf { q } } ( y _ { 1 : T } \mid \mathbf { I } _ { 1 : K } ) } { \mathbf { p } ( y _ { 1 : T } ) } \right) \Big \vert \mathbf { I } _ { 1 : K } \right] \right] = \mathbb { E } _ { \mathbf { I } _ { 1 : K } } \left[ \mathrm { K L } ( \widetilde { \mathbf { q } } _ { Y _ { 1 : T } | \mathbf { I } _ { 1 : K } } \| \mathbf { p } _ { Y _ { 1 : T } } ) \right] .
$$

Then, the minimax regret can be further lower bounded by

$$
\begin{array} { r l } { \mathcal { V } _ { T } ( \mathcal { U } _ { C } ) \geq \underset { f _ { 1 : \mathcal K } } { \operatorname* { i n f } } \mathbb { E } _ { \mathbf { I } _ { 1 : \mathcal K } } [ \mathrm { K L } ( \widetilde { \mathbf { q } } _ { Y _ { 1 : T } | \mathbf { I } _ { 1 : \mathcal K } } \| \mathbf { p } _ { Y _ { 1 : T } } ) ] } \\ { = \mathbb { E } _ { \mathbf { I } _ { 1 : \mathcal K } } [ \mathrm { K L } ( \widetilde { \mathbf { q } } _ { Y _ { 1 : T } | \mathbf { I } _ { 1 : \mathcal K } } \| \widetilde { \mathbf { q } } _ { Y _ { 1 : T } } ) ] + \underset { f _ { 1 : \mathcal K } } { \operatorname* { i n f } } \mathbb { E } _ { \mathbf { I } _ { 1 : \mathcal K } } [ \displaystyle \sum _ { y _ { 1 : T } \in \mathcal { Y } ^ { T } } \widetilde { \mathbf { q } } ( y _ { 1 : T } \mid \mathbf { I } _ { 1 : \mathcal K } ) \log \frac { \widetilde { \mathbf { q } } ( y _ { 1 : T } ) } { \operatorname* { p } ( y _ { 1 : T } ) } ] } \\ { = \mathbb { E } _ { \mathbf { I } _ { 1 : \mathcal K } } [ \mathrm { K L } ( \widetilde { \mathbf { q } } _ { Y _ { 1 : T } | \mathbf { I } _ { 1 : \mathcal K } } \| \widetilde { \mathbf { q } } _ { Y _ { 1 : T } } ) ] + \underset { f _ { 1 : \mathcal K } } { \operatorname* { i n f } } \displaystyle \sum _ { y _ { 1 : T } \in \mathcal { Y } ^ { T } } \mathbb { E } _ { \mathbf { I } _ { 1 : \mathcal K } } [ \widetilde { \mathbf { q } } ( y _ { 1 : T } \mid \mathbf { I } _ { 1 : \mathcal K } ) ] \log \frac { \widetilde { \mathbf { q } } ( y _ { 1 : T } ) } { \operatorname* { p } ( y _ { 1 : T } ) } } \\  = \mathbb { E } _ { \mathbf { I } _ { 1 : \mathcal K } } [ \mathrm { K L } ( \widetilde { \mathbf { q } } _  Y _ { 1 : T } | \mathbf { I } _  \end{array}
$$

where $\bar { \mathbf { q } } ( y _ { 1 : T } ) = \mathrm { P r } ( Y _ { 1 : T } = y _ { 1 : T } ) = \mathbb { E } _ { \mathbf { I } _ { 1 : K } } [ \widetilde { \mathbf { q } } ( y _ { 1 : T } \mid \mathbf { I } _ { 1 : K } ) ]$ is the marginal probability mass function of the label sequence obtained by averaging over the environment indices. With the same abuse of notation, we use $\bar { \mathbf q } _ { Y _ { 1 : T } }$ to denote the corresponding marginal distribution. The penultimate equality holds because both probability mass functions $\bar { \bf q }$ and p are independent of $\mathbf { I } _ { 1 : K }$

Then, let $y _ { \mathbb { T } _ { k } } = \{ y _ { t } \} _ { t \in \mathbb { T } _ { k } }$ be the sequence of labels on interval $\mathcal { T } _ { k }$ , and define its conditional probability mass function by

$$
\widetilde { \mathbf { q } } _ { k } ( y _ { \mathcal { T } _ { k } } \mid \mathbf { I } _ { k } ) = \operatorname* { P r } \left( Y _ { \mathcal { T } _ { k } } = y _ { \mathcal { T } _ { k } } \mid \mathbf { I } _ { k } \right) = \prod _ { t \in \mathcal { T } _ { k } } [ \widetilde { \mathbf { u } } _ { t } ] _ { y _ { t } } .
$$

With a slight abuse of notation, we use $\widetilde { \mathbf { q } } _ { Y _ { \mathcal { T } _ { k } } | \mathbf { I } _ { k } }$ to denote the corresponding conditional distribution. Conditional on $\mathbf { I } _ { 1 : K } = { \bigl ( } \mathbf { I } _ { 1 } , \cdot \cdot \cdot , \mathbf { I } _ { K } { \bigr ) }$ , the label sequences on diferent intervals are independent. Therefore, the conditional probability mass function of the full label sequence factorizes as $\begin{array} { r } {  { \widetilde { \mathbf { q } } } ( y _ { 1 : T } \ | \ \mathbf { I } _ { 1 : K } ) = \prod _ { k = 1 } ^ { K }  { \widetilde { \mathbf { q } } } _ { k } ( y _ {  { \mathcal { T } } _ { k } } \ | \ \mathbf { I } _ { k } ) } \end{array}$ . Using the independence of the environment indices $\mathbf { I } _ { 1 } , \ldots , \mathbf { I } _ { K }$ , we further have

$$
\begin{array} { l } { \displaystyle \bar { \mathbf { q } } ( y _ { 1 : T } ) = \mathbb { E } _ { \mathbf { I } _ { 1 : K } } \left[ \widetilde { \mathbf { q } } ( y _ { 1 : T } \mid \mathbf { I } _ { 1 : K } ) \right] = \mathbb { E } _ { \mathbf { I } _ { 1 : K } } \left[ \prod _ { k = 1 } ^ { K } \widetilde { \mathbf { q } } _ { k } ( y _ { \mathcal { T } _ { k } } \mid \mathbf { I } _ { k } ) \right] } \\ { \displaystyle \quad = \prod _ { k = 1 } ^ { K } \mathbb { E } _ { \mathbf { I } _ { k } } \left[ \widetilde { \mathbf { q } } _ { k } ( y _ { \mathcal { T } _ { k } } \mid \mathbf { I } _ { k } ) \right] = \prod _ { k = 1 } ^ { K } \bar { \mathbf { q } } _ { k } ( y _ { \mathcal { T } _ { k } } ) , } \end{array}
$$

where we define the marginal probability mass function on block k as $\bar { \mathbf q } _ { k } ( y _ { \mathbb { Z } _ { k } } ) =$ $\operatorname* { P r } ( Y _ { \mathcal { T } _ { k } } = y _ { \mathcal { T } _ { k } } ) = \mathbb { E } _ { \mathbf { I } _ { k } } [ \widetilde { \mathbf { q } } _ { k } ( y _ { \mathcal { T } _ { k } } \mid \mathbf { I } _ { k } ) ]$ and use $\bar { \mathbf q } _ { Y _ { \mathcal L _ { k } } }$ to denote the corresponding marginal distribution. The minimax regret can be further bounded by

$$
\mathcal { V } _ { T } ( \mathcal { U } _ { C } ) \geq \mathbb { E } _ { \mathbf { I } _ { 1 : K } } \left[ \sum _ { k = 1 } ^ { K } { \mathrm { K L } } ( \widetilde { \mathbf { q } } _ { Y _ { T _ { k } } | \mathbf { I } _ { k } } \| \bar { \mathbf { q } } _ { Y _ { T _ { k } } } ) \right] = \sum _ { k = 1 } ^ { K } \mathbb { E } _ { \mathbf { I } _ { k } } \left[ { \mathrm { K L } } ( \widetilde { \mathbf { q } } _ { Y _ { T _ { k } } | \mathbf { I } _ { k } } \| \bar { \mathbf { q } } _ { Y _ { T _ { k } } } ) \right] ,\tag{15}
$$

where the blockwise decomposition follows from the additivity of the KL divergence for product distributions, and the equality holds because the conditional distribution $\widetilde { \mathbf { q } } _ { Y _ { \mathcal { T } _ { k } } | \mathbf { I } _ { k } }$ depends only on ${ \mathbf I } _ { k }$

We next analyze the KL divergence contributed by each block. For every $j \in [ d - 1 ]$ ， define

$$
Z _ { k , j } = \mathbb { 1 } \left\{ \sum _ { \boldsymbol { t } \in \mathcal { T } _ { k } } \mathbb { 1 } \{ Y _ { t } = j \} \geq 1 \right\} ,
$$

and write $\mathbf { Z } _ { k } = \left( Z _ { k , 1 } , \ldots , Z _ { k , d - 1 } \right)$ . Thus, $Z _ { k , j }$ indicates whether label $j$ is observed at least once on block k. Since $\mathbf { Z } _ { k }$ is a deterministic function of $Y _ { \mathcal { T } _ { k } }$ , we have

$$
\mathbb { E } _ { { \mathbf { I } } _ { k } } \left[ \mathrm { K L } \big ( \widetilde { \mathbf { q } } _ { Y _ { \mathcal { T } _ { k } } | \mathbf { I } _ { k } } \| \bar { \mathbf { q } } _ { Y _ { \mathcal { T } _ { k } } } \big ) \right] = \mathrm { I } \left( \mathbf { I } _ { k } ; Y _ { \mathcal { T } _ { k } } \right) \geq \mathrm { I } \left( \mathbf { I } _ { k } ; \mathbf { Z } _ { k } \right) \geq \sum _ { j = 1 } ^ { d - 1 } \mathrm { I } \left( I _ { k , j } ; Z _ { k , j } \right) ,\tag{16}
$$

where $\operatorname { I } ( \cdot ; \cdot )$ denotes mutual information. The first equality follows from $\operatorname { I } ( X ; Y ) =$ $\mathbb { E } _ { X } [ \mathrm { K L } ( P _ { Y | X } \| P _ { Y } ) ]$ by taking $X = \mathbf { I } _ { k }$ and $Y = Y _ { \mathbb { Z } _ { k } }$ . The first inequality follows from the data-processing inequality. The last inequality follows from the independence of the coordinates of $\mathbf { I } _ { k }$ and the fact that conditioning reduces entropy, since $H ( \mathbf { I } _ { k } \mid$ $\begin{array} { r } { \mathbf Z _ { k } ) \leq \sum _ { j = 1 } ^ { d - 1 } H ( I _ { k , j } \mid \mathbf Z _ { k } ) \leq \sum _ { j = 1 } ^ { d - 1 } H ( I _ { k , j } \mid Z _ { k , j } ) } \end{array}$

We next analyze the information contributed by each coordinate. Conditional on $I _ { k , j } = 0$ , label $j$ has zero probability, and hence $Z _ { k , j } = 0$ almost surely. Conditional on $I _ { k , j } = 1$ , label $j$ is generated with probability $\epsilon / ( d - 1 )$ at every round. We have

$$
p _ { k } : = \operatorname* { P r } ( Z _ { k , j } = 1 \mid I _ { k , j } = 1 ) = 1 - \left( 1 - { \frac { \epsilon } { d - 1 } } \right) ^ { | Z _ { k } | } \geq 1 - e ^ { - 1 } ,
$$

where the last inequality holds because $| { \mathcal { T } } _ { k } | \geq L \geq ( d - 1 ) / \epsilon$

Since $I _ { k , j } \sim$ Bern $( 1 / 2 )$ , we have $\operatorname* { P r } ( Z _ { k , j } = 1 ) = p _ { k } / 2$ and $H ( I _ { k , j } ) = \log 2$ . Moreover, conditional on $Z _ { k , j } = 1$ , the environment coordinate $I _ { k , j }$ must be equal to one, and hence $H ( I _ { k , j } \mid Z _ { k , j } = 1 ) = 0$ . Since $H ( I _ { k , j } \mid Z _ { k , j } = 0 ) \le \log 2$ , we obtain

$$
\begin{array} { r l } & { \operatorname { I } \left( I _ { k , j } ; Z _ { k , j } \right) = H ( I _ { k , j } ) - H ( I _ { k , j } \mid Z _ { k , j } ) } \\ & { \qquad \ge \log 2 - \operatorname* { P r } ( Z _ { k , j } = 0 ) \log 2 } \\ & { \qquad = \operatorname* { P r } ( Z _ { k , j } = 1 ) \log 2 = \frac { p _ { k } } { 2 } \log 2 \ge \frac { 1 - e ^ { - 1 } } { 2 } \log 2 . } \end{array}
$$

Combining the above inequalities, the KL divergence contributed by each block satisfies

$$
\mathbb { E } _ { { \mathbf { I } } _ { k } } \left[ { \mathrm { K L } } ( \widetilde { \mathbf { q } } _ { Y _ { \widetilde { L } _ { k } } | \mathbf { I } _ { k } } | | \bar { \mathbf { q } } _ { Y _ { \widetilde { L } _ { k } } } ) \right] \geq \frac { ( d - 1 ) ( 1 - e ^ { - 1 } ) } { 2 } \log 2 .
$$

Then, we can conclude that

$$
\mathcal { V } _ { T } ( \mathcal { U } _ { C } ) \geq \frac { ( d - 1 ) ( 1 - e ^ { - 1 } ) } { 2 } K \log 2 .
$$

It remains to lower bound the number of intervals K. Since $\epsilon \leq 1 / 2$ , we have $L = \lceil ( d - 1 ) / \epsilon \rceil \leq 2 ( d - 1 ) / \epsilon$ . Moreover, $\epsilon \geq ( d - 1 ) / T$ implies $L \leq T$ . Therefore,

$$
K = \left\lfloor { \frac { T } { L } } \right\rfloor \geq { \frac { T } { 2 L } } \geq { \frac { T \epsilon } { 4 ( d - 1 ) } } .
$$

It follows that, in the main regime,

$$
\mathcal { V } _ { T } ( \mathcal { U } _ { C } ) \geq \frac { 1 - e ^ { - 1 } } { 8 } T \epsilon \log 2 = \frac { ( 1 - e ^ { - 1 } ) \log 2 } { 8 \sqrt { 2 } } \sqrt { ( d - 1 ) T C } .\tag{17}
$$

Hanlding Corner Cases. It remains to handle the values of C outside the main regime.

If $C < 2 ( d - 1 ) / T$ , we use the same construction with $\epsilon _ { 0 } = ( d - 1 ) / T$ . In this case, $L = T$ and $K = 1$ , so that the comparator sequence is constant and has zero path length. The same one-block analysis gives

$$
\mathcal { V } _ { T } ( \mathcal { U } _ { C } ) \geq \frac { ( d - 1 ) ( 1 - e ^ { - 1 } ) } { 2 } \log 2 = \Omega ( d ) .
$$

Moreover, since $C < 2 ( d - 1 ) / T$ , we have $\sqrt { d T C } < \sqrt { 2 d ( d - 1 ) } = O ( d )$ . Therefore, the above one-block lower bound already dominates the desired dynamic term. Combining this one-block estimate with (17), we obtain the required $\Omega ( { \sqrt { d T C } } )$ lower bound for every $C \leq T / ( 2 ( d - 1 ) )$ .

If $C > T / ( 2 ( d - 1 ) )$ , we apply the preceding construction with the smaller budget $C _ { 0 } = T / ( 2 ( d - 1 ) )$ . Since $\mathcal { U } _ { C _ { 0 } } \subseteq \mathcal { U } _ { C }$ , the same lower bound continues to hold. Moreover, applying (17) with $C _ { 0 }$ gives

$$
\mathcal { V } _ { T } ( \mathcal { U } _ { C } ) \geq \mathcal { V } _ { T } ( \mathcal { U } _ { C _ { 0 } } ) = \Omega \left( \sqrt { ( d - 1 ) T C _ { 0 } } \right) = \Omega ( T ) .
$$

Combining these cases, we obtain

$$
\mathcal { V } _ { T } ( \mathcal { U } _ { C } ) \geq \Omega \left( \operatorname* { m i n } \left\{ T , \sqrt { d T C } \right\} \right) .\tag{18}
$$

On the other hand, since any fixed comparator is also contained by $\displaystyle { \mathcal { U } } _ { C } .$ , the standard lower bound for sequential probability assignment(Cesa-Bianchi and Lugosi, 2006, Chapter 9.1) shows that $V _ { T } ( \mathcal { U } _ { C } ) = \Omega ( d \log T )$ for $C = 0$ , which leads to the lower bound of $\mathcal { V } _ { T } ( \mathcal { U } _ { C } ) \geq \Omega ( \operatorname* { m a x } \{ d \log T $ , min $\big \{ T , \sqrt { d T C } \big \} \big \}$ , which completes the proof. ■

## B.2. Proof of Lemma 2

Proof of Lemma 2 For any interval $\mathcal { T } = [ s , e ] \subseteq [ T ]$ , let $\begin{array} { r } { \mathbf { w } _ { \mathcal { T } } = \arg \operatorname* { m i n } _ { \mathbf { w } \in \Delta _ { d } } \sum _ { t \in \mathcal { T } } \ell _ { t } ( \mathbf { w } ) } \end{array}$ be the optimal fixed prediction on interval I. Then, for any sequence $\{ { \mathbf { u } } _ { t } \} _ { t \in \mathbb { Z } } ,$ we have

$$
\sum _ { t \in \mathcal { T } } \ell _ { t } ( \mathbf { w } _ { \mathcal { T } } ) - \sum _ { t \in \mathcal { T } } \ell _ { t } ( \mathbf { u } _ { t } ) = \sum _ { t \in \mathcal { T } } \log \left( \frac { \mathbf { u } _ { t } ^ { \top } \mathbf { x } _ { t } } { \mathbf { w } _ { \mathcal { T } } ^ { \top } \mathbf { x } _ { t } } \right) = \log \left( \frac { \prod _ { t \in \mathcal { T } } \mathbf { u } _ { t } ^ { \top } \mathbf { x } _ { t } } { \prod _ { t \in \mathcal { T } } \mathbf { w } _ { \mathcal { T } } ^ { \top } \mathbf { x } _ { t } } \right) .\tag{19}
$$

Let $z _ { i } ^ { \mathrm { m a x } } = \mathrm { m a x } _ { t \in \mathcal { T } } \{ u _ { t , i } \}$ , and define $\mathbf { z } _ { \mathrm { m a x } } = [ z _ { 1 } ^ { \mathrm { m a x } } , \dots , z _ { d } ^ { \mathrm { m a x } } ] ^ { \top }$ together with its normalized version $\bar { \mathbf { u } } = \mathbf { z } _ { \operatorname* { m a x } } / \Vert \mathbf { z } _ { \operatorname* { m a x } } \Vert _ { 1 }$ . Since $\mathbf { x } _ { t } \in \mathbb { R } _ { + } ^ { d }$ , we can further upper bound (19) by

$$
\begin{array} { r l } & { \log \left( \displaystyle \frac { \prod _ { t \in \mathcal { T } } \mathbf { u } _ { t } ^ { \top } \mathbf { x } _ { t } } { \prod _ { t \in \mathcal { T } } \mathbf { w } _ { T } ^ { \top } \mathbf { x } _ { t } } \right) \leq \log \left( \frac { \prod _ { t \in \mathcal { T } } \mathbf { z } _ { \operatorname* { m a x } } ^ { \top } \mathbf { x } _ { t } } { \prod _ { t \in \mathcal { T } } \mathbf { w } _ { T } ^ { \top } \mathbf { x } _ { t } } \right) } \\ & { \qquad = \log \left( \displaystyle \frac { \prod _ { t \in \mathcal { T } } \bar { \mathbf { u } } ^ { \top } \mathbf { x } _ { t } } { \prod _ { t \in \mathcal { T } } \mathbf { w } _ { T } ^ { \top } \mathbf { x } _ { t } } \right) + | \mathcal { Z } | \log \| \mathbf { z } _ { \operatorname* { m a x } } \| _ { 1 } } \\ & { \qquad \leq | \mathcal { Z } | \log \| \mathbf { z } _ { \operatorname* { m a x } } \| _ { 1 } , } \end{array}\tag{20}
$$

where the last inequality holds because $\mathbf { w } _ { \mathcal { T } }$ minimizes the cumulative loss over the interval I. For each coordinate of $\mathbf { z } _ { \mathrm { m a x } }$ , we have $\begin{array} { r } { z _ { i } ^ { \operatorname* { m a x } } \leq u _ { s , i } + \sum _ { t = s + 1 } ^ { e } [ u _ { t , i } - u _ { t - 1 , i } ] + ; } \end{array}$ which implies

$$
\begin{array} { l } { \| \mathbf { z } _ { \operatorname* { m a x } } \| _ { 1 } \leq \displaystyle \sum _ { i = 1 } ^ { d } u _ { s , i } + \displaystyle \sum _ { t = s + 1 } ^ { e } \sum _ { i = 1 } ^ { d } [ u _ { t , i } - u _ { t - 1 , i } ] _ { + } } \\ { = 1 + \displaystyle \frac { 1 } { 2 } \sum _ { t = s + 1 } ^ { e } \| \mathbf { u } _ { t } - \mathbf { u } _ { t - 1 } \| _ { 1 } = 1 + \displaystyle \frac { 1 } { 2 } P _ { \mathbb { Z } } , } \end{array}\tag{21}
$$

where $\begin{array} { r } { P _ { \mathbb { Z } } = \sum _ { t = s + 1 } ^ { e } \| \mathbf { u } _ { t } - \mathbf { u } _ { t - 1 } \| _ { 1 } } \end{array}$ . The equality uses the simplex identity that $\scriptstyle \sum _ { i = 1 } ^ { d } [ v _ { i } ] _ { + }$ equals ${ \frac { 1 } { 2 } } \| v \| _ { 1 }$ whenever $\begin{array} { r } { \sum _ { i = 1 } ^ { d } v _ { i } = 0 } \end{array}$ , applied to $v = \mathbf u _ { t } - \mathbf u _ { t - 1 }$

Combining (19), (20), and (21) with condition (2), we obtain that for any interval $\mathcal { T } \subseteq [ T ]$ 2

$$
\sum _ { t \in \mathcal { T } } \ell _ { t } ( \mathbf { w } _ { t } ) - \sum _ { t \in \mathcal { T } } \ell _ { t } ( \mathbf { u } _ { t } ) \leq B ( T ) + | \mathcal { Z } | \log \left( 1 + \frac { P _ { \mathcal { T } } } { 2 } \right) \leq B ( T ) + \frac { | \mathcal { Z } | P _ { \mathcal { T } } } { 2 } .\tag{22}
$$

We now construct a partition into consecutive maximal blocks. Set $s _ { 1 } = 1$ . Given the start $s _ { m } .$ , let $e _ { m }$ be the largest index $e \in \{ s _ { m } , \ldots , T \}$ for which the product $( e - s _ { m } + 1 ) P _ { [ s _ { m } , e ] }$ is at most $B ( T )$ . Such an index always exists because $P _ { [ s _ { m } , s _ { m } ] } = 0$ If $e _ { m } < T$ , set $s _ { m + 1 } = e _ { m } + 1$ and continue; otherwise stop. This produces a partition $\{ \mathcal { T } _ { m } = [ s _ { m } , e _ { m } ] \} _ { m = 1 } ^ { M }$ of [T], and every block satisfies $| \mathcal { T } _ { m } | P _ { \mathcal { T } _ { m } } \leq B ( T )$ . Therefore, (22) gives

$$
\sum _ { t = 1 } ^ { T } \ell _ { t } ( \mathbf { w } _ { t } ) - \sum _ { t = 1 } ^ { T } \ell _ { t } ( \mathbf { u } _ { t } ) = \sum _ { m = 1 } ^ { M } \left( \sum _ { t \in \mathcal { T } _ { m } } \ell _ { t } ( \mathbf { w } _ { t } ) - \sum _ { t \in \mathcal { T } _ { m } } \ell _ { t } ( \mathbf { u } _ { t } ) \right) \leq \frac { 3 } { 2 } M B ( T ) .\tag{23}
$$

If $M \ = \ 1$ , (23) already gives the required bound $\mathrm { D - R e g } _ { T } ( \{ \mathbf { u } _ { t } \} _ { t = 1 } ^ { T } ) \ \leq \ \frac { 3 } { 2 } B ( T )$ Henceforth, assume $M > 1$ . It remains to bound M. Write $L _ { m } = \left| \mathcal { T } _ { m } \right|$ . For every nonfinal block, maximality implies that the product $( L _ { m } + 1 ) P _ { [ s _ { m } , e _ { m } + 1 ] }$ exceeds $B ( T )$ for $m = 1 , \ldots , M - 1$ . The increments in $P _ { [ s _ { m } , e _ { m } + 1 ] }$ are indexed by $t = s _ { m } + 1 , \ldots , e _ { m } + 1$ Those in the next extended block begin at $t = s _ { m + 1 } + 1 = e _ { m } + 2$ , so the sets of increments are disjoint. Consequently, the sum $\begin{array} { r } { \sum _ { m = 1 } ^ { M - 1 } P _ { [ s _ { m } , e _ { m } + 1 ] } } \end{array}$ is at most $P _ { T }$ Moreover, since every $\begin{array} { r } { L _ { m } \geq 1 , \sum _ { m = 1 } ^ { M - 1 } ( L _ { m } + 1 ) } \end{array}$ is at most $2 \textstyle \sum _ { m = 1 } ^ { M - 1 } L _ { m }$ , which is at most 2T. Taking square roots in the nonfinal-block inequality and summing gives the first line below. Cauchy–Schwarz and the two preceding sum bounds then yield

$$
\begin{array} { r l r } {  { ( M - 1 ) \sqrt { B ( T ) } < \sum _ { m = 1 } ^ { M - 1 } \sqrt { ( L _ { m } + 1 ) P _ { [ s _ { m } , e _ { m } + 1 ] } } } } \\ & { } & { \leq \sqrt { ( \sum _ { m = 1 } ^ { M - 1 } ( L _ { m } + 1 ) ) ( \displaystyle \sum _ { m = 1 } ^ { M - 1 } P _ { [ s _ { m } , e _ { m } + 1 ] } ) } } \\ & { } & { \leq \sqrt { 2 T P _ { T } } . } \end{array}
$$

Thus $M \leq 1 + \sqrt { 2 T P _ { T } / B ( T ) }$ . Substitution into (23) shows that $\mathrm { D - R e g } _ { T } \big ( \{ \mathbf { u } _ { t } \} _ { t = 1 } ^ { T } \big )$ is at most $\begin{array} { r } { \frac { 3 } { 2 } B ( T ) + \frac { 3 } { \sqrt { 2 } } \sqrt { B ( T ) T P _ { T } } } \end{array}$ , which is $\mathcal { O } \Big ( B ( T ) + \sqrt { B ( T ) T P _ { T } } \Big )$ , this completes the proof. 7

## Appendix C. Omitted Proofs for Section 3.2

This section presents the omitted proofs of Theorems 3 and 4. We first establish several auxiliary lemmas and then present the main proofs.

## C.1. Useful Lemmas

Lemma 8 For any distribution $Q _ { t }$ with $\operatorname { s u p p } ( Q _ { t } ) \subseteq \operatorname { s u p p } ( P _ { t } )$ , it holds that

$$
\ell _ { t } ( { \mathbf { w } } _ { t } ) = \mathbb { E } _ { { \mathbf { u } } \sim Q _ { t } } [ \ell _ { t } ( { \mathbf { u } } ) ] + { \mathrm { K L } } ( Q _ { t } \parallel P _ { t } ) - { \mathrm { K L } } ( Q _ { t } \parallel \tilde { P } _ { t + 1 } ) ,
$$

where $\tilde { P } _ { t + 1 } ( \mathbf { u } ) \propto P _ { t } ( \mathbf { u } ) \exp ( - \ell _ { t } ( \mathbf { u } ) )$ for all $\mathbf { u } \in \mathrm { s u p p } ( P _ { t } )$

Proof of Lemma 8 This lemma holds by an exact identity for distributions defined over $\Delta _ { d }$ . Let $Z _ { t } = \mathbb { E } _ { \mathbf { w } \sim P _ { t } } [ \exp ( - \ell _ { t } ( \mathbf { w } ) ) ]$ , we have

$$
\begin{array} { r l } & { \mathrm { K L } ( Q _ { t }  P _ { t } ) - \mathrm { K L } ( Q _ { t }  \tilde { P } _ { t + 1 } ) = \mathbb { E } _ { \mathbf { u } \sim Q _ { t } } [ \mathrm { l n } ( \frac { \tilde { P } _ { t + 1 } ( \mathbf { u } ) } { P _ { t } ( \mathbf { u } ) } ) ] } \\ & { \qquad = \mathbb { E } _ { \mathbf { u } \sim Q _ { t } } [ \mathrm { l n } ( \frac { \exp ( - \ell _ { t } ( \mathbf { u } ) ) } { Z _ { t } } ) ] } \\ & { \qquad = \ - \ \mathbb { E } _ { \mathbf { u } \sim Q _ { t } } [ \ell _ { t } ( \mathbf { u } ) ] - \mathrm { l n } Z _ { t } } \\ & { \qquad = \ - \ \mathbb { E } _ { \mathbf { u } \sim Q _ { t } } [ \ell _ { t } ( \mathbf { u } ) ] + \ell _ { t } ( \mathbf { w } _ { t } ) , } \end{array}
$$

where the second equality follows from $\tilde { P } _ { t + 1 } ( \mathbf { u } ) = P _ { t } ( \mathbf { u } ) \exp ( - \ell _ { t } ( \mathbf { u } ) ) / Z _ { t }$

Lemma 16 Let $P _ { t }$ be the distribution updated by Algorithm 1 with $\mu _ { t } = 1 / t$ . Then, ln $P _ { t } ( \mathbf { u } ) \leq ( d - 1 ) \ln ( T + 1 ) + d ( \ln d + 1 )$ for all $t \in [ T ]$ and $\mathbf { u } \in \Delta _ { d }$

Proof of Lemma 16 According to the EW and fixed-share update steps in Algorithm 1, for any $\mathbf { w } \in \Delta _ { d }$ , we have

$$
P _ { t + 1 } ( \mathbf { w } ) = ( 1 - \mu _ { t + 1 } ) \tilde { P } _ { t + 1 } ( \mathbf { w } ) + \mu _ { t + 1 } P _ { 1 } ( \mathbf { w } ) = \frac { ( 1 - \mu _ { t + 1 } ) e ^ { - \ell _ { t } ( \mathbf { w } ) } } { \mathbb { E } _ { \mathbf { u } \sim P _ { t } } \left[ e ^ { - \ell _ { t } ( \mathbf { u } ) } \right] } P _ { t } ( \mathbf { w } ) + \mu _ { t + 1 } P _ { 1 } ( \mathbf { w } ) .
$$

Unrolling this recursion shows that $P _ { t + 1 }$ is a convex combination of posterior distributions initialized at the diferent restart times. Specifically, define

$$
P _ { t + 1 , i } ( \mathbf { w } ) = \frac { 1 } { W _ { t + 1 , i } } P _ { 1 } ( \mathbf { w } ) e ^ { - \sum _ { s = i } ^ { t } \ell _ { s } ( \mathbf { w } ) } \mathrm { ~ f o r ~ a l l ~ } i \leq t + 1 ,\tag{24}
$$

where $W _ { t + 1 , i } = \mathbb { E } _ { \mathbf { u } \sim P _ { 1 } } [ e ^ { - \sum _ { s = i } ^ { t } \ell _ { s } ( \mathbf { u } ) } ]$ is the normalization factor and the empty sum is zero when $i = t + 1$ . An induction on t gives nonnegative weights $\alpha _ { t + 1 , 1 } , \ldots , \alpha _ { t + 1 , t + 1 }$ that sum to one and satisfy

$$
P _ { t + 1 } ( \mathbf { w } ) = \sum _ { i = 1 } ^ { t + 1 } \alpha _ { t + 1 , i } P _ { t + 1 , i } ( \mathbf { w } ) \leq \operatorname* { m a x } _ { i \in [ t + 1 ] } P _ { t + 1 , i } ( \mathbf { w } ) .\tag{25}
$$

For $i = t + 1$ , we have $P _ { t + 1 , t + 1 } ( \mathbf { w } ) = P _ { 1 } ( \mathbf { w } )$ for all $\mathbf { w } \in \Delta _ { d }$ and ln $P _ { t + 1 , t + 1 } ( \mathbf { w } ) =$ ln $\Gamma ( d )$ since $P _ { 1 }$ is a uniform distribution on $\Delta _ { d }$ . Then, to prove the lemma it is

suficient to upper bound ln $P _ { t + 1 , i } ( \mathbf { w } )$ for all $i \in [ t ]$ and $\mathbf { w } \in \Delta _ { d } .$ . According to the definition (24), the distribution $P _ { t + 1 , i } ( \mathbf { w } ) \propto P _ { t , i } ( \mathbf { w } ) \cdot \exp ( - \ell _ { t } ( \mathbf { w } ) )$ for all $i \leq t$ . Then, we have

$$
\begin{array} { l } { { \displaystyle \ln P _ { t + 1 , i } ( { \bf w } ) = \ln P _ { t , i } ( { \bf w } ) - \ln ( \mathbb { E } _ { { \bf u } \sim P _ { t , i } } [ e ^ { - \ell _ { s } ( { \bf u } ) } ] ) - \ell _ { t } ( { \bf w } ) } \ ~ } \\ { { \displaystyle ~ = \ln P _ { 1 } ( { \bf w } ) - \sum _ { s = i } ^ { t } \ln ( \mathbb { E } _ { { \bf u } \sim P _ { s , i } } [ e ^ { - \ell _ { s } ( { \bf u } ) } ] ) - \sum _ { s = i } ^ { t } \ell _ { s } ( { \bf w } ) } \ ~ } \\ { { \displaystyle ~ = \ln P _ { 1 } ( { \bf w } ) + \sum _ { s = i } ^ { t } \mathbb { E } _ { { \bf u } \sim Q } [ \ell _ { s } ( { \bf u } ) ] - \sum _ { s = i } ^ { t } \ell _ { s } ( { \bf w } ) } \ ~ } \\ { { \displaystyle ~ + \mathrm { K L } \left( Q \parallel P _ { 1 } \right) - \mathrm { K L } \left( Q \parallel P _ { t + 1 , i } \right) } . } \end{array}\tag{26}
$$

for any distribution $Q$ over $\Delta _ { d }$ whose support is contained in that of $P _ { 1 }$ . The second equality is due to the recursive definition of $P _ { t , i }$ . Let

$$
\mathbf { w } _ { * } ^ { t + 1 , i } = \underset { \mathbf { w } \in \Delta _ { d } } { \arg \operatorname* { m a x } } \ln P _ { t + 1 , i } ( \mathbf { w } ) , \qquad Q _ { * } ^ { t + 1 , i } = \mathrm { D i r } ( \mathbf { 1 } + T \mathbf { w } _ { * } ^ { t + 1 , i } ) .
$$

Then, the same arguments used to upper bound term (a) in the proof of Theorem 3 yield

$$
\sum _ { s = i } ^ { t } \mathbb { E } _ { \mathbf { u } \sim Q _ { * } ^ { t + 1 , i } } [ \ell _ { s } ( \mathbf { u } ) ] - \sum _ { s = i } ^ { t } \ell _ { s } ( \mathbf { w } _ { * } ^ { t + 1 , i } ) \leq T \ln \left( 1 + \frac { d } { T } \right) \leq d .
$$

Besides, based on Lemma 15, we have $\mathrm { K L } ( Q _ { * } ^ { t + 1 , i } \parallel P _ { 1 } ) \leq ( d - 1 ) \ln ( 1 + T )$ . Plugging the above inequalities into (26) with the fact that ln $P _ { 1 } ( \mathbf { w } ) = \ln \Gamma ( d )$ yields

$$
\begin{array} { r } { \ln P _ { t + 1 , i } ( \mathbf w ) \leq \ln \Gamma ( d ) + d + ( d - 1 ) \ln ( 1 + T ) } \\ { \leq ( d - 1 ) \ln ( 1 + T ) + d ( \ln d + 1 ) } \end{array}
$$

for any $\mathbf { w } \in \Delta _ { d }$ and $i \in [ t ]$ . Finally, (25) gives

$$
P _ { t + 1 } ( \mathbf { w } ) \leq \operatorname* { m a x } _ { i \in [ t + 1 ] } P _ { t + 1 , i } ( \mathbf { w } ) ,
$$

which completes the proof.

Lemma 17 Let $\gamma > 0$ and $Q _ { \mathbf { u } } = \mathrm { D i r } ( \mathbf { 1 } + \gamma \mathbf { u } ) ~ f o r ~ \mathbf { u } \in \Delta _ { d }$ . Then, for any u, $\mathbf { v } \in \Delta _ { d }$

$$
\int _ { \Delta _ { d } } | { \cal Q } _ { \bf u } ( { \bf w } ) - { \cal Q } _ { \bf v } ( { \bf w } ) | \mathrm { d } { \bf w } \leq \operatorname* { m i n } \left\{ 2 , 2 \sqrt { 2 \gamma } \mathrm { J S } ( { \bf u } , { \bf v } ) \right\} .
$$

Proof of Lemma 17 For u, $\mathbf { v } \in \Delta _ { d } ,$ let $\mathbf { m } = ( \mathbf { u } + \mathbf { v } ) / 2$ . We have

$$
\begin{array} { r } { \| Q _ { \mathbf { u } } - Q _ { \mathbf { v } } \| _ { 1 } \leq \sqrt { 2 \mathrm { K L } ( Q _ { \mathbf { m } } \| Q _ { \mathbf { u } } ) } + \sqrt { 2 \mathrm { K L } ( Q _ { \mathbf { m } } \| Q _ { \mathbf { v } } ) } } \\ { = \sqrt { 2 B _ { F } ( \gamma \mathbf { u } \| \gamma \mathbf { m } ) } + \sqrt { 2 B _ { F } ( \gamma \mathbf { v } \| \gamma \mathbf { m } ) } , } \end{array}\tag{27}
$$

where the first line follows from the triangle and Pinsker inequalities, and the second follows from van der Hoeven et al. (2018, Lemma 10) since Dirichlet distributions form an exponential family. Here $B _ { F } ( { \mathbf a } | | { \mathbf b } ) = F ( { \mathbf a } ) - F ( { \mathbf b } ) - \langle \nabla F ( { \mathbf b } ) , { \mathbf a } - { \mathbf b } \rangle$ is the Bregman divergence induced by the Dirichlet log-partition function

$$
F ( \mathbf { z } ) = \sum _ { i = 1 } ^ { d } \ln \Gamma ( 1 + z _ { i } ) - \ln \Gamma \left( d + \sum _ { i = 1 } ^ { d } z _ { i } \right) .
$$

We next bound $B _ { F }$ by the KL divergence. Let $\psi$ denote the digamma function. Since the trigamma function satisfies $\psi ^ { \prime } ( 1 + x ) \leq 1 / x$ for $x ~ > ~ 0$ , the function $h ( x ) = x \ln x - \ln \Gamma ( 1 + x )$ is convex on $( 0 , \infty )$ . The first-order convexity inequality $h ( a ) \geq h ( b ) + h ^ { \prime } ( b ) ( a - b )$ indicates that

$$
\ln { \frac { \Gamma ( 1 + a ) } { \Gamma ( 1 + b ) } } - ( a - b ) \psi ( 1 + b ) \leq a \ln { \frac { a } { b } } - a + b\tag{28}
$$

for any $a , b > 0$ . With the convention 0 ln $0 = 0$ , both sides are continuous in a at zero, so the inequality also holds for $a = 0$ . Since $m _ { i } = 0$ implies $u _ { i } = v _ { i } = 0$ such coordinates contribute zero to $B _ { F } ( \gamma \mathbf { u } \| \gamma \mathbf { m } )$ . We therefore only need to consider coordinates with $m _ { i } > 0$ . Applying (28) coordinatewise, we obtain

$$
\begin{array} { r l r } {  { B _ { F } \big ( \gamma \mathbf { u } \| \gamma \mathbf { m } \big ) = \sum _ { i : m _ { i } > 0 } [ \ln \frac { \Gamma ( 1 + \gamma u _ { i } ) } { \Gamma ( 1 + \gamma m _ { i } ) } - \gamma ( u _ { i } - m _ { i } ) \psi ( 1 + \gamma m _ { i } ) ] } } \\ & { } & { \leq \gamma \sum _ { i : m _ { i } > 0 } [ u _ { i } \ln \frac { u _ { i } } { m _ { i } } - u _ { i } + m _ { i } ] = \gamma \mathrm { K L } ( \mathbf { u } \| \mathbf { m } ) , } \end{array}
$$

where the first and last equalities use $\begin{array} { r } { \sum _ { i = 1 } ^ { d } u _ { i } = \sum _ { i = 1 } ^ { d } m _ { i } = 1 } \end{array}$ . The same argument applies to v.

Substituting the above displayed inequality into (27) and applying the Cauchy-Schwarz inequality yields

$$
\begin{array} { r l } & { \| Q _ { \mathbf { u } } - Q _ { \mathbf { v } } \| _ { 1 } \leq \sqrt { 2 \gamma \mathrm { K L } ( \mathbf { u } \| \mathbf { m } ) } + \sqrt { 2 \gamma \mathrm { K L } ( \mathbf { v } \| \mathbf { m } ) } } \\ & { \qquad \leq 2 \sqrt { \gamma \big ( \mathrm { K L } ( \mathbf { u } \| \mathbf { m } ) + \mathrm { K L } ( \mathbf { v } \| \mathbf { m } ) \big ) } } \\ & { \qquad = 2 \sqrt { 2 \gamma } \mathrm { J S } ( \mathbf { u } , \mathbf { v } ) . } \end{array}
$$

Finally, we complete the proof by noting that $\| Q _ { \mathbf { u } } - Q _ { \mathbf { v } } \| _ { 1 } \leq 2$ for any u, $\mathbf { v } \in \Delta _ { d }$

## C.2. Proof of Theorem 3

Proof of Theorem 3 Let $P _ { 1 } = \mathrm { D i r } ( \mathbf { 1 } )$ . According to the definition of the loss function ${ \ell _ { t } } ( { \mathbf w } ) = - \ln ( { \mathbf w } ^ { { \top } } { \mathbf x } _ { t } )$ and the prediction $\mathbf { w } _ { t } = \mathbb { E } _ { \mathbf { w } \sim P _ { t } } [ \mathbf { w } ]$ , we have

$$
\begin{array} { r l } & { \ell _ { t } ( \mathbf w _ { t } ) = \mathbf \Psi - \ln \left( \mathbb { E } _ { \mathbf w \sim P _ { t } } [ \exp ( - \ell _ { t } ( \mathbf w ) ) ] \right) } \\ & { \qquad = \mathbb { E } _ { \mathbf w \sim Q _ { t } } [ \ell _ { t } ( \mathbf w ) ] + \mathrm { K L } \left( Q _ { t } \parallel P _ { t } \right) - \mathrm { K L } \left( Q _ { t } \parallel \tilde { P } _ { t + 1 } \right) } \\ & { \qquad = \mathbb { E } _ { \mathbf w \sim Q _ { t } } [ \ell _ { t } ( \mathbf w ) ] + \mathrm { K L } \left( Q _ { t } \parallel P _ { t } \right) - \mathrm { K L } \left( Q _ { t } \parallel P _ { t + 1 } \right) + \mathbb { E } _ { \mathbf w \sim Q _ { t } } \left[ \ln \frac { \tilde { P } _ { t + 1 } ( \mathbf w ) } { P _ { t + 1 } ( \mathbf w ) } \right] } \\ & { \qquad \leq \mathbb { E } _ { \mathbf w \sim Q _ { t } } [ \ell _ { t } ( \mathbf w ) ] + \mathrm { K L } \left( Q _ { t } \parallel P _ { t } \right) - \mathrm { K L } \left( Q _ { t } \parallel P _ { t + 1 } \right) + \ln \frac { 1 } { 1 - \mu _ { t + 1 } } } \end{array}
$$

where $\begin{array} { r } { \tilde { P } _ { t + 1 } ( \mathbf { w } ) \propto P _ { t } ( \mathbf { w } ) \exp ( - \ell _ { t } ( \mathbf { w } ) ) } \end{array}$ for all $\mathbf { w } \in \Delta _ { d }$ . In the above, the last inequality holds since $P _ { t + 1 } ( \mathbf { w } ) = ( 1 - \mu _ { t + 1 } ) \tilde { P } _ { t + 1 } ( \mathbf { w } ) + \mu _ { t + 1 } P _ { 1 } ( \mathbf { w } )$ for all $\mathbf { w } \in \Delta _ { d }$ and $t \geq 1$ . We can further upper bound the term ln $( 1 / ( 1 - \mu _ { t + 1 } ) ) \leq 1 / t$ by the setting $\mu _ { t + 1 } = 1 / ( 1 + t )$ Taking the sum from $t = 1$ to $T$ rounds and rearranging the terms yields

$$
\begin{array} { r l } & { \displaystyle \sum _ { t = 1 } ^ { T } \ell _ { t } ( \mathbf { w } _ { t } ) \leq \sum _ { \underbrace { t = 1 } _ { \mathrm { \ t e r n ~ ( a ) } } } ^ { T } \mathbb { E } _ { \mathbf { w } \sim Q _ { t } } [ \ell _ { t } ( \mathbf { w } ) ] } \\ & { \quad \quad + \underbrace { \sum _ { t = 2 } ^ { T } ( \mathrm { K L } ( Q _ { t } \| P _ { t } ) - \mathrm { K L } ( Q _ { t - 1 } \| P _ { t } ) ) } _ { \mathrm { t e r n ~ ( b ) } } } \\ & { \quad \quad + \underbrace { \mathrm { K L } ( Q _ { 1 } \| P _ { 1 } ) } _ { \mathrm { t e r n ~ ( c ) } } + ( 1 + \ln T ) . } \end{array}
$$

The next step is to choose $Q _ { t }$ to make the bound tight. Here, we use the Dirichlet distribution $Q _ { t } = \operatorname * { D i r } ( { \bf 1 } + \gamma { \bf u } _ { t } )$ , where $\gamma > 0$ is a parameter to be tuned later and $\mathbf { u } _ { t }$ is the time-varying comparator sequence. Next, we bound the three terms separately.

Bounding term (a). For each round $t \in [ T ]$ , the return vector $\mathbf { x } _ { t }$ has at least one nonzero entry. For simplicity, we assume that the first $\bar { d } _ { t }$ entries of $\mathbf { x } _ { t }$ are nonzero. Besides, for each dimension $i \in [ d ]$ , let $\{ Z _ { t , i } \} _ { i = 1 } ^ { d }$ be independent random variables with $Z _ { t , i } \sim \mathrm { G a m m a } ( 1 + \gamma u _ { t , i } , 1 )$ , and define $\begin{array} { r } { Z _ { t , 0 } = \sum _ { i = 1 } ^ { d } Z _ { t , i } \sim \mathrm { G a m m a } ( d + \gamma , 1 ) } \end{array}$ It is known that the random vector $( Z _ { t , 1 } / Z _ { t , 0 } , \ldots , Z _ { t , d } / Z _ { t , 0 } )$ follows the Dirichlet

distribution $\mathrm { D i r } ( { \bf 1 } + \gamma { \bf u } _ { t } )$ . We have

$$
\begin{array} { r l } { \mathrm { c e r r a t ~ ( a ) } = } & { \displaystyle \sum _ { i = 1 } ^ { T } \mathbb { E } _ { Z _ { i + 1 } , \ldots , Z _ { i , i } } \left[ - \mathrm { l i } \left( \displaystyle \sum _ { i = 1 } ^ { \theta } x _ { i } , Z _ { i + 1 } \right) \right] } \\ & { = \displaystyle \sum _ { i = 1 } ^ { T } \mathbb { E } _ { Z _ { i + 1 } , \ldots , Z _ { i , i } } \left[ \mathrm { l i } ( Z _ { i , 0 } ) \right] + \displaystyle \sum _ { i = 1 } ^ { T } \mathbb { E } _ { Z _ { i + 1 } , \ldots , Z _ { i , i } } \left[ - \mathrm { l i } \left( \displaystyle \sum _ { i = 1 } ^ { \theta } x _ { i } , Z _ { i + 1 } \right) \right] } \\ & { = \displaystyle \sum _ { i = 1 } ^ { T } \psi ( d + \gamma _ { i } ) + \sum _ { i = 1 } ^ { T } \mathbb { E } _ { Z _ { i + 1 } , \ldots , Z _ { i , i } } \left[ - \mathrm { l i } \left( \displaystyle \sum _ { i = 1 } ^ { \theta } x _ { i } , Z _ { i + 1 } \right) \right] } \\ & { \leq \displaystyle \sum _ { i = 1 } ^ { T } \psi ( d + \gamma _ { i } ) + \sum _ { i = 1 } ^ { T } \mathbb { E } _ { Z _ { i + 1 } , \ldots , Z _ { i , i } } \left[ - \displaystyle \sum _ { i = 1 } ^ { \theta } \eta _ { i } , \mathbf { i } \left( x _ { i } , Z _ { i + 1 } \right/ p _ { i , * } \right) \right] } \\ & { = \displaystyle \sum _ { i = 1 } ^ { T } \psi ( d + \gamma _ { i } ) - \displaystyle \sum _ { i = 1 } ^ { T } \left( \displaystyle \sum _ { i = 1 } ^ { \theta } p _ { s _ { i } } ( \mathrm { l i } x _ { i } , + \psi ( 1 + \gamma _ { i } u _ { i } ) ) - \displaystyle \sum _ { i = 1 } ^ { \theta } p _ { s , \mathrm { l i } } \left( p _ { s , i } , \mathbf { i } \left( p _ { s , i } , \right) \right) . } \end{array}\tag{29}
$$

In the third line, we use the identity $\mathbb { E } [ \ln ( Z _ { t , 0 } ) ] = \psi ( d + \gamma )$ for $Z _ { t , 0 } \sim \mathrm { G a m m a } ( d + \gamma , 1 )$ and note that $\begin{array} { r } { \sum _ { i = 1 } ^ { d } x _ { t , i } Z _ { t , i } = \sum _ { i = 1 } ^ { \bar { d } _ { t } } x _ { t , i } Z _ { t , i } } \end{array}$ <sub>i</sub> since the remaining entries are zero. For the fourth line, since $x _ { t , i } Z _ { t , i } > 0$ for all $i \in [ \bar { d } _ { t } ]$ and $t \in [ T ]$ , we apply the inequality ln $\left( \textstyle \sum _ { i = 1 } ^ { \bar { d } _ { t } } a _ { i } \right) \ge \sum _ { i = 1 } ^ { \bar { d } _ { t } } p _ { i }$ ln $\left( { \frac { a _ { i } } { p _ { i } } } \right)$ , which holds for all $a _ { i } > 0$ and any probability vector $\mathbf { p } = ( p _ { 1 } , \dots , p _ { \bar { d } _ { t } } ) \in \mathrm { r i } ( \Delta _ { \bar { d } _ { t } } )$ , where $\mathrm { r i } ( \Delta _ { \bar { d } _ { t } } )$ denotes the relative interior of the simplex $\Delta _ { \bar { d } _ { t } }$

Then, we can tune the probability vector $\mathbf { p } _ { t }$ to make the bound (29) tight. The goal is to solve the optimization problem

$$
V _ { t } ^ { * } = \operatorname* { m a x } _ { \mathbf { p } _ { t } \in \mathrm { r i } ( \Delta _ { \bar { d } _ { t } } ) } \sum _ { i = 1 } ^ { \bar { d } _ { t } } p _ { t , i } \big ( \ln x _ { t , i } + \psi ( 1 + \gamma u _ { t , i } ) \big ) - \sum _ { i = 1 } ^ { \bar { d } _ { t } } p _ { t , i } \ln ( p _ { t , i } ) ,
$$

which has the closed-form solution by $\begin{array} { r } { V _ { t } ^ { * } = \ln \left( \sum _ { i = 1 } ^ { \bar { d } _ { t } } x _ { t , i } \exp ( \psi ( 1 + \gamma u _ { t , i } ) ) \right) } \end{array}$ achieved at $p _ { t , i } ^ { * } \propto x _ { t , i } \exp ( \psi ( 1 + \gamma u _ { t , i } ) )$ . Plugging the optimal solution back to (29) yields

$$
\begin{array} { r l } { \mathbf { t e r m \thinspace \thinspace \thinspace ( a ) \leq \sum _ { t = 1 } ^ { T } } \psi ( d + \gamma ) - \displaystyle \sum _ { t = 1 } ^ { T } \ln \left( \displaystyle \sum _ { i = 1 } ^ { d _ { t } } x _ { t , i } \exp ( \psi ( 1 + \gamma u _ { t , i } ) ) \right) } \\ { \displaystyle \leq \sum _ { t = 1 } ^ { T } \psi ( d + \gamma ) - \displaystyle \sum _ { t = 1 } ^ { T } \ln \left( \gamma \displaystyle \sum _ { i = 1 } ^ { d _ { t } } x _ { t , i } u _ { t , i } \right) } \\ { \displaystyle = \sum _ { t = 1 } ^ { T } \ell _ { t } ( \mathbf { u } _ { t } ) + T ( \psi ( d + \gamma ) - \ln \gamma ) } \\ { \displaystyle \leq \sum _ { t = 1 } ^ { T } \ell _ { t } ( \mathbf { u } _ { t } ) + T \ln \left( 1 + \frac { d } { \gamma } \right) } \end{array}
$$

where the second line holds since $\psi ( 1 + x ) \geq$ ln x for all $x > 0$ . The second inequality also holds for the case $u _ { t , i } = 0$ since $\exp ( \psi ( 1 ) ) > 0$ . The third line follows from the fact that only the first ${ \bar { d } } _ { t }$ entries of $\mathbf { x } _ { t }$ are nonzero. The last inequality holds since $\psi ( x ) \leq \ln ( x )$ for all $x > 0$

Bounding term (b): As for term (b), we have

$$
\begin{array} { r l } { \mathrm { t e r m ~ ( b ) } = \displaystyle \sum _ { i = 2 } ^ { T } \left( \mathrm { K L } \left( Q _ { t } \mid P _ { t } \right) - \mathrm { K L } \left( Q _ { t - 1 } \mid P _ { t } \right) \right) } & { } \\ & { = \displaystyle \sum _ { t = 2 } ^ { T } \mathbb { E } _ { \mathrm { w } \sim Q _ { t } } \left[ \ln \frac { Q _ { t } ( \mathbf { w } ) } { P _ { t } ( \mathbf { w } ) } \right] - \mathbb { E } _ { \mathrm { w } \sim Q _ { t - 1 } } \left[ \ln \frac { Q _ { t - 1 } ( \mathbf { w } ) } { P _ { t } ( \mathbf { w } ) } \right] } \\ & { = \displaystyle \sum _ { t = 2 } ^ { T } \left( H ( Q _ { t - 1 } ) - H ( Q _ { t } ) \right) + \sum _ { t = 2 } ^ { T } \mathbb { E } _ { \mathrm { w } \sim Q _ { t - 1 } } \left[ \ln P _ { t } ( \mathbf { w } ) \right] - \mathbb { E } _ { \mathrm { w } \sim Q _ { t } } \left[ \ln P _ { t } ( \mathbf { w } ) \right] } \\ & { = \displaystyle \sum _ { t = 2 } ^ { T } \int _ { \mathrm { w } \in \Delta _ { d } } \left( Q _ { t - 1 } ( \mathbf { w } ) - Q _ { t } ( \mathbf { w } ) \right) \ln P _ { t } ( \mathbf { w } ) \mathrm { d } \mathbf { w } + \underbrace { H ( Q _ { 1 } ) - H ( Q _ { T } ) } _ { \mathrm { t e r m ~ ( b r o ) } } } \end{array}
$$

We then proceed to bound term (b-1) and term (b-2) separately. For notational simplicity, we denote by $C _ { T , d } = ( d - 1 ) \ln ( T + 1 ) + d ( \ln d + 1 )$ . We can upper bound term (b-1) by

$$
\begin{array} { r l } & { \mathrm { t e r m ~ ( b - 1 ) } \le \displaystyle \sum _ { t = 2 } ^ { T } \int _ { \mathbf { w } \in \Delta _ { d } } \vert Q _ { t - 1 } ( \mathbf { w } ) - Q _ { t } ( \mathbf { w } ) \vert \cdot \vert \ln P _ { t } ( \mathbf { w } ) \vert \mathrm { d } \mathbf { w } } \\ & { \quad \le C _ { T , d } \displaystyle \sum _ { t = 2 } ^ { T } \int _ { \Delta _ { d } } \vert Q _ { t - 1 } ( \mathbf { w } ) - Q _ { t } ( \mathbf { w } ) \vert \mathrm { d } \mathbf { w } , } \end{array}\tag{30}
$$

The last inequality holds because |ln $P _ { t } ( \mathbf { w } ) \vert \leq C _ { T , d }$ for all $t \in [ T ]$ and $\mathbf { w } \in \Delta _ { d }$ . Indeed, the upper bound ln $P _ { t } ( \mathbf { w } ) \leq C _ { T , d }$ follows from Lemma 16. For the lower bound, the fixed-share update (5) and $P _ { 1 } ( \mathbf { w } ) = \Gamma ( d )$ give $P _ { t } ( \mathbf { w } ) \geq \mu _ { t } P _ { 1 } ( \mathbf { w } ) = \Gamma ( d ) / t \geq 1 / T$ , so ln $P _ { t } ( \mathbf { w } ) \geq - \ln T \geq - C _ { T , d }$

Then, we can further bound the total variation between $Q _ { t }$ and $Q _ { t - 1 }$ by

$$
\sum _ { t = 2 } ^ { T } \int _ { \Delta _ { d } } | Q _ { t - 1 } ( \mathbf { w } ) - Q _ { t } ( \mathbf { w } ) | \mathrm { d } \mathbf { w } \leq 2 \sqrt { 2 \gamma } \sum _ { t = 2 } ^ { T } \mathrm { J S } ( \mathbf { u } _ { t } , \mathbf { u } _ { t - 1 } ) = 2 \sqrt { 2 \gamma } P _ { T } ^ { \mathrm { J S } } ,\tag{31}
$$

where the inequality holds by Lemma 17, and the equality follows from the definition of $P _ { T } ^ { \mathrm { J S } }$ . Combining (30) and (31), we arrive at

$$
\mathsf { t e r m } \left( \mathsf { b } - 1 \right) \le 2 \sqrt { 2 } C _ { T , d } \sqrt { \gamma } P _ { T } ^ { \mathrm { J S } } .
$$

As for term (b-2), we have

$$
\mathsf { t e r m ~ } \left( \mathsf { b } { - } 2 \right) = H ( Q _ { 1 } ) - H ( Q _ { T } ) = \mathrm { K L } \left( Q _ { T } \parallel P _ { 1 } \right) - \mathrm { K L } \left( Q _ { 1 } \parallel P _ { 1 } \right)
$$

since $P _ { 1 } = \operatorname { D i r } ( \mathbf { 1 } )$ is a uniform distribution over the simplex. Finally, we arrive at

$$
\begin{array} { r } { \mathbf { t e r m } ( \mathbf { b } ) \leq 2 \sqrt { 2 } C _ { T , d } \sqrt { \gamma } P _ { T } ^ { \mathrm { J S } } + \mathrm { K L } ( Q _ { T }  P _ { 1 } ) - \mathrm { K L } ( Q _ { 1 }  P _ { 1 } ) . } \end{array}
$$

Combining All. Combining the bounds on terms (a) and (b), we get

$$
\begin{array} { l } { \displaystyle \sum _ { t = 1 } ^ { T } \ell _ { t } ( \mathbf w _ { t } ) \leq \sum _ { t = 1 } ^ { T } \ell _ { t } ( \mathbf { u } _ { t } ) + T \ln \left( 1 + \frac { d } { \gamma } \right) + 2 \sqrt { 2 } C _ { T , d } \sqrt { \gamma } P _ { T } ^ { \mathrm { J S } } } \\ { \displaystyle \qquad + \mathrm { K L } \left( Q _ { T } \parallel P _ { 1 } \right) + ( 1 + \ln T ) . } \end{array}\tag{32}
$$

We can further upper bound the KL divergence term by

$$
\begin{array} { r l } & { \mathrm { K L } \left( Q _ { T } \left\| P _ { 1 } \right) \leq \mathrm { K L } \left( \mathrm { D i r } ( \mathbf 1 + \gamma \mathbf e _ { i } ) \right\| P _ { 1 } \right) } \\ & { \qquad = \ln \frac { \Gamma ( d + \gamma ) } { \Gamma ( d ) \Gamma ( 1 + \gamma ) } + \gamma ( \psi ( 1 + \gamma ) - \psi ( d + \gamma ) ) } \\ & { \qquad \leq ( d - 1 ) \ln ( 1 + \gamma ) . } \end{array}
$$

where $\mathbf { e } _ { i }$ is a one-hot vector with 1 at the i-th position and 0 elsewhere. The first inequality is due to Lemma 15. For the last inequality, we use the identities $\Gamma ( d + \gamma ) / \Gamma ( 1 + \gamma ) = \prod _ { i = 1 } ^ { d - 1 } ( \gamma + i )$ and $\begin{array} { r } { \Gamma ( d ) = \prod _ { i = 1 } ^ { d - 1 } i } \end{array}$ , together with $\psi ( 1 + \gamma ) - \psi ( d +$ $\gamma ) \leq 0$ . Then, the regret bound (32) becomes

$$
\begin{array} { l } { { \displaystyle \sum _ { t = 1 } ^ { T } \ell _ { t } ( { \mathbf w } _ { t } ) - \sum _ { t = 1 } ^ { T } \ell _ { t } ( { \mathbf u } _ { t } ) } } \\ { { \le T \ln \left( 1 + \frac { d } { \gamma } \right) + 2 \sqrt { 2 } C _ { T , d } \sqrt { \gamma } P _ { T } ^ { \mathrm { J S } } + ( d - 1 ) \ln ( 1 + \gamma ) + ( 1 + \ln T ) , } } \end{array}
$$

where $C _ { T , d } = ( d - 1 ) \ln ( T + 1 ) + d ( \ln d + 1 ) = \mathcal { O } \bigl ( d \ln ( T d ) \bigr )$ . Since $\gamma$ appears only in the analysis, we can choose it to optimize the regret bound. We consider the following two cases, depending on the value of $P _ { T } ^ { \mathrm { J S } }$

• Case 1 $\left( P _ { T } ^ { \mathrm { J S } } \leq 1 / T \ \right)$ . We set $\gamma = T$ , which yields an $\mathcal { O } ( d \ln ( d T ) )$ regret bound.

• Case 2 $\left( P _ { T } ^ { \mathrm { J S } } > 1 / T \right)$ . We choose $\gamma = T ^ { \frac 2 3 } \big ( P _ { T } ^ { \mathrm { J S } } \big ) ^ { - \frac 2 3 } ( \ln ( d T ) ) ^ { - \frac 2 3 }$ . Substituting this choice of $\gamma$ into the bound gives

$$
\sum _ { t = 1 } ^ { T } \ell _ { t } ( \mathbf { w } _ { t } ) - \sum _ { t = 1 } ^ { T } \ell _ { t } ( \mathbf { u } _ { t } ) = \mathcal { O } \left( d \Big ( T ^ { \frac { 1 } { 3 } } \big ( P _ { T } ^ { \mathrm { J S } } \big ) ^ { \frac { 2 } { 3 } } ( \ln ( d T ) ) ^ { \frac { 2 } { 3 } } + \ln ( d T ) \Big ) \right) .
$$

We have completed the proof by combining the two cases.

## C.3. Proof of Theorem 4

Proof of Theorem 4 The proof follows the same overall argument as in the proof of Theorem 1. The main diference lies in the hard example construction. The lower bound in Theorem 1 relies on a hard instance with comparator sequences near the boundary of the simplex. Here, we instead construct a hard instance showing that the $T ^ { 1 / 3 } ( P _ { T } ^ { \mathrm { J S } } ) ^ { 2 / 3 }$ dependence is optimal up to logarithmic factors even for uniformly interior comparator sequences.

Our goal remains to establish a lower bound on the minimax regret.

$$
\mathcal { W } _ { T } ( \mathcal { U } _ { C } ^ { \mathrm { J S } } ) = \operatorname* { i n f } _ { f _ { 1 : T } } \operatorname* { s u p } _ { \substack { \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { T } \in \mathbb { R } _ { + } ^ { d } \mathbf { u } _ { 1 : T } \in \mathcal { U } _ { C } ^ { \mathrm { J S } } } } \left( \sum _ { t = 1 } ^ { T } \ell _ { t } ( \mathbf { w } _ { t } ) - \sum _ { t = 1 } ^ { T } \ell _ { t } ( \mathbf { u } _ { t } ) \right) ,
$$

where $f _ { 1 : T }$ denotes the sequence of online prediction rules and $\mathbf { w } _ { t } = f _ { t } ( \mathbf { x } _ { 1 : t - 1 } )$ is the algorithm’s prediction based only on past observations. Here,

$$
\mathcal { U } _ { C } ^ { \mathrm { J S } } = \left\{ \mathbf { u } _ { 1 : T } \in \Delta _ { d } ^ { T } \left| \sum _ { t = 2 } ^ { T } \mathrm { J S } \left( \mathbf { u } _ { t } , \mathbf { u } _ { t - 1 } \right) \le C \right. \right\}
$$

is the set of all comparator sequences whose JS-path length is at most $C$ . The same reduction as in the proof of Theorem 1 gives

$$
\mathcal { W } _ { T } ( \mathcal { U } _ { C } ^ { \mathrm { J S } } ) \geq \mathcal { V } _ { T } ( \mathcal { U } _ { C } ^ { \mathrm { J S } } ) : = \operatorname* { i n f } _ { f _ { 1 : T } } \operatorname* { s u p } _ { y _ { 1 } , \ldots , y _ { T } \in \mathcal { V } } \operatorname* { s u p } _ { \mathbf { u } _ { 1 : T } \in \mathcal { U } _ { C } ^ { \mathrm { J S } } } \left( \sum _ { t = 1 } ^ { T } \ell _ { \log } ( \mathbf { w } _ { t } , y _ { t } ) - \sum _ { t = 1 } ^ { T } \ell _ { \log } ( \mathbf { u } _ { t } , y _ { t } ) \right) ,
$$

where $\mathcal { V } = [ d ]$ is an alphabet of size d and $\ell _ { \mathrm { l o g } } ( \mathbf { w } , y ) = - \log [ \mathbf { w } ] _ { y }$ for any $\mathbf { w } \in \Delta _ { d }$

Hard Example Construction. We first restrict attention to the main regime $\begin{array} { r } { C \in \left\lceil \sqrt { \frac { 2 ( d - 1 ) } { T } } , \frac { T } { 4 \sqrt { 2 } ( d - 1 ) } \right\rceil } \end{array}$ , which is non-empty by the assumption $T > 4 ( d - 1 )$ . As in the proof of Theorem 1, we partition the horizon into $K = \lfloor T / L \rfloor$ blocks, where the first $K - 1$ blocks have length

$$
L = \left\lceil { \frac { d - 1 } { \epsilon ^ { 2 } } } \right\rceil { \mathrm { ~ w i t h ~ } } \epsilon = \left( { \frac { ( d - 1 ) C } { \sqrt { 2 } T } } \right) ^ { \frac { 1 } { 3 } } .
$$

Under the main regime, $\sqrt { ( d - 1 ) / T } \leq \epsilon \leq 1 / 2 ,$ so $L \leq T$ . The final block has length $T - ( K - 1 ) L \in [ L , 2 L - 1 ]$ . We denote the k-th block by $\mathcal { T } _ { k } = \left\lceil s _ { k } , e _ { k } \right\rceil$

For each block $k \in [ K ]$ , let $\mathbf { I } _ { k } = [ I _ { k , 1 } , \ldots , I _ { k , d - 1 } ] \in \{ 0 , 1 \} ^ { d - 1 }$ , where $I _ { k , j } \sim \mathrm { B e r n } ( 1 / 2 )$ independently for every $j \in [ d - 1 ]$ and $k \in [ K ]$ ]. For any $t \in \mathcal { Z } _ { k }$ , we define a probability vector $\widetilde { \mathbf { u } } _ { t }$ in the interior of $\Delta _ { d }$ by

$$
\begin{array} { r l } & { [ \widetilde { \mathbf { u } } _ { t } ] _ { j } = \displaystyle \frac { 1 } { 2 ( d - 1 ) } + \frac { \epsilon } { d - 1 } \left( I _ { k , j } - \frac { 1 } { 2 } \right) \quad \mathrm { f o r ~ a l l ~ } j \in [ d - 1 ] , \ t \in \mathbb { Z } _ { k } , } \\ & { [ \widetilde { \mathbf { u } } _ { t } ] _ { d } = \displaystyle \frac { 1 } { 2 } - \frac { \epsilon } { d - 1 } \displaystyle \sum _ { j = 1 } ^ { d - 1 } \left( I _ { k , j } - \frac { 1 } { 2 } \right) \quad \mathrm { f o r ~ a l l ~ } t \in \mathbb { Z } _ { k } . } \end{array}
$$

Indeed, since $\epsilon \leq 1 / 2$ , we have $\begin{array} { r } { [ \widetilde { \mathbf { u } } _ { t } ] _ { j } \in \left[ \frac { 1 } { 4 ( d - 1 ) } , \frac { 3 } { 4 ( d - 1 ) } \right] } \end{array}$ for all $j \in [ d - 1 ]$ and $[ \widetilde { \mathbf { u } } _ { t } ] _ { d } \in$ $[ 1 / 4 , 3 / 4 ]$ . We then generate $Y _ { t } \sim \mathrm { C a t } ( \widetilde { \mathbf { u } } _ { t } )$ independently conditional on $\mathbf { I } _ { 1 : K }$

We next show that the comparator sequence $\widetilde { \mathbf { u } } _ { 1 : T }$ has JS-path length at most $C .$ Since $\widetilde { \mathbf { u } } _ { t }$ is constant within each block, it can change only at the $K - 1$ block boundaries. It therefore sufices to bound the JS distance between $\widetilde { \mathbf { u } } _ { s _ { k } - 1 }$ and $\widetilde { \mathbf { u } } _ { s _ { k } }$ , the comparator vectors on blocks $k - 1$ and $k .$ , respectively. For any $k \in \{ 2 , \ldots , K \}$ , we have

$$
\begin{array} { l } { \displaystyle \mathrm { J S } ( \widetilde { \mathbf { u } } _ { s _ { k } } , \widetilde { \mathbf { u } } _ { s _ { k - 1 } } ) ^ { 2 } = \frac { 1 } { 2 } \sum _ { i = 1 } ^ { d } \left[ [ \widetilde { \mathbf { u } } _ { s _ { k } } ] _ { i } \ln \frac { 2 [ \widetilde { \mathbf { u } } _ { s _ { k } } ] _ { i } } { [ \widetilde { \mathbf { u } } _ { s _ { k } } ] _ { i } + [ \widetilde { \mathbf { u } } _ { s _ { k } - 1 } ] _ { i } } + [ \widetilde { \mathbf { u } } _ { s _ { k } - 1 } ] _ { i } \ln \frac { 2 [ \widetilde { \mathbf { u } } _ { s _ { k } - 1 } ] _ { i } } { [ \widetilde { \mathbf { u } } _ { s _ { k } } ] _ { i } + [ \widetilde { \mathbf { u } } _ { s _ { k } - 1 } ] _ { i } } \right] } \\ { \displaystyle \leq \frac { 1 } { 2 } \sum _ { i = 1 } ^ { d } \frac { \left( [ \widetilde { \mathbf { u } } _ { s _ { k } } ] _ { i } - [ \widetilde { \mathbf { u } } _ { s _ { k } - 1 } ] _ { i } \right) ^ { 2 } } { [ \widetilde { \mathbf { u } } _ { s _ { k } } ] _ { i } + [ \widetilde { \mathbf { u } } _ { s _ { k } - 1 } ] _ { i } } , } \end{array}\tag{33}
$$

where the inequality uses ln $x \leq x - 1$ . We bound the contributions of the first $d - 1$ coordinates and the last coordinate separately. For each $j \in [ d - 1 ]$ , the coordinate $[ \widetilde { \mathbf { u } } _ { t } ] _ { j }$ takes one of the two values $( 1 - \epsilon ) / ( 2 ( d - 1 ) )$ and $( 1 + \epsilon ) / ( 2 ( d - 1 ) )$ . Hence,

$$
\sum _ { j = 1 } ^ { d - 1 } \frac { \left( \left[ \widetilde { \mathbf { u } } _ { s _ { k } } \right] _ { j } - \left[ \widetilde { \mathbf { u } } _ { s _ { k } - 1 } \right] _ { j } \right) ^ { 2 } } { \left[ \widetilde { \mathbf { u } } _ { s _ { k } } \right] _ { j } + \left[ \widetilde { \mathbf { u } } _ { s _ { k } - 1 } \right] _ { j } } \leq \frac { 2 \epsilon ^ { 2 } } { d - 1 } \sum _ { j = 1 } ^ { d - 1 } | I _ { k , j } - I _ { k - 1 , j } | \leq 2 \epsilon ^ { 2 } ,\tag{34}
$$

where the first inequality uses $[ \widetilde { \mathbf { u } } _ { s _ { k } } ] _ { j } + [ \widetilde { \mathbf { u } } _ { s _ { k } - 1 } ] _ { j } \geq 1 / ( 2 ( d - 1 ) )$ and $[ \widetilde { \mathbf { u } } _ { s _ { k } } ] _ { j } - [ \widetilde { \mathbf { u } } _ { s _ { k } - 1 } ] _ { j } =$ $\epsilon ( I _ { k , j } - I _ { k - 1 , j } ) / ( d - 1 )$ .

For the last coordinate, we have

$$
\frac { \left( [ \widetilde { \mathbf { u } } _ { s _ { k } } ] _ { d } - [ \widetilde { \mathbf { u } } _ { s _ { k } - 1 } ] _ { d } \right) ^ { 2 } } { [ \widetilde { \mathbf { u } } _ { s _ { k } } ] _ { d } + [ \widetilde { \mathbf { u } } _ { s _ { k } - 1 } ] _ { d } } \leq 2 \left( [ \widetilde { \mathbf { u } } _ { s _ { k } } ] _ { d } - [ \widetilde { \mathbf { u } } _ { s _ { k } - 1 } ] _ { d } \right) ^ { 2 } \leq 2 \epsilon ^ { 2 } .\tag{35}
$$

The first inequality uses $[ \widetilde { \mathbf { u } } _ { s _ { k } } ] _ { d } , [ \widetilde { \mathbf { u } } _ { s _ { k } - 1 } ] _ { d } \geq 1 / 4$ . The last inequality follows from the definition of $\widetilde { \mathbf { u } } _ { t }$ , which gives

$$
\lvert [ \widetilde { \mathbf { u } } _ { s _ { k } } ] _ { d } - [ \widetilde { \mathbf { u } } _ { s _ { k } - 1 } ] _ { d } \rvert = \frac { \epsilon } { d - 1 } \left. \sum _ { j = 1 } ^ { d - 1 } \left( I _ { k , j } - I _ { k - 1 , j } \right) \right. \leq \frac { \epsilon } { d - 1 } \sum _ { j = 1 } ^ { d - 1 } \lvert I _ { k , j } - I _ { k - 1 , j } \rvert \leq \epsilon .
$$

Combining (34) and (35) with (33), we obtain

$$
P _ { T } ^ { \mathrm { J S } } = \sum _ { k = 2 } ^ { K } \mathrm { J S } \left( \widetilde { \mathbf { u } } _ { s _ { k } } , \widetilde { \mathbf { u } } _ { s _ { k } - 1 } \right) \leq ( K - 1 ) \sqrt { 2 } \epsilon \leq \frac { \sqrt { 2 } T \epsilon ^ { 3 } } { d - 1 } = C .\tag{36}
$$

Here, the second inequality uses $K - 1 \le T / L$ and $L \ge ( d - 1 ) / \epsilon ^ { 2 }$ , and the last equality follows from the definition of ϵ. Thus, every realization of the comparator sequence $\widetilde { \mathbf { u } } _ { 1 : T }$ belongs to $\mathcal { U } _ { C } ^ { \mathrm { J S } }$

Lower bounding the minimax regret. For each block $k \in [ K ]$ , let $Y _ { \mathcal { T } _ { k } } : = ( Y _ { t } ) _ { t \in \mathcal { T } _ { k } }$ denote the random label sequence on block k, and let $y _ { \mathcal { T } _ { k } } : = ( y _ { t } ) _ { t \in \mathcal { T } _ { k } } \in \mathcal { V } ^ { | \mathcal { T } _ { k } | }$ denote one of its realizations. Conditional on the environment index $\mathbf { I } _ { k } .$ the probability mass function of $Y _ { \mathcal { T } _ { k } }$ is $\begin{array} { r } { \widetilde { \mathbf q } _ { k } ( y _ { \mathbb { T } _ { k } } \mid \mathbf I _ { k } ) : = \prod _ { t \in \mathcal { T } _ { k } } [ \widetilde { \mathbf { u } } _ { t } ] _ { y _ { t } } } \end{array}$ , and we use $\widetilde { \mathbf { q } } _ { Y _ { \mathcal { T } _ { k } } | \mathbf { I } _ { k } }$ to denote the corresponding conditional distribution. We further define its marginal probability mass function by $\bar { \mathbf q } _ { k } ( y _ { \mathcal { T } _ { k } } ) : = \mathbb { E } _ { \mathbf { I } _ { k } } [ \widetilde { \mathbf q } _ { k } ( y _ { \mathcal { T } _ { k } } \ | \ \mathbf I _ { k } ) ]$ and use $\bar { \mathbf q } _ { Y _ { \boldsymbol L _ { k } } }$ to denote the corresponding marginal distribution. The same argument used to derive (15) in the proof of Theorem 1 then shows that

$$
\mathcal { V } _ { T } ( \mathcal { U } _ { C } ^ { \mathrm { J S } } ) \geq \sum _ { k = 1 } ^ { K } \mathbb { E } _ { { \mathbf { I } } _ { k } } \left[ { \mathrm { K L } } \left( \widetilde { \mathbf { q } } _ { Y _ { \mathcal { T } _ { k } } | { \mathbf { I } } _ { k } } \Vert \bar { \mathbf { q } } _ { Y _ { \mathcal { T } _ { k } } } \right) \right] .\tag{37}
$$

We next lower bound the information contributed by each block. Recall that $\mathbf { I } _ { k } = \left( I _ { k , 1 } , \ldots , I _ { k , d - 1 } \right)$ is the binary environment index of block $k _ { : }$ , where $I _ { k , j } \in \{ 0 , 1 \}$ determines whether the $j { \mathrm { - t h } }$ coordinate of $\widetilde { \mathbf { u } } _ { s _ { k } }$ is $( 1 - \epsilon ) / ( 2 ( d - 1 ) ) \mathrm { o r } ( 1 + \epsilon ) / ( 2 ( d - 1 ) )$ . For every $j \in [ d - 1 ]$ , we define

$$
Z _ { k , j } = \sum _ { t = s _ { k } } ^ { s _ { k } + L - 1 } \mathbb { 1 } \{ Y _ { t } = j \} ,
$$

which counts the number of occurrences of label $j$ in the first L rounds of block k. We further define $\mathbf { Z } _ { k } = \left( Z _ { k , 1 } , \ldots , Z _ { k , d - 1 } \right)$ . Given a realization $I _ { k , j } = b \in \{ 0 , 1 \}$ , label $j$ is observed independently at each round with probability $\begin{array} { r } { \pi _ { b } = \frac { 1 } { 2 ( d - 1 ) } + \frac { \epsilon } { d - 1 } ( b - \frac { 1 } { 2 } ) } \end{array}$ . Hence,

conditional on $I _ { k , j } = b$ , the random variable $Z _ { k , j }$ follows the binomial distribution $P _ { b } = \mathrm { B i n } ( L , \pi _ { b } )$ , whose probability mass function is

$$
P _ { b } ( z ) = \operatorname* { P r } ( Z _ { k , j } = z \mid I _ { k , j } = b ) = { \binom { L } { z } } \pi _ { b } ^ { z } ( 1 - \pi _ { b } ) ^ { L - z } , \qquad z \in \{ 0 , \ldots , L \} .
$$

Since $\mathbf { Z } _ { k }$ is a deterministic function of $Y _ { \mathcal { T } _ { k } }$ , the same argument used to derive (16) in the proof of Theorem 1 gives

$$
\mathbb { E } _ { { \mathbf { I } } _ { k } } \left[ \mathrm { K L } \left( \widetilde { \mathbf { q } } _ { Y _ { \mathcal { I } _ { k } } | { \mathbf { I } } _ { k } } \Vert \bar { \mathbf { q } } _ { Y _ { \mathcal { I } _ { k } } } \right) \right] \geq \sum _ { j = 1 } ^ { d - 1 } { \mathrm { I } } \left( I _ { k , j } ; Z _ { k , j } \right) \geq \frac { d - 1 } { 2 } \mathrm { T V } ( P _ { 0 } , P _ { 1 } ) ^ { 2 } .\tag{38}
$$

Here, I(X; Y) denotes the mutual information between the random variables X and $Y$ , and $\begin{array} { r } { \mathrm { T V } ( P , Q ) = \frac { 1 } { 2 } \sum _ { z } | P ( z ) - Q ( z ) | } \end{array}$ denotes the total variation distance. The first inequality follows from the independence of the environment indices and the data-processing inequality, as in (16). The last inequality follows from Pinsker’s inequality since $I _ { k , j } \sim \mathrm { B e r n } ( 1 / 2 )$ and the marginal distribution of $Z _ { k , j }$ is the equally weighted mixture $M = ( P _ { 0 } + P _ { 1 } ) / 2$ of its two conditional distributions. In particular,

$$
\mathrm { I } ( I _ { k , j } ; Z _ { k , j } ) = \frac 1 2 \mathrm { K L } ( P _ { 0 } \| M ) + \frac 1 2 \mathrm { K L } ( P _ { 1 } \| M ) \ge \frac 1 2 \mathrm { T V } ( P _ { 0 } , P _ { 1 } ) ^ { 2 } .
$$

We next lower bound the total variation distance between the two conditional distributions. Let $\rho = \sqrt { \pi _ { 0 } \pi _ { 1 } } + \sqrt { ( 1 - \pi _ { 0 } ) ( 1 - \pi _ { 1 } ) }$ . We have

$$
\mathrm { T V } ( P _ { 0 } , P _ { 1 } ) = 1 - \sum _ { z = 0 } ^ { L } \operatorname* { m i n } \{ P _ { 0 } ( z ) , P _ { 1 } ( z ) \} \geq 1 - \sum _ { z = 0 } ^ { L } \sqrt { P _ { 0 } ( z ) P _ { 1 } ( z ) } .
$$

The equality follows from $| a - b | = a + b - 2 \operatorname* { m i n } \{ a , b \}$ and the fact that each probability mass function sums to one. The inequality uses min $\{ a , b \} \leq { \sqrt { a b } }$ for $a , b \geq 0$ . Using the probability mass functions of the two binomial distributions, we obtain

$$
\begin{array} { r l r } {  { \sum _ { z = 0 } ^ { L } \sqrt { P _ { 0 } ( z ) P _ { 1 } ( z ) } = \sum _ { z = 0 } ^ { L } { \binom { L } { z } } ( \sqrt { \pi _ { 0 } \pi _ { 1 } } ) ^ { z } ( \sqrt { ( 1 - \pi _ { 0 } ) ( 1 - \pi _ { 1 } ) } ) ^ { L - z } } } \\ & { } & { = ( \sqrt { \pi _ { 0 } \pi _ { 1 } } + \sqrt { ( 1 - \pi _ { 0 } ) ( 1 - \pi _ { 1 } ) } ) ^ { L } = \rho ^ { L } . } \end{array}
$$

The second equality follows by expanding the L-th power of the sum. To bound $\rho ,$ we note that

$$
\begin{array} { l } { \displaystyle 1 - \rho = \frac 1 2 \left[ ( \sqrt { \pi _ { 1 } } - \sqrt { \pi _ { 0 } } ) ^ { 2 } + ( \sqrt { 1 - \pi _ { 1 } } - \sqrt { 1 - \pi _ { 0 } } ) ^ { 2 } \right] } \\ { \displaystyle \geq \frac 1 2 \frac { ( \pi _ { 1 } - \pi _ { 0 } ) ^ { 2 } } { ( \sqrt { \pi _ { 1 } } + \sqrt { \pi _ { 0 } } ) ^ { 2 } } \geq \frac { \epsilon ^ { 2 } } { 4 ( d - 1 ) } , } \end{array}\tag{39}
$$

where the last inequality uses $\pi _ { 1 } - \pi _ { 0 } = \epsilon / ( d - 1 )$ and $( \sqrt { \pi _ { 1 } } + \sqrt { \pi _ { 0 } } ) ^ { 2 } \leq 2 ( \pi _ { 0 } + \pi _ { 1 } ) =$ $2 / ( d - 1 )$ . Substituting these bounds into the total variation inequality above gives

$$
\mathrm { T V } ( P _ { 0 } , P _ { 1 } ) \geq 1 - \rho ^ { L } \geq 1 - \exp \left( - \frac { L \epsilon ^ { 2 } } { 4 ( d - 1 ) } \right) \geq 1 - e ^ { - 1 / 4 } ,\tag{40}
$$

where the second inequality uses $1 - x \leq e ^ { - x }$ with $x = 1 - \rho ,$ and the last inequality follows from $L \epsilon ^ { 2 } / ( d - 1 ) \geq 1$ . Substituting (40) into (38), we obtain

$$
\mathbb { E } _ { { \mathbf { I } } _ { k } } \left[ \mathrm { K L } \left( \widetilde { \mathbf { q } } _ { Y _ { \mathcal { T } _ { k } } | { \mathbf { I } } _ { k } } \Vert \bar { \mathbf { q } } _ { Y _ { \mathcal { T } _ { k } } } \right) \right] \geq \frac { d - 1 } { 2 } \left( 1 - e ^ { - 1 / 4 } \right) ^ { 2 } .\tag{41}
$$

Combining (37) and (41) yields

$$
\mathcal { V } _ { T } ( \mathcal { U } _ { C } ^ { \mathrm { J S } } ) \geq \frac { d - 1 } { 2 } ( 1 - e ^ { - 1 / 4 } ) ^ { 2 } K \geq \frac { ( 1 - e ^ { - 1 / 4 } ) ^ { 2 } } { 8 } T \epsilon ^ { 2 } = \Omega \left( ( d - 1 ) ^ { \frac { 2 } { 3 } } T ^ { \frac { 1 } { 3 } } C ^ { \frac { 2 } { 3 } } \right) .\tag{42}
$$

Here, the second inequality uses $K = \lfloor T / L \rfloor \ge T / ( 2 L ) \ge T \epsilon ^ { 2 } / ( 4 ( d - 1 ) )$ , since $T / L \geq 1$ and $L = \lceil ( d - 1 ) / \epsilon ^ { 2 } \rceil \leq 2 ( d - 1 \bar { ) } / \epsilon ^ { 2 }$ in the main regime. The last equality follows from the definition $\epsilon = ( ( d - 1 ) C / ( \sqrt { 2 } T ) ) ^ { 1 / 3 }$

Handling Corner Cases. We next consider the two regimes outside the main regime. First, suppose that $C < \sqrt { 2 ( d - 1 ) / T }$ . We use the same construction with $\epsilon = \epsilon _ { 0 } : = \sqrt { ( d - 1 ) / T }$ . In this case, $L = T$ and $K = 1$ , so the comparator sequence is constant and has zero JS-path length. Applying the one-block estimate in (41) gives

$$
\mathcal { V } _ { T } ( \mathcal { U } _ { C } ^ { \mathrm { J S } } ) \geq \frac { d - 1 } { 2 } \left( 1 - e ^ { - 1 / 4 } \right) ^ { 2 } = \Omega ( d ) .
$$

Moreover, the condition on C implies $( d - 1 ) ^ { \frac { 2 } { 3 } } T ^ { \frac { 1 } { 3 } } C ^ { \frac { 2 } { 3 } } < 2 ^ { \frac { 1 } { 3 } } ( d - 1 ) = \mathcal { O } ( d )$ . Thus, the one-block lower bound already dominates the desired dynamic term in this regime.

Next, suppose that $C > T / ( 4 \sqrt { 2 } ( d - 1 ) )$ . We apply the construction from the main regime with the smaller path length budget $C _ { 0 } = T / ( 4 \sqrt { 2 } ( d - 1 ) )$ . Since $\mathcal { U } _ { C _ { 0 } } ^ { \mathrm { J S } } \subseteq \mathcal { U } _ { C } ^ { \mathrm { J S } }$ the monotonicity of the comparator classes and (42) give

$$
\mathcal { V } _ { T } ( \mathcal { U } _ { C } ^ { \mathrm { J S } } ) \geq \mathcal { V } _ { T } ( \mathcal { U } _ { C _ { 0 } } ^ { \mathrm { J S } } ) = \Omega \left( ( d - 1 ) ^ { \frac { 2 } { 3 } } T ^ { \frac { 1 } { 3 } } C _ { 0 } ^ { \frac { 2 } { 3 } } \right) = \Omega ( T ) .
$$

Combining the main regime with the two boundary regimes, we conclude that

$$
\mathcal { V } _ { T } ( \mathcal { U } _ { C } ^ { \mathrm { J S } } ) \geq \Omega \left( \operatorname* { m i n } \left\{ T , d ^ { \frac { 2 } { 3 } } T ^ { \frac { 1 } { 3 } } C ^ { \frac { 2 } { 3 } } \right\} \right) .\tag{43}
$$

Combining this bound with the classical static lower bound $\Omega ( d \log ( 1 + T / d ) )$ for $C = 0$ completes the proof. ■

## C.4. Proof of Corollary 5

Proof of Corollary 5 Fix any comparator sequence $\mathbf { u } _ { 1 } , \ldots , \mathbf { u } _ { T } \in \Delta _ { d }$ and let $\beta \in ( 0 , 1 )$ be a certain parameter for mixing the comparator. We define the interior counterpart of $\mathbf { u } _ { t }$ by $\begin{array} { r } { \tilde { \mathbf { u } } _ { t } = ( 1 - \beta ) \mathbf { u } _ { t } + \frac { \beta } { d } \mathbf { 1 } } \end{array}$ for all $t \in [ T ]$ . Clearly, we have mi $\boldsymbol { 1 } _ { i \in [ d ] } \tilde { \boldsymbol { u } } _ { t , i } \ge \beta / d$ Besides, the gap between $\tilde { \mathbf { u } } _ { t }$ and $\mathbf { u } _ { t }$ can be bounded by

$$
\sum _ { t = 1 } ^ { T } \ell _ { t } ( \tilde { \mathbf { u } } _ { t } ) - \sum _ { t = 1 } ^ { T } \ell _ { t } ( \mathbf { u } _ { t } ) = \sum _ { t = 1 } ^ { T } \ln \left( \frac { \mathbf { u } _ { t } ^ { \top } \mathbf { x } _ { t } } { ( 1 - \beta ) \mathbf { u } _ { t } ^ { \top } \mathbf { x } _ { t } + \frac { \beta } { d } \mathbf { 1 } ^ { \top } \mathbf { x } _ { t } } \right) \leq T \ln \left( \frac { 1 } { 1 - \beta } \right) .\tag{44}
$$

We first relate the JS-path length of the smoothed sequence directly to the $L _ { 1 } .$ -path length of the original sequence. For any $\mathbf { p } , \mathbf { q } \in \Delta _ { d }$ with $p _ { i } , q _ { i } \ge \beta / d$ , let $\mathbf { m } = ( \mathbf { p } + \mathbf { q } ) / 2$ Applying ln $x \leq x - 1$ coordinate-wise gives

$$
\begin{array} { l } { \displaystyle \mathrm { J S } ( \mathbf { p } , \mathbf { q } ) ^ { 2 } = \frac { 1 } { 2 } \sum _ { i = 1 } ^ { d } \left( p _ { i } \ln \frac { p _ { i } } { m _ { i } } + q _ { i } \ln \frac { q _ { i } } { m _ { i } } \right) } \\ { \displaystyle \quad \leq \frac { 1 } { 2 } \sum _ { i = 1 } ^ { d } \left[ p _ { i } \left( \frac { p _ { i } } { m _ { i } } - 1 \right) + q _ { i } \left( \frac { q _ { i } } { m _ { i } } - 1 \right) \right] } \\ { \displaystyle \quad = \frac { 1 } { 2 } \sum _ { i = 1 } ^ { d } \frac { ( p _ { i } - q _ { i } ) ^ { 2 } } { p _ { i } + q _ { i } } \leq \frac { d } { 4 \beta } \| \mathbf { p } - \mathbf { q } \| _ { 2 } ^ { 2 } . } \end{array}
$$

Since $\tilde { \mathbf { u } } _ { t } - \tilde { \mathbf { u } } _ { t - 1 } = ( 1 - \beta ) ( \mathbf { u } _ { t } - \mathbf { u } _ { t - 1 } )$ , it follows that

$$
P _ { T } ^ { \mathrm { J S } } ( \tilde { \mathbf { u } } _ { 1 : T } ) \leq \frac { 1 - \beta } { 2 } \sqrt { \frac { d } { \beta } } \sum _ { t = 2 } ^ { T } \lVert \mathbf { u } _ { t } - \mathbf { u } _ { t - 1 } \rVert _ { 2 } \leq \frac { 1 - \beta } { 2 } \sqrt { \frac { d } { \beta } } P _ { T } .\tag{45}
$$

Then, we can upper bound the dynamic regret with respect to any comparator ${ \mathbf { u } } _ { t } \in \Delta _ { d }$ by

$$
\begin{array} { r l } & { \mathrm { D } \mathrm { - } \mathrm { R e } _ { T } \big ( \{ \mathbf { u } _ { t } \} _ { t = 1 } ^ { T } \big ) = \displaystyle \sum _ { t = 1 } ^ { T } \ell _ { t } ( \mathbf { w } _ { t } ) - \displaystyle \sum _ { t = 1 } ^ { T } \ell _ { t } ( \tilde { \mathbf { u } } _ { t } ) + \displaystyle \sum _ { t = 1 } ^ { T } \ell _ { t } ( \tilde { \mathbf { u } } _ { t } ) - \displaystyle \sum _ { t = 1 } ^ { T } \ell _ { t } ( \mathbf { u } _ { t } ) } \\ & { \qquad \leq \displaystyle \sum _ { t = 1 } ^ { T } \ell _ { t } ( \mathbf { w } _ { t } ) - \displaystyle \sum _ { t = 1 } ^ { T } \ell _ { t } ( \tilde { \mathbf { u } } _ { t } ) \ + \ T \ln \left( \displaystyle \frac { 1 } { 1 - \beta } \right) } \\ & { \qquad \leq \mathcal { O } \left( d ^ { \frac 4 3 } B \beta ^ { - \frac 1 3 } ( 1 - \beta ) ^ { \frac 2 3 } T ^ { \frac 1 3 } P _ { T } ^ { \frac 2 3 } + d \ln ( d T ) + T \ln \left( \displaystyle \frac { 1 } { 1 - \beta } \right) \right) , } \end{array}
$$

where the first inequality follows from (44), and the last inequality follows from (45) and the regret guarantee assumption in Corollary 5. We now choose $\beta$ to make the bound tight:

• Case 1 $( P _ { T } = 0 )$ . We choose $\beta = 1 / T$ , which yields

$$
\mathrm { D - R e g } _ { T } ( \{ \mathbf { u } _ { t } \} _ { t = 1 } ^ { T } ) \leq \mathcal { O } \left( d \ln ( d T ) \right) .
$$

• Case $2 \ ( P _ { T } > 0 )$ . To balance the dynamic regret of $\tilde { \mathbf { u } } _ { 1 : T }$ and smoothing terms, write $a = \beta / ( 1 - \beta ) > 0$ . Using $( 1 + a ) ^ { - 1 / 3 } \leq 1$ and ln $\left( 1 + a \right) \leq a$ , we obtain

$$
\begin{array} { r } { \mathrm { D - R e g } _ { T } ( \{ \mathbf { u } _ { t } \} _ { t = 1 } ^ { T } ) \leq \mathcal { O } \left( d ^ { \frac { 4 } { 3 } } B T ^ { \frac { 1 } { 3 } } P _ { T } ^ { \frac { 2 } { 3 } } a ^ { - \frac { 1 } { 3 } } + T a + d \ln ( d T ) \right) . } \end{array}
$$

Balancing the first two terms gives $a = d B ^ { 3 / 4 } \sqrt { P _ { T } / T }$ . With this choice, both terms equal $d B ^ { 3 / 4 } \sqrt { T P _ { T } }$ , yielding

$$
\mathrm { D - R e g } _ { T } ( \{ \mathbf { u } _ { t } \} _ { t = 1 } ^ { T } ) \leq \mathcal { O } \left( d B ^ { \frac { 3 } { 4 } } \sqrt { T P _ { T } } + d \ln ( d T ) \right) .
$$

Combining the two cases proves the claim.

## C.5. Proof of Proposition 6

Proof of Proposition 6 Fix an interval $\mathcal { T } = [ r , s ]$ and a comparator $\mathbf { u } \in \Delta _ { d }$ . If $\ell _ { t } ( { \mathbf { u } } ) = + \infty$ for some $t \in \mathcal { Z }$ , the interval-regret claim is immediate, so suppose that its loss is finite throughout I. Following the same steps as in the proof of Theorem $3 ,$ for any distribution $Q _ { t }$ over $\Delta _ { d }$ , we have

$$
\ell _ { t } ( \mathbf { w } _ { t } ) \leq \mathbb { E } _ { \mathbf { w } \sim Q _ { t } } [ \ell _ { t } ( \mathbf { w } ) ] + \mathrm { K L } ( Q _ { t } \| P _ { t } ) - \mathrm { K L } ( Q _ { t } \| P _ { t + 1 } ) + \ln \frac { 1 } { 1 - \mu _ { t + 1 } } .
$$

For each round $t \in \mathcal { T } = [ r , s ]$ , we take the fixed distribution $Q _ { t } = Q _ { \mathbb { Z } } = \operatorname { D i r } ( { \bf 1 } + \gamma { \bf u } )$ Summing the above inequality over $\mathcal { T }$ with $\mu _ { t } = 1 / t$ yields

$$
\sum _ { t \in \mathbb { Z } } \ell _ { t } ( { \mathbf w } _ { t } ) \le \underbrace { \sum _ { t \in \mathbb { Z } } \mathbb { E } _ { { \mathbf w } \sim Q _ { \mathbb { Z } } } [ \ell _ { t } ( { \mathbf w } ) ] } _ { \mathrm { t e r m ~ ( a ) } } + \underbrace { \mathrm { K L } ( Q _ { \mathbb { Z } } \| P _ { r } ) } _ { \mathrm { t e r m ~ ( b ) } } - \mathrm { K L } ( Q _ { \mathbb { Z } } \| P _ { s + 1 } ) + \underbrace { \sum _ { t \in \mathbb { Z } } \ln \left( 1 + { \frac { 1 } { t } } \right) } _ { t \in \mathbb { Z } } .
$$

For term (a), the same argument as in the proof of Theorem 3 gives

$$
\mathrm { t e r m ~ \mathsf { \Omega } ( a ) } \le \sum _ { t \in \mathbb { Z } } \ell _ { t } ( { \mathbf { u } } ) + | \mathbb { Z } | \ln \left( 1 + { \frac { d } { \gamma } } \right) .
$$

For term (b), the fixed-share update gives $P _ { r } ( \mathbf { w } ) \geq \mu _ { r } P _ { 1 } ( \mathbf { w } )$ , and hence

$$
\begin{array} { l } { \displaystyle \mathrm { t e r m ~ } \left( \mathbf { b } \right) = \mathrm { K L } ( Q _ { \mathcal { I } } | | P _ { 1 } ) + \mathbb { E } _ { \mathbf { w } \sim Q _ { \mathcal { I } } } \left[ \ln \frac { P _ { 1 } \left( \mathbf { w } \right) } { P _ { r } \left( \mathbf { w } \right) } \right] } \\ { \displaystyle \quad \leq \mathrm { K L } ( Q _ { \mathcal { I } } | | P _ { 1 } ) + \ln \frac { 1 } { \mu _ { r } } } \\ { \displaystyle \leq \left( d - 1 \right) \ln ( 1 + \gamma ) + \ln T , } \end{array}
$$

where the last inequality follows from Lemma 15 and the calculation in the proof of Theorem 3. Taking $\gamma = T$ , dropping the nonpositive KL term, and using

$$
| \mathcal { Z } | \ln \left( 1 + \frac { d } { T } \right) \leq d \quad \mathrm { a n d } \quad \sum _ { t = r } ^ { s } \ln \left( 1 + \frac { 1 } { t } \right) = \ln \frac { s + 1 } { r } \leq \ln ( T + 1 )
$$

complete the proof for the interval regret.

For the switching guarantee, partition [T] into the $\mathsf { S } _ { T } + 1$ maximal intervals on which the comparator is constant. Applying the interval-regret bound on each interval and summing gives

$$
\begin{array} { r } { \mathrm { D - R e g } _ { T } ( \{ \mathbf { u } _ { t } \} _ { t = 1 } ^ { T } ) \leq \mathcal { O } \big ( d ( \mathsf { S } _ { T } + 1 ) \ln ( d T ) \big ) , } \end{array}
$$

which completes the proof.

## Appendix D. Omitted Proofs for Section 3.3

## D.1. Proof of Theorem 7

Proof of Theorem 7 Fix any $q \in [ 0 , 1 ]$ and any comparator sequence $\mathbf { u } _ { 1 } , \ldots , \mathbf { u } _ { T } \in \Delta _ { d }$ We use the same mixability-based decomposition as in the proof of Theorem 3, with the virtual comparator distribution $Q _ { t } = \operatorname * { D i r } ( { \bf 1 } + \gamma { \bf u } _ { t } )$ . The bounds on the expected-loss, endpoint, and fixed-share terms remain unchanged. The only modification concerns the total variation between $Q _ { t }$ and $Q _ { t - 1 }$ in term (b-1). For every $t \geq 2$ , Lemma 17 gives

$$
\int _ { \Delta _ { d } } \left| Q _ { t } ( \mathbf { w } ) - Q _ { t - 1 } ( \mathbf { w } ) \right| \mathrm { d } \mathbf { w } \leq \operatorname* { m i n } \left\{ 2 , 2 \sqrt { 2 \gamma } \mathrm { J S } ( \mathbf { u } _ { t } , \mathbf { u } _ { t - 1 } ) \right\} .\tag{46}
$$

Therefore, under the endpoint convention above, for every $q \in [ 0 , 1 ]$

$$
\begin{array} { r l r } {  { \int _ { \Delta _ { d } }  Q _ { t } ( \mathbf { w } ) - Q _ { t - 1 } ( \mathbf { w } )  \mathrm { d } \mathbf { w } \leq \operatorname* { m i n } \{ 2 , 2 \sqrt { 2 \gamma } \mathrm { J S } ( \mathbf { u } _ { t } , \mathbf { u } _ { t - 1 } ) \} } } \\ & { } & { \leq 2 ( 2 \gamma ) ^ { q / 2 } \mathrm { J S } ( \mathbf { u } _ { t } , \mathbf { u } _ { t - 1 } ) ^ { q } . } \end{array}
$$

Summing over time and using $2 ^ { q / 2 } \leq { \sqrt { 2 } }$ yields

$$
\sum _ { t = 2 } ^ { T } \int _ { \Delta _ { d } } \lvert Q _ { t } ( { \bf w } ) - Q _ { t - 1 } ( { \bf w } ) \rvert \mathrm { d } { \bf w } \leq 2 \sqrt { 2 } \gamma ^ { q / 2 } P _ { T , q } ^ { \mathrm { J S } } .\tag{47}
$$

Substituting (47) into (30) and retaining the other bounds from the proof of Theorem 3, we obtain

$$
\begin{array} { r l r } & { } & { \mathrm { D } { \cdot } \mathrm { R e g } _ { T } ( \{ \mathbf { u } _ { t } \} _ { t = 1 } ^ { T } ) \leq T \ln \left( 1 + \displaystyle \frac { d } { \gamma } \right) + 2 \sqrt { 2 } C _ { T , d } \gamma ^ { q / 2 } P _ { T , q } ^ { \mathrm { J S } } } \\ & { } & { \quad + \left( d - 1 \right) \ln ( 1 + \gamma ) + ( 1 + \ln T ) , } \end{array}\tag{48}
$$

where $C _ { T , d } = ( d - 1 ) \ln ( T + 1 ) + d ( \ln d + 1 ) = \mathcal { O } ( d \ln ( d T ) )$ . If $P _ { T , q } ^ { \mathrm { J S } } \leq T ^ { - q / 2 }$ , choose $\gamma = T$ . Equation (48) then gives $\mathcal { O } ( d \ln ( d T ) )$ . Otherwise, choose

$$
\gamma = T ^ { \frac { 2 } { q + 2 } } \bigl ( P _ { T , q } ^ { \mathrm { J S } } \bigr ) ^ { - \frac { 2 } { q + 2 } } \bigl ( \ln ( d T ) \bigr ) ^ { - \frac { 2 } { q + 2 } } .
$$

The condition of this case ensures $\gamma \leq T$ . Using $\ln ( 1 + x ) \leq x$ in (48), the first two terms are both bounded by

$$
\mathcal { O } \left( d T ^ { \frac { q } { q + 2 } } \left( P _ { T , q } ^ { \mathrm { J S } } \right) ^ { \frac { 2 } { q + 2 } } \left( \ln ( d T ) \right) ^ { \frac { 2 } { q + 2 } } \right) ,
$$

while the remaining terms are $\mathcal { O } ( d \ln ( d T ) )$ . Combining the two cases proves the theorem.

## D.2. Calculations for the Rising Concave Path

For the rising concave path, write $p _ { t } = [ \mathbf { u } _ { t } ] _ { 1 } = 3 / 4 - 1 / ( 4 t )$ and $J _ { t } = \mathrm { J S } ( \mathbf u _ { t } , \mathbf u _ { t - 1 } )$ For every $t \geq 2$ , let $\delta _ { t } : = p _ { t } - p _ { t - 1 } = 1 / ( 4 t ( t - 1 ) )$ . Let $m _ { t } = ( p _ { t } + p _ { t - 1 } ) / 2$ and $\bar { \mathbf { u } } _ { t } = ( \mathbf { u } _ { t } + \mathbf { u } _ { t - 1 } ) / 2$ . The JS distance between the consecutive Bernoulli distributions satisfies

$$
J _ { t } ^ { 2 } = \frac { 1 } { 2 } \mathrm { K L } ( \mathbf { u } _ { t } \| \bar { \mathbf { u } } _ { t } ) + \frac { 1 } { 2 } \mathrm { K L } ( \mathbf { u } _ { t - 1 } \| \bar { \mathbf { u } } _ { t } ) .\tag{49}
$$

For either $\theta = p _ { t }$ or $\theta = p _ { t - 1 }$ , the inequality ln $x \leq x - 1$ gives

$$
\mathrm { K L } \bigl ( \mathrm { B e r n } ( \theta ) \| \mathrm { B e r n } ( m _ { t } ) \bigr ) \leq \frac { ( \theta - m _ { t } ) ^ { 2 } } { m _ { t } ( 1 - m _ { t } ) } = \frac { \delta _ { t } ^ { 2 } } { 4 m _ { t } ( 1 - m _ { t } ) } .
$$

Since $m _ { t } \in [ 1 / 2 , 3 / 4 ]$ , each of the two KL divergences is at most $4 \delta _ { t } ^ { 2 } / 3$ . On the other hand, Pinsker’s inequality bounds each of them below by $\delta _ { t } ^ { 2 } / 2$ , since

$$
\left\| \mathbf { u } _ { t } - \bar { \mathbf { u } } _ { t } \right\| _ { 1 } = \left\| \mathbf { u } _ { t - 1 } - \bar { \mathbf { u } } _ { t } \right\| _ { 1 } = \delta _ { t } .
$$

Therefore,

$$
\frac { \delta _ { t } } { \sqrt { 2 } } \leq J _ { t } \leq \frac { 2 \delta _ { t } } { \sqrt { 3 } } .\tag{50}
$$

Consequently, $J _ { t } = \Theta ( [ t ( t - 1 ) ] ^ { - 1 } )$ . Because every transition is nonzero, $P _ { T , 0 } ^ { \mathrm { J S } } = T - 1$ For each fixed $q \in ( 0 , 1 ]$ , summing (50) gives

$$
P _ { T , q } ^ { \mathrm { J S } } = \Theta \left( \sum _ { t = 2 } ^ { T } [ t ( t - 1 ) ] ^ { - q } \right) = \left\{ \begin{array} { l l } { \Theta \big ( T ^ { 1 - 2 q } \big ) , } & { 0 < q < \frac { 1 } { 2 } , } \\ { \Theta ( \ln T ) , } & { q = \frac { 1 } { 2 } , } \\ { \Theta ( 1 ) , } & { \frac { 1 } { 2 } < q \leq 1 . } \end{array} \right.\tag{51}
$$

Suppressing logarithmic factors, substitution into Theorem 7 gives a regret bound of order $d T ^ { \alpha ( q ) }$ , where

$$
\alpha ( q ) = \left\{ \begin{array} { l l } { \displaystyle \frac { 2 - 3 q } { q + 2 } , } & { 0 \leq q \leq \frac { 1 } { 2 } , } \\ { \displaystyle \frac { q } { q + 2 } , } & { \frac { 1 } { 2 } \leq q \leq 1 . } \end{array} \right.\tag{52}
$$

The first branch is strictly decreasing and the second is strictly increasing, so $\alpha ( q )$ is uniquely minimized at $q = 1 / 2$ , where $\alpha ( 1 / 2 ) = 1 / 5$ . In comparison, $\alpha ( 0 ) = 1$ and $\alpha ( 1 ) = 1 / 3$ , yielding the three rates stated in Section 3.3.

## Appendix E. Omitted Proofs for Section 4

Proof of Theorem 10 We begin with a similar mixability-based regret decomposition as Zhang et al. (2025). Let $\begin{array} { r } { \tilde { m } _ { t } ( P _ { t } ) = - \frac { 1 } { \eta } \ln \Big ( \mathbb { E } _ { \mathbf { u } \sim P _ { t } } \big [ e ^ { - \eta \tilde { \ell } _ { t } ( \mathbf { u } ) } \big ] \Big ) } \end{array}$ be the mix loss. The dynamic regret can be decomposed by

$$
\begin{array} { r l } & { \underbrace { \displaystyle \sum _ { t = 1 } ^ { T } \ell _ { t } ( { \mathbf { w } _ { t } } ) - \sum _ { t = 1 } ^ { T } \ell _ { t } ( { \mathbf { u } _ { t } } ) } _ { \mathrm { t = 1 } } \le \displaystyle \sum _ { t = 1 } ^ { T } \widetilde { \ell } _ { t } ( { \mathbf { w } _ { t } } ) - \displaystyle \sum _ { t = 1 } ^ { T } \widetilde { \ell } _ { t } ( { \mathbf { u } _ { t } } ) } \\ & { = \underbrace { \displaystyle \sum _ { t = 1 } ^ { T } \widetilde { \ell } _ { t } ( { \mathbf { w } _ { t } } ) - \sum _ { t = 1 } ^ { T } \widetilde { m } _ { t } ( P _ { t } ) } _ { \mathrm { t e r m ~ \pm \infty ~ } ( \infty ) } + \underbrace { \sum _ { t = 1 } ^ { T } \widetilde { m } _ { t } ( P _ { t } ) - \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { E } _ { \mathbf { u } \sim \mathcal { Q } _ { t } } [ \widetilde { \ell } _ { t } ( { \mathbf { u } } ) ] } _ { \mathrm { t e r m ~ \pm \infty ~ } ( \infty ) } } \\ & { \quad + \underbrace { \sum _ { t = 1 } ^ { T } \mathbb { E } _ { \mathbf { u } \sim \mathcal { Q } _ { t } } [ \widetilde { \ell } _ { t } ( { \mathbf { u } } ) ] - \displaystyle \sum _ { t = 1 } ^ { T } \widetilde { \ell } _ { t } ( { \mathbf { u } _ { t } } ) } _ { \mathrm { t e r m ~ \pm \infty ~ } ( \epsilon ) } , } \end{array}
$$

where the first line is due to Hazan (2016, Lemma 4.2) under the step size setting $\begin{array} { r } { \eta = \frac { 1 } { 5 } \operatorname* { m i n } \left\{ \frac { 1 } { 2 G } , \kappa \right\} } \end{array}$ . In the analysis we choose $Q _ { t } = \mathcal { N } ( \mathbf { u } _ { t } , \sigma ^ { 2 } I _ { d } )$ , where $\sigma > 0$ is a parameter that can be virtually tuned to make the bound tight.

One can handle terms (a) and (c) using arguments similar to those in Zhang et al. (2025). However, the most challenging part is the analysis of term (b), where the domain constraint is enforced via an intractable I-projection of the distribution $P _ { t }$ onto a set of infinite Gaussian mixtures. In our algorithm, instead of performing this intractable projection, we identify that it is suficient to project each component of the mixture distribution $P _ { t }$ individually rather than projecting the mixture as a whole. The latter would require a more in-depth analysis that leverages the two-layer structure of Algorithm 2. This leads to an eficient method. In what follows, we first analyze terms $\mathrm { ( a ) }$ and (c) using arguments similar to those in (Zhang et al., 2025), and then turn to the most challenging term (b).

Bounding term (a). Since $\tilde { \ell } _ { t } ( \mathbf { w } )$ is a quadratic function and the initial distribution of each base base-leaner $B _ { i }$ is a Gaussian, according to van der Hoeven et al. (2018, Theorem 5) shows that the distribution $P _ { t + 1 , i } = \mathcal { N } ( \mathbf { w } _ { t + 1 , i } , H _ { t + 1 , i } ^ { - 1 } )$ for any base-learner $B _ { i }$ updated by (7) is also a Gaussian distribution. More precisely, the mean and covariance matrix can be updated by

$$
\left\{ \begin{array} { r l } & { H _ { t + 1 , i } = H _ { t , i } + 2 \eta ^ { 2 } \mathbf { g } _ { t } \mathbf { g } _ { t } ^ { \top } } \\ & { \mathbf { w } _ { t + 1 , i } ^ { \prime } = \mathbf { w } _ { t , i } - \eta H _ { t + 1 , i } ^ { - 1 } ( 1 - 2 \eta \mathbf { g } _ { t } ^ { \top } ( \mathbf { w } _ { t } - \mathbf { w } _ { t , i } ) ) \mathbf { g } _ { t } } \\ & { \mathbf { w } _ { t + 1 , i } = \arg \operatorname* { m i n } _ { \mathbf { u } \in \mathcal { W } } \lVert \mathbf { u } - \mathbf { w } _ { t + 1 , i } ^ { \prime } \rVert _ { H _ { t + 1 , i } } } \end{array} \right.\tag{53}
$$

The above essentially follows the update procedure of online Newton step (Hazan et al., 2007). The design matrix is also symmetric positive definite and $H _ { t + 1 , i } =$ $\begin{array} { r } { I _ { d } + 2 \eta ^ { 2 } \sum _ { s = i } ^ { t } \mathbf { g } _ { s } \mathbf { g } _ { s } ^ { \top } \leq ( 1 + \frac { d t } { 2 } ) I _ { d } \leq d T I _ { d } } \end{array}$ for any $B _ { i } \in H _ { t }$ and $\mathbf { w } _ { t , i } \in \mathcal { W }$ due to the projection step.

Our goal is to show term $\mathbf { \Sigma } ( \mathsf { a } ) \mathbf { \Sigma } \leq \mathbf { \Sigma } 0$ To show this, it is suficient to have $\mathbb { E } _ { \mathbf { u } \sim P _ { t } } \big [ \exp ( - \eta \tilde { \ell } _ { t } ( \mathbf { u } ) ) \big ] \le \exp \big ( - \eta \tilde { \ell } _ { t } ( \mathbf { w } _ { t } ) \big ) = 1$ for each iteration. This can be achieved by the following arguments

$$
\begin{array} { r l } { \mathbb { E } _ { \mathbf { u } \sim P _ { t } } \left[ \exp ( - \eta \tilde { \ell } _ { t } ( \mathbf { u } ) ) \right] = } & { \displaystyle \sum _ { B _ { i } \in \mathcal { H } _ { t } } p _ { t , i } \mathbb { E } _ { \mathbf { u } \sim P _ { t , i } } [ \exp ( - \eta \tilde { \ell } _ { t } ( \mathbf { u } ) ) ] } \\ & { = \displaystyle \sum _ { B _ { i } \in \mathcal { H } _ { t } } p _ { t , i } \mathbb { E } _ { \mathbf { u } \sim P _ { t , i } } \left[ \exp \left( \eta \mathbf { g } _ { t } ^ { \top } ( \mathbf { w } _ { t } - \mathbf { u } ) - \eta ^ { 2 } \| \mathbf { u } - \mathbf { w } _ { t } \| _ { \mathbf { g } \times \mathbb { R } ^ { \tau } } ^ { 2 } \right) \right] } \\ & { \leq \displaystyle \sum _ { B _ { i } \in \mathcal { H } _ { t } } p _ { t , i } \exp \left( \eta \mathbf { g } _ { t } ^ { \top } ( \mathbf { w } _ { t } - \mathbf { w } _ { t , i } ) - \eta ^ { 2 } \| \mathbf { w } _ { t } - \mathbf { w } _ { t , i } \| _ { \mathbf { g } \times \mathbb { R } ^ { \tau } } ^ { 2 } \right) } \\ & { \leq \displaystyle \sum _ { B _ { i } \in \mathcal { H } _ { t } } p _ { t , i } \left( 1 + \eta \mathbf { g } _ { t } ^ { \top } ( \mathbf { w } _ { t } - \mathbf { w } _ { t , i } ) \right) = 1 , } \end{array}
$$

where the first inequality is due to (van Erven and Koolen, 2016, Lemma 10) under the condition $\eta \leq 1 / ( 1 0 G )$ . The second inequality holds because $e ^ { z - z ^ { 2 } } \leq 1 + z$ for any $z \geq - { \frac { 2 } { 3 } }$ . Then, we can have

$$
\mathrm { { t e r m } \left( a \right) \leq 0 . }\tag{54}
$$

Bounding term (c). A direct calculation according to the definition of $\tilde { \ell } _ { t }$ shows that

$$
\begin{array} { r l r } { \mathrm { t e r m } } & { \langle \mathbf { c } \rangle = \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { E } _ { \mathbf { u } \sim Q _ { t } } \big [ \tilde { \ell } _ { t } ( \mathbf { u } ) \big ] - \sum _ { t = 1 } ^ { T } \tilde { \ell } _ { t } ( \mathbf { u } _ { t } ) } & \\ & { } & \\ & { \quad } & { \quad = \eta \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { E } _ { \mathbf { u } \sim Q _ { t } } \big [ \big ( \mathbf { g } _ { t } ^ { \top } ( \mathbf { u } - \mathbf { u } _ { t } ) \big ) ^ { 2 } \big ] } \\ & { } & \\ & { \quad } & { \quad = \eta \sigma ^ { 2 } \displaystyle \sum _ { t = 1 } ^ { T } \lVert \mathbf { g } _ { t } \rVert _ { 2 } ^ { 2 } \le \eta d G ^ { 2 } T \sigma ^ { 2 } , } \end{array}\tag{55}
$$

where the last inequality is due to $\| \mathbf { g } _ { t } \| _ { 2 } \leq \sqrt { d } \| \mathbf { g } _ { t } \| _ { \infty } \leq \sqrt { d } G$

Bounding term (b). As for term (b), diferent from the previous work (Zhang et al., 2025), we decompose the mix loss by exploiting the two-layer structure. Denote by

$$
\tilde { m } _ { t } ( P _ { t , i } ) = - \frac { 1 } { \eta } \ln \left( \mathbb { E } _ { P _ { t , i } } [ \exp ( - \eta \tilde { \ell } _ { t } ( \mathbf { u } ) ) ] \right)
$$

the mix loss for the individual distribution $P _ { t , i }$ . We can rewrite the mix loss for the aggregated distribution $P _ { t }$ as

$$
\begin{array} { l } { \displaystyle \tilde { m } _ { t } ( P _ { t } ) = - \frac { 1 } { \eta } \ln \left( \sum _ { B _ { i } \in \mathcal { H } _ { t } } p _ { t , i } \cdot \mathbb { E } _ { { \mathbf { u } } \sim P _ { t , i } } [ \exp ( - \eta \tilde { \ell } _ { t } ( { \mathbf { u } } ) ) ] \right) } \\ { = - \frac { 1 } { \eta } \ln \left( \sum _ { B _ { i } \in \mathcal { H } _ { t } } p _ { t , i } \cdot \exp ( - \eta \tilde { m } _ { t } ( P _ { t , i } ) ) \right) } \\ { = \sum _ { B _ { i } \in \mathcal { H } _ { t } } q _ { t , i } \cdot \tilde { m } _ { t } ( P _ { t , i } ) + \frac { 1 } { \eta } \left( \mathrm { K L } \left( \mathbf { q } _ { t } \parallel \mathbf { p } _ { t } \right) - \mathrm { K L } \left( \mathbf { q } _ { t } \parallel \tilde { \mathbf { p } } _ { t + 1 } \right) \right) , } \end{array}\tag{56}
$$

where $\mathbf { p } _ { t } \in \Delta _ { | \mathcal { H } _ { t } | }$ denotes the probability vector over the base-learner pool with the i-th entry $p _ { t , i }$ and $\tilde { p } _ { t + 1 , i } \propto p _ { t , i } \cdot \exp ( - \eta \tilde { m } _ { t } ( P _ { t , i } ) ) = p _ { t , i } \cdot \mathbb { E } _ { P _ { t , i } } [ \exp ( - \eta \tilde { \ell } _ { t } ( \mathbf { u } ) ) ]$ is the same as the one defined in Algorithm 2. The last line holds for any $\mathbf { q } _ { t } \in \Delta _ { | \mathcal { H } _ { t } | }$ that assigns weight to each elements in $\mathcal { H } _ { t }$ and $q _ { t , i }$ is the i-th entry of $\mathbf { q } _ { t }$ . Furthermore, we can also rewrite the mix loss for each base learner $B _ { i }$ as

$$
\begin{array} { r } { \tilde { m } _ { t } ( P _ { t , i } ) = \mathbb { E } _ { \mathbf { u } \sim Q _ { t } } [ \tilde { \ell } _ { t } ( \mathbf { u } ) ] + \displaystyle \frac { 1 } { \eta } \left( \mathrm { K L } \left( Q _ { t } \parallel P _ { t , i } \right) - \mathrm { K L } \left( Q _ { t } \parallel P _ { t + 1 , i } ^ { \prime } \right) \right) } \\ { \leq \mathbb { E } _ { \mathbf { u } \sim Q _ { t } } [ \tilde { \ell } _ { t } ( \mathbf { u } ) ] + \displaystyle \frac { 1 } { \eta } \left( \mathrm { K L } \left( Q _ { t } \parallel P _ { t , i } \right) - \mathrm { K L } \left( Q _ { t } \parallel P _ { t + 1 , i } \right) \right) , } \end{array}\tag{57}
$$

where $P _ { t + 1 , i } ^ { \prime } \propto P _ { t , i } \exp ( - \eta \tilde { \ell } _ { t } ( \mathbf { u } ) )$ is the same as (7) and the last line is to the Pythagorean theorem for KL divergence since $\begin{array} { r } { P _ { t + 1 , i } = \arg \operatorname* { m i n } _ { P ^ { \prime } \in \mathcal { W } } \mathrm { K L } \left( P ^ { \prime } \left\| P _ { t + 1 , i } ^ { \prime } \right) \right. } \end{array}$ and $\mathcal { W }$ is a convex set. Then, plugging (57) back into (56), we arrive

$$
\begin{array} { r l r } & { } & { \tilde { m } _ { t } ( P _ { t } ) \leq \mathbb { E } _ { \mathbf { u } \sim Q _ { t } } [ \tilde { \ell } _ { t } ( \mathbf { u } ) ] + \underbrace { \frac { 1 } { \eta } ( \displaystyle \sum _ { B _ { i } \in \mathcal { H } _ { t } } q _ { t , i } \mathrm { K L } ( Q _ { t } \| P _ { t , i } ) + \mathrm { K L } ( \mathbf { q } _ { t } \| \mathbf { p } _ { t } ) ) } _ { \mathrm { t e r m ~ ( b - 1 ) } }  } \\ & { } & { \qquad \quad \underbrace { - \frac { 1 } { \eta } ( \displaystyle \sum _ { B _ { i } \in \mathcal { H } _ { t } } q _ { t , i } \mathrm { K L } ( Q _ { t } \| P _ { t + 1 , i } ) + \mathrm { K L } ( \mathbf { q } _ { t } \| \tilde { \mathbf { p } } _ { t + 1 } ) ) } _ { \mathrm { t e r m ~ ( b - 2 ) } } } \end{array}\tag{58}
$$

holds for any $\mathbf { q } _ { t } \in \Delta _ { | \mathcal { H } _ { t } | }$ and $Q _ { t }$ . Here, we specify $\mathbf { q } _ { t }$ as the minimizer of the optimization problem

$$
\mathbf { q } _ { t } = \underset { \mathbf { q } \in \Delta _ { | \mathcal { H } _ { t } | } } { \arg \operatorname* { m i n } } \sum _ { \boldsymbol { B } _ { i } \in \mathcal { H } _ { t } } q _ { i } \mathrm { K L } \left( Q _ { t } \parallel P _ { t , i } \right) + \mathrm { K L } \left( \mathbf { q } \parallel \mathbf { p } _ { t } \right) ,
$$

whose optimal value has the close form formulation as

$$
V _ { t } ( Q _ { t } ) = - \ln \left( \sum _ { B _ { i } \in \mathcal { H } _ { t } } p _ { t , i } \cdot \exp ( - \mathrm { K L } \left( Q _ { t } \left. P _ { t , i } \right) \right) \right) \ge 0 .
$$

The value $V _ { t } ( Q _ { t } )$ is always greater than 0 since the objective function of the above optimization problem is non-negative. Then, we have

$$
\mathrm { t e r m ~ \ ( b { - } 1 ) } \leq { \frac { 1 } { \eta } } V _ { t } ( Q _ { t } ) .
$$

As for term (b-2), we can similarly define

$$
\widetilde { \mathbf { q } } _ { t + 1 } = \underset { \mathbf { q } \in \Delta _ { | \mathcal { H } _ { t } | } } { \arg \operatorname* { m i n } } \sum _ { \mathcal { B } _ { i } \in \mathcal { H } _ { t } } q _ { i } \mathrm { K L } ( Q _ { t }  P _ { t + 1 , i } ) + \mathrm { K L } ( \mathbf { q }  \widetilde { \mathbf { p } } _ { t + 1 } ) .
$$

We also have $\begin{array} { r } { \tilde { V } _ { t + 1 } ( Q _ { t } ) = - \ln \Big ( \sum _ { \mathcal { B } _ { i } \in \mathcal { H } _ { t } } \tilde { p } _ { t + 1 , i } \cdot \exp ( - \mathrm { K L } \left( Q _ { t } \parallel P _ { t + 1 , i } \right) ) \Big ) } \end{array}$ as the optimal value of the above optimization problem. Clearly, we have

$$
\mathrm { t e r m ~ \Gamma ( b - 2 ) } = - \frac { 1 } { \eta } \left( \sum _ { B _ { i } \in \mathcal { H } _ { t } } q _ { t , i } \mathrm { K L } \left( Q _ { t } \parallel P _ { t + 1 , i } \right) + \mathrm { K L } \left( \mathbf { q } _ { t } \parallel \tilde { \mathbf { p } } _ { t + 1 } \right) \right) \leq - \frac { 1 } { \eta } \tilde { V } _ { t + 1 } ( Q _ { t } ) ,
$$

since $\mathbf { q } _ { t , i }$ is not the minimizer of the objective function in term (b-2). Plugging the upper bound of term (b-1) and term (b-2) into (58), we have

$$
\tilde { m } _ { t } ( P _ { t } ) \leq \mathbb { E } _ { \mathbf { u } \sim Q _ { t } } [ \tilde { \ell } _ { t } ( \mathbf { u } ) ] + \frac { 1 } { \eta } \left( V _ { t } ( Q _ { t } ) - \tilde { V } _ { t + 1 } ( Q _ { t } ) \right) .\tag{59}
$$

Then, we related $\tilde { V } _ { t + 1 } ( Q _ { t } )$ to $V _ { t + 1 } ( Q _ { t } )$ by

$$
\begin{array} { r l } { { \cal V } _ { t 1 1 } ( Q _ { t } ) = } & { - \ln \left( \displaystyle \sum _ { B _ { i } \in \mathcal { H } _ { t + 1 , i } } p _ { t + 1 , i } \cdot \exp ( - \mathrm { K L } ( Q _ { t } \mid P _ { t _ { 1 } ( 1 , i ) } ) ) \right) } \\ & { = - \ln \left( \left( 1 - \mu _ { t + 1 } \right) \displaystyle \sum _ { B _ { i } \in \mathcal { H } _ { t } } \tilde { p } _ { t + 1 , i } \cdot \exp ( - \mathrm { K L } \left( Q _ { t } \mid P _ { t + 1 , i } \right) ) \right. } \\ & { \qquad \left. + \mu _ { t + 1 } \exp ( - \mathrm { K L } ( Q _ { t } \mid N _ { 0 } ) ) \right) } \\ & { \leq - \ln \left( \displaystyle \sum _ { B _ { i } \in \mathcal { H } _ { t + 1 , i } } \cdot \exp ( - \mathrm { K L } \left( Q _ { t } \mid P _ { t + 1 , i } ) \right) \right) + \ln \left( \displaystyle \frac { 1 } { 1 - \mu _ { t + 1 } } \right) } \\ & { = \tilde { V } _ { t + 1 } ( Q _ { t } ) + \log \left( \displaystyle \frac { t + 1 } { t } \right) , } \end{array}\tag{60}
$$

where the second line is due to the fixed-share update (9) with $N _ { 0 } = \mathcal { N } ( \mathbf { u } _ { 0 } , I _ { d } )$ and the last equality is due to the parameter setting $\mu _ { t + 1 } = 1 / ( t + 1 )$ . Plugging (60) back into (59) and taking a summation over T rounds, we obtain

$$
\begin{array} { r l r } {  { \sum _ { t = 1 } ^ { T } \tilde { m } _ { t } ( P _ { t } ) \le \sum _ { t = 1 } ^ { T } \mathbb { E } _ { \mathbf { u } \sim Q _ { t } } [ \tilde { \ell } _ { t } ( \mathbf { u } ) ] + \frac { 1 } { \eta } \sum _ { t = 1 } ^ { T } ( V _ { t } ( Q _ { t } ) - V _ { t + 1 } ( Q _ { t } ) ) + \sum _ { t = 1 } ^ { T } \frac { 1 } { \eta } \log ( \frac { t + 1 } { t } ) } } \\ & { } & { \le \sum _ { t = 1 } ^ { T } \mathbb { E } _ { \mathbf { u } \sim Q _ { t } } [ \tilde { \ell } _ { t } ( \mathbf { u } ) ] + \frac { 1 } { \eta } \sum _ { t = 2 } ^ { T } ( V _ { t } ( Q _ { t } ) - V _ { t } ( Q _ { t - 1 } ) ) + \frac { 1 } { \eta } V _ { 1 } ( Q _ { 1 } ) + \frac { \log ( T + 1 ) } { \eta } . } \end{array}\tag{61}
$$

where the second inequality holds because $V _ { t } ( Q )$ is always non-negative for any $Q$

It remains to handle the variation term $V _ { t } ( Q _ { t } ) - V _ { t } ( Q _ { t - 1 } )$ . Denote by

$$
h _ { t , i } ( \mathbf { u } ) = \frac { 1 } { 2 } \left( \log | H _ { t , i } ^ { - 1 } | + \sigma ^ { 2 } \mathrm { T r } ( H _ { t , i } ) + \| \mathbf { u } - \mathbf { w } _ { t , i } \| _ { H _ { t , i } } ^ { 2 } \right) ,\tag{62}
$$

where $\operatorname { T r } ( A )$ indicates the trace of a matrix A. Then, the variation term can be expressed as

$$
\begin{array} { r l } & { V _ { t } ( Q _ { t } ) - V _ { t } ( Q _ { t - 1 } ) = \ln \Big ( \displaystyle \frac { \sum _ { B _ { i } \in \mathcal { H } _ { t } } p _ { t , i } \cdot \exp ( - \mathrm { K L } ( Q _ { t - 1 } \| P _ { t , i } ) ) } { \sum _ { B _ { i } \in \mathcal { H } _ { t } } p _ { t , i } \cdot \exp ( - \mathrm { K L } ( Q _ { t } \| P _ { t , i } ) ) } \Big ) } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad = \ln \Big ( \displaystyle \frac { \sum _ { B _ { i } \in \mathcal { H } _ { t } } p _ { t , i } \cdot \exp ( - h _ { t , i } ( \mathbf { u } _ { t - 1 } ) ) } { \sum _ { B _ { i } \in \mathcal { H } _ { t } } p _ { t , i } \cdot \exp ( - h _ { t , i } ( \mathbf { u } _ { t } ) ) } \Big ) } \\ & { \quad \quad \quad \quad \quad \quad \quad = J _ { t } ( \mathbf { u } _ { t } ) - J _ { t } ( \mathbf { u } _ { t - 1 } ) } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \leq \underset { \mathrm { u \in \mathcal { W } } } { \operatorname* { s u p } } \| \nabla J _ { t } ( \mathbf { u } ) \| _ { 2 } \cdot \| \mathbf { u } _ { t } - \mathbf { u } _ { t - 1 } \| _ { 2 } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \leq \underset { \mathrm { u \in \mathcal { W } } } { \operatorname* { s u p } } \| \nabla J _ { t } ( \mathbf { u } ) \| _ { 2 } \cdot \| \mathbf { u } _ { t } - \mathbf { u } _ { t - 1 } \| _ { 1 } } \end{array}\tag{63}
$$

where the second line is due to the definition of KL divergence for Gaussian distributions and we define $\begin{array} { r } { J _ { t } ( \mathbf { u } ) = - \ln \left( \sum _ { \boldsymbol { B } _ { i } \in \mathcal { H } _ { t } } p _ { t , i } \cdot \exp ( - h _ { t , i } ( \mathbf { u } ) ) \right) } \end{array}$ in the last line. The next step is to control the gradient of the function $J _ { t } ( \mathbf { u } )$ , which can be calculated as

$$
\nabla J _ { t } ( \mathbf { u } ) = \sum _ { \boldsymbol { \mathcal { B } } _ { i } \in \mathcal { H } _ { t } } \beta _ { t , i } ( \mathbf { u } ) H _ { t , i } ( \mathbf { u } - \mathbf { w } _ { t , i } ) \quad \forall \mathbf { u } \in \mathcal { W } ,
$$

where $\beta _ { t , i } ( \mathbf { u } ) \in \Delta _ { | \mathcal { H } _ { t } | }$ and $\beta _ { t , i } ( \mathbf { u } ) \propto p _ { t , i } \exp ( - h _ { t , i } ( \mathbf { u } ) )$ . We then bound the norm of $\nabla J _ { t } ( \mathbf { u } )$ by

$$
\begin{array} { r l } { \| \nabla J _ { t } ( \mathbf { u } ) \| _ { 2 } \leq } & { \displaystyle \sum _ { B _ { i } \in \mathcal { H } _ { t } } \beta _ { t , i } ( \mathbf { u } ) \cdot \| H _ { t , i } ( \mathbf { u } - \mathbf { w } _ { t , i } ) \| _ { 2 } } \\ & { \leq \displaystyle \sum _ { B _ { i } \in \mathcal { H } _ { t } } \beta _ { t , i } ( \mathbf { u } ) \cdot \sqrt { \mathrm { T r } ( H _ { t , i } ) } \cdot \| \mathbf { u } - \mathbf { w } _ { t , i } \| _ { H _ { t , i } } } \\ & { \leq \sqrt { \displaystyle \sum _ { B _ { i } \in \mathcal { H } _ { t } } \beta _ { t , i } ( \mathbf { u } ) \| \mathbf { u } - \mathbf { w } _ { t , i } \| _ { H _ { t , i } } ^ { 2 } } \sqrt { \displaystyle \sum _ { B _ { i } \in \mathcal { H } _ { t } } \beta _ { t , i } \mathrm { T r } ( H _ { t , i } ) } } \end{array}\tag{64}
$$

where the second inequality is by Cauchy–Schwarz inequality and $\beta _ { t , i } ( \mathbf { u } ) \in \Delta _ { | \mathcal { H } _ { t } | }$ . The third line holds because $\begin{array} { r } { \| H _ { t , i } ( \mathbf { u } - \mathbf { w } _ { t , i } ) \| _ { 2 } ^ { 2 } \leq \| H _ { t , i } \| _ { 2 } \| \mathbf { u } - \mathbf { w } _ { t , i } \| _ { H _ { t , i } } ^ { 2 } \leq \mathrm { T r } ( H _ { t , i } ) \| \mathbf { u } - \mathbf { w } _ { t , i } \| _ { H _ { t , i } } ^ { 2 } } \end{array}$ i for a symmetric positive definite matrix $H _ { t , i }$

Then, we proceed to relate the above two terms back to $h _ { t , i } ( \mathbf { u } )$ . As shown in (53), $\begin{array} { r } { H _ { t , i } = I _ { d } + 2 \eta ^ { 2 } \sum _ { s = i } ^ { t - 1 } g _ { s } g _ { s } ^ { \top } \preceq \left( 1 + \frac { t d } { 2 } \right) I _ { d } \preceq d T I _ { d } } \end{array}$ for any $\boldsymbol { B } _ { i } \in \mathcal { H } _ { t }$ and $t \in [ T ]$ , which implies $\lambda _ { \operatorname* { m i n } } ( H _ { t , i } ^ { - 1 } ) \geq 1 / ( d T )$ . Consequently, one has log| $H _ { t , i } ^ { - 1 } | \geq - d \log ( d T )$ . Plugging the lower bound into (62) yields

$$
\mathrm { T r } ( H _ { t , i } ) \leq \frac { 2 h _ { t , i } ( \mathbf { u } ) + d \log ( d T ) } { \sigma ^ { 2 } } \quad \mathrm { a n d } \quad \| \mathbf { u } - \mathbf { w } _ { t , i } \| _ { H _ { t , i } } ^ { 2 } \leq 2 h _ { t , i } ( \mathbf { u } ) + d \log ( d T ) .\tag{65}
$$

Then, plugging (65) into (64), we can further upper bound the gradient norm by

$$
\begin{array} { r l } { \| \nabla J _ { t } ( \mathbf u ) \| _ { 2 } \leq \frac { 1 } { \sigma } \left( d \log ( d T ) - 2 \sum _ { b \in \{ M \} } \beta _ { b , ( \mathbf u ) } j _ { b , ( \mathbf u ) } \right) } & { } \\ { \leq \frac { 1 } { \sigma } \left( d \log ( d T ) - 2 \sum _ { b \in \{ M \} } \beta _ { b , ( \mathbf u ) } j _ { b , ( \mathbf u ) } + 2 \mathrm { K L } \left( \beta _ { b } ( \mathbf u ) \right) \| _ { \mathrm { P J } } \right) } & { } \\ { = \frac { 1 } { \sigma } \left( d \log ( d T ) - 2 \ln \left( \sum _ { b \in \{ M \} } p _ { i , \mathrm { c } } \exp ( - h _ { t , \mathrm { c } } ( \mathbf u ) ) \right) \right) } & { } \\ { \leq \frac { 1 } { \sigma } \left( d \log ( d T ) - 2 \ln \left( \mu _ { b \in \ } e ^ { - h _ { t , t } ( \mathbf u ) } \right) \right) } & { } \\ { = \frac { 1 } { \sigma } \left( d \log ( d T ) - 2 \ln \left( \mu _ { b } \cdot e ^ { - a _ { i } \cdot ( \mathbf u ) } \right) \right) } & { } \\ { = \frac { 1 } { \sigma } \left( d \log ( d T ) \mid 2 \ln 1 + \sigma ^ { 2 } d + \| \mathbf u - \mathbf u _ { 0 } \| _ { \infty } ^ { 2 } \right) } & { } \\ { \leq \frac { ( d + 2 ) \log ( d T ) + 4 } { \sigma } d ( 1 + \sigma ^ { 2 } ) } & { } \end{array}\tag{66}
$$

where the third line is by the definition $\beta _ { t , i } ( \mathbf { u } ) \propto p _ { t , i } \exp ( - h _ { t , i } ( \mathbf { u } ) )$ . The fourth line is due to the fixed share update (9) such that there always exists a base algorithm with weight $\mu _ { t } = 1 / t$ and distribution $P _ { t , t } = N _ { 0 } = \mathcal { N } ( \mathbf { u } _ { 0 } , I _ { d } )$ . The last line holds since $\| \mathbf { u } - \mathbf { u } _ { 0 } \| _ { 2 } \leq \| \mathbf { u } - \mathbf { u } _ { 0 } \| _ { 1 } \leq 2$ for any $\mathbf { u } \in \mathcal { W }$ by Assumption 2 and $\sigma \leq 1 + \sigma ^ { 2 }$

Finally, we can upper bound term (b) by

term (b)

$$
= \sum _ { t = 1 } ^ { T } \tilde { m } _ { t } ( P _ { t } ) - \sum _ { t = 1 } ^ { T } \mathbb { E } _ { \mathbf { u } \sim Q _ { t } } [ \tilde { \ell } _ { t } ( \mathbf { u } ) ]
$$

$$
\begin{array} { r l } & { \leq \displaystyle \frac { 1 } { \eta } \left( \frac { ( d + 2 ) \log ( d T ) + 4 } { \sigma } + d \sigma ^ { 2 } + d \right) \sum _ { t = 2 } ^ { T } \lVert \mathbf { u } _ { t } - \mathbf { u } _ { t - 1 } \rVert _ { 1 } + \displaystyle \frac { 1 } { \eta } V _ { 1 } ( Q _ { 1 } ) + \frac { 2 } { \eta } \log ( d T ) } \\ & { = \displaystyle \frac { 1 } { \eta } \left( \frac { ( d + 2 ) \log ( d T ) + 4 } { \sigma } + d \sigma ^ { 2 } + d \right) P _ { T } + \frac { 1 } { \eta } \mathrm { K L } ( Q _ { 1 } \lVert P _ { 1 } ) + \displaystyle \frac { 2 } { \eta } \log ( d T ) } \\ & { \leq \displaystyle \frac { 1 } { \eta } \left( \frac { ( d + 2 ) \log ( d T ) + 4 } { \sigma } + d \sigma ^ { 2 } + d \right) P _ { T } + \displaystyle \frac { 1 } { \eta } \left( 3 + d \log \left( \frac { 1 } { \sigma } \right) + \frac { d \sigma ^ { 2 } } { 2 } + 2 \log ( d T ) \right) , } \end{array}\tag{67}
$$

where the first inequality comes from a combination of (61), (63) and (66). The second equality is by the definition of $V _ { 1 } ( Q _ { 1 } )$ and the last inequality is due to the closed-form expression for the KL divergence between two Gaussian distributions.

Combining All. Combining the upper bounds (54), (67) and (55) on term (a), term (b) and term (c), we obtain

$$
\begin{array} { r l } & { \quad \mathrm { D } { \cdot } \mathrm { R e } _ { T } ( \{ \mathbf { u } _ { t } \} _ { t = 1 } ^ { T } ) } \\ & { \leq \frac { \left( \left( d + 2 \right) \log ( d T ) + 4 \right) P _ { T } } { \eta \sigma } + d \left( \frac { P _ { T } + 1 / 2 } { \eta } + \eta G ^ { 2 } T \right) \sigma ^ { 2 } } \\ & { \quad + \displaystyle \frac { 1 } { \eta } \left( d P _ { T } + 3 + d \log \displaystyle \frac { 1 } { \sigma } + 2 \log ( d T ) \right) } \\ & { \leq \frac { \left( \left( d + 2 \right) \log ( d T ) + 4 \right) P _ { T } } { \eta \sigma } + \frac { 5 d T \sigma ^ { 2 } } { 2 \eta } + \displaystyle \frac { 1 } { \eta } \left( d P _ { T } + 3 + d \log \displaystyle \frac { 1 } { \sigma } + 2 \log ( d T ) \right) . } \end{array}
$$

The last inequality uses $P _ { T } \leq 2 T , \eta \leq 1 / ( 2 G )$ , and $T \geq 2$ . We consider the following two cases:

• Case 1: $P _ { T } \leq T ^ { - 1 / 2 }$ . We choose $\sigma = T ^ { - 1 / 2 }$ . Substituting this choice into the above bound gives

$$
\mathrm { D - R e g } _ { T } ( \{ \mathbf { u } _ { t } \} _ { t = 1 } ^ { T } ) \leq \mathcal { O } \left( \frac { d } { \eta } \ln ( d T ) \right) .
$$

• Case 2: $P _ { T } > T ^ { - 1 / 2 }$ . We choose $\sigma = ( P _ { T } \ln ( e d T ) / T ) ^ { 1 / 3 }$ . Substituting this choice into the above bound gives

$$
\mathrm { D - R e g } _ { T } ( \{ \mathbf { u } _ { t } \} _ { t = 1 } ^ { T } ) \leq \mathcal { O } \left( \frac { d } { \eta } \left[ \ln ( d T ) + T ^ { 1 / 3 } P _ { T } ^ { 2 / 3 } ( \ln ( d T ) ) ^ { 2 / 3 } \right] \right) .
$$

The proof is completed by combining the two cases.

## Appendix F. Technical Lemmas

Lemma 18 Let $\begin{array} { r } { \psi ( u ) = \frac { \mathrm { d } } { \mathrm { d } u } } \end{array}$ $\Gamma ( u )$ be the digamma function. Then, $g ( u ) = \gamma u \psi ( 1 +$ $\gamma u ) - \ln \Gamma ( 1 + \gamma u )$ is a convex function for $u > 0$ and $\gamma > 0$

Proof of Lemma 18 Let $\begin{array} { r } { \psi ( u ) = \frac { \mathrm { d } } { \mathrm { d } u } } \end{array}$ ln $\Gamma ( u )$ and define $g ( u ) = \gamma u \psi ( 1 + \gamma u ) - \ln \Gamma ( 1 +$ $\gamma u )$ for $\gamma > 0$ . Set $x = 1 + \gamma u$ and define $h ( x ) = ( x - 1 ) \psi ( x ) - \ln \Gamma ( x )$ for $x > 1$ Since $g ( u ) = h ( 1 + \gamma u )$ , it sufices to show that h is convex on $( 1 , \infty )$

A direct computation yields $h ^ { \prime } ( x ) = ( x - 1 ) \psi ^ { \prime } ( x )$ and $h ^ { \prime \prime } ( x ) = \psi ^ { \prime } ( x ) + ( x - 1 ) \psi ^ { \prime \prime } ( x )$ Using the integral representations of the polygamma functions, we have

$$
\psi ^ { \prime } ( x ) = \int _ { 0 } ^ { \infty } { \frac { t e ^ { - x t } } { 1 - e ^ { - t } } } \mathrm { d } t , \qquad \psi ^ { \prime \prime } ( x ) = - \int _ { 0 } ^ { \infty } { \frac { t ^ { 2 } e ^ { - x t } } { 1 - e ^ { - t } } } \mathrm { d } t ,
$$

for $x > 1$ . Then, by the integration by parts arguments, we obtain for $x > 1$

$$
{ \begin{array} { r l } & { h ^ { \prime \prime } ( x ) = \displaystyle \int _ { 0 } ^ { \infty } { \frac { t } { e ^ { t } - 1 } } \left( 1 - ( x - 1 ) t \right) e ^ { - ( x - 1 ) t } \mathrm { d } t } \\ & { \qquad = \displaystyle \int _ { 0 } ^ { \infty } { \frac { t } { e ^ { t } - 1 } } { \frac { \mathrm { d } } { \mathrm { d } t } } { \Bigl ( } t e ^ { - ( x - 1 ) t } { \Bigr ) } \mathrm { d } t } \\ & { \qquad = \displaystyle \left[ { \frac { t ^ { 2 } e ^ { - ( x - 1 ) t } } { e ^ { t } - 1 } } \right] _ { 0 } ^ { \infty } - \int _ { 0 } ^ { \infty } { \frac { \mathrm { d } } { \mathrm { d } t } } { \Bigl ( } { \frac { t } { e ^ { t } - 1 } } { \Bigr ) } t e ^ { - ( x - 1 ) t } \mathrm { d } t } \\ & { \qquad = \displaystyle - \int _ { 0 } ^ { \infty } { \frac { \mathrm { d } } { \mathrm { d } t } } { \Bigl ( } { \frac { t } { e ^ { t } - 1 } } { \Bigr ) } t e ^ { - ( x - 1 ) t } \mathrm { d } t . } \end{array} }
$$

The last equality holds because the function $\frac { t ^ { 2 } e ^ { - ( x - 1 ) t } } { e ^ { t } - 1 }  0$ when $t \to 0$ and $t \to \infty$ Since $\frac { \mathrm { d } } { \mathrm { d } t } \left( \frac { t } { e ^ { t } - 1 } \right) < 0$ for $t > 0$ , the integrand is nonnegative and not identically zero. Hence $h ^ { \prime \prime } ( x ) > \bar { 0 }$ for all $x > 1$ , so h is strictly convex. Therefore $g ( u ) = h ( 1 + \gamma u )$ is strictly convex for $u > 0$ and $\gamma > 0$ 7