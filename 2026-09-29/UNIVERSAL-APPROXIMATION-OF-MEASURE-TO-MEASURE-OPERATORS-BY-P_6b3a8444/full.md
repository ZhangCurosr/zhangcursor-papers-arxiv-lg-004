# UNIVERSAL APPROXIMATION OF MEASURE-TO-MEASURE OPERATORS BY PUSHFORWARDS

Takashi Furuya Doshisha University, RIKEN AIP tfuruya@mail.doshisha.ac.jp

Nicholas H. Nelsen UT Austin nnelsen@oden.utexas.edu

Frank Cole   
UCLA   
fcole99@g.ucla.edu

## ABSTRACT

Many learning tasks map an input distribution to an output distribution. A natural way to model such an operator is to transform each input sample using a continuous function that may depend on the entire input distribution, and then take the distribution of the transformed samples. This defines a measure-dependent pushforward model and includes measure-theoretic formulations of transformers. We ask when such models can approximate arbitrary continuous operators between spaces of probability measures. We first show that universal approximation fails when atomic inputs are allowed: some continuous measure-to-measure operators that split or redistribute atomic mass cannot be approximated arbitrarily well by deterministic pushforward models. We then introduce the uniform level set condition, which requires a continuous measure-dependent scalarization whose shrinking level set neighborhoods carry uniformly vanishing mass over the input family. This condition is satisfied, in particular, by compact families of absolutely continuous measures. On every compact family satisfying this condition, we prove that any continuous measure-to-measure operator with outputs of finite p-th moment can be uniformly approximated, in the p-Wasserstein distance, by continuous measuredependent pushforwards. Combining our theorem with existing approximation results for measure-dependent in-context maps yields universal approximation by measure-theoretic transformers. We also extend the framework to continuouslyvarying source measures, yielding a corresponding universality result for a class of pushforward models that are closely aligned with cross-attention architectures.

## 1 INTRODUCTION

Many scientific and learning problems involving distributions can naturally be formulated as maps between spaces of probability measures. Given an input distribution, one would like to produce another distribution that depends continuously on the input. Examples include the prior-to-posterior map in Bayesian inference (Sprungk, 2020; Stuart, 2010), proximal maps on Wasserstein spaces (Ambrosio et al., 2008), and solution maps of McKean–Vlasov dynamics (Kolokoltsov, 2010).

A particularly important class of measure-to-measure maps is given by pushforward models. In such models, each sample from the input distribution is transformed by a map that may itself depend on the entire input distribution, and the collection of transformed samples determines the output distribution. This structure appears naturally in measure-theoretic formulations of transformers and attention mechanisms (Castin et al., 2025; Furuya et al., 2025b; Geshkovski et al., 2025; Rigollet, 2026; Sander et al., 2022; Vuckovic et al., 2020). Closely related pushforward constructions also arise in generative modeling, including flow matching (Lipman et al., 2023) and conditional normalizing flows (Winkler et al., 2019). This leads to the following fundamental approximation question:

To what extent can an arbitrary continuous operator between spaces of probability measures be approximated by continuous pushforward models whose transformation is allowed to depend on the input measure?

In this paper, we first show that universal approximation by continuous pushforward models fails in general when atomic input measures are allowed. We provide counterexamples induced by continuous measure-to-measure operators that split or redistribute atomic mass, which cannot in general be approximated by deterministic pushforward models.

We then identify a sufficient geometric-measure-theoretic condition on a compact family of input measures, which we call the uniform level set condition. Roughly speaking, this condition requires the existence of a continuous measure-dependent scalarization whose shrinking level set neighborhoods carry uniformly vanishing mass across the entire family of input measures. Under this condition, we prove that every continuous measure-to-measure operator with finite p-th moment outputs can be uniformly approximated in the p-Wasserstein distance by continuous pushforward models.

The uniform level set condition is substantially more flexible than absolute continuity with respect to the Lebesgue measure. In particular, it is automatically satisfied by compact families of non-atomic measures in one dimension, as well as by compact families dominated by a common non-atomic reference measure in higher dimensions. It also applies to certain families of singular measures supported on continuously-varying lower-dimensional structures, such as normalized arclength measures on a compact family of injective curves. At the same time, it necessarily excludes atomic measures, which is consistent with the obstruction described previously.

Our results have direct implications for the expressive power of transformers. Measure-theoretic transformers act on an input distribution through a measure-dependent transformation of its tokens and therefore induce a pushforward map at the level of probability measures. Combining our main approximation theorem with existing universality results for in-context maps, we establish universal approximation of continuous measure-to-measure operators by measure-theoretic transformers on compact families satisfying the uniform level set condition. Last, we extend the pushforward approximation framework to continuously-varying source measures, which relates to cross-attention.

Related work. The universal approximation properties of transformers have been extensively studied in the classical sequence-to-sequence setting (Vaswani et al., 2017). Early results established universality for transformers and their sparse variants (Yun et al., 2020a;b), followed by extensions concerning constrained architectures, prompting, approximation rates, and more refined descriptions of the expressive power of attention (Calvello et al., 2025; Chen et al., 2025; Cheng et al., 2026; Havrilla & Liao, 2024; Jiang & Li, 2024; Jiao et al., 2026; Kratsios et al., 2022; Liu et al., 2026; Petrov et al., 2024; Shen et al., 2026; Takakura & Suzuki, 2023; Wang & E, 2024).

A complementary line of work studies transformers in a measure-theoretic formulation, where a collection of tokens is represented by its empirical measure and attention is extended to probability measures. This viewpoint has led to mathematical descriptions of attention and transformer dynamics (Bach et al., 2025; 2026; Burger et al., 2025; Castin et al., 2025; Geshkovski et al., 2025; Rigollet, 2026; Sander et al., 2022; Vuckovic et al., 2020), as well as approximation results for functions whose inputs involve probability measures (Biswal et al., 2025; Cole et al., 2026b; Fraiman, 2026; Furuya et al., 2025b; 2026; Geshkovski et al., 2026) and characterizations of the smoothness of attention layers acting on probability measures (Castin et al., 2024; Kim et al., 2021; Vuckovic et al., 2021).

More closely related to the present work are approximation and exact representation results in which both the input and output are probability measures. In work of Furuya et al. (2025a), transformerinduced measure-to-measure maps are studied through the lens of support-preserving transformations. Related exact transport map representations of continuous measure-to-measure transformations are investigated by Lavenant & Savare (2026). These works naturally lead to structural conditions under´ which the output does not split the mass carried by individual input particles. Such conditions are particularly well suited to empirical measures, since an empirical input is then transformed into another empirical measure with compatible particle structure.

The question studied in the present paper is different. We ask under what conditions on a family of input measures can an arbitrary continuous measure-to-measure operator be approximated by deterministic pushforward models without imposing a support-preserving or non-splitting condition on the target map. This distinction is essential: natural maps such as Bayesian updating or joint measure conditioning may continuously change the weights of atoms and hence fall outside exact deterministic pushforward representations. Our results characterize a regime in which this obstruction disappears at the level of uniform approximation and thereby connect general continuous measure-tomeasure operators with transformer-induced pushforward models.

Contributions. The main contributions of this work are summarized as follows.

(C1) An obstruction caused by atomic measures. We show that continuous measure-dependent pushforward models are not universal on general compact subsets of probability measures (Proposition 2.2). In particular, deterministic pushforwards cannot split atomic mass or freely modify atomic weights. This yields explicit continuous measure-to-measure operators that remain separated from every deterministic pushforward model by a strictly positive 1-Wasserstein distance (Examples 2.3 and 2.4).

(C2) Universal approximation under a uniform level set condition. We introduce the uniform level set condition for compact families of input measures (Assumption 3.1). We prove that, under this condition, every continuous measure-to-measure operator with outputs of finite p-th moment can be uniformly approximated in the p-Wasserstein distance by continuous measure-dependent pushforward maps (Theorem 3.6). The condition necessarily excludes atomic measures, but is satisfied by compact families of non-atomic measures in one dimension, compact families dominated by a common non-atomic reference measure, and certain families of singular measures supported on continuously-varying lower-dimensional structures (Examples 3.3, 3.4, and 3.5).

(C3) Consequences for transformer architectures. By combining our main approximation theorem with existing universality results for measure-theoretic in-context maps, we establish universal approximation of continuous measure-to-measure operators by measure-theoretic transformers on compact families satisfying the uniform level set condition (Theorem 4.4). We further extend the pushforward approximation framework to continuously-varying source measures, which has applications to cross-attention (Corollary 4.6).

Notation. Let | · | denote the Euclidean norm. We denote by ${ \mathcal { P } } ( \mathbb { R } ^ { d } )$ the set of all Borel probability measures on $\mathbb { R } ^ { d }$ and, for $1 \leq p < \infty$ , by $\begin{array} { r } { \mathcal { P } _ { p } ( \mathbb { R } ^ { d } ) : = \{ \mu \in \mathcal { P } ( \mathbb { R } ^ { d } ) \colon \int _ { \mathbb { R } ^ { d } } | x | ^ { p } \mu ( d x ) < \infty \} } \end{array}$ the set of probability measures with finite p-th moment. The subset of non-atomic or atomless probability measures is

$$
\begin{array} { r } { \mathcal { P } _ { \mathrm { n a } } ( \mathbb { R } ^ { d } ) : = \left\{ \mu \in \mathcal { P } ( \mathbb { R } ^ { d } ) \colon \mu ( \{ x \} ) = 0 \ \mathrm { f o r } \mathrm { e v e r y } \ x \in \mathbb { R } ^ { d } \right\} . } \end{array}\tag{1.1}
$$

We equip ${ \mathcal { P } } ( \mathbb { R } ^ { d } )$ with the topology of weak convergence of probability measures. Whenever $\mathcal { P } _ { p } ( \mathbb { R } ^ { d } )$ is considered, it is equipped with the $p \mathrm { . }$ Wasserstein metric $\mathsf { W } _ { p }$ given by

$$
\mathsf { W } _ { p } ( \mu , \nu ) : = \biggl ( \operatorname* { i n f } _ { \pi \in \Pi ( \mu , \nu ) } \int _ { \mathbb { R } ^ { d } \times \mathbb { R } ^ { d } } | x - y | ^ { p } \pi ( d x , d y ) \biggr ) ^ { 1 / p } ,
$$

where $\Pi ( \mu , \nu )$ is the set of couplings of $\mu$ and $\nu .$ . For a measurable map $T \colon  { \mathbb { R } } ^ { d } \to  { \mathbb { R } } ^ { d ^ { \prime } }$ and $\mu \in \mathcal P ( \mathbb { R } ^ { d } )$ , the pushforward measure $T _ { \# } \mu \in \mathcal { P } ( \mathbb { R } ^ { d ^ { \prime } } )$ is defined by $T _ { \# } \mu ( A ) : = \mu ( T ^ { - 1 } ( A ) )$ ) for all $A \subset  { \mathbb { R } } ^ { d ^ { \prime } }$ Borel. For topological spaces $X$ and $Y$ , we denote by ${ \mathcal { C } } ( X , Y )$ the space of continuous maps from X to $Y ;$ the topology on spaces of probability measures is always understood as above.

## 2 THE PUSHFORWARD APPROXIMATION PROBLEM

This section formulates the approximation problem studied in this work, illustrates it with two examples of continuous measure-to-measure operators, and identifies atomic obstructions.

## 2.1 THE MEASURE-TO-MEASURE APPROXIMATION PROBLEM

In what follows, let $p \in [ 1 , \infty ) , d \in \mathbb { N } .$ , and $d ^ { \prime } \in \mathbb { N }$ . We investigate whether for every compact set $K \subset \mathcal { P } _ { p } ( \mathbb { R } ^ { d } )$ with respect to $\mathsf { W } _ { p } .$ , every $F \in \mathcal { C } ( K , \mathcal { P } _ { p } ( \mathbb { R } ^ { d ^ { \prime } } ) )$ , and every $\varepsilon > 0$ , there exists a continuous map $G \in \mathcal { C } ( K \times \mathbb { R } ^ { d } , \mathbb { R } ^ { d ^ { \prime } } )$ such that

$$
\operatorname* { s u p } _ { \mu \in K } \mathsf { W } _ { p } \big ( G ( \mu , \cdot ) _ { \# } \mu , F ( \mu ) \big ) < \varepsilon .
$$

In other words, we ask whether continuous measure-to-measure operators can be uniformly approximated by continuous measure-dependent pushforward models. Pushforward models are especially attractive from a computational perspective (Marzouk et al., 2017). This approximation problem arises naturally in several settings. We illustrate it with an example.

Example 2.1 (Prior-to-posterior map in Bayesian inference). In Bayesian inverse problems, updating a prior probability measure using observed data modeled by a likelihood naturally defines a map from prior to posterior measures (Nelsen & Yang, 2026; Stuart, 2010). Let Φ: $\mathbb { R } ^ { d } \to ]$ R denote the negative log-likelihood associated with the observed data. For a prior measure $\mu \in \mathcal P _ { p } ( \mathbb { R } ^ { d } )$ , the corresponding posterior measure is defined by

$$
F _ { \Phi } ( \mu ) ( d u ) : = \frac { \exp \bigl ( - \Phi ( u ) \bigr ) } { Z _ { \mu } } \mu ( d u ) , \quad \mathrm { w h e r e } \quad Z _ { \mu } : = \int _ { \mathbb { R } ^ { d } } \exp \bigl ( - \Phi ( v ) \bigr ) \mu ( d v ) .
$$

Thus, Bayesian inference defines the prior-to-posterior map $F _ { \Phi } \colon { \mathcal { P } } _ { p } ( \mathbb { R } ^ { d } ) \to { \mathcal { P } } _ { p } ( \mathbb { R } ^ { d } )$ . If Φ is continuous and bounded from below, then $F _ { \Phi }$ is well-defined and $\mathsf { W } _ { p } .$ -continuous (Sprungk, 2020, Lemma 16). Notice that $F _ { \Phi }$ generally changes the weights of the input measure while preserving its null sets. In particular, for an atomic prior $\begin{array} { r } { \mu = \sum _ { i = 1 } ^ { N } a _ { i } \delta _ { x _ { i } } } \end{array}$ , the posterior is

$$
F _ { \Phi } ( \mu ) = \sum _ { i = 1 } ^ { N } \frac { a _ { i } e ^ { - \Phi ( x _ { i } ) } } { \sum _ { j = 1 } ^ { N } a _ { j } e ^ { - \Phi ( x _ { j } ) } } \delta _ { x _ { i } } .
$$

Hence, although the support is unchanged, the weights are in general different. Consequently, $F _ { \Phi } ( \mu )$ cannot in general be represented as a deterministic pushforward of the form $G ( \mu , \cdot ) _ { \# } \mu$ . This provides a natural example of a continuous measure-to-measure operator that lies beyond exact deterministic pushforward representations on atomic and nonatomic measures, while still falling within the approximation problem considered in this work.

See Appendix A.1 for a further example based on proximal maps between Wasserstein spaces.

## 2.2 OBSTRUCTIONS CAUSED BY ATOMIC MEASURES

In general, the answer to the approximation question posed in Section 2.1 is negative when atomic input measures are allowed. More precisely, the following holds.

Proposition 2.2. There exist a compact set $K \subset \mathcal { P } _ { p } ( \mathbb { R } ^ { d } )$ in $\mathsf { W } _ { p } ,$ , a continuous map $F \in \mathbf { \Sigma }$ $\mathcal { C } ( K , \mathcal { P } _ { p } ( \mathbb { R } ^ { d ^ { \prime } } ) )$ , and $\varepsilon > 0$ such that there is no continuous map G : $K \times \mathbb { R } ^ { d } \to \mathbb { R } ^ { d ^ { \prime } }$ satisfying

$$
\operatorname* { s u p } _ { \mu \in K } \mathsf { W } _ { p } \big ( G ( \mu , \cdot ) _ { \# } \mu , F ( \mu ) \big ) < \varepsilon .
$$

The obstruction already appears in the simplest possible examples.

Example 2.3 (Different support cardinalities). Let $K : = \{ \delta _ { 0 } \} \subset \mathcal { P } ( \mathbb { R } )$ and define

$$
F ( \delta _ { 0 } ) : = \frac { 1 } { 2 } \delta _ { - 1 } + \frac { 1 } { 2 } \delta _ { 1 } .
$$

Since K is a singleton, $F \colon K \to { \mathcal { P } } _ { 1 } ( \mathbb { R } )$ is continuous. For any continuous $G \colon K \times \mathbb { R } \to \mathbb { R }$ , we have $G ( \delta _ { 0 } , \cdot ) _ { \# } \delta _ { 0 } = \delta _ { G ( \delta _ { 0 } , 0 ) }$ . Therefore,

$$
\mathsf { W } _ { 1 } \big ( G ( \delta _ { 0 } , \cdot ) _ { \# } \delta _ { 0 } , F ( \delta _ { 0 } ) \big ) = \frac { 1 } { 2 } \left| G ( \delta _ { 0 } , 0 ) + 1 \right| + \frac { 1 } { 2 } \left| G ( \delta _ { 0 } , 0 ) - 1 \right| \geq 1
$$

by the triangle inequality. Hence

$$
\operatorname* { i n f } _ { G \in { \mathcal { C } } ( K \times \mathbb { R } , \mathbb { R } ) } \operatorname* { s u p } _ { \mu \in K } \mathbb { W } _ { 1 } \big ( G ( \mu , \cdot ) _ { \# } \mu , F ( \mu ) \big ) \geq 1 .
$$

Thus, a deterministic pushforward cannot approximate a map that splits a single input atom into two.

The preceding example changes the number of atoms. However, this is not the essential issue: an obstruction remains even when the input and output measures have the same support cardinality. Example 2.4 (Same support cardinality, different weights). Let $\begin{array} { r } { K : = \left\{ \frac { 1 } { 2 } \delta _ { - 1 } + \frac { 1 } { 2 } \delta _ { 1 } \right\} } \end{array}$ and define

$$
\nu = \frac { 1 } { 2 } \delta _ { - 1 } + \frac { 1 } { 2 } \delta _ { 1 } \quad \mathrm { a n d } \quad F ( \nu ) : = \frac { 1 } { 3 } \delta _ { - 1 } + \frac { 2 } { 3 } \delta _ { 1 } .
$$

Again, $F \colon K \to { \mathcal { P } } _ { 1 } ( \mathbb { R } )$ is continuous. For any continuous map $G \colon K \times \mathbb { R } \to \mathbb { R }$ , it holds that

$$
G ( \nu , \cdot ) _ { \# } \nu = { \frac { 1 } { 2 } } \delta _ { G ( \nu , - 1 ) } + { \frac { 1 } { 2 } } \delta _ { G ( \nu , 1 ) } .
$$

Proposition A.2 delivers the lower bound

$$
\mathsf { W } _ { 1 } \big ( G ( \nu , \cdot ) _ { \# } \nu , F ( \nu ) \big ) \geq \frac { 1 } { 3 } .
$$

Consequently,

$$
\operatorname* { i n f } _ { G \in { \mathcal { C } } ( K \times \mathbb { R } , \mathbb { R } ) } \operatorname* { s u p } _ { \mu \in K } \mathsf { W } _ { 1 } \big ( G ( \mu , \cdot ) _ { \# } \mu , F ( \mu ) \big ) \geq \frac { 1 } { 3 } .
$$

Thus, even when the input and output distributions have the same number of atoms, a deterministic pushforward cannot in general modify their weights.

## 3 UNIVERSAL APPROXIMATION BY CONTINUOUS PUSHFORWARD MODELS

This section establishes our main universal approximation result for continuous measure-dependent pushforward models.

## 3.1 UNIFORM LEVEL SET CONDITION

We introduce a sufficient condition on the family K of input measures under which universal approximation holds.

Assumption 3.1. Let $K \subset \mathcal { P } _ { p } ( \mathbb { R } ^ { d } )$ be $\mathsf { W } _ { p }$ -compact. There exists $\varphi \in \mathcal { C } ( K \times \mathbb { R } ^ { d } , \mathbb { R } )$ such that

$$
\operatorname* { l i m } _ { r \downarrow 0 } \operatorname* { s u p } _ { \mu \in K } \operatorname* { s u p } _ { t \in \mathbb { R } } \mu \big ( \{ x \in \mathbb { R } ^ { d } \colon | \varphi ( \mu , x ) - t | \leq r \} \big ) = 0 .\tag{3.1}
$$

We call (3.1) the uniform level set condition. Intuitively, Assumption 3.1 requires the existence of a continuous scalarization $\varphi$ whose level sets carry uniformly vanishing mass over the family K. In particular, the condition is satisfied when all measures in $\dot { K }$ are absolutely continuous with respect to Lebesgue measure: one may take, for example, any linear projection $\varphi ( \mu , x ) : = x \cdot v$ in nonzero direction v independently of $\mu .$ . More generally, as shown below, absolute continuity with respect to Lebesgue measure is far from necessary. The importance of Assumption 3.1 is that it enables us to represent the one-dimensional uniform distribution Unif[0, 1] exactly by a continuous pushforward model. This is a key ingredient in the proof of Theorem $3 . 6 ,$ , our main approximation result.

The uniform level set condition is closely related to the atomless property (1.1).

Remark 3.2 (Uniform level set condition implies non-atomic family). If K is such that Assumption 3.1 holds, then $\dot { K } \subset \mathcal { P } _ { \mathrm { n a } } ( \mathbb { R } ^ { d } )$ ). Indeed, fix $\mu \in K$ and $x _ { 0 } \in \mathbb { R } ^ { d }$ , and set $t = \varphi ( \mu , x _ { 0 } )$ . Then

$$
\{ x _ { 0 } \} \subset \{ x \in \mathbb { R } ^ { d } \colon | \varphi ( \mu , x ) - \varphi ( \mu , x _ { 0 } ) | \leq r \}
$$

for any $r \geq 0$ . Hence

$$
\mu \big ( \{ x _ { 0 } \} \big ) \leq \operatorname* { s u p } _ { \nu \in K } \operatorname* { s u p } _ { t \in \mathbb { R } } \nu \big ( \{ x \in \mathbb { R } ^ { d } \colon | \varphi ( \nu , x ) - t | \leq r \} \big )  0
$$

as $r \downarrow 0$ by (3.1). Since $x _ { 0 }$ was arbitrary, $\mu$ is non-atomic as asserted.

Moreover, the uniform level set condition (3.1) corresponding to $K$ and $\varphi$ is equivalent to the family $\{ \varphi ( \mu , \cdot ) _ { \# } \mu \} _ { \mu \in K } \subset \mathcal { P } _ { \mathrm { n a } } ( \mathbb { R } )$ of scalar pushforwards being non-atomic; see Remark ${ \mathrm { A } } . 3$

To better understand Assumption 3.1, we present several examples of compact families that satisfy it. Example 3.3 (One-dimensional case). Let $K \subset \mathcal { P } _ { p } ( \mathbb { R } )$ be compact with respect to $\mathsf { W } _ { p }$ . Then the identity function $\varphi : \mathbb { R }  \mathbb { R }$ given by $\varphi ( x ) : = x$ satisfies

$$
\operatorname* { l i m } _ { r \downarrow 0 } \operatorname* { s u p } _ { \mu \in K } \operatorname* { s u p } _ { t \in \mathbb { R } } \mu \big ( \{ x \in \mathbb { R } \colon | \varphi ( x ) - t | \leq r \} \big ) = 0 .
$$

In particular, every compact family of non-atomic probability measures on $\mathbb { R }$ satisfies the uniform level set condition (3.1). See Appendix B.1 for the proof.

Remark 3.2 shows that if a compact subset $K \subset \mathcal { P } _ { p } ( \mathbb { R } ^ { d } )$ satisfies Assumption 3.1, then every element of $K$ is atomless. Example 3.3 establishes the converse when $d = 1$ ; this gives a complete characterization of subsets satisfying Assumption 3.1 in the one-dimensional setting.

Example 3.4 (Families dominated by a common non-atomic measure). Let $\rho \in \mathcal { P } _ { \operatorname* { n a } } ( \mathbb { R } ^ { d } )$ and let

$$
K \subset \mathcal { P } _ { \mathrm { a c } , p } ( \rho ) : = \{ \mu \in \mathcal { P } _ { p } ( \mathbb { R } ^ { d } ) \colon \mu \ll \rho \}
$$

be compact with respect to $\mathsf { W } _ { p }$ . Then there exists a continuous function $\varphi \colon  { \mathbb { R } ^ { d } } \to  { \mathbb { R } }$ such that

$$
\operatorname* { l i m } _ { r \downarrow 0 } \operatorname* { s u p } _ { \mu \in K } \operatorname* { s u p } _ { t \in \mathbb { R } } \mu \big ( \{ x \in \mathbb { R } ^ { d } \colon | \varphi ( x ) - t | \leq r \} \big ) = 0 .
$$

See Appendix B.2 for the proof. By taking $\rho$ to be the Lebesgue measure, Example 3.4 implies that Assumption 3.1 is satisfied by any $\mathsf { W } _ { p }$ -compact family of densities. Importantly, the only requirement in Example 3.4 is that $\rho$ be non-atomic. In particular, Example 3.4 applies to compact families of probability measures that are supported on a low-dimensional manifold M and admit a density with respect to some fixed atomless reference measure on ${ \mathcal { M } } , \mathbf { e } . \mathbf { g } .$ ., a uniform measure on $\mathcal { M } .$ . Such families of measures arise naturally under the manifold hypothesis in machine learning.

Example 3.5 (Measures supported on a compact family of curves). Let $Q \subset \mathbb { R } ^ { d }$ be compact and let $K ^ { \prime } \subset \mathsf { \bar { C } } ^ { 1 } ( [ 0 , 1 ] , Q )$ be compact. Assume that there exists $c > 0$ such that

$$
| \dot { \gamma } ( s ) | \geq c \quad \mathrm { f o r } \quad \gamma \in K ^ { \prime } \quad \mathrm { a n d } \quad s \in [ 0 , 1 ] ,
$$

and that every $\gamma \in K ^ { \prime }$ is injective. For each $\gamma \in K ^ { \prime }$ , define the normalized arclength measure

$$
\sigma _ { \gamma } : = \left( \int _ { 0 } ^ { 1 } | { \dot { \gamma } } ( s ) | d s \right) ^ { - 1 } \gamma _ { \# } { \big ( } | { \dot { \gamma } } ( s ) | d s { \big ) } .
$$

Then there exists a continuous function $\widetilde { \varphi } \colon K ^ { \prime } \times Q \to [ 0 , 1 ]$ such that ${ \widetilde { \varphi } } ( \gamma , \gamma ( s ) ) = s \operatorname { f o r } \gamma \in K ^ { \prime }$ and $s \in [ 0 , 1 ]$ , and

$$
\operatorname* { l i m } _ { r \downarrow 0 } \operatorname* { s u p } _ { \gamma \in K ^ { \prime } } \operatorname* { s u p } _ { t \in \mathbb { R } } \sigma _ { \gamma } \big ( \{ x \in Q \colon | \widetilde { \varphi } ( \gamma , x ) - t | \leq r \} \big ) = 0 .
$$

Assume, in addition, that the curves in $K ^ { \prime }$ are uniquely determined by their images, that is,

$$
\gamma _ { 1 } ( [ 0 , 1 ] ) = \gamma _ { 2 } ( [ 0 , 1 ] ) \quad \mathrm { i m p l i e s } \quad \gamma _ { 1 } = \gamma _ { 2 } .
$$

Then the family $K : = \{ \sigma _ { \gamma } : \gamma \in K ^ { \prime } \} \subset \mathcal { P } _ { p } ( Q )$ satisfies the uniform level set condition (3.1).

See Appendix B.3 for the proof. Example 3.5 illustrates that the uniform level set condition can accommodate singular measures whose supports themselves vary continuously. Such families naturally arise when data distributions are concentrated near evolving or parameter-dependent lowdimensional structures, rather than on a single fixed reference manifold.

## 3.2 MAIN RESULT

We are now ready to state our main universal approximation theorem. The key idea is to use the uniform level set condition to find a continuous in-context map that exactly pushes forward each input measure to a common uniform distribution on [0, 1].

Theorem 3.6. Let $K \subset \mathcal { P } _ { p } ( \mathbb { R } ^ { d } )$ satisfy Assumption 3.1. Thenfor any $F \in { \mathcal { C } } ( K , { \mathcal { P } } _ { p } ( \mathbb { R } ^ { d ^ { \prime } } ) )$ and any $\varepsilon > 0 ,$ , there exists a continuous map $G \colon K \times \mathbb { R } ^ { d } \to \mathbb { R } ^ { d ^ { \prime } }$ such that

$$
\operatorname* { s u p } _ { \mu \in K } \mathsf { W } _ { p } \big ( G ( \mu , \cdot ) _ { \# } \mu , F ( \mu ) \big ) < \varepsilon .
$$

Sketch ofproof. The full proof is given in Appendix B.4. Here we describe the construction together with the estimates used below. For $R > 0$ , let

$$
F _ { R } ( { \boldsymbol { \mu } } ) : = ( \Pi _ { R } ) _ { \# } F ( { \boldsymbol { \mu } } ) ,
$$

where $\Pi _ { R }$ is the metric projection onto the closed ball $\overline { { B _ { R } ( 0 ) } }$ of radius R. Define

$$
E _ { \mathrm { t a i l } , p } ( R ) : = \operatorname* { s u p } _ { \nu \in F ( K ) } \left( \int _ { \{ | y | > R \} } | y | ^ { p } \nu ( d y ) \right) ^ { 1 / p } .
$$

By Lemma B.2, $E _ { \mathrm { t a i l } , p } ( R ) \to 0$ as $R \to \infty$ and

$$
\operatorname* { s u p } _ { \mu \in K } { \mathsf { W } } _ { p } \big ( F ( \mu ) , F _ { R } ( \mu ) \big ) \leq E _ { \mathrm { t a i l } , p } ( R ) .
$$

Next, for $h > 0 ,$ choose an h-net $\{ y _ { 1 } , \dotsc , y _ { N } \} \subset { \overline { { B _ { R } ( 0 ) } } }$ and a continuous partition of unity subordinate to the corresponding cover. The covering number may be chosen so that

$$
N = N ( R , h ) \leq C _ { d ^ { \prime } } \left( 1 + \frac { R } { h } \right) ^ { d ^ { \prime } } ,
$$

where $C _ { d ^ { \prime } } > 0$ depends only on $d ^ { \prime } .$ . This yields continuous weights $\alpha _ { i } \colon K \to [ 0 , 1 ]$ such that

$$
\operatorname* { s u p } _ { \mu \in K } \mathsf { W } _ { p } \big ( F _ { R , h } ( \mu ) , F _ { R } ( \mu ) \big ) \leq h , \quad \mathrm { w h e r e } \quad F _ { R , h } ( \mu ) : = \sum _ { i = 1 } ^ { N } \alpha _ { i } ( \mu ) \delta _ { y _ { i } } .
$$

Writing $\begin{array} { r } { s _ { i } ( { \boldsymbol \mu } ) : = \sum _ { j = 1 } ^ { i } \alpha _ { j } ( { \boldsymbol \mu } ) } \end{array}$ and $s _ { 0 } ( \mu ) : = 0$ , the discontinuous interval map

$$
u \mapsto Q _ { 0 } ( \mu , u ) : = \sum _ { i = 1 } ^ { N } \mathbf { 1 } _ { ( s _ { i - 1 } ( \mu ) , s _ { i } ( \mu ) ] } ( u ) y _ { i }
$$

pushes $\mathrm { U n i f } [ 0 , 1 ]$ exactly onto $F _ { R , h } ( \mu )$ . We smooth the transition points of $Q _ { 0 }$ on intervals of width $\delta > 0$ to obtain a bounded continuous map $Q _ { \delta }$ whose $\mathsf { W } _ { p }$ contribution to errors in the transition regions is bounded by $2 R { \left( 2 N ( R , h ) \delta \right) } ^ { 1 / p }$ . Under Assumption 3.1, the construction in Lemma B.3 delivers the existence of $H _ { 0 } \in \mathcal { C } ( K \times \mathbb { R } ^ { d } , [ 0 , 1 ] )$ such that

$$
H _ { 0 } ( \mu , \cdot ) _ { \# } \mu = \mathrm { U n i f } [ 0 , 1 ] \quad \mathrm { f o r ~ a l l } \quad \mu \in K .
$$

Combining this exact continuous uniformizer with the preceding arguments and defining

$$
G ( \mu , x ) : = Q _ { \delta } \bigl ( \mu , H _ { 0 } ( \mu , x ) \bigr ) ,
$$

the triangle inequality gives the asserted approximation after successively choosing R, h, and δ appropriately as functions of ε. □

For a fixed $\mu \in \mathcal { P } _ { \mathrm { n a } } ( \mathbb { R } ^ { d } )$ , it is known that the set of deterministic Monge couplings $( \operatorname { i d } , g ) _ { \# } \mu$ induced by continuous transport maps $g \colon  { \mathbb { R } } ^ { d } \to  { \mathbb { R } } ^ { d ^ { \prime } }$ with source measure $\mu$ is dense in the set of all joint distributions in $\mathcal { P } ( \mathbb { R } ^ { d } \times \mathbb { R } ^ { d ^ { \prime } } )$ with first marginal $\mu$ (Beiglbock & Lacker, 2018, Prop. 2.2).¨ Theorem 3.6 can then be interpreted as the operator analog of this “diagonal approximation” result in which $\mu$ is no longer fixed, but is instead allowed to vary uniformly over the compact set $K$

Remark 3.7 (Quantitative error bound). The preceding proof actually yields the upper bound

$$
\operatorname* { s u p } _ { \mu \in K } { \mathsf { W } } _ { p } \big ( G ( \mu , \cdot ) _ { \# } \mu , F ( \mu ) \big ) \leq E _ { \mathrm { t a i l } , p } ( R ) + h + 2 R \left[ 2 C _ { d ^ { \prime } } \left( 1 + \frac R h \right) ^ { d ^ { \prime } } \delta \right] ^ { 1 / p } .
$$

Moreover, the construction gives the uniform bound $\| G \| _ { \mathcal { C } ( K \times \mathbb { R } ^ { d } , \mathbb { R } ^ { d ^ { \prime } } ) } \leq R$

## 4 APPLICATIONS TO TRANSFORMER ARCHITECTURES

We now apply our main approximation theorem to measure-theoretic transformer architectures.

## 4.1 MEASURE-THEORETIC TRANSFORMERS

We first recall the measure-theoretic formulation of transformers.

Definition 4.1 (Measure-theoretic in-context attention maps). For $H \in \mathbb { N } .$ , a measure-theoretic in-context multi-head attention map $\Gamma _ { \theta } \colon \mathcal { P } ( \mathbb { R } ^ { d } ) \times \mathbb { R } ^ { d } \to \mathbb { R } ^ { \dot { d } }$ is defined by

$$
\Gamma _ { \theta } ( \mu , x ) : = x + \sum _ { h = 1 } ^ { H } W ^ { h } \int _ { \mathbb { R } ^ { d } } \frac { \exp \bigl ( \langle Q ^ { h } x , K ^ { h } y \rangle \bigr ) } { \int _ { \mathbb { R } ^ { d } } \exp \bigl ( \langle Q ^ { h } x , K ^ { h } z \rangle \bigr ) \mu ( d z ) } V ^ { h } y \mu ( d y ) ,\tag{4.1}
$$

where $\theta : = ( W ^ { h } , Q ^ { h } , K ^ { h } , V ^ { h } ) _ { h = 1 } ^ { H } , ( \mu , x ) \in \mathcal { P } ( \mathbb { R } ^ { d } ) \times \mathbb { R } ^ { d }$ , and $W ^ { h } , Q ^ { h } , K ^ { h }$ , and $V ^ { h }$ are the head, query, key, and value matrices, respectively. See Remark 4.5 for a note about finiteness of (4.1).

In (4.1), $\Gamma _ { \theta }$ acts on a distinguished token x while depending on the input probability measure $\mu ,$ and can therefore be viewed as an in-context map. The standard discrete attention mechanism is recovered when $\mu$ is an empirical measure $\begin{array} { r } { \mu = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \delta _ { x _ { i } } } \end{array}$ associated with tokens $x _ { 1 } , \ldots , x _ { n }$ . In this case, the integral in the definition of $\Gamma _ { \theta }$ reduces to the usual finite sum over the tokens. The measure-theoretic formulation removes the need to fix the number of tokens in advance and provides a unified description of attention for arbitrary probability measures, including non-atomic measures. For further details on measure-theoretic formulations of attention and in-context maps, see, e.g., the work of Castin et al. (2024; 2025); Furuya et al. (2025a;b); Geshkovski et al. (2023; 2025; 2026).

Definition 4.2 (Composition of measure-theoretic in-context maps). Let

$$
\Gamma _ { 1 } \colon \mathcal { P } ( \mathbb { R } ^ { d _ { 0 } } ) \times \mathbb { R } ^ { d _ { 0 } } \to \mathbb { R } ^ { d _ { 1 } } \quad \mathrm { ~ a n d ~ } \quad \Gamma _ { 2 } \colon \mathcal { P } ( \mathbb { R } ^ { d _ { 1 } } ) \times \mathbb { R } ^ { d _ { 1 } } \to \mathbb { R } ^ { d _ { 2 } } .
$$

Their composition is defined by

$$
\big ( \Gamma _ { 2 } \diamond \Gamma _ { 1 } \big ) ( \mu , x ) : = \Gamma _ { 2 } \big ( \Gamma _ { 1 } ( \mu , \cdot ) _ { \# } \mu , \Gamma _ { 1 } ( \mu , x ) \big ) .
$$

Definition 4.3 (Measure-theoretic transformer). Let $\Gamma _ { \theta _ { 1 } } , \dots , \Gamma _ { \theta _ { I } }$ be measure-theoretic in-context attention maps as in (4.1) and $\phi _ { 1 } , \ldots , \phi _ { L }$ be context-free multilayer perceptron (MLP) maps. The associated deep transformer in-context map T: $\mathcal { P } ( \mathbb { R } ^ { d } ) \times \mathbb { R } ^ { d }  \bar { \mathbb { R } ^ { d ^ { \prime } } }$ is defined by

$$
{ \sf T } : = \phi _ { L } \diamond \Gamma _ { \theta _ { L } } \diamond \cdot \cdot \cdot \diamond \phi _ { 1 } \diamond \Gamma _ { \theta _ { 1 } } .\tag{4.2}
$$

The corresponding measure-theoretic transformer is the measure-to-measure operator defined by

$$
\mathcal { P } ( \mathbb { R } ^ { d } ) \ni \mu \mapsto \mathsf { T } ( \mu , \cdot ) _ { \# } \mu \in \mathcal { P } ( \mathbb { R } ^ { d ^ { \prime } } ) .
$$

Furuya et al. (2025b) show that transformer in-context maps of the form (4.2) are universal approxi mators for general continuous in-context mappings defined on compact token domains; the crucial difficulty is that the present paper works on the whole of $\mathbb { R } ^ { d }$ , not a compact domain. Nevertheless, combining modifications of this universality result (Proposition A.7) with Theorem 3.6 yields the following universal approximation result for measure-to-measure operators.

Theorem 4.4. Let $K \subset \mathcal { P } _ { p } ( \mathbb { R } ^ { d } )$ be a $\mathsf { W } _ { p }$ -compact set satisfying Assumption 3.1. Then for any $F \in$ $\mathcal { C } ( K , \mathcal { P } _ { p } ( \mathbb { R } ^ { d ^ { \prime } } ) )$ and any $\varepsilon > 0$ , there exists a deep transformer in-context map T : $\mathcal { P } ( \mathbb { R } ^ { d } ) \times \mathbb { R } ^ { d }  \mathbb { R } ^ { d ^ { \prime } }$ oftheform (4.2) such that

$$
\operatorname* { s u p } _ { \mu \in K } \mathsf { W } _ { p } \big ( \mathsf { T } ( \mu , \cdot ) _ { \# } \mu , F ( \mu ) \big ) < \varepsilon .
$$

See Appendix C.1 for the proof. Theorem 4.4 proves that the measure-to-measure operators defined by transformers are universal approximators of continuous measure-to-measure operators defined on a compact set $K \subset \mathcal { P } _ { p } ( \mathbb { R } ^ { d } )$ , provided that $K$ satisfies the uniform level set condition (3.1). In many practical scenarios, the transformer T is not applied to $\mu \in K$ directly—which is atomless by Remark 3.2—but instead to an atomic measure $\mu _ { N }$ supported on N samples from $\mu .$ The error analysis in this case can still be handled by first controlling the $\mathsf { W } _ { p }$ error between $\mathsf { T } ( \mu _ { N } , \cdot ) _ { \# } \mu _ { N }$ and $\mathsf { T } ( { \boldsymbol \mu } , \cdot \mathsf { ) } _ { \# } { \boldsymbol \mu }$ uniformly over K (Cole et al., 2026a) and then applying Theorem 4.4 to bound the $\mathsf { W } _ { p }$ error between ${ \sf T } ( \mu , \cdot ) _ { \# } \mu$ and $F ( \mu )$ . This approach requires Assumption 3.1 and discretizes $\mu \in K$ with samples. A different approach removes Assumption 3.1 on the $\mathsf { W } _ { p }$ -compact set K altogether by convolving the (now possibly atomic) input measure $\mu \in K$ with a standard isotropic Gaussian distribution that has a small variance $\sigma ^ { 2 }$ to obtain a distribution $\mu _ { \sigma } .$ ; see Corollary A.6 in Appendix A.3 for details. Given samples $X _ { 1 } , \ldots , X _ { N }$ from $\mu ,$ , one can combine the two approaches by approximating $\mu _ { \sigma }$ by an empirical measure of the form $\begin{array} { r } { \widehat { \mu } _ { \sigma } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \delta _ { X _ { i } + \sigma Z _ { i } } } \end{array}$ , where $Z _ { 1 } , \dots , Z _ { N }$ are independent Gaussian samples with mean zero and identity covariance matrix.

Remark 4.5 (Domain of measure-theoretic attention). To be precise, the measure-theoretic attention map Γ in (4.1) is only well-defined for inputs $( \mu , x )$ such that

$$
\int _ { { \mathbb R } ^ { d } } \exp \bigr ( \langle Q ^ { h } x , K ^ { h } y \rangle \bigr ) \mu ( d y ) < \infty \quad \mathrm { a n d } \quad \int _ { { \mathbb R } ^ { d } } \exp \bigr ( \langle Q ^ { h } x , K ^ { h } y \rangle \bigr ) V ^ { h } y \mu ( d y ) < \infty
$$

for all $h \in \{ 1 , \ldots , H \}$ . These conditions always hold when $\mu$ has compact support; Cole et al. (2026a) give more general conditions on $\mu$ based on sub-Gaussianity. Although Theorem 4.4 applies to input measures $\mu \in K$ with possibly full support on $\mathbb { R } ^ { d }$ , the specific construction only requires evaluating self-attention layers at compactly supported measures. This is because the transformer constructed in the proof of Theorem 4.4 utilizes an explicit initial MLP layer which truncates the input measure to a common compact set.

## 4.2 CONTINUOUSLY-VARYING SOURCE MEASURES AND CONNECTION TO CROSS-ATTENTION

The preceding result is an approximation theorem for measure-theoretic transformers based on self-attention, in which the source measure of the transport is the input measure itself. The next corollary generalizes the main pushforward approximation result, Theorem 3.6 from Section 3.2, to the setting in which the source measure of the pushforward is allowed to depend continuously on the input measure, but need not equal the input measure. This is particularly relevant for analyzing cross-attention architectures, where the measure carrying the query tokens may differ from the measure representing the context and may even live on a different ambient space.

Corollary 4.6. Let $K \subset \mathcal { P } _ { p } ( \mathbb { R } ^ { d } )$ be $\mathsf { W } _ { p }$ -compact. Let η : $K  \mathcal { P } _ { p } ( \mathbb { R } ^ { m } )$ be $\mathsf { W } _ { p }$ -continuous. Assume that there exists $\varphi \in \mathcal { C } ( K \times \mathbb { R } ^ { m } , \mathbb { R } )$ such that

$$
\operatorname* { l i m } _ { r \downarrow 0 } \operatorname* { s u p } _ { \mu \in K } \operatorname* { s u p } _ { t \in \mathbb { R } } \eta ( \mu ) \big ( \{ x \in \mathbb { R } ^ { m } \colon | \varphi ( \mu , x ) - t | \leq r \} \big ) = 0 .\tag{4.3}
$$

Then for every $F \in { \mathcal { C } } ( K , { \mathcal { P } } _ { p } ( \mathbb { R } ^ { d ^ { \prime } } ) )$ ) and every $\varepsilon > 0 ,$ , there exists $G \in \mathcal { C } ( K \times \mathbb { R } ^ { m } , \mathbb { R } ^ { d ^ { \prime } } )$ such that

$$
\operatorname* { s u p } _ { \mu \in K } \mathsf { W } _ { p } \big ( G ( \mu , \cdot ) _ { \# } \eta ( \mu ) , F ( \mu ) \big ) < \varepsilon .
$$

Typical source measures covered by Corollary 4.6 include a fixed non-atomic measure, $\eta ( \mu ) \equiv \rho$ (e.g., an isotropic Gaussian), and a smoothed input measure, $\eta ( \mu ) = \rho * \mu$ , where $\rho$ is absolutely continuous. The proof of Corollary 4.6 follows the same construction as that of Theorem 3.6 and may be found in Appendix C.2.

Remark 4.7 (Relation to cross-attention). Cross-attention allows one collection of query tokens to attend to a distinct collection of context tokens and is a central component of encoder–decoder transformers (Vaswani et al., 2017) and related architectures such as the Perceiver (Jaegle et al., 2021). Corollary 4.6 is directly connected to architectures in which the source tokens attend to the context tokens through cross-attention and are otherwise processed token-wise, without self-attention among the source tokens. Combined with universality for measure-dependent in-context maps—see, e.g., Theorem 1 due to Furuya et al. (2025b) or Proposition A.7—it yields an analog of Theorem 4.4 for this particular cross-attention configuration. However, this argument does not by itself establish universality for more general architectures that combine both self-attention among source tokens with cross-attention among context tokens.

## 5 CONCLUSION

In this paper, we studied the approximation of continuous measure-to-measure operators by continuous pushforward models, a subset of mappings which are naturally motivated by transformers. We showed that atomic measures present an obstruction to universal approximation due to the inability of pushforward models to split or adjust the weights of atoms. To remove this obstruction, we identified a sufficient condition for universal approximation on a compact set $K \subset \mathcal { P } _ { p } ( \mathbb { R } ^ { d } )$ , which we termed the uniform level set condition. Notably, the uniform level set condition is weaker than absolute continuity and allows for measures whose supports admit low-dimensional structure. Our main result also implies a uniform approximation theorem for measure-theoretic transformers.

This work is the first to study universal approximation properties of pushforward models under general assumptions on the target operator; however, several important questions still remain. First, while the uniform level set condition is sufficient for universal approximation, it may not be necessary. An interesting mathematical question is to identify the minimal assumptions required on a compact set $K \subset { \mathcal { P } } _ { p } ( \mathbb { R } ^ { d } )$ to enable universal approximation of continuous operators on K by pushforward models. Second, certain measure-to-measure operators of interest, such as the joint-to-conditional operator in Bayesian inference, may not be continuous (Tsimpos et al., 2026, Appendix E); approximation of such maps by pushforward models is thus beyond the scope of this paper. Last, an important problem is obtaining quantitative approximation results for measure-theoretic transformers, where the number of parameters is controlled by the smoothness of the target operator and the covering number of the domain of approximation. We leave these important directions to future work.

## AI USE STATEMENT

In this work, we used generative AI tools to assist in the writing of proofs, formulating mathematical claims and refining them, providing critical ingredients for proving mathematical claims, and assist with translation. We also used generative AI tools for checking proof arguments, improving the organization, presentation, readability, and structure of the paper, language editing, brainstorming, and literature search assistance. We have not used generative AI tools to help develop theoretica models or conceptual frameworks, propose or refine hypotheses, design or provide feedback on research methodology or experiments, or interpret results. Moreover, we did not use generative AI tools to generate synthetic data sets, implement methods, clean and reformat datasets, or support qualitative and thematic data analysis; these tasks are not applicable to this theoretical work.

All mathematical statements and proofs suggested or modified with the assistance of generative AI were independently examined and verified by the authors. Relevant references and attribution were checked against the original sources, and the final mathematical arguments and wording were reviewed and revised by the authors. We take responsibility for the final content of this work, including text, claims, and other artifacts produced with the aid of generative AI.

## ACKNOWLEDGMENTS

T.F. acknowledges funding from JSPS KAKENHI (grants JP24K16949 and 25H01453), JST CREST (JPMJCR24Q5), and JST ASPIRE (JPMJAP2329). The research of N.H.N. is supported by a Klarman Fellowship through Cornell University’s College of Arts & Sciences and by startup funds at The University of Texas at Austin. This project arose out of the 2nd Workshop on Machine Learning in Infinite Dimensions, which was sponsored by the ETH Zurich and the ProbAI Hub on¨ the Mathematical and Computational Foundations of AI.

## REFERENCES

Luigi Ambrosio, Nicola Gigli, and Giuseppe Savare.´ Gradient flows: In metric spaces and in the space ofprobability measures. Springer Science & Business Media, 2008.

Eviatar Bach, Ricardo Baptista, Edoardo Calvello, Bohan Chen, and Andrew M Stuart. Learning enhanced ensemble filters. Journal ofComputational Physics, art. 114550, 2025.

Eviatar Bach, Ricardo Baptista, Jochen Brocker, Bohan Chen, and Andrew M Stuart. Learning¨ probabilistic filters with strictly proper scoring rules. preprint arXiv:2606.26497, 2026.

Mathias Beiglbock and Daniel Lacker. Denseness of adapted processes among causal couplings.¨ preprint arXiv:1805.03185, 2018.

Shiba Biswal, Karthik Elamvazhuthi, and Rishi Sonthalia. Universal approximation of mean-field models via transformers. In International Conference on Machine Learning, pp. 4455–4470, 2025.

Martin Burger, Samira Kabri, Yury Korolev, Tim Roith, and Lukas Weigand. Analysis of mean-field models arising from self-attention dynamics in transformer architectures with layer normalization. Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences, 383(2298):20240233, 2025.

Edoardo Calvello, Nikola B Kovachki, Matthew E Levine, and Andrew M Stuart. Continuum attention for neural operators. Journal ofMachine Learning Research, 26(300):1–52, 2025.

Valerie Castin, Pierre Ablin, and Gabriel Peyr´ e. How smooth is attention? In´ International Conference on Machine Learning, pp. 5817–5840. PMLR, 2024.

Valerie Castin, Pierre Ablin, Jos´ e Antonio Carrillo, and Gabriel Peyr´ e. A unified perspective on the´ dynamics of deep transformers. Foundations ofComputational Mathematics, 2025.

Yifang Chen, Xiaoyu Li, Yingyu Liang, Zhenmei Shi, and Zhao Song. Fundamental limits of visual autoregressive transformers: Universal approximation abilities. In International Conference on Machine Learning, 2025.

Jingpu Cheng, Ting Lin, Zuowei Shen, and Qianxiao Li. A unified framework for establishing the universal approximation of transformer-type architectures. In Advances in Neural Information Processing Systems, volume 38, pp. 129131–129163, 2026.

Frank Cole, Nicholas H Nelsen, and Takashi Furuya. Stability of measure-to-measure transformers on sub-Gaussian data. In preparation, 2026a.

Frank Cole, Dixi Wang, Yineng Chen, Yulong Lu, and Rongjie Lai. In-context operator learning on the space of probability measures. preprint arXiv:2601.09979, 2026b.

James Dugundji. An extension of Tietze’s theorem. Pacific Journal ofMathematics, 1(3):353–367, 1951.

Demian Fraiman. On the expressive power of contextual relations in transformers. ´ preprint arXiv:2603.25860, 2026.

Takashi Furuya, Maarten V de Hoop, and Matti Lassas. Transformers through the lens of supportpreserving maps between measures. preprint arXiv:2509.25611, 2025a.

Takashi Furuya, Maarten V de Hoop, and Gabriel Peyre. Transformers are universal in-context´ learners. In International Conference on Learning Representations, 2025b.

Takashi Furuya, Davide Murari, and Carola-Bibiane Schonlieb. Approximation theory for Lipschitz¨ continuous transformers. In International Conference on Machine Learning, 2026.

Borjan Geshkovski, Cyril Letrouit, Yury Polyanskiy, and Philippe Rigollet. The emergence of clusters in self-attention dynamics. In Advances in Neural Information Processing Systems, volume 36, pp. 57026–57037, 2023.

Borjan Geshkovski, Cyril Letrouit, Yury Polyanskiy, and Philippe Rigollet. A mathematical perspective on transformers. Bulletin ofthe American Mathematical Society, 62(3):427–479, 2025.

Borjan Geshkovski, Philippe Rigollet, and Domenec Ruiz-Balet. Measure-to-measure interpolation\` using transformers. Foundations of Computational Mathematics, pp. 1–50, 2026.

Alex Havrilla and Wenjing Liao. Understanding scaling laws with statistical and approximation theory for transformer neural networks on intrinsically low-dimensional data. In Advances in Neural Information Processing Systems, volume 37, pp. 42162–42210, 2024.

Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, and Joao Carreira. Perceiver: General perception with iterative attention. In International Conference on Machine Learning, pp. 4651–4664. PMLR, 2021.

Haotian Jiang and Qianxiao Li. Approximation rate of the transformer architecture for sequence modeling. In Advances in Neural Information Processing Systems, volume 37, pp. 68926–68955, 2024.

Yuling Jiao, Yanming Lai, Defeng Sun, Yang Wang, and Bokai Yan. Approximation bounds for transformer networks with application to regression. In International Conference on Machine Learning, 2026.

Hyunjik Kim, George Papamakarios, and Andriy Mnih. The Lipschitz constant of self-attention. In International Conference on Machine Learning, pp. 5562–5571. PMLR, 2021.

Vassili N Kolokoltsov. Nonlinear Markov processes and kinetic equations, volume 182. Cambridge University Press, 2010.

Anastasis Kratsios, Behnoosh Zamanlooy, Tianlin Liu, and Ivan Dokmanic. Universal approxima-´ tion under constraints is possible with transformers. In International Conference on Learning Representations, 2022.

Hugo Lavenant and Giuseppe Savare. Continuous transformations of probability measures and their´ transport representations. preprint arXiv:2604.16653, 2026.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023.

Hude Liu, Jerry Yao-Chieh Hu, Zhao Song, and Han Liu. Attention mechanism, max-affine partition, and universal approximation. In Advances in Neural Information Processing Systems, volume 38, pp. 120443–120511, 2026.

Youssef Marzouk, Tarek Moselhy, Matthew Parno, and Alessio Spantini. Sampling via measure transport: An introduction. Springer Books, pp. 785–825, 2017.

Nicholas H Nelsen and Yunan Yang. Operator learning meets inverse problems: A probabilistic perspective. In Andreas Hauptmann, Bangti Jin, and Carola-Bibiane Schonlieb (eds.),¨ Handbook ofNumerical Analysis, volume 27. Elsevier, 2026.

Aleksandar Petrov, Philip Torr, and Adel Bibi. Prompting a pretrained transformer can be a universal approximator. In International Conference on Machine Learning, pp. 40523–40550. PMLR, 2024.

Philippe Rigollet. The mean-field dynamics of transformers. In International Congress of Mathematicians 2026, pp. 389–404. SIAM, 2026.

Michael E Sander, Pierre Ablin, Mathieu Blondel, and Gabriel Peyre. Sinkformers: Transformers with´ doubly stochastic attention. In International Conference on Artificial Intelligence and Statistics, pp. 3515–3530, 2022.

Zhaiming Shen, Alexander Hsu, Rongjie Lai, and Wenjing Liao. Understanding in-context learning on structured manifolds: Bridging attention to kernel methods. In International Conference on Learning Representations, volume 2026, pp. 42067–42103, 2026.

Bjorn Sprungk. On the local Lipschitz stability of Bayesian inverse problems.¨ Inverse Problems, 36 (5):055015, 2020.

Andrew M Stuart. Inverse problems: A Bayesian perspective. Acta Numerica, 19:451–559, 2010.

Shokichi Takakura and Taiji Suzuki. Approximation and estimation ability of transformers for sequence-to-sequence functions with infinite dimensional input. In International Conference on Machine Learning, pp. 33416–33447. PMLR, 2023.

Panos Tsimpos, Edoardo Calvello, Ayoub Belhadji, and Nicholas H Nelsen. One operator for many densities: Amortized approximation of conditioning by neural operators. preprint arXiv:2605.06873, 2026.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, 2017.

James Vuckovic, Aristide Baratin, and Remi Tachet des Combes. A mathematical theory of attention. preprint arXiv:2007.02876, 2020.

James Vuckovic, Aristide Baratin, and Remi Tachet des Combes. On the regularity of attention. preprint arXiv:2102.05628, 2021.

Mingze Wang and Weinan E. Understanding the expressive power and mechanisms of transformer for sequence modeling. In Advances in Neural Information Processing Systems, volume 37, pp. 25781–25856, 2024.

Christina Winkler, Daniel Worrall, Emiel Hoogeboom, and Max Welling. Learning likelihoods with conditional normalizing flows. preprint arXiv:1912.00042, 2019.

Chulhee Yun, Srinadh Bhojanapalli, Ankit Singh Rawat, Sashank J Reddi, and Sanjiv Kumar. Are transformers universal approximators of sequence-to-sequence functions? In International Conference on Learning Representations, 2020a.

Chulhee Yun, Yin-Wen Chang, Srinadh Bhojanapalli, Ankit Singh Rawat, Sashank Reddi, and Sanjiv Kumar. O(n) connections are expressive enough: Universal approximability of sparse transformers. In Advances in Neural Information Processing Systems, volume 33, pp. 13783–13794, 2020b.

# Supplementary Material for: Universal Approximation of Measure-to-Measure Operators by Pushforwards

## A AUXILIARY RESULTS

This appendix contains auxiliary results and additional supporting material.

## A.1 ADDITIONAL MATERIAL FOR SECTION 2

We now provide another example of a continuous measure-to-measure operator; this one is based on proximal maps. Proximal maps and minimizing-movement schemes can be defined on general metric spaces; see, e.g., the work of Ambrosio et al. (2008). Here we consider such a construction on the 1-Wasserstein space.

Example A.1 $( \mathsf { W } _ { 1 }$ -proximal map). Let $\Omega \subset \mathbb { R } ^ { d }$ be compact with positive Lebesgue measure. Define the entropy functional

$$
\operatorname { E n t } ( \nu ) : = { \left\{ \begin{array} { l l } { \int _ { \Omega } \rho ( x ) \log \rho ( x ) d x , } & { { \mathrm { i f ~ } } \mathrm { d e n s i t y ~ } \rho : = { \frac { d \nu } { d x } } { \mathrm { ~ e x i s t s } } , } \\ { + \infty , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }
$$

For $\tau > 0$ and $\mu \in \mathscr { P } _ { 1 } ( \Omega )$ , consider the variational problem

$$
F _ { \tau } ( \mu ) : = \operatorname * { a r g m i n } _ { \nu \in \mathcal { P } _ { 1 } ( \Omega ) } \left\{ \mathrm { E n t } ( \nu ) + \frac { 1 } { 2 \tau } \mathsf { W } _ { 1 } ^ { 2 } ( \mu , \nu ) \right\} .
$$

Since $\Omega$ is compact, $\mathcal { P } _ { 1 } ( \Omega ) = \mathcal { P } ( \Omega )$ and this space is compact with respect to $\mathsf { W } _ { 1 }$ . Moreover, Ent is lower semicontinuous and strictly convex, while

$$
\nu \mapsto \mathsf { W } _ { 1 } ^ { 2 } ( \mu , \nu )
$$

is convex. Hence, the above minimization problem admits a unique minimizer. Therefore, $F _ { \tau }$ defines a measure-to-measure operator

$$
\begin{array} { r } { F _ { \tau } \colon \mathcal { P } _ { 1 } ( \Omega ) \to \mathcal { P } _ { 1 } ( \Omega ) , } \\ { \mu \mapsto F _ { \tau } ( \mu ) . } \end{array}
$$

We next observe that $F _ { \tau }$ is continuous with respect to $\mathsf { W } _ { 1 }$ . Indeed, let $\mu _ { n } \to \mu$ in $\mathsf { W } _ { 1 }$ and set

$$
\nu _ { n } : = F _ { \tau } ( \mu _ { n } ) .
$$

By compactness of $\mathscr { P } _ { 1 } ( \Omega )$ , every subsequence of $( \nu _ { n } ) _ { n \in \mathbb { N } }$ has a further subsequence, still denoted by $( \nu _ { n } ) _ { n \in \mathbb { N } } ,$ , such that

$$
\nu _ { n }  \nu _ { * } \mathrm { ~ \quad ~ i n ~ \quad ~ W _ { 1 } ~ }
$$

for some $\nu _ { \ast } \in \mathscr { P } _ { 1 } ( \Omega )$ . For every $\eta \in \mathcal { P } _ { 1 } ( \Omega )$ , the minimizing property gives

$$
\operatorname { E n t } ( \nu _ { n } ) + \frac { 1 } { 2 \tau } \mathsf { W } _ { 1 } ^ { 2 } ( \mu _ { n } , \nu _ { n } ) \leq \operatorname { E n t } ( \eta ) + \frac { 1 } { 2 \tau } \mathsf { W } _ { 1 } ^ { 2 } ( \mu _ { n } , \eta ) .
$$

Using the lower semicontinuity of Ent and the continuity of $\mathsf { W } _ { 1 }$ , we obtain

$$
\operatorname { E n t } ( \nu _ { * } ) + \frac { 1 } { 2 \tau } \mathsf { W } _ { 1 } ^ { 2 } ( \mu , \nu _ { * } ) \leq \operatorname { E n t } ( \eta ) + \frac { 1 } { 2 \tau } \mathsf { W } _ { 1 } ^ { 2 } ( \mu , \eta ) .
$$

Since $\eta$ is arbitrary, $\nu _ { * }$ is a minimizer of the variational problem associated with $\mu .$ By uniqueness,

$$
\nu _ { * } = F _ { \tau } ( \mu ) .
$$

Hence every convergent subsequence of $( \nu _ { n } ) _ { n \in \mathbb { N } }$ has the same limit $F _ { \tau } ( \mu )$ , and therefore

$$
F _ { \tau } ( \mu _ { n } ) \to F _ { \tau } ( \mu ) \quad \mathrm { i n } \quad \mathsf { W } _ { 1 } .
$$

Thus,

$$
\begin{array} { r } { F _ { \tau } \in \mathcal { C } \big ( \mathcal { P } _ { 1 } ( \Omega ) , \mathcal { P } _ { 1 } ( \Omega ) \big ) . } \end{array}
$$

This map also illustrates a limitation of exact deterministic pushforward representations on atomic measures. Suppose that $0 \in \Omega$ and consider the input

$$
\mu : = \delta _ { 0 } .
$$

The minimum value in the definition of $F _ { \tau } ( \delta _ { 0 } )$ is finite, since there exist absolutely continuous probability measures on Ω with finite entropy. Consequently,

$$
\begin{array} { r } { \operatorname { E n t } \left( F _ { \tau } ( \delta _ { 0 } ) \right) < \infty , } \end{array}
$$

and hence

$$
F _ { \tau } ( \delta _ { 0 } ) \ll d x .
$$

In particular, $F _ { \tau } ( \delta _ { 0 } )$ is non-atomic. On the other hand, for every deterministic map

$$
G \colon \mathcal { P } _ { 1 } ( \Omega ) \times \Omega  \Omega ,
$$

we have

$$
G ( \delta _ { 0 } , \cdot ) _ { \# } \delta _ { 0 } = \delta _ { G ( \delta _ { 0 } , 0 ) } ,
$$

which is a Dirac measure. Therefore,

$$
F _ { \tau } ( \delta _ { 0 } ) \neq G ( \delta _ { 0 } , \cdot ) _ { \# } \delta _ { 0 }
$$

for every such $G .$ Thus, $\mathsf { W } _ { 1 }$ -proximal maps provide natural examples of continuous measure-tomeasure operators which, on atomic inputs, cannot in general be represented exactly by deterministic pushforward models.

To conclude this appendix, we provide the proof of the lower bound asserted in Example 2.4.

Proposition A.2. Let

$$
\nu _ { u , v } : = \frac { 1 } { 2 } ( \delta _ { u } + \delta _ { v } ) a n d \nu ^ { * } : = \frac { 1 } { 3 } \delta _ { - 1 } + \frac { 2 } { 3 } \delta _ { 1 } ,
$$

where $u \leq v .$ . Then

$$
\operatorname* { i n f } _ { u \leq v } \mathsf { W } _ { 1 } ( \nu _ { u , v } , \nu ^ { * } ) = \frac { 1 } { 3 } .
$$

The minimum is attained at $( u , v ) = ( - 1 , 1 )$

Proof. In one dimension, the 1-Wasserstein distance admits the quantile representation

$$
\mathsf { W } _ { 1 } ( \mu , \nu ) = \int _ { 0 } ^ { 1 } \left| Q _ { \mu } ( t ) - Q _ { \nu } ( t ) \right| d t ,
$$

where $Q _ { \mu }$ and $Q _ { \nu }$ denote the corresponding quantile functions. For

$$
\nu _ { u , v } = \frac { 1 } { 2 } ( \delta _ { u } + \delta _ { v } ) \quad \mathrm { a n d } \quad u \leq v ,
$$

we have

$$
Q _ { \nu _ { u , v } } ( t ) = { \left\{ \begin{array} { l l } { u , } & { 0 < t < { \frac { 1 } { 2 } } , } \\ { v , } & { { \frac { 1 } { 2 } } \leq t < 1 , } \end{array} \right. }
$$

whereas

$$
Q _ { \nu ^ { * } } ( t ) = { \left\{ \begin{array} { l l } { - 1 , } & { 0 < t < { \frac { 1 } { 3 } } , } \\ { 1 , } & { { \frac { 1 } { 3 } } \leq t < 1 . } \end{array} \right. }
$$

Hence

$$
\begin{array} { l } { \displaystyle \mathsf { W } _ { 1 } \big ( \nu _ { u , v } , \nu ^ { * } \big ) = \int _ { 0 } ^ { 1 / 3 } \big | u + 1 \big | d t + \int _ { 1 / 3 } ^ { 1 / 2 } \big | u - 1 \big | d t + \int _ { 1 / 2 } ^ { 1 } \big | v - 1 \big | d t } \\ { \displaystyle \qquad = \frac 1 3 | u + 1 | + \frac 1 6 | u - 1 | + \frac 1 2 | v - 1 | . } \end{array}
$$

We now minimize this expression under the constraint $u \leq v$ . If $u \leq 1$ , the choice $v = 1$ is admissible and minimizes the last term. Thus it remains to minimize

$$
f ( u ) : = \frac 1 3 | u + 1 | + \frac 1 6 | u - 1 | \quad \mathrm { f o r } \quad u \leq 1 .
$$

For $u \leq - 1$

$$
f ( u ) = - \frac { 1 } { 2 } u - \frac { 1 } { 6 } ,
$$

which is decreasing as u increases and therefore attains its minimum at $u = - 1$ . For $- 1 \leq u \leq 1$

$$
f ( u ) = \frac { 1 } { 2 } + \frac { 1 } { 6 } u ,
$$

which is increasing and hence again attains its minimum at $u = - 1$ . Therefore,

$$
\operatorname* { i n f } _ { u \leq 1 } f ( u ) = f ( - 1 ) = { \frac { 1 } { 3 } } .
$$

If $u > 1$ , then the constraint $v \geq u$ implies

$$
| v - 1 | = v - 1 \geq u - 1 .
$$

Consequently,

$$
\begin{array} { l } { \displaystyle { \mathsf { W } _ { 1 } \big ( \nu _ { u , v } , \nu ^ { * } \big ) \geq \frac { 1 } { 3 } \big ( u + 1 \big ) + \frac { 1 } { 6 } \big ( u - 1 \big ) + \frac { 1 } { 2 } \big ( u - 1 \big ) } } \\ { \displaystyle { \phantom { \frac { 1 } { 1 } \bigg ( } = u - \frac { 1 } { 3 } > \frac { 1 } { 3 } . } } \end{array}
$$

Thus the global minimum is $\textstyle { \frac { 1 } { 3 } }$ , attained at

$$
( u , v ) = ( - 1 , 1 )
$$

as claimed.

## A.2 ADDITIONAL MATERIAL FOR SECTION 3

As discussed in Section 3, the uniform level set condition (3.1) for a compact set $K$ and continuous scalarization $\varphi \in \mathcal { C } ( K \times \mathbb { R } ^ { d } , \mathbb { R } )$ is equivalent to the condition that $\varphi ( \mu , \cdot ) _ { \# } \mu$ is non-atomic for every $\mu \in K$ . We now elaborate on the proof of this fact.

Remark A.3 (Equivalence of uniform level set condition and pointwise non-atomic family). Let $K \subset \mathcal { P } _ { p } ( \mathbb { R } ^ { d } )$ be W<sub>p</sub>-compact. Let $\varphi \in \mathcal { C } ( K \times \mathbb { R } ^ { d }$ , R). The following are equivalent:

$$
\operatorname* { l i m } _ { r \downarrow 0 } \operatorname* { s u p } _ { \mu \in K } \operatorname* { s u p } _ { t \in \mathbb { R } } \mu \big ( \{ x \in \mathbb { R } ^ { d } \colon | \varphi ( \mu , x ) - t | \leq r \} \big ) = 0 \quad \mathrm { a n d }\tag{A.1}
$$

$$
\mu { \big ( } \{ x \in \mathbb { R } ^ { d } \colon \varphi ( \mu , x ) = t \} { \big ) } = 0 \quad { \mathrm { f o r ~ a l l } } \quad \mu \in K \quad { \mathrm { a n d ~ a l l } } \quad t \in \mathbb { R } .\tag{A.2}
$$

Indeed, $( \mathsf { A } . 1 )$ implies $( \mathsf { A } . 2 )$ by the argument leading to (B.5) in the proof of Theorem 3.6. To see the other direction, first define $\nu _ { \mu } : = \varphi ( \mu , \cdot ) _ { \# } \mu$ . Let (A.2) hold, which is equivalent to

$$
\nu _ { \mu } \in \mathcal { P } _ { \mathrm { n a } } ( \mathbb { R } ) \quad \mathrm { f o r ~ a l l } \quad \mu \in K .
$$

Define

$$
L : = \{ \nu _ { \mu } \colon \mu \in K \} \subset { \mathcal { P } } _ { \mathrm { n a } } ( \mathbb { R } ) .
$$

We claim that the map

$$
\mu \mapsto \nu _ { \mu }
$$

is continuous from $K \subset ( \mathcal { P } _ { p } ( \mathbb { R } ^ { d } ) , \mathsf { W } _ { p } )$ into ${ \mathcal { P } } ( \mathbb { R } )$ with the weak topology. To this end, let $\mu _ { n }  \mu$ in $\mathsf { W } _ { p }$ and hence in distribution. Let f be any continuous and bounded function on R. Then

$$
\int _ { \mathbb { R } ^ { d } } f \big ( \varphi ( \mu , x ) \big ) \mu _ { n } ( d x )  \int _ { \mathbb { R } ^ { d } } f \big ( \varphi ( \mu , z ) \big ) \mu ( d z ) \quad \mathrm { a s } \quad n  \infty
$$

by the weak convergence of $\mu _ { n } \to \mu$ because $f \circ \varphi ( \mu , \cdot )$ is bounded and continuous. On the other hand, for any $R > 0$ , it holds that

$$
\left| \int _ { \mathbb { R } ^ { d } } \Bigl [ f \bigl ( \varphi ( \mu _ { n } , x ) \bigr ) - f \bigl ( \varphi ( \mu , x ) \bigr ) \Bigr ] \mu _ { n } ( d x ) \right| \le \operatorname* { s u p } _ { x \in \overline { { B _ { R } ( 0 ) } } } \bigl | f \bigl ( \varphi ( \mu _ { n } , x ) \bigr ) - f \bigl ( \varphi ( \mu , x ) \bigr ) \bigr |
$$

by splitting the integral. For fixed $R ,$ the first term on the right-hand side of the preceding display tends to zero as $n \to \infty$ by continuity of $f \circ \varphi$ and compactness of closed $\mathbb { R } ^ { d }$ balls. Then sending $R \to \infty$ , the second term converges to zero by uniform tightness of the weakly convergent sequence $\{ \mu _ { n } \} _ { n \in \mathbb { N } }$ . An application of the triangle inequality then shows that

$$
\int _ { \mathbb R } f ( t ) \nu _ { \mu _ { n } } ( d t ) = \int _ { \mathbb R ^ { d } } f \big ( \varphi ( \mu _ { n } , x ) \big ) \mu _ { n } ( d x )  \int _ { \mathbb R ^ { d } } f \big ( \varphi ( \mu , z ) \big ) \mu ( d z ) = \int _ { \mathbb R } f ( s ) \nu _ { \mu } ( d s )
$$

a $s n  \infty , s 0 \nu _ { \mu _ { n } }  \nu _ { \mu }$ in distribution. Thus, L is compact in the weak topology. The assertion that $( \mathsf { A } . 1 )$ holds then follows by exactly the same contradiction argument used in the proof of Example 3.4.

## A.3 ADDITIONAL MATERIAL FOR SECTION 4

By considering Gaussian convolutions, Theorem 4.4 can be generalized to compact sets K which do not necessarily satisfy the uniform level set condition; in particular, this generalization allows discrete measures to belong to K. To this end, given $\sigma > 0$ and $\bar { \mu } \in \mathcal { P } ( \mathbb { R } ^ { d } )$ , let $\mu _ { \sigma }$ denote the distribution of the random variable $X + \sigma Z .$ , where $X \sim \mu$ and $Z \sim \mathcal { N } ( 0 , I _ { d } )$ are independent. That is,

$$
\mu _ { \sigma } : = \operatorname { L a w } ( X + \sigma Z ) = \int _ { \mathbb { R } ^ { d } } \mathcal { N } ( x , \sigma ^ { 2 } I _ { d } ) \mu ( d x ) .
$$

For a compact set $K \subset \mathcal { P } _ { p } ( \mathbb { R } ^ { d } )$ , define

$$
K _ { \sigma } = \{ \mu _ { \sigma } \colon \mu \in K \} .
$$

We first prove that the mapping $K \mapsto K _ { \sigma }$ preserves $\mathsf { W } _ { p }$ -compactness.

Lemma A.4. If $K \subset \mathcal { P } _ { p } ( \mathbb { R } ^ { d } )$ is $\mathsf { W } _ { p }$ -compact and $\sigma > 0$ , then $K _ { \sigma }$ is $\mathsf { W } _ { p }$ -compact.

Proof. It suffices to show that the mapping $\mu \mapsto \mu _ { \sigma }$ is $\mathsf { W } _ { p }$ -continuous. In fact, we will show that it is 1-Lipschitz continuous uniformly in $\sigma .$ . Fix $\mu \in \mathcal P _ { p } ( \mathbb { R } ^ { d } )$ and $\nu \in \mathcal P _ { p } ( \mathbb { R } ^ { d } )$ . Let $( X , Y )$ be random variables such that $X \sim \mu , Y \sim \nu$ , and X and $Y$ are $\mathsf { W } _ { p }$ -optimally coupled. $\mathrm { ~ f ~ } \dot { Z } \sim \dot { \mathcal { N } } ( 0 , I _ { d } )$ , then $( X + \sigma Z , Y + \sigma Z )$ defines a coupling of $\mu _ { \sigma }$ and $\nu _ { \sigma }$ . Therefore,

$$
\begin{array} { r l } & { \mathsf { W } _ { p } ( \mu _ { \sigma } , \nu _ { \sigma } ) \leq \big ( \mathbb { E } \| ( X + \sigma Z ) - ( Y + \sigma Z ) \| ^ { p } \big ) ^ { 1 / p } } \\ & { \qquad = \big ( \mathbb { E } \| X - Y \| ^ { p } \big ) ^ { 1 / p } } \\ & { \qquad = \mathsf { W } _ { p } ( \mu , \nu ) . } \end{array}
$$

This proves the desired claim.

Moreover, $\mu _ { \sigma }$ uniformly approximates $\mu$ over $K$ with an explicit rate.

Lemma A.5. If $\operatorname { \mathrm { : } } K \subset { \mathcal { P } } _ { p } ( \mathbb { R } ^ { d } )$ is $\mathsf { W } _ { p }$ -compact and $\sigma > 0 _ { : }$ , then

$$
\operatorname* { s u p } _ { \mu \in K } \mathsf { W } _ { p } ( \mu _ { \sigma } , \mu ) \leq \big ( \mathbb { E } _ { Z \sim \mathcal { N } ( 0 , I _ { d } ) } \| Z \| ^ { p } \big ) ^ { 1 / p } \sigma .
$$

Proof. If $\mu \in K , X \sim \mu$ , and $Z \sim { \mathcal { N } } ( 0 , I _ { d } )$ , then $( X , X + \sigma Z )$ defines a coupling of $\mu$ and $\mu _ { \sigma }$ Therefore,

$$
\begin{array} { l } { \displaystyle \operatorname* { s u p } _ { \mu \in { \cal K } } { \sf W } _ { p } ( \mu , \mu _ { \sigma } ) \leq \displaystyle \operatorname* { s u p } _ { \mu \in { \cal K } } \left( \mathbb { E } _ { X \sim \mu , { \cal Z } \sim \mathcal { N } ( 0 , { \cal I } _ { d } ) } \| X - ( X + \sigma { \cal Z } ) \| ^ { p } \right) ^ { 1 / p } } \\ { = \left( \mathbb { E } _ { { \cal Z } \sim \mathcal { N } ( 0 , { \cal I } _ { d } ) } \| { \cal Z } \| ^ { p } \right) ^ { 1 / p } \sigma . } \end{array}
$$

This completes the proof.

Notice that $K _ { \sigma } \subset \mathcal { P } _ { \mathrm { a c } } ( \mathbb { R } ^ { d } )$ for any $\sigma > 0$ , even if $K$ contains singular or atomic measures. Application of Theorem 4.4 to $K _ { \sigma }$ , together with the preceding lemmas, yields the following generalization to Theorem 4.4 that is valid for any $\mathsf { W } _ { p }$ -compact set $K ;$ ; in particular, the uniform level set condition (3.1) is not required.

Corollary A.6. Fix $p \in [ 1 , \infty )$ . Let $F \colon \mathcal { P } _ { p } ( \mathbb { R } ^ { d } ) \to \mathcal { P } _ { p } ( \mathbb { R } ^ { d ^ { \prime } } )$ be $\mathsf { W } _ { p }$ -continuous. Let $K \subset \mathcal { P } _ { p } ( \mathbb { R } ^ { d } )$ be $\mathsf { W } _ { p } { - } c o m p a c t .$ . For any $\varepsilon > 0 ,$ , there exists $\sigma > 0$ and a transformer T oftheform (4.2) such that

$$
\operatorname* { s u p } _ { \mu \in K } \mathsf { W } _ { p } \big ( \mathsf { T } ( \mu _ { \sigma } , \cdot ) _ { \# } \mu _ { \sigma } , F ( \mu ) \big ) < \varepsilon .
$$

Proof. Let $\varepsilon > 0$ . Fix $\sigma > 0$ to be determined. By Lemma A.4, $K _ { \sigma }$ is a $\mathsf { W } _ { p }$ -compact subset of $\mathcal { P } _ { \mathrm { a c } } ( \mathbb { R } ^ { d } )$ . Thus, by Example $3 . 4 , K _ { \sigma }$ satisfies the uniform level set condition (3.1) from Assumption 3.1. Consequently, by Theorem 4.4, there exists a deep transformer in-context map $\mathsf { T } = \mathsf { T } _ { \sigma , \varepsilon } \colon \mathcal { P } ( \mathbb { R } ^ { d } ) \times \mathbb { R } ^ { d } \to \mathbb { R } ^ { d ^ { \prime } }$ of the form (4.2) such that

$$
\operatorname* { s u p } _ { \nu \in K _ { \sigma } } \mathsf { W } _ { p } \big ( \mathsf { T } ( \nu , \cdot ) _ { \# } \nu , F ( \nu ) \big ) < \frac { \varepsilon } { 2 } .
$$

By Lemma A.5, the continuity of $F .$ , and a result on uniform continuity near compact sets (Tsimpos et al., 2026, Lemma C.2), we may choose $\sigma = \sigma ( \varepsilon )$ such that

$$
\operatorname * { s u p } _ { \mu \in { \cal K } } { \ W _ { p } \big ( { \cal F } ( \mu _ { \sigma } ) , { \cal F } ( \mu ) \big ) } < { \frac { \varepsilon } { 2 } } .
$$

Application of the triangle inequality to obtain

$$
\operatorname* { s u p } _ { \mu \in K } \mathsf { W } _ { p } \big ( \mathsf { T } ( \mu _ { \sigma } , \cdot ) _ { \# } \mu _ { \sigma } , F ( \mu ) \big ) \leq \operatorname* { s u p } _ { \nu \in K _ { \sigma } } \mathsf { W } _ { p } \big ( \mathsf { T } ( \nu , \cdot ) _ { \# } \nu , F ( \nu ) \big ) + \operatorname* { s u p } _ { \mu \in K } \mathsf { W } _ { p } \big ( F ( \mu _ { \sigma } ) , F ( \mu ) \big ) < \varepsilon
$$

completes the proof.

We conclude with the following observation. The proof of Theorem 4.4—in particular the analysis leading to (C.13)—implies an independently interesting result on the universality of deep transformer in-context maps over $\mathsf { W } _ { p }$ -compact sets $K \doteq \mathcal { P } _ { p } ( \mathbb { R } ^ { d } )$ of probability measures supported on the whole of $\mathbb { R } ^ { d }$ instead of on a compact token domain. We state this result as a proposition.

Proposition A.7. Fix $p \in [ 1 , \infty )$ . Let $\Omega \subset \mathbb { R } ^ { d }$ be compact and $K \subset \mathcal { P } _ { p } ( \mathbb { R } ^ { d } )$ be $\mathsf { W } _ { p } { - } c o m p a c t .$ Then for any $G \in \mathcal { C } ( K \times \Omega , \mathbb { R } ^ { d ^ { \prime } } )$ and any $\varepsilon > 0 _ { : }$ , there exists a deep transformer in-context map $\mathsf { T } \colon \mathcal { P } ( \mathbb { R } ^ { d } ) \times \mathbb { R } ^ { d } \to \mathbb { R } ^ { d ^ { \prime } }$ oftheform (4.2) such tha

$$
\operatorname* { s u p } _ { ( \mu , x ) \in K \times \Omega } \left| \mathsf { T } ( \mu , x ) - G ( \mu , x ) \right| < \varepsilon .
$$

## B PROOFS OF RESULTS IN SECTION 3

This appendix provides the remaining proofs of results from Section 3 in the main text.

## B.1 PROOF OF EXAMPLE 3.3

Proof. Since $\varphi ( x ) = x ,$ , we have

$$
\mu \big ( \{ x \in \mathbb { R } \colon | \varphi ( x ) - t | \leq r \} \big ) = \mu \big ( [ t - r , t + r ] \big ) .
$$

Thus, it is enough to prove that

$$
\operatorname* { l i m } _ { r \downarrow 0 } \operatorname* { s u p } _ { \mu \in K } \operatorname* { s u p } _ { t \in \mathbb { R } } \mu \big ( [ t - r , t + r ] \big ) = 0 .
$$

Suppose by contradiction that this fails. Then there exist

$$
\varepsilon > 0 , \qquad r _ { n } \downarrow 0 , \qquad \mu _ { n } \in K , \qquad t _ { n } \in \mathbb { R }
$$

such that

$$
\mu _ { n } { \big ( } [ t _ { n } - r _ { n } , t _ { n } + r _ { n } ] { \big ) } \geq \varepsilon \quad { \mathrm { f o r ~ a l l } } \quad n \geq 1 .
$$

Since K is compact, it is tight. Hence there exists $R > 0$ such that

$$
\mu { \big ( } [ - R , R ] { \big ) } > 1 - { \frac { \varepsilon } { 2 } } \quad { \mathrm { f o r ~ a l l } } \quad \mu \in K .
$$

We claim that $( t _ { n } ) _ { n \in \mathbb { N } }$ is bounded. Indeed, if $( t _ { n } ) _ { n \in \mathbb { N } }$ were unbounded, then, after passing to a subsequence,

$$
| t _ { n } | \to \infty .
$$

Since $r _ { n } \to 0$ , for all sufficiently large $n ,$ it holds that

$$
[ t _ { n } - r _ { n } , t _ { n } + r _ { n } ] \subset \mathbb { R } \setminus [ - R , R ] .
$$

Therefore

$$
\mu _ { n } \big ( [ t _ { n } - r _ { n } , t _ { n } + r _ { n } ] \big ) \leq \mu _ { n } \big ( \mathbb { R } \setminus [ - R , R ] \big ) < \frac { \varepsilon } { 2 } ,
$$

which contradicts the choice of $\mu _ { n }$ and $t _ { n } .$ Thus $( t _ { n } ) _ { n \in \mathbb { N } }$ is bounded. Passing to a subsequence, we may assume that

$$
t _ { n } \to t _ { * }
$$

for some $t _ { * } \in \mathbb { R }$ . Since $K$ is compact, after passing to a further subsequence,

$$
\mu _ { n } \to \mu _ { * }
$$

for some $\mu _ { * } \in K$ . Since $\mu _ { * }$ is non-atomic,

$$
\mu _ { * } ( \{ t _ { * } \} ) = 0 .
$$

Therefore, by continuity from above,

$$
\mu _ { * } \big ( [ t _ { * } - \delta , t _ { * } + \delta ] \big )  \mu _ { * } \big ( \{ t _ { * } \} \big ) = 0 \quad \mathrm { a s } \quad \delta \downarrow 0 .
$$

Hence we may choose $\delta > 0$ such that

$$
\mu _ { * } \big ( [ t _ { * } - \delta , t _ { * } + \delta ] \big ) < \frac { \varepsilon } { 2 } .
$$

For all sufficiently large $n ,$ it holds that

$$
[ t _ { n } - r _ { n } , t _ { n } + r _ { n } ] \subset [ t _ { * } - \delta , t _ { * } + \delta ] .
$$

Consequently,

$$
\begin{array} { r } { \varepsilon \leq \mu _ { n } \big ( [ t _ { * } - \delta , t _ { * } + \delta ] \big ) . } \end{array}
$$

Since $[ t _ { * } - \delta , t _ { * } + \delta ]$ is closed, the Portmanteau theorem gives

$$
\operatorname* { l i m } _ { n \to \infty } \operatorname* { s u p } _ { \mu _ { n } } \bigl ( \bigl [ t _ { * } - \delta , t _ { * } + \delta \bigr ] \bigr ) \leq \mu _ { * } \bigl ( \bigl [ t _ { * } - \delta , t _ { * } + \delta \bigr ] \bigr ) .
$$

It follows that

$$
\varepsilon \le \mu _ { * } \big ( [ t _ { * } - \delta , t _ { * } + \delta ] \big ) < \frac { \varepsilon } { 2 } ,
$$

a contradiction. Therefore,

$$
\operatorname* { l i m } _ { r \downarrow 0 } \operatorname* { s u p } _ { \mu \in K } \operatorname* { s u p } _ { t \in \mathbb { R } } \mu \big ( [ t - r , t + r ] \big ) = 0 ,
$$

which proves the assertion.

## B.2 PROOF OF EXAMPLE 3.4

Proof. Since $\rho$ is non-atomic, by Lemma B.1 there exists $v \in \mathbb { S } ^ { d - 1 }$ such that

$$
x \mapsto \varphi ( x ) : = x \cdot v
$$

satisfies

$$
\varphi _ { \# } \rho \in { \mathcal { P } } _ { \mathrm { n a } } ( \mathbb { R } ) .
$$

Fix $\mu \in K$ . Since $\mu \ll \rho ,$ , for every Borel set $B \subset \mathbb { R } .$ , it holds that

$$
( \varphi _ { \# } \rho ) ( B ) = 0 \quad { \mathrm { i m p l i e s } } \quad \rho { \big ( } \varphi ^ { - 1 } ( B ) { \big ) } = 0 .
$$

Hence

$$
\mu \big ( \varphi ^ { - 1 } ( B ) \big ) = 0 ,
$$

and therefore

$$
( \varphi _ { \# } \mu ) ( B ) = 0 .
$$

Thus

$$
\varphi _ { \# } \mu \ll \varphi _ { \# } \rho .
$$

Since $\varphi _ { \# } \rho$ is non-atomic, $\varphi _ { \# } \mu$ is also non-atomic. Define

$$
L : = \{ \varphi _ { \# } \mu \colon \mu \in K \} \subset { \mathcal { P } } _ { p } ( \mathbb { R } ) .
$$

Since $\varphi$ is Lipschitz, the map

$$
\mu \mapsto \varphi _ { \# } \mu
$$

is continuous. Therefore $L$ is compact in ${ \mathcal { P } } _ { p } ( \mathbb { R } )$ . Moreover,

$$
L \subset { \mathcal { P } } _ { \mathrm { n a } } ( \mathbb { R } ) .
$$

We claim that

$$
\operatorname* { l i m } _ { r \downarrow 0 } \operatorname* { s u p } _ { \nu \in L } \operatorname* { s u p } _ { t \in \mathbb { R } } \nu \big ( [ t - r , t + r ] \big ) = 0 .
$$

Suppose by contradiction that this fails. Then there exist $\varepsilon > 0 , r _ { n } \downarrow 0 , \nu _ { n } \in L .$ , and $t _ { n } \in \mathbb { R }$ such that

$$
\nu _ { n } \big ( [ t _ { n } - r _ { n } , t _ { n } + r _ { n } ] \big ) \geq \varepsilon \quad \mathrm { f o r ~ a l l } \quad n \geq 1 .
$$

Since $L$ is compact, it is tight. Hence there exists $R > 0$ such that

$$
\nu \big ( [ - R , R ] \big ) > 1 - \frac { \varepsilon } { 2 } \quad \mathrm { f o r \ a l l } \quad \nu \in L .
$$

Consequently, $( t _ { n } ) _ { n \in \mathbb { N } }$ must be bounded; otherwise, for infinitely many $n ,$ it would hold that

$$
[ t _ { n } - r _ { n } , t _ { n } + r _ { n } ] \subset \mathbb { R } \setminus [ - R , R ] ,
$$

which would imply

$$
\nu _ { n } \big ( [ t _ { n } - r _ { n } , t _ { n } + r _ { n } ] \big ) \leq \frac { \varepsilon } { 2 } .
$$

This is a contradiction. Now passing to a subsequence,

$$
t _ { n } \to t _ { * }
$$

for some $t _ { * } \in \mathbb { R }$ . Since L is compact, after passing to a further subsequence,

$$
\nu _ { n }  \nu _ { * }
$$

for some $\nu _ { * } \in L$ . Because $\nu _ { * }$ is non-atomic,

$$
\nu _ { * } \bigl ( \{ t _ { * } \} \bigr ) = 0 .
$$

Hence there exists $\delta > 0$ such that

$$
\nu _ { * } \big ( [ t _ { * } - \delta , t _ { * } + \delta ] \big ) < \frac { \varepsilon } { 2 } .
$$

For sufficiently large $n _ { \mathrm { { ; } } }$

$$
[ t _ { n } - r _ { n } , t _ { n } + r _ { n } ] \subset [ t _ { * } - \delta , t _ { * } + \delta ] .
$$

Therefore

$$
\varepsilon \leq \nu _ { n } \big ( [ t _ { * } - \delta , t _ { * } + \delta ] \big ) .
$$

Since $[ t _ { * } - \delta , t _ { * } + \delta ]$ is closed, the Portmanteau theorem yields

$$
\operatorname* { l i m } _ { n \to \infty } \nu _ { n } \big ( [ t _ { * } - \delta , t _ { * } + \delta ] \big ) \leq \nu _ { * } \big ( [ t _ { * } - \delta , t _ { * } + \delta ] \big ) .
$$

Hence

$$
\varepsilon \leq \nu _ { * } \big ( [ t _ { * } - \delta , t _ { * } + \delta ] \big ) ,
$$

contradicting the choice of $\delta .$ . Thus

$$
\operatorname* { l i m } _ { r \downarrow 0 } \operatorname* { s u p } _ { \mu \in K } \operatorname* { s u p } _ { t \in \mathbb { R } } \mu \big ( \{ x \in \mathbb { R } ^ { d } \colon | \varphi ( x ) - t | \leq r \} \big ) = \operatorname* { l i m } _ { r \downarrow 0 } \operatorname* { s u p } _ { \nu \in L } \operatorname* { s u p } _ { t \in \mathbb { R } } \nu \big ( [ t - r , t + r ] \big ) = 0
$$

as asserted.

In the preceding proof, we required the following lemma concerning the non-atomicity of almost every one-dimensional projection.

Lemma B.1. Let $\rho \in \mathcal { P } _ { \operatorname* { n a } } ( \mathbb { R } ^ { d } )$ . Then for surface-almost every $v \in \mathbb { S } ^ { d - 1 }$ , it holds that

$$
\begin{array} { r } { ( \pi _ { v } ) _ { \# } \rho \in \mathcal { P } _ { \mathrm { n a } } ( \mathbb { R } ) , \quad \mathit { w h e r e } \quad \pi _ { v } ( x ) : = x \cdot v . } \end{array}
$$

Proof. The case $d = 1$ is immediate because ${ \mathbb S } ^ { 0 } = \{ - 1 , 1 \}$ and $\pi _ { v } ( x ) = \pm x$ . We therefore assume that $d \geq 2$ . Let σ denote the surface measure on ${ \mathbb S } ^ { d - 1 }$ . For $v \in \mathbb { S } ^ { d - 1 }$ , set

$$
\nu _ { v } : = ( \pi _ { v } ) _ { \# } \rho .
$$

Let

$$
\Delta : = \{ ( s , s ) \colon s \in \mathbb { R } \} \subset \mathbb { R } ^ { 2 } .
$$

By the definition of pushforward measures,

$$
( \nu _ { v } \otimes \nu _ { v } ) ( \Delta ) = ( \rho \otimes \rho ) \big ( \{ ( x , y ) \in \mathbb { R } ^ { d } \times \mathbb { R } ^ { d } \colon v \cdot ( x - y ) = 0 \} \big ) .
$$

Integrating with respect to v and applying Tonelli’s theorem gives

$$
\int _ { \mathbb { S } ^ { d - 1 } } ( \nu _ { v } \otimes \nu _ { v } ) ( \Delta ) \sigma ( d v ) = \int _ { \mathbb { R } ^ { d } \times \mathbb { R } ^ { d } } \sigma \big ( \{ v \in \mathbb { S } ^ { d - 1 } \colon v \cdot ( x - y ) = 0 \} \big ) ( \rho \otimes \rho ) ( d x , d y ) .
$$

If x $\neq y .$ , then

$$
\{ v \in \mathbb { S } ^ { d - 1 } \colon v \cdot ( x - y ) = 0 \}
$$

is a $( d - 2 )$ -dimensional great subsphere of $\mathbb { S } ^ { d - 1 }$ and therefore has zero surface measure. I $: x = y$ the above set is the whole sphere. Hence

$$
\int _ { \mathbb S ^ { d - 1 } } ( \nu _ { v } \otimes \nu _ { v } ) ( \Delta ) \sigma ( d v ) = \sigma ( \mathbb S ^ { d - 1 } ) ( \rho \otimes \rho ) \big ( \{ ( x , x ) \colon x \in \mathbb R ^ { d } \} \big ) .
$$

Since $\rho$ is non-atomic,

$$
( \rho \otimes \rho ) { \big ( } \{ ( x , x ) \colon x \in \mathbb { R } ^ { d } \} { \big ) } = \int _ { \mathbb { R } ^ { d } } \rho ( \{ x \} ) \rho ( d x ) = 0 .
$$

Therefore,

$$
\int _ { \mathbb { S } ^ { d - 1 } } ( \nu _ { v } \otimes \nu _ { v } ) ( \Delta ) \sigma ( d v ) = 0 .
$$

Since the integrand is nonnegative,

$$
( \nu _ { v } \otimes \nu _ { v } ) ( \Delta ) = 0
$$

for $\sigma$ almost every $v \in \mathbb { S } ^ { d - 1 }$ . Last, for any Borel probability measure ν on R, it holds that

$$
( \nu \otimes \nu ) ( \Delta ) = \int _ { \mathbb { R } } \nu ( \{ t \} ) \nu ( d t ) .
$$

It follows that $( \nu \otimes \nu ) ( \Delta ) = 0$ if and only if ν is non-atomic because if ν has an atom $t _ { 0 }$ of mass $w > 0$ , then

$$
( \nu \otimes \nu ) ( \Delta ) \geq ( \nu \otimes \nu ) { \big ( } \{ ( t _ { 0 } , t _ { 0 } ) \} { \big ) } = ( \nu \otimes \nu ) { \big ( } \{ t _ { 0 } \} \times \{ t _ { 0 } \} { \big ) } = \nu ( \{ t _ { 0 } \} ) ^ { 2 } = w ^ { 2 } > 0 .
$$

Consequently,

$$
\nu _ { v } = ( \pi _ { v } ) _ { \# } \rho \in \mathcal { P } _ { \mathrm { n a } } ( \mathbb { R } )
$$

for σ almost every $v \in \mathbb { S } ^ { d - 1 }$

## B.3 PROOF OF EXAMPLE 3.5

Proof. Define

$$
E : = \Big \{ \big ( \gamma , \gamma ( s ) \big ) \colon \gamma \in K ^ { \prime } , \ s \in [ 0 , 1 ] \Big \} \subset K ^ { \prime } \times Q .
$$

Consider the map

$$
\Psi \colon K ^ { \prime } \times [ 0 , 1 ] \to E , \qquad \Psi ( \gamma , s ) : = { \bigl ( } \gamma , \gamma ( s ) { \bigr ) } .
$$

The map $\Psi$ is continuous. Indeed, convergence in $\mathcal { C } ^ { 1 } ( [ 0 , 1 ] , Q )$ implies uniform convergence, so the evaluation map $( \gamma , s ) \mapsto \gamma ( s )$ is continuous. Moreover, Ψ is injective. In fact, if

$$
( \gamma , \gamma ( s ) ) = ( \widetilde { \gamma } , \widetilde { \gamma } ( t ) ) ,
$$

then $\gamma = \widetilde { \gamma } ,$ and hence $\gamma ( s ) = \gamma ( t )$ . Since γ is injective, we obtain $s = t$ . Since $K ^ { \prime } \times [ 0 , 1 ]$ is compact and $K ^ { \prime } \times Q$ is Hausdorff, Ψ is a homeomorphism from $K ^ { \prime } \times [ 0 , 1 ]$ onto $E$ . In particular, E is compact. Define

$$
f \colon E  [ 0 , 1 ] , \quad { \mathrm { w h e r e } } \quad f ( \gamma , \gamma ( s ) ) : = s .
$$

Equivalently,

$$
f = \pi _ { 2 } \circ \Psi ^ { - 1 } ,
$$

where $\pi _ { 2 } ( \gamma , s ) = s$ . Hence f is continuous.

Since $K ^ { \prime } \times Q$ is a compact metric space, it is normal. By the Tietze extension theorem, there exists

$$
\widetilde { \varphi } \in \mathcal { C } ( K ^ { \prime } \times Q , [ 0 , 1 ] )
$$

such that

$$
{ \widetilde { \varphi } } | _ { E } = f .
$$

Therefore,

$$
\widetilde { \varphi } \left( \gamma , \gamma ( s ) \right) = s \quad \mathrm { f o r \ a l l } \quad \gamma \in K ^ { \prime } \quad \mathrm { a n d } \quad s \in [ 0 , 1 ] .
$$

Since $K ^ { \prime }$ is compact in $\mathcal { C } ^ { 1 } ( [ 0 , 1 ] , Q )$ , there exists $C < \infty$ such that

$$
| \dot { \gamma } ( s ) | \leq C \qquad \mathrm { ~ f o r ~ a l l ~ } \gamma \in K ^ { \prime } \quad \mathrm { a n d } \quad s \in [ 0 , 1 ] .
$$

On the other hand, by assumption,

$$
\int _ { 0 } ^ { 1 } | { \dot { \gamma } } ( s ) | d s \geq c \quad { \mathrm { f o r ~ a l l } } \quad \gamma \in K ^ { \prime } .
$$

Fix $\gamma \in K ^ { \prime }$ and $t \in \mathbb { R }$ . By the definition of $\sigma _ { \gamma }$ and the identity $\widetilde { \varphi } ( \gamma , \gamma ( s ) ) = s$ , we have

$$
\sigma _ { \gamma } \{ x \in Q \colon | \widetilde { \varphi } ( \gamma , x ) - t | \leq r \} \} = \frac { \displaystyle \int _ { 0 } ^ { 1 } \mathbf { 1 } _ { \{ | s - t | \leq r \} } | \dot { \gamma } ( s ) | d s } { \displaystyle \int _ { 0 } ^ { 1 } | \dot { \gamma } ( s ) | d s } .
$$

Hence

$$
\sigma _ { \gamma } \big ( \{ x \in Q \colon | \widetilde { \varphi } ( \gamma , x ) - t | \le r \} \big ) \le \frac { C } { c } \mathcal { L } ^ { 1 } \big ( [ 0 , 1 ] \cap [ t - r , t + r ] \big ) \le \frac { 2 C } { c } r .
$$

The estimate is uniform in $\gamma \in K ^ { \prime }$ and $t \in \mathbb { R } .$ Therefore,

$$
\operatorname* { l i m } _ { r \downarrow 0 } \operatorname* { s u p } _ { \gamma \in K ^ { \prime } } \operatorname* { s u p } _ { t \in \mathbb { R } } \sigma _ { \gamma } \big ( \{ x \in Q \colon | \widetilde { \varphi } ( \gamma , x ) - t | \leq r \} \big ) = 0 .
$$

It remains to relate this parametrized statement to the uniform level set condition on a family of measures. First, the map

$$
K ^ { \prime } \ni \gamma \mapsto \sigma _ { \gamma } \in { \mathcal { P } } _ { p } ( Q )
$$

is continuous. Indeed, if $\gamma _ { n }  \gamma \mathrm { i n } { \mathcal { C } } ^ { 1 }$ , then

$$
\gamma _ { n }  \gamma \quad \mathrm { a n d } \quad | \dot { \gamma } _ { n } |  | \dot { \gamma } |
$$

uniformly on $[ 0 , 1 ]$ , and hence, for every $h \in { \mathcal { C } } ( Q )$ , it holds that

$$
\int _ { Q } h d \sigma _ { \gamma _ { n } } = \frac { \displaystyle \int _ { 0 } ^ { 1 } h ( \gamma _ { n } ( s ) ) | \dot { \gamma } _ { n } ( s ) | d s } { \displaystyle \int _ { 0 } ^ { 1 } | \dot { \gamma } _ { n } ( s ) | d s } \to \frac { \displaystyle \int _ { 0 } ^ { 1 } h ( \gamma ( s ) ) | \dot { \gamma } ( s ) | d s } { \displaystyle \int _ { 0 } ^ { 1 } | \dot { \gamma } ( s ) | d s } = \int _ { Q } h d \sigma _ { \gamma } .
$$

Thus, $\sigma _ { \gamma _ { n } }  \sigma _ { \gamma }$ weakly. Since $Q$ is compact, this is equivalent to convergence in $\mathsf { W } _ { p } .$ . Consequently,

$$
K : = \{ \sigma _ { \gamma } : \gamma \in K ^ { \prime } \}
$$

is compact in $\mathcal { P } _ { p } ( Q )$

Assume now that distinct curves in $K ^ { \prime }$ have distinct images, i.e.,

$$
\gamma _ { 1 } ( [ 0 , 1 ] ) = \gamma _ { 2 } ( [ 0 , 1 ] ) \quad \mathrm { i m p l i e s } \quad \gamma _ { 1 } = \gamma _ { 2 } .
$$

Then, $\gamma \mapsto \sigma _ { \gamma }$ is injective. Since it is a continuous bijection from the compact space $K ^ { \prime }$ onto the Hausdorff space $K$ , its inverse

$$
K \ni \sigma _ { \gamma } \mapsto \gamma \in K ^ { \prime }
$$

is continuous. We may therefore define

$$
\varphi \colon K \times Q  [ 0 , 1 ] , \quad \mathrm { w h e r e } \quad \varphi ( \sigma _ { \gamma } , x ) : = \widetilde { \varphi } ( \gamma , x ) .
$$

This map is continuous, and the preceding estimate gives

$$
\operatorname* { s u p } _ { \mu \in K } \operatorname* { s u p } _ { t \in \mathbb { R } } \mu \big ( \{ x \in Q \colon | \varphi ( \mu , x ) - t | \leq r \} \big ) \leq \frac { 2 C } { c } r .
$$

Hence, K satisfies the uniform level set condition (3.1).

## B.4 PROOF OF THEOREM 3.6

Proof. The proof proceeds in six steps.

Step 1 (Truncate the output space). For $R > 0$ , define the truncation map

$$
\Pi _ { R } \colon \mathbb { R } ^ { d ^ { \prime } }  \overline { { B _ { R } ( 0 ) } } \subset \mathbb { R } ^ { d ^ { \prime } } \quad \mathfrak { b } \mathbf { y } \quad \Pi _ { R } ( y ) : = \{ { y , \atop R \frac { y } { | y | } } , \quad | y | > R , 
$$

and set

$$
F _ { R } ( \mu ) : = ( \Pi _ { R } ) _ { \# } F ( \mu ) \quad { \mathrm { f o r } } \quad \mu \in K .
$$

Then a direct calculation shows that

$$
\mathsf { W } _ { p } ^ { p } \big ( F ( \mu ) , F _ { R } ( \mu ) \big ) \le \int _ { \{ | y | > R \} } | y | ^ { p } F ( \mu ) ( d y ) .
$$

Hence

$$
\operatorname* { s u p } _ { \mu \in K } { \mathsf { W } } _ { p } \bigl ( F ( \mu ) , F _ { R } ( \mu ) \bigr ) \leq E _ { \mathrm { t a i l } , p } ( R ) ,
$$

where

$$
E _ { \mathrm { t a i l } , p } ( R ) : = \left( \operatorname* { s u p } _ { \nu \in F ( K ) } \int _ { \{ | y | > R \} } | y | ^ { p } \nu ( d y ) \right) ^ { 1 / p } .
$$

Since $F ( K )$ is compact in $( \mathcal { P } _ { p } ( \mathbb { R } ^ { d ^ { \prime } } ) , \mathsf { W } _ { p } )$ by the continuity of F, Lemma B.2 shows that

$$
\begin{array} { r } { E _ { \mathrm { t a i l } , p } ( R ) \to 0 \quad \mathrm { a s } \quad R \to \infty . } \end{array}
$$

Step 2 (Cover the truncated output space). Fix $h > 0$ . Since $\overline { { B _ { R } ( 0 ) } } \subset \mathbb { R } ^ { d ^ { \prime } }$ is compact, we can choose an h-net

$$
\{ y _ { 1 } , \dotsc , y _ { N } \} \subset { \overline { { B _ { R } ( 0 ) } } }
$$

and a continuous partition of unity $\{ \psi _ { i } \} _ { i = 1 } ^ { N } \subset \mathcal { C } ( \mathbb { R } ^ { d ^ { \prime } } )$ such that

$$
0 \leq \psi _ { i } \leq 1 , \qquad \sum _ { i = 1 } ^ { N } \psi _ { i } \equiv 1 \quad \mathrm { o n } \quad \overline { { B _ { R } ( 0 ) } } ,
$$

and

$$
\mathrm { s u p p } ( \psi _ { i } ) \cap \overline { { B _ { R } ( 0 ) } } \subset B _ { h } ( y _ { i } )
$$

for each i. Moreover, the covering may be chosen so that

$$
N = N ( R , h ) \leq C _ { d ^ { \prime } } \left( 1 + \frac { R } { h } \right) ^ { d ^ { \prime } }
$$

by a standard volumetric bound. Next, define

$$
\alpha _ { i } ( \mu ) : = \int _ { \mathbb { R } ^ { d ^ { \prime } } } \psi _ { i } ( \boldsymbol { y } ) F _ { R } ( \mu ) ( d \boldsymbol { y } ) \quad \mathrm { a n d } \quad F _ { R , h } ( \mu ) : = \sum _ { i = 1 } ^ { N } \alpha _ { i } ( \mu ) \delta _ { \boldsymbol { y } _ { i } } .
$$

Since $F _ { R }$ is continuous by the 1-Lipschitz property of $\Pi _ { R }$ and each $\psi _ { i }$ is bounded and continuous, $\alpha _ { i } \colon K \to [ 0 , 1 ]$ is continuous. For each $\mu \in K$ , define a probability measure $\pi _ { \mu }$ on $\mathbb { R } ^ { d ^ { \prime } } \times \mathbb { R } ^ { d ^ { \prime } }$ by

$$
\pi _ { \mu } : = \sum _ { i = 1 } ^ { N } \bigl ( \mathrm { i d } , y _ { i } \bigr ) _ { \# } \bigl ( \psi _ { i } F _ { R } ( \mu ) \bigr ) = \sum _ { i = 1 } ^ { N } \bigl ( \psi _ { i } F _ { R } ( \mu ) \bigr ) \otimes \delta _ { y _ { i } } ,
$$

where ψ $ , _ { i } F _ { R } ( \mu )$ denotes the (sub-probability) measure with density $\psi _ { i }$ with respect to $F _ { R } ( \mu )$ . Because $F _ { R } ( \mu )$ is supported on $\overline { { B _ { R } ( 0 ) } }$ and $\begin{array} { r } { \sum _ { i = 1 } ^ { N } \psi _ { i } = 1 \mathrm { o n } \overline { { B _ { R } ( 0 ) } } } \end{array}$ , it holds that $\pi _ { \mu }$ is a coupling of $F _ { R } ( \mu )$ and $F _ { R , h } ( \mu )$ . Moreover, for each $i ,$ the support condition implies that if $\psi _ { i } ( \dot { y } ) > 0$ , then $| y - y _ { i } | < h$ for $F _ { R } ( \mu )$ -almost every $y .$ Hence

$$
\begin{array} { r l } {  { \operatorname { W } _ { p } ^ { p } \big ( F _ { R } ( \mu ) , F _ { R , h } ( \mu ) \big ) \le \int _ { \mathord { \mathbb { R } } ^ { d ^ { \prime } } \times \mathord { \mathbb { R } } ^ { d ^ { \prime } } } | y - z | ^ { p } \pi _ { \mu } ( d y , d z ) } } \\ & { = \displaystyle \sum _ { i = 1 } ^ { N } \int _ { \mathord { \mathbb { R } } ^ { d ^ { \prime } } } | y - y _ { i } | ^ { p } \psi _ { i } ( y ) F _ { R } ( \mu ) ( d y ) } \\ & { \le h ^ { p } \sum _ { i = 1 } ^ { N } \int _ { \mathord { \mathbb { R } } ^ { d ^ { \prime } } } \psi _ { i } ( y ) F _ { R } ( \mu ) ( d y ) = h ^ { p } . } \end{array}
$$

Therefore,

$$
\operatorname* { s u p } _ { \mu \in K } { \mathsf { W } } _ { p } \big ( F _ { R } ( \mu ) , F _ { R , h } ( \mu ) \big ) \leq h \quad \mathrm { f o r ~ a l l } \quad R > 0 .
$$

Step 3 (Transport to and from the uniform distribution). Since $K \subset \mathcal { P } _ { p } ( \mathbb { R } ^ { d } )$ satisfies the uniform level set condition (3.1) from Assumption 3.1 by hypothesis, the construction in Lemma B.3 delivers the existence of $H _ { 0 } \in \mathcal { C } ( K \times \mathbb { R } ^ { d } , [ 0 , 1 ] )$ such that

$$
H _ { 0 } ( \mu , \cdot ) _ { \# } \mu = \mathrm { U n i f } [ 0 , 1 ] \quad \mathrm { f o r ~ a l l } \quad \mu \in K .\tag{B.1}
$$

Next, define the cumulative weights

$$
s _ { 0 } ( \mu ) : = 0 , \quad \mathrm { a n d } \quad s _ { i } ( \mu ) : = \sum _ { \ell = 1 } ^ { i } \alpha _ { \ell } ( \mu ) \quad \mathrm { f o r ~ a l l } \quad i = 1 , \dots , N .
$$

Then $s _ { N } ( \mu ) = 1$ . Also define the discontinuous map

$$
( \mu , u ) \mapsto Q _ { 0 } ( \mu , u ) : = \sum _ { i = 1 } ^ { N } \mathbf { 1 } _ { ( s _ { i - 1 } ( \mu ) , s _ { i } ( \mu ) ] } ( u ) y _ { i } .
$$

If $U \sim \mathrm { U n i f } [ 0 , 1 ]$ , then a direct calculation shows that

$$
\operatorname { L a w } \big ( Q _ { 0 } ( \mu , U ) \big ) = Q _ { 0 } ( \mu , \cdot ) _ { \# } \operatorname { U n i f } [ 0 , 1 ] = F _ { R , h } ( \mu ) .
$$

In particular, (B.1) shows that

$$
F _ { R , h } ( \mu ) = Q _ { 0 } \big ( \mu , H _ { 0 } ( \mu , \cdot ) \big ) _ { \# } \mu .
$$

Step 4 (Regularize the discontinuous pushforward map). Let $\delta \in ( 0 , 1 )$ . Let $\kappa \in \mathcal { C } _ { c } ^ { \infty } ( ( - 1 , 1 ) )$ be nonnegative with $\begin{array} { r } { \int _ { \mathbb { R } } \kappa ( r ) d r = 1 } \end{array}$ . Set

$$
\kappa _ { \delta } ( r ) : = \delta ^ { - 1 } \kappa ( r / \delta ) .
$$

Extend $Q _ { 0 } ( \mu , \cdot )$ to R by setting

$$
\overline { { Q } } _ { 0 } ( \mu , u ) : = \left\{ \begin{array} { l l } { y _ { 1 } , } & { u \leq 0 , } \\ { Q _ { 0 } ( \mu , u ) , } & { 0 < u \leq 1 , } \\ { y _ { N } , } & { u > 1 . } \end{array} \right.
$$

Define

$$
Q _ { \delta } ( \mu , u ) : = \int _ { \mathbb { R } } \kappa _ { \delta } ( u - v ) \overline { { Q } } _ { 0 } ( \mu , v ) d v .
$$

Equivalently,

$$
Q _ { \delta } ( \mu , u ) = \sum _ { i = 1 } ^ { N } w _ { i } ^ { \delta } ( \mu , u ) y _ { i } ,
$$

where the nonnegative continuous weights $w _ { i } ^ { \delta }$ are obtained by integrating $\kappa _ { \delta } ( u - \cdot )$ over the intervals associated with $y _ { i } .$ . In particular,

$$
\sum _ { i = 1 } ^ { N } w _ { i } ^ { \delta } ( \mu , u ) = 1
$$

for all $\mu \in K$ and $u \in [ 0 , 1 ]$

Last, define our final approximation $G$ to be

$$
( \mu , x ) \mapsto G ( \mu , x ) : = G _ { R , \delta , h } ( \mu , x ) : = Q _ { \delta } \big ( \mu , H _ { 0 } ( \mu , x ) \big ) .
$$

Since $H _ { 0 }$ and $Q _ { \delta }$ are continuous, it holds that

$$
G _ { R , \delta , h } \in \mathcal { C } ( K \times \mathbb { R } ^ { d } , \mathbb { R } ^ { d ^ { \prime } } )
$$

is also continuous.

Step 5 (Bound the regularized pushforward error). We next compare $Q _ { \delta }$ with $Q _ { 0 }$ . Let

$$
T _ { \delta } ( \mu ) : = \bigcup _ { i = 1 } ^ { N - 1 } \left\{ u \in [ 0 , 1 ] \colon | u - s _ { i } ( \mu ) | < \delta \right\}
$$

for every $\mu \in K$ . Then

$$
| T _ { \delta } ( \mu ) | \le 2 N ( R , h ) \delta ,
$$

and $Q _ { \delta } ( \mu , \cdot ) = Q _ { 0 } ( \mu , \cdot )$ outside $T _ { \delta } ( \mu )$ , apart from a set of Lebesgue measure zero. Since both maps take values in $B _ { R } ( 0 )$ , it holds that

$$
| Q _ { \delta } ( \mu , u ) - Q _ { 0 } ( \mu , u ) | \leq 2 R
$$

for each $\mu$ and u. Coupling them by the same random variable $U \sim \mathrm { U n i f } [ 0 , 1 ]$ gives

$$
\begin{array} { r l } { \mathsf { W } _ { p } ^ { p } \big ( G _ { R , \delta , h } ( \mu , \cdot ) _ { \# } \mu , F _ { R , h } ( \mu ) \big ) = \mathsf { W } _ { p } ^ { p } \big ( Q _ { \delta } ( \mu , \cdot ) _ { \# } \mathrm { U n i f } [ 0 , 1 ] , Q _ { 0 } ( \mu , \cdot ) _ { \# } \mathrm { U n i f } [ 0 , 1 ] \big ) } & { } \\ & { = \mathsf { W } _ { p } ^ { p } \big ( \mathrm { L a w } ( Q _ { \delta } ( \mu , U ) ) , \mathrm { L a w } ( Q _ { 0 } ( \mu , U ) ) \big ) } \\ & { \leq \mathbb { E } | Q _ { \delta } ( \mu , U ) - Q _ { 0 } ( \mu , U ) | ^ { p } } \\ & { \leq ( 2 R ) ^ { p } | T _ { \delta } ( \mu ) | } \\ & { \leq ( 2 R ) ^ { p } 2 N ( R , h ) \delta . } \end{array}
$$

Therefore,

$$
\operatorname* { s u p } _ { \mu \in K } { \cal W } _ { p } \big ( G _ { R , \delta , h } ( \mu , \cdot ) _ { \# } \mu , F _ { R , h } ( \mu ) \big ) \leq 2 R \big ( 2 N ( R , h ) \delta \big ) ^ { 1 / p } .
$$

Step 6 (Combine the estimates). Combining Steps 1–5 and using the triangle inequality, we obtain

$$
\operatorname* { s u p } _ { \mu \in K } { \cal W } _ { p } \big ( G _ { R , \delta , h } ( \mu , \cdot ) _ { \# } \mu , F ( \mu ) \big ) \leq 2 R \big ( 2 N ( R , h ) \delta \big ) ^ { 1 / p } + h + E _ { \mathrm { t a i l } , p } ( R ) .\tag{B.2}
$$

Choose $R > 0$ sufficiently large such that

$$
E _ { \mathrm { t a i l } , p } ( R ) < \frac { \varepsilon } { 3 } .
$$

Next, choose $h > 0$ sufficiently small such that

$$
h < { \frac { \varepsilon } { 3 } } .
$$

With R and h fixed as in the preceding displays, choose $\delta \in ( 0 , 1 )$ sufficiently small such that

$$
2 R \big ( 2 N ( R , h ) \delta \big ) ^ { 1 / p } < \frac { \varepsilon } { 3 } .
$$

Then (B.2) yields

$$
\operatorname* { s u p } _ { \mu \in K } \mathsf { W } _ { p } \big ( G _ { R , \delta , h } ( \mu , \cdot ) _ { \# } \mu , F ( \mu ) \big ) < \varepsilon .
$$

This completes the proof.

The following result is a well-known property of compact sets in Wasserstein space that is used in the preceding argument; we provide a proof for the sake of completeness.

Lemma B.2 (Uniform integrability of $p \mathrm { - }$ th moments). Let $d \in \mathbb { N }$ and $1 \leq p < \infty$ . Let $K \subset \mathcal { P } _ { p } ( \mathbb { R } ^ { d } )$ be $\mathsf { W } _ { p } { - } c o m p a c t .$ Then

$$
\operatorname* { l i m } _ { R \to \infty } \operatorname* { s u p } _ { \nu \in K } \int _ { \{ | y | > R \} } | y | ^ { p } \nu ( d y ) = 0 .\tag{B.3}
$$

Proof. Suppose by contradiction that (B.3) fails. Then there exist $\delta > 0$ , a sequence $( \nu _ { n } ) _ { n \in \mathbb { N } } \subset K$ and a sequence $R _ { n } \to \infty$ such that

$$
\int _ { \{ | y | > R _ { n } \} } | y | ^ { p } \nu _ { n } ( d y ) \geq \delta \quad { \mathrm { f o r ~ a l l } } \quad n \in \mathbb { N } .
$$

Since K is compact with respect to $\mathsf { W } _ { p }$ , after passing to a subsequence, we may assume that

$$
\nu _ { n } \to \nu \quad \mathrm { i n } \quad \mathsf { W } _ { p }
$$

for some $\nu \in K$

Fix $\eta > 0$ . Since $\nu \in \mathcal P _ { p } ( \mathbb { R } ^ { d } )$ , there exists $M > 0$ such that

$$
\int _ { \{ | y | > M \} } | y | ^ { p } \nu ( d y ) < \eta .
$$

Choose a continuous function $\chi _ { M } \colon [ 0 , \infty ) \to [ 0 , 1 ]$ such that

$$
\chi _ { M } ( r ) = 0 \quad \mathrm { f o r ~ a l l } \quad r \leq M \quad \mathrm { a n d } \quad \chi _ { M } ( r ) = 1 \quad \mathrm { f o r ~ a l l } \quad r \geq M + 1 ,
$$

and define

$$
\varphi _ { M } ( y ) : = | y | ^ { p } \chi _ { M } ( | y | ) \quad { \mathrm { f o r ~ a l l } } \quad y \in \mathbb { R } ^ { d } .
$$

Then $\varphi _ { M }$ is continuous and satisfies

$$
0 \leq \varphi _ { M } ( y ) \leq | y | ^ { p } ,
$$

as well as

$$
\varphi _ { M } ( y ) = 0 \quad { \mathrm { i f } } \quad | y | \leq M \quad { \mathrm { a n d } } \quad \varphi _ { M } ( y ) = | y | ^ { p } \quad { \mathrm { i f } } \quad | y | \geq M + 1 .
$$

Since $\nu _ { n } \to \nu$ in $\mathsf { W } _ { p }$ and $\varphi _ { M }$ is continuous with at most p-th-order growth,

$$
\int _ { \mathbb { R } ^ { d } } \varphi _ { M } ( y ) \nu _ { n } ( d y ) \to \int _ { \mathbb { R } ^ { d } } \varphi _ { M } ( y ) \nu ( d y ) \quad \mathrm { a s } \quad n \to \infty .
$$

Moreover,

$$
\int _ { \mathbb { R } ^ { d } } \varphi _ { M } ( y ) \nu ( d y ) \leq \int _ { \{ | y | > M \} } | y | ^ { p } \nu ( d y ) < \eta .
$$

Hence, for all sufficiently large n, it holds that

$$
\int _ { \mathbb { R } ^ { d } } \varphi _ { M } ( y ) \nu _ { n } ( d y ) < 2 \eta .
$$

Since $R _ { n } \to \infty$ , we also have $R _ { n } \geq M + 1$ for all sufficiently large n. Therefore,

$$
\begin{array} { r l } & { \displaystyle \int _ { \{ | y | > R _ { n } \} } | y | ^ { p } \nu _ { n } ( d y ) \leq \int _ { \{ | y | > M + 1 \} } | y | ^ { p } \nu _ { n } ( d y ) } \\ & { \qquad \quad \leq \displaystyle \int _ { \mathbb { R } ^ { d } } \varphi _ { M } ( y ) \nu _ { n } ( d y ) } \\ & { \qquad \quad < 2 \eta . } \end{array}
$$

Choosing $\eta < \delta / 2$ gives a contradiction.

The core of the proof of Theorem 3.6 relies on transporting to the one-dimensional uniform distribution with a jointly continuous in-context map.

Lemma B.3 (Continuous exact uniformizer). Let $p \in [ 1 , \infty )$ . Let $K \subset \mathcal { P } _ { p } ( \mathbb { R } ^ { d } )$ satisfy Assumption 3.1. Then there exists

$$
H _ { 0 } \in \mathcal { C } ( K \times \mathbb { R } ^ { d } , [ 0 , 1 ] )
$$

such that

$$
H _ { 0 } ( \mu , \cdot ) _ { \# } \mu = \mathrm { U n i f } [ 0 , 1 ] \quad f o r a l l \quad \mu \in { \cal K } .
$$

Proof. Let $\varphi$ be the continuous scalarization from Assumption 3.1. Fix $r > 0$ . Define

$$
\omega _ { K } ( r ) : = \operatorname* { s u p } _ { \mu \in K } \operatorname* { s u p } _ { t \in \mathbb { R } } \mu \bigl ( \{ x \in \mathbb { R } ^ { d } \colon | \varphi ( \mu , x ) - t | \leq r \} \bigr ) .\tag{B.4}
$$

Let $\Theta \colon  { \mathbb { R } } \to [ 0 , 1 ]$ be continuous and nondecreasing such that

$$
\Theta ( s ) = 0 \quad { \mathrm { f o r ~ a l l } } \quad s \leq - 1 \quad { \mathrm { a n d } } \quad \Theta ( s ) = 1 \quad { \mathrm { f o r ~ a l l } } \quad s \geq 1 .
$$

Define

$$
H _ { r } ( \mu , x ) : = \int _ { \mathbb { R } ^ { d } } \Theta \left( \frac { \varphi ( \mu , x ) - \varphi ( \mu , z ) } { r } \right) \mu ( d z ) .
$$

Clearly $H _ { r }$ takes values in [0, 1].

We first show that

$$
H _ { r } \in \mathcal { C } ( K \times \mathbb { R } ^ { d } , [ 0 , 1 ] ) .
$$

To this end, let $( \mu _ { n } , x _ { n } ) \to ( \mu , x )$ in $K \times \mathbb { R } ^ { d }$ as $n \to \infty$ and set

$$
g _ { n } ( z ) : = \Theta \left( \frac { \varphi ( \mu _ { n } , x _ { n } ) - \varphi ( \mu _ { n } , z ) } { r } \right) \quad \mathrm { a n d } \quad g ( z ) : = \Theta \left( \frac { \varphi ( \mu , x ) - \varphi ( \mu , z ) } { r } \right) .
$$

Since $\varphi$ is continuous, $g _ { n }  g$ uniformly on every compact subset of $\mathbb { R } ^ { d }$ . Moreover,

$$
0 \leq g _ { n } \leq 1 \quad { \mathrm { a n d } } \quad 0 \leq g \leq 1
$$

and each are continuous. Since $\mu _ { n } \to \mu$ in $\mathsf { W } _ { p } .$ in particular $\mu _ { n }$ converges to $\mu$ in distribution, and the family $\{ \mu _ { n } : n \in \mathbb { N } \} \cup \{ \mu \}$ is uniformly tight. It follows that

$$
\int _ { \mathbb { R } ^ { d } } g _ { n } ( z ) \mu _ { n } ( d z ) \to \int _ { \mathbb { R } ^ { d } } g ( z ) \mu ( d z )
$$

as $n \to \infty$ by the same argument used in Remark A.3. Hence

$$
H _ { r } ( \mu _ { n } , x _ { n } ) \to H _ { r } ( \mu , x ) \quad { \mathrm { a s } } \quad n \to \infty ,
$$

so $H _ { r }$ is continuous.

Now fix $\mu \in K$ and define

$$
\nu _ { \mu } : = \varphi ( \mu , \cdot ) _ { \# } \mu \in \mathcal { P } ( \mathbb { R } ) .
$$

Assumption 3.1 implies that $\nu _ { \mu }$ is non-atomic. To see this, note that for every $t \in \mathbb { R }$ , it holds that

$$
\nu _ { \mu } ( \{ t \} ) = \mu { \bigl ( } \{ x \in \mathbb { R } ^ { d } \colon \varphi ( \mu , x ) = t \} { \bigr ) }
$$

and

$$
\nu _ { \mu } ( \{ t \} ) \leq \mu \big ( \{ x \in \mathbb { R } ^ { d } \colon | \varphi ( \mu , x ) - t | \leq \eta \} \big )
$$

for every $\eta > 0$ . Letting $\eta \downarrow 0$ and using the uniform level set condition gives

$$
\nu _ { \mu } ( \{ t \} ) = 0\tag{B.5}
$$

as claimed.

Continuing, we let

$$
F _ { \mu } ( t ) : = \nu _ { \mu } { \big ( } ( - \infty , t ] { \big ) } \quad { \mathrm { f o r } } \quad t \in \mathbb { R }
$$

be the cumulative distribution function of $\nu _ { \mu }$ . Define its smoothed version by

$$
F _ { \mu , r } ( t ) : = \int _ { \mathbb { R } } \Theta \left( \frac { t - s } { r } \right) \nu _ { \mu } ( d s ) .
$$

Then

$$
H _ { r } ( \mu , x ) = F _ { \mu , r } \big ( \varphi ( \mu , x ) \big ) ,
$$

and consequently

$$
H _ { r } ( \mu , \cdot ) _ { \# } \mu = ( F _ { \mu , r } ) _ { \# } \nu _ { \mu } .
$$

For every $t \in \mathbb { R } .$ , it holds that

$$
| F _ { \mu , r } ( t ) - F _ { \mu } ( t ) | \le \nu _ { \mu } ( [ t - r , t + r ] ) .
$$

Indeed, the functions

$$
s \mapsto \Theta \left( \frac { t - s } { r } \right) \quad \mathrm { a n d } \quad s \mapsto \mathbf { 1 } _ { ( - \infty , t ] } ( s )
$$

coincide outside the set $[ t - r , t + r ]$ . Hence,

$$
\operatorname* { s u p } _ { t \in \mathbb { R } } | F _ { \mu , r } ( t ) - F _ { \mu } ( t ) | \leq \omega _ { K } ( r ) \quad { \mathrm { f o r ~ a l l } } \quad \mu \in K .\tag{B.6}
$$

Since $\nu _ { \mu }$ is non-atomic, $F _ { \mu }$ is continuous. The probability integral transform gives

$$
( F _ { \mu } ) _ { \# } \nu _ { \mu } = \mathrm { U n i f } [ 0 , 1 ] .
$$

Last, define

$$
\begin{array} { c } { { H _ { 0 } \colon K \times \mathbb { R } ^ { d } \to [ 0 , 1 ] , } } \\ { { ( \mu , x ) \mapsto F _ { \mu } \bigl ( \varphi ( \mu , x ) \bigr ) . } } \end{array}
$$

By the preceding two displays, it holds that

$$
H _ { 0 } ( \mu , \cdot ) _ { \# } \mu = ( F _ { \mu } ) _ { \# } \nu _ { \mu } = \mathrm { U n i f } [ 0 , 1 ]
$$

for all $\mu \in K$ . It remains to show that $H _ { 0 }$ is jointly continuous. To this end, let $( \mu _ { n } , x _ { n } ) \to ( \mu , x )$ as $n \to \infty$ in $K \times \mathbb { R } ^ { d }$ . For any $r > 0$ , we estimate

$$
\begin{array} { r l } & { \left| H _ { 0 } ( \mu _ { n } , x _ { n } ) - H _ { 0 } ( \mu , x ) \right| \le \left| H _ { 0 } ( \mu _ { n } , x _ { n } ) - H _ { r } ( \mu _ { n } , x _ { n } ) \right| } \\ & { \qquad + \left| H _ { r } ( \mu _ { n } , x _ { n } ) - H _ { r } ( \mu , x ) \right| } \\ & { \qquad + \left| H _ { r } ( \mu , x ) - H _ { 0 } ( \mu , x ) \right| } \\ & { \qquad \le 2 \omega _ { K } ( r ) + \left| H _ { r } ( \mu _ { n } , x _ { n } ) - H _ { r } ( \mu , x ) \right| } \end{array}
$$

by (B.4) and (B.6). The second term in the last line of the preceding display tends to zero as $n \to \infty$ for fixed r by continuity of $H _ { r } . \mathrm { ~ A ~ }$ fter sending $n  \infty ,$ the first term $2 \omega _ { K } ( r )$ tends to zero as $r  0$ by Assumption 3.1. Thus, $H _ { 0 }$ is continuous as asserted. □

## C PROOFS OF RESULTS IN SECTION 4

This appendix provides the remaining proofs of results from Section 4 in the main text.

## C.1 PROOF OF THEOREM 4.4

Proof. Fix $\varepsilon > 0$ . By Theorem 3.6 and Remark 3.7, there exists a bounded continuous map

$$
G \colon K \times \mathbb { R } ^ { d } \to \mathbb { R } ^ { d ^ { \prime } }
$$

such that

$$
\operatorname* { s u p } _ { \mu \in K } \mathsf { W } _ { p } \big ( G ( \mu , \cdot ) _ { \# } \mu , F ( \mu ) \big ) < \frac { \varepsilon } { 2 } .\tag{C.1}
$$

Set

$$
M _ { G } : = \operatorname* { s u p } _ { ( \mu , x ) \in K \times \mathbb { R } ^ { d } } | G ( \mu , x ) | < \infty .
$$

We first compactify the input space. Let

$$
\Omega : = [ - 1 , 1 ] ^ { d }
$$

and define

$$
\begin{array} { c } { \displaystyle \iota \colon \mathbb R ^ { d } \to ( - 1 , 1 ) ^ { d } , } \\ { \displaystyle x \mapsto \iota ( x ) : = \frac { x } { 1 + | x | } . } \end{array}
$$

The map ι is bounded, continuous, and injective. Since it is 1-Lipschitz, the induced pushforward

$$
\begin{array} { r } { \iota _ { \# } \colon \mathcal { P } _ { p } ( \mathbb { R } ^ { d } ) \to \mathcal { P } ( \Omega ) , } \\ { \mu \mapsto \iota _ { \# } \mu } \end{array}
$$

is continuous with respect to $\mathsf { W } _ { p }$ . Define

$$
K ^ { \iota } : = \{ \iota _ { \# } \mu \in { \mathcal { P } } ( \Omega ) \colon \mu \in K \} \subset { \mathcal { P } } _ { p } ( \mathbb { R } ^ { d } ) .
$$

Since $K$ is compact, $K ^ { \iota }$ is also compact by continuity of $\iota _ { \# }$ . Moreover,

$$
\iota \# | \kappa \colon K \to K ^ { \iota }
$$

is a homeomorphism.

Since K is compact in $( \mathcal { P } _ { p } ( \mathbb { R } ^ { d } ) , \mathsf { W } _ { p } )$ , it is uniformly tight. Hence

$$
\tau _ { R } : = \operatorname* { s u p } _ { \mu \in { \cal K } } \mu \bigl ( \mathbb { R } ^ { d } \setminus B _ { R } ( 0 ) \bigr ) \to 0 \quad \mathrm { a s } \quad R \to \infty .
$$

Choose $R > 0$ sufficiently large such that

$$
2 M _ { G } \tau _ { R } ^ { 1 / p } < \frac { \varepsilon } { 4 } .\tag{C.2}
$$

Set

$$
C _ { R } : = \iota ( \overline { { B _ { R } ( 0 ) } } ) \subset \Omega .
$$

Then $C _ { R }$ is compact. Define

$$
\widetilde { G } _ { R } \colon K ^ { \iota } \times C _ { R } \to \overline { { B _ { M _ { G } } ( 0 ) } }
$$

by

$$
{ \widetilde { G } } _ { R } { \big ( } \iota _ { \# } \mu , \iota ( x ) { \big ) } : = G ( \mu , x ) \quad { \mathrm { f o r ~ a l l } } \quad \mu \in K \quad { \mathrm { a n d } } \quad x \in { \overline { { B _ { R } ( 0 ) } } } .
$$

This map is well-defined and continuous because ι and $\iota \# | _ { K }$ are homeomorphisms onto their images.

Since $K ^ { \iota } \times C _ { R }$ is a closed subset of the compact metric space $\mathscr { P } ( \Omega ) \times \Omega$ and $\overline { { B _ { M _ { G } } ( 0 ) } }$ is convex, the Dugundji extension theorem (Dugundji, 1951) delivers the existence of a continuous map

$$
\widehat { G } _ { R } : \mathcal { P } ( \Omega ) \times \Omega  \overline { { B _ { M _ { G } } ( 0 ) } }
$$

such that

$$
{ \widehat { G } } _ { R } { \big ( } \iota _ { \# } \mu , \iota ( x ) { \big ) } = G ( \mu , x ) \quad { \mathrm { f o r ~ a l l } } \quad \mu \in K \quad { \mathrm { a n d } } \quad x \in { \overline { { B _ { R } ( 0 ) } } } ,\tag{C.3}
$$

and

$$
\operatorname* { s u p } _ { ( \nu , z ) \in \mathcal { P } ( \Omega ) \times \Omega } | \widehat { G } _ { R } ( \nu , z ) | \leq M _ { G } .\tag{C.4}
$$

Choose $\delta \in ( 0 , 1 )$ sufficiently small such that

$$
\delta < \frac { \varepsilon } { 8 } .\tag{C.5}
$$

By the universal approximation theorem for measure-theoretic transformer in-context maps on the compact domain $\dot { \mathcal { P } ( \Omega ) } \times \Omega$ (Furuya et al., 2025b, Theorem 1), there exists a measure-theoretic transformer

$$
\mathsf T _ { \delta } \colon \mathcal P ( \Omega ) \times \Omega  \mathbb R ^ { d ^ { \prime } }
$$

of the form (4.2)—also depending on R—such that

$$
\operatorname* { s u p } _ { ( \nu , z ) \in \mathcal { P } ( \Omega ) \times \Omega } \bigl | \mathsf { T } _ { \delta } ( \nu , z ) - \widehat { G } _ { R } ( \nu , z ) \bigr | < \delta .\tag{C.6}
$$

In particular, by (C.4) and the preceding display, it holds that

$$
\operatorname* { s u p } _ { ( \nu , z ) \in \mathcal { P } ( \Omega ) \times \Omega } | \mathsf { T } _ { \delta } ( \nu , z ) | < \delta + M _ { G } .\tag{C.7}
$$

Since Ω is compact, $\mathcal { P } ( \Omega ) \times \Omega$ is compact. Since $\mathsf { T } _ { \delta }$ is continuous on $\mathscr { P } ( \Omega ) \times \Omega$ (Furuya et al., 2026), it is uniformly continuous. Therefore, there exists a nondecreasing function

$$
\omega _ { \delta } \colon [ 0 , \infty )  [ 0 , \infty )
$$

such that li $\mathrm { 1 } _ { r \downarrow 0 } \omega _ { \delta } ( r ) = 0$ and

$$
\left| \mathsf T _ { \delta } (  { \boldsymbol \nu } , z ) - \mathsf T _ { \delta } (  { \boldsymbol \nu } ^ { \prime } , z ^ { \prime } ) \right| \le \omega _ { \delta } \bigl ( \mathsf { W } _ { p } (  { \boldsymbol \nu } ,  { \boldsymbol \nu } ^ { \prime } ) + | z - z ^ { \prime } | \bigr )\tag{C.8}
$$

for every $\nu$ and $\nu ^ { \prime }$ in ${ \mathcal { P } } ( \Omega )$ and z and $z ^ { \prime }$ in Ω. Now choose $\rho > 0$ such that

$$
\omega _ { \delta } ( \rho ) < \frac { \varepsilon } { 8 } .\tag{C.9}
$$

After $\delta$ and $\rho$ have been fixed, choose $S \geq R$ sufficiently large such that

$$
\tau _ { S } : = \operatorname* { s u p } _ { \mu \in { \cal K } } \mu \bigl ( \mathbb { R } ^ { d } \setminus B _ { S } ( 0 ) \bigr ) < \biggl ( \frac { \rho } { 4 \sqrt { d } } \biggr ) ^ { p } .\tag{C.10}
$$

We next approximate ι on $\overline { { B _ { S } ( 0 ) } }$ by a token-wise MLP. Define

$$
\mathrm { c l i p } ( t ) : = - 1 + \mathrm { R e L U } ( t + 1 ) - \mathrm { R e L U } ( t - 1 )
$$

and

$$
\mathrm { C l i p } ( z ) : = \bigl ( \exp ( z _ { 1 } ) , \dots , \mathrm { c l i p } ( z _ { d } ) \bigr ) .
$$

Then Clip is exactly representable by a shallow ReLU neural network and satisfies

$$
{ \mathrm { C l i p } } ( \mathbb { R } ^ { d } ) \subset \Omega .
$$

For any $\eta > 0$ , there exists a ReLU MLP $\widetilde { N } _ { \eta } \colon  { \mathbb { R } ^ { d } } \to  { \mathbb { R } ^ { d } }$ satisfying

$$
\operatorname* { s u p } _ { x \in B _ { S } ( 0 ) } | \widetilde { N } _ { \eta } ( x ) - \iota ( x ) | < \eta
$$

by the universal approximation theorem for ReLU MLPs. Set

$$
N _ { \eta } : = \mathrm { C l i p } \circ \widetilde { N } _ { \eta } ,
$$

which is also a MLP. Since Clip is the metric projection onto $\Omega = [ - 1 , 1 ] ^ { d }$ , it is 1-Lipschitz. Thus,

$$
\operatorname* { s u p } _ { x \in \overline { { B _ { S } ( 0 ) } } } \vert N _ { \eta } ( x ) - \iota ( x ) \vert = \operatorname* { s u p } _ { x \in \overline { { B _ { S } ( 0 ) } } } \vert \mathrm { C l i p } \big ( \widetilde { N } _ { \eta } ( x ) \big ) - \mathrm { C l i p } \big ( \iota ( x ) \big ) \vert < \eta .\tag{C.11}
$$

Moreover,

$$
N _ { \eta } ( \mathbb { R } ^ { d } ) \subset \Omega .
$$

Choose $\eta > 0$ sufficiently small such that

$$
\eta < { \frac { \rho } { 4 } } .\tag{C.12}
$$

Let $\Gamma _ { \mathrm { i d } }$ be an attention layer of the form (4.1) with $Q ^ { h } = K ^ { h } = V ^ { h } = 0$ for every head h. Then

$$
\Gamma _ { \mathrm { i d } } ( \mu , x ) = x
$$

for every $( \mu , x ) \in \mathcal { P } ( \mathbb { R } ^ { d } ) \times \mathbb { R } ^ { d }$ . Define T: $\mathcal { P } ( \mathbb { R } ^ { d } ) \times \mathbb { R } ^ { d }  \mathbb { R } ^ { d ^ { \prime } }$ by

$$
\mathsf { T } : = \mathsf { T } _ { \delta } \diamondsuit N _ { \eta } \diamondsuit \Gamma _ { \mathrm { i d } } .
$$

Equivalently,

$$
\mathsf { T } ( \mu , x ) = \mathsf { T } _ { \boldsymbol { \delta } } \big ( ( N _ { \eta } ) _ { \# } \mu , N _ { \eta } ( x ) \big ) .
$$

The initial identity attention layer and the token-wise MLP $N _ { \eta }$ put T in the form (4.2). The initial layer $\Gamma _ { \mathrm { i d } }$ is defined for every probability measure. Then $\mu \mapsto ( N _ { \eta } ) _ { \# } \mu$ maps ${ \mathcal { P } } ( \mathbb { R } ^ { d } )$ into $\mathcal { P } ( [ - 1 , 1 ] ^ { d } )$ Consequently, all subsequent attention layers act on measures supported on compact subsets of Euclidean space because measure-theoretic attention operators map compactly supported probability measures to compactly supported probability measures. Thus, T is well-defined on $\mathcal { P } ( \mathbb { R } ^ { \dot { d } } ) \times \mathbb { R } ^ { d }$

We next estimate the error incurred by replacing ι with $N _ { \eta }$ . Let $\mu \in K$ . The probability measure $( N _ { \eta } , \iota ) _ { \# } \mu$ is a coupling of $( N _ { \eta } ) _ { \# } \mu$ and $\nu \# \mu$ . Hence

$$
\begin{array} { r l } & { \mathsf { W } _ { p } ^ { p } \bigl ( ( N _ { \eta } ) _ { \# } \mu , \iota _ { \# } \mu \bigr ) \le \displaystyle \int _ { \mathbb { R } ^ { d } } | N _ { \eta } ( x ) - \iota ( x ) | ^ { p } \mu ( d x ) } \\ & { \qquad \le \eta ^ { p } + ( 2 \sqrt { d } ) ^ { p } \mu \bigl ( \mathbb { R } ^ { d } \setminus B _ { S } ( 0 ) \bigr ) } \end{array}
$$

by splitting the integral and invoking (C.11). Using $( | a | ^ { p } + | b | ^ { p } ) ^ { 1 / p } \leq | a | + | b |$ yields

$$
\operatorname* { s u p } _ { \mu \in K } \mathsf { W } _ { p } \bigl ( ( N _ { \eta } ) _ { \# } \mu , \iota _ { \# } \mu \bigr ) \le \eta + 2 \sqrt { d } \tau _ { S } ^ { 1 / p } ,
$$

where $\tau _ { S }$ is as in (C.10). So, for $x \in B _ { R } ( 0 ) \subset B _ { S } ( 0 )$ , it holds that

$$
\mathsf { W } _ { p } \big ( ( N _ { \eta } ) _ { \# } \mu , \iota _ { \# } \mu \big ) + | N _ { \eta } ( x ) - \iota ( x ) | \le 2 \eta + 2 \sqrt { d } \tau _ { S } ^ { 1 / p } < \frac { \rho } { 2 } + \frac { \rho } { 2 } = \rho
$$

by (C.12) and (C.10). Then by (C.8) and (C.9), it holds that

$$
\left. \mathsf { T } ( \mu , x ) - \mathsf { T } _ { \delta } \big ( \iota _ { \# } \mu , \iota ( x ) \big ) \right. = \left. \mathsf { T } _ { \delta } \big ( ( N _ { \eta } ) _ { \# } \mu , N _ { \eta } ( x ) \big ) - \mathsf { T } _ { \delta } \big ( \iota _ { \# } \mu , \iota ( x ) \big ) \right. \le \omega _ { \delta } ( \rho ) < \frac { \varepsilon } { 8 } .
$$

Moreover, (C.3) and (C.6) yield

$$
\operatorname* { s u p } _ { \mu \in K } \left| \mathsf T _ { \delta } \big ( \iota _ { \# } \mu , \iota ( x ) \big ) - G ( \mu , x ) \right| < \delta .
$$

Thus, $\mathrm { i f } x \in B _ { R } ( 0 )$ , then

$$
| \mathsf { T } ( \mu , x ) - G ( \mu , x ) | < \frac { \varepsilon } { 8 } + \delta .\tag{C.13}
$$

In the other hand, if $x \in \mathbb { R } ^ { d } \setminus B _ { R } ( 0 )$ , then (C.7) and a crude bound show that

$$
\vert \mathsf { T } ( \mu , x ) - G ( \mu , x ) \vert \le ( \delta + M _ { G } ) + M _ { G } = \delta + 2 M _ { G } .
$$

Consequently,

$$
| \mathsf { T } ( \mu , x ) - G ( \mu , x ) | \leq \frac { \varepsilon } { 8 } \mathbf { 1 } _ { B _ { R } ( 0 ) } ( x ) + \delta + 2 M _ { G } \mathbf { 1 } _ { \mathbb { R } ^ { d } \setminus B _ { R } ( 0 ) } ( x ) \quad \mathrm { f o r ~ a l l } \quad x \in \mathbb { R } ^ { d } .\tag{C.14}
$$

Using the common random variable $X ~ \sim ~ \mu$ to couple $\mathsf { T } ( \mu , X )$ and $G ( \mu , X )$ , followed by Minkowski’s inequality and (C.14), we obtain the estimate

$$
\operatorname* { s u p } _ { \mu \in K } \mathsf { W } _ { p } \big ( \mathsf { T } ( \mu , \cdot ) _ { \# } \mu , G ( \mu , \cdot ) _ { \# } \mu \big ) \le \frac { \varepsilon } { 8 } + \delta + 2 M _ { G } \tau _ { R } ^ { 1 / p } < \frac { \varepsilon } { 2 }\tag{C.15}
$$

by (C.5) and (C.2). Last, the triangle inequality, (C.15), and (C.1) deliver the bound

$$
\begin{array} { l } { \displaystyle \operatorname* { s u p } _ { \mu \in K } \mathsf { W } _ { p } \bigl ( \mathsf { T } ( \mu , \cdot ) _ { \# } \mu , F ( \mu ) \bigr ) \leq \displaystyle \operatorname* { s u p } _ { \mu \in K } \mathsf { W } _ { p } \bigl ( \mathsf { T } ( \mu , \cdot ) _ { \# } \mu , G ( \mu , \cdot ) _ { \# } \mu \bigr ) } \\ { \displaystyle \qquad + \operatorname* { s u p } _ { \mu \in K } \mathsf { W } _ { p } \bigl ( G ( \mu , \cdot ) _ { \# } \mu , F ( \mu ) \bigr ) } \\ { \displaystyle \qquad < \frac { \varepsilon } { 2 } + \frac { \varepsilon } { 2 } = \varepsilon } \end{array}
$$

as asserted.

## C.2 PROOF OF COROLLARY 4.6

Proof. The proof follows the same construction as that of Theorem 3.6, but now with the input measure $\mu$ replaced by the continuously-varying source measure $\eta ( \mu )$ . Indeed, the assumption (4.3), together with the continuity of η, yields, by the same argument as in Lemma B.3, a continuous map

$$
\widetilde { H } _ { 0 } \in \mathcal { C } \big ( K \times \mathbb { R } ^ { m } , [ 0 , 1 ] \big )
$$

such that

$$
\widetilde { H } _ { 0 } ( \mu , \cdot ) _ { \# } \eta ( \mu ) = \mathrm { U n i f } [ 0 , 1 ] \quad \mathrm { f o r ~ a l l } \quad \mu \in { \cal K } .
$$

The remainder of the proof is identical to that of Theorem 3.6, except now with $H _ { 0 } ( \mu , \cdot ) _ { \# } \mu$ replaced by $\widetilde { H } _ { 0 } ( \mu , \cdot ) _ { \# } \eta ( \mu )$ □