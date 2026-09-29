# Weighting Schedules Govern What and When Score-Based Generative Models Learn from Multimodal Data

Jérémie Klinger\*<sup>1</sup>, Raphaël Urfin<sup>1</sup>, Giulio Biroli<sup>1</sup>, and Marylou Gabrié<sup>1</sup>

<sup>1</sup>Laboratoire de Physique de l’École normale supérieure, ENS, Université PSL, CNRS, Sorbonne Université, Université Paris Cité, F-75005 Paris, France

## Abstract

Score-based generative models generate new samples by integrating a time-dependent drift that carries Gaussian noise onto the target distribution. In practice this drift is modeled by a neural network, trained on a loss integrated over time t with a weighting schedule w(t). Along the backward dynamics, and for multi-modal distributions, trajectories commit to modes of the target within a narrow time window, the speciation time. In this work, focusing on high-dimensional data, we decompose the integrated loss into its single-time contributions and analyze each at fixed signal-to-noise ratio Λ(t): we show that Λ(t) sets the rate at which each feature of a multimodal target—the mode directions and their relative weights—is acquired during training. Crucially, at high Λ(t) all mode directions are acquired together, on a single timescale insensitive to their amplitudes, while the relative weights are not learned at all. Only near the speciation time, where Λ(t) becomes of order one, do all features become learnable, each on its own timescale: the weights are acquired jointly with the directions, and the directions at rates set by their relative amplitudes. For models trained on time-integrated objectives, the learning dynamics is then governed by how much of the weighting effectively sits near the speciation time, which provides insights on w(t) design choices. These results follow from an exact high-dimensional analysis of the training dynamics of unbalanced and hierarchical Gaussian mixtures. Numerical experiments on image and human genome haplotype generation recover the predicted hierarchy of learning timescales in more complex settings.

Keywords: Score-Based Generative Models | Multimodal Data | Weighting Scheme

## 1 Introduction

Over the last decade, the fast-paced development of transport-based generative models has led to tremendous progress in diverse and complex sampling tasks across both natural and artificial modalities. Vision modeling represents one such novelty, with latent diffusion models constituting state-of-the-art image [15, 35] and video generators [29, 42]. In the physical and chemical sciences, Boltzmann generators [31] offer an alternative to traditional sampling methods such as MCMC, either directly replacing them or augmenting them [17, 20, 33]. While this empirical progress has repeatedly outpaced our ability to explain it, significant leverage still lies in a precise theoretical understanding of training and the subsequent generative process.

The unified framework of Stochastic Interpolants [SI, 1, 2] proposes a shared theoretical setting to discuss score-based generative models. Building such models amounts to optimizing a drift field driving Gaussian samples towards the target distribution, with generation quality directly impacted by practitioners’ training and sampling choices [25]. Within this paradigm, the statistical physics analysis of reverse time dynamics [11] underlying the generation process has led to the identification of a narrow critical timescale dubbed speciation time: the brief window in which the generative trajectory commits to one mode of the target distribution, plainly identifying where compute effort should be spent during the sampling phase. While the practical impact of this result is undeniable, it assumes the optimal drift to be known. In practice that drift is learned, and which features of the target the converged score actually resolves is decided by the choices made during the training phase.

Chief among those is the weighting schedule, which sets the noise levels the objective emphasizes and drives much of the reported progress in sample quality and training efficiency [12, 21, 16]. These gains, however, rest largely on empirical tuning: how the weighting schedule governs the acquisition of data features during training still lacks a theoretical account. We bridge this gap by considering the training dynamics of two controlled families of structured data distributions: unbalanced and hierarchical Gaussian Mixture Models (GMM). We show that the design of the training objective quantitatively sets the rate at which each feature — mode direction, relative weight, mode amplitude — is acquired. In turn we empirically demonstrate that these results generalize to standard SI pipelines, with direct implications for the choice of weighting schedule.

Main contributions. Stochastic interpolant optimization is generically recast as minimizing

$$
\mathcal { L } ( \pmb { \theta } ) = d ^ { - 1 } \mathbb { E } _ { \mathbf { x } _ { 0 } , \pmb { \xi } } \left[ \int _ { 0 } ^ { t _ { \operatorname* { m a x } } } \mathrm { d } t w ( t ) \| \beta ( t ) \mathbf { s } _ { \pmb { \theta } } \left( \alpha ( t ) \mathbf { x } _ { 0 } + \beta ( t ) \pmb { \xi } , t \right) + \pmb { \xi } \| ^ { 2 } \right]\tag{1}
$$

with respect to the model parameters $\theta ,$ where $\mathbf { x } _ { \mathrm { 0 } }$ is drawn from the target distribution $P _ { 0 }$ and $\boldsymbol { \xi }$ from a standard Gaussian $\sqrt { ( 0 , \bar { I _ { d } } ) }$ . With unlimited training time and exact minimization, the integrated loss is minimized by the exact score at every noise level, whatever the weighting schedule w. In practice, however, training is stopped after a finite time and is at finite precision. The noising schedule $( \alpha , \beta )$ and weighting schedule w then constitute important design choices whose influence on learning dynamics for high-dimensional data is the primary concern of this paper.

• Training at fixed noise level - For both unbalanced and structured GMM variants characterized by finitely many modes with distinct directions and weights, we analyze the training dynamics of a network ${ \bf s } _ { \theta }$ specifically tailored to the GMM datasets at a single noising time $t ^ { * }$ . The timescale on which each mode direction and mode weight are learned critically depends on $t ^ { * }$ . In particular, for $t ^ { * }$ small with respect to the speciation time $t _ { s } ,$ only the mode directions are recovered, on a timescale insensitive to the mode amplitudes, while focusing training around $t _ { s }$ leads to weights and directions being learned jointly, at rates set by the modes’ relative amplitude.

• Influence of the weighting schedule - Using the signal-to-noise ratio as the shared noising metric, we precisely show how the choice of weighting schedule w imposes effective weights for the relevant noise levels. As a result, the dominant noising regions control the learning dynamics of mode direction and weights. Importantly, we emphasize that Diffusion and Flow Matching (FM) correspond to different effective weights, resulting in distinct learning dynamics.

• Empirical validation on real datasets - We evaluate our results against i) handmade unbalanced and hierarchical MNIST dataset and ii) Haplotypes from the Human Genome Dataset (HGD). We sweep over a breadth of SI frameworks, completing the GMM Denoising Score Matching experiments with Flow Matching, as well as the more recent Masked Discrete Diffusion framework for the HGD, and recover quantitative agreement between theory and the empirical timescales governing hierarchical feature emergence.

## Related works

Where structure appears at generation time. Mode commitment during the generation dynamics, first identified in Raya and Ambrogioni [32], is thoroughly analyzed in Biroli and Mézard [10], Biroli et al. [11]: they introduce the speciation time, characterize it analytically for high-dimensional GMMs through overlap summary statistics, and generalize it to arbitrary distributions. The analysis is further extended to mixtures of strongly log-concave densities in Li and Chen [27]. Behjoo and Chertkov [9], Sclocchi et al. [37] probe the same windows empirically through forward-backward U-turn protocols on pretrained denoisers. All of these assume the score to be exact and locate a narrow commitment window along the sampling trajectory. The complementary question, at what training time each feature becomes available, is the one we address.

Where structure appears at training time. A series of works characterizes the optimal score reachable at finite sample complexity for a given neural network architecture, without studying the training dynamics: Cui et al. [13] consider a two-layer autoencoder at a fixed noising time for a symmetric bimodal GMM, completed by Aranguri and Insulla [5], who study mode weight recovery for asymmetric GMMs, while George et al. [19] derive precise learning curves for random feature networks. Another line of work focuses on the training dynamics itself. Outside the diffusion paradigm, Bachtis et al. [7] follow the training dynamics of energy-based models on hierarchically structured data, and find the levels of the hierarchy to be learned successively. Within the diffusion framework, Cui et al. [14] extend the two-layer autoencoder to all noising times, at the price of an encoding of time with low expressivity, Nicoletti et al. [30] track how data structure and imbalance shape the dynamics in a random feature model, and Bardone et al. [8] show that moments of a distribution are learned sequentially. Closest to our question, Wang and Pehlevan [41] find the covariance eigendirections of a Gaussian target to display sequential alignment during training. Whereas all the structure lies in the covariance, we here focus on multimodal structure.

Noising and weighting schedules. Kingma and Gao [26] show that most standard diffusion objectives are the ELBO under different weightings; Choi et al. [12], Karras et al. [25], Hang et al. [21] and Esser et al. [16] propose state-of-the-art SNR-based schedules and weighting functions. On the theory side, Aranguri et al. [6] optimize noise schedules in high dimension, while Aranguri and Insulla [5] analyze mode weight recovery using ad-hoc weighting schedules, and Gagneux et al. [18] study the influence of the choice of weighting scheme on the performance of trained neural networks. Instead, we here focus on learning of mode weights and hierarchy in standard SI pipelines and analyze the consequences of the resulting weighting schemes.

## 2 Setting

Stochastic interpolants. The Stochastic Interpolant (SI) framework $[ 1 , 2 ]$ encompasses both Diffusion Models (DM) [38, 39, 23] and Flow Matching (FM) [28]. A SI defines a bridge between a target distribution $P _ { 0 }$ and Gaussian white noise by introducing the interpolation

$$
\mathbf { x } _ { t } = \alpha ( t ) \mathbf { x } _ { 0 } + \beta ( t ) \pmb { \xi } ,\tag{2}
$$

where $( \mathbf { x } _ { 0 } , \pmb { \xi } ) \sim P _ { 0 } \times \mathcal { N } ( 0 , \pmb { I } _ { d } ) , t \in [ 0 , t _ { \operatorname* { m a x } } ] ^ { 1 }$ , and the schedule functions $( \alpha , \beta )$ defining the SI obey the boundary conditions $\alpha ( 0 ) = 1 , \ \beta ( 0 ) = 0 , \ \alpha ( t _ { \mathrm { m a x } } ) = 0 , \ \beta ( t _ { \mathrm { m a x } } ) = 1$ . By construction, the marginal distribution $\mathbf { x } _ { t } \sim P _ { t }$ interpolates between $P _ { 0 }$ and $\mathcal { N } ( 0 , \pmb { I } _ { d } )$ . Following the pioneering works of [4] and [22], the interpolant $\mathbf { x } _ { t }$ can also be written as the solution at time t of the deterministic flow

$$
\dot { { \mathbf x } } ( t ) = { \mathbf b } ( { \mathbf x } ( t ) , t ) , { \mathbf x } ( t _ { \mathrm { m a x } } ) \sim \mathcal { N } ( 0 , { I } _ { d } )\tag{3}
$$

integrated backward in time, with velocity field

$$
\mathbf { b } ( x , t ) = \mathbb { E } [ { \dot { \mathbf { x } } } _ { t } \mid \mathbf { x } _ { t } = \mathbf { x } ] = { \frac { { \dot { \alpha } } ( t ) } { \alpha ( t ) } } \mathbf { x } - \left( { \dot { \beta } } ( t ) - { \frac { { \dot { \alpha } } ( t ) } { \alpha ( t ) } } \beta ( t ) \right) \beta ( t ) \nabla _ { \mathbf { x } } \log P _ { t } ( \mathbf { x } ) ,\tag{4}
$$

so that integrating equation 3 generates new samples from $P _ { 0 }$ . The term $\mathbf { s } ^ { * } ( \mathbf { x } , t ) \equiv \nabla _ { \mathbf { x } }$ log $P _ { t } ( \mathbf { x } )$ , dubbed the score function, is generally not available in closed form, but it is the minimizer of the Denoising Score Matching (DSM) loss [24, 40]

$$
\mathcal { L } _ { \mathrm { D S M } } ( \mathbf { s } ) = \frac { 1 } { d } \mathbb { E } _ { ( \mathbf { x } _ { 0 } , \pm ) \sim P _ { 0 } \times \mathcal { N } ( 0 , I _ { d } ) } \left[ \int _ { 0 } ^ { t _ { \operatorname* { m a x } } } \mathrm { d } t w ( t ) \| \beta ( t ) \mathbf { s } \left( \mathbf { x } _ { t } , t \right) + \pmb { \xi } \| ^ { 2 } \right] ,\tag{5}
$$

where the weighting function $w ( t )$ is arbitrary. In practice the score function is approximated by a neural network with parameters θ and optimized with local methods such as stochastic gradient descent [34].

Schedule and weighting functions $( \alpha , \beta , w )$ are crucial design choices: they set the relative weights of various noise levels and thus which features of the target distribution $P _ { 0 }$ can possibly be recovered. The precise interplay between these choices and the training or sampling dynamics remains poorly understood, and the few results available assume an exact, optimally trained score [10, 11, 6]. They underlie our own analysis and are recalled below.

Data distributions. We focus on two Gaussian mixtures with closed-form scores, one unbalanced and one hierarchical (see Fig. 1 and 2 for illustrative histograms):

$\left( \mathcal { D } _ { 1 } \right)$ , an unbalanced bimodal distribution,

$$
\begin{array} { r l } & { P _ { 0 } ( \mathbf { x } ) = \omega \mathcal { N } ( \mu , \sigma ^ { 2 } I _ { d } ) + ( 1 - \omega ) \mathcal { N } ( - \mu , \sigma ^ { 2 } I _ { d } ) , \quad | | \mu | | ^ { 2 } = d , \quad \omega , \sigma ^ { 2 } = O _ { d } ( 1 ) , } \\ & { \nabla \log P _ { t } ( \mathbf { x } ) = - \cfrac { \mathbf { x } } { \Gamma _ { t } } + \cfrac { \alpha ( t ) \mu } { \Gamma _ { t } } \operatorname { t a n h } \left( \cfrac { \alpha ( t ) \mu ^ { T } \mathbf { x } } { \Gamma _ { t } } + \frac { 1 } { 2 } \log \frac { \omega } { 1 - \omega } \right) , } \end{array}\tag{6}
$$

with $\Gamma _ { t } = \alpha ^ { 2 } ( t ) \sigma ^ { 2 } + \beta ^ { 2 } ( t )$

$\left( \mathcal { D } _ { 2 } \right)$ , a quadrimodal distribution inspired by Bachtis et al. [7],

$$
\begin{array} { l } { { \displaystyle P _ { 0 } ( { \bf x } ) = \frac { 1 } { 4 } \sum _ { s _ { 1 } = \pm 1 } \sum _ { s _ { 2 } = \pm 1 } \mathcal { N } ( s _ { 1 } \mu _ { 1 } + s _ { 2 } \mu _ { 2 } , \sigma ^ { 2 } I _ { d } ) } , \ ~ } \\ { { \displaystyle \nabla \log P _ { t } ( { \bf x } ) = - \frac { { \bf x } } { \Gamma _ { t } } + \frac { \alpha ( t ) \mu _ { 1 } } { \Gamma _ { t } } \operatorname { t a n h } \left( \frac { \alpha ( t ) { \bf \Sigma } \mu _ { 1 } ^ { T } { \bf x } } { \Gamma _ { t } } \right) + \frac { \alpha ( t ) \mu _ { 2 } } { \Gamma _ { t } } \operatorname { t a n h } \left( \frac { \alpha ( t ) { \bf \Sigma } \mu _ { 2 } ^ { T } { \bf x } } { \Gamma _ { t } } \right) } , \ ~ } \end{array}\tag{7}
$$

where $\mu _ { 1 } \in \mathbb { R } ^ { d }$ is supported on the first κd coordinates and $\pmb { \mu } _ { 2 }$ on the remaining $( 1 - \kappa ) d$ with $\| \pmb { \mu } _ { 1 } \| ^ { 2 } = \kappa d , \| \pmb { \mu } _ { 2 } \| ^ { 2 } = ( 1 - \kappa ) d , \kappa = O _ { d } ( 1 )$ . We work in the high-dimensional limit $d \gg 1$ , for two reasons: realistic data are high-dimensional, and, as we show below, this is precisely the regime in which the choice of $w ( t )$ matters most.

Diffusion Regimes and Speciation Time. Consider a symmetric bimodal target $\begin{array} { r } { P _ { 0 } = \frac { 1 } { 2 } \mathcal { N } ( \pmb { \mu } , \pmb { I } _ { d } ) + } \end{array}$ $\scriptstyle { \frac { 1 } { 2 } } { \mathcal { N } } ( - \mu , I _ { d } )$ and a diffusion model, i.e. $\alpha ( t ) = e ^ { - t } , \beta ( t ) = \sqrt { 1 - e ^ { - 2 t } } , t \in [ 0 , t _ { \operatorname* { m a x } } ]$ , for which the exact score $\mathbf { s } ^ { * } ( \mathbf { x } , t )$ is known and the reverse-time drift equation 4 can be analyzed in closed form. Biroli et al. [11] show that the mode overlap<sup>2</sup> $q _ { t } = \mu ^ { T } \mathbf { x } _ { t } / d$ evolves in an effective time-dependent potential $V _ { t } ( q )$ whose shape sharply shifts around the speciation time $t _ { s } \equiv { \frac { 1 } { 2 } } \log ( d )$ . In Regime $\scriptstyle \mathrm { I , }$ corresponding to times $t \gg t _ { s } , V _ { t }$ is effectively unimodal, and no information is retained about the bimodal structure of $P _ { 0 }$ . Conversely, in Regime $\mathrm { I I } , t \ll t _ { s } , V _ { t }$ consists of two well-separated harmonic wells centered on 1: the bimodality of $P _ { 0 }$ is fully resolved and the generated sample has committed to one of the modes. The speciation time $t _ { s }$ is the narrow window in which $V _ { t }$ develops a cusp that decides the generation outcome, and specifies where sampling compute is best spent when integrating equation 4. The speciation phenomenon has been confirmed in other theoretical models [10, 3, 27] and in real cases by numerical experiments [9, 37].

Note however that these results only concern sampling and they do not address how a trained score approaches $\mathbf { s } ^ { * }$ as optimization of equation 5 is carried out, nor whether $t _ { s }$ plays a significant role, or whether the picture holds once modes carry structure beyond the simple bimodal balanced GMM - unequal mode weights, or a hierarchy of sub-modes at different scales. The aim of this paper is to study training dynamics under such structure, for general stochastic interpolants.

## 3 Analytical results

In the following, we study the training of a parametrized score model $\mathbf { s } _ { \pmb { \theta } } ( \mathbf { x } , t )$ by gradient descent on the population loss equation $5 ^ { 3 }$ , in the infinitesimal learning rate limit known as Gradient Flow (GF)

$$
\frac { \mathrm { d } \pmb { \theta } } { \mathrm { d } \tau } = - \nabla _ { \pmb { \theta } } \mathcal { L } _ { \mathrm { D S M } } ( \mathbf { s } _ { \pmb { \theta } } ) ,\tag{8}
$$

where the training time τ should not be confused with the sampling time t. The aim is to elucidate the dynamical timescale dependencies on the dimension d and structure parameters $\omega , \kappa .$ . To isolate dependencies between the schedules $( \alpha , \beta )$ and $w ,$ it is insightful to consider first a single noise level, equivalent to a Dirac weighting function $w ( t ) = \delta ( t - t ^ { * } )$ . In this case, the synthetic distributions $\mathcal { D } _ { 1 , 2 } ,$ which allow for analytical treatment, explicitly showcase the sequential learning of structured data features.

## 3.1 Single fixed noising time

Score models. At fixed noising time $t ,$ we parametrize the trainable score as a two-layer network with skip connection and tanh activation: for $\left( \mathcal { D } _ { 1 } \right)$ ,

$$
\mathbf { s } _ { \theta = ( c , b , \mathbf { w } ) } ( \mathbf { x } , t ) = c \mathbf { x } + \mathbf { w } \operatorname { t a n h } ( \mathbf { w } ^ { T } \mathbf { x } + b ) , \qquad c , b \in \mathbb { R } , \ \mathbf { w } \in \mathbb { R } ^ { d } ,\tag{9}
$$

with the learned weight read out of the bias, $\rho = e ^ { 2 b } / ( 1 + e ^ { 2 b } )$ ; and for $\left( \mathcal { D } _ { 2 } \right)$

$$
\begin{array} { r } { { \bf s } _ { \theta = ( c , { \bf w } _ { 1 } , { \bf w } _ { 2 } ) } ( { \bf x } , t ) = c { \bf x } + { \bf w } _ { 1 } \operatorname { t a n h } ( { \bf w } _ { 1 } ^ { T } { \bf x } ) + { \bf w } _ { 2 } \operatorname { t a n h } ( { \bf w } _ { 2 } ^ { T } { \bf x } ) , } \end{array}\tag{10}
$$

where $\mathbf { w } _ { 1 }$ lives in the same block dimensional space as $\pmb { \mu } _ { 1 }$ (resp. for $\big ( \mathbf { w } _ { 2 } , \mu _ { 2 } \big ) \big )$ . Under the GF dynamics equation 8, we show in appendix $( \ S \mathrm { A } . 2 . 1 )$ and $( \ S \mathrm { A } . 2 . 2 )$ that $\mathcal { L } _ { \mathrm { D S M } }$ depends on w (resp. $\mathbf { w } _ { 1 } , \mathbf { w } _ { 2 } )$ only through its projection onto the mode direction(s) and its orthogonal component, so training reduces to a closed system of ODEs on the summary statistics

$$
m = { \frac { \mathbf { w } ^ { T } \boldsymbol { \mu } } { \alpha ( t ) d } } , \quad q = { \frac { \| \mathbf { w } ^ { \bot } \| ^ { 2 } } { \alpha ( t ) ^ { 2 } d } } \qquad \left( \mathrm { r e s p . ~ } m _ { i } = { \frac { \mathbf { w } _ { i } ^ { T } \mu _ { i } } { \alpha ( t ) \kappa _ { i } d } } , \quad q _ { i } = { \frac { \| \mathbf { w } _ { i } ^ { \bot } \| ^ { 2 } } { \alpha ( t ) ^ { 2 } \kappa _ { i } d } } , \ i = 1 , 2 \right)\tag{11}
$$

together with c (and b for $\mathcal { D } _ { 1 } )$ , where $\mathbf { w } ^ { \perp }$ is the component of w orthogonal to $\pmb { \mu } .$ The parameterizations equation 9 and equation 10 are expressive enough to represent the exact score. Our aim is therefore to determine whether, and on what timescales, the summary statistics reach their optimal values during training.

Signal-to-noise ratio and speciation time for generic SI. As shown in appendix $( \ S \mathrm { A . 1 } )$ , the speciation time analysis carried out in [11] holds for generic SI schedules $( \alpha , \beta )$ . Mode selection during sampling is now governed by the signal-to-noise (SNR) ratio

$$
\Lambda ( t ) = \frac { \alpha ^ { 2 } ( t ) d } { \alpha ^ { 2 } ( t ) \sigma ^ { 2 } + \beta ^ { 2 } ( t ) } ,\tag{12}
$$

redefining the speciation time by the convention $\Lambda ( t _ { s } ) \equiv 1$ (equivalently $\alpha / \beta \sim d ^ { - 1 / 2 } )$ . The underlying principle is clear: mode mixing occurs at speciation and this analysis carries over to the training paradigm. Indeed, training deep in Regime $\Pi , \Lambda _ { t } \stackrel { - } { = } O ( d )$ , exposes the network only to noise levels at which the data structure is fully resolved<sup>4</sup>; training around speciation exposes it to meaningfully collapsed data structures. By characterizing training at a given $\Lambda _ { t }$ in Regime II and around speciation, for each of $( \mathcal { D } _ { 1 } ) , ( \mathcal { D } _ { 2 } )$ , we show that its choice determines not only how fast $( m , q )$ are learned, but what is learned at all. Our findings are summarized in the following results.

Result 1 (Training dynamics for $\left( \mathcal { D } _ { 1 } \right) \left( \ S \mathrm { A } . 2 . 1 \right) )$ . Let $\mathbf { w } ( 0 ) \in \mathbb { S } ^ { d - 1 }$ with $c ( 0 ) , b ( 0 ) = O _ { d } ( 1 )$ , and let $( c , \mathbf { w } , b )$ evolve under GF equation 8for the score model equation 9 atfixed $\Lambda _ { t }$ . In the $d \to \infty$ limit the d-dimensionalflow closes on the summary statistics $( m , q , c , b )$ , which obey a finite system of ODEs given in §A.2.1. Specifically:

(a) Regime II $( \Lambda _ { t } = O _ { d } ( d ) )$ . On $\tau = O _ { d } ( 1 )$ times $c \to - 1 / ( \Gamma _ { t } + \alpha _ { t } ^ { 2 } )$ , while $( m , q , b )$ stay at initialization. $O n \tau = O _ { d } ( d )$ times $( m , q , b )$ relax to $( 1 / \Gamma _ { t } , 0 , b ( 0 ) )$ while c relaxes instantaneously $t o \ c ^ { * } ( m , q )$

(b) Speciation $( \Lambda _ { t } = O _ { d } ( 1 ) )$ . On $\tau = O _ { d } ( 1 )$ times $c  - 1 / \Gamma _ { t }$ while on $\tau = O _ { d } ( d )$ times $( m , q , b ) $ $\begin{array} { r } { ( 1 / \Gamma _ { t } , 0 , \frac { 1 } { 2 } \log \frac { \omega } { 1 - \omega } ) } \end{array}$

On $\tau = O _ { d } ( 1 )$ times only the skip connection c evolves; the model learns the score associated to $\mathcal { N } ( 0 , ( \Gamma _ { t } + \alpha _ { t } ^ { 2 } ) I _ { d } )$ , the best unimodal approximation to the bimodal target. Multimodality is then acquired on $\tau = O _ { d } ( d )$ times and in a regime-dependent fashion: in Regime II the network recovers the exact score equation 6 up to its bias, so the direction is learned but the imbalance is not $( { \mathrm { F i g . ~ } } 1 , l e f t ) .$ ; at speciation, direction and imbalance are recovered jointly (Fig. 1, middle).

Result 2 (Training dynamics for $\left( \mathcal { D } _ { 2 } \right) \left( \ S \mathrm { A } . 2 . 2 \right) )$ . Let $\begin{array} { r } { \kappa _ { 1 } = \kappa , \kappa _ { 2 } = 1 - \kappa , \mathbf { w } _ { 1 } ( 0 ) \in \mathbb { S } ^ { \kappa _ { 1 } d - 1 } , \mathbf { w } _ { 2 } ( 0 ) \in \mathbb { S } ^ { \kappa _ { 2 } d - 1 } } \end{array}$ and $c ( 0 ) = O _ { d } ( 1 )$ . Let $\left( c , \mathbf { w } _ { 1 } , \mathbf { w } _ { 2 } \right)$ evolve under the GF equation 8for the score model equation 10 atfixed $\Lambda _ { t } .$ . In the $d \to \infty$ limit the d-dimensional flow closes on the summary statistics $( m _ { 1 } , q _ { 1 } , m _ { 2 } , q _ { 2 } , c )$ , each pair $( m _ { i } , q _ { i } )$ obeying the system of Result 1 at $\omega = 1 / 2 , b = 0$ and rescaled SNR $\gamma _ { i } ^ { 2 } = \kappa _ { i } \Lambda _ { t . }$ , the two systems being coupled only through c. Specifically:

(a) Regime $I I \left( \Lambda _ { t } = { \cal O } _ { d } ( d ) \right)$ . On $\tau = O _ { d } ( 1 )$ times $c \to - 1 / ( \Gamma _ { t } + \alpha _ { t } ^ { 2 } )$ , while $( m _ { i } , q _ { i } )$ stay at initialization. On $\tau = O _ { d } ( d )$ times c relaxes instantaneously to $c ^ { * } ( m _ { 1 } , q _ { 1 } , m _ { 2 } , q _ { 2 } )$ while $( m _ { i } , q _ { i } )  ( 1 / \Gamma _ { t } , 0 )$ on a κ-independent timescale.

(b) Speciation $( \Lambda _ { t } = O _ { d } ( 1 ) )$ . On $\tau = O _ { d } ( 1 )$ times $c  - 1 / \Gamma _ { t } ,$ the two systems decouple and, for $i = 1 , 2 $ $( m _ { i } , q _ { i } )  ( 1 / \Gamma _ { t } , 0 )$ on κ-dependent timescales $\tau _ { i } .$ . In the small κ limit, $\tau _ { 1 } / \tau _ { 2 } \sim 1 / \kappa$

<sup>4</sup>This presupposes that the structural parameters ω themselves remain $O _ { d } ( 1 )$ , contrary to the results presented in Aranguri and Insulla [5]

![](images/832e8b05202823e40921bd554f8d1155d8a51a370732dc8a0520de109f6fcf4e.jpg)

![](images/84a6196ed22e4023140701f3542ed4614ee0e59bf55084bb1b478e8d1c2b2a20.jpg)

![](images/ca31c6aa4536541589955cdd49d79753df77c6dc7f432b947983da70c0578a2a.jpg)  
Figure 1: Exact score parametrization for $\left( \mathcal { D } _ { 1 } \right)$ . Overlap m (dashed), skip connection c (solid) and mode weight (dotted), against training steps rescaled by d, for the unbalanced GMM at $\omega = 0 . 8 ,$ , different values of d and two noising times, t = 0.5 (left) and $t = t _ { s }$ (middle). Typical bimodal histograms of projected samples are shown on the $( r i g h t )$ panel. In Regime II (left) c plateaus on $O ( 1 )$ times and m converges on $O ( d )$ times, while the weight stays frozen at its initial value $1 / 2 .$ . At speciation (right), c first reaches its final value and then weight and overlap rise together, on a shared $O ( d )$ timescale.

As in the unbalanced case, on $\tau = O _ { d } ( 1 )$ times the model only adapts its skip connection and learns the best unimodal approximation of the target. The two directions are then acquired on a $O _ { d } ( d )$ timescale, in a regime-dependent fashion. In Regime II both overlaps evolve jointly, independently of the block asymmetry $( { \mathrm { F i g } } . 2 , l e f t ) ;$ at speciation however each block evolves at its own SNR set by $\kappa .$ Consequently, learning is hierarchical, with the dominant direction $\pmb { \mu } _ { 2 }$ acquired before $\pmb { \mu } _ { 1 }$ (Fig. 2, middle).

Since the dynamics of $( m _ { i } , q _ { i } )$ are coupled, the rate of growth of $m _ { i }$ drifts as $q _ { i }$ relaxes, making its κ dependence intractable. To proceed further, we impose the extra constraint of fixed $\mathbf { w } _ { i }$ norm.

Result 2 bis (Heuristic growth rate §A.2.3). Constrain each $\mathbf { w } _ { i }$ to its target norm, $\| \mathbf { w } _ { i } \| = \alpha ( t ) \| \pmb { \mu } _ { i } \| ,$ , i.e. $m _ { i } ^ { 2 } + q _ { i } = 1$ . The small $m _ { i }$ linearized dynamics of Result 2(b) then reduces to the single ODE

$$
\dot { m } _ { i } = d ^ { - 1 } \lambda ( \gamma _ { i } ) m _ { i } + { \cal { O } } ( m _ { i } ^ { 3 } ) ,\tag{13}
$$

with $\lambda ( \gamma )$ given in closedform in equation 103 of §A.2.3.

While this assumption is provably not exact, it is closed-form and recovers 2(b) exactly as $\kappa  0$ Consequently, we use $\tau _ { i } \propto 1 / \lambda ( \gamma _ { i } )$ as the reference timescale against which training time is rescaled in Fig. 2 and throughout.

![](images/8732e71a9d8b0a168b289be7b2c0b46606f4e7bf7c247da02329a2243cb06afa.jpg)  
Figure 2: Exact score parametrization for $\left( \mathcal { D } _ { 2 } \right)$ . Block overlaps $m _ { 1 }$ (dashed) and $m _ { 2 }$ (solid), against training steps, for the quadrimodal GMM at $d = 1 0 2 4$ , structure parameter $\kappa \in \{ 0 . 1 , 0 . 2 , 0 . 3 \}$ and two noising times, $t = 0 . 5 ( l e f t )$ and $t = t _ { s }$ (middle and right). In Regime II (left) the two overlaps are learned on the same training timescale, independently of κ. At speciation (middle) they evolve on different timescales with $\pmb { \mu } _ { 1 }$ being learned after $\pmb { \mu } _ { 2 }$ . Rescaling each block’s training time by its own closed-form rate $\lambda ( \gamma _ { i } )$ of equation 13, with $\gamma _ { i } ^ { 2 } = \bar { \kappa _ { i } } \Lambda _ { t } \Gamma _ { i }$ <sub>t</sub> and $\kappa _ { 1 } = \kappa , \kappa _ { 2 } = 1 - \kappa ,$ collapses the curves (right). The inset shows $\left( \mathcal { D } _ { 2 } \right)$ samples, projected on $( \mu _ { 1 } , \mu _ { 2 } )$ . The smaller $\kappa ,$ the closer the modes pair up along $\pmb { \mu } _ { 1 }$ and the more hierarchical the target.

## 3.2 Integrated loss

Standard SI pipelines optimize the integrated loss equation 5 rather than a single-time one, exposing the model to many noise levels at once. Since this loss is a weighted superposition of fixed-SNR denoising problems, Results 1 and 2 suggest that training depends on how much weight is placed on each SNR region, which we now make precise.

Rewriting a denoising objective in terms of SNR rather than time is now standard [26], and underlies state-of-the-art schedule designs [25, 16]. Switching from $t \mapsto \Lambda$ in the DSM loss equation $5 ,$ we collect every schedule $( \alpha , \beta )$ and weighting w into a single effective SNR weight $w _ { \mathrm { e f f } } ( \bar { \Lambda ) } = w ( t ( \Lambda ) ) | \mathrm { d } t / \mathrm { d } \Lambda | .$ which fully characterizes a given SI framework (§A.3). In light of the fixed-SNR analysis, most of the structural information is passed to the model around speciation: the relevant signal peaks at $\Lambda = O _ { d } ( 1 )$ and comparing pipelines amounts to comparing the weight they place around that scale. Under this lens, standard Diffusion (DM) and Flow Matching $( \mathrm { \check { F } M } ) ^ { 5 }$ pipelines differ explicitly. For $\Lambda \ll d ,$

$$
w _ { \mathrm { e f f } } ^ { \mathrm { D M } } ( \Lambda ) \propto \Lambda ^ { - 1 } \qquad w _ { \mathrm { e f f } } ^ { \mathrm { F M } } ( \Lambda ) \propto \sqrt { d } \Lambda ^ { - 3 / 2 } .\tag{14}
$$

Diffusion is scale-free and allocates the same denoising effort to every decade of SNR, exposing the network to a broad mixture of noise levels from Regime II down past speciation. Conversely, Flow Matching carries the extra factor $\sqrt { d / \Lambda }$ and tilts training towards low SNR. Note that this concentration is not a design choice but a consequence of the canonical FM schedule itself. We therefore expect the FM training dynamics to be dominated by speciation-specific effects, whereas Diffusion should display more mixed behavior. In turn, this predicts distinct structure learning signatures, which we test on $( \dot { \mathcal { D } } _ { 1 } )$ and $\left( \mathcal { D } _ { 2 } \right)$ by replacing the analytical score of §3.1 with a generic ResNet denoiser trained under both FM and DM effective SNR weighting functions (§B.2). Neither pipeline is expected to reproduce the fixed-SNR analysis exactly — the network shares parameters across noise levels, so the fixed-SNR dynamics do not simply superpose — but we still find qualitative agreement.

On $\left( \mathcal { D } _ { 1 } \right)$ (Fig. 3a), once two modes appear in the generated samples, uniform DSM first relaxes towards an almost even split, as predicted by Result 1(a), before converging to ω; its earlier rise in positive fraction is spurious, reporting a displaced mean while the histograms are still unimodal at (1). Under the FM weighting, weight and direction are instead acquired together, as at speciation. On $\left( \mathcal { D } _ { 2 } \right) \left( \mathrm { F i g } . 3 \mathrm { b } \right)$ , the overlap with the subdominant direction $\pmb { \mu } _ { 1 }$ separates with κ under both weightings, more sharply under FM, for which rescaling training steps by $\lambda ( { \sqrt { \kappa } } )$ collapses the curves.

![](images/291adab82a3d3f9b9681c5545b0df4465123448ee0cab34c978bfacf90b8c4b0.jpg)  
(a)

![](images/61767b098b987755c84b254f33475e0b0526a02965d83e2d90ba829ad3c40fef.jpg)

![](images/c1323218ba35efbe386c6ffcf03de93369c32c981e32fe467ac819ce039d77dc.jpg)  
(b)

![](images/f1917b2b12a9ccd6e75aa4a6bb92c1557a71d8dd5efebc24b1e9939562dd234c.jpg)  
Figure 3: ResNet for GMMs at $d = 6 4 .$ , under uniform DSM (left) and FM-weighted DSM (right) in each panel, against training steps. (a) $\left( \mathcal { D } _ { 1 } \right)$ at $\omega = 0 . 8 \mathrm { : }$ cosine similarity with $\pmb { \mu }$ (blue) and positive fraction of $p = \mathbf { x } ^ { T } \pmb { \mu } / \Vert \pmb { \mu } \Vert ^ { 2 }$ (red); insets show the histogram of p at the three marked steps. (b) $\left( \mathcal { D } _ { 2 } \right)$ : cosine similarity with $\pmb { \mu } _ { 1 }$ for $\kappa \in \{ 0 . 2 , 0 . 3 , 0 . 4 \}$ ; the inset replots the FM curves against training steps rescaled by $\lambda ( { \sqrt { \kappa } } )$

## 4 Structure learning in complex data

While synthetic GMMs allow for a clear interpretation of the structural features composing the data, such identification in real world datasets is typically open to interpretation. Nevertheless we demonstrate that our theory captures learning dynamics across data modalities and SI design choices through three numerical experiments of increasing complexity. Precise details on all numerical experiments are provided in appendix (§B).

Weight learning in unbalanced MNIST. The paradigmatic multi-class MNIST dataset offers a natural route to test coordinated learning of class index and class proportion. To mimic a bimodal distribution we prune MNIST to two classes $3 / 6$ with relative tunable proportion $\omega .$ . We compare on $\mathrm { F i g . }$ 4a out-of-the-box DSM and FM implementations for a fixed imbalance $\omega = 0 . 8$ in favor of the 6s and use a pretrained digit classifier for evaluating empirical class proportion.

Under Flow Matching, class proportion and direction evolve jointly: as denoising progresses, the fraction of blurry images classified as 6s ramps up to ω concurrently with increasing image crispness, as expected of a pipeline overweighting low SNR regions. Unlike the controlled ResNet experiments discussed above, where only w changes, however, the comparison with DSM is inconclusive: proportion and direction appear to be learned on the same timescale under both objectives. We attribute this to the weaker mode separation of unbalanced MNIST, which shortens the relevant SNR range distinguishing the two weighting schemes (the two out-of-the-box pipelines also differ in more than w, §B.3.3).

Learning hierarchy in (Tinted MNIST/FM). To investigate hierarchical feature emergence during training we design a new dataset by adding a digit-orthogonal mode to the $3 / 6$ pruned MNIST. Specifically, we randomly colorize each image in either red or green with tunable intensity $\alpha \in ( 0 , 1 ] ;$ see Fig. 6 for representative samples and appendix B.3.1 for details on the dataset construction. Importantly, the resulting dataset is reminiscent of $\mathcal { D } _ { 2 }$ , with orthogonal digit and color directions. To match the GMM analysis, the squared amplitude of the color direction scales as $\alpha ^ { 2 }$ and can thus be made arbitrarily small.

Within the FM pipeline, training is dominated by the speciation region; consequently, digit learning occurs on a single α-independent timescale, whereas the color direction is acquired on an α dependent one, its onset spanning close to a decade across the sweep (Fig. 4b). We emphasize that even on this complex, image-based dataset, the GMM rescaling ansatz equation 13 still quantitatively captures the onset of color mode emergence, provided the experiment is matched to the single SNR analysis by identifying the small mode amplitude $| \mu _ { 1 } | \propto \alpha$

![](images/5d2e55431116b3ba0941ec399119ac55d05fda02cd5759e61159836e2a9f4f1c.jpg)  
(a)

![](images/532c3e7f0a74875f163fd1136da08c065b619b67b10b1f803e7d4bb7625ea5da.jpg)

![](images/4c664d32a73caeb44cdb107bd8f085fb352a632eddecea4fc382e296b4224fa7.jpg)

(b)  
![](images/b9c17f9cc91b5fd1216685ca14d6f7f3ac528b742747f6e423929f69e68de1ec.jpg)  
Figure 4: Experiments on MNIST. (a) Cosine similarity with the digit direction (blue) and fraction of $6 s$ (orange), against training steps, for MNIST restricted to 3s and 6s, of which a fraction $\omega = 0 . 8$ are 6s (dotted line), under DSM $( l e f t )$ and FM (right). Both objectives acquire direction and proportion within a single decade of training, MNIST offering too short an SNR range for the two weightings to separate. (b) Overlap with the digit direction (left) and with the color direction $( r i g h t ) ,$ , against training steps, for tinted MNIST trained with FM at tint intensities $\alpha ,$ the dataset being built so that the squared amplitude of the color mode scales as $\alpha ^ { 2 }$ . The digit direction is learned at a rate nearly insensitive to $\alpha ,$ while the color direction separates with it; rescaling training time by λ(α) collapses the color curves (inset).

Sequentially learning structured features in genomic data (HGD/MDLM). Departing from synthetic datasets, we finally consider haplotype data (binary sequences) from the Human Genome Dataset (HGD), following Bachtis et al. [7]. To test the scope of our theory, we consider yet another SI framework, Masked Discrete Diffusion, adapting the original implementation of Sahoo et al. [36] to the binary data at hand (see App. B.4 for implementation details). Unlike in the GMM and tinted MNIST cases, feature structure in the HGD is not made obvious by construction. To assess sequential learning of data features we pre-compute the training data’s PCA eigenvectors and track the statistics of inferred samples projected on these fixed directions, shown in Fig. 5.

The complexity of the HGD is made apparent in the left panel of Fig. 5, where the first and second eigenvector projections display clear multi-modality. The right panel shows that the generated data acquires the top PCA eigendirections sequentially, the leading two being separated by close to half a decade of training while the subsequent ones only begin to rise at the end of the run. Following the single SNR analysis, we rescale training time by $\lambda ( \sqrt { v _ { k } } )$ , with $v _ { k }$ the variance explained along direction k (App. B.4). The collapse holds for the two leading directions, precisely those along which the projected data is visibly multimodal (left panel), and degrades beyond: the smaller eigendirections carry no resolved modes, so the mode amplitude the ansatz rests on has no counterpart there. The MDLM loss is moreover scale free (App. A.5), which, as in the ResNet experiment of Fig. 3b, does not prevent sequential emergence.

![](images/7007216c32e677fa3369e1baed12bf4d8f6c8101d6e29bfdf1b267ab64aee9eb.jpg)

![](images/e2d2c75176d32b2e04c8528fa3e7a4339bba743263cf66fbec65c4d2ef3cee8c.jpg)  
Figure 5: Sequential feature learning on HGD. (Left) HGD samples projected onto the first two PCA eigenvectors, showing a complex multimodal structure. $( R i g h t )$ Overlap of the generated samples with the leading PCA components $\mu _ { i } , i = 1 , \ldots , 5 ,$ against training steps, under Masked Discrete Diffusion: the directions are recovered one after the other, and rescaling training time by the closed-form rate $\lambda ( \sqrt { v _ { i } } )$ of equation 13, with v<sub>i</sub> the variance explained along the i-th reference PCA direction, collapses the two leading ones (inset).

## 5 Conclusion and future work

Reading a training objective through its effective SNR weighting turns an arbitrary design choice into a predictive one: which features a model resolves, and when, follows from where that weighting places its mass relative to the speciation scale. Gaussian mixtures make this concrete. Training deep in Regime II of Biroli et al. [11] recovers every mode direction on a common timescale but leaves the relative weights untouched, however long one trains. Training around speciation instead acquires weights and directions jointly, and separates the directions themselves at rates set by their relative amplitudes. The same phenomenology holds for networks trained on unbalanced MNIST and, quantitatively, on tinted MNIST and genomic haplotype data.

The same statement, read backwards, is a design principle. Rather than inheriting a weighting schedule, one could tailor it to the structure of a given target, so as to acquire a chosen feature first or to reach a desired structure within a fixed compute budget. We leave this to future work.

## Acknowledgment.

GB acknowledges support from the French government under the management of ANR: PEPR-IA (project MAGICALL ANR-25-PEIA-0004) and PR[AI]RIE-PSAI (ANR-23-IACL- 0008). JK acknowledges financial support from the postdoctoral Junior Research and Teaching Laplace chair, funded by Capital Fund Management and the École Normale Supérieure.

## References

[1] Michael S. Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. In International Conference on Learning Representations, 2023. URL https://arxiv.org/ abs/2209.15571.

[2] Michael S. Albergo, Nicholas M. Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and diffusions. Journal ofMachine Learning Research, 26(209):1–80, 2025. URL https://jmlr.org/papers/v26/23-1605.html.

[3] Luca Ambrogioni. The statistical thermodynamics of generative diffusion models, 2023. URL https://arxiv.org/abs/2310.17467.

[4] Brian D. O. Anderson. Reverse-time diffusion equation models. Stochastic Processes and their Applications, 12(3):313–326, 1982. URL https://doi.org/10.1016/0304-4149(82)90051-5.

[5] Santiago Aranguri and Francesco Insulla. Phase-aware training schedule simplifies learning in flow-based generative models, 2024. URL https://arxiv.org/abs/2412.07972.

[6] Santiago Aranguri, Giulio Biroli, Marc Mézard, and Eric Vanden-Eijnden. Optimizing noise schedules of generative models in high dimensions, 2025. URL https://arxiv.org/abs/2501. 00988.

[7] Dimitrios Bachtis, Giulio Biroli, Aurélien Decelle, and Beatriz Seoane. Cascade of phase transitions in the training of energy-based models. In Advances in Neural Information Processing Systems, volume 37, pages 55591–55619, 2024. URL https://proceedings.neurips.cc/paper\_files/paper/ 2024/file/648a5a590ca6f2bb5de53f938e230160-Paper-Conference.pdf.

[8] Lorenzo Bardone, Claudia Merger, and Sebastian Goldt. A theory of learning data statistics in diffusion models, from easy to hard, 2026. URL https://arxiv.org/abs/2603.12901.

[9] Hamidreza Behjoo and Michael Chertkov. U-turn diffusion. Entropy, 27(4):343, March 2025. ISSN 1099-4300. doi: 10.3390/e27040343. URL http://dx.doi.org/10.3390/e27040343.

[10] Giulio Biroli and Marc Mézard. Generative diffusion in very large dimensions. Journal of Statistical Mechanics: Theory and Experiment, 2023(9):093402, 2023. URL https://doi.org/10.1088/ 1742-5468/acf8ba.

[11] Giulio Biroli, Tony Bonnaire, Valentin de Bortoli, and Marc Mézard. Dynamical regimes of diffusion models. Nature Communications, 15(1):9957, 2024. URL https://doi.org/10.1038/ s41467-024-54281-3.

[12] Jooyoung Choi, Jungbeom Lee, Chaehun Shin, Sungwon Kim, Hyunwoo Kim, and Sungroh Yoon. Perception prioritized training of diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11462–11471, 2022. URL https://arxiv.org/abs/2204. 00227.

[13] Hugo Cui, Florent Krzakala, Eric Vanden-Eijnden, and Lenka Zdeborová. Analysis of learning a flowbased generative model from limited sample complexity. In International Conference on Learning Representations, pages 51929–51955, 2024. URL https://proceedings.iclr.cc/paper\_files/ paper/2024/file/e45a448dfa778f6d62729a7bc8633c06-Paper-Conference.pdf.

[14] Hugo Cui, Cengiz Pehlevan, and Yue Lu. A solvable model of learning generative diffusion: Theory and insights. In Advances in Neural Information Processing Systems, volume 38, pages 5253– 5296, 2025. URL https://proceedings.neurips.cc/paper\_files/paper/2025/file/ 082d3d795520c43214da5123e56a3a34-Paper-Conference.pdf.

[15] Prafulla Dhariwal and Alexander Quinn Nichol. Diffusion models beat GANs on image synthesis. In Advances in Neural Information Processing Systems, 2021. URL https://openreview.net/forum? id=OU98jZWS3x\_.

[16] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis. In Proceedings of the 41st International Conference on Machine Learning, 2024. URL https://arxiv.org/abs/2403.03206.

[17] Marylou Gabrié, Grant M. Rotskoff, and Eric Vanden-Eijnden. Adaptive Monte Carlo augmented with normalizing flows. Proceedings ofthe National Academy ofSciences, 119(10):e2109420119, 2022. URL https://doi.org/10.1073/pnas.2109420119.

[18] Anne Gagneux, Ségolène Martin, Rémi Gribonval, and Mathurin Massias. The generation phases of flow matching: A denoising perspective, 2025. URL https://arxiv.org/abs/2510.24830.

[19] Anand Jerry George, Rodrigo Veiga, and Nicolas Macris. Denoising score matching with random features: Insights on diffusion models from precise learning curves, 2026. URL https://arxiv. org/abs/2502.00336.

[20] Louis Grenioux, Leonardo Galliano, Ludovic Berthier, Giulio Biroli, and Marylou Gabrié. Boltzmann generators for amorphous particle systems, 2025. URL https://arxiv.org/abs/2512.16607.

[21] Tiankai Hang, Shuyang Gu, Chen Li, Jianmin Bao, Dong Chen, Han Hu, Xin Geng, and Baining Guo. Efficient diffusion training via min-SNR weighting strategy. In IEEE/CVF International Conference on Computer Vision (ICCV), 2023. URL https://arxiv.org/abs/2303.09556.

[22] U. G. Haussmann and E. Pardoux. Time reversal of diffusions. The Annals of Probability, 14(4): 1188–1205, 1986. URL https://doi.org/10.1214/aop/1176992362.

[23] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, volume 33, pages 6840– 6851, 2020. URL https://proceedings.neurips.cc/paper\_files/paper/2020/file/ 4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf.

[24] Aapo Hyvärinen. Estimation of non-normalized statistical models by score matching. Journal of Machine Learning Research, 6(24):695–709, 2005. URL https://jmlr.org/papers/v6/ hyvarinen05a.html.

[25] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusionbased generative models, 2022. URL https://arxiv.org/abs/2206.00364.

[26] Diederik P. Kingma and Ruiqi Gao. Understanding diffusion objectives as the ELBO with simple data augmentation. In Advances in Neural Information Processing Systems, volume 36, 2023. URL https://arxiv.org/abs/2303.00848.

[27] Marvin Li and Sitan Chen. Critical windows: Non-asymptotic theory for feature emergence in diffusion models, 2024. URL https://arxiv.org/abs/2403.01633.

[28] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023. URL https://arxiv.org/abs/2210.02747.

[29] Yixin Liu, Kai Zhang, Yuan Li, Zhiling Yan, Chujie Gao, Ruoxi Chen, Zhengqing Yuan, Yue Huang, Hanchi Sun, Jianfeng Gao, Lifang He, and Lichao Sun. Sora: A review on background, technology, limitations, and opportunities of large vision models, 2024. URL https://arxiv.org/abs/ 2402.17177.

[30] Flavio Nicoletti, Chenxiao Ma, Enrico Ventura, Luca Saglietti, and Stefano Sarao Mannelli. The interplay of data structure and imbalance in the learning dynamics of diffusion models, 2026. URL https://arxiv.org/abs/2605.06367.

[31] Frank Noé, Simon Olsson, Jonas Köhler, and Hao Wu. Boltzmann generators: Sampling equilibrium states of many-body systems with deep learning. Science, 365(6457):eaaw1147, 2019. URL https: //doi.org/10.1126/science.aaw1147.

[32] Gabriel Raya and Luca Ambrogioni. Spontaneous symmetry breaking in generative diffusion models. In Advances in Neural Information Processing Systems, volume 36, pages 66377– 66389, 2023. URL https://proceedings.neurips.cc/paper\_files/paper/2023/file/ d0da30e312b75a3fffd9e9191f8bc1b0-Paper-Conference.pdf.

[33] Danyal Rehman, Charlie B. Tan, Yoshua Bengio, Avishek Joey Bose, and Alexander Tong. Autoregressive Boltzmann generators, 2026. URL https://arxiv.org/abs/2606.27361. ICML 2026 (spotlight).

[34] Herbert Robbins and Sutton Monro. A stochastic approximation method. The Annals of Mathematical Statistics, 22(3):400–407, 1951. URL https://doi.org/10.1214/aoms/1177729586.

[35] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10674–10685, 2022. URL https://doi.org/10.1109/ CVPR52688.2022.01042.

[36] Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T. Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and effective masked diffusion language models. In Advances in Neural Information Processing Systems, 2024. URL https://arxiv.org/ abs/2406.07524.

[37] Antonio Sclocchi, Alessandro Favero, and Matthieu Wyart. A phase transition in diffusion models reveals the hierarchical nature of data. Proceedings of the National Academy of Sciences, 122(1): e2408799121, 2025. URL https://doi.org/10.1073/pnas.2408799121.

[38] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In Proceedings ofthe 32nd International Conference on Machine Learning, volume 37, pages 2256–2265, 2015. URL https://proceedings.mlr.press/ v37/sohl-dickstein15.html.

[39] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021. URL https://arxiv.org/abs/2011.13456.

[40] Pascal Vincent. A connection between score matching and denoising autoencoders. Neural Computation, 23(7):1661–1674, 2011. URL https://doi.org/10.1162/NECO\_a\_00142.

[41] Binxu Wang and Cengiz Pehlevan. An analytical theory of spectral bias in the learning dynamics of diffusion models. In Advances in Neural Information Processing Systems, 2026. URL https: //openreview.net/forum?id=SDhOClkyqC.

[42] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Yuxuan Zhang, Weihan Wang, Yean Cheng, Bin Xu, Xiaotao Gu, Yuxiao Dong, and Jie Tang. CogVideoX: Text-to-video diffusion models with an expert transformer. In International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=LQzN6TRFg9.

# Weighting Schedules Govern What and When Score-Based Generative Models Learn from Multimodal Data

Supplementary Material (SM)

Jérémie Klinger, Raphaël Urfin, Giulio Biroli, Marylou Gabrié

## A Details on the theoretical results

## A.1 Dynamical regimes and signal-to-noise ratio

In this section we adapt the computation of Biroli et al. [11], Aranguri et al. [6] to stochastic interpolants, in order to identify the speciation time i.e. the time at which the generative trajectory commits to one of the modes of the target. Let us consider as a target distribution a bimodal symmetric isotropic Gaussian distribution

$$
P _ { 0 } = \frac { 1 } { 2 } \mathcal { N } ( \pmb { \mu } , \sigma ^ { 2 } \pmb { I } _ { d } ) + \frac { 1 } { 2 } \mathcal { N } ( - \pmb { \mu } , \sigma ^ { 2 } \pmb { I } _ { d } ) , \qquad \| \pmb { \mu } \| ^ { 2 } = d .\tag{15}
$$

Since the stochastic interpolant at time t is ${ \bf x } _ { t } = \alpha ( t ) { \bf x } _ { 0 } + \beta ( t ) \pmb { \xi }$ , the marginal $P _ { t }$ is also a bimodal symmetric isotropic mixture,

$$
\begin{array} { r } { P _ { t } = \frac { 1 } { 2 } \mathcal { N } ( \alpha ( t ) \mu , \Gamma _ { t } I _ { d } ) + \frac { 1 } { 2 } \mathcal { N } ( - \alpha ( t ) \mu , \Gamma _ { t } I _ { d } ) , \qquad \Gamma _ { t } = \alpha ^ { 2 } ( t ) \sigma ^ { 2 } + \beta ^ { 2 } ( t ) , } \end{array}\tag{16}
$$

whose score reads

$$
\nabla _ { \mathbf { x } } \log P _ { t } ( \mathbf { x } ) = - \frac { \mathbf { x } } { \Gamma _ { t } } + \frac { \alpha ( t ) \pmb { \mu } } { \Gamma _ { t } } \operatorname { t a n h } \left( \frac { \alpha ( t ) \pmb { \mu } ^ { T } \mathbf { x } } { \Gamma _ { t } } \right) .\tag{17}
$$

Signal-to-noise ratio. We study the evolution, during the generative dynamics, of the overlap

$$
q _ { t } = \frac { \pmb { \mu } ^ { T } \mathbf { x } _ { t } } { \sqrt { d } } .\tag{18}
$$

Writing $\mathbf { x } _ { 0 } = s \pmb { \mu } + \sigma \mathbf { z } ,$ with $s = \pm 1$ the label of the mode from which the sample is drawn and $\mathbf { z } \sim \mathcal { N } ( 0 , \pmb { I } _ { d } )$ and projecting $\mathbf { x } _ { t } = \alpha ( t ) \mathbf { x } _ { 0 } + \beta ( t ) \boldsymbol \xi$ on $\mu / { \sqrt { d } }$ gives

$$
\begin{array} { r } { q _ { t } = \underbrace { s \alpha ( t ) \sqrt { d } } _ { \mathrm { s i g n a l } } + \underbrace { \alpha ( t ) \sigma g _ { 1 } + \beta ( t ) g _ { 2 } } _ { \mathrm { n o i s e } } , \qquad g _ { 1 } , g _ { 2 } \overset { \mathrm { i i d } } { \sim } \mathcal { N } ( 0 , 1 ) , } \end{array}\tag{19}
$$

where we used $\| \pmb { \mu } \| ^ { 2 } = d$ and the fact that ${ \pmb \mu } ^ { T } { \bf z } / \sqrt { d }$ and $\pmb { \mu } ^ { T } \pmb { \xi } / \sqrt { d }$ are standard Gaussians. The two noise contributions are independent, of total variance $\alpha ^ { 2 } ( t ) \bar { \sigma ^ { 2 } } + \beta ^ { 2 } ( t ) = \Gamma _ { t }$ . In turn, we define the time-dependent signal-to-noise ratio $\Lambda _ { t }$ as

$$
\Lambda _ { t } = \frac { \left( \mathbb { E } \left[ q _ { t } \mid s \right] \right) ^ { 2 } } { \operatorname { V a r } \left( q _ { t } \mid s \right) } = \frac { \alpha ^ { 2 } ( t ) \| \pmb { \mu } \| ^ { 2 } } { \Gamma _ { t } } = \frac { \alpha ^ { 2 } ( t ) d } { \alpha ^ { 2 } ( t ) \sigma ^ { 2 } + \beta ^ { 2 } ( t ) } ,\tag{20}
$$

where $\Lambda _ { t }$ decreases from $\Lambda _ { t } = d / \sigma ^ { 2 }$ at $t = 0$ to $\Lambda _ { t } = 0$ at $t = t _ { \operatorname* { m a x } }$

Effective potential. Project the generative ODE equation 4

$$
{ \dot { \mathbf { x } } } ( t ) = \mathbf { b } ( \mathbf { x } ( t ) , t ) , \qquad \mathbf { b } ( \mathbf { x } , t ) = { \frac { { \dot { \alpha } } ( t ) } { \alpha ( t ) } } \mathbf { x } - \left( { \dot { \beta } } ( t ) - { \frac { { \dot { \alpha } } ( t ) } { \alpha ( t ) } } \beta ( t ) \right) \beta ( t ) \nabla _ { \mathbf { x } } \log P _ { t } ( \mathbf { x } )\tag{21}
$$

onto $\mu / { \sqrt { d } } .$ . Using equation 17 and $\pmb { \mu } ^ { T } \mathbf { x } = \sqrt { d } q$ yields a closed equation for $q ,$

$$
- \frac { \mathrm { d } q _ { t } } { \mathrm { d } t } = - \frac { \partial V _ { \mathrm { e f f } } } { \partial q } ( q _ { t } ) ,\tag{22}
$$

where t is the forward time and with

$$
V _ { \mathrm { e f f } } ( q ) = \frac { A _ { t } + B _ { t } / \Gamma _ { t } } { 2 } q ^ { 2 } - B _ { t } \log \cosh \left( \frac { \alpha ( t ) \sqrt { d } } { \Gamma _ { t } } q \right)\tag{23}
$$

$$
= \frac { A _ { t } + B _ { t } / \Gamma _ { t } } { 2 } q ^ { 2 } - B _ { t } \log \cosh \left( \sqrt { \Lambda _ { t } / \Gamma _ { t } } q \right)\tag{24}
$$

where $A _ { t } = { \dot { \alpha } } ( t ) / \alpha ( t )$ and $B _ { t } = \big ( \dot { \beta } ( t ) - A _ { t } \beta ( t ) \big ) \beta ( t )$ . We realize that $\sqrt { \Lambda _ { t } / \Gamma _ { t } }$ is the scaling variable controlling the argument of the second term, and that $\Lambda _ { t }$ alone controls it wherever $\Gamma _ { t } = O _ { d } ( \bar { 1 } )$ . Expanding the potential in low and high SNR limits, it reads, close to the origin $q = 0 ,$

$$
V _ { \mathrm { e f f } } ( q ) \simeq \left\{ \begin{array} { l l } { \displaystyle \frac { 1 } { 2 } \left( A _ { t } + \frac { B _ { t } } { \Gamma _ { t } } \left( 1 - \Lambda _ { t } \right) \right) q ^ { 2 } , } & { \Lambda _ { t } \ll 1 , } \\ { \displaystyle \frac { 1 } { 2 } \left( A _ { t } + \frac { B _ { t } } { \Gamma _ { t } } \right) q ^ { 2 } - B _ { t } \frac { \alpha ( t ) \sqrt { d } } { \Gamma _ { t } } \vert q \vert , } & { \Lambda _ { t } \gg 1 . } \end{array} \right.\tag{25}
$$

For $\Lambda _ { t } \ll 1$ the potential is purely quadratic and the trajectory is not driven towards either mode. For $\Lambda _ { t } \gg 1$ it acquires a linear cusp at the origin which makes $q = 0$ unstable: the trajectory commits to one of the two branches $q \gtrless 0$ and is carried towards the corresponding mode, reaching $q = \pm \sqrt { d }$ at the end of generation.

The transition between these two regimes occurs at some $\Lambda _ { t } = O ( 1 )$ , which defines the speciation time $t _ { s }$ . We adopt the convention $\Lambda _ { t _ { s } } = \bar { 1 } ^ { 6 }$ . This yields

$$
{ \frac { \alpha ^ { 2 } ( t _ { s } ) d } { \alpha ^ { 2 } ( t _ { s } ) \sigma ^ { 2 } + \beta ^ { 2 } ( t _ { s } ) } } = 1 \qquad \Longleftrightarrow \qquad { \frac { \alpha ^ { 2 } ( t _ { s } ) } { \beta ^ { 2 } ( t _ { s } ) } } = { \frac { 1 } { d - \sigma ^ { 2 } } } \underset { d \gg 1 } { \sim } { \frac { 1 } { d } } .\tag{26}
$$

This generalizes the analysis of Biroli et al. [11] to arbitrary stochastic interpolants.

Specific speciations times. Let us specialize this criterion for the two most widespread SI frameworks, namely Diffusion and Flow Matching.

• For Diffusion, $\alpha ( t ) = e ^ { - t }$ and $\beta ( t ) = \sqrt { 1 - e ^ { - 2 t } } ,$ , so that $\Gamma _ { t } = 1 - e ^ { - 2 t } ( 1 - \sigma ^ { 2 } )$ and

$$
\Lambda _ { t } ^ { \mathrm { { D M } } } = \frac { e ^ { - 2 t } d } { 1 - e ^ { - 2 t } ( 1 - \sigma ^ { 2 } ) } , \qquad t _ { s } = \frac 1 2 \log \left( d + 1 - \sigma ^ { 2 } \right) \underset { d \gg 1 } { \sim } \frac 1 2 \log d ,\tag{27}
$$

recovering the result of Biroli et al. [11].

• For the linear interpolant of Flow Matching, $\alpha ( t ) = 1 - t$ and $\beta ( t ) = t$ with $t \in [ 0 , 1 ]$ , so that $\Gamma _ { t } = ( 1 - t ) ^ { 2 } \sigma ^ { 2 } + t ^ { 2 }$ and

$$
\Lambda _ { t } ^ { \mathrm { F M } } = \frac { ( 1 - t ) ^ { 2 } d } { ( 1 - t ) ^ { 2 } \sigma ^ { 2 } + t ^ { 2 } } , \qquad \frac { t _ { s } } { 1 - t _ { s } } = \sqrt { d - \sigma ^ { 2 } } \quad \Longleftrightarrow \quad 1 - t _ { s } = \frac { 1 } { 1 + \sqrt { d - \sigma ^ { 2 } } } \underset { d \gg 1 } { \sim } \frac { 1 } { \sqrt { d } } .\tag{28}
$$

## A.2 Single time analysis

## A.2.1 Unbalanced distribution: exact loss and gradient flow

This section is self-contained. We derive, for the score model actually used in our experiments, the DSM loss and gradient flow exactly as functions of a finite set of summary statistics, valid for any signal-to-noise ratio, and only then specialize to the two regimes of interest by scaling analysis.

Setup. The target is the unbalanced mixture

$$
P _ { 0 } = \omega \mathcal { N } ( \mu , \sigma ^ { 2 } I _ { d } ) + ( 1 - \omega ) \mathcal { N } ( - \mu , \sigma ^ { 2 } I _ { d } ) , \qquad \| \mu \| ^ { 2 } = d , \qquad \omega , \sigma ^ { 2 } = O _ { d } ( 1 ) ,\tag{29}
$$

and we model the score with a two-layer network with trainable skip connection,

$$
\mathbf { s } _ { \theta = ( c , \mathbf { w } , b ) } ( \mathbf { x } , t ) = c \mathbf { x } + \mathbf { w } \operatorname { t a n h } \left( \mathbf { w } ^ { T } \mathbf { x } + b \right) , \qquad c , b \in \mathbb { R } , \ \mathbf { w } \in \mathbb { R } ^ { d } .\tag{30}
$$

Note that equation 30 has no explicit dependence on the schedule $( \alpha , \beta )$ : whatever amplitude and argument scale the exact score requires at the noising time $t ,$ the network must produce through training. We work at a fixed noising time $t ,$ writing $\alpha = \alpha ( t ) , \beta = \beta ( t )$ and

$$
\Gamma = \alpha ^ { 2 } \sigma ^ { 2 } + \beta ^ { 2 } , \quad \quad \Lambda = \frac { \alpha ^ { 2 } d } { \Gamma } , \quad \quad \gamma ^ { 2 } : = \alpha ^ { 2 } d = \Lambda \Gamma ,\tag{31}
$$

where Λ is the signal-to-noise ratio and $\gamma$ will turn out to be the natural scale controlling the tanh. Training is gradient flow in training time τ on the single-time DSM loss,

$$
{ \mathcal { L } } ( \theta ) = { \frac { 1 } { d } } \mathbb { E } _ { { \mathbf { x } } _ { 0 } \sim P _ { 0 } } , \boldsymbol { \xi } \sim { \mathcal { N } } ( 0 , I _ { d } ) \left[ \left\| \beta \mathbf { s } _ { \theta } ( \mathbf { x } _ { t } , t ) + \boldsymbol { \xi } \right\| ^ { 2 } \right] , \qquad \mathbf { x } _ { t } = \alpha \mathbf { x } _ { 0 } + \beta \boldsymbol { \xi } , \qquad { \frac { \mathrm { d } \theta } { \mathrm { d } \tau } } = - \nabla _ { \theta } { \mathcal { L } } .\tag{32}
$$

Summary statistics and initialization. Because equation 30 carries no $\alpha ,$ the weight vector must itself grow to the scale set by the exact score. It is therefore convenient to measure w in units of α and define

$$
m = \frac { { \bf w } ^ { T } \mu } { \alpha d } , \qquad q = \frac { \| { \bf w } ^ { \perp } \| ^ { 2 } } { \alpha ^ { 2 } d } , \qquad \mathrm { s o t h a t } \qquad \| { \bf w } \| ^ { 2 } = \alpha ^ { 2 } d \left( m ^ { 2 } + q \right) ,\tag{33}
$$

with $\mathbf { w } ^ { \perp }$ the component of w orthogonal to $\pmb { \mu } .$ Equivalently, m and $q$ are the usual overlap and orthogonal norm of the rescaled vector $\mathbf { u } = \mathbf { w } / \alpha$ . Together with c and $b ,$ these are the only quantities the loss will depend on.

We initialize $c ( 0 ) , b ( 0 ) = O _ { d } ( 1 )$ ) and draw $\mathbf { w } ( 0 ) = \mathbf { w } _ { 0 }$ uniformly on the unit sphere $\mathbb { S } ^ { d - 1 } ( 1 )$ , as in our experiments. Then $\pmb { \nu } _ { 0 } ^ { T } \pmb { \mu }$ has zero mean and variance $\| \mu \| ^ { 2 } / d = 1$ , while $\lVert \mathbf { w } _ { 0 } ^ { \perp } \rVert ^ { 2 } = 1 - O _ { d } ( 1 / d )$ , so that

$$
m _ { 0 } = O _ { d } \biggl ( \frac { 1 } { \alpha d } \biggr ) , \qquad q _ { 0 } = \frac { 1 } { \alpha ^ { 2 } d } \left( 1 + O _ { d } \bigl ( \textstyle { \frac { 1 } { d } } \bigr ) \right) = \frac { 1 } { \gamma ^ { 2 } } + O _ { d } \bigl ( \textstyle { \frac { 1 } { d } } \bigr ) .\tag{34}
$$

The initial condition thus depends on the noising time through α alone. Everything below establishes the following statement, the long form of Result 1 of the main text.

Result (1, long form: training dynamics, unbalanced distribution). Let ${ \bf w } ( 0 )$ be uniform on $\mathbb { S } ^ { d - 1 } ( 1 )$ $c ( 0 ) , b ( 0 ) = O _ { d } ( 1 )$ , and let $( c , \mathbf { w } , b )$ evolve under GF equation 8 for the score model equation 9 at fixed t. As $d \to \infty :$

(a) Regime II $( \Lambda = O _ { d } ( d ) )$ . On $\tau = O _ { d } ( 1 )$ times, $c \to c _ { 1 } ^ { * } = - 1 / ( \Gamma + \alpha ^ { 2 } )$ while $( m , q , b )$ remain at initialization; on $\tau = O _ { d } ( d )$ times,

$$
\dot { q } = - \frac { 4 \beta ^ { 2 } } { d } q , \qquad \dot { m } = - \frac { 2 \beta ^ { 2 } \Gamma } { d ( \Gamma + \alpha ^ { 2 } ) } \Big ( m - \frac { 1 } { \Gamma } \Big ) ,\tag{35}
$$

so $( m , q )  ( 1 / \Gamma , 0 ) .$ , c is slaved to the value $o f \left( m , q \right)$ and b stays frozen at its initial value: the network recovers the exact score equation 6 up to its bias, and the imbalance ω is not learned.

(b) Speciation time $( \Lambda = O _ { d } ( 1 ) )$ . Here $\Gamma  1$ and $\gamma ^ { 2 } = \Lambda \Gamma = O _ { d } ( 1 )$ , with initialization $q ( 0 ) = 1 / \gamma ^ { 2 } =$ $O _ { d } ( 1 ) , m ( 0 ) = O _ { d } ( 1 / \sqrt { d } )$ . On $\tau = O _ { d } ( 1 )$ times $c  - 1 / \Gamma$ . On $\tau = O _ { d } ( d )$ times, i.e. in the rescaled training time $\check { \tau } = \beta ^ { 2 } \tau / d ,$ , the orthogonal component relaxes on its own while, at small $( m , b )$ , the overlap and the bias leave the origin jointly,

$$
\frac { \mathrm { d } q } { \mathrm { d } \tilde { \tau } } = - 4 q \Psi ( q , \gamma ) + O ( m ^ { 2 } , b ^ { 2 } ) , \qquad \frac { \mathrm { d } } { \mathrm { d } \tilde { \tau } } \binom { m } { b } = { \bf M } ( q , \gamma ) \binom { m } { b } + O ( m ^ { 2 } , b ^ { 2 } ) ,\tag{36}
$$

with the scalar Ψ given in equation 62 and the explicit $2 \times 2$ matrix M in equation 64. Theflow converges to $q  0 , m  1 / \Gamma$ and $\begin{array} { r } { b  \bar { b } ^ { * } = \frac { 1 } { 2 } } \end{array}$ log $\frac { \omega } { 1 - \omega }$ : the imbalance is learned jointly with the mode direction, on the same $\tau = O _ { d } ( d )$ timescale on which Regime II recovers the direction alone.

Detailed dynamics.

1. Effective noises. Decompose the data as ${ \bf x } _ { 0 } = s \pmb { \mu } + \sigma { \bf z } ,$ with $s = \pm 1$ with probabilities $\omega , 1 - \omega$ and $\mathbf { z } \sim \mathcal { N } ( 0 , \pmb { I } _ { d } )$ , and introduce

$$
\begin{array} { r } { q _ { \mathbf { z } } = \displaystyle \frac { \mathbf { w } ^ { T } \mathbf { z } } { \alpha \sqrt { d ( m ^ { 2 } + q ) } } , \qquad q _ { \boldsymbol { \xi } } = \frac { \mathbf { w } ^ { T } { \boldsymbol { \xi } } } { \alpha \sqrt { d ( m ^ { 2 } + q ) } } , \qquad q _ { \mathbf { z } } , q _ { \boldsymbol { \xi } } \overset { \mathrm { i i d } } { \sim } \mathcal { N } ( 0 , 1 ) , } \end{array}\tag{37}
$$

using $\| \mathbf { w } \| ^ { 2 } = \alpha ^ { 2 } d ( m ^ { 2 } + q )$ from equation 33. The projections of data and noise onto w then read

$$
\mathbf { w } ^ { T } \mathbf { x } _ { 0 } = \alpha \left( s m d + \sigma \sqrt { d ( m ^ { 2 } + q ) } q _ { \mathbf { z } } \right) , \qquad \mathbf { w } ^ { T } \boldsymbol { \xi } = \alpha \sqrt { d ( m ^ { 2 } + q ) } q _ { \boldsymbol { \xi } } .\tag{38}
$$

2. Effective Loss. Inserting $\mathbf { x } _ { t } = \alpha \mathbf { x } _ { 0 } + \beta \pmb { \xi }$ into equation 30 and collecting terms along $\mathbf { x } _ { \mathrm { 0 } } , \pmb { \xi }$ and $\mathbf { w } ,$

$$
\beta \mathbf { s } _ { \theta } ( \mathbf { x } _ { t } , t ) + \boldsymbol { \xi } = \alpha \beta c \mathbf { x } _ { 0 } + \left( 1 + \beta ^ { 2 } c \right) \boldsymbol { \xi } + \beta T \mathbf { w } , \qquad T : = \operatorname { t a n h } \bigl ( \mathbf { w } ^ { T } \mathbf { x } _ { t } + b \bigr ) .\tag{39}
$$

Squaring, dividing by $d ,$ and using the concentration of $\begin{array} { r } { \frac { 1 } { d } \| \mathbf { x } _ { 0 } \| ^ { 2 } \to 1 + \sigma ^ { 2 } , \frac { 1 } { d } \| \pmb { \xi } \| ^ { 2 } \to 1 } \end{array}$ and $\begin{array} { r } { \frac { 1 } { d } \mathbf { x } _ { 0 } ^ { T } \pmb { \xi }  0 , } \end{array}$ together with equation 38, gives the effective loss

$$
\begin{array} { l } { { \displaystyle { \mathcal { L } = \left( 1 + \beta ^ { 2 } c \right) ^ { 2 } + \alpha ^ { 2 } \beta ^ { 2 } \left[ c ^ { 2 } \left( 1 + \sigma ^ { 2 } \right) + \left( m ^ { 2 } + q \right) \mathbb { E } \left[ T ^ { 2 } \right] + 2 c m \mathbb { E } [ s T ] \right] } } } \\ { { \displaystyle ~ + \frac { 2 \alpha \beta \sqrt { m ^ { 2 } + q } } { \sqrt { d } } \Big ( \alpha \beta c \sigma \mathbb { E } [ q _ { z } T ] + \left( 1 + \beta ^ { 2 } c \right) \mathbb { E } [ q _ { \xi } T ] \Big ) . } } \end{array}\tag{40}
$$

While the last line of equation 40 carries an explicit $1 / { \sqrt { d } } ,$ it is not subleading in d. Indeed, T is correlated with $q _ { \mathbf { z } } , q _ { \xi } .$ , and Gaussian integration by parts (Stein’s lemma), using $\partial T / \partial q _ { \xi } = \alpha \beta \sqrt { d ( m ^ { 2 } + q ) } \left( 1 - T ^ { 2 } \right)$ and $\partial T / \partial q _ { \bf z } = \alpha ^ { 2 } \sigma \sqrt { d ( m ^ { 2 } + q ) } ( 1 - T ^ { 2 } )$ from equation 38, gives

$$
\begin{array} { r l r } { \mathbb { E } [ q _ { \xi } T ] = \alpha \beta \sqrt { d ( m ^ { 2 } + q ) } \mathbb { E } \left[ 1 - T ^ { 2 } \right] , } & { } & { \mathbb { E } [ q _ { \ z } T ] = \alpha ^ { 2 } \sigma \sqrt { d ( m ^ { 2 } + q ) } \mathbb { E } \left[ 1 - T ^ { 2 } \right] . } \end{array}\tag{41}
$$

such that each expectation carries a compensating ${ \sqrt { d } } .$ Substituting into equation 40, the explicit $1 / \sqrt { d }$ cancels exactly and, using $1 + \beta ^ { 2 } c + \alpha ^ { 2 } \sigma ^ { \hat { 2 } } c = 1 + \check { c } \Gamma$ , the whole line collapses to the $O _ { d } ( 1 )$ contribution

$$
2 \alpha ^ { 2 } \beta ^ { 2 } \left( m ^ { 2 } + q \right) \left( 1 + c \Gamma \right) \mathbb { E } \left[ 1 - T ^ { 2 } \right] .\tag{42}
$$

yielding

$$
\begin{array} { r l } & { \mathcal { L } = \left( 1 + \beta ^ { 2 } c \right) ^ { 2 } } \\ & { \quad \quad + \alpha ^ { 2 } \beta ^ { 2 } \left[ c ^ { 2 } \left( 1 + \sigma ^ { 2 } \right) + \left( m ^ { 2 } + q \right) \mathbb { E } \left[ T ^ { 2 } \right] + 2 c m \mathbb { E } [ s T ] \right] } \\ & { \quad \quad + 2 \alpha ^ { 2 } \beta ^ { 2 } \left( m ^ { 2 } + q \right) \left( 1 + c \Gamma \right) \mathbb { E } \left[ 1 - T ^ { 2 } \right] . } \end{array}\tag{43}
$$

3. tanh analysis. From equation 38, $\mathbf { w } ^ { T } \mathbf { x } _ { t } = \alpha ^ { 2 } s m d + \alpha \sqrt { d ( m ^ { 2 } + q ) } \left( \alpha \sigma q _ { \mathbf { z } } + \beta q _ { \pm } \right)$ . The two independent Gaussians combine into a single $g \sim \mathcal { N } ( 0 , 1 )$ with total variance $\alpha ^ { 2 } \sigma ^ { 2 } + \beta ^ { 2 } = \Gamma$ , so that, using $\alpha ^ { \hat { 2 } } d = \gamma ^ { 2 } .$

$$
\mathbf { w } ^ { T } \mathbf { x } _ { t } + b = s \gamma ^ { 2 } m + \gamma \sqrt { \Gamma \left( m ^ { 2 } + q \right) } g + b .\tag{44}
$$

Averaging over s with probabilities $\omega , 1 - \omega ,$ and flipping $g  - g$ in the $s ~ = ~ - 1$ branch, the two expectations left in equation 40 reduce to one-dimensional Gaussian integrals,

$$
h : = \mathbb { E } [ s T ] = \omega I _ { 1 } ( b ) + ( 1 - \omega ) I _ { 1 } ( - b ) , \qquad k : = \mathbb { E } \big [ T ^ { 2 } \big ] = \omega I _ { 2 } ( b ) + ( 1 - \omega ) I _ { 2 } ( - b ) ,\tag{45}
$$

$$
I _ { 1 } ( b ) = \mathbb { E } _ { g } \Big [ \operatorname { t a n h } \Big ( \gamma ^ { 2 } m + \gamma \sqrt { \Gamma ( m ^ { 2 } + q ) } g + b \Big ) \Big ] , \qquad I _ { 2 } ( b ) = \mathbb { E } _ { g } \Big [ \operatorname { t a n h } ^ { 2 } \Big ( \gamma ^ { 2 } m + \gamma \sqrt { \Gamma ( m ^ { 2 } + q ) } g + b \Big ) \Big ] .
$$

4. Exact loss. Collecting equation 40, equation 42 and equation 45,

$$
\mathcal { L } ( c , m , q , b ) = \left( 1 + \beta ^ { 2 } c \right) ^ { 2 } + \alpha ^ { 2 } \beta ^ { 2 } \biggl [ c ^ { 2 } \left( 1 + \sigma ^ { 2 } \right) + \left( m ^ { 2 } + q \right) \left( k + 2 \left( 1 + c \Gamma \right) \left( 1 - k \right) \right) + 2 c m h \biggr ] ,\tag{46}
$$

exactly, for any $\Lambda ,$ ω and $b ,$ with h, k given by equation 45.

5. Exact gradient flow. The flow acts on $\mathbf { w } \in \mathbb { R } ^ { d } .$ , while equation 46 depends on it only through m and $q .$ From equation 33,

$$
\nabla _ { \mathbf { w } } m = \frac { \mu } { \alpha d } , \qquad \nabla _ { \mathbf { w } } q = \frac { 2 \mathbf { w } ^ { \perp } } { \alpha ^ { 2 } d } ,\tag{47}
$$

so that, projecting dw $\mathbf { \nabla } \cdot / \mathrm { d } \tau = - \nabla _ { \mathbf { w } } \mathcal { L }$ onto $\pmb { \mu }$ and onto $\mathbf { w } ^ { \perp }$ and using $\pmb { \mu } ^ { T } \mathbf { w } ^ { \perp } = 0 , \| \pmb { \mu } \| ^ { 2 } = d$ and $\| \mathbf { w } ^ { \perp } \| ^ { 2 } =$ $\alpha ^ { 2 } d q$

$$
{ \frac { \mathrm { d } c } { \mathrm { d } \tau } } = - { \frac { \partial { \mathcal { L } } } { \partial c } } , { \frac { \mathrm { d } m } { \mathrm { d } \tau } } = - { \frac { 1 } { \gamma ^ { 2 } } } { \frac { \partial { \mathcal { L } } } { \partial m } } , { \frac { \mathrm { d } q } { \mathrm { d } \tau } } = - { \frac { 4 q } { \gamma ^ { 2 } } } { \frac { \partial { \mathcal { L } } } { \partial q } } , { \frac { \mathrm { d } b } { \mathrm { d } \tau } } = - { \frac { \partial { \mathcal { L } } } { \partial b } } .\tag{48}
$$

The vector parameters $m , q$ inherit a factor $1 / \gamma ^ { 2 } = 1 / ( \alpha ^ { 2 } d )$ from the change of variables that the scalars $c , b$ do not. This asymmetry, and the fact that $\gamma ^ { 2 }$ is $O _ { d } ( d )$ in one regime and $O _ { d } ( 1 )$ in the other, is what drives everything below. Equations equation 46–equation 48 are the exact reduction of the problem. The two regimes are now obtained by scaling analysis, i.e. by evaluating h, k from equation 45 in the large d limit.

Regime II: $\Lambda = O _ { d } ( d )$ , i.e. $\alpha = O _ { d } ( 1 )$ . Here $\gamma ^ { 2 } = \alpha ^ { 2 } d = O _ { d } ( d )$ , so as soon as $m ^ { 2 } + q = O _ { d } ( 1 )$ the Gaussian term in equation 44 has standard deviation $\gamma \sqrt { \Gamma ( m ^ { 2 } + q ) } \ : = \ : O _ { d } ( \sqrt { d } ) \ :$ : the tanh argument diverges for almost every g and $T $ sign. Consequently $k  1 .$ , and the $O _ { d } ( 1 )$ bias b is swamped, so $h \to \bar { \mathbb { E } } [ \mathrm { s i g n } ( \nu + g ) ] = \mathrm { e r f } ( \nu / \sqrt { 2 } )$ with

$$
\nu = { \sqrt { \Lambda } } { \frac { m } { { \sqrt { m ^ { 2 } + q } } } } .\tag{49}
$$

Crucially $1 - k \to 0$ exponentially in γ (a Gaussian tail), so the term equation 42 is genuinely negligible here, and h is independent of b. Since the imbalance $\omega$ enters the exact score equation 6 only through the offset ${ \frac { 1 } { 2 } } \log { \frac { \omega } { 1 - \omega } }$ , and $\partial \mathcal { L } / \partial b  0 .$ , the bias never moves in this regime. The loss reduces to

$$
\mathcal { L } ( c , m , q ) = \left( 1 + \beta ^ { 2 } c \right) ^ { 2 } + \alpha ^ { 2 } \beta ^ { 2 } \left[ c ^ { 2 } \left( 1 + \sigma ^ { 2 } \right) + m ^ { 2 } + q + 2 c m \operatorname * { e r f } \left( \frac { \nu } { \sqrt { 2 } } \right) \right] ,\tag{50}
$$

and equation 48 closes on $( c , m , q )$ , with all the $\alpha ^ { 2 }$ prefactors cancelling against $1 / \gamma ^ { 2 }$ in the m, q equations:

$$
\frac { \mathrm { d } c } { \mathrm { d } \tau } = - 2 \beta ^ { 2 } \left( 1 + \beta ^ { 2 } c \right) - 2 \alpha ^ { 2 } \beta ^ { 2 } \left[ c \left( 1 + \sigma ^ { 2 } \right) + m \operatorname { e r f } \left( \frac { \nu } { \sqrt { 2 } } \right) \right] ,\tag{51}
$$

$$
\frac { \mathrm { d } m } { \mathrm { d } \tau } = - \frac { 2 \beta ^ { 2 } } { d } \left[ m + c \mathrm { e r f } \left( \frac \nu { \sqrt 2 } \right) + \sqrt { \frac 2 \pi } \frac { c q m \sqrt { \Lambda } } { \left( m ^ { 2 } + q \right) ^ { 3 / 2 } } e ^ { - \nu ^ { 2 } / 2 } \right] ,\tag{52}
$$

$$
\frac { \mathrm { d } q } { \mathrm { d } \tau } = - \frac { 4 \beta ^ { 2 } } { d } q \left[ 1 - \sqrt { \frac { 2 } { \pi } } \frac { c m ^ { 2 } \sqrt { \Lambda } } { \left( m ^ { 2 } + q \right) ^ { 3 / 2 } } e ^ { - \nu ^ { 2 } / 2 } \right] .\tag{53}
$$

The right-hand sides of the m and q equations are $O _ { d } ( 1 / d )$ while that of c is $O _ { d } ( 1 )$ : the dynamics splits into two phases.

Phase $1 , \tau = O _ { d } ( 1 )$ . Only c moves. By equation 34, $m _ { 0 } , q _ { 0 } = O _ { d } ( 1 / d )$ , so m er $\dot { \iota } ( \nu / \sqrt { 2 } )$ is negligible and

$$
\frac { \mathrm { d } c } { \mathrm { d } \tau } = - 2 \beta ^ { 2 } \left( \Gamma + \alpha ^ { 2 } \right) \left( c - c _ { 1 } ^ { * } \right) , \qquad c _ { 1 } ^ { * } = - \frac { 1 } { \Gamma + \alpha ^ { 2 } } = - \frac { d } { \mathbb { E } \| \mathbf { x } _ { t } \| ^ { 2 } } ,\tag{54}
$$

so c relaxes exponentially to the value that best fits the isotropic Gaussian approximation of $P _ { t . }$ , while $( m , q , b )$ sit at initialization.

Phase $2 , \tau = O _ { d } ( d )$ . Now m, q evolve while $c ,$ being d times faster, is slaved through $\partial _ { c } { \mathcal { L } } = 0 ,$ i.e. $c ^ { * } = - ( 1 + \alpha ^ { 2 } m \operatorname { e r f } ( \nu / \sqrt { 2 } ) ) / ( \Gamma + \alpha ^ { 2 } )$ . Note from equation 34 and equation 49 that $\nu = O _ { d } ( 1 )$ already at initialization in this regime: the drive $- c \mathrm { e r f } ( \nu / \bar { \sqrt { 2 } } ) > 0$ is present from the outset and m grows immediately, with no instability threshold to cross. Once ν  1 the error function saturates and, taking $m > 0$ without loss of generality, the flow linearizes,

$$
\frac { \mathrm { d } q } { \mathrm { d } \tau } = - \frac { 4 \beta ^ { 2 } } { d } q , \qquad \frac { \mathrm { d } m } { \mathrm { d } \tau } = - \frac { 2 \beta ^ { 2 } \Gamma } { d \left( \Gamma + \alpha ^ { 2 } \right) } \left( m - \frac 1 { \Gamma } \right) ,\tag{55}
$$

so $q \to 0$ and $m  1 / \Gamma$ on $\tau = O _ { d } ( d )$ . At the fixed point $\mathbf { w } = \alpha \pmb { \mu } / \Gamma$ and $c = - 1 / \Gamma$ , giving

$$
\mathbf { s } _ { \pmb { \theta } } ( \mathbf { x } , t ) \longrightarrow - \frac { \mathbf { x } } { \Gamma } + \frac { \alpha \pmb { \mu } } { \Gamma } \operatorname { t a n h } \biggl ( \frac { \alpha \pmb { \mu } ^ { T } \mathbf { x } } { \Gamma } + b _ { 0 } \biggr ) ,\tag{56}
$$

which is the exact score equation 6 of the target except for the frozen bias $b _ { 0 }$ in place of $\begin{array} { r } { b ^ { * } = \frac { 1 } { 2 } \log \frac { \omega } { 1 - \omega } } \end{array}$

Speciation time: $\Lambda = { \cal O } _ { d } ( 1 ) , { \bf i . e . } \alpha ^ { 2 } = { \cal O } _ { d } ( 1 / d ) $ . Now $\gamma ^ { 2 } = \Lambda \Gamma = O _ { d } ( 1 )$ and the tanh argument equation 44 stays $O _ { d } ( 1 )$ : no saturation, and the bias is no longer negligible. Two simplifications are available. First, $\dot { \alpha } ^ { 2 } = \dot { O _ { d } } ( 1 / d )  0$ forces $t \to t _ { \operatorname* { m a x } } ,$ hence $\beta  1$ and

$$
\Gamma = \alpha ^ { 2 } \sigma ^ { 2 } + \beta ^ { 2 } \longrightarrow 1 ,\tag{57}
$$

so we set $\Gamma = 1$ and $\gamma ^ { 2 } = \Lambda$ throughout. Second, by equation 34 the initial condition is now

$$
q _ { 0 } = \frac { 1 } { \gamma ^ { 2 } } = O _ { d } ( 1 ) , \qquad m _ { 0 } = O _ { d } \biggl ( \frac { 1 } { \gamma \sqrt { d } } \biggr ) = O _ { d } \biggl ( \frac { 1 } { \sqrt { d } } \biggr ) ,\tag{58}
$$

so m starts vanishingly small against an $O _ { d } ( 1 )$ value of q - the opposite of Regime II, where both were $O _ { d } ( 1 / d )$ . Since $\alpha ^ { 2 } \beta ^ { 2 } = \mathsf { \bar { O } } _ { d } ( 1 / d )$ , the loss equation 46 is dominated by $( 1 + \beta ^ { 2 } c ) ^ { 2 }$ and c relaxes on $\tau = O _ { d } ( 1 )$ to $c ^ { * } = - 1 / \Gamma + O _ { d } ( 1 / d ) = - 1 + O _ { d } ( 1 / d )$ , before $( m , q , b )$ move. Substituting $c = c ^ { * }$ makes the $( 1 + c \Gamma )$ term in equation 46 $O _ { d } ( 1 / d )$ , leaving $\mathcal { L } = \mathrm { c o n s t } + \alpha ^ { 2 } \beta ^ { 2 } \mathcal { F } + O _ { d } ( 1 / d ^ { 2 } )$ with

$$
{ \mathcal { F } } ( m , q , b ) = \mathbb { E } _ { g , s } { \Big [ } \left( m ^ { 2 } + q \right) T ^ { 2 } - 2 s m T { \Big ] } , \qquad T = \operatorname { t a n h } ( A ) , \qquad A = s \gamma ^ { 2 } m + \gamma { \sqrt { m ^ { 2 } + q } } g + b .\tag{59}
$$

Measuring training time in units of $\check { \tau } = \beta ^ { 2 } \tau / d ,$ the exact flow equation 48 becomes

$$
\frac { \mathrm { d } m } { \mathrm { d } \tilde { \tau } } = - \frac { \partial \mathcal { F } } { \partial m } , \qquad \frac { \mathrm { d } q } { \mathrm { d } \tilde { \tau } } = - 4 q \frac { \partial \mathcal { F } } { \partial q } , \qquad \frac { \mathrm { d } b } { \mathrm { d } \tilde { \tau } } = - \gamma ^ { 2 } \frac { \partial \mathcal { F } } { \partial b } ,\tag{60}
$$

where the $\alpha ^ { 2 }$ of the loss has cancelled against the $1 / \gamma ^ { 2 } = 1 / ( \alpha ^ { 2 } d )$ of equation 48 in the $m , q$ equations, while the b equation retains a factor $\gamma ^ { 2 } = \overset { \cdot } { O _ { d } } ( 1 )$ . All four parameters therefore move at comparable rates in $\check { \tau } , \mathrm { i } . \mathbf { e } . \tau = O _ { d } ( d )$

Linear stability of $m = b = 0$ . Expanding equation $5 9$ to second order in $( m , b )$ at fixed $q ,$ and introducing the Gaussian averages

$$
\Theta _ { 1 } ( q , \gamma ) = \mathbb { E } _ { g } \left[ \operatorname { t a n h } ^ { 2 } ( \gamma \sqrt { q } g ) \right] , \qquad \Theta _ { 2 } ( q , \gamma ) = \mathbb { E } _ { g } \left[ \left( 1 - \operatorname { t a n h } ^ { 2 } \right) \left( 1 - 3 \operatorname { t a n h } ^ { 2 } \right) \left( \gamma \sqrt { q } g \right) \right] ,\tag{61}
$$

one finds $\partial _ { m } \mathcal { F } = \partial _ { b } \mathcal { F } = 0 \mathrm { a t } m = b = 0$ , both integrands being odd. Writing

$$
\begin{array} { r l } & { \Psi ( q , \gamma ) = \Theta _ { 1 } + \gamma ^ { 2 } q \Theta _ { 2 } , } \\ & { \mu ( q , \gamma ) = 4 \gamma ^ { 2 } \left( 1 - \Theta _ { 1 } \right) - 2 \Theta _ { 1 } - 2 \gamma ^ { 2 } q \left( 1 + \gamma ^ { 2 } \right) \Theta _ { 2 } , } \\ & { \Xi ( q , \gamma ) = \left( 1 - \Theta _ { 1 } \right) - \gamma ^ { 2 } q \Theta _ { 2 } , } \end{array}\tag{62}
$$

the zeroth order of the expansion gives the autonomous relaxation of $q ,$

$$
\frac { \mathrm { d } q } { \mathrm { d } \check { \tau } } = - 4 q \Psi ( q , \gamma ) + { \cal O } ( m ^ { 2 } , b ^ { 2 } ) ,\tag{63}
$$

which decays monotonically from $q _ { 0 } = 1 / \gamma ^ { 2 }$ , while the first order gives the coupled linear system obeyed by the overlap and the bias at fixed $q ,$

$$
\frac { \mathrm { d } } { \mathrm { d } \tilde { \tau } } \binom { m } { b } = \mathbf { M } ( q , \gamma ) \binom { m } { b } , \qquad \mathbf { M } ( q , \gamma ) = \binom { \mu } { 2 \gamma ^ { 2 } \left( 2 \omega - 1 \right) \Xi } \quad - 2 \gamma ^ { 2 } q \Theta _ { 2 } \Bigr ) .\tag{64}
$$

The off-diagonal terms are proportional to $2 \omega - 1 \cdot$ : for balanced data the overlap and the bias decouple entirely, m growing at the rate $\mu ( q , \gamma )$ while b decays to 0. For $\omega \neq 1 / 2$ the two are coupled and leave the

origin together, the flow converging to $q = 0 , m = 1 / \Gamma$ and

$$
b \longrightarrow b ^ { * } = { \frac { 1 } { 2 } } \log { \frac { \omega } { 1 - \omega } } , \qquad \mathrm { i . e . } \qquad \rho \longrightarrow \omega .\tag{65}
$$

Contrary to Regime II, where the bias stays frozen at $b _ { 0 }$ and only the direction is recovered, training around the speciation time learns the mode direction and the imbalance, jointly and on the same $\tau = O _ { d } ( d )$ timescale.

## A.2.2 GMM with four modes: exact loss and gradient flow

This section is self-contained, and mirrors $\ S \mathrm { A } . 2 . 1$ for the hierarchical target, using the score model of the main text, equation 10.

Setup. The target is the block-structured quadrimodal mixture

$$
P _ { 0 } ( { \bf x } ) = \frac { 1 } { 4 } \sum _ { s _ { 1 } = \pm 1 } \sum _ { s _ { 2 } = \pm 1 } \mathcal { N } \big ( s _ { 1 } \pm s _ { 2 } \pmb { \mu } _ { 2 } , \sigma ^ { 2 } { \pmb { I } } _ { d } \big ) ,\tag{66}
$$

where $\pmb { \mu } _ { 1 }$ is supported on the first κd coordinates and $\pmb { \mu } _ { 2 }$ on the remaining $( 1 - \kappa ) d .$ . We write $\kappa _ { 1 } = \kappa ,$ $\kappa _ { 2 } = 1 - \kappa ,$ , both $O _ { d } ( 1 )$ , and normalize $\| \mu _ { i } \| ^ { 2 } = \kappa _ { i } d$ so that every coordinate of the means is $O _ { d } ( 1 )$ . The score model is

$$
\mathbf { s } _ { \theta = ( c , \mathbf { w } _ { 1 } , \mathbf { w } _ { 2 } ) } ( \mathbf { x } , t ) = c \mathbf { x } + \mathbf { w } _ { 1 } \operatorname { t a n h } \bigl ( \mathbf { w } _ { 1 } ^ { T } \mathbf { x } \bigr ) + \mathbf { w } _ { 2 } \operatorname { t a n h } \bigl ( \mathbf { w } _ { 2 } ^ { T } \mathbf { x } \bigr ) ,\tag{67}
$$

with $\mathbf { w } _ { i }$ supported on block $i ,$ so that $\mathbf { v } _ { 1 } ^ { T } \mathbf { w } _ { 2 } = 0$ and $\mathbf { w } _ { i } ^ { T } \pmb { \mu } _ { j } = 0$ for $i \neq j .$ . As in $\ S \mathrm { A } . 2 . 1$ we work at fixed noising time $t ,$ write $\alpha = \alpha ( t ) , \beta = \beta ( t ) , \Gamma = \alpha ^ { 2 } \sigma ^ { 2 } + \beta ^ { 2 } , \tilde { \Lambda } = \alpha ^ { 2 } d / \Gamma$ , and train by gradient flow on the single-time DSM loss $\begin{array} { r } { \mathcal { L } = \frac { 1 } { d } \mathbb { E } \| \beta \mathbf { s } _ { \theta } ( \mathbf { x } _ { t } , t ) + \pmb { \xi } \| ^ { 2 } } \end{array}$ with $\mathbf { x } _ { t } = \alpha \mathbf { x } _ { 0 } + \beta \pmb { \xi }$ . Note that equation $6 7$ carries no bias: each block of equation 66 is balanced, so there is no mode weight to learn, and the structural parameter is κ instead.

Summary statistics, block SNR, initialization. Per block we define, as in equation 11,

$$
m _ { i } = \frac { { \bf w } _ { i } ^ { T } \mu _ { i } } { \alpha \kappa _ { i } d } , \qquad q _ { i } = \frac { \| { \bf w } _ { i } ^ { \perp } \| ^ { 2 } } { \alpha ^ { 2 } \kappa _ { i } d } , \qquad \mathrm { s o ~ t h a t } \qquad \| { \bf w } _ { i } \| ^ { 2 } = \alpha ^ { 2 } \kappa _ { i } d \left( m _ { i } ^ { 2 } + q _ { i } \right) ,\tag{68}
$$

with $\mathbf { w } _ { i } ^ { \perp }$ the component of $\mathbf { w } _ { i }$ orthogonal to $\pmb { \mu _ { i } }$ inside block i. Because block i has $\kappa _ { i } d$ coordinates and $\| \mu _ { i } \| ^ { 2 } = \kappa _ { i } d ,$ the natural signal-to-noise ratio and tanh scale of block i are those of $\ S \mathrm { A } . 2 . 1$ with $d \to \kappa _ { i } d \colon$

$$
\Lambda ^ { ( i ) } = \frac { \alpha ^ { 2 } \| \pmb { \mu } _ { i } \| ^ { 2 } } { \Gamma } = \kappa _ { i } \Lambda , \qquad \gamma _ { i } ^ { 2 } : = \alpha ^ { 2 } \kappa _ { i } d = \kappa _ { i } \Lambda \Gamma = \kappa _ { i } \gamma ^ { 2 } .\tag{69}
$$

The block asymmetry κ enters only through this rescaling of the SNR. Initializing each ${ \bf w } _ { i } ( 0 )$ uniformly on the unit sphere of its own block gives, exactly as in equation 34,

$$
m _ { i } ( 0 ) = O _ { d } \left( \frac { 1 } { \alpha \kappa _ { i } d } \right) , \qquad q _ { i } ( 0 ) = \frac { 1 } { \gamma _ { i } ^ { 2 } } + O _ { d } \bigl ( { \textstyle \frac { 1 } { d } } \bigr ) , \qquad \mathrm { s o ~ t h a t } \qquad \gamma _ { i } \sqrt { q _ { i } ( 0 ) } = 1\tag{70}
$$

for both blocks, whatever $\kappa \cdot$ the two blocks start at the same point in the rescaled variable that controls the tanh, and differ only through $\gamma _ { i }$ . Together with (§A.2.3), everything below establishes the following statement, the long form of Result 2 of the main text.

Result (2, long form: training dynamics, hierarchical distribution). Let ${ \bf w } _ { 1 } ( 0 ) , { \bf w } _ { 2 } ( 0 )$ be independent and uniform on the unit sphere of their respective κd (resp. $( 1 - \kappa ) d )$ dimensional blocks, $c ( 0 ) = \hat { O _ { d } ( 1 ) }$ , and let $\left( c , \mathbf { w } _ { 1 } , \mathbf { w } _ { 2 } \right)$ evolve under GF equation 8 for the score model equation 10 at fixed t. Define $\kappa _ { 1 } = \kappa , \kappa _ { 2 } = 1 - \kappa ,$ each pair $( m _ { i } , q _ { i } )$ obeys the dynamics of Result 1 at $\omega = 1 / 2 , b = 0$ and rescaled SNR $\Lambda \to \kappa _ { i } \Lambda _ { . }$ , the two blocks being coupled only through the shared skip connection c. As d  :

(a) Regime II. On $\tau = O _ { d } ( 1 )$ times only c moves, relaxing to $c _ { 1 } ^ { * } = - 1 / ( \Gamma + \alpha ^ { 2 } )$ while both $( m _ { i } , q _ { i } )$ stay at initialization. On $\tau = O _ { d } ( d )$ times c is slaved to the overlaps, and $( m _ { 1 } , q _ { 1 } ) , ( m _ { 2 } , q _ { 2 } )$ both converge to $( 1 / \Gamma , 0 )$ , independently of $\kappa _ { i } \dot { . }$ the two mode directions $\pmb { \mu } _ { 1 } , \pmb { \mu } _ { 2 }$ are recovered together, on the same timescale, regardless of the block asymmetry.

(b) Speciation time $( \Lambda = O _ { d } ( 1 ) )$ . Here $\Gamma  1$ , each block carries its own SNR $\gamma _ { i } ^ { 2 } = \kappa _ { i } \Lambda \Gamma$ and starts from $\dot { q _ { i } } ( 0 ) = 1 / \gamma _ { i } ^ { 2 }$ . On $\tau = O _ { d } ( 1 )$ times $c  - 1 / \Gamma$ and stays therefor the remainder of the dynamics; with c thus frozen the two blocks decouple entirely, and since $\omega = 1 / 2$ makes ${ \bf M } ( q _ { i } , \gamma _ { i } )$ diagonal, each pair obeys the closed autonomous system

$$
\frac { \mathrm { d } q _ { i } } { \mathrm { d } \check { \tau } } = - 4 q _ { i } \Psi ( q _ { i } , \gamma _ { i } ) , \qquad \frac { \mathrm { d } m _ { i } } { \mathrm { d } \check { \tau } } = \mu ( q _ { i } , \gamma _ { i } ) m _ { i } + O ( m _ { i } ^ { 3 } ) ,\tag{71}
$$

with Ψ and the growth rate $\mu$ as given in equation $6 2 .$ . The orthogonal component $q _ { i }$ decreases monotonically on the same timescale as $m _ { i } ,$ , and µ is a strictly decreasing function of $q _ { i } \colon$ it is the decay of $q _ { i }$ that drives the overlap unstable. In the $q _ { i } \to 0$ limit, $\mu ( q _ { i } , \gamma _ { i } ) \sim 4 \gamma _ { i } ^ { 2 } \equiv 4 \kappa _ { i } \Lambda \colon$ the rate of block i is proportional to $\kappa _ { i }$ The two directions are learned on different timescales - for κ small the dominant direction $\pmb { \mu } _ { 2 }$ is learned on $\tau = O _ { d } ( d )$ , while the subdominant $\pmb { \mu } _ { 1 }$ lags by a factor $1 / \kappa ,$ on $\tau = O _ { d } ( d / \kappa )$

(c) Heuristic growth rate. The system equation 71 is two-dimensional, so at finite $\gamma _ { i }$ the rate $\mu ( q _ { i } , \gamma _ { i } )$ drifts as $q _ { i }$ relaxes. Assuming instead that $\| \mathbf { w } _ { i } \|$ is initialized and kept at the norm $\alpha \lVert \pmb { \mu } _ { i } \rVert$ of the target direction, i.e. $m _ { i } ^ { 2 } + q _ { i } = 1 .$ , eliminates $q _ { i }$ from equation $7 1$ and reduces the dynamics to a single ODE $\mathrm { d } m _ { i } / \mathrm { d } \check { \tau } = \lambda ( \gamma _ { i } ) m _ { i } + { \cal O } ( m _ { i } ^ { 3 } )$ , with the closed-form rate equation 104 of $\mathrm { \ S } A . 2 . 3 ,$

$$
\lambda ( \gamma ) = \mu ( 1 , \gamma ) + 2 \Psi ( 1 , \gamma ) .\tag{72}
$$

Note that this constraint is not exactly met by the initialization above, which gives $q _ { i } ( 0 ) = 1 / \gamma _ { i } ^ { 2 }$ rather than $q _ { i } ( 0 ) = 1 ;$ for $\kappa _ { i } = O _ { d } ( 1 )$ , however, the two differ only by the $O _ { d } ( 1 )$ factor $\gamma _ { i } ^ { 2 } = \kappa _ { i } \Lambda \Gamma ,$ , so both start the dynamics at the same order in $d .$ The reduction moreover recovers (b) exactly as $\kappa  0$ , since $\lambda ( \gamma ) = 4 \dot { \gamma } ^ { 2 } - 6 \gamma ^ { 4 } + O ( \gamma ^ { 6 } )$ . Away from that limit it is a heuristic, but a closed-form one, and we use $\tau _ { i } \propto 1 / \lambda ( \gamma _ { i } )$ as the reference timescale against which the measured dynamics are rescaled in $F i g . 2 .$

Detailed dynamics.

1. Exact loss. Decompose ${ \bf x } _ { 0 } = s _ { 1 } \mu _ { 1 } + s _ { 2 } \mu _ { 2 } + \sigma { \bf z }$ with $s _ { 1 } , s _ { 2 } = \pm 1$ independent and uniform, $\mathbf { z } \sim$ $\mathcal { N } ( 0 , \pmb { I } _ { d } )$ , and set $q _ { \mathbf { z } _ { i } } = \mathbf { w } _ { i } ^ { T } \mathbf { z } / ( \alpha \sqrt { \kappa _ { i } d ( m _ { i } ^ { 2 } + q _ { i } ) } )$ ) and $q _ { \pmb { \xi } _ { i } } = \mathbf { w } _ { i } ^ { T } \pmb { \xi } / ( \alpha \sqrt { \kappa _ { i } d ( m _ { i } ^ { 2 } + q _ { i } ) } ) ;$ the four are i.i.d. standard Gaussians because w $\mathbf { \Delta } _ { 1 } ^ { T } \mathbf { w } _ { 2 } = 0$ . Since $\mathbf { w } _ { i } ^ { T } \pmb { \mu } _ { j } = 0$ for $i \neq j$ , block i only ever sees $s _ { i } ,$ and

$$
\mathbf { w } _ { i } ^ { T } \mathbf { x } _ { 0 } = \alpha \left( s _ { i } m _ { i } \kappa _ { i } d + \sigma \sqrt { \kappa _ { i } d ( m _ { i } ^ { 2 } + q _ { i } ) } q _ { \mathbf { z } _ { i } } \right) , \qquad \mathbf { w } _ { i } ^ { T } \boldsymbol { \xi } = \alpha \sqrt { \kappa _ { i } d ( m _ { i } ^ { 2 } + q _ { i } ) } q _ { \pmb { \xi } _ { i } } .\tag{73}
$$

Inserting $\mathbf { x } _ { t } = \alpha \mathbf { x } _ { 0 } + \beta \pmb { \xi }$ into equation $6 7$ and regrouping,

$$
\beta \mathbf { s } _ { \theta } ( \mathbf { x } _ { t } , t ) + \pmb { \xi } = \alpha \beta c \mathbf { x } _ { 0 } + \left( 1 + \beta ^ { 2 } c \right) \pmb { \xi } + \beta \sum _ { i = 1 , 2 } T _ { i } \mathbf { w } _ { i } , \qquad T _ { i } : = \operatorname { t a n h } \bigl ( \mathbf { w } _ { i } ^ { T } \mathbf { x } _ { t } \bigr ) .\tag{74}
$$

Squaring and dividing by d: the cross term between the two blocks vanishes by $\begin{array} { r } { \mathbf { w } _ { 1 } ^ { T } \mathbf { w } _ { 2 } = 0 , \frac { 1 } { d } \mathbb { E } \| \mathbf { x } _ { 0 } \| ^ { 2 }  } \end{array}$ $1 + \sigma ^ { 2 }$ since $\| \pmb { \mu } _ { 1 } \| ^ { 2 } + \| \pmb { \mu } _ { 2 } \| ^ { 2 } = d ,$ and the terms in ${ q _ { \mathbf { z } _ { i } } } , { q _ { \xi } } ,$ carry an explicit $1 / { \sqrt { d } }$ which is again compensated by Stein’s lemma,

$$
\mathbb { E } [ q _ { \xi _ { i } } T _ { i } ] = \alpha \beta \sqrt { \kappa _ { i } d ( m _ { i } ^ { 2 } + q _ { i } ) } \mathbb { E } \big [ 1 - T _ { i } ^ { 2 } \big ] , \quad \mathbb { E } [ q _ { z _ { i } } T _ { i } ] = \alpha ^ { 2 } \sigma \sqrt { \kappa _ { i } d ( m _ { i } ^ { 2 } + q _ { i } ) } \mathbb { E } \big [ 1 - T _ { i } ^ { 2 } \big ] ,\tag{75}
$$

the two recombining through $1 + \beta ^ { 2 } c + \alpha ^ { 2 } \sigma ^ { 2 } c = 1 + c \Gamma$ . Finally, from equation $7 3$ and $\alpha \sqrt { \kappa _ { i } d } = \gamma _ { i }$ , the argument of the i-th tanh is

$$
\mathbf { w } _ { i } ^ { T } \mathbf { x } _ { t } = s _ { i } \gamma _ { i } ^ { 2 } m _ { i } + \gamma _ { i } \sqrt { \Gamma \left( m _ { i } ^ { 2 } + q _ { i } \right) } g _ { i } , \qquad g _ { i } \sim { \mathcal { N } } ( 0 , 1 ) ,\tag{76}
$$

and, $s _ { i }$ being uniform, averaging over it with $g _ { i }  s _ { i } g _ { i }$ removes $s _ { i }$ altogether. Collecting,

$$
\mathcal { L } ( c , m _ { 1 } , q _ { 1 } , m _ { 2 } , q _ { 2 } ) = \left( 1 + \beta ^ { 2 } c \right) ^ { 2 } + \alpha ^ { 2 } \beta ^ { 2 } \left[ c ^ { 2 } \left( 1 + \sigma ^ { 2 } \right) + \sum _ { i = 1 , 2 } \kappa _ { i } \mathcal { G } _ { i } \right] ,\tag{77}
$$

$$
\mathcal { G } _ { i } = \left( m _ { i } ^ { 2 } + q _ { i } \right) \left( k _ { i } + 2 \left( 1 + c \Gamma \right) \left( 1 - k _ { i } \right) \right) + 2 c m _ { i } h _ { i } ,\tag{78}
$$

with the one-dimensional Gaussian averages

$$
h _ { i } = \mathbb { E } _ { g } [ \operatorname { t a n h } A _ { i } ] , \quad k _ { i } = \mathbb { E } _ { g } \left[ \operatorname { t a n h } ^ { 2 } A _ { i } \right] , \quad A _ { i } = \gamma _ { i } ^ { 2 } m _ { i } + \gamma _ { i } \sqrt { \Gamma \left( m _ { i } ^ { 2 } + q _ { i } \right) } g .\tag{79}
$$

Comparing with equation 46: each $\mathcal { G } _ { i }$ is exactly the bracket of the unbalanced problem at $\omega = 1 / 2$ and $b = 0 ,$ , with $\gamma \to \gamma _ { i }$ . The hierarchical target is therefore two independent copies of the single-mode problem at rescaled SNR $\kappa _ { i } \Lambda ,$ , weighted by $\kappa _ { i }$ and coupled only through the shared skip connection c.

2. Exact gradient flow. With $\nabla _ { \mathbf { w } _ { i } } m _ { i } = \mu _ { i } / ( \alpha \kappa _ { i } d )$ and $\nabla _ { { \bf w } _ { i } } q _ { i } = 2 { \bf w } _ { i } ^ { \perp } / ( \alpha ^ { 2 } \kappa _ { i } d )$ , projecting dw $_ { i } / \mathrm { d } \tau =$ $- \nabla _ { \mathbf { w } _ { i } } \mathcal { L }$ as in equation 47 gives

$$
\frac { \mathrm { d } c } { \mathrm { d } \tau } = - \frac { \partial \mathcal { L } } { \partial c } , \qquad \frac { \mathrm { d } m _ { i } } { \mathrm { d } \tau } = - \frac { 1 } { \gamma _ { i } ^ { 2 } } \frac { \partial \mathcal { L } } { \partial m _ { i } } = - \frac { \alpha ^ { 2 } \beta ^ { 2 } \kappa _ { i } } { \gamma _ { i } ^ { 2 } } \frac { \partial \mathcal { G } _ { i } } { \partial m _ { i } } , \qquad \frac { \mathrm { d } q _ { i } } { \mathrm { d } \tau } = - \frac { 4 q _ { i } } { \gamma _ { i } ^ { 2 } } \frac { \partial \mathcal { L } } { \partial q _ { i } } = - \frac { 4 q _ { i } \alpha ^ { 2 } \beta ^ { 2 } \kappa _ { i } } { \gamma _ { i } ^ { 2 } } \frac { \partial \mathcal { G } _ { i } } { \partial q _ { i } } .\tag{80}
$$

Using $\gamma _ { i } ^ { 2 } = \alpha ^ { 2 } \kappa _ { i } d$ and the rescaled training time $\check { \tau } = \beta ^ { 2 } \tau / d$ yields

$$
\frac { \mathrm { d } m _ { i } } { \mathrm { d } \check { \tau } } = - \frac { \partial \mathcal { G } _ { i } } { \partial m _ { i } } , \qquad \frac { \mathrm { d } q _ { i } } { \mathrm { d } \check { \tau } } = - 4 q _ { i } \frac { \partial \mathcal { G } _ { i } } { \partial q _ { i } } .\tag{81}
$$

The block asymmetry acts on the dynamics exclusively through $\gamma _ { i } ^ { 2 } = \kappa _ { i } \Lambda \Gamma$ inside $\mathcal { G } _ { i } ,$ i.e. purely as a rescaling of the block SNR.

Regime II: $\Lambda = O _ { d } ( d )$ Then $\gamma _ { i } ^ { 2 } = \kappa _ { i } \Lambda \Gamma = O _ { d } ( d )$ for both blocks, the argument equation 76 diverges, $T _ { i } $ sign, $k _ { i } \to 1$ with $1 - k _ { i }$ exponentially small, and $h _ { i }  \mathrm { e r f } ( \nu _ { i } / \sqrt { 2 } )$ with $\nu _ { i } = \sqrt { \kappa _ { i } \Lambda } m _ { i } / \sqrt { m _ { i } ^ { 2 } + q _ { i } }$ The loss becomes

$$
\mathcal { L } = \left( 1 + \beta ^ { 2 } c \right) ^ { 2 } + \alpha ^ { 2 } \beta ^ { 2 } \left[ c ^ { 2 } \left( 1 + \sigma ^ { 2 } \right) + \sum _ { i = 1 , 2 } \kappa _ { i } \left( m _ { i } ^ { 2 } + q _ { i } + 2 c m _ { i } \operatorname { e r f } \left( \frac { \nu _ { i } } { \sqrt { 2 } } \right) \right) \right] ,\tag{82}
$$

and equation 80 gives

$$
\frac { \mathrm { d } c } { \mathrm { d } \tau } = - 2 \beta ^ { 2 } \left( 1 + \beta ^ { 2 } c \right) - 2 \alpha ^ { 2 } \beta ^ { 2 } \left[ c \left( 1 + \sigma ^ { 2 } \right) + \sum _ { i } \kappa _ { i } m _ { i } \operatorname { e r f } \left( \frac { \nu _ { i } } { \sqrt { 2 } } \right) \right] ,\tag{83}
$$

$$
\frac { \mathrm { d } m _ { i } } { \mathrm { d } \tau } = - \frac { 2 \beta ^ { 2 } } { d } \left[ m _ { i } + c \mathrm { e r f } \left( \frac { \nu _ { i } } { \sqrt { 2 } } \right) + \sqrt { \frac { 2 } { \pi } } \frac { c q _ { i } m _ { i } \sqrt { \kappa _ { i } \Lambda } } { \left( m _ { i } ^ { 2 } + q _ { i } \right) ^ { 3 / 2 } } e ^ { - \nu _ { i } ^ { 2 } / 2 } \right] ,\tag{84}
$$

$$
\frac { \mathrm { d } q _ { i } } { \mathrm { d } \tau } = - \frac { 4 \beta ^ { 2 } } { d } q _ { i } \left[ 1 - \sqrt { \frac { 2 } { \pi } } \frac { c m _ { i } ^ { 2 } \sqrt { \kappa _ { i } \Lambda } } { \left( m _ { i } ^ { 2 } + q _ { i } \right) ^ { 3 / 2 } } e ^ { - \nu _ { i } ^ { 2 } / 2 } \right] .\tag{85}
$$

Again the c equation is $O _ { d } ( 1 )$ and the others $O _ { d } ( 1 / d )$ , giving two phases. On $\tau = O _ { d } ( 1 )$ , with $m _ { i } , q _ { i } =$ $\bar { O _ { d } } ( 1 / d )$ at initialization, c relaxes to

$$
c _ { 1 } ^ { * } = - \frac { 1 } { \Gamma + \alpha ^ { 2 } }\tag{86}
$$

while both $( m _ { i } , q _ { i } )$ stay put. On $\tau = O _ { d } ( d )$ , c is slaved through $\begin{array} { r } { \partial _ { c } \mathcal { L } = 0 , \mathrm { i . e . } c ^ { * } = - \left( 1 + \alpha ^ { 2 } \sum _ { j } \kappa _ { j } m _ { j } \mathrm { e r f } ( \nu _ { j } / \sqrt { 2 } ) \right) / ( \Gamma + } \end{array}$ $\alpha ^ { 2 } )$ and linearizing

$$
\frac { \mathrm { d } q _ { i } } { \mathrm { d } \tau } = - \frac { 4 \beta ^ { 2 } } { d } q _ { i } , \qquad \frac { \mathrm { d } m _ { i } } { \mathrm { d } \tau } = - \frac { 2 \beta ^ { 2 } } { d } \left[ m _ { i } - \frac { 1 + \alpha ^ { 2 } \sum _ { j } \kappa _ { j } m _ { j } } { \Gamma + \alpha ^ { 2 } } \right] .\tag{87}
$$

Both blocks obey the same dynamics and converge together. Using $\textstyle \sum _ { j } \kappa _ { j } \ = \ 1$ , the fixed point is $m _ { 1 } = m _ { 2 } = 1 / \Gamma , q _ { i } = 0 , c = - 1 / \Gamma$ , whence $\mathbf { w } _ { i } = \alpha \pmb { \mu } _ { i } / \Gamma$ and

$$
{ \bf s } _ { \theta } ( { \bf x } , t ) \longrightarrow - \frac { { \bf x } } { \Gamma } + \sum _ { i = 1 , 2 } \frac { \alpha \pmb { \mu } _ { i } } { \Gamma } \operatorname { t a n h } \left( \frac { \alpha \mu _ { i } ^ { T } \mathbf { x } } { \Gamma } \right) ,\tag{88}
$$

exactly the target score equation $7 .$ In Regime II both mode directions are recovered on the same $\tau = O _ { d } ( d )$ timescale, independently of κ: training at high SNR is blind to the block asymmetry.

Speciation time: $\Lambda = O _ { d } ( 1 )$ . Now $\gamma _ { i } ^ { 2 } = \kappa _ { i } \Lambda \Gamma = O _ { d } ( 1 )$ , and as in $\ S \mathrm { A } . 2 . 1 \alpha ^ { 2 } = { \cal O } _ { d } ( 1 / d )$ forces $\Gamma  1$ so $\gamma _ { i } ^ { 2 } \to \kappa _ { i } \Lambda$ . The skip connection relaxes on $\tau = O _ { d } ( 1 )$ to $c ^ { * } = - 1 + O _ { d } ( 1 / d ) .$ ; substituting it kills the $( 1 + c \Gamma )$ term in equation 78 and leaves $\mathcal { G } _ { i } = \mathcal { F } _ { i } + O _ { d } ( 1 / d )$ with

$$
{ \mathcal F } _ { i } ( m _ { i } , q _ { i } ) = { \mathbb E } _ { g } \left[ \left( m _ { i } ^ { 2 } + q _ { i } \right) \operatorname { t a n h } ^ { 2 } A _ { i } - 2 m _ { i } \operatorname { t a n h } A _ { i } \right] , \qquad A _ { i } = \gamma _ { i } ^ { 2 } m _ { i } + \gamma _ { i } \sqrt { m _ { i } ^ { 2 } + q _ { i } } \ g .\tag{89}
$$

With c frozen, the two blocks are now completely decoupled: by equation 81 each pair $( m _ { i } , q _ { i } )$ follows its own autonomous flow, on the timescale

$$
\begin{array} { r } { \check { \tau } = O _ { d } ( 1 ) \qquad \Longleftrightarrow \qquad \tau = O _ { d } ( d ) . } \end{array}\tag{90}
$$

Expanding equation 89 in $m _ { i }$ at fixed $q _ { i }$ (the $b = 0 , \omega = 1 / 2 )$ , and writing $\Theta _ { 1 , 2 }$ for the Gaussian averages equation 61 evaluated at $( q _ { i } , \gamma _ { i } )$

$$
\frac { \mathrm { d } q _ { i } } { \mathrm { d } \check { \tau } } = - 4 q _ { i } \Psi ( q _ { i } , \gamma _ { i } ) + O ( m _ { i } ^ { 2 } ) , \qquad \frac { \mathrm { d } m _ { i } } { \mathrm { d } \check { \tau } } = \mu ( q _ { i } , \gamma _ { i } ) m _ { i } + O ( m _ { i } ^ { 3 } ) ,\tag{91}
$$

with $\mu$ as in equation 62. Since, $\Theta _ { 1 } \to 0$ and $\Theta _ { 2 } \to 1 \mathrm { a s } q _ { i } \to 0 , \mu$ increases monotonically along the flow towards

$$
\mu ( q _ { i } \to 0 , \gamma _ { i } ) = 4 \gamma _ { i } ^ { 2 } = 4 \kappa _ { i } \Lambda \Gamma \ \xrightarrow [ \Gamma \to 1 ] { } \ 4 \kappa _ { i } \Lambda :\tag{92}
$$

the asymptotic growth rate of block i is linear in $\kappa _ { i }$ . We expect the escape time of $m _ { i }$ to therefore scale as $1 / \kappa _ { i }$ in $\check { \tau } ,$ i.e.

$$
\tau _ { i } = O _ { d } \biggl ( \frac { d } { \kappa _ { i } } \biggr ) .\tag{93}
$$

Taking κ small, so that $\kappa _ { 1 } = \kappa \ll 1$ and $\kappa _ { 2 } = 1 - \kappa \simeq 1$ , the dominant direction $\pmb { \mu } _ { 2 }$ is learned on $\tau = O _ { d } ( d )$ while the subdominant direction $\pmb { \mu } _ { 1 }$ lags by a factor $1 / \kappa ,$ on $\tau = O _ { d } ( d / \kappa )$ . Once escaped, each block converges to $q _ { i } = 0 , m _ { i } = 1 / \Gamma$ , recovering equation 88.

## A.2.3 Fixed-norm ansatz: a one-dimensional reduction at the speciation time

The analyses of §A.2.1 and §A.2.2 involve two coupled order parameters, the overlap m and the orthogonal norm $q ,$ whose joint relaxation makes the growth rate of m depend on the instantaneous value of $q .$ We show here that constraining the weight vector to a fixed norm removes q entirely, reduces the dynamics to a single scalar ODE, and yields a closed-form growth rate at small overlap m.

Fixed norm constraint. We keep the notation of §A.2.1: score model equation 30, order parameters equation 33, and $\gamma ^ { 2 } = \alpha ^ { 2 } d = \Lambda \Gamma$ . We now impose that w retain throughout training the norm of the target direction,

$$
\| { \bf w } \| = \alpha \sqrt { d } \qquad \Longleftrightarrow \qquad m ^ { 2 } + q = 1 ,\tag{94}
$$

using $\| \mathbf { w } \| ^ { 2 } = \alpha ^ { 2 } d ( m ^ { 2 } + q )$ . This is the natural scale: the exact score equation 6 is reproduced at $\mathbf { w } ^ { * } = { \alpha \pmb { \mu } } / \Gamma$ , of norm $\alpha \sqrt { d } / \Gamma$ . We place ourselves at the speciation time, $\Lambda = O _ { d } ( 1 )$ , and set $\Gamma = 1$ throughout, so that $\gamma ^ { 2 } = \Lambda$

The loss loses its dependence on q. The whole $( m , q )$ dependence of the loss enters through the law equation 44 of the tanh argument, whose mean is m and whose standard deviation is $\gamma \sqrt { \Gamma ( m ^ { 2 } + q ) }$ Under equation 94 the latter is constant,

$$
\begin{array} { r } { \mathbf { w } ^ { T } \mathbf { x } _ { t } + b = s \gamma ^ { 2 } m + \gamma g + b , \qquad g \sim \mathcal { N } ( 0 , 1 ) , } \end{array}\tag{95}
$$

so the constraint freezes the width of the Gaussian and lets only its mean move with $m .$ Writing $\mathcal { F }$ for the reduced loss equation 59 and substituting $q = 1 - m ^ { 2 }$ , the prefactor $( m ^ { 2 } + q )$ becomes 1 and

$$
\Phi ( m , b ) : = \mathcal { F } \big ( m , 1 - m ^ { 2 } , b \big ) = k ( m , b ) - 2 m h ( m , b ) ,\tag{96}
$$

with $h .$ , k the Gaussian averages equation 45 evaluated on equation 95. For the balanced case $( \omega = 1 / 2 ,$ $b = 0 )$ relevant to each block of $( \bar { \mathcal { D } _ { 2 } } )$ these reduce to

$$
h ( m ) = \mathbb { E } _ { g } \left[ \operatorname { t a n h } \bigl ( \gamma ^ { 2 } m + \gamma g \bigr ) \right] , \qquad k ( m ) = \mathbb { E } _ { g } \left[ \operatorname { t a n h } ^ { 2 } \bigl ( \gamma ^ { 2 } m + \gamma g \bigr ) \right] .\tag{97}
$$

The problem is now one-dimensional: a single scalar $m ,$ and a single parameter $\gamma .$

Constrained gradient flow. Restricting the flow to the sphere $\| \mathbf { w } \| = \alpha { \sqrt { d } }$ means projecting out the radial component, dw $\mathbf { \Psi } / \mathrm { d } \tau = - \left( I - \hat { \mathbf { w } } \hat { \mathbf { w } } ^ { T } \right) \nabla _ { \mathbf { w } } \mathcal { L }$ with $\hat { \textbf { w } } = \mathbf { w } / \| \mathbf { w } \|$ . Using $\nabla _ { \mathbf { w } } m = \pmb { \mu } / ( \alpha d )$ and $\nabla _ { \mathbf { w } } q = 2 \mathbf { w } ^ { \perp } / ( \alpha ^ { 2 } d )$ from equation 47, together with $\begin{array} { r } { \mathbf { w } ^ { T } \nabla _ { \mathbf { w } } \mathcal { L } = m \partial _ { m } \mathcal { L } + 2 q \partial _ { q } \mathcal { L } , } \end{array}$ , one finds

$$
\frac { \mathrm { d } m } { \mathrm { d } \tau } = - \frac { 1 } { \gamma ^ { 2 } } \Big [ \left( 1 - m ^ { 2 } \right) \partial _ { m } \mathcal { L } - 2 m q \partial _ { q } \mathcal { L } \Big ] = - \frac { 1 - m ^ { 2 } } { \gamma ^ { 2 } } \Big [ \partial _ { m } \mathcal { L } - 2 m \partial _ { q } \mathcal { L } \Big ] ,\tag{98}
$$

the second equality using $q = 1 - m ^ { 2 }$ . The bracket is precisely the total derivative of the loss along the constraint surface, $\Phi ^ { \prime } ( m ) \bar { = } \partial _ { m } \mathcal { F } - 2 m \partial _ { q } \mathcal { F }$ . Since ${ \mathcal { L } } =$ cons $\bar { \cdot } + \alpha ^ { 2 } \beta ^ { 2 } \mathcal { F } + O _ { d } ( 1 / d ^ { 2 } )$ and $\alpha ^ { 2 } \beta ^ { 2 } / \gamma ^ { 2 } = \breve { \beta } ^ { 2 } / d ,$ the flow closes on m alone in the rescaled time $\check { \tau } = \beta ^ { 2 } \tau / d$ of $\ S \mathrm { A } . 2 . 1$

$$
\frac { \mathrm { d } m } { \mathrm { d } \check { \tau } } = - \left( 1 - m ^ { 2 } \right) \Phi ^ { \prime } ( m ) ,\tag{99}
$$

so that the dynamics again unfolds on $\tau = O _ { d } ( d )$

Linearized dynamics. Everything is controlled by the same two Gaussian integrals equation 61 as in the unconstrained case, now evaluated at $q = 1$ since the width in equation 95 is fixed:

$$
\theta _ { 1 } ( \gamma ) : = \Theta _ { 1 } ( 1 , \gamma ) = \mathbb { E } _ { g } \big [ \operatorname { t a n h } ^ { 2 } ( \gamma g ) \big ] ,\tag{100}
$$

$$
\theta _ { 2 } ( \gamma ) : = \Theta _ { 2 } ( 1 , \gamma ) = \mathbb { E } _ { g } \big [ \big ( 1 - \operatorname { t a n h } ^ { 2 } ( \gamma g ) \big ) \big ( 1 - 3 \operatorname { t a n h } ^ { 2 } ( \gamma g ) \big ) \big ] .\tag{101}
$$

Expanding equation 97 in m around 0 and using that tanh is odd $( \operatorname { s o } \mathbb { E } _ { g } [ \operatorname { t a n h } ( \gamma g ) ] = \mathbb { E } _ { g } [ \operatorname { t a n h } \operatorname { t a n h } ^ { \prime } ( \gamma g ) ] =$ 0),

$$
h ( m ) = \gamma ^ { 2 } \left( 1 - \theta _ { 1 } \right) m + O ( m ^ { 3 } ) , \qquad k ( m ) = \theta _ { 1 } + \gamma ^ { 4 } \theta _ { 2 } m ^ { 2 } + O ( m ^ { 4 } ) ,\tag{102}
$$

so that Φ is even in m and

$$
\Phi ( m ) = \theta _ { 1 } - \frac { \lambda ( \gamma ) } { 2 } m ^ { 2 } + O ( m ^ { 4 } ) , \qquad \lambda ( \gamma ) = 4 \gamma ^ { 2 } \left( 1 - \theta _ { 1 } ( \gamma ) \right) - 2 \gamma ^ { 4 } \theta _ { 2 } ( \gamma ) .\tag{103}
$$

Inserting into equation 99, the overlap grows exponentially from its $O _ { d } ( 1 / \sqrt { d } )$ initial value,

$$
\frac { \mathrm { d } m } { \mathrm { d } \tilde { \tau } } = \lambda ( \gamma ) m + { \cal O } ( m ^ { 3 } ) .\tag{104}
$$

One can check that $\lambda ( \gamma ) > 0$ for all $\gamma > 0$ and, unlike the unconstrained rate $\mu ( q , \gamma )$ of equation $6 2 ,$ which drifts as q relaxes, λ is a fixed number once γ is chosen.

Small $\gamma$ and consistency with §A.2.2. For $\gamma \ll 1 .$ , tanh $( \gamma g ) = \gamma g - { \textstyle \frac { 1 } { 3 } } \gamma ^ { 3 } g ^ { 3 } + O ( \gamma ^ { 5 } )$ gives $\theta _ { 1 } = \gamma ^ { 2 } - 2 \gamma ^ { 4 } +$ $O ( \gamma ^ { 6 } )$ and $\theta _ { 2 } = 1 - 4 \gamma ^ { 2 } + { \cal O } ( \gamma ^ { 4 } )$ , hence

$$
\lambda ( \gamma ) = 4 \gamma ^ { 2 } \left( 1 - \gamma ^ { 2 } \right) - 2 \gamma ^ { 4 } + { \cal O } ( \gamma ^ { 6 } ) = 4 \gamma ^ { 2 } - 6 \gamma ^ { 4 } + { \cal O } ( \gamma ^ { 6 } ) .\tag{105}
$$

Applied blockwise to $\left( \mathcal { D } _ { 2 } \right)$ , where $\gamma _ { i } ^ { 2 } = \kappa _ { i } \Lambda \Gamma \to \kappa _ { i } \Lambda$ by equation 69, this gives

$$
\lambda ( \gamma _ { i } ) \simeq 4 \gamma _ { i } ^ { 2 } = 4 \kappa _ { i } \Lambda ,\tag{106}
$$

identical to the limiting rate equation 92 obtained there without the constraint.

Note that the two descriptions do not coincide away from this limit. For instance, a direct comparison at $q = 1$ gives the exact identity

$$
\lambda ( \gamma ) - \mu ( 1 , \gamma ) = 2 \left( \theta _ { 1 } + \gamma ^ { 2 } \theta _ { 2 } \right) = 2 \Psi ( 1 , \gamma ) > 0 .\tag{107}
$$

## A.3 Integrated loss: SNR reparametrization and effective weightings

The ResNet experiments of Fig. 3 do not compare two distinct pipelines. Both panels train the same denoiser, on the same denoising score matching objective equation 5 and the same noising schedule; the two curves differ only through the weighting function $w ( \bar { t } )$ , taken either uniform or chosen so as to reproduce the allocation of noise levels that Flow Matching performs implicitly. This appendix makes that construction precise. We first show that, once expressed in the signal-to-noise ratio, any stochastic interpolant pipeline is characterized by a single effective SNR weighting $w _ { \mathrm { e f f } } ( \Lambda )$ , so that two pipelines sharing it train identically whatever their time parametrization. We then compute $w _ { \mathrm { e f f } }$ in closed form for Diffusion and Flow Matching; the weighting w(t) that installs the latter inside a DSM run is derived in §A.4, and is the one used in Fig. 3.

Setup. In practice a single network represents the denoiser at every noise level at once, and is trained on the loss integrated over the noising time with a weighting function $w ( t )$

$$
\mathcal { L } ( \pmb { \theta } ) = \frac { 1 } { d } \int _ { 0 } ^ { t _ { \mathrm { m a x } } } \mathrm { d } t w ( t ) \mathbb { E } _ { \mathbf { x } _ { 0 } , \pmb { \xi } } \left\| \pmb { \xi } - \hat { \pmb { \xi } } _ { \pmb { \theta } } \left( \alpha ( t ) \mathbf { x } _ { 0 } + \beta ( t ) \pmb { \xi } , t \right) \right\| ^ { 2 } .\tag{108}
$$

Here $\hat { \xi } _ { \pmb { \theta } } : = - \beta ( t ) \mathbf { s } _ { \pmb { \theta } }$ is the noise prediction associated with the score model, so that equation 108 is identically the loss $\| \beta \mathbf { s } _ { \pmb { \theta } } + \pmb { \xi } \| ^ { 2 }$ of equation 5 used in the single time analysis: the two are the same residual written in two parametrizations, for every $t ,$ with no approximation on $\beta .$ . The single SNR results of $\ S _ { \mathrm { A } . 2 . 1 - \ S _ { \mathrm { A } . 2 . 2 } }$ may therefore be inserted into equation 108 as they stand.

SNR reparametrization. Two pipelines with different $( \alpha , \beta , w )$ may present the network with the same collection of denoising problems, since what a training example teaches depends on its noise level only through the SNR

$$
\Lambda _ { t } = \frac { \alpha ^ { 2 } ( t ) d } { \alpha ^ { 2 } ( t ) \sigma ^ { 2 } + \beta ^ { 2 } ( t ) } ,\tag{109}
$$

which, assuming $\beta / \alpha$ monotonically increases, decreases monotonically from $\Lambda = d / \sigma ^ { 2 }$ at $t = 0$ to $\Lambda = 0$ at $t = t _ { \mathrm { m a x } } ( \ S \mathrm { A } . 1 )$ . Being monotone, $t \mapsto \Lambda _ { t }$ is invertible and may be used as the integration variable in equation 108,

$$
\mathcal { L } ( \pmb { \theta } ) = \frac { 1 } { d } \int _ { 0 } ^ { d / \sigma ^ { 2 } } \mathrm { d } \Lambda ~ w _ { \mathrm { e f f } } ( \Lambda ) ~ \mathbb { E } _ { \mathbf { x } _ { 0 } , \pmb { \xi } } \left\| \pmb { \xi } - \hat { \pmb { \xi } } _ { \pmb { \theta } } \right\| ^ { 2 } , \qquad w _ { \mathrm { e f f } } ( \Lambda ) = w \big ( t ( \Lambda ) \big ) \left| \frac { \mathrm { d } t } { \mathrm { d } \Lambda } \right| .\tag{110}
$$

We call $w _ { \mathrm { e f f } }$ the effective SNR weighting of the scheme. Two pipelines with the same $w _ { \mathrm { e f f } }$ train identically whatever their time parametrizations; conversely, the same w produces very different $w _ { \mathrm { e f f } }$ for different interpolants. All the comparisons below are therefore comparisons of $w _ { \mathrm { e f f } }$

Flow Matching as a reweighted denoising loss. Diffusion regresses the noise directly, so equation 108 applies verbatim. Flow Matching instead regresses the velocity field along the linear interpolant $\mathbf { x } _ { t } =$

$( 1 - t ) \mathbf { x } _ { 0 } + t \pmb { \xi }$ , whose target is $\dot { \mathbf { x } } _ { t } = \pmb { \xi } - \mathbf { x } _ { 0 }$

$$
\mathcal { L } _ { \mathrm { F M } } = \int _ { 0 } ^ { 1 } \mathrm { d } t w ( t ) \mathbb { E } \left\| \hat { \pmb v } - ( { \pmb \xi } - { \bf x } _ { 0 } ) \right\| ^ { 2 } .\tag{111}
$$

Eliminating $\mathbf { x } _ { 0 } = ( \mathbf { x } _ { t } - t \pmb { \xi } ) / ( 1 - t )$ gives $\pmb { \xi } - \mathbf { x } _ { 0 } = ( \pmb { \xi } - \mathbf { x } _ { t } ) / ( 1 - t )$ , so that with the change of variables $\begin{array} { r } { \hat { \pmb { \xi } } = ( 1 - t ) \hat { \pmb { v } } + { \bf x } _ { t } , } \end{array}$

$$
\mathcal { L } _ { \mathrm { F M } } = \int _ { 0 } ^ { 1 } \mathrm { d } t \ \frac { w ( t ) } { ( 1 - t ) ^ { 2 } } \mathbb { E } \left\| \hat { \pmb { \xi } } - { \pmb { \xi } } \right\| ^ { 2 } .\tag{112}
$$

Flow Matching is thus a denoising loss in disguise, but one that silently reweights the noise levels by $1 / ( 1 - t ) ^ { 2 } = 1 / \bar { \alpha } ^ { 2 } ( t )$ . This Jacobian is not a cosmetic detail: since $\alpha  0$ at the noisy end, it is precisely what will concentrate FM on the low-SNR region below.

Effective weightings of DM and FM. We take $w \equiv 1 ;$ a non-uniform w simply multiplies the results by $w ( t ( \Lambda ) )$

For $\mathrm { D M } , \alpha = e ^ { - t }$ and $\beta = { \sqrt { 1 - e ^ { - 2 t } } } ,$ so $\Gamma _ { t } = 1 - e ^ { - 2 t } ( 1 - \sigma ^ { 2 } )$ and equation 109 inverts as $e ^ { - 2 t } =$ $\Lambda / ( d + ( 1 - \sigma ^ { 2 } ) \Lambda )$ . Differentiating $t = - { \textstyle { \frac { 1 } { 2 } } } \log e ^ { - 2 t }$

$$
w _ { \mathrm { e f f } } ^ { \mathrm { D M } } ( \Lambda ) = \frac { d } { 2 \Lambda \left( d + ( 1 - \sigma ^ { 2 } ) \Lambda \right) } \underset { \Lambda \ll d } { \simeq } \frac { 1 } { 2 \Lambda } .\tag{113}
$$

For FM, $\alpha = 1 - t$ and $\beta = t ,$ so $\Gamma _ { t } = ( 1 - t ) ^ { 2 } \sigma ^ { 2 } + t ^ { 2 }$ and, writing $r = ( 1 - t ) / t ,$ equation 109 reads $\Lambda = r ^ { 2 } d / ( 1 + r ^ { 2 } \sigma ^ { 2 } )$ , i.e. $r = \sqrt { \Lambda / ( d - \sigma ^ { 2 } \Lambda ) }$ . Including the Jacobian of equation 112,

$$
w _ { \mathrm { e f f } } ^ { \mathrm { F M } } ( \Lambda ) = \frac { d } { 2 \Lambda ^ { 3 / 2 } \sqrt { d - \sigma ^ { 2 } \Lambda } ~ \stackrel { \sim } { \sim } _ { d } ~ \frac { \sqrt { d } } { 2 \Lambda ^ { 3 / 2 } } } .\tag{114}
$$

<sub>The</sub> <sub>two</sub> <sub>differ</sub> <sub>by</sub> <sub>a</sub> <sub>factor</sub> √<sub>Λ:</sub> $w _ { \mathrm { e f f } } ^ { \mathrm { F M } } / w _ { \mathrm { e f f } } ^ { \mathrm { D M } } \simeq \sqrt { d / \Lambda }$ for $\Lambda \ll d ,$ so FM up-weights low SNR ever more strongly as Λ decreases. Since the SNR spans several decades, it is enlightening to consider the alternative log-SNR coordinate $u = \log \Lambda , w _ { \mathrm { e f f } } \mathrm { d } \Lambda = \tilde { w } ( u )$ du with

$$
\tilde { w } ( u ) = \Lambda w _ { \mathrm { e f f } } ( \Lambda ) , \qquad \tilde { w } _ { \mathrm { D M } } \simeq \frac { 1 } { 2 } , \qquad \tilde { w } _ { \mathrm { F M } } \simeq \frac { 1 } { 2 } \sqrt { \frac { d } { \Lambda } } ,\tag{115}
$$

for $\Lambda \ll d .$ Under this change of variable, it clearly appears that DM is scale-free: it deposits the same weight on every decade of SNR, while FM is not: its weight per decade grows as $\sqrt { d / \Lambda }$ towards low SNR. Evaluated at speciation scale yields

$$
\frac { \tilde { w } _ { \mathrm { F M } } } { \tilde { w } _ { \mathrm { D M } } } \bigg | _ { \Lambda = O _ { d } ( 1 ) } = O _ { d } \Big ( \sqrt { d } \Big ) .\tag{116}
$$

## A.4 Matching Diffusion and Flow Matching at equal SNR

The comparison of §A.3 is between weightings, not between implementations. To isolate that effect experimentally we keep everything else fixed – same $\boldsymbol { \xi }$ prediction network, same $\mathrm { V P }$ forward process $\mathbf { x } _ { t } \mathbf { \hat { \xi } } = e ^ { - t } \mathbf { x } _ { 0 } + \mathbf { \dot { \xi } } \sqrt { 1 - e ^ { - 2 t } } \pmb { \xi } ,$ , same sampler – and change only the weighting $w ( t )$ in equation 108: the baseline uses $w \equiv 1$ , i.e. plain DSM, and the “FM-weighted” run uses the $w ( t )$ that reproduces the effective SNR weighting of Flow Matching. We derive that weight here.

The matched time. Write $u ( t ) = \sqrt { e ^ { 2 t } - 1 }$ , so that the VP interpolant has $\alpha _ { \mathrm { D M } } / \beta _ { \mathrm { D M } } = e ^ { - t } / \sqrt { 1 - e ^ { - 2 t } } =$ $1 / u$ . The linear interpolant ${ \bf x } _ { r } = r { \bf x } _ { 0 } + ( 1 - r ) \pmb { \xi }$ has $\alpha _ { \mathrm { { F M } } } / \beta _ { \mathrm { { F M } } } = r / ( 1 - r )$ . Since $\Lambda$ is a monotone function of $\alpha / \beta$ alone, the two processes present the same denoising problem when

$$
\frac { r } { 1 - r } = \frac { 1 } { u } \qquad \Longleftrightarrow \qquad r ( t ) = \frac { 1 } { 1 + u ( t ) } .\tag{117}
$$

Differentiating equation 117 with $\mathrm { d } u / \mathrm { d } t = e ^ { 2 t } / u \operatorname { g i v e s } { \left| \mathrm { d } r / \mathrm { d } t \right| } = e ^ { 2 t } / \big ( u ( 1 + u ) ^ { 2 } \big )$ . The two combine into

$$
w _ { \mathrm { F M  D M } } ( t ) = ( 1 + u ) ^ { 2 } \times \frac { e ^ { 2 t } } { u ( 1 + u ) ^ { 2 } } = \frac { e ^ { 2 t } } { \sqrt { e ^ { 2 t } - 1 } } .\tag{118}
$$

One checks directly that uniform t sampling with this weight induces exactly $w _ { \mathrm { e f f } } ^ { \mathrm { F M } }$ of equation 114. The two limits are the ones anticipated in $\ S \mathrm { A } . 3 \colon$ at large $t , u \simeq e ^ { t }$ and $w _ { \mathrm { F M \to D M } } \simeq e ^ { t } = \sqrt { d / \Lambda }$ , recovering the tilt of equation 115; at small t it diverges as $1 / \sqrt { 2 t } ,$ , which is regularized in practice by the cutoff $t _ { \mathrm { m i n } } > 0$ used in training.

## A.5 Masked discrete diffusion

The SNR construction of $\ S \mathrm { A } . 3$ is not tied to Gaussian interpolants. We treat here the masked discrete diffusion (MDLM) pipeline of Sahoo et al. [36] used on the genomic data of §B.4.

Noise injection and SNR. In this case, each of the d tokens of $\mathbf { x } _ { \mathrm { 0 } }$ is independently replaced by MASK with probability $1 - e ^ { - \sigma ( t ) }$ , so that a token survives with probability $e ^ { - \sigma ( t ) }$ , and σ increases from $\sigma _ { \mathrm { m i n } }$ at $t = 0 \mathrm { t o } \sigma _ { \mathrm { m a x } } \mathrm { a t } t = 1$

In the discrete masking paradigm, a masked token carries no information at all, while an unmasked one carries it intact: masking destroys information by removing tokens rather than by attenuating them. The natural signal-to-noise ratio is therefore simply the expected number of tokens that survive,

$$
\Lambda _ { t } = e ^ { - \sigma ( t ) } d ,\tag{119}
$$

which plays exactly the role that $\Lambda _ { t } ~ = ~ \alpha ^ { 2 } d / \Gamma _ { t }$ plays in the Gaussian case, both counting effective informative units (we absorb the $O _ { d } ( 1 )$ information per token into the normalization).

Effective weighting. As implemented, the ELBO equation 130 averages the cross-entropy over the masked positions and reweights it by $\dot { \sigma } / ( 1 - e ^ { - \sigma } )$ , so that the per-token loss carries the time weighting $w ( t ) = \bar { \sigma } / ( 1 - e ^ { - \sigma } )$ . With $\Lambda \bar { = } e ^ { - \sigma } d$ we have $\vert \mathrm { d } t / \mathrm { d } \Lambda \vert = 1 / ( \dot { \sigma } e ^ { - \hat { \sigma } } d )$ , and the σ˙ cancels:

$$
w _ { \mathrm { e f f } } ^ { \mathrm { M D L M } } ( \Lambda ) = \frac { \dot { \sigma } } { 1 - e ^ { - \sigma } } \cdot \frac { 1 } { \dot { \sigma } e ^ { - \sigma } d } = \frac { d } { \Lambda \left( d - \Lambda \right) } \underset { \Lambda \ll d } { \simeq } \frac { 1 } { \Lambda } .\tag{120}
$$

Two consequences follow. First, equation 120 does not involve $\sigma ( t )$ : the effective weighting is independent of the noise schedule, a discrete counterpart of the known schedule invariance of the masked diffusion ELBO. Only the endpoints $\sigma _ { \mathrm { m i n } } , \sigma _ { \mathrm { m a x } }$ matter, through the range of Λ they expose. Second, comparing with equation 113, MDLM has the same $1 / \Lambda$ low SNR behavior as Diffusion: measured in SNR, masked discrete diffusion is scale-free and rather trains like a standard DSM than a FM.

## B Numerical details

## B.1 Fig. 1 and Fig. 2

In these experiments, we consider synthetic Gaussian-mixture data in $\mathbb { R } ^ { d }$ as described in the main text, focusing on two datasets:

• Unbalanced (two-mode GMM): a single signal direction µ with unequal mode probability $\omega = 0 . 8 ,$ swept over the dimension $d \in \{ 6 4 $ , 128, 256 at fixed $\omega .$

• Kappa dependence (Quadrimodal GMM): two orthogonal signal directions $\pm \mu _ { 1 } \perp \mu _ { 2 }$ of unequal strength $\left( \kappa \mathbf { v s } . 1 - \kappa \right)$ , each with its own mode probability $\omega _ { 1 } = \omega _ { 2 } = 0 . 7$ <sup>(asymmetric</sup> ± <sup>sign</sup> <sup>per</sup> mode). d = 1024 fixed, swept over $\kappa \in \{ 0 . 1 , 0 . \bar { 2 } , 0 . 3 \}$

In these experiments the score’s functional form is known exactly from the Gaussian-mixture log-density at all noise levels, and only a handful of scalar/vector coefficients in that closed form are learned, by denoising score matching (DSM) of a VP-SDE at a single, fixed noise level. In both settings we compare two fixed DSM noise times $t \in \{ 0 . 5 , t _ { s } \}$ side by side, where $\begin{array} { r } { t _ { s } = \frac { 1 } { 2 } \log d . } \end{array}$ . There is no sampling or sample generation anywhere in either experiment: because the score’s closed form is known exactly, model performance is evaluated directly from the learned coefficients themselves, rather than from samples generated by the model.

## B.1.1 Data

During training, data samples x are generated as follows:

Unbalanced GMM (dimension sweep). $\mathbf { x } = s { \pmb { \mu } } + \mathbf { z } , \pmb { \mu } = \mathbf { 1 } _ { d }$ (all-ones, so $\| \pmb { \mu } \| ^ { 2 } = d ) , \mathbf { z } \sim \mathcal { N } ( 0 , \pmb { I } _ { d } )$ $s = + 1$ with probability $\omega = 0 . 8 ( { \mathrm { e l s e - 1 } } )$ . Swept over d 64, 128, 256 at fixed $\omega = 0 . 8$

Quadrimodal GMM (kappa-dependence sweep). $\mathbf { x } = s _ { 1 } \pmb { \mu } _ { 1 } + s _ { 2 } \pmb { \mu } _ { 2 } + \mathbf { z } , \mathbf { z } \sim \mathcal { N } ( 0 , \pmb { I } _ { d } ) , d = 1 0 2 4 . \mu _ { 1 } , \mu _ { 2 }$ are orthogonal block-indicator vectors $( \| { \pmb \mu } _ { 1 } \| ^ { 2 } = k _ { 1 } = \lfloor \kappa d \rfloor , \| { \pmb \mu } _ { 2 } \| ^ { 2 } = k _ { 2 } = \lfloor ( 1 - \kappa ) d \rfloor )$ , with unbalanced signs: $s _ { 1 } = + 1$ with probability $\omega _ { 1 } = 0 . 7 ( \mathrm { e l s e - 1 } )$ , independently $s _ { 2 } = + 1$ with probability $\omega _ { 2 } = 0 . 7$ Swept over $\kappa \in \{ 0 . 1 , 0 . 2 , 0 . 3 \}$

## B.1.2 Model architecture

As discussed in depth in the main text, both experiments exploit closed-form knowledge of the true score, parameterizing it with a small number of scalar/vector coefficients rather than a generic function approximator; metrics are read directly off these learned coefficients.

Unbalanced. The exact score of the two-mode mixture factorizes as

$$
\nabla _ { \mathbf { x } } \log P _ { t } ( \mathbf { x } ) = - c \mathbf { x } + \mathbf { w } \operatorname { t a n h } ( \mathbf { x } ^ { T } \mathbf { w } + b ) ,\tag{121}
$$

with $\mathbf { w } \in \mathbb { R } ^ { d }$ playing the role of the learned µ-direction, c the pseudo variance, and the log weight $\begin{array} { r } { b  \frac { 1 } { 2 } \log \frac { \omega } { 1 - \omega } } \end{array}$ at convergence. The model thus has $( d + 2 )$ learnable parameters in total.

Quadrimodal. Because the four modes have equal weight $( \mathrm { g i v e n } s _ { 1 } , s _ { 2 } )$ and $\mu _ { 1 } \perp \mu _ { 2 }$ by construction, the exact score of the VP-SDE-noised mixture at a fixed time t similarly factorizes:

$$
\begin{array} { r } { \nabla _ { \mathbf x } \log P _ { t } ( \mathbf x ) = - c \mathbf x + \mathbf w _ { 1 } \operatorname { t a n h } ( \mathbf x ^ { T } \mathbf w _ { 1 } + b _ { 1 } ) + \mathbf w _ { 2 } \operatorname { t a n h } ( \mathbf x ^ { T } \mathbf w _ { 2 } + b _ { 2 } ) , } \end{array}
$$

with $\mathbf { w } _ { 1 } \in \mathbb { R } ^ { k _ { 1 } } , \mathbf { w } _ { 2 } \in \mathbb { R } ^ { k _ { 2 } }$ each compactly supported on their own mode’s subspace, plus scalars $c , b _ { 1 } , b _ { 2 } \colon$ $d { + 3 }$ learnable parameters total.

Both models use a small initialization: $\mathbf { w } _ { i } \sim d ^ { - \frac { 1 } { 2 } } \mathcal { N } ( 0 , I _ { d } )$ (respectively $\mathbf { w } _ { i } \sim k _ { i } ^ { - \frac { 1 } { 2 } } \mathcal { N } ( 0 , \pmb { I } _ { k _ { i } } ) )$ , such that initially $\| \mathbf { w } _ { i } \| ^ { 2 } = O ( 1 )$

## B.1.3 Training procedure

For each fixed DSM time $t ,$ we let $\alpha _ { t } = e ^ { - t } , \sigma _ { t } = \sqrt { 1 - e ^ { - 2 t } }$ , and optimize the loss

$$
\mathcal { L } = \mathbb { E } \Vert \sigma _ { t } \operatorname { s c o r e } ( \mathbf { x } _ { t } ) + \pmb { \xi } \Vert ^ { 2 }\tag{122}
$$

where $\mathbf { x } _ { t } = \alpha _ { t } \mathbf { x } _ { 0 } + \sigma _ { t } \pmb { \xi }$ and the expectation is taken over training data $\mathbf { x } _ { \mathrm { 0 } }$ and $\pmb { \xi } \sim \mathcal { N } ( 0 , \pmb { I } _ { d } )$

## B.1.4 Unbalanced, dimension dependence, Fig. 1

The two panels are the two fixed noising times: $t ~ = ~ 0 . 5 ~ ( l e f t )$ , which sits deep in Regime II, and $\begin{array} { r } { t = t _ { s } = \frac { 1 } { 2 } \log d \left( r i g h t \right) } \end{array}$ , the speciation time. colors index $d \in \{ 6 4 , 1 2 8 , 2 5 6 \}$ , and the three curves of the legend are, for each d:

• Overlap - the direction overlap $\mathbb { E } _ { + } [ \mathbf { w } ^ { T } \pmb { \mu } ] / ( e ^ { - t } d )$ , normalized to 1 at convergence.

• c - the learned scalar $c ,$ which plays the role of an inverse-variance-like precision term in the exact-score formula.

• weight - the learned mode weight $\omega = 0 . 5 + | \mathrm { s i g m o i d } ( 2 b ) - 0 . 5 |$ , read off the bias. The dotted horizontal line marks its target value $\omega = 0 . 8$

The x axis is the number of SGD steps divided by d.

## B.1.5 Kappa dependence, Fig. 2

The figure has three panels: $t = 0 . 5 ( l e f t )$ , deep in Regime II; $t = t _ { s } ( m i d d l e )$ , at speciation; and the same speciation data replotted against a rescaled x axis (right). colors index $\kappa \in \{ 0 . 1 , 0 . 2 , 0 . 3 \}$ , and for each κ the two curves of the legend are the overlaps along the two orthogonal directions,

$$
m _ { 1 } = \frac { \mathbb { E } _ { + } [ { \bf w } _ { 1 } ^ { T } { \pmb \mu } _ { 1 } ] } { e ^ { - t } \kappa d } , \qquad m _ { 2 } = \frac { \mathbb { E } _ { + } [ { \bf w } _ { 2 } ^ { T } { \pmb \mu } _ { 2 } ] } { e ^ { - t } ( 1 - \kappa ) d } ,\tag{123}
$$

each normalized to 1 at convergence $( m _ { 1 }$ dashed, $m _ { 2 }$ solid). The learned biases $b _ { i }$ are not displayed here. The x axis of the first two panels is the raw number of SGD steps. In the third it is rescaled by the growth rate of the corresponding block, written $\lambda ( \sqrt { \kappa _ { i } } )$ in the figure and equal to the closed-form rate $\lambda ( \gamma _ { i } )$ of equation 13, evaluated at $\gamma _ { i } ^ { 2 } = \kappa _ { i } \Lambda _ { t } \Gamma _ { t }$ with $\kappa _ { 1 } = \kappa$ and $\kappa _ { 2 } = 1 - \kappa .$

Relation to $\left( \mathcal { D } _ { 2 } \right)$ . The hierarchical target equation $7$ of the main text is balanced, all four modes carrying weight $1 / 4 ,$ whereas the runs here draw each sign with probability $\omega _ { 1 } = \omega _ { 2 } = 0 . 7$ . The exact score model keeps one bias per block, $b _ { 1 } , b _ { 2 } ,$ initialized at zero, so this imbalance is representable and is learned alongside the overlaps. Strictly, each block then follows the coupled $( m _ { i } , b _ { i } )$ dynamics of Result 1 rather than the balanced reduction of Result $^ { 2 , }$ the off-diagonal coupling being proportional to $2 \omega _ { i } - 1$ . This does not affect what the figure tests: the growth rate of block i remains a function of $( q _ { i } , \gamma _ { i } )$ alone, hence of $\kappa _ { i } \Lambda$ , so the collapse variable is unchanged and the balanced closed form $\lambda ( \gamma _ { i } )$ is used as the rescaling ansatz. The ResNet experiments below instead use the balanced target exactly.

Training hyperparameters are reported below:
<table><tr><td>Unbalanced and Kappa-dependence</td></tr><tr><td>Optimizer plain SGD (no momentum, no weight decay, constant LR)</td></tr><tr><td>Learning rate  $1 0 ^ { - 1 }$ </td></tr><tr><td>Batch size 512</td></tr><tr><td>Training steps  $1 0 ^ { 6 }$  (SGD steps, no early stopping)</td></tr><tr><td>tnoise 0.5 or  $\begin{array} { r } { t _ { s } = \frac { 1 } { 2 } \log { \bar { d } } } \end{array}$ </td></tr><tr><td>Seeds per config 20</td></tr><tr><td>Sweep values  $d \in \{ 6 4 , 1 2 8 , 2 5 6 \}$  at ω = 0.8  $\kappa \in \{ 0 . 1 , 0 . 2 , 0 . 3 \}$  at  $d = 1 0 2 4$ </td></tr></table>

Table 1: Hyperparameters shared by all 20 runs averaged to produce each figure. Metrics are logged at 200 log-spaced steps.

## B.2 Fig. 3a and Fig. 3b

In these experiments, we return to the same two Gaussian-mixture datasets introduced above — unbalanced and quadrimodal GMMs, with data generated as in §B.1.1 except that the quadrimodal signs are now drawn with equal probability, $\omega _ { 1 } = \omega _ { 2 } = 0 . 5$ , so that the target is exactly the balanced $\left( \mathcal { D } _ { 2 } \right)$ of equation $7 ;$ the unbalanced target keeps $\omega = 0 . 8 -$ but replace the analytically exact score parameterization with a generic fully-connected ResNet epsilon-predictor, trained by denoising score matching (DSM) over a continuous range of noise levels rather than at a single fixed t. Correspondingly, since the score is no longer known in closed form, model performance can no longer be read directly off a handful of coefficients: instead, we evaluate it by drawing samples from the model via reverse-time SDE integration and comparing statistics of those samples to the training distribution. We contrast two training objectives, standard DSM and FM-weighted DSM.

## B.2.1 Model architecture

Both experiments use the same fully-connected ResNet epsilon-predictor $\hat { \xi } _ { \theta } ( \mathbf { x } , t ) : \mathbb { R } ^ { d } \times \mathbb { R }  \mathbb { R } ^ { d } \mathrm { : }$ a linear input projection to width 128, 4 residual blocks (LayerNorm, additive sinusoidal-time-embedding conditioning via a linear projection, then a 2-layer SiLU MLP, residual add), and a zero-initialized linear output projection back to $\mathbb { R } ^ { i }$ . Sinusoidal time embedding dimension 128. The score is reparameterized with a skip connection. The input/output projections are the only d-dependent layers.

## B.2.2 Training procedure

In both experiments we follow the standard denoising score matching procedure, where noised samples are obtained via the forward continuous-time variance-preserving Ornstein-Uhlenbeck SDE: ${ \mathrm { d } } { \bf x } _ { t } =$ $- { \mathbf x } _ { t } \mathbf d t + \sqrt { 2 } { \mathbf d } { \mathbf b } _ { t } ,$ with closed-form marginal $\mathbf { x } _ { t } \mid \mathbf { x } _ { 0 } \sim { \mathcal { N } } ( e ^ { - t } \mathbf { x } _ { 0 } , { \sqrt { ( 1 - e ^ { - 2 t } ) } } I )$ . The network $\hat { \xi } _ { \pmb { \theta } }$ predicts the injected noise $\xi ,$ and the score used in the backward inference dynamics is recovered via Tweedie’s formula, $\nabla _ { \mathbf x }$ log $p _ { t } ( \mathbf { x } ) = - \hat { \xi } _ { \theta } ( \mathbf { x } , t ) / \sigma ( t )$ . Sampling integrates the reverse SDE with Euler–Maruyama from $t _ { \mathrm { m a x } }$ down to $t _ { \mathrm { m i n } } > 0$

To evaluate the impact of weighting schemes, we use a weighted denoising score matching loss

$$
\mathcal { L } = \mathbb { E } _ { t \sim \mathcal { U } ( t _ { \mathrm { m i n } } , t _ { \mathrm { m a x } } ) } [ w ( t ) \| \pmb { \xi } - \hat { \xi } _ { \pmb { \theta } } ( \mathbf { x } _ { t } , t ) \| ^ { 2 } ]\tag{124}
$$

with two choices of $w ( t ) \ d t =$

• Uniform DSM: $w ( t ) = 1$

• FM-weighted DSM: $w ( t ) = e ^ { 2 t } / \sqrt { e ^ { 2 t } - 1 }$ , the matched weight equation 118 derived in $\ S \mathrm { A . 4 }$ For each training we use plain SGD with a constant learning rate and neither momentum nor weight decay. The curves report the mean over the generated samples at each logged step and the shaded band the corresponding 95% CI, so the band measures sampling error at fixed model rather than run-to-run variability. Inference and training hyperparameter values are reported below:

<table><tr><td></td><td>Unbalanced GMM Quadrimodal GMM</td><td></td></tr><tr><td> $d$ </td><td>64</td><td>64</td></tr><tr><td>Optimizer</td><td colspan="2">plain SGD (no momentum, no weight decay, constant LR)</td></tr><tr><td>Learning rate</td><td> $\mathrm { \hat { 1 } 0 ^ { - 2 } }$ </td><td> $1 0 ^ { - 2 }$ </td></tr><tr><td>Batch size</td><td>2048</td><td>2048</td></tr><tr><td>Training steps (SGD)</td><td> $1 0 ^ { 6 }$ </td><td> $1 0 ^ { 6 }$ </td></tr><tr><td> $t _ { \mathrm { m i n } } , t _ { \mathrm { m a x } }$ </td><td> $5 \times 1 0 ^ { - 3 } , 6 . 2$ </td><td> $1 0 ^ { - 3 } , \ 6 . 2$ </td></tr><tr><td>Reverse SDE steps (eval)</td><td>30</td><td>30</td></tr><tr><td>Eval samples / logged step</td><td> $5 \times 1 0 ^ { 4 }$ </td><td> $5 \times 1 0 ^ { 4 }$ </td></tr><tr><td>Sweep values</td><td> $\omega = 0 . 8$ </td><td> $\kappa \in \{ 0 . 2 , 0 . 3 , 0 . 4 \}$ </td></tr></table>

Table 2: Hyperparameters associated to each figure. Both objectives (uniform / FM-weighted DSM) use identical hyperparameters otherwise.

## B.2.3 Unbalanced GMM, Fig. 3a

The two panels are the two weighting schemes of equation 124, labeled Uniform DSM $( w ( t ) = 1 )$ and FM-weighted DSM, at the single imbalance $\omega = 0 . 8$ marked by the dotted horizontal line. For each we monitor the two curves of the legend:

• Cosine Similarity - the cosine between an inferred sample and the mode direction, cos ${ \bf \nabla } _ { \bf \left\{ x , \right\}  \mu } =$ $\mathbf { x } ^ { T } \pmb { \mu } / ( | \mathbf { x } | | \pmb { \mu } | )$ . Since the target is symmetric, we average the mean cosine over the two signs of $\mathbf { x } ^ { T } \boldsymbol { \mu }$ and report $\begin{array} { r } { \frac { 1 } { 2 } \big ( | \mathbb { E } _ { + } [ \cos ( \breve { \mathbf { x } } , \pmb { \mu } ) ] | + | \mathbb { E } _ { - } [ \cos ( \mathbf { x } , \pmb { \mu } ) ] | \big ) } \end{array}$ . This tracks direction learning: a transition to $O ( 1 )$ values indicates that the inferred samples are spread along the preferential training direction $\pmb { \mu } .$ Note that this does not mean the inferred distribution is necessarily bimodal, only that its covariance matrix is no longer isotropic.

• Positive fraction - the fraction of generated samples falling on the dominant side of the mode direction, $\scriptstyle { \frac { 1 } { N } } \# \{ \mathbf { x } : \mathbf { x } ^ { T } \pmb { \mu } > 0 \}$ . It converges to ω once the imbalance has been acquired.

Each panel carries an inset showing the histogram of the projection $p = \mathbf { x } ^ { T } \pmb { \mu } / \Vert \pmb { \mu } \Vert ^ { 2 }$ of the generated samples, at the three training times marked by the vertical dashed lines; the dotted verticals mark the target mode locations 1.

## B.2.4 Quadrimodal GMM, Fig. 3b

Again the two panels are the two weighting schemes, Uniform DSM and FM-weighted DSM, and colors index $\kappa \in \{ 0 . 2 , 0 . 3 , 0 . 4 \}$ . The monitored quantity is the cosine similarity between the inferred samples and the subdominant direction $\pmb { \mu } _ { 1 }$ , defined as in §B.2.3 with $\pmb { \mu }$ replaced by $\mu _ { 1 } ;$ since $\| \mu _ { 1 } \| ^ { 2 } = \kappa d ,$ smaller κ means a weaker subdominant mode.

Only the FM-weighted DSM panel carries an inset, which replots that panel’s curves against the rescaled x axis, training steps $\times \lambda ( { \sqrt { \kappa } } )$ , with λ the closed-form rate equation 13 evaluated at $\check { \gamma _ { i } ^ { 2 } } = \kappa _ { i } \Lambda$

## B.3 Fig. 4a and Fig. 4b

We extend the ResNet/GMM experiments with two companion experiments on MNIST digits 3, 6 , sharing the same generative architecture and varying a different axis of the data-generating process:

• Weight learning (class-imbalanced grayscale MNIST): only digit identity varies, but the two classes appear in an imbalanced ratio $\omega = P ( { \mathrm { c l a s s } } = 6 )$ in the training set.

• Structure learning (tinted MNIST): both digits are shown in a random, class-independent color (red or green) at a fixed contrast α. The data has two orthogonal directions of variation — digit identity (“3 vs. 6”) and color (“red vs. green”) — and α controls the strength of the (weaker) color signal.

## B.3.1 Data

Both experiments restrict MNIST (train split) to digits 3 and $\rangle \left( n _ { 3 } = 6 , 1 3 1 , n _ { 6 } = 5 , 9 1 8 \mathrm { i m a g e s } , 2 8 \times 2 8 \right)$

Class-imbalanced MNIST. Grayscale (1 channel), normalized to [ 1, 1]. All n<sub>6</sub> sixes are kept and threes are subsampled to hit a target mixture weight $\omega \in \{ 0 . 5 , 0 . 8 \} \colon n _ { \mathrm { t o t } } = \lfloor n _ { 6 } / \omega \rfloor , n _ { 3 } = n _ { \mathrm { t o t } } - n _ { 6 } \left( \mathbf { s o } \omega = 0 . 5 \right)$ balanced, $n _ { 3 } = n _ { 6 } = 5 , 9 1 8 ; \omega = 0 . 8 \colon n _ { 3 } = 1 , 4 7 9 , n _ { 6 } = 5 , 9 1 8 ,$ total 7,397).

Tinted MNIST. All 12,049 available 3s and 6s are used, no subsampling. Each grayscale image x $[ 0 , 1 ] ^ { 2 8 \times 2 8 }$ is expanded to 3 channels and randomly tinted red or green (uniformly, independently of digit label) at contrast $\alpha \in \{ 0 . 4 , 0 . 5 , 0 . 6 , 0 . 7 , 0 . 8 , 0 . 9 , 1 . 0 \}$ :

$$
\begin{array} { r } { \mathbf { x } _ { \mathrm { t i n t e d } } = ( 1 - \alpha ) \bar { \mathbf { x } } + \alpha \bar { \mathbf { x } } \odot c , \qquad \bar { \mathbf { x } } = ( \mathbf { x } , \mathbf { x } , \mathbf { x } ) , \quad c \in \{ ( 1 , 0 , 0 ) , ( 0 , 1 , 0 ) \} , } \end{array}\tag{125}
$$

then normalized to [ 1, 1] per channel. $\alpha = 0$ would give plain grayscale (no hue signal); $\alpha = 1$ gives a fully saturated tint; see Fig. 6 for a sample.

![](images/1d41a65ee6dc089253b002c82f1df2c3b9293d72c551cc7ee0cdb756c8057056.jpg)  
Figure 6: Image sample at tint $\alpha = 0 . 7$

## B.3.2 Model architecture

Both experiments use the same U-Net backbone (torchcfm.models.unet.UNetModel): model channels 32, 2 residual blocks per resolution for the imbalance experiment and 1 for the tinted one, channel multiplier (1, 2, 2) (the package’s default for 28 28 inputs), self-attention at the coarsest resolution (single head), no class conditioning, dropout 0, GroupNorm+SiLU residual blocks with a sinusoidal timestep embedding. Input/output channels are 3 for the tinted experiment and 1 for the grayscale one, both $\mathrm { 1 . 0 8 \times 1 0 ^ { 6 } }$ parameters.

That backbone is used in two different roles. Under Flow Matching it is the velocity field $\mathbf { v } _ { \pmb { \theta } } ( \mathbf { x } , t )$ regressed in equation 128. Under DSM it parametrizes the score through a trainable skip connection,

$$
\mathbf { \boldsymbol { s } } _ { \theta } ( \mathbf { \boldsymbol { x } } , t ) = - \mathbf { \boldsymbol { x } } + \mathrm { U N e t } _ { \theta } ( \mathbf { \boldsymbol { x } } , t ) ,\tag{126}
$$

rather than the usua $- \hat { \xi } _ { \theta } ( \mathbf { x } , t ) / \sigma ( t ) ;$ : at initialization the network output is small, so the score defaults to $- \mathbf { x } ,$ the score of a standard normal, instead of an untrained noise prediction divided by a possibly tiny $\sigma ( t )$ . This keeps reverse-SDE sampling stable before the network has learned anything.

## B.3.3 Flow-matching training procedure

In the colorized experiment, we use the standard conditional Flow Matching (CFM): for a clean sample $\mathbf { x } _ { 1 }$ and $\mathbf { x } _ { 0 } \sim \mathcal { N } ( 0 , I )$

$$
{ \bf x } _ { t } = ( 1 - t ) { \bf x } _ { 0 } + t { \bf x } _ { 1 } , \qquad t \sim \mathcal { U } ( 0 , 1 ) ,\tag{127}
$$

and the network is trained to regress the constant conditional velocity $\dot { \mathbf { x } } _ { t } = \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 }$ with an MSE loss,

$$
\mathcal { L } ( \pmb { \theta } ) = \mathbb { E } _ { t , \mathbf { x } _ { 0 } , \mathbf { x } _ { 1 } } \| \mathbf { v } _ { \pmb { \theta } } ( \mathbf { x } _ { t } , t ) - ( \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } ) \| ^ { 2 }\tag{128}
$$

Sampling integrates $\mathbf { v } _ { \pmb { \theta } }$ from $t = 0$ (Gaussian noise) to $t = 1$ with an adaptive-step Dormand–Prince (dopri5) ODE solver (torchdyn, atol $= \mathrm { r t o l } = 1 0 ^ { - 4 } )$ . The training and inference parameters are reported below:

<table><tr><td></td><td>Structure sweep (tinted)</td><td>Weight sweep (imbalanced)</td></tr><tr><td>Optimizer</td><td colspan="2">AdamW, weight decay  $1 0 ^ { - 2 }$  (default), constant LR</td></tr><tr><td>Learning rate</td><td colspan="2"> $1 0 ^ { - 3 }$ </td></tr><tr><td>Batch size</td><td>128</td><td></td></tr><tr><td>Training steps</td><td>2,000</td><td> $1 0 { , } 0 0 0$ </td></tr><tr><td>Sweep values</td><td> $\alpha \in \{ 0 . 4 , \ldots , 1 . 0 \}$  (step 0.1)</td><td> $\omega = 0 . 8$ </td></tr><tr><td>Seeds per value</td><td>10</td><td>5</td></tr><tr><td>Eval samples / logged step</td><td> $1 0 , 0 0 0 \left( 1 , 0 0 0 \times 1 0 \mathrm { r e p e a t s } \right)$ </td><td>5,000 (1,000 × 5 repeats)</td></tr><tr><td>Classifier epochs</td><td>6</td><td>20</td></tr></table>

Table 3: Hyperparameters for the two experiments.

The imbalance experiment is run twice, once with the Flow Matching objective above and once with the score-based DSM objective of §B.2.2, which is the pair compared in Fig. 4a. The DSM side uses the same U-Net with the score skip connection equation 126, a unit-rate Ornstein-Uhlenbeck forward process dx $= - \mathbf { x } \mathrm { d } t + \sqrt { 2 } \mathrm { d } \mathbf { b } _ { t }$ run over $t \in [ 5 \times 1 0 ^ { - 3 }$ , 3.3], and generates by integrating the reverse SDE over 50 steps with cutoff $\epsilon _ { t } = 5 \times 1 0 ^ { - 3 }$ . All optimization hyperparameters are those of Tab. 3.

## B.3.4 Weight learning, Fig. 4a

Both panels show the same unbalanced dataset at fixed $\omega = 0 . 8$ (dotted horizontal line); what differs is the training objective, DSM on the left and Flow Matching on the right, so the two panels isolate the effect of SI pipeline at fixed data. Curves are the mean over the 5 seeds, the band a 95% CI across seeds, Gaussian-smoothed along the step axis.

At each logged step we draw samples from the current model and pass them through two fixed evaluation probes, neither of which is ever used to train the generative model:

• An auxiliary digit classifier, a 1-channel, 2-layer CNN trained once per run on real data with a matching noise-augmentation scheme (classify $( 1 - t ) \mathbf { x } _ { 0 } + t \mathbf { x } _ { 1 }$ against the true label, $t \sim \mathcal { U } ( 0 , 1 )$ so it stays calibrated at the high-noise end of the path). It is always fit on a balanced $( \omega = 0 . 5 )$ split, so it judges generated class frequency without inheriting the training-set imbalance.

• Class-mean projections. $\pmb { \mu } _ { 3 } , \pmb { \mu } _ { 6 }$ , the pixel-space mean image of each digit class in the training set, precomputed per run; every generated sample x is projected onto both, $\mathbf { x } ^ { T } \pmb { \mu } _ { 3 }$ and $\mathbf { x } ^ { T } \pmb { \mu } _ { 6 }$ The two curves of the legend are then:

• Cosine Similarity - E- max $\big ( \cos ( { \mathbf { x } } , \mu _ { 3 } ) , \cos ( { \mathbf { x } } , \mu _ { 6 } ) \big ) \big ]$ , the cosine between a generated sample and the nearer of the two class-mean images.

• Fraction of 6s $\mathbf { \nabla } - \mathbb { E } [ \mathbf { 1 } _ { C ( \mathbf { x } ) = 6 } ]$ , the empirical fraction of generated samples the classifier assigns to digit 6.

## B.3.5 Structure learning, Fig. 4b

Two panels, the mean over the 10 seeds per $\alpha ,$ the band a 95% CI across seeds, Gaussian-smoothed along the step axis. The digit direction is read off the same class-mean projections as above; the color direction uses a hue projection, the mean red minus mean green pixel intensity of a generated image, $r - g$ (“color spread”).

• Left panel - the digit direction overlap, defined as $\mathbb { E } \big [ \operatorname* { m a x } ( \frac { | { \bf x } ^ { T } { \pmb \mu } _ { 3 } | } { | { \pmb \mu } _ { 3 } | ^ { 2 } } , \frac { | { \bf x } ^ { T } { \pmb \mu } _ { 6 } | } { | { \pmb \mu } _ { 6 } | ^ { 2 } } ) \big ]$

• Right panel - the color spread. Since $r - g$ is bimodal by construction we keep only its positive part and monitor $\mathbb { E } _ { + } [ \alpha \cdot { \dot { ( } r - g ) } ] / \alpha ^ { 2 }$ . The $\alpha ^ { - 2 }$ normalization reflects the squared norm of the training color spread scaling as $\alpha ^ { 2 }$ , mirroring the $| \mu _ { 3 , 6 } | ^ { 2 }$ normalization of the digit direction.

To rescale the time axis of the subdominant direction we identify the relative direction amplitude $| \mu _ { 1 } | ^ { 2 } / | \mu _ { 2 } | ^ { 2 }$ as proportional to $\alpha ^ { 2 } ;$ rescaling training time by equation 13 then collapses the curves.

## B.4 Fig. 5

## B.4.1 Dataset

We use the 805-SNP subset of phased haplotypes from the 1000 Genomes Project (805\_SNP\_1000G\_real.hapt). Each of the 2,504 sequenced individuals contributes two phased haplotypes, giving $N = 5$ ,008 binary sequences of length $\mathbf { \dot { \boldsymbol { L } } } = 8 0 5 , \mathbf { x } \in \{ 0 , 1 \} ^ { L }$ . The data is split into a $9 0 \bar { / } 1 0$ training and validation set. $\mathrm { A }$ reference PCA basis $\{ \pmb { \mu _ { k } } \} _ { k = 1 } ^ { K } , K = 5 0$ , together with the reference eigenvalues $\{ \check { \lambda _ { k } } \} _ { k = 1 } ^ { K } ,$ is fit once on the training split and kept fixed throughout training for evaluation purposes.

## B.4.2 Masked Discrete Diffusion process

We use continuous-time absorbing-state (masked) discrete diffusion (MDLM) [36], adapted to a binary alphabet. Each token from the original sequence $\mathbf { x } _ { \mathrm { 0 } }$ is independently replaced in the noised sequence $\mathbf { x } _ { t }$ by MASK with probability $\alpha ( t ) = 1 - e ^ { - { \dot { \sigma } } ( t ) }$ , under a log-linear noise schedule

$$
\sigma ( t ) = \sigma _ { \operatorname* { m i n } } ^ { 1 - t } \sigma _ { \operatorname* { m a x } } ^ { t } , \qquad t \in [ 0 , 1 ] ,\tag{129}
$$

with $\sigma _ { \mathrm { m i n } } = 1 0 ^ { - 4 } , \sigma _ { \mathrm { m a x } } = 2 0$ . The model is trained with the continuous-time ELBO (masked crossentropy at masked positions only, reweighted by $\dot { \sigma } ( t ) / ( 1 - e ^ { - \sigma ( t ) } ) )$ , with $t \sim \mathcal { U } ( \xi , 1 ) , \epsilon = 1 0 ^ { - 4 }$ , resampled independently for every sequence in a minibatch:

$$
\mathcal { L } ( \pmb \theta ) = - \int _ { \epsilon } ^ { 1 } \frac { \dot { \sigma } ( t ) } { \left( 1 - e ^ { - \sigma ( t ) } \right) } \mathbb { E } _ { \mathbf { x } _ { t } , \mathbf { x } _ { 0 } } \left[ \mathrm { l o g } ( p _ { \theta } ( \mathbf { x } _ { 0 } | \mathbf { x } _ { t } ) ) \right] .\tag{130}
$$

Sampling uses the ancestral reverse process as originally proposed in the MDLM paper, starting from a fully masked sequence and unmasking tokens over $n _ { \mathrm { s t e p s } } = 3 0$ discretized reverse steps. At each logged step, $1 0 \times 5 0 0$ samples are drawn from the current model to evaluate PCA overlap statistics.

## B.4.3 Model architecture

The denoiser $p _ { \theta }$ is a small Transformer encoder, operating on a 3-token vocabulary 0, 1, MASK :

• Input embedding. A learned token embedding $( 3 \times d _ { \mathrm { m o d e l } } )$ plus a learned absolute positional embedding $( L \times d _ { \mathrm { m o d e l } } )$ , summed.

• Time conditioning. The noise level enters through log $\sigma ( t )$ , mapped by a sinusoidal embedding of dimension $d _ { \mathrm { c o n d } }$ followed by a 2-layer MLP $( d _ { \mathrm { c o n d } }  4 d _ { \mathrm { c o n d } }  d _ { \mathrm { m o d e l } } $ , GELU nonlinearity), and added to every token position (broadcast over the sequence).

• Backbone. $n _ { \mathrm { l a y e r s } }$ pre-norm blocks $( d _ { \mathrm { m o d e l } } , \ n _ { \mathrm { h e a d s } }$ heads, feed-forward width $4 d _ { \mathrm { m o d e l } }$ , GELU, dropout $p )$ , full self-attention.

• Output head. LayerNorm linear projection to 3 logits, followed by the SUBS parameterization of MDLM: the MASK logit is set to $- \infty ;$ at unmasked positions the output distribution is forced to a point mass on the observed token; at masked positions the model outputs a free softmax over 0, 1 . The output projection is zero-initialized.

For the run analyzed here: $d _ { \mathrm { m o d e l } } = 6 4 , n _ { \mathrm { h e a d s } } = 4 , n _ { \mathrm { l a y e r s } } = 2 , d _ { \mathrm { c o n d } } = 6 4 ,$ , dropout $p = 0 . 1 $ , giving ≈ $1 . 8 5 \times 1 0 ^ { 5 }$ trainable parameters.

Training parameters are reported below  
Optimizer AdamW, weight decay $1 0 ^ { - 2 }$   
Peak learning rate $3 \times 1 0 ^ { - 2 }$   
LR schedule linear warm-up (50 steps)  constant  linear decay to $1 0 ^ { - 3 } \times$ peak over the   
final 20% of training (decay starts at step 80,000)   
Batch size 256   
Training steps 100,000   
Gradient clipping global norm 1.0   
Seeds 4 independent runs  
Table 4: Optimization hyperparameters for the run shown in Fig. 5

## B.4.4 Figure 5: PCA-learning collapse

At every logged training step s, and for each of the 5,000 freshly generated samples, we:

1. Fit a fresh K-component PCA, $\{ \mu _ { k } ^ { \mathrm { i n f } } ( s ) \}$ , on the generated batch (“inferred” PCA at step s).

2. Compute the overlap matrix between the inferred and the fixed reference eigenvectors,

$$
O _ { i j } ( s ) = \frac { 1 } { \vert \mu _ { j } ^ { \mathrm { r e f } } \vert ^ { 2 } } \mu _ { i } ^ { \mathrm { i n f } } ( s ) \cdot \mu _ { j } ^ { \mathrm { r e f } } , \qquad i , j = 1 , \ldots , K ,\tag{131}
$$

3. Pair inferred and reference directions one to one, through the permutation $\pi _ { s }$ of $\{ 1 , \ldots , K \}$ maximizing the total matched overlap $\begin{array} { r l } { \sum _ { j } | O _ { \pi _ { s } ( j ) j } ( s ) | } \end{array}$ (Hungarian algorithm). The matched value $\vert \mu _ { \mathrm { i n f } } \cdot \bar { \mu _ { \mathrm { r e f } } } \vert _ { k } ( s ) : = \vert O _ { \pi _ { s } ( k ) k } ( s ) \vert$ tracks how well the k-th reference principal direction has emerged in the generated samples by training step s.

Ranking the entries of O and reading the k-th largest as the k-th direction would be ambiguous on two counts: the curves would then be non-increasing in k by construction, whatever the model has learned, and a single dominant inferred component leaking onto several reference eigenvectors could occupy several of the top slots. Both matter here, since the inset of Fig. 5 rescales curve k by the timescale of reference direction k specifically. The two constructions agree on the two leading directions, each carried by a single inferred component, and differ on the subleading ones. The reported curves are the mean and 95% CI of $| \mu _ { \mathrm { i n f } } \cdot \mu _ { \mathrm { r e f } } | _ { k } ( s )$ across the 4 training seeds, lightly smoothed along the step axis (Gaussian filter, $\sigma = 0 . 1$ index units), for the top 5 modes.

Left panel. A plain scatter of $\mathbf { x } ^ { T } \pmb { \mu } _ { 0 }$ vs. $\mathbf { x } ^ { T } \pmb { \mu } _ { 1 }$ (the projections of each haplotype onto the top-two reference PCA directions), using every one of the $N _ { \mathrm { t r a i n } } = 4 { , } 5 0 8$ real training haplotypes – no subsampling. Both the scatter and the reference directions $\{ \mu _ { k } \}$ come from the same reference PCA fit.

Right panel. $| \mu _ { \mathrm { i n f } } \cdot \mu _ { \mathrm { r e f } } | _ { k } ,$ , labeled PCA Overlap on the figure, plotted against the raw training step s; the inset shows the same curves against the rescaled step $\lambda _ { k } \times s ,$ , where they collapse. To identify the rescaling timescales we extract the explained variance $v _ { k }$ along each of the reference PCA directions. The GMM growth rate equation 13 is a function of the block SNR $\gamma _ { i } ,$ which at the speciation scale reduces to $\gamma _ { i } = \sqrt { \kappa _ { i } } ,$ with $\kappa _ { i } = \dot { \| } \mu _ { i } \| ^ { 2 } / d$ the squared amplitude of direction i. Identifying that squared amplitude with the variance explained along direction $k ,$ we rescale the time axis by

$$
\lambda _ { k } \equiv \lambda ( \sqrt { v _ { k } } ) .\tag{132}
$$

The PCA projected Human Genome dataset is of course not a GMM, and the rescaling should only be expected to hold where the projected data actually resolves into modes. This is the case for the two leading directions, which collapse onto one another; the subsequent ones, along which no mode separation is visible, do not, and the ansatz carries less meaning there.

## C LLM usage

We acknowledge the use of LLMs to assist in drafting the manuscript and refining its clarity. The tool was also used to support the derivation of mathematical proofs and the writing of the code. All AI-generated content was rigorously reviewed, verified, and edited by the authors. The authors bear full responsibility for the originality and scientific integrity of this work.