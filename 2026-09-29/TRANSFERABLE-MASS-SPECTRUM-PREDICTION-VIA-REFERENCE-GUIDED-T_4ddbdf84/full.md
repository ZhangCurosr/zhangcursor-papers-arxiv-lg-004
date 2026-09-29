# TRANSFERABLE MASS SPECTRUM PREDICTION VIA REFERENCE-GUIDED TEST-TIME SPECIALIZATION

Yunhua Zhong<sup>1,2,†</sup>, Runting Li<sup>1,3</sup>, Yifan Li<sup>1</sup>, Pan Liu<sup>1</sup>, Zhiwen Yang<sup>1,4</sup>, Zikun Wang<sup>1</sup>, Yixuan Tang<sup>1,5</sup>, Jun Xia<sup>1,6,∗</sup>

<sup>†</sup>This work was done during an undergraduate internship at HKUST-GZ.

<sup>1</sup>The Hong Kong University of Science and Technology (Guangzhou)

<sup>2</sup>The University of Hong Kong, <sup>3</sup>South China University of Technology

<sup>4</sup>Hong Kong Polytechnic University, <sup>5</sup>Jinan University

<sup>6</sup>The Hong Kong University of Science and Technology

<sup>∗</sup>Corresponding Author.

## ABSTRACT

Tandem mass spectrum prediction supports compound identification across metabolomics, natural-product discovery, and environmental analysis. However, pretrained predictors often degrade under shifts in chemical space and acquisition conditions, while retraining domain-specific models from scratch is costly. We introduce SPARC, a retrieval-guided test-time specialization framework that adapts a pretrained predictor using a spectral reference library without accessing test-query spectra. For each target query, SPARC retrieves chemically related reference spectra to recalibrate fragment intensities within the learned fragmentation space. During Transfer, SPARC combines reference-guided spectral adaptation with reliability-aware consistency, using reconstruction behavior on retrieved spectra to selectively preserve trustworthy predictions during continual specialization. Across MassSpecGym, NPLIB1 and application-specific GNPS libraries, SPARC improves spectral prediction under multiple transfer settings. These results establish retrieval-guided test-time specialization as a practical strategy for extending pretrained MS/MS predictors to specific chemical and acquisition domains, with continual test-time training providing further refinement during deployment.

## 1 INTRODUCTION

Tandem mass spectrometry is central to molecular annotation, natural-product discovery (Duhrkop¨ et al., 2019; Wang et al., 2016) and environmental analysis (Schymanski et al., 2014). By estimating fragment-ion masses and intensities from molecular structures, molecule-to-spectrum predictors extend spectral reference coverage beyond available experimental measurements (Goldman et al., 2024; Murphy et al., 2023; Nowatzky et al., 2025; Young et al., 2024a). Their performance, however, depends on the chemical and acquisition distributions represented during training (Bremer et al., 2022; Young et al., 2024a). Large spectral libraries expose a model to broader chemical space and fragmentation behavior than a single study can provide (Gupta et al., 2026; Murphy et al., 2023). Real-world deployment rarely queries this space uniformly. Each application emphasizes a particular molecular distribution together with its own adducts, ionization modes, instruments, collision energies, and metadata. These factors can change spectral similarity and prediction performance (Bremer et al., 2022; Hoang et al., 2024; Liu et al., 2025). Consequently, broad training coverage does not ensure accurate predictions for the chemical space and acquisition condition of a particular study.

This mismatch produces a practical generalization problem. Changes in acquisition conditions alter the mapping from molecular structure to fragment intensities (Bremer et al., 2022). Study-defined molecular distributions can also concentrate predictions in regions that were sparse during source training (Bushuiev et al., 2024). Adduct-dependent fragmentation provides a concrete example. Compounds retained under one precursor ion may lack paired spectra under another condition. The same molecule can also fragment differently across adducts (Schmid et al., 2021), creating coverage and chemical-composition gaps. When measured spectra are unavailable for the target molecules, chemically related reference spectra provide an alternative source of domain-specific information. Prior work on spectral representation learning and analogue search has shown that MS/MS neighborhoods can recover chemically related molecules (de Jonge et al., 2023; Huber et al., 2021a). This motivates test-time specialization that adapts the predictor to the queried domain.

![](images/74a597fa663fd028d19c50daf289895e9d3a9ccab11a64ef98f89047535f7b10.jpg)  
Figure 1: Overview of SPARC. Top: A predictor pretrained on a broad source collection is specialized to a target research domain. Bottom: Target molecules retrieve acquisition-compatible and chemically similar support spectra. The fragment generator remains frozen, while support reconstruction errors calibrate bin-wise reliability of an exponential moving average teacher. The student intensity model optimizes similarity-weighted support reconstruction and reliability-weighted teacher–student consistency, with stochastic restoration and rollback constraining adaptation drift.

Test-time training specializes a pretrained model during inference. In classification, TENT usesprediction entropy as a confidence signal, while CoTTA combines teacher averaging with stochastic restoration to limit error accumulation and forgetting (Wang et al., 2020; 2022). Transferring these ideas to MS/MS prediction, however, requires complementing confidence-based adaptation with an external spectral signal indicating how predictions change.

Spectral entropy summarizes the dispersion of normalized peak intensities (Li et al., 2021), but it does not encode whether peaks occur at the correct locations. Entropy can therefore decrease by suppressing weak peaks without improving the predicted fragmentation pattern. Moreover, target molecular structures identify the chemical region being queried but do not specify how the predicted spectrum should change. Chemically related measured reference spectra provide the missing external signal, while teacher predictions regularize the update and preserve reliable source-model structure.

We introduce SPARC (Spectral Prediction via Adaptation with Retrieval-calibrated Consistency), a retrieval-guided test-time training framework that combines support-guided specialization with drift-aware continual adaptation for pretrained MS/MS predictors (Fig. 1). For each target, SPARC retrieves chemically related labeled spectra using acquisition-aware prioritization and uses their measured intensities to supervise adaptation. In addition, SPARC uses reconstruction errors on retrieved support spectra to weight teacher–student consistency, relaxing teacher constraints in bins with larger support errors. An exponential moving average teacher, stochastic restoration and rollback mechanism constrain continual updates together.

We evaluate SPARC on MassSpecGym and NPLIB1 (Bushuiev et al., 2024; Duhrkop et al., 2021),¨ covering specialization to an adduct-defined subset and transfer between spectral libraries. We additionally curate five application-specific GNPS collections using a shared processing pipeline to represent focused scientific domains (Wang et al., 2016). These evaluations show robust improvements in the main benchmark transfer settings and higher entropy similarity across all application libraries. Peak-level analyses further characterize how adaptation improves predictions within the preserved candidate space, revealing increases in precision and F1-score at the peak and intensity level.

## 2 RELATED WORK

MS/MS Spectrum Prediction. Computational mass spectrometry encompasses both spectrum interpretation and molecule-to-spectrum prediction. SIRIUS combines isotope patterns and fragmentation trees for molecular formula and structure annotation (Duhrkop et al., 2019). Earlier deep¨ learning approaches focused on direct spectrum regression, including NEIMS for fingerprint-based EI prediction (Wei et al., 2019), 3DMolMS for 3D structure-based MS/MS prediction (Hong et al., 2023), and GrAFF-MS for graph-based high-resolution spectrum prediction (Murphy et al., 2023). More recently, explicit fragmentation models have further connected predicted peaks to molecular substructures. Iceberg combines autoregressive fragmentation-graph generation with fragment inten sity prediction (Goldman et al., 2024), whereas FIORA estimates fragment-ion probabilities from single-bond cleavages and their local molecular neighborhoods (Nowatzky et al., 2025). MassFormer uses graph transformers to model molecular structure, whereas FraGNNet introduces a structured probabilistic model for high-resolution spectrum prediction (Young et al., 2024a;b).

Test-Time Training. Test-time training updates a trained model using information available during inference. TENT minimizes prediction entropy by updating normalization parameters (Wang et al., 2020), while CoTTA supports non-stationary streams through an exponential-moving-average teacher, augmentation pseudo-labels, and stochastic restoration (Wang et al., 2022). Beyond computer vision, TAIP adapts interatomic potentials to out-of-distribution molecular configurations using global- and local-structure self-supervision (Cui et al., 2025). Test-time learning has also been explored in mass spectrometry for peptide-spectrum prediction (Ye et al., 2024) and de novo small-molecule generation from observed spectra (Mismetti et al., 2026).

Mass Spectral Libraries and Benchmarks. Public repositories such as GNPS, MassBank, and MetaboLights aggregate spectra across laboratories, scientific domains, and acquisition conditions, providing broad coverage but substantial heterogeneity (Horai et al., 2010; Wang et al., 2016; Yurekten et al., 2024). To support systematic machine learning evaluation, several curated benchmarks have since been developed. NPLIB1 provides a natural-product-oriented collection, MassSpecGym standardizes prediction and retrieval benchmarks, and MS<sup>n</sup>Lib and SpectraVerse broaden coverage across compounds, adducts, and ionization modes (Brungs et al., 2025; Bushuiev et al., 2024; Duhrkop¨ et al., 2021; Gupta et al., 2026). In addition, application-specific collections such as oxylipin libraries and GNPS community subsets reflect concrete scientific settings (Elloumi et al., 2024), from which we curate five application-grounded target domains for SPARC.

## 3 METHOD

We consider MS/MS spectrum prediction from a known molecular structure and its acquisition conditions. Let $\mathbf { x } = ( \mathcal { G } , \mathbf { c } )$ , where $\mathcal { G }$ is the molecular graph and c contains the available metadata, including precursor adduct, instrument type, and collision energy. The output is a non-negative intensity vector $\mathbf { y } \in \mathbb { R } _ { \geq 0 } ^ { B }$ over B fixed $m / z$ resolution bins.

A predictor $F _ { \omega }$ estimates the intensity vector as $\hat { \mathbf { y } } = F _ { \omega } ( \mathbf { x } )$ . At deployment, the target molecules and acquisition conditions may differ from those represented during source training. Given a pretrained predictor $F _ { \omega _ { 0 } } { : }$ , a target query $x _ { t } ,$ and a labeled reference library ${ \cal { S } } = \{ ( x _ { i } ^ { s } , y _ { i } ^ { s } ) \} _ { i = 1 } ^ { M }$ , our goal is to specialize the predictor using the query input and reference spectra without using the measured query spectrum in the adaptation objective.

We consider two deployment settings, both with access to a labeled reference library. In the targetwith-validation setting, labeled target-domain validation spectra guide update acceptance and checkpoint selection. In the target-only setting, no target validation labels are used, and updates are accepted by an entropy-based guard. Measured test-query spectra are reserved for evaluation only.

## 3.1 FRAGMENT-SPACE PRESERVING ADAPTATION

SPARC instantiates $F _ { \omega }$ with Iceberg, which separates discrete fragment generation from continuous intensity prediction (Goldman et al., 2024). Writing $\omega = ( \psi , \theta )$ for the parameters of these two

stages respectively, we have

$$
\mathcal { H } _ { \mathbf { x } } = G _ { \psi } ( \mathbf { x } ) , \qquad \hat { \mathbf { y } } = f _ { \theta } ( \mathbf { x } , \mathcal { H } _ { \mathbf { x } } ) .\tag{1}
$$

Here, $G _ { \psi }$ autoregressively constructs a directed acyclic graph fragmentation $\mathcal { H } _ { \mathbf { x } }$ of candidate fragments, and $f _ { \theta }$ predicts their contributions to the spectrum. The source checkpoint provides the initial parameters $( \psi _ { 0 } , \theta _ { 0 } )$ . Iceberg-Generate supplies a ranked set of candidate fragments to Iceberg-Score, allowing us to reuse the pretrained generator as a fixed fragmentation prior. Accordingly, SPARC fixes $G _ { \psi _ { 0 } }$ and adapts only θ. Conditioned on molecular and fragment representations and acquisition metadata, the intensity model uses chemically related reference spectra to learn target spectral patterns, emphasizing characteristic fragments and suppressing less relevant candidates. Adaptation therefore changes intensity allocation within the existing candidate space.

The adaptation pipeline uses retrieved spectra in two complementary ways. Their measured intensities directly supervise the student, while the reconstruction errors of an EMA teacher determine the binwise weights for consistency on the queries. Student and teacher states persist across successive queries, with stochastic restoration and rollback governing continual updates.

## 3.2 CONDITION-AWARE CHEMICAL RETRIEVAL

SPARC constructs the chemical-local support neighborhood for each query, so the supervision follows the chemical region being processed. For a molecule x, let $\phi ( \mathbf { x } )$ be the $\ell _ { 2 }$ -normalized concatenation of its Morgan and hashed AtomPair bit fingerprints (Carhart et al., 1985; Rogers & Hahn, 2010). Each reference spectrum is scored by its fingerprint cosine similarity to the target molecule.

$$
r _ { t , i } = \frac { \phi ( \boldsymbol { x _ { t } } ) ^ { \top } \phi ( \boldsymbol { x _ { i } ^ { s } } ) } { \lVert \phi ( \boldsymbol { x _ { t } } ) \rVert _ { 2 } \lVert \phi ( \boldsymbol { x _ { i } ^ { s } } ) \rVert _ { 2 } }\tag{2}
$$

Acquisition compatibility takes precedence over chemical ranking. With the default adduct-based retrieval, references matching the query adduct form the candidate pool. Within this pool, references also matching the query instrument are ranked first, followed by the remaining adduct-compatible references. Each group is sorted by $r _ { t , i }$ , and the first $K$ entries form $S _ { t }$ . The implementation falls back to global retrieval only when the adduct-compatible pool is empty. Collision energy and precursor $m / z$ remain model inputs but are not hard retrieval filters.

Retrieved spectra contribute according to their chemical relevance. Writing $K _ { t } = | S _ { t } |$ and reindexing locally reindexing the entries of $S _ { t }$ , we define

$$
a _ { t , i } = \frac { \exp ( r _ { t , i } / \tau ) } { \sum _ { j = 1 } ^ { K _ { t } } \exp ( r _ { t , j } / \tau ) } , \qquad \tau > 0 ,\tag{3}
$$

and minimize the similarity-weighted support reconstruction loss

$$
\mathcal { L } _ { \mathrm { s u p } } ( \theta ; S _ { t } ) = \frac { 1 } { B } \sum _ { i = 1 } ^ { K _ { t } } a _ { t , i } \left\| f _ { \theta } \big ( \mathbf { x } _ { i } ^ { s } \big ) - \mathbf { y } _ { i } ^ { s } \right\| _ { 2 } ^ { 2 } .\tag{4}
$$

This term anchors adaptation in measured spectra rather than relying exclusively on the model’s own predictions.

## 3.3 SUPPORT-CALIBRATED TEACHER CONSISTENCY

The teacher supplies a target prediction, but its reliability need not be uniform across the spectrum (Tarvainen & Valpola, 2017). We initialize the student θ and teacher <sup>¯</sup>θ from $\theta _ { 0 }$ . Before each update, we evaluate the teacher on the retrieved labeled spectra and estimate an error profile across $m / z$ bins:

$$
e _ { t , b } = \frac { 1 } { K _ { t } } \sum _ { i = 1 } ^ { K _ { t } } [ f _ { \bar { \theta } } ( \mathbf { x } _ { i } ^ { s } ) _ { b } - y _ { i , b } ^ { s } ] ^ { 2 } , \qquad c _ { t , b } = \exp ( - \gamma e _ { t , b } ) , \gamma \geq 0 .\tag{5}
$$

Support errors are averaged uniformly over the retrieved set, whereas retrieval-similarity weights modulate the supervised reconstruction loss. Chemical locality enters the error estimate through support selection. The resulting $c _ { t , b }$ is computed without gradients and controls the strength of teacher

consistency on the shared $m / z$ grid: larger support reconstruction errors yield weaker constraints. We use these weights as a support-derived regularization signal, rather than as calibrated probabilities of correctness for individual query fragments.

We regularize the student toward the unperturbed teacher prediction, reducing pressure in bins that the teacher reconstructs poorly:

$$
\mathcal { L } _ { \mathrm { c o n } } = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } c _ { t , b } \left[ f _ { \theta } ( x _ { t } ) _ { b } - \mathrm { s g } ( f _ { \bar { \theta } } ( x _ { t } ) _ { b } ) \right] ^ { 2 }\tag{6}
$$

where sg denotes stop-gradient. The core adaptation objective is

$$
{ \mathcal { L } } _ { \mathrm { a d a p t } } = { \mathcal { L } } _ { \mathrm { c o n } } + \lambda _ { \mathrm { s u p } } { \mathcal { L } } _ { \mathrm { s u p } }\tag{7}
$$

where $\lambda _ { s u p }$ controls the weight of direct support supervision. The measured support spectra drive student updates, while teacher consistency limits changes on the current query according to reliability. These weights determine how student model is constrained by the teacher at each bin.

## 3.4 CONTINUAL UPDATES AND DRIFT CONTROL

Mechanisms for stabilizing continual updates. SPARC uses two CoTTA-inspired mechanisms to stabilize continual updates, an EMA teacher and stochastic restoration (Wang et al., 2022). For each query and its retrieved support set, the student takes gradient steps on Eq. (7), and the accepted state is carried to the next query. At inner step u, the optimizer first produces a trial student state $\theta _ { u } ^ { + }$ ; the teacher then tracks this state according to

$$
\bar { \theta } _ { u + 1 } = \alpha \bar { \theta } _ { u } + ( 1 - \alpha ) \theta _ { u } ^ { + } .\tag{8}
$$

The EMA decay coefficient α is fixed in our implementation. After the teacher update, each trainable student weight or bias entry is independently restored to its source value with probability $\rho \colon$

$$
m _ { u , j } \sim \mathrm { B e r n o u l l i } ( \rho ) ,
$$

$$
\theta _ { u + 1 , j } = m _ { u , j } \theta _ { 0 , j } + ( 1 - m _ { u , j } ) \theta _ { u , j } ^ { + } .\tag{9}
$$

Thus, the teacher averages the optimized student before restoration, whereas restoration anchors the student to the pretrained parameters. This ordering smooths the teacher trajectory while limiting cumulative drift in the model that is updated online.

Supervised and unsupervised rollback. Besides stabilizer, SPARC controls the remaining drift with a rollback rule whose evidence depends on whether labeled validation spectra are available. Before each trial update, we snapshot the whole model and optimizer states.

When validation spectra are available, a trial update is accepted only if its post-update validation CosSim remains at least η times the pre-update value. This comparison uses validation spectra for the rollback decision, while the adaptation loss remains unchanged.

When validation spectra are unavailable, the same rollback operation uses a label-free entropy guard instead. For a predicted intensity vector $\mathbf { z } ,$ define normalized spectral entropy as

$$
\begin{array} { l } { { \displaystyle p _ { b } ( { \bf z } ) = \frac { \operatorname* { m a x } ( z _ { b } , 0 ) } { \sum _ { j = 1 } ^ { B } \operatorname* { m a x } ( z _ { j } , 0 ) + \epsilon } } , } \\ { { \displaystyle h ( { \bf z } ) = - \frac { 1 } { \log B } \sum _ { b = 1 } ^ { B } p _ { b } ( { \bf z } ) \log [ p _ { b } ( { \bf z } ) + \epsilon ] } . } \end{array}\tag{10}
$$

Let $\bar { y } _ { u } ^ { - }$ and $\bar { y } _ { u } ^ { + }$ denote the unperturbed teacher predictions for the current query before and after the trial update, respectively. We define

$$
H _ { u } ^ { - } = h ( \bar { y } _ { u } ^ { - } ) , \qquad H _ { u } ^ { + } = h ( \bar { y } _ { u } ^ { + } ) .\tag{11}
$$

We reject the update when $H _ { u } ^ { + } - H _ { u } ^ { - } > \delta$ . This label-free guard rejects updates that abruptly broaden the predicted intensity distribution, providing a lightweight stability criterion.

## 4 EXPERIMENTS

Datasets and benchmarks Following prior work, we first evaluate SPARC on MassSpecGym (MSG) (Bushuiev et al., 2024) and NPLIB1 (Duhrkop et al., 2021). We consider adduct-focused¨ specialization from the full MSG dataset to MSG-MNa subset and cross-library transfer from MSG to NPLIB1. MSG→MSG-MNa is an adduct-defined subset specialization, as the full-MSG source checkpoint includes M+Na examples. We further use five experimental libraries on GNPS: Drugs of Abuse, 3-Hydroxy Acyl Amides, ECG Acyl Amides C4–C24, Alkylamines–Bile Acids, and SelleckChem FDA. These collections span forensic toxicology, acyl-amide and lipid chemistry, bile-acid conjugates, and approved pharmaceuticals. For all five GNPS target libraries, the MSG training split serves as the labeled reference support library, while target spectra are reserved only for evaluation. To prevent information leakage, Murcko scaffolds were computed for molecules, and support molecules sharing a scaffold with any validation or test molecule were excluded before retrieval (Bemis & Murcko, 1996). Complete preprocessing setting are provided in Supplementary.

Evaluation We evaluate spectral fidelity using cosine similarity (CosSim), Jensen–Shannon entropy similarity (EntSim), and mean squared error (MSE), with entropy as an auxiliary diagnostic. Before scoring, we retain the 100 highest-intensity predicted bins. Comparisons use the successful intersection within methods, and coverage is reported in Supplementary. We additionally evaluate peak-presence and ion-current quality using precision, recall, and F1 after exact-bin matching, with peak- and intensity-level metrics computed from matched prediction and ground-truth.

Implementation details and baselines SPARC retrieves 64 support spectra for each query. During adaptation, SPARC optimizes its objective with Adam using a learning rate of $1 \times \mathrm { 1 0 ^ { - 4 } }$ . Each molecular representation concatenates 2,048 radius-2 Morgan fingerprint bits with 3,072 hashed atom-pair bits. We set the support-loss coefficient to $\lambda _ { \operatorname* { s u p } } = 1$ and use $\tau = 0 . 1 , \gamma = 1 0 , \alpha = 0 . 9 9$ $\rho = 0 . 0 2$ , and $\delta = 0 . 1$ for the remaining adaptation and regularization terms. Finally, SPARC use student model for prediction. We implement all models on NVIDIA A800 (80GB) GPUs.

We compare against NEIMS, GrAFF-MS, FIORA and pretrained Iceberg models as predictor baselines, and TENT and CoTTA on Iceberg as source-free adaptation baselines. Architectural details are reported in Supplementary.

Adaptation protocols. For the MSG and NPLIB1 benchmarks, labeled target-training spectra provide support supervision. SPARC(val) specializes the source checkpoint using target-validation spectra for rollback and checkpoint selection, and is then evaluated without test-time updates. SPARC– TTT starts from the same selected checkpoint and continues adapting on test-query structures and acquisition metadata using entropy-based rollback. The GNPS experiments instead omit validation specialization and use MSG-training spectra as the reference library, with entropy-based rollback throughout online adaptation. Measured test-query spectra are never used in adaptation losses, rollback decisions, or checkpoint selection.

## 5 RESULTS

5.1 PREDICTION HETEROGENEITY AND SPECTRAL DOMAIN SHIFTS.

Cross-adduct shift changes both chemistry and fragmentation. In MassSpecGym, only 19.0% of molecules with M+H spectra also have M+Na spectra. These molecules are enriched for N-free, polyol/carbohydrate-like, and O-rich structures, but depleted for basic amines, aromatic nitrogen, and sulfone/sulfonamides, consistent with known adduct preferences (Kruve et al., 2013). Even among matched molecules, over 50% share no peaks and few neutral losses, reflecting strong adductdependent fragmentation (de Jonge et al., 2026; Liu et al., 2025). M+Na spectra also show lower entropy. Detailed analysis are provided in Supplementary.

Locality motivates retrieval. We next ask whether the prediction error is uniform. We visualize this by projecting DreaMS embeddings and molecule fingerprints of dataset samples into a twodimensional UMAP and coloring each sample by its prediction performance (Bushuiev et al., 2026). Regions of high and low accuracy occur within every split and in both molecule and spectra views (Fig. 2). This qualitative pattern is supported in Supplementary. Because errors cluster locally, fine-tuning on the whole support set can improve performance across the entire local neighborhood, turning test-time training into genuine domain specialization (Huber et al., 2021b).

![](images/ba2b6cbcf525519e53e1ed8b6e7dfe2b55ef6d13a804197e84a3f196fb3cb799.jpg)  
Figure 2: Prediction performance is locally structured in spectral and molecule spaces.

## 5.2 SPARC SPECIALIZES ACROSS AND WITHIN DOMAINS

We first evaluate SPARC under explicit domain shifts. On MSG→MSG-MNa, SPARC raises EntSim and CosSim from 0.180/0.280 for the pretrained MSG checkpoint to 0.274/0.370 after validationstage specialization, with SPARC–TTT further reaching a CosSim of 0.376 (Table 1). The adapted model also exceeds the Iceberg model trained directly on M+Na, which obtains 0.262/0.348. SPARC therefore recovers a substantial part of the adduct-specific prediction gap through reference-guided specialization of the pretrained model.

Table 1: Test performance for MSG→MSG-MNa and MSG→NPLIB1 transfer.
<table><tr><td>Method Entropy</td></tr><tr><td>MSE EntSim CosSim MSG→MSG-MNa (MSG initialized)</td></tr><tr><td>NEIMS 4.150 14.461 0.072 0.072</td></tr><tr><td>GrAFF-MS 4.134 15.822 0.075 0.071</td></tr><tr><td>FIORA 3.015 1.020 0.063 0.071</td></tr><tr><td>Iceberg-MSG 3.461 1.242 0.180 0.280</td></tr><tr><td>Iceberg-MNa 2.485 1.006 0.262 0.348</td></tr><tr><td>SPARC(val) 2.715 0.895 0.274 0.370</td></tr><tr><td>SPARC-TTT 2.923 0.882 0.284 0.376</td></tr><tr><td></td></tr><tr><td>Iceberg(tent) 2.077 1.183 0.229 0.290 Iceberg(cotta) 3.419 1.189 0.191 0.288</td></tr></table>

<table><tr><td>Method Entropy MSE EntSim</td></tr><tr><td>CosSim MSG→NPLIB1 (MSG initialized)</td></tr><tr><td>NEIMS 3.568 20.246 0.392 0.363</td></tr><tr><td>GrAFF-MS 3.396 21.170 0.330 0.280</td></tr><tr><td>FIORA 1.900 1.317 0.369 0.412</td></tr><tr><td>Iceberg-MSG 3.586 1.269 0.470 0.517</td></tr><tr><td>Iceberg-NPLIB1 3.516 1.217 0.521 0.586</td></tr><tr><td>SPARC(val) 3.112 1.063 0.559 0.630</td></tr><tr><td>SPARC-TTT 3.128 1.041 0.575 0.650</td></tr><tr><td></td></tr><tr><td>Iceberg(tent) 3.052 1.252 0.490 0.517 Iceberg(cotta) 3.560 1.245 0.472 0.518</td></tr></table>

Starting from the MSG checkpoint, specialization using NPLIB1 training spectra as support and validation-guided rollback and checkpoint selection improves EntSim and CosSim from 0.470/0.517 to 0.559/0.630 (Table 1). Across both transfer settings, validation-guided specialization provides the main performance gain, with test-stream continuation yielding additional refinement as target queries arrive. Thus, the benefit is not confined to the adduct subset but persists when transferring to a distinct library. A separate controlled comparison matches source initialization, target-training support access, validation protocol, and held-out test evaluation to isolate the benefit beyond retrieval-only fine-tuning. On NPLIB1, similarity-weighted retrieved-support fine-tuning achieves 0.5873 CosSim, compared with 0.5545 for global fine-tuning, supporting the value of chemically conditioned training beyond access to labeled target-domain spectra alone.

Table 2: Test performance for MSG M+H and NPLIB1.
<table><tr><td>Method</td><td>Entropy</td><td>MSE</td><td>EntSim</td><td>CosSim</td><td>Method</td><td>Entropy</td><td>MSE</td><td>EntSim</td><td>CosSim</td></tr><tr><td colspan="5">MSG M+H (MSG initialized)</td><td colspan="5">NPLIB1 (NPLIB1 initialized)</td></tr><tr><td>NEIMS</td><td>3.982</td><td>16.180</td><td>0.184</td><td>0.162</td><td>NEIMS</td><td>3.349</td><td>20.652</td><td>0.324</td><td>0.280</td></tr><tr><td>GrAFF-MS</td><td>3.756</td><td>16.713</td><td>0.169</td><td>0.137</td><td>GrAFF-MS</td><td>3.521</td><td>19.689</td><td>0.384</td><td>0.362</td></tr><tr><td>FIORA</td><td>1.742</td><td>0.917</td><td>0.434</td><td>0.460</td><td>FIORA</td><td>1.596</td><td>1.237</td><td>0.419</td><td>0.484</td></tr><tr><td>Iceberg-MSG</td><td>3.477</td><td>0.853</td><td>0.462</td><td>0.529</td><td>Iceberg-NPLIB1</td><td>3.516</td><td>1.217</td><td>0.521</td><td>0.586</td></tr><tr><td>SPARC(val)</td><td>2.866</td><td>0.907</td><td>0.494</td><td>0.532</td><td>SPARC(val)</td><td>2.925</td><td>1.107</td><td>0.559</td><td>0.616</td></tr><tr><td>SPARC-TTT</td><td>2.794</td><td>0.880</td><td>0.505</td><td>0.547</td><td>SPARC-TTT</td><td>2.944</td><td>1.092</td><td>0.565</td><td>0.628</td></tr></table>

SPARC can also refine predictors that are already matched. We apply the same adaptation procedure within MSG M+H and within NPLIB1, where the backbone was trained on the target domain itself (Table 2). On the dominant MSG M+H subset, SPARC increases EntSim and CosSim from 0.462/0.529 to 0.505/0.547, and increases EntSim and CosSim from 0.521/0.586 to 0.565/0.628 on NPLIB1. These within-domain gains show that retrieved reference spectra provide a useful local refinement signal even after the predictor has already learned the target-domain distribution, making SPARC a test-time specialization stage rather than only a mechanism for correcting domain shifts.

## 5.3 SPARC IMPROVES SPECTRAL PURITY BY SUPPRESSING SPURIOUS PEAKS

We next examine how SPARC redistributes intensity within the frozen fragmentation space. For peak-quality evaluation on M+H and M+Na, we follow the Iceberg evaluation setting: retain the 100 highest-intensity predicted bins, remove peaks below 1% of the predicted base-peak intensity, and match the remaining bins to the experimental spectrum. Both SPARC and SPARC–TTT improve intensity-weighted precision and F1 (Table 3), concentrating more predicted intensity on experimentally supported peaks. We further rank the saved M+Na top-100 predictions by intensity and assess agreement with experimental peak support and intensities using average precision (AP), NDCG, and Spearman correlation on matched bins. Improvements across these measures indicate better prioritization of supported peaks and more faithful relative intensity ordering.

Besides, we compare Iceberg and the validation-selected SPARC checkpoint over the complete candidate space, progressively adding peaks in descending predicted-intensity order until matched experimental peaks account for 25%, 50%, 75%, or 100% of the experimental ion current recoverable within the frozen candidate space. We compare purity at each coverage level and find that purity increases at full recoverable coverage, showing that less unsupported predicted intensity accompanies comparable experimental signal recovery.

Table 3: Peak quality, candidate ranking, and spectral purity of Iceberg and SPARC.
<table><tr><td>Method</td><td>MSG M+H Peak Prec. Recall</td><td>MSG M+H Inten Prec. Recall</td><td>F1</td><td>MSG M+Na Peak Prec. Recall</td><td>F1</td><td>Prec. Recall</td><td>MSG M+Na Inten F1</td></tr><tr><td>Iceberg SPARC SPARC-TTT</td><td>0.203 0.550 0.264 0.449 0.271 0.438</td><td>0.297 0.465 0.333 0.581 0.335 0.593</td><td>0.756 0.576 0.691 0.632 0.687 0.637</td><td>0.039 0.061 0.045</td><td>0.159 0.062 0.097 0.075 0.112 0.074</td><td>0.153 0.361 0.340</td><td>0.3140.206 0.2970.326 0.312 0.327</td></tr><tr><td>Method</td><td>AP N@10 N@50 Spear</td><td>MSG M+Na Ranking</td><td>P@25 P@50 P@75 P@100</td><td>MSG M+Na Purity</td><td>P@25</td><td>MSG M+H Purity</td><td>P@50 P@75 P@100</td></tr><tr><td>Iceberg</td><td>|0.402 0.508</td><td>0.567 0.268</td><td>0.459 0.444</td><td>0.414</td><td>0.383</td><td>0.804 0.765</td><td>0.722 0.664</td></tr><tr><td>SPARC SPARC-TTT</td><td>0.474 0.612 0.485 0.624</td><td>0.6480.409 0.664 0.410</td><td>0.561 0.573</td><td>0.569</td><td>0.567</td><td>0.809 0.771</td><td>0.740 0.699</td></tr></table>

Fig. 3 illustrates this effect: the source prediction assigns substantial intensity to low-m/z peaks, whereas the adapted predictions suppress many of these peaks and concentrate relative intensity on the dominant peak. Together, these results show that useful target-domain corrections can be made within the existing candidate space by changing which fragments receive substantial intensity.

![](images/ea84a6e5972bb9ae02c75676d7774682ea5cffc7f2932f60714d16316567ec7a.jpg)  
Figure 3: Representative example of prediction from SPARC. Each displayed spectrum is normalized, and peaks with intensities below 1% are removed.

## 5.4 SPARC TRANSFERS TO APPLICATION-SPECIFIC TARGET DOMAINS

We next evaluate target-only specialization on five application-specific GNPS libraries using MSGtraining spectra as references, without a validation-specialization stage. Retrieval follows the acquisition-aware policy in Sec. 3.2, and online updates use entropy-based rollback.

Measured target-library spectra are reserved for evaluation. SPARC achieves the best EntSim among the compared predictors on every library and the best scores across all three reported metrics on GNPS-A/B and SC-FDA. These results demonstrate specialization to application-defined query streams using an external reference library.

Table 4: Performance on Drugs of Abuse, 3HAA, ECG, GNPS-A/B, and SC-FDA. All models are initially trained on the MSG dataset.
<table><tr><td rowspan="2">Method</td><td rowspan="2">MSE EntSim</td><td colspan="2">DoA</td><td rowspan="2">MSE</td><td colspan="2">3HAA</td><td rowspan="2">MSE</td><td colspan="2">ECG</td><td colspan="2">GNPS-A/B</td><td rowspan="2"></td><td colspan="2">SC-FDA Cos</td></tr><tr><td></td><td> $\mathbf { C o s } \vert$ </td><td>EntSim</td><td> $\mathbf { C o s } \vert$ </td><td>EntSim</td><td> ${ \mathrm { C o s } }$ </td><td>MSE EntSim</td><td> ${ \mathrm { C o s } }$ </td><td>MSE</td><td>EntSim</td></tr><tr><td>NEIMS</td><td>19.90</td><td>0.198</td><td>0.127</td><td>12.84</td><td>0.386</td><td>0.382 34.33</td><td></td><td>0.184 0.145</td><td>18.05</td><td>0.409</td><td>0.242</td><td>14.75</td><td>0.148</td><td>0.110</td></tr><tr><td>GrAFF-MS</td><td>19.98</td><td>0.149</td><td>0.084</td><td>15.23</td><td>0.238</td><td>0.173</td><td>34.49</td><td>0.145</td><td>0.117</td><td>19.84</td><td>0.324 0.179</td><td>15.57</td><td>0.157</td><td>0.107</td></tr><tr><td>FIORA</td><td>1.078</td><td>0.427</td><td>0.458</td><td>0.935</td><td>0.395</td><td>0.397</td><td>2.836</td><td>0.140 0.117</td><td>1.009</td><td>0.393</td><td>0.488</td><td>0.893</td><td>0.362</td><td>0.405</td></tr><tr><td>Iceberg</td><td>1.057</td><td>0.415</td><td>0.483</td><td>0.904</td><td>0.490</td><td>0.509</td><td>2.621</td><td>0.178 0.223</td><td>0.960</td><td></td><td>0.504 0.511</td><td>0.925</td><td>0.322</td><td>0.379</td></tr><tr><td>SPARC</td><td>1.058</td><td>0.482</td><td>0.509</td><td>0.901</td><td>0.508</td><td>0.514</td><td>2.654</td><td>0.205 0.246</td><td>0.919</td><td>0.534</td><td>0.569</td><td>0.839</td><td>0.389</td><td>0.435</td></tr><tr><td>w/o rollback | 1.068</td><td></td><td>0.451</td><td>0.488 |0.940</td><td></td><td>0.493</td><td>0.497 |2.631</td><td></td><td>0.202</td><td>0.252|1.109</td><td>0.516</td><td></td><td>0.536|0.920</td><td>0.367</td><td>0.391</td></tr></table>

## 5.5 COMPONENT ABLATION

Table 5 compares component ablations under adaptation on the MSG → MSG-MNa transfer. All tested ablations reduce CosSim relative to the full method. Removing similarity-based support retrieval produces the largest decrease, whereas removing stochastic restoration has the smallest effect. These comparisons identify similarity-based support use as the most influential component among the tested ablations.

These results support the roles of chemically relevant references, direct support supervision, and stochastic restoration in continual adaptation. Removing similarity-based support produces the clearest degradation, indicating that adaptation depends largely on local chemical rel-

Table 5: Ablation Study of SPARC Components.
<table><tr><td>Method</td><td>MSE EntSim</td><td>CosSim</td></tr><tr><td>-similarity-based support retrieval</td><td>0.9859 0.2595</td><td>0.3441</td></tr><tr><td>-support-calibrated bin confidence</td><td>0.8889 0.2637</td><td>0.3514</td></tr><tr><td>-stochastic restoration</td><td>0.8840 0.2720</td><td>0.3692</td></tr><tr><td>-per-query rollback</td><td>0.8977 0.2656</td><td>0.3631</td></tr><tr><td>SPARC</td><td>0.8804 0.2742</td><td>0.3755</td></tr></table>

evance. Removing stochastic restoration leads to the smallest decrease in CosSim, suggesting that this mechanism is not uniformly necessary and that allowing the adapted model to move farther from the source parameters may be beneficial for mass spectrum prediction.

## 6 LIMITATIONS

SPARC has two limitations. First, it is bounded by the fragmentation space inherited from the pretrained predictor. Freezing the fragment generator stabilizes adaptation but restricts SPARC to reweighting candidate fragments and it can’t recover an omitted diagnostic fragment. Jointly expanding the candidate space without destabilizing online adaptation can be an important direction for domains with unseen fragmentation pathways.

Second, SPARC assumes that chemically similar, acquisition-compatible support spectra provide transferable supervision. Poor support coverage or a fingerprint-caused mismatch can bias both the supervised update and the support-derived reliability estimate. Continual updates further introduce dependence on target-stream order, while entropy-based rollback limits abrupt drift without guaranteeing correct peak locations. Query-specific uncertainty estimates and order-robust safeguards aligned more directly with spectral fidelity could mitigate these limitations.

## 7 CONCLUSION

We introduced SPARC, a retrieval-guided test-time specialization framework for pretrained MS/MS predictors. SPARC preserves the fragmentation space and adapts fragment intensities using chemically related reference spectra, with support reconstruction errors weighting teacher consistency. Validation-assisted specialization improves benchmark spectral similarity, with further gains from online continuation. Without validation specialization, entropy-guarded adaptation using MSG references improves EntSim across all GNPS application libraries. Peak-level analyses associate these gains with higher spectral purity and intensity redistribution within the existing candidate space. Together, these results support reference-guided specialization as an extension of pretrained predictors when compatible labeled reference spectra are available.

## REFERENCES

Guy W. Bemis and Mark A. Murcko. The properties of known drugs. 1. molecular frameworks. Journal ofMedicinal Chemistry, 39(15):2887–2893, 1996. doi: 10.1021/jm9602928.

Parker Ladd Bremer, Arpana Vaniya, Tobias Kind, Shunyang Wang, and Oliver Fiehn. How well can we predict mass spectra from structures? benchmarking competitive fragmentation modeling for metabolite identification on untrained tandem mass spectra. Journal ofChemical Information and Modeling, 62(17):4049–4056, 2022. doi: 10.1021/acs.jcim.2c00936.

Corinna Brungs, Robin Schmid, Steffen Heuckeroth, et al. MS<sup>n</sup>Lib: Efficient generation of open multi-stage fragmentation mass spectral libraries. Nature Methods, 22:2028–2031, 2025. doi: 10.1038/s41592-025-02813-0.

Roman Bushuiev, Anton Bushuiev, Niek F de Jonge, Adamo Young, Fleming Kretschmer, Raman Samusevich, Janne Heirman, Fei Wang, Luke Zhang, Kai Duhrkop, et al. Massspecgym: A¨ benchmark for the discovery and identification of molecules. Advances in Neural Information Processing Systems, 37:110010–110027, 2024.

Roman Bushuiev, Anton Bushuiev, Raman Samusevich, Corinna Brungs, Josef Sivic, and Toma´sˇ Pluskal. Self-supervised learning of molecular representations from millions of tandem mass spectra using dreams. Nature Biotechnology, 44(4):630–640, 2026.

Raymond E. Carhart, Dennis H. Smith, and R. Venkataraghavan. Atom pairs as molecular features in structure-activity studies: Definition and applications. Journal of Chemical Information and Computer Sciences, 25(2):64–73, 1985. doi: 10.1021/ci00046a002.

Taoyong Cui, Chenyu Tang, Dongzhan Zhou, Yuqiang Li, Xingao Gong, Wanli Ouyang, Mao Su, and Shufei Zhang. Online test-time adaptation for better generalization of interatomic potentials to outof-distribution data. Nature Communications, 16:1891, 2025. doi: 10.1038/s41467-025-57101-4.

Niek F. de Jonge, Joris J. R. Louwen, Elena Chekmeneva, Stephane Camuzeaux, Femke J. Vermeir, Robert S. Jansen, Florian Huber, and Justin J. J. van der Hooft. MS2Query: Reliable and scalable MS2 mass spectra-based analogue search. Nature Communications, 14:1752, 2023. doi: 10.1038/s41467-023-37446-4.

Niek F. de Jonge, Elena Chekmeneva, Robin Schmid, David Joas, Lem-Joe Truong, Justin J. J. van der Hooft, and Florian Huber. Cross ionization mode chemical similarity prediction between tandem mass spectra in metabolomics. Nature Communications, 17:2483, 2026.

Kai Duhrkop, Markus Fleischauer, Marcus Ludwig, Alexander A. Aksenov, Alexey V. Melnik,¨ Marvin Meusel, Pieter C. Dorrestein, Juho Rousu, and Sebastian Bocker. SIRIUS 4: A rapid tool¨ for turning tandem mass spectra into metabolite structure information. Nature Methods, 16(4): 299–302, 2019. doi: 10.1038/s41592-019-0344-8.

Kai Duhrkop, Louis-F¨ elix Nothias, Markus Fleischauer, et al. Systematic classification of unknown´ metabolites using high-resolution fragmentation mass spectra. Nature Biotechnology, 39(4): 462–471, 2021. doi: 10.1038/s41587-020-0740-8.

Anis Elloumi, Lindsay Mas-Normand, Jamie Bride, et al. From MS/MS library implementation to molecular networks: Exploring oxylipin diversity with NEO-MSMS. Scientific Data, 11:193, 2024. doi: 10.1038/s41597-024-03034-4.

Samuel Goldman, Janet Li, and Connor W. Coley. Generating molecular fragmentation graphs with autoregressive neural networks. Analytical Chemistry, 96(8):3419–3428, 2024. doi: 10.1021/acs. analchem.3c04654.

Vishu Gupta, Hantao Qiang, Hsin-Hsiang Chung, Ehud Herbst, and Michael A. Skinnider. Comprehensive curation and harmonization of small-molecule MS/MS libraries in Spectraverse. Analytical Chemistry, 98(5):3934–3943, 2026. doi: 10.1021/acs.analchem.5c06256.

Corey Hoang, Winnie Uritboonthai, Linh Hoang, Elizabeth M. Billings, Aries Aisporna, Farshad A. Nia, Rico J. E. Derks, James R. Williamson, Martin Giera, and Gary Siuzdak. Tandem mass spectrometry across platforms. Analytical Chemistry, 96(14):5478–5488, 2024. doi: 10.1021/acs. analchem.3c05576.

Yuhui Hong, Sujun Li, Christopher J Welch, Shane Tichy, Yuzhen Ye, and Haixu Tang. 3dmolms: prediction of tandem mass spectra from 3d molecular conformations. Bioinformatics, 39(6): btad354, 2023. doi: 10.1093/bioinformatics/btad354.

Hisayuki Horai, Masanori Arita, Shigehiko Kanaya, et al. MassBank: A public repository for sharing mass spectral data for life sciences. Journal of Mass Spectrometry, 45(7):703–714, 2010. doi: 10.1002/jms.1777.

Florian Huber, Lars Ridder, Stefan Verhoeven, Jurriaan H. Spaaks, Faruk Diblen, Simon Rogers, and Justin J. J. van der Hooft. Spec2Vec: Improved mass spectral similarity scoring through learning of structural relationships. PLOS Computational Biology, 17(2):e1008724, 2021a. doi: 10.1371/journal.pcbi.1008724.

Florian Huber, Sven van der Burg, Justin J. J. van der Hooft, and Lars Ridder. MS2DeepScore: A novel deep learning similarity measure to compare tandem mass spectra. Journal ofCheminformatics, 13:84, 2021b. doi: 10.1186/s13321-021-00558-4.

Anneli Kruve, Karl Kaupmees, Jaanus Liigand, Merit Oss, and Ivo Leito. Sodium adduct formation efficiency in ESI source. Journal of Mass Spectrometry, 48(6):695–702, 2013. doi: 10.1002/jms. 3218.

Yuanyue Li, Tobias Kind, Jacob Folz, Arpana Vaniya, Sajjan Singh Mehta, and Oliver Fiehn. Spectral entropy outperforms MS/MS dot product similarity for small-molecule compound identification. Nature Methods, 18(12):1524–1531, 2021. doi: 10.1038/s41592-021-01331-z.

Botao Liu, Zhifeng Tang, and Tao Huan. Adduct-induced variability in tandem mass spectrometry. Analytical Chemistry, 97(31):17058–17066, 2025. doi: 10.1021/acs.analchem.5c02792.

Laura Mismetti, Marvin Alberts, Andreas Krause, and Mara Graziani. Test-time tuned language models enable end-to-end de novo molecular structure generation from MS/MS spectra. arXiv preprint arXiv:2510.23746, 2026. URL https://arxiv.org/abs/2510.23746.

Michael Murphy, Stefanie Jegelka, Ernest Fraenkel, Tobias Kind, David Healey, and Thomas Butler. Efficiently predicting high resolution mass spectra with graph neural networks. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pp. 25549–25562. PMLR, 2023.

Yannek Nowatzky, Francesco Friedrich Russo, Jan Lisec, Alexander Kister, Knut Reinert, Thilo Muth, and Philipp Benner. Fiora: Local neighborhood-based prediction of compound mass spectra from single fragmentation events. Nature Communications, 16:2298, 2025. doi: 10.1038/ s41467-025-57422-4.

David Rogers and Mathew Hahn. Extended-connectivity fingerprints. Journal of Chemical Information and Modeling, 50(5):742–754, 2010. doi: 10.1021/ci100050t.

Robin Schmid, Daniel Petras, Louis-Felix Nothias, Mingxun Wang, Allegra T. Aron, et al. Ion identity molecular networking for mass spectrometry-based metabolomics in the GNPS environment. Nature Communications, 12:3832, 2021. doi: 10.1038/s41467-021-23953-9.

Emma L. Schymanski, Junho Jeon, Rebekka Gulde, Kathrin Fenner, Matthias Ruff, Heinz P. Singer, and Juliane Hollender. Identifying small molecules via high resolution mass spectrometry: Communicating confidence. Environmental Science & Technology, 48(4):2097–2098, 2014. doi: 10.1021/es5002105.

Antti Tarvainen and Harri Valpola. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. In Advances in Neural Information Processing Systems, volume 30, 2017.

Dequan Wang, Evan Shelhamer, Shaoteng Liu, Bruno Olshausen, and Trevor Darrell. Tent: Fully test-time adaptation by entropy minimization. arXiv preprint arXiv:2006.10726, 2020.

Mingxun Wang, Jeremy J. Carver, Vanessa V. Phelan, et al. Sharing and community curation of mass spectrometry data with global natural products social molecular networking. Nature Biotechnology, 34(8):828–837, 2016. doi: 10.1038/nbt.3597.

Qin Wang, Olga Fink, Luc Van Gool, and Dengxin Dai. Continual test-time domain adaptation. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 7191–7201. IEEE, 2022.

Jennifer N Wei, David Belanger, Ryan P Adams, and D Sculley. Rapid prediction of electron– ionization mass spectrometry using neural networks. ACS Central Science, 5(4):700–708, 2019. doi: 10.1021/acscentsci.9b00085.

Jianbai Ye, Xiangnan He, Shujuan Wang, Meng-Qiu Dong, Feng Wu, Shan Lu, and Fuli Feng. Test-time training for deep MS/MS spectrum prediction improves peptide identification. Journal ofProteome Research, 23(2):550–559, 2024. doi: 10.1021/acs.jproteome.3c00229.

Adamo Young, Hannes Rost, and Bo Wang. Tandem mass spectrum prediction for small molecules¨ using graph transformers. Nature Machine Intelligence, 6(4):404–416, 2024a. doi: 10.1038/ s42256-024-00816-8.

Adamo Young, Fei Wang, David S Wishart, Bo Wang, Russell Greiner, and Hannes Rost. Fragnnet: a¨ deep probabilistic model for tandem mass spectrum prediction. arXiv preprint arXiv:2404.02360, 2024b.

Ozgur Yurekten, Thomas Payne, Noemi Tejera, Felix Xavier Amaladoss, Callum Martin, Mark Williams, and Claire O’Donovan. Metabolights: open data repository for metabolomics. Nucleic acids research, 52(D1):D640–D646, 2024.