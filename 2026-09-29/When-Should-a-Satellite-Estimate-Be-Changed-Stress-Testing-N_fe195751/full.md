![](images/c679896d9e1919e1fc6a7d759580ef9f893222c08c0c1ecc342711dd506946aa.jpg)

# When Should a Satellite Estimate Be Changed? Stress-Testing Neural Corrections for Evapotranspiration

Marco Trotta

IRRIGANT

Irrigant, Idaho, USA m@irrigant.xyz

September 25, 2026

## Abstract

Neural residuals can improve satellite evapotranspiration (ET) estimates, but selectors must predict when a correction helps and reject unsupported inputs. We evaluate ten-member models on 16,366 flux-tower observations from 151 stations paired with OpenET, across nine rolling years and five spatial folds. At one held-out station, Gain accepted corrections on all 32 physically invalid records: it predicted a mean benefit of 0.83 mm day<sup>−1</sup>, but the corrections increased mean absolute error by $2 1 . 6 \ \mathrm { m m } \mathrm { d a y } ^ { - 1 }$ versus OpenET. On spatially held-out unit errors, SupportGain reduced station-macro MAE versus Gain by $0 . 1 4 8 \mathrm { m m } \mathrm { d a y } ^ { - 1 }$ under wind x3.6 (simultaneous 95% interval, 0.070 to 0.226), with 9.3% acceptance versus Gain’s 51.8%; on clean inputs, its $0 . 0 0 6 \mathrm { m m } \mathrm { d a y } ^ { - 1 }$ advantage had an interval that includes zero. These fault analyses are exploratory; none of 40 preplanned temporal comparisons passed Holm correction, while a separate predeclared cropland contrast found 0.041 mm day<sup>−1</sup> lower station-macro MAE with crop-only training (95% interval, 0.009 to 0.079).

Keywords: selective regression; satellite evapotranspiration; input faults; spatial validation; cropland training

## 1 Introduction

OpenET already estimates actual evapotranspiration (ET). A neural correction can move that estimate closer to a flux-tower measurement, or farther away. The selector must answer a sharper question than whether its neural model is uncertain: will this correction improve the estimate already available?

That decision becomes harder when weather inputs fall outside the conditions represented in training. A selector may predict benefit from familiar patterns while missing an unsupported input. This paper tests whether a simple support rule can catch such cases, and whether any apparent advantage holds on clean observations.

We evaluate a ten-network residual model on 16,366 screened records from 151 stations. Rolling tests cover 2012 through 2020, but reuse stations across years; they do not test unseen-site transfer. Every rejected correction returns the original OpenET estimate, and the score includes every test record. This prevents rejection from appearing accurate by hiding dificult cases.

The evidence gives a mixed answer. Clean data do not establish a reliable advantage for support screening over learned benefit prediction. An exploratory station-level failure shows how far a confident correction can miss, while controlled weather changes reveal when support screening can reduce error by rejecting many corrections. A predeclared cropland comparison also favors training on cropland records, subject to method-selection uncertainty.

The study makes three contributions. First, it evaluates a correction by how much error it removes from the existing satellite estimate, using spatially cross-fitted selector data. Second, it separates one observed input failure from controlled weather transformations and from the wider record of physical rule violations. Third, it compares all-station and crop-only training on the same held-out cropland records. These are retrospective prediction results; they do not identify irrigation status or measure irrigation response.

## 2 Related work

Selective regression asks when a model should defer to an available alternative. Prior work studies multi-expert deferral and conformalized selective prediction [6, 13]. Here, the alternative is the existing OpenET estimate. The selector predicts improvement over that estimate, and evaluation counts the fallback on rejected records. The decision rule is established; our contribution is its environmental stress test, not a new deferral theory.

Environmental prediction methods use input distance and cross-fitting to assess whether a record resembles the training data [7, 8]. Ensemble disagreement can also change outside the training distribution [1, 5], while neural networks may extrapolate along unsupported inputs [18]. We test a nearestneighbor distance screen beside these learned scores. It is a simple check, not a calibrated guarantee of reliable predictions.

Machine-learning methods already estimate and correct ET using water-balance supervision, physical covariates, and geographic transfer [2, 3, 11, 12, 17]. OpenET and the associated flux archive supply the satellite estimates and measurements used here [14–16]. Because the same archive supports this evaluation, our results do not independently validate OpenET development. Land-cover-specific OpenET biases also motivate testing a cropland training population [9].

Environmental records are spatially dependent, so random row splits can overstate transfer [10]. We group nearby stations for inner selection and report a separate spatial holdout. The rolling-year analysis tests later dates at observed stations, not new locations.

Reference ET describes a standardized surface; the target here is measured actual ET. We use reference ET only as an input and make no claim about crop water need, irrigation response, or reference-ET forecasting [4].

## 3 Data and evaluation

## 3.1 Observations and inputs

We use the OpenET model archive, Zenodo record 10119477, and the processed flux archive, record 7636781. The stationdate join contains 16,447 labeled observations from 152 stations. A physical screen retains 16,366 observations from 151 stations. It removes 24 temperature violations, eight gridMET vapor-pressure-deficit (VPD) violations, 54 invalid weather values from the station archive, and three missing ET labels. These counts overlap. A rule violation does not diagnose an instrument fault. The retained stations form 102 proximity groups at 10 km.

The target is energy-balance-corrected measured ET from the archive’s ET\_corr field. The observations are sparse satellite validation dates, not a daily time series. We retain negative target values and do not clip predictions. The Croplands subset contains 5,203 observations from 58 stations and 30 groups. Land-cover labels do not identify irrigation status.

Let $y _ { s t }$ denote measured ET and $O _ { s t }$ the OpenET estimate at station � and date �. The model uses that estimate, reference ET, seasonal position, temperature, VPD, and wind:

$$
\begin{array} { r } { x _ { s t } = [ o _ { s t } , E T o _ { s t } , h _ { t } , T _ { s t } , V P D _ { s t } , u _ { s t } ] , } \\ { h _ { t } = [ \sin ( 2 \pi d _ { t } / 3 6 5 ) , \cos ( 2 \pi d _ { t } / 3 6 5 ) ] . } \end{array}\tag{1}
$$

Here $d _ { t }$ is day of year, � is gridMET average temperature, � is gridMET wind speed, and ETo is reference ET. The seasonal encoding repeats every 365 days. The inputs and satellite estimate refer to the target date, but we do not reconstruct their release times. This is retrospective estimation, not forecasting.

## 3.2 Rolling temporal evaluation

For each test year from 2012 through 2020, train on rows from earlier years and test on that year. The pooled test set has 7,842 rows from 101 stations and 64 groups. Stations can occur in both training and test years. This design tests temporal transfer, not transfer to unseen stations. The cropland test subset has 3,234 rows from 49 stations and 24 groups.

Each training period uses three inner folds that withhold 10 km proximity groups. The same saved row assignments support the all-station and crop-only training arms. The crop-only arm filters each saved partition to Croplands rows. Outer test labels never determine model fits, selector scores, or thresholds. We save row identifiers for every fit, validation, and test partition.

## 3.3 Scoring and uncertainty

The primary score calculates mean absolute error (MAE) within each test station, then averages stations:

$$
\mathrm { M A E } _ { \mathrm { s t a t i o n } } ( f ) = \frac { 1 } { S } \sum _ { s = 1 } ^ { S } \frac { 1 } { n _ { s } } \sum _ { t = 1 } ^ { n _ { s } } | y _ { s t } - f ( x _ { s t } ) | .\tag{2}
$$

This gives each station equal weight, regardless of its number of observations. It does not estimate area-weighted agricultural error. Coverage is the station-weighted share of observations that receive a correction. The complete-system score uses OpenET wherever the selector rejects a correction.

We report paired diferences as baseline error minus candidate error. Positive diferences favor the candidate. The primary and extended selector families contain 40 and 52 preplanned comparisons, respectively. Both use Holm’s multipletest correction. The cropland analysis has one predeclared contrast and a paired group bootstrap with 2,000 draws and seed 20261005. Intervals resample proximity groups but keep fitted models and selectors fixed. They omit retraining uncertainty. Repeated split settings are stability checks, not independent trials.

## 4 Neural correction models

## 4.1 Ten-network residual model

The main gridMET and cropland experiments use an ensemble of ten residual networks. Each network has two 32-unit ReLU layers and one scalar output. Each model has 1,345 trainable parameters and seven input features. Training uses Adam with learning rate 0.001, batch size 256, L2 parameter 0.001, and 120 iterations. Each fit standardizes its inputs and training residuals. The ten fixed seeds run from 20260713 through 20260722. The model predicts the residual from OpenET to measured ET. The selector uses the ensemble mean correction and its betweennetwork spread. Full applies every correction; OpenET applies none. Training does not use test-based stopping or architecture search. We retain every fit warning in the run records.

## 5 When should the correction be applied?

## 5.1 Predicting benefit difers from predicting error

Let �(�) denote the available satellite estimate and $g ( x )$ a fitted neural residual. A selector $a ( x ) \in \{ 0 , 1 \}$ returns:

$$
f _ { a } ( x ) = o ( x ) + a ( x ) g ( x ) .\tag{3}
$$

Every observation receives a prediction: the corrected estimate if accepted, or OpenET if rejected. The correction’s realized benefit is

$$
D ( x , y ) = | y - o ( x ) | - | y - o ( x ) - g ( x ) | .\tag{4}
$$

Positive � means the correction lowers absolute error. For a fixed fitted model, the system’s mean absolute error satisfies

$$
R ( a ) = R ( o ) - \mathbb { E } [ a ( X ) D ( X , Y ) ] .\tag{5}
$$

The selector’s target is the expected benefit given its inputs, $\mu ( x ) = \mathbb { E } [ D ( X , Y ) \mid X = x ]$ . It should apply the correction when $\mu ( x ) > 0 .$ A selector that predicts low neural error need not predict positive improvement over �.

Consider two observations. For the first, OpenET and measured ET are both 0, while the correction is 0.1; the correction adds 0.1 error. For the second, measured ET is 10, OpenET is 0, and the correction is 9; it removes 9 error. The neural errors are 0.1 and 1, respectively. A rule that rejects the larger neural error would keep the harmful correction and discard the helpful one. Ensemble spread measures disagreement between fitted networks, not whether either correction improves on OpenET.

## 5.2 Learning and checking the selectors

The main experiment fixes the ten-network residual ensemble from Section 4. Each outer training set produces three sets of neural predictions on held-out proximity groups. These predictions train the selectors and set their thresholds. The neural ensemble is then refitted on the full outer training set. Outer test targets do not set model parameters, selector scores, thresholds, or input scales. Figure 1 shows the flow.

For residual predictions $g _ { 1 } , \ldots , g _ { 1 0 }$ , we use their mean $\bar { g }$ and sample spread:

$$
s _ { g } ( x ) = \left[ { \frac { 1 } { 9 } } \sum _ { j = 1 } ^ { 1 0 } ( g _ { j } ( x ) - \bar { g } ( x ) ) ^ { 2 } \right] ^ { 1 / 2 } .\tag{6}
$$

This is a mean-only ensemble inspired by Lakshminarayanan et al. [5]. The input-support score is the average distance to the five nearest training observations:

$$
d _ { 5 } ( x ) = \frac { 1 } { 5 } \sum _ { z \in N _ { 5 } ( x ) } \| S ( x ) - S ( z ) \| _ { 2 } ,\tag{7}
$$

where � standardizes inputs using the training set and $N _ { 5 }$ contains the five nearest training observations. We rebuild the neighbor index within each training partition and set thresholds from held-out inner distances. This simple screen follows prior work on training support [7], but it does not guarantee reliable predictions under shift.

Spread95 accepts observations below the station-weighted 95th percentile of inner ensemble spread. Support95 applies the same rule to input-support distance. Gain predicts the benefit in Equation 4 with a fixed boosted-tree regressor. Its six inputs are $( \bar { g } , | \bar { g } | , s _ { g } , \log ( 1 + d _ { 5 } ) , o , \mathrm { E T o } )$ . It uses 80 iterations, learning rate 0.05, seven leaves, minimum leaf size 50, L2 parameter 10, and no early stopping. Training weights give equal total weight to each station and are normalized to mean one. Gain accepts a correction only when predicted benefit is positive. SupportGain also requires Support95 to accept. The neural estimator is the same for every selector.

MonoGain requires predicted benefit to decrease as ensemble spread and support distance increase. AugmentedGain adds randomized weather faults to benefit-model training. The extended comparison adds TunedGain, LogisticSign, and ConformalCSR under a frozen protocol. Uniform, LocalShrinkage, and Clip appear in the earlier exploratory analysis in Appendix A. No selector hyperparameters change after the outer results are viewed.

## 5.3 Earlier measured-weather probes

An earlier analysis compares the seven archived inputs with a six-input version that omits VPD. It uses both proximity withholding and joint proximity-and-time withholding. Each natural comparison scores every test observation, including OpenET fallbacks, and reports cropland results separately. The 95% inner threshold does not guarantee 95% acceptance on test observations.

Additional probes multiply VPD or wind by 10, or add $2 0 ^ { \circ } \mathrm { C }$ to temperature. They change the test inputs while holding labels and fitted parameters fixed. Only rows with nonnegative exported actual vapor pressure are eligible. The VPD probe is a negative control when the model omits VPD. These transformations are not weather interventions and do not estimate fault prevalence. The protocol sets each magnitude before the new run.

## 6 Results

## 6.1 One station shows how a correction can fail

In one exploratory three-network run, Gain accepts all 32 records with physically invalid weather at the held-out manilacotton station. It predicts a mean benefit of 0.83 mm day<sup>−1</sup>. Instead, the corrections raise station-macro MAE from OpenET’s 1.095 to 22.681 mm day<sup>−1</sup>, a 21.586 mm day<sup>−1</sup> increase. SupportGain rejects all 32 corrections and returns OpenET. The mean correction magnitude is 21.98 mm day<sup>−1</sup>, far above the inner 95th percentile of 1.17 mm $\mathrm { d a y ^ { - 1 } }$ Ensemble spread is also above its inner 95th percentile: 39.42 versus 0.37 mm day<sup>−1</sup>. This is one station and one group, with no multi-group interval.

We then exclude that station and compare 133 other ruleviolating records from 21 stations in 17 groups. Support-Gain has 0.015 mm day<sup>−1</sup> lower MAE than Gain (95% groupbootstrap interval [0.000, 0.034]). Only two groups show a diference; 15 show none. OpenET has lower MAE than either selector: 0.755 mm day<sup>−1</sup>, compared with 0.791 for Gain and 0.776 for SupportGain. These records do not establish a device fault or a broad selector advantage.

![](images/3e45d9d80e947fe7f696310a8b3d84b79a0afdc62d3f5cedd2e76e53e76fd423.jpg)  
Figure 1. The selectors use predictions from inner groups that the neural networks did not train on. Outer test targets measure performance only. A rejected correction returns the original OpenET estimate.

## 6.2 Support screening helps under altered inputs, but accepts fewer records

The all-station protocol defines 40 comparisons before fitting. None has a Holm-adjusted one-sided �-value below 0.05; the smallest is 0.126. These tests do not directly compare SupportGain with Gain.

Across 30 shared split-seed and proximity settings, SupportGain lowers clean MAE over Gain by a mean of 0.00438 mm day<sup>−1</sup> (standard deviation 0.00289). Only one of 30 nominal group-bootstrap intervals excludes zero. The settings reuse observations, so they do not show a stable clean advantage.

SupportGain has lower point MAE than Gain in all 30 settings under each of twelve synthetic weather transforms. Across the 30 settings, temperature increased by 32 <sup>◦</sup>C yields a mean improvement of 0.55623 mm day<sup>−1</sup> (SD 0.04753; range 0.44878 to 0.62684). These comparisons reuse observations and keep labels and fitted models fixed. They describe sensitivity to the chosen transforms, not natural fault prevalence. Figure 4 shows the wind, temperature, and VPD probes. The VPD panel tests a kPa-to-hPa unit error at one magnitude.

A separate post hoc five-fold spatial holdout excludes each test station from its fold’s fit. Across 102 groups, the clean Gain-minus-SupportGain diference is 0.0059 mm day<sup>−1</sup> (simultaneous 95% interval [-0.0069, 0.0186]). Under wind multiplied by 3.6, SupportGain lowers MAE by 0.1478 mm day<sup>−1</sup> (simultaneous 95% interval [0.0697, 0.2260]). It accepts 9.3% of records, compared with 51.8% for Gain. Coverage falls below 0.3% under Fahrenheit-as-Celsius and VPD multiplied by 10. This exploratory analysis uses the same archive and conditions on fitted models; Appendix A.3 reports the full estimates.

## 6.3 Crop-only training lowers error on the tested cohort

SupportGain had the lowest clean all-station error among seven selectors in the extended comparison, at 0.801 mm day<sup>−1</sup>. That ranking used the same archive and included the cropland test records. The paired comparison therefore does not include uncertainty from selecting SupportGain.

On the same 3,234 cropland test records from 49 stations and 24 groups, all-station SupportGain scores 0.829 mm day<sup>−1</sup>

MAE. Cropland-only SupportGain scores 0.788 mm day<sup>−1</sup>. The predeclared diference is 0.0414 mm day<sup>−1</sup> (95% groupbootstrap interval [0.0085, 0.0788]). The interval uses 2,000 draws and conditions on the fitted models, selectors, and thresholds. Figure 6 shows the annual estimates.

OpenET scores 0.837 mm day<sup>−1</sup> on these records. The post hoc Full comparison also favors crop-only training: 0.777 versus 0.885 mm day<sup>−1</sup> MAE. Its diference is 0.1080 mm day<sup>−1</sup> (95% interval [0.0388, 0.2060]). The interval uses 2,000 groupbootstrap draws (seed 20261006) across the same 24 groups and conditions on the fitted models. We reviewed point estimates before this comparison, so it is not independent confirmation or part of the primary family. Crop-only SupportGain accepts 57.4% of records; all-station SupportGain accepts 64.1%. This result supports population-matched training within this archive, not a general crop or irrigation claim.

Additional risk-coverage curves and exploratory audits remain in the reproducibility package.

## 7 Discussion and limits

The study does not establish a reliable clean-input advantage for support screening. The 40-test family does not compare SupportGain directly with Gain, and the direct clean diferences are small and uncertain. The fixed weather transformations show how selectors respond to chosen input changes; they do not estimate natural fault rates.

One station shows a severe failure: Gain accepts large corrections on physically invalid weather, while SupportGain falls back to OpenET. The other 17 groups do not show a broad benefit, and OpenET has the lowest error on those records. An exploratory audit finds adjusted SupportGain advantages over Gain in 10 of 13 conditions and over MonoGain in nine, but none over AugmentedGain. The outcomes were visible before that audit, so independent sites with documented faults must test the pattern. Physical rule violations alone do not prove sensor faults.

Training only on cropland records lowers error on the tested cropland cohort. However, SupportGain was selected from outcomes on the same archive, including those test records. The reported interval omits that selection and model refitting.

![](images/60fbddb384791cc9fd37ad248ce3e4bf61dee0f9af9f1abbf2407b5ab44aed64.jpg)  
C Clean correction inputs

![](images/738374cd5565275b0ed327bb553c3f6068666898345158f0dfb7b06f58bb42a2.jpg)

![](images/2b274497c7c2b425bfaa52e122fe2a2dc02ff43bddb0af8506c531fe638e704f.jpg)

D Faulted correction inputs  
![](images/0148966f5c4487b6641efc092671bb6d05e2efdc348957d1ff7766a33e4bc15f.jpg)  
Figure 2. Each point is one of 32 records from the held-out station. Gain predicts benefit even though correction size and ensemble spread exceed their inner-training 95th percentiles.

The result supports a population-matching hypothesis within this archive, not general crop performance.

The rolling-year tests reuse stations and do not measure unseen-site transfer. Land-cover labels do not identify irrigated fields, and this study does not test irrigation interventions or forecast skill. All 360 neural fits in the cropland run reach the 120-iteration limit and report convergence warnings. This limit may afect the fitted predictor and selector. The measuredweather analysis is exploratory, and the available gridMET data do not provide a device-accuracy scale.

## 8 Conclusion

The study does not identify a generally reliable rule for when a neural correction should change a clean satellite estimate. It exposes one severe selector failure and finds lower error after crop-only training on the tested cropland cohort, with methodselection uncertainty. Independent sites and documented input faults must confirm these findings before they support a broader claim.

## Acknowledgements

I thank Meetpal S. Kukal for research mentorship and critical feedback on an earlier manuscript. I thank the OpenET and flux-data contributors for making the source datasets available.

Figure styling adapts Chen Liu’s figures4papers repository, with attribution and license information in the accompanying materials. The analysis and conclusions are the responsibility of the author.

## Data, code, and AI assistance

The source datasets are available through the cited Zenodo records. The accompanying reproducibility package contains checksums, cohort rows, split assignments, model predictions, protocols, tests, and figure-generation scripts. The arXiv source package includes the files required to rebuild this manuscript. The repository contains the model runs, selector protocols, and saved inner and outer predictions. The numerical environment uses Python 3.13.5, NumPy 2.4.3, pandas 2.3.3, and scikit-learn 1.8.0. Codex assists with code development, data auditing, analysis, figures, and manuscript drafting. All reported experimental values come from executed scripts and saved prediction records.

A Error on 133 rows  
![](images/92e29a092b424b1f4ba984577c695599124cd471c4ffb52037882fd9b5ab411e.jpg)

B Group differences  
![](images/b656bb73146b0dc0105c0d216bb860f93f03deef255f0948907a8ce231196d0a.jpg)  
17 groups · 21 stations · paired group interval [0.000, 0.034] mm/day · fitted models fixed  
Figure 3. Among 133 rule-violating records outside the known station, only two of 17 groups show a diference between Gain and SupportGain. OpenET has the lowest station-macro MAE. Rejected corrections return OpenET.

Clean condition across split and proximity settings  
![](images/975b8ca4e11d7ada8d419779299cd20e6b01999b8c341895231429450f500175.jpg)  
Figure 4. Ten-network results under fixed weather changes. The top row shows MAE reduction over OpenET; the bottom row shows station-weighted acceptance. Descriptive 95% intervals condition on fitted models. Wind labels give multipliers; temperature labels give Celsius ofsets. “F as C” reads Fahrenheit values as Celsius. The VPD panel shows one kPa-to-hPa error.

![](images/e28b30ab7ebc08400211273ebe29431a16eac2b487eddeb584c288f7f8f75345.jpg)  
Whiskers: per-run 95% group-bootstrap intervals. Shading: mean ± 1 SD across ten seeds.  
Figure 5. Clean SupportGain improvement over Gain for ten inner split seeds and three proximity thresholds. Whiskers show 95% group-bootstrap intervals; shading shows the mean plus or minus one standard deviation across split seeds.

![](images/295892fdaaa0285afe4d452d2926e93a798cc2d2ab31f06e827ba7703401a1f2.jpg)  
How to read the result. Eight of nine yearly estimates favor crop-only training. The years reuse stations and groups, so they show a pattern within this cohort, not independent replications.  
Figure 6. Yearly and pooled all-station minus crop-only SupportGain error. Positive values favor crop-only training. The pooled diference is 0.0414 mm day<sup>−1</sup> (95% group-bootstrap interval [0.0085, 0.0788]).

## A Exploratory measured-weather fault and repair analyses

This appendix reports a corrected three-member analysis on an earlier measured-weather cohort. It uses seeds 20260713 through 20260715 and recomputes the correction, spread, and support distance for transformed inputs. We had inspected the original results before correction, so these analyses remain exploratory. The cohort has 7,758 rows from 84 stations and 62 proximity groups.

## A.1 Weather fault types

The dose tests multiply wind by 0, 0.447, 1, 2.237, 3.6, 5, or 10. They add temperature ofsets of 0, 5, 10, 20, or 32 degrees Celsius. Other tests read Fahrenheit values as Celsius and multiply VPD by ten. Additional wind faults set values to zero, replace missing values with the training median, hold values for seven days, or use a training-derived day-of-year climatology. The tests keep labels and fitted model parameters fixed. The source workbook has no instrument accuracy specification, so these runs add no sensor-specification noise. The ten-member gridMET dose responses appear in Figure 4. Figure 7 shows the additiona measured-weather fault types.

![](images/3cab2f13f7348cd942aa5d8ede30e56197d3dbfe78f1a80fd7580ec1edb59813.jpg)  
Figure 7. Exploratory fault-type results from the corrected three-member run. Each transformation keeps fitted model parameters and labels fixed.

## A.2 Selector repairs

Three controls test alternative correction choices. Uniform selects a single multiplier from {0, 0.25, 0.5, 0.75, 1} by inner station MAE. LocalShrinkage estimates the conditional moments $\mathbb { E } [ ( y - o ) g \mid x ] { \mathrm { ~ a n d ~ } } \mathbb { E } [ g ^ { 2 } \mid x ]$ with two gradient-boosted regressors. It multiplies �(�) by $\lambda ( x ) = { \mathrm { c l i p } } ( \widehat { \mathbb { E } } [ ( y - o ) g \mid x ] / \widehat { \mathbb { E } } [ g ^ { 2 } \mid x ] , 0 , 1 )$ , or sets $\lambda ( x ) = 0$ when the denominator is zero. The ratio targets squared error, not the primary absolute-error score. The regressors use inner out-of-group predictions, equal station weights, and the Gain feature set. Clip limits weather inputs to the outer-training 1st and 99th percentiles at inference.

LocalShrinkage has clean station-macro MAE 0.806 mm/day; SupportGain has 0.825. The paired interval for its improvement is [0.001, 0.037] across 62 groups. The Holm-adjusted p-value is 0.154 across eleven post hoc comparisons. LocalShrinkage and Uniform difer by 0.003 mm/day, with an interval of [-0.009, 0.019].

Under tenfold wind, MonoGain has station-macro MAE 1.230 mm/day and Gain has 1.242. SupportGain has 0.853 on these same probe rows. Corrected AugmentedGain has MAE 0.860 on clean rows, against 0.825 for SupportGain. Under tenfold wind, AugmentedGain has 0.862 and SupportGain has 0.853. The 95% interval for the AugmentedGain improvement is [-0.0231, 0.0037] mm/day. None of its four comparisons with SupportGain passes Holm correction. These results do not identify a confirmed selector repair.

## A.3 Spatial holdout sensitivity

This post hoc analysis pools five spatial folds on the clean gridMET cohort. It uses 16,366 rows from 151 stations and 102 spatial groups. Each fold’s fit excludes every row from its test groups. Each row receives one prediction from a model that excludes its station. Training uses all dates from other groups, so this does not test forecasting. The selector and condition choices were visible before this analysis.

![](images/1bed161910f9ab85f046f504afc44f2c0f6dad8d13a3b2ceb2125811faf0ca15.jpg)  
Figure 8. Gain-minus-SupportGain MAE diference and acceptance on held-out spatial groups. Positive values favor SupportGain. The upper panel shows simultaneous 95% group-bootstrap intervals.

Table 1. Gain-minus-SupportGain station-macro MAE (mm/day) across five spatial folds. Positive values favor SupportGain. Intervals are simultaneous across the five test conditions.
<table><tr><td>Test input</td><td></td><td>Difference Simultaneous 95% interval</td></tr><tr><td>Clean</td><td>0.0059</td><td>[-0.0069, 0.0186]</td></tr><tr><td>Wind x2.237</td><td>0.0384</td><td>[0.0015,0.0752]</td></tr><tr><td>Wind x3.6</td><td>0.1478</td><td>[0.0697,0.2260]</td></tr><tr><td>Fahrenheit as Celsius</td><td>0.4899</td><td>[0.3454, 0.6344]</td></tr><tr><td>VPD x10</td><td>1.3698</td><td>[0.4886, 2.2511]</td></tr></table>

The group bootstrap uses 2,000 draws with seed 20261060. No p-values were calculated. These intervals condition on fitted models and omit retraining uncertainty. The four unit-error conditions preserve labels and model parameters. They do not estimate natural fault prevalence

The full prediction tables, risk-coverage curves, matched-budget analyses, original benchmark, and earlier covariate ablations remain in the reproducibility package.

## References

[1] Antoine de Mathelin, Francois Deheeger, Mathilde Mougeot, and Nicolas Vayatis. Deep anti-regularized ensembles provide reliable out-of-distribution uncertainty quantification. arXiv:2304.04042, 2023. URL https://arxiv.org/abs/ 2304.04042.

[2] Aleksei Dobrokhotov, Oliver Lopez, Ludmila Kozyreva, and Daria Mukhina. A machine learning residual correction framework for enhancing physical-based evapotranspiration estimates over croplands. European Journal of Remote Sensing, 59: 2695909, 2026. doi: 10.1080/22797254.2026.2695909.

[3] Tristan E. M. Hascoet, Victor Pellet, and Filipe Aires. Learning evapotranspiration dataset corrections from water cycle closure supervision. In NeurIPS Workshop on Tackling Climate Change with Machine Learning, 2022. URL https: //www.climatechange.ai/papers/neurips2022/23.

[4] Meetpal S. Kukal, Richard G. Allen, Ayse Kilic, et al. Artificial intelligence/machine learning estimation of FAO and ASCE standardized reference ET erodes the physical basis and intended purpose of ET standardization. Agricultural Water Management, 333:110657, 2026. doi: 10.1016/j.agwat.2026.110657.

[5] Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. In Advances in Neural Information Processing Systems, volume 30, pages 6402– 6413, 2017. URL https://papers.nips.cc/paper/2017/hash/ 9ef2ed4b7fd2c810847fa5fa85bce38-Abstract.html.

[6] Anqi Mao, Mehryar Mohri, and Yutao Zhong. Regression with multi-expert deferral. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pages 34738–34759, 2024. URL https://proceedings.mlr.press/v235/mao24d.html.

[7] Hanna Meyer and Edzer Pebesma. Predicting into unknown space? estimating the area of applicability of spatial prediction models. Methods in Ecology and Evolution, 12:1620–1633, 2021. doi: 10.1111/2041-210X.13650.

[8] Khoa Nguyen, Daniel Serino, Aviral Prakash, and Marc Klasky. GeoQ: Geometry-aware conditional quantile error estimation for scientific surrogate models. arXiv:2608.21652, 2026. URL https://arxiv.org/abs/2608.21652.

[9] OpenET. Accuracy and known issues, n.d. URL https: //etdata.org/accuracy-known-issues/. Accessed September 24, 2026.

[10] David R. Roberts, Volker Bahn, Simone Ciuti, et al. Crossvalidation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. Ecography, 40:913–929, 2017. doi: 10.1111/ecog.02881.

[11] Aleksei Rozanov, Samikshya Subedi, Vasudha Sharma, and Bryan C. Runck. Knowledge-guided machine learning models to upscale evapotranspiration in the U.S. Midwest, 2025. URL https://arxiv.org/abs/2510.11505.

[12] Haiyang Shi and Ximing Cai. Extrapolability improvement of machine learning-based evapotranspiration models via domain-adversarial neural networks. Environmental Modelling & Software, 187:106383, 2025. doi: 10.1016/ j.envsoft.2025.106383. URL https://www.sciencedirect.com/ science/article/abs/pii/S1364815225000672.

[13] Anna Sokol, Nuno Moniz, and Nitesh V. Chawla. Conformalized selective regression. Discover Data, 4:14, 2026. doi: 10.1007/ s44248-026-00113-2. URL https://link.springer.com/article/ 10.1007/s44248-026-00113-2.

[14] John M. Volk, Justin L. Huntington, Forrest S. Melton, et al. Assessing the accuracy of OpenET satellite-based evapotranspiration data to support water resource and land management applications. Nature Water, 2:193–205, 2024. doi: 10.1038/s44221-023-00181-7.

[15] John M. Volk et al. Post-processed data and graphical tools for a CONUS-wide eddy flux evapotranspiration dataset, 2023. URL https://zenodo.org/records/7636781.

[16] John M. Volk et al. OpenET model data for assessing the accuracy of OpenET satellite-based evapotranspiration data to support water resource and land management applications, 2023. URL https://zenodo.org/records/10119477.

[17] Jiaxing Wei, Shaomin Liu, Lisheng Song, et al. Development and validation of physically constrained machine learning for improving remote sensing-based evapotranspiration estimation. Remote Sensing ofEnvironment, 341:115460, 2026. doi: 10.1016/j.rse.2026.115460.

[18] Keyulu Xu, Mozhi Zhang, Jingling Li, Simon Shaolei Du, Kenichi Kawarabayashi, and Stefanie Jegelka. How neural networks extrapolate: From feedforward to graph neural networks. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=UH-cmocLJC.