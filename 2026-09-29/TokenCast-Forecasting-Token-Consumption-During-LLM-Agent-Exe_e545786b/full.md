# TokenCast: Forecasting Token Consumption During LLM Agent Execution

Chaoqian Ouyang<sup>1\*</sup> Ling Yue<sup>2\*</sup> Libin Zheng<sup>1†</sup> Huanghui Guo<sup>3</sup> Shengxiang Xu<sup>3</sup> YiShu Wang<sup>3</sup> Ran Li<sup>4</sup> Jian Yin<sup>1</sup> Shaowu Pan<sup>2</sup> Shimin Di<sup>3†</sup>

<sup>1</sup>Sun Yat-Sen University <sup>2</sup>Rensselaer Polytechnic Institute <sup>3</sup>Southeast University <sup>4</sup>Hong Kong University of Science and Technology zhenglb6@mail.sysu.edu.cn shimin.di@seu.edu.cn

## Abstract

When a large language model (LLM) agent executes the same task, token consumption can vary by over an order of magnitude across runs. The agent chooses its next steps based on tool feedback and intermediate results, while the growing context steadily inflates the input size of every subsequent call. The total consumption of a task is therefore hard to predict before execution and the prediction must be revised as the run unfolds. In this paper, we propose TokenCast, which learns a composable cost representation for each execution segment, recording its own consumption and the context growth it introduces. Composing adjacent segments yields a cumulative estimate that captures the extra input cost incurred when context from earlier segments is re-read by every later call. As execution unfolds, newly observed evidence refreshes the forecast, requiring no additional LLM calls and incurring a mean cumulative prediction time of 32.8 ms per run on SWE-bench Verified. Across 4 task suites and 6 agent models, TokenCast’s mean absolute error reduction against the strongest comparator averages 14.5% over 96 evaluated combinations. In offline budget-control replay, TokenCast uses 21.3% fewer tokens on average than a fixed-budget policy at matched trace completion. The code is available at https://github.com/DEFENSE-SEU/TokenCast.

## 1 Introduction

The applications of large language models (LLMs) are expanding from simple question answering to complex tasks such as software engineering and deep research. Completing these tasks typically relies on LLM-based agents that repeatedly plan, modify code, invoke tools, and verify results [Yao et al., 2023, Yang et al., 2024, Wang et al., 2025]. A given task may be resolved in a single edit or may require multiple rounds of retries before converging, and the dialogue and tool outputs generated in each round accumulate into the context of subsequent requests [Yang et al., 2024, Xiao et al., 2026, Zhu et al., 2026]. The final token consumption of a task is therefore difficult to predict before execution completes, making it hard for users to forecast costs and plan usage [Bai et al., 2026].

The challenge of predicting tokens lies in the fact that every action an agent takes can trigger new model calls, and consumption across steps is interdependent: the longer the context left by earlier steps, the larger the input to every subsequent call, causing consumption to accumulate and amplify [Salim et al., 2026, Zhu et al., 2026, Xiao et al., 2026]. The model’s generative behavior further compounds the difficulty. Given the same input, the model may choose different courses of action, generate outputs of different lengths, and consequently undergo different numbers of verification or retry cycles. In practice, token consumption across different executions of the same task can differ by up to 30× [Bai et al., 2026], making accurate prediction challenging.

Existing work can be organized by prediction scope and observation time. The scope ranges from a single response to an entire agent task; the forecast can be made before execution or updated during it. These dimensions give four settings (Figure 1): response length before generation, remaining response length during generation, total task consumption before execution, and remaining task consumption as an agent runs. Prior methods address these settings with different targets and access assumptions [Shahout et al., 2025, Xie et al., 2026, Bai et al., 2026].

At the request level, prior work studies how to predict the response length of a single model call. Because the input is known before generation begins, these methods typically estimate length from input features for resource allocation [Jin et al., 2023, Qiu et al., 2024, Fu et al., 2024, Zheng et al., 2026], while some also refine the estimate during generation using intermediate states or en-

![](images/6c45497cda0c0711cec6bb4999422addcf1f545ea995df839682cc2e4d9d9f1b.jpg)  
Figure 1: Four token-consumption prediction settings. At the call level, the input context supports a forecast before generation (1), and the output prefix updates it during generation (2). At the task level, total consumption is forecast before execution (3), and remaining consumption is updated as calls complete and context accumulates (4).

tropy statistics [Shahout et al., 2025, Xie et al., 2026, Merzouk et al., 2026].

At the agent-task level, prior work estimates total token consumption before execution, either from the task description or through the model’s own cost assessment [Bai et al., 2026]. Multi-step LLM execution frameworks organize or schedule requests according to program structures, semantic dependencies, or workflow paths available before the corresponding requests are executed [Khattab et al., 2023, Lin et al., 2024, Zheng et al., 2024, Ni et al., 2026, Yu et al., 2026]. Open-ended agents present a different situation: their behavior depends on tool feedback, environment state, and intermediate results, and no complete dependency graph exists before execution begins [Yao et al., 2023, Yang et al., 2024, Wang et al., 2025].

Multi-step forecasting has also been studied in structured workflows [Ni et al., 2026]. We focus on remaining provider-accounted input and output consumption in sequential agent execution. Each future request can bill retained context again, coupling the remaining cost to both future call count and input lengths. TokenCast updates this forecast from observed execution information without querying an LLM for the cost estimate.

Figure 2 summarizes TokenCast. It represents each execution segment by call count, net input-length change, and a cost residual. An exact composition identity exposes how context growth in one segment affects the cost baseline of later calls. The task predictor combines a direct forecast with a prefix–suffix forecast conditioned on a predicted boundary state, and refreshes its features as execution proceeds. Calibrated quantile models provide prediction intervals. The predictor makes no additional LLM calls.

![](images/c0e94f57793af0db94b5cdee2f628cd24299d4ded46a442cc9388e72bef339fb.jpg)  
Figure 2: Overview of TokenCast. Observed execution evidence supports call- and task-level forecasts, while the compositional path propagates predicted context growth into later-call costs.

Our contributions are as follows:

• We introduce a segment-cost factorization and an exact composition identity that accounts for repeated input consumption across calls.

• We use this identity in a staged prefix–suffix predictor that combines direct and compositional forecasts and updates from execution evidence without additional LLM calls.

• Across four benchmarks and six agent models, TokenCast’s MAE reduction averages 14.5% across 96 comparisons with the strongest comparator in each. Budget-control replay saves 21.3% of tokens at matched trace completion.

## 2 Related Work

Request-level output length prediction. Early methods extract features from the prompt to estimate response length for memory reservation, batching, or shortest-job-first scheduling [Jin et al., 2023, Qiu et al., 2024]. Prompt-only estimates remain highly uncertain, and prompt-conditioned response lengths can exhibit broad or heavy-tailed distributions [Perez-Ramirez et al., 2025, Wang et al., 2026]. Subsequent work therefore models more than a single point estimate: one line of work learns pairwise length orderings among requests to set scheduling priorities [Fu et al., 2024], and another fits a heavy-tailed log-t distribution that lets the scheduler trade off average and tail latency [Zheng et al., 2026]. Once generation begins, the decoder’s own intermediate states become available. Several recent studies show that mid-layer representations or entropy statistics can continuously refine the remaining-length estimate during decoding, narrowing the gap between the initial guess and the actual length [Shahout et al., 2025, Merzouk et al., 2026, Xie et al., 2026]. These methods share a common set of assumptions: the prediction target is a single response, the prompt is known, and the system has access to the prompt or to model internals. In agent tasks, later requests do not exist until earlier actions and tool calls complete, so none of these assumptions holds.

Task-level and multi-step prediction. Prior work has extended prediction to entire tasks. Self-Prediction has a coding agent inspect its environment before execution and estimate input, output, and total token consumption by stage [Bai et al., 2026]. DSPy [Khattab et al., 2023], Parrot [Lin et al., 2024], and SGLang [Zheng et al., 2024] represent multi-step LLM applications through program structures or semantic dependencies available to serving runtimes. Open-ended agents expose no such structure before execution.

Chimera predicts remaining workflow output with a CPU-based quantile random forest [Ni et al., 2026]. Pythia profiles historical traces to infer likely workflow paths and role-level output lengths [Yu et al., 2026]. TokenCast’s segment identity addresses the repeated input cost induced by carried context, and its forecasts use no additional LLM calls.

Trace studies of deployed agents report extensive iterative review loops in multi-agent software pipelines, and long contexts with short outputs and heavy prefix reuse in real coding-agent sessions [Salim et al., 2026, Zhu et al., 2026]. Broader analyses organize token use and efficiency across single-agent, multi-agent, and agent-ecosystem settings [Chen et al., 2026]. These studies characterize consumption after the fact and do not forecast it for a running task.

## 3 Method

## 3.1 Problem Formulation

An agent executes a task through a sequence of LLM calls interleaved with tool interactions. Let $C _ { k }$ denote the provider-accounted input and output token consumption of call k. A task terminates upon completion, failure, or when an execution limit is reached. For a task with K calls, its total consumption is $\begin{array} { r } { T = \sum _ { j = 1 } ^ { K } C _ { j } } \end{array}$ . After call k completes, the confirmed consumption is $\begin{array} { r } { S _ { k } = \sum _ { j = 1 } ^ { k } C _ { j } } \end{array}$ and the remaining consumption is $R _ { k } = T - S _ { k }$ . Each forecast uses only the task and execution information available at its prediction point.

We consider four prediction settings along the execution trajectory.

• Task Start predicts the total consumption $T$ before execution begins.

• Call Start predicts the consumption $C _ { k }$ of the current call after its request has been assembled and before any output token is generated.

• In-call Update updates the prediction of $C _ { k }$ as the generated prefix becomes available.

• Task Update predicts the remaining consumption $R _ { k }$ after call k completes, yielding an updated forecast of the task total, $\widehat { T } _ { k } = S _ { k } + \widehat { R } _ { k }$

## 3.2 Call–Task Forecasting

TokenCast represents segment cost relative to its starting input length. The next segment inherits the context produced by the preceding one, yielding an exact composition rule for adjacent segments.

Segment representation. A contiguous block of one or more calls forms a segment. Let segment A start with input length $L _ { A }$ , span $n _ { A }$ calls, and consume $C _ { A }$ tokens in total. Its representation is $\phi _ { A } = \left( n _ { A } , g _ { A } , b _ { A } \right)$ , where $g _ { A }$ is the net change in input length across the segment and $b _ { A } =$ $C _ { A } - n _ { A } L _ { A }$ is the residual after subtracting the starting-input baseline $n _ { A } L _ { A }$ , covering generation and within-segment context growth. The next segment starts with input length $L _ { A } + g _ { A }$ . Reasoning tokens billed by the provider contribute to $b _ { A } ;$ when they do not persist in the conversation context, they do not contribute to the context change $g _ { A }$

When segment B immediately follows A, the combined representation is

$$
\phi _ { A \circ B } = \left( n _ { A } + n _ { B } , g _ { A } + g _ { B } , b _ { A } + b _ { B } + n _ { B } g _ { A } \right) .\tag{1}
$$

This identity follows from the definitions and the boundary condition $L _ { B } = L _ { A } + g _ { A }$ . The third term $n _ { B } \ g _ { A }$ arises from realigning the baseline: $b _ { B }$ is defined relative to $B ^ { * } { \mathrm { s } }$ actual starting point $L _ { A } + g _ { A }$ whereas the combined residual is relative to $L _ { A }$ . The difference on each call is exactly $g _ { A }$ , and $n _ { B }$ calls accumulate to $n _ { B } \ g _ { A }$

Forecasting with composition. The decomposition converts aggregate remaining-cost prediction into three sub-problems: the prefix segment’s representation, the boundary state, and the suffix segment’s representation. TokenCast fits LightGBM models for these predictions and updates after completed calls. Appendix D.6 compares alternative base predictors. Input features are drawn from the task and execution information available at the prediction point. Appendix B.1 details these features and when each becomes available.

Call-level forecasts predict the current call’s cost from the features visible at Call Start or In-call Update. Task-level forecasts maintain two paths. The direct path predicts remaining total cost as a single target. The compositional path separates the current or next segment from the subsequent suffix. Their cost representations are predicted separately and combined using Eq. (1).

The compositional path predicts the prefix representation $( \widehat { n } _ { A } , \widehat { g } _ { A } , \widehat { b } _ { A } )$ and its ending state. The predicted context change sets the suffix input baseline, while the predicted ending state and features visible at the original prediction point condition the suffix model. The suffix representation is converted to a remaining-cost forecast through Eq. (1).

Updating forecasts. Each completed step during execution produces new observations, and Token-Cast refreshes its forecasts accordingly. At Call Start, the current request has been assembled and the actual input length is known, so the call-level forecast can be based on it directly. At In-call Update, the committed generated prefix and streaming timing supply additional features for the current-call forecast. At Task Update, the completed call supplies confirmed cumulative cost $S _ { k }$ and any completed tool outcomes. The next request’s input length remains predicted until that request i assembled. TokenCast uses the available evidence to re-predict $\widehat { R } _ { k }$ and reports $\widehat { T } _ { k } = S _ { k } + \widehat { R } _ { k }$

## 3.3 Compositional Learning

Staged fitting. TokenCast fits the direct, prefix, and suffix predictors as separate LightGBM models using labels extracted from completed traces. The direct path predicts remaining total cost. The compositional path predicts the prefix and suffix representations together with the prefix-ending boundary variables. Prefix context-change and suffix call-count errors can affect downstream cost through the composition identity, motivating the following weights on their local absolute losses:

$$
\ell _ { g } = n _ { B } \left| \widehat { g } _ { A } - g _ { A } \right| , \qquad \ell _ { n } = L _ { B } \left| \widehat { n } _ { B } - n _ { B } \right| .\tag{2}
$$

Cross-fitting. The suffix model is trained on predicted prefix boundaries. Tasks are partitioned into F folds, and each fold’s prefix predictions are generated by models trained on the remaining folds. The predicted context change sets the suffix input baseline, so its residual label is recomputed as $\begin{array} { r } { b _ { B } ^ { \mathrm { t r a i } \overline { { \bf n } } } = C _ { B } - n _ { B } \widehat { L } _ { B } } \end{array}$ , where $\widehat { L } _ { B } = L _ { A } + \widehat { g } _ { A } ^ { \mathrm { o o f } }$ . The loss weights in Eq. (2) are motivated by the composition identity; Appendix B.3 distinguishes this motivation from the error decomposition under the rebased suffix training label.

Correction model. After the direct, prefix, and suffix models are fixed, a correction model is trained on out-of-fold outputs of the complete forecasting pipeline:

$$
\psi ^ { * } = \underset { \psi } { \arg \operatorname* { m i n } } \sum _ { i } \left| y _ { i } - \widehat { C } _ { \mathrm { c o m p } , i } - h _ { \psi } ( \mathbf { q } _ { i } ) \right| .\tag{3}
$$

The input $\mathbf { q } _ { i }$ contains the visible features, the direct and compositional forecasts, their difference, and the predicted boundary variables. The final task-level forecast is $\widehat { C } _ { \mathrm { c o m p } } + h _ { \psi } ( \mathbf { q } )$

Prediction intervals. Separate LightGBM quantile models produce the 0.05 and 0.95 endpoints at each prediction point. The endpoints are widened symmetrically by a quantile of interval residuals

Table 1: MAE ↓ in tokens at four prediction points. Norm. Avg. is the macro-average of MAE normalized at each prediction point by the MAE of the corresponding history-median predictor. k denotes thousands of tokens. Best and second-best results are in bold and underlined, respectively. Marks are assigned using unrounded values.
<table><tr><td>Agent LLM</td><td>Prediction point</td><td>TokenCast (Ours)</td><td>TRAIL</td><td>EGTP</td><td>TIE</td><td>Self-Pred.</td></tr><tr><td></td><td colspan="6">SWE-bench Verified</td></tr><tr><td rowspan="4">GPT-5.4</td><td>Task Start</td><td>144.0k</td><td>165.3k</td><td>160.9k</td><td>157.0k</td><td>152.0k</td></tr><tr><td>Call Start</td><td>64.5</td><td>71.1</td><td>70.3</td><td>69.5</td><td>70.9</td></tr><tr><td>In-call Update</td><td>38.9</td><td>78.9</td><td>74.6</td><td>77.4</td><td>80.2</td></tr><tr><td>Task Update</td><td>80.0k</td><td>115.0k</td><td>117.0k</td><td>119.0k</td><td>126.0k</td></tr><tr><td></td><td>Norm. Avg.</td><td>0.69</td><td>0.94</td><td>0.92</td><td>0.92</td><td>0.94</td></tr><tr><td rowspan="4">Qwen3.8-27B</td><td>Task Start</td><td>192.0k</td><td>232.0k</td><td>209.0k</td><td>215.0k</td><td>221.0k</td></tr><tr><td>Call Start</td><td>87.0</td><td>109.0</td><td>102.9</td><td>99.0</td><td>105.5</td></tr><tr><td>In-call Update</td><td>81.9</td><td>120.0</td><td>116.0</td><td>111.4</td><td>108.9</td></tr><tr><td>Task Update</td><td>124.0k</td><td>176.0k</td><td>171.0k</td><td>164.0k</td><td>173.0k</td></tr><tr><td></td><td>Norm. Avg.</td><td>0.65</td><td>0.86</td><td>0.81</td><td>0.79</td><td>0.82</td></tr><tr><td></td><td></td><td>Search-R1</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="4">GPT-5.4</td><td>Task Start</td><td>34.2k</td><td>37.8k</td><td>37.6k</td><td>36.4k</td><td>36.9k</td></tr><tr><td>Call Start</td><td>34.5</td><td>37.3</td><td>36.5</td><td>34.0</td><td>39.8</td></tr><tr><td>In-call Update</td><td>31.3</td><td>39.4</td><td>38.5</td><td>36.8</td><td>39.9</td></tr><tr><td>Task Update</td><td>17.6k</td><td>23.1k</td><td>23.7k</td><td>23.3k</td><td>24.2k</td></tr><tr><td></td><td>Norm. Avg.</td><td>0.75</td><td>0.89</td><td>0.88</td><td>0.85</td><td>0.91</td></tr><tr><td rowspan="4">Qwen3.8-27B</td><td>Task Start</td><td>39.0k</td><td>53.5k</td><td>49.8k</td><td>47.6k</td><td>51.0k</td></tr><tr><td>Call Start</td><td>56.7</td><td>65.9</td><td>71.4</td><td>67.7</td><td>76.1</td></tr><tr><td>In-call Update</td><td>57.0</td><td>84.6</td><td>80.7</td><td>75.5</td><td>83.2</td></tr><tr><td>Task Update</td><td>24.0k</td><td>39.5k</td><td>34.0k</td><td>35.4k</td><td>38.4k</td></tr><tr><td></td><td>Norm. Avg.</td><td>0.59</td><td>0.83</td><td>0.79</td><td>0.77</td><td>0.84</td></tr></table>

on held-out calibration tasks and are bounded below by consumption already confirmed within the prediction scope. Appendix B.4 gives the procedure.

## 4 Experiments

## 4.1 Experimental Setup

Tasks and execution traces. We evaluate TokenCast on SWE-bench Verified [Jimenez et al., 2024, Chowdhury et al., 2024], Search-R1 [Jin et al., 2025a], MMLU-Pro [Wang et al., 2024b], and LongBench-v2 [Bai et al., 2025], covering software engineering, retrieval-based question answering, knowledge-based reasoning, and long-context understanding. We collect 11,712 execution traces from 240 benchmark tasks with six agent LLMs: GPT-5.4 [OpenAI, 2026], Claude Opus 4.6 [Anthropic, 2026], Gemini 3.1 Pro [Google DeepMind, 2026], DeepSeek-V4-Pro [DeepSeek-AI, 2026], Qwen3.8- 27B [Qwen Team, 2026], and Llama-3.2-3B-Instruct [Meta AI, 2024]. For reproducibility, we specify gpt-5.4-2026-03-05 for the GPT-5.4 API and qwen3.8-27b-20260815 for the self-hosted Qwen checkpoint; Table 5 lists model access and reasoning configurations. Traces are collected using DeepSeek Harness [DeepSeek, 2026] and OpenHands [Wang et al., 2025]. Repeated executions support the analysis of run-to-run variation, with additional repeats for anchor tasks. Table 2 specifies the collection design for each benchmark and harness. All runs of the same task remain in one partition. Training, validation, calibration, and test tasks are separated as described in Appendix C. Generalization experiments also use independently released trajectories (Appendix C.1).

Baselines. We compare TokenCast with three output-length predictors, TRAIL [Shahout et al.,

2025], EGTP [Xie et al., 2026], and TIE [Zheng et al., 2026], and the agent-level consumption estimator Self-Prediction [Bai et al., 2026]. We adapt these methods to the prediction targets and observations available at the four prediction points. At Task Start, Self-Prediction inspects the task environment before estimating total consumption. At the other prediction points, it uses the observed execution prefix. Appendix C.3 details each adaptation. Appendix D.6 compares direct regression, compositional forecasting, their average, and the full correction pipeline; Appendix D.6 evaluates the segment representation and composition procedure.

Metrics and implementation. We report mean absolute error (MAE) and weighted absolute percentage error (WAPE) at the four prediction points. At Call Start, the target includes the assembled request’s known input tokens. For 90% prediction intervals, we report empirical coverage, mean width, and mean interval score (MIS) [Gneiting and Raftery, 2007]. Tasks receive equal weight, with that weight distributed across their runs and evaluated checkpoints. Cross-configuration summaries divide MAE by that of a history-median predictor, which outputs the median training target for the same benchmark, agent LLM, and prediction point. The local platform provides eight NVIDIA A100 GPUs. Internal-state baselines use the agent model when accessible and a proxy for API-based models. Appendix C gives metrics, data partitions, training settings, and timing procedures.

## 4.2 Experimental Results and Analysis

## 4.2.1 Main Results

Table 1 shows SWE-bench Verified and Search-R1 results for GPT-5.4 and Qwen3.8-27B; Figure 3 covers all four benchmarks and six agent LLMs. On SWE-bench Verified with GPT-5.4, TokenCast reduces MAE relative to the strongest comparator by 47.9% at In-call Update, from EGTP’s 74.6 to 38.9 tokens, and by 30.4% at Task Update, from TRAIL’s 115k to 80k tokens. At Task Start, the strongest comparator is Self-Prediction, and the reduction is 5.3%, from 152k to 144k tokens. Across the 96 benchmark–model–prediction-point combinations in Appendix D.2, its MAE reduction against the lowest comparator MAE averages 14.5%. The averages are −2.2% at Task Start, 1.9% at Call Start, 30.4% at In-call Update, and 27.8% at Task Update. TokenCast trails the strongest comparator in 24 combinations: 15 at Task Start and nine at Call Start. None of these losses occurs at In-call Update or Task Update, where execution evidence accumulates.

Error relative to consumption. WAPE complements MAE by expressing absolute error relative to mean target consumption. For GPT-5.4 at Task Start, TokenCast’s MAE of 32.2k tokens on LongBench-v2 corresponds to a WAPE of 4.2%, whereas its MAE of 8.0k on MMLU-Pro corresponds to 15.8%. Their mean target consumptions are 771.2k and 50.6k tokens, respectively. Table 11 in Appendix D.3 reports the full WAPE results.

## 4.2.2 Prediction Reliability

Run-to-run variation. Repeated GPT-5.4 executions of the same task show substantial consumption spread across all four benchmarks. Figure 6 in Appendix D.1 shows the distributions for 48 anchor tasks and gives the repetition counts per benchmark.

Interval reliability. On the anchor tasks, TokenCast’s calibrated intervals reduce MIS relative to Self-Prediction’s native intervals at all four prediction points, by 32.0% on average. At Task Start, TokenCast covers 82.0% of outcomes against a nominal 90% level, while Self-Prediction covers 52.7%. Table 3 reports interval width, coverage, and MIS at every prediction point.

## 4.2.3 Generalization

Unseen task types. On independently released LiveClawBench trajectories [Long et al., 2026], zero-shot leave-one-domain-out transfer yields MAE ratios of 1.31 at Call Start and 1.47 at Task Update relative to Self-Prediction. With 20 target-domain tasks for adaptation, the ratios fall to 0.82 and 0.85. Appendix D.4.1 gives the protocol and intermediate results.

![](images/a02e76b2775fd9df1e7310b4964d78656c6c1ffcdfd3cf459c4f194378ca650b.jpg)  
Figure 3: Normalized average MAE across four benchmarks and six agent LLMs. Each panel corresponds to one benchmark, and each bar group corresponds to one agent LLM. For each method, MAE is normalized by the corresponding history-median MAE at each prediction point and macroaveraged over the four prediction points. Lower is better.

Table 2: Execution traces per benchmark and harness, over six agent LLMs.
<table><tr><td>Benchmark</td><td>Harness</td><td>Tasks</td><td>Runs</td><td>Anchors</td><td>Anchor runs</td><td>Traces</td></tr><tr><td>SWE-bench Verified</td><td>DeepSeek Harness</td><td>144</td><td>2</td><td>24</td><td>8</td><td>2,592</td></tr><tr><td>SWE-bench Verified</td><td>OpenHands</td><td>144</td><td>2</td><td>24</td><td>8</td><td>2,592</td></tr><tr><td>Search-R1</td><td>DeepSeek Harness</td><td>32</td><td>4</td><td>8</td><td>16</td><td>1,344</td></tr><tr><td>Search-R1</td><td>OpenHands</td><td>32</td><td>4</td><td>8</td><td>16</td><td>1,344</td></tr><tr><td>MMLU-Pro</td><td>DeepSeek Harness</td><td>32</td><td>4</td><td>8</td><td>8</td><td>960</td></tr><tr><td>MMLU-Pro</td><td>OpenHands</td><td>32</td><td>4</td><td>8</td><td>8</td><td>960</td></tr><tr><td>LongBench-v2</td><td>DeepSeek Harness</td><td>32</td><td>4</td><td>8</td><td>8</td><td>960</td></tr><tr><td>LongBench-v2</td><td>OpenHands</td><td>32</td><td>4</td><td>8</td><td>8</td><td>960</td></tr><tr><td>Total</td><td></td><td></td><td></td><td></td><td></td><td>11,712</td></tr></table>

Unseen agent LLMs. At Call Start, transfer to a held-out agent LLM yields 1.02 times the target history-median MAE without target-model tasks, improving to 0.95 with 20 tasks. With 3–10 target tasks, transfer outperforms target-only training. Appendix D.4.2 gives the protocol and a separate remaining-output-token Task Update analysis; other generalization results appear in Appendix D.4.

## 4.3 Case Study

TokenCast’s forecasts are revised as execution unfolds. In a code-repair run, a verification failure raises the task-total forecast, which decreases after a compatibility workaround passes the reported assertions. In long-context QA, file operations after answer generation consume further tokens and raise the forecast. Both traces are in Appendix E. We next examine token-budget control driven by these online forecasts on SWE-bench Verified.

Budget control. On 288 GPT-5.4 runs from 144 SWE-bench Verified tasks, Figure 4 shows that TokenCast uses 21.3% fewer tokens on average across seven replay budgets while matching fixedbudget trace completion at every budget. A run is trace-complete when it reaches its recorded terminal state; the replay does not measure task resolution. Prediction costs are included. Appendix D.7 gives the stopping rules, per-budget results, and overhead (Table 21).

![](images/99aaa8d6995bdd5f58eb682ff4b796f92b911ac0abfeaba56947edf02b740754.jpg)  
(a) Tokens

![](images/53a359a7b9f78a64acc962d93c0b8db31f083fa60bf147821187a91317a066c4.jpg)  
(b) Wall time  
Figure 4: Trace completion under stopping limits on SWE-bench Verified. Panel (a) plots trace completion against mean tokens per run, and panel (b) plots it against replay-accounted wall time per run. Prediction overhead is included. The dashed curve is the fixed-budget baseline.

Table 3: 90% prediction intervals over repeated executions of anchor tasks. TokenCast uses held-out calibration; Self-Prediction uses its reported 5th and 95th percentiles.
<table><tr><td></td><td colspan="3">TokenCast</td><td colspan="3">Self-Prediction</td></tr><tr><td>Prediction point</td><td>Coverage (%)</td><td>Width</td><td>MIS↓</td><td>Coverage (%)</td><td>Width</td><td>MIS↓</td></tr><tr><td>Task Start</td><td>82.0</td><td>752.5k</td><td>1547.9k</td><td>52.7</td><td>354.3k</td><td>3334.6k</td></tr><tr><td>Call Start</td><td>89.8</td><td>315</td><td>549</td><td>73.0</td><td>223</td><td>638</td></tr><tr><td>In-call Update</td><td>92.8</td><td>319</td><td>576</td><td>75.2</td><td>237</td><td>732</td></tr><tr><td>Task Update</td><td>91.2</td><td>379.1k</td><td>574.3k</td><td>77.0</td><td>232.0k</td><td>941.6k</td></tr></table>

## 4.4 Ablation Study

Feature analysis. Figure 5 shows that the useful evidence changes over a run. Request length leads at Task Start and Call Start, with relative importance of 0.24 and 0.30. During generation, committed prefix length dominates at 0.58. After a call completes, last-call input tokens, remaining plan items, and calls without progress carry similar importance at 0.28, 0.25, and 0.23.

Segment representation. The segment triple improves both task-level and call-level forecasts. Replacing it with total token count, input length, and output length raises their normalized MAEs from 0.71 to 0.88 and from 0.68 to 0.82. The no-composition variant has a normalized average MAE of 0.78 and call-level MAE of 0.74, compared with 0.69 and 0.68 for the full method. Within the triple, omitting context growth yields 0.76, while omitting the residual yields 0.73.

Forecasting strategy. Before execution, direct regression has lower normalized MAE than composition, 0.82 versus 0.87. The ranking reverses at Task Update after calls have completed: composition reaches 0.66 and direct regression 0.72. Averaging the two forecasts reduces Task Update MAE to 0.64, and the full correction model reaches 0.62 while raising interval coverage from 88.1% under direct regression to 90.6%. Appendix D.6 gives the full comparison.

Pipeline components. The largest degradation comes from dropping both cross-fitting and cost weighting. Normalized average MAE rises from 0.69 to 0.78, while 90% interval coverage falls from 90.6% to 87.4%. Dropping cross-fitting alone gives 0.73 MAE, and dropping cost weighting alone gives 0.72. The correction model and predicted boundary variables also contribute: their separate ablations give 0.74 and 0.72 MAE. Among six base predictors, LightGBM gives the lowest error and the highest coverage with 0.8 ms model time. Appendix D.6 reports the complete component and base-predictor results in Tables 19 and 16.

![](images/028fb0462d665277bab1d9d41377a776cdcdd052a40a224a1784ba2cf5966d30.jpg)  
Figure 5: Feature analysis at the four prediction points. For each point, the panels show the relative importance of selected features and contribution patterns for two representative features. Colors indicate the feature groups defined in Appendix B.1.

Update frequency. Updating after every call takes 32.8 ms of mean cumulative prediction time and makes 19.7 forecasts per run. Updating every three calls cuts the time to 11.9 ms and the forecast count to 6.9, and a single forecast at Task Start takes 2.1 ms. These full-pipeline times include feature extraction and model inference, so every-call updating adds under 0.03% to the 129 s median wall time of a GPT-5.4 run. Appendix D.5 reports the full frequency sweep with per-run standard deviations for all four update intervals.

## 5 Conclusion

TokenCast addresses the problem of forecasting token consumption during LLM agent execution, where context accumulation causes each call’s cost to depend on the entire preceding trajectory. The method represents each execution segment with a composable cost triple that separates the startinginput baseline from incremental consumption, and composes adjacent segments to propagate context growth into downstream cost estimates. Across four benchmarks and six agent LLMs, TokenCast’s MAE reduction against the strongest comparator averages 14.5% over 96 evaluated combinations and, in offline budget-control replay, it matches a fixed budget’s trace-completion rate while consuming 21.3% fewer tokens. Its mean cumulative task-level prediction time is 32.8 ms per run on SWE-bench Verified. Because the predictors are learned from recorded traces, transfer to a new task domain or agent LLM improves with a small set of target tasks from the new setting. This work provides a lightweight prediction layer for agent token consumption, and we leave the integration of these forecasts into a runtime decision framework that actively manages execution under budget constraints as a direction for future work.

## Ethics Statement

Our experiments use public benchmarks and recorded agent trajectories to study token consumption in LLM agents. The study involves no human subjects or collection of private user data. Models and datasets are used under their respective licenses and terms of use.

## Reproducibility Statement

Code associated with this study is linked at https://github.com/DEFENSE-SEU/TokenCast. Section 3 presents the forecasting formulation and training objective, with implementation details in Appendix B. Appendix C documents benchmark sampling, trajectory collection, agent and harness configurations, data splits, baseline implementations, and evaluation metrics. The budget-control replay protocol and complete Self-Prediction prompt are provided in Appendices D.7 and F, respectively.

## AI Use Statement

During manuscript preparation, large language models were used solely as general-purpose writing assistants for grammar checking, word refinement, and improving clarity. LLMs did not contribute to the research ideation, methodological design, or experimental execution. All suggestions produced by the LLMs were reviewed, edited, and vetted by the authors, who take full responsibility for the entire content of the paper.

## References

Anthropic. Introducing Claude Opus 4.6, 2026. URL https://www.anthropic.com/news/ claude-opus-4-6.

Longju Bai, Zhemin Huang, Xingyao Wang, Jiao Sun, Rada Mihalcea, Erik Brynjolfsson, Alex Pentland, and Jiaxin Pei. How do AI agents spend your money? analyzing and predicting token consumption in agentic coding tasks. CoRR, abs/2604.22750, 2026. doi: 10.48550/ARXIV.2604. 22750. URL https://doi.org/10.48550/arXiv.2604.22750.

Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng, Xiaozhi Wang, Xin Lv, Shulin Cao, Jiazheng Xu, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. Longbench v2: Towards deeper understanding and reasoning on realistic long-context multitasks. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), ACL 2025, Vienna, Austria, July 27 - August 1, 2025, pages 3639–3664. Association for Computational Linguistics, 2025. doi: 10.18653/V1/2025.ACL-LONG.183. URL https://doi.org/10.18653/v1/2025. acl-long.183.

Yuxi Chen, Junming Chen, Chenyu He, Yiwei Li, Yicheng Ji, Yifan Wu, Dingyu Yang, Lansong Diao, Lidan Shou, Hongliang Zhang, et al. Token economics for llm agents: A dual-view study from computing and economics. arXiv preprint arXiv:2605.09104, 2026.

Neil Chowdhury, James Aung, Chan Jun Shern, Oliver Jaffe, Dane Sherburn, Giulio Starace, Evan Mays, Rachel Dias, Marwan Aljubeh, Mia Glaese, Carlos E. Jimenez, John Yang, Leyton Ho, Tejal Patwardhan, Kevin Liu, and Aleksander Madry. Introducing SWE-bench verified, 2024. URL https://openai.com/index/introducing-swe-bench-verified/.

DeepSeek. DeepSeek Harness, 2026. URL https://www.deepseek.com/harness/en/.

DeepSeek-AI. Deepseek-v4: Towards highly efficient million-token context intelligence, 2026.

Yichao Fu, Siqi Zhu, Runlong Su, Aurick Qiao, Ion Stoica, and Hao Zhang. Efficient llm scheduling by learning to rank. Advances in Neural Information Processing Systems, 37:59006–59029, 2024.

Tilmann Gneiting and Adrian E Raftery. Strictly proper scoring rules, prediction, and estimation. Journal ofthe American statistical Association, 102(477):359–378, 2007.

Google DeepMind. Gemini 3.1 Pro model card, February 2026. URL https://deepmind. google/models/model-cards/gemini-3-1-pro/.

Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning steps. In Proceedings ofthe 28th International Conference on Computational Linguistics, pages 6609–6625, 2020.

Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik R. Narasimhan. Swe-bench: Can language models resolve real-world github issues? In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id= VTF8yNQM66.

Bowen Jin, Hansi Zeng, Zhenrui Yue, Dong Wang, Hamed Zamani, and Jiawei Han. Searchr1: Training llms to reason and leverage search engines with reinforcement learning. CoRR, abs/2503.09516, 2025a. doi: 10.48550/ARXIV.2503.09516. URL https://doi.org/10. 48550/arXiv.2503.09516.

Jiajie Jin, Yutao Zhu, Zhicheng Dou, Guanting Dong, Xinyu Yang, Chenghao Zhang, Tong Zhao, Zhao Yang, and Ji-Rong Wen. Flashrag: A modular toolkit for efficient retrieval-augmented generation research. In Guodong Long, Michale Blumestein, Yi Chang, Liane Lewin-Eytan, Zi Helen Huang, and Elad Yom-Tov, editors, Companion Proceedings ofthe ACM on Web Conference 2025, WWW 2025, Sydney, NSW, Australia, 28 April 2025 - 2 May 2025, pages 737–740. ACM, 2025b. doi: 10.1145/3701716.3715313. URL https://doi.org/10.1145/3701716.3715313.

Yunho Jin, Chun-Feng Wu, David Brooks, and Gu-Yeon Wei. S<sup>3</sup>: Increasing GPU utilization during generative inference for higher throughput. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper\_files/paper/2023/hash/ 3a13be0c5dae69e0f08065f113fb10b8-Abstract-Conference.html.

Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1601– 1611, 2017.

Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri Vardhamanan, Saiful Haq, Ashutosh Sharma, Thomas T. Joshi, Hanna Moazam, Heather Miller, Matei Zaharia, and Christopher Potts. Dspy: Compiling declarative language model calls into selfimproving pipelines. CoRR, abs/2310.03714, 2023. doi: 10.48550/ARXIV.2310.03714. URL https://doi.org/10.48550/arXiv.2310.03714.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. Natural questions: a benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:453–466, 2019.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Jason Flinn, Margo I. Seltzer, Peter Druschel, Antoine Kaufmann,

and Jonathan Mace, editors, Proceedings ofthe 29th Symposium on Operating Systems Principles, SOSP 2023, Koblenz, Germany, October 23-26, 2023, pages 611–626. ACM, 2023. doi: 10.1145/ 3600006.3613165. URL https://doi.org/10.1145/3600006.3613165.

Chaofan Lin, Zhenhua Han, Chengruidong Zhang, Yuqing Yang, Fan Yang, Chen Chen, and Lili Qiu. Parrot: Efficient serving of llm-based applications with semantic variable. In Ada Gavrilovska and Douglas B. Terry, editors, 18th USENIX Symposium on Operating Systems Design and Implementation, OSDI 2024, Santa Clara, CA, USA, July 10-12, 2024, pages 929–945. USENIX Association, 2024. URL https://www.usenix.org/conference/osdi24/ presentation/lin-chaofan.

Xiang Long, Li Du, Yilong Xu, RongJian Xu, Qiyanhui Lu, Ying Gao, Qinhua Xie, Fangcheng Liu, Ning Ding, Haoqing Wang, et al. Liveclawbench: Benchmarking llm agents on complex, real-world assistant tasks. arXiv preprint arXiv:2604.13072, 2026.

Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. When not to trust language models: Investigating effectiveness of parametric and non-parametric memories. In Proceedings of the 61st annual meeting of the association for computational linguistics (volume 1: Long papers), pages 9802–9822, 2023.

Mohamed Amine Merzouk, Dmitri Carpov, Mirko Bronzi, Damiano Fornasiere, and Adam Oberman. How much is left? llms linearly encode their remaining output length. CoRR, abs/2607.05316, 2026. doi: 10.48550/ARXIV.2607.05316. URL https://doi.org/10.48550/arXiv. 2607.05316.

Meta AI. Llama-3.2-3B-Instruct. Hugging Face, 2024. URL https://huggingface.co/ meta-llama/Llama-3.2-3B-Instruct.

Kangqi Ni, Wenyue Hua, Xiaoxiang Shi, Jiang Guo, Shiyu Chang, and Tianlong Chen. Chimera: Latency- and performance-aware multi-agent serving for heterogeneous llms. CoRR, abs/2603.22206, 2026. doi: 10.48550/ARXIV.2603.22206. URL https://doi.org/10. 48550/arXiv.2603.22206.

OpenAI. Introducing GPT-5.4, March 2026. URL https://openai.com/index/ introducing-gpt-5-4/.

Daniel F. Perez-Ramirez, Dejan Kostic, and Magnus Boman. CASTILLO: characterizing response length distributions of large language models. CoRR, abs/2505.16881, 2025. doi: 10.48550/ ARXIV.2505.16881. URL https://doi.org/10.48550/arXiv.2505.16881.

Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A Smith, and Mike Lewis. Measuring and narrowing the compositionality gap in language models. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2023, pages 5687–5711, 2023.

Haoran Qiu, Weichao Mao, Archit Patke, Shengkun Cui, Saurabh Jha, Chen Wang, Hubertus Franke, Zbigniew T. Kalbarczyk, Tamer Basar, and Ravishankar K. Iyer. Efficient interactive LLM serving with proxy model-based sequence length prediction. CoRR, abs/2404.08509, 2024. doi: 10.48550/ ARXIV.2404.08509. URL https://doi.org/10.48550/arXiv.2404.08509.

Qwen Team. Qwen3.8-Max: A new bar for coding and cowork, August 2026. URL https: //qwen.ai/blog?id=qwen3.8.

Stephen Robertson and Hugo Zaragoza. The probabilistic relevance framework: Bm25 and beyond. Foundations and trends® in information retrieval, 4(1-2):1–174, 2009.

Mohamad Salim, Jasmine Latendresse, SayedHassan Khatoonabadi, and Emad Shihab. Tokenomics: Quantifying where tokens are used in agentic software engineering. arXiv preprint arXiv:2601.14470, 2026.

Rana Shahout, Eran Malach, Chunwei Liu, Weifan Jiang, Minlan Yu, and Michael Mitzenmacher. Don’t stop me now: Embedding based scheduling for LLMS. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025. URL https://openreview.net/forum?id=7JhGdZvW4T.

Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. musique: Multihop questions via single-hop question composition. Transactions of the Association for Computational Linguistics, 10:539–554, 2022.

Jing Wang, Yu-Yang Qian, Ke Xue, Chao Qian, Peng Zhao, and Zhi-Hua Zhou. Robust length prediction: A perspective from heavy-tailed prompt-conditioned distributions. CoRR, abs/2604.07931, 2026. doi: 10.48550/ARXIV.2604.07931. URL https://doi.org/10.48550/arXiv. 2604.07931.

Xingyao Wang, Yangyi Chen, Lifan Yuan, Yizhe Zhang, Yunzhu Li, Hao Peng, and Heng Ji. Executable code actions elicit better llm agents. arXiv preprint arXiv:2402.01030, 2024a.

Xingyao Wang, Boxuan Li, Yufan Song, Frank F. Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, Hoang H. Tran, Fuqiang Li, Ren Ma, Mingzhang Zheng, Bill Qian, Yanjun Shao, Niklas Muennighoff, Yizhe Zhang, Binyuan Hui, Junyang Lin, and et al. Openhands: An open platform for AI software developers as generalist agents. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025. URL https://openreview.net/forum?id=OJd3ayDDoF.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Zhuang, Rongqi Fan, Xiang Yue, and Wenhu Chen. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, editors, Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024b. URL http://papers.nips.cc/paper\_files/paper/ 2024/hash/ad236edc564f3e3156e1b2feafb99a24-Abstract-Datasets\_ and\_Benchmarks\_Track.html.

Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff, Defu Lian, and Jian-Yun Nie. C-pack: Packed resources for general chinese embeddings. In Grace Hui Yang, Hongning Wang, Sam Han, Claudia Hauff, Guido Zuccon, and Yi Zhang, editors, Proceedings ofthe 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR 2024, Washington DC, USA, July 14-18, 2024, pages 641–649. ACM, 2024. doi: 10.1145/3626772.3657878. URL https://doi.org/10.1145/3626772.3657878.

Yuan-An Xiao, Pengfei Gao, Chao Peng, and Yingfei Xiong. Reducing cost of LLM agents with trajectory reduction. Proc. ACM Softw. Eng., 3(FSE):1241–1263, 2026. doi: 10.1145/3797084. URL https://doi.org/10.1145/3797084.

Huanyi Xie, Yubin Chen, Liangyu Wang, Lijie Hu, and Di Wang. Predicting LLM output length via entropy-guided representations. CoRR, abs/2602.11812, 2026. doi: 10.48550/ARXIV.2602.11812. URL https://doi.org/10.48550/arXiv.2602.11812.

John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. Swe-agent: Agent-computer interfaces enable automated software engineering. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, editors, Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024. URL http://papers.nips.cc/paper\_files/paper/2024/ hash/5a7c947568c1b1328ccc5230172e1e7c-Abstract-Conference.html.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 conference on empirical methods in natural language processing, pages 2369–2380, 2018.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview.net/forum?id=WE\_vluYUL-X.

Shan Yu, Junyi Shu, Yuanjiang Ni, Kun Qian, Xue Li, Yang Wang, Jinyuan Zhang, Ziyi Xu, Shuo Yang, Lingjun Zhu, et al. Pythia: Exploiting workflow predictability for efficient agent-native llm serving. arXiv preprint arXiv:2604.25899, 2026.

Haoyu Zheng, Yongqiang Zhang, Fangcheng Fu, Xiaokai Zhou, Hao Luo, Hongchao Zhu, Yuanyuan Zhu, Hao Wang, Xiao Yan, and Jiawei Jiang. Scheduling LLM inference with uncertainty-aware output length predictions. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/forum?id=I5IMkvVKd7.

Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark W. Barrett, and Ying Sheng. Sglang: Efficient execution of structured language model programs. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, editors, Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024. URL http://papers.nips.cc/paper\_files/paper/2024/hash/ 724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html.

Kan Zhu, Mathew Jacob, Chenxi Ma, Yi Pan, Stephanie Wang, Arvind Krishnamurthy, and Baris Kasikci. Tracelab: Characterizing coding agent workloads for llm serving. arXiv preprint arXiv:2606.30560, 2026.

## Appendix

## A Comparison with Related Methods

Table 4 compares prediction targets, evidence, and update points. TokenCast forecasts call and remaining-task consumption from execution features and a segment-cost representation.

Table 4: Prediction targets and estimators in related work. Appendix C.3 describes our adaptations.
<table><tr><td>Work</td><td>Prediction target</td><td>Evidence and estimator</td><td>Update point</td></tr><tr><td>TRAIL</td><td>Response length</td><td>Last-token hidden state and length bins</td><td>Before generation</td></tr><tr><td>EGTP</td><td>Response and remaining- response length</td><td>Hidden states and entropy pooling</td><td>Pre/mid generation</td></tr><tr><td>TIE</td><td>Response length distribution</td><td>Text embedding and log-t model</td><td>Before generation</td></tr><tr><td>Chimera</td><td>Remaining workflow output</td><td>Prompt and workflow features; quantile forest</td><td>Workflow request</td></tr><tr><td>Pythia</td><td>Workflow path and role output length</td><td>Historical trace profiles</td><td>Incoming request</td></tr><tr><td>Self-Prediction</td><td>Total task tokens</td><td>Agent inspection of task environ- Before execution ment</td><td></td></tr><tr><td>TokenCast</td><td>Provider-accounted call and remaining-task tokens</td><td>Execution features; direct and com- Four positional predictors</td><td>prediction points</td></tr></table>

## B Method Details

## B.1 Execution Evidence

The observation record v contains only information available at the prediction point. Token usage comes from provider records, and a logical call aggregates its initial request and any retries. The input length of an assembled request is measured directly; a future request’s input length remains predicted until assembly. The visible prefix contains streamed text and tool-call arguments committed so far, with a checkpoint every 128 UTF-8 bytes. Unavailable fields are masked. Prediction-point observations remain separate from the prefix-ending predictions passed to the suffix model.

Task representation. The task representation d uses TF-IDF features over word unigrams and bigrams and character 3- to 5-grams, with 12,000 terms for each representation. When a benchmark provides a task-difficulty label, the label is prepended to the task statement before encoding.

Action types. Reads include file reads, glob matches, and searches. Edits write or replace file content. Tracking updates the to-do list. A shell command is labeled as a test when it invokes pytest, tox, nox, or unittest, or refers to a test directory, and other shell commands are labeled as runs. An action is marked as failed when a tool reports an error or a shell command exits with a nonzero status. The stage of a call is determined by its most recent action.

Execution history. The progress record contains the numbers of planned, completed, and remaining to-do items, together with the number of calls since the list was last updated. The consumption history contains the input, output, and reasoning usage of the latest completed call, their changes from the preceding call, their mean, median, and least-squares trends over the last five and ten calls, and running totals of calls, model requests, tool actions, and confirmed tokens. Task Update additionally records the numbers of failed requests and tool errors in the latest call, together with the numbers of consecutive calls that failed, required a retry, or made no progress. Progress is defined as an edit, test, tracking update, or completed plan item. A window statistic is masked when the required observations are unavailable.

Recent tool actions. The w most recent tool actions fill the slots in recency order. Each slot records the action type, status, result size, calls since execution, and the reported result: an exit code, the number of lines returned by a read, the number of matches returned by a search, or the passed and failed counts of a test run. We select w per fold from {3, 5, 8} using the validation tasks.

Generated prefix. The lowercased prefix is hashed by words and word pairs into 256 signed buckets with weight 1 + log n for n occurrences, and its last line is hashed into 64 buckets. A single scan records whether the prefix is a valid JSON prefix, its nesting depth, whether it ends inside a string and under which tool-argument key, and, within string contents, unmatched brackets, quote parity, open code fences, heredocs, and the termination pattern of the last line.

Prefix recall. Earlier completed model requests from the same run serve as references, each represented by its hashed prefix at every checkpoint and its final generated length. The current prefix is compared with these references using cosine similarity, and the record includes the final and remaining lengths associated with the most similar requests and checkpoints.

Stream timing. For the current request, let s denote its start time and let $c _ { 1 } < \cdots < c _ { m }$ denote the output-checkpoint times. The timing record contains the time to the first checkpoint $c _ { 1 } - s ,$ elapsed time $c _ { m } \mathrm { ~ - ~ } s _ { \mathrm { ~ \scriptsize ~ { ~ m ~ } ~ } }$ , streaming time $c _ { m } - c _ { 1 }$ , committed bytes divided by streaming time, time since the previous checkpoint, the longest gap between checkpoints, and the initial-delay fraction $( c _ { 1 } - s ) / ( c _ { m } - s )$ . Fields with undefined denominators or too few observations are masked.

## B.2 Predictors and Training

Predictors. TokenCast uses separate LightGBM models for the four prediction settings. Call Start and In-call Update predict the complete consumption of the current call. Task Start and Task Update use a direct path and a compositional path. The direct path predicts remaining task consumption as one target. The compositional path predicts the current or next segment, its ending boundary, and the subsequent suffix, then combines the two segment representations using Eq. (1).

Training instances. Labels are extracted from completed traces. Call-level instances use the realized complete-call consumption. Task-level instances contain the remaining-consumption target, the prefix representation and ending boundary, and the suffix representation. Model inputs contain only the evidence available at the corresponding prediction point.

Task-level fitting. The direct and prefix models are fitted first. Task-level cross-fitting then produces out-of-fold prefix boundaries for suffix training. The suffix residual label is recomputed against the predicted input baseline as described in Section 3.3. After these models are fixed, the complete pipeline produces out-of-fold forecasts used to fit the correction model in Eq. (3).

Inference. Call-level forecasts use the corresponding fitted model directly. Task-level forecasts compute the direct prediction, the prefix boundary, the suffix prediction, and the compositional forecast before applying the correction model. The point models remain fixed during execution. Raw interval endpoints are produced by the quantile models in Appendix B.4.

## B.3 Theoretical Analysis

Composition properties. Let $L _ { A }$ and $L _ { B }$ be the initial input lengths of adjacent segments A and B. Since $L _ { B } = L _ { A } + g _ { A }$ and $C _ { X } = n _ { X } L _ { X } + b _ { X }$ for $X \in \{ A , B \}$ ,

$$
C _ { A \circ B } = ( n _ { A } + n _ { B } ) L _ { A } + b _ { A } + b _ { B } + n _ { B } g _ { A } = C _ { A } + C _ { B } .
$$

The cross-term $n _ { B } g _ { A }$ arises when the cost of B is expressed relative to the initial input length of A. For three consecutive segments $A , B ,$ , and D, either grouping yields the residual $b _ { A } + b _ { B } + b _ { D } +$ $n _ { B } g _ { A } + n _ { D } \big ( g _ { A } + g _ { B } \big )$ . The composition is therefore associative.

Single-call representation. A single call has representation $\phi _ { j } = ( 1 , L _ { j + 1 } - L _ { j } , C _ { j } - L _ { j } )$ . For the terminal call, set $L _ { K + 1 } = L _ { K }$ to close the notation; no suffix follows it. Composing these representations in execution order recovers the representation of any contiguous segment. For a

complete trace of K calls, the resulting residual is $\begin{array} { r } { \sum _ { j = 1 } ^ { K } C _ { j } - K L _ { 1 } } \end{array}$ . Hence

$$
T = K L _ { 1 } + b _ { \phi _ { 1 } \circ \cdots \circ \phi _ { K } } ,
$$

regardless of how the trace is partitioned.

Error propagation and loss weights. In a compositional forecast, the suffix is predicted from the estimated boundary of the prefix. Let $L _ { B } = L _ { A } + g _ { A }$ be the true input length at the start of the suffix, and let $\delta g _ { A }$ and $\delta n _ { B }$ denote prediction errors in the prefix’s context change and the suffix’s call count. With $L _ { A }$ fixed, and writing $\delta { b } _ { B }$ for the error in the suffix residual, the suffix cost error is

$$
\Delta C _ { B } = n _ { B } \delta g _ { A } + L _ { B } \delta n _ { B } + \delta n _ { B } \delta g _ { A } + \delta b _ { B } .
$$

The coefficients $n _ { B }$ and $L _ { B }$ motivate the local weights in Eq. (2) under a true-boundary residual. Suffix training instead uses $b _ { B } ^ { \mathrm { t r a i n } } = C _ { B } - n _ { B } \widehat { L } _ { B }$ . Writing $e _ { b } = \widehat { b } _ { B } - b _ { B } ^ { \mathrm { t r a i n } }$ and $e _ { n } = { \widehat { n } } _ { B } - n _ { B }$ gives $\widehat { C } _ { B } - C _ { B } = \widehat { L } _ { B } e _ { n } + e _ { b }$ . Boundary prediction can affect the learned residual and the suffix model’s inputs. Cross-fitting exposes suffix training to out-of-fold boundary errors.

## B.4 Interval Calibration

After the point models are fixed, separate LightGBM quantile models are trained for the 0.05 and 0.95 endpoints at each prediction setting. They use the evidence available at that prediction point and are fitted on the training tasks. Their outputs form the raw interval $[ \ell _ { i } , u _ { i } ]$

Intervals are calibrated separately for Task Start, Call Start, In-call Update, and Task Update using the held-out calibration tasks of each fold. For target $y _ { i } .$ , the absolute interval residual is

$$
s _ { i } = \operatorname* { m a x } \left\{ \ell _ { i } - y _ { i } , y _ { i } - u _ { i } , 0 \right\} .\tag{4}
$$

For each prediction setting, the corresponding calibration quantile of $\{ s _ { i } \}$ is added symmetrically to the raw endpoints. At runtime, the calibrated endpoints are bounded below by consumption already confirmed within the prediction scope. Calibration parameters remain fixed during inference.

## C Experimental Setup

## C.1 Trace Collection

Benchmarks. From the 500 instances of SWE-bench Verified [Jimenez et al., 2024, Chowdhury et al., 2024], we take 144. The four smallest repositories, flask, seaborn, requests, and pylint, enter in full. pytest, xarray, astropy, scikit-learn, and matplotlib contribute 15 each, and sphinx, sympy, and django 16 each. Search-R1 [Jin et al., 2025a] contributes 32 questions from seven FlashRAG evaluation sets [Jin et al., 2025b], half single-hop from NQ [Kwiatkowski et al., 2019], TriviaQA [Joshi et al., 2017], and PopQA [Mallen et al., 2023], and half multi-hop from HotpotQA [Yang et al., 2018], 2WikiMultihopQA [Ho et al., 2020], MuSiQue [Trivedi et al., 2022], and Bamboogle [Press et al., 2023]. MMLU-Pro [Wang et al., 2024b] contributes 32 test questions with two or three per category, and LongBench-v2 [Bai et al., 2025] has 32 questions spread over its six domains and three length labels. Within each stratum, we take instances in the stable-hash order of their id, and the anchors are the first two per SWE-bench Verified repository, one for flask and three for django, and the first eight in each other benchmark.

Task packs. A task is a statement and a seed workspace. For SWE-bench, the seed is the repository at the base commit. The agent may not edit tests and finishes by writing submission.patch, the diff of its working tree. For the other benchmarks, the seed holds README.md and an empty answer.md, the statement provides the question and its options, for LongBench-v2 together with the context document, and the agent writes its answer into answer.md. A Search-R1 statement also names a shell command that queries a local BM25 server [Robertson and Zaragoza, 2009, Jin et al., 2025a], the official Search-R1 retriever over the 21,015,324-passage wiki-18 corpus, and prints three passages. Correctness is scored after collection, with the SWE-bench Verified harness for patches, letter match for MMLU-Pro and LongBench-v2, and alias match for Search-R1.

Agent LLMs. Table 5 lists the models and their reasoning configurations. GPT-5.4 [OpenAI, 2026], Claude Opus 4.6 [Anthropic, 2026], and Gemini 3.1 Pro [Google DeepMind, 2026] use the lowest reasoning setting available through their respective APIs. DeepSeek-V4-Pro [DeepSeek-AI, 2026] and Qwen3.8-27B [Qwen Team, 2026] run with thinking enabled, while Llama-3.2-3B-Instruct [Meta AI, 2024] has no reasoning mode. Temperature is 0 where supported. The self-hosted models run on vLLM [Kwon et al., 2023] in bf16 on eight A100 GPUs with full context windows.

Table 5: Agent LLMs used for trace collection, with access modes and reasoning configurations.
<table><tr><td>Model</td><td>Vendor</td><td>Access</td><td>Reasoning configuration</td></tr><tr><td>GPT-5.4</td><td>OpenAI</td><td>API</td><td>Low reasoning effort</td></tr><tr><td>Claude Opus 4.6</td><td>Anthropic</td><td>API</td><td>Adaptive thinking; low effort</td></tr><tr><td>Gemini 3.1 Pro</td><td>Google</td><td>API</td><td>Low thinking level</td></tr><tr><td>DeepSeek-V4-Pro</td><td>DeepSeek</td><td>API</td><td>Thinking enabled</td></tr><tr><td>Qwen3.8-27B</td><td>Alibaba</td><td>Self-hosted</td><td>Thinking enabled</td></tr><tr><td>Llama-3.2-3B-Instruct</td><td>Meta</td><td>Self-hosted</td><td>No reasoning mode</td></tr></table>

Harnesses. DeepSeek Harness [DeepSeek, 2026] runs at version 0.1.1-rc.2, commit b150a551, in its headless profile with 12 tools: read, write, edit, str\_replace\_editor, glob, grep, pwsh, job\_list, job\_output, job\_kill, todo\_write, and skill. Commands are executed without a sandbox or approval prompts. Context compaction, result pruning, generated session titles, web access, and subagents are turned off, so every provider request is an agent call. Each run starts a fresh session in a fresh copy of the seed workspace on Windows.

OpenHands [Wang et al., 2025] runs at version 1.18.0 with the CodeAct [Wang et al., 2024a] agent and its execute\_bash, str\_replace\_editor, and task-tracking tools, with browsing and the condenser disabled. It runs SWE-bench Verified in the official image of each instance and the other benchmarks in a plain Linux image and calls the same retrieval script.

Both harnesses cap a run at 7,200 s, 500 calls, 4 retries per call, 65,536 output tokens per model request, and 30,000,000 tokens. A run ends when the agent submits, the model gives a final response without submitting, or a cap is reached. A logical call consists of its initial model request and any retry requests issued before tool feedback is received. The observer timestamps every model request and emits a checkpoint every 128 bytes of committed output. Token usage is taken from the provider response for each request and aggregated at the logical-call level.

Repeated executions. Table 2 lists the runs per task. SWE-bench Verified tasks run twice per model, and the other tasks run four times. Anchors run eight times, or 16 for Search-R1.

Trace statistics. Table 6 lists the median tokens and calls per run and the share of correct runs for each benchmark and model. For GPT-5.4 on SWE-bench Verified, the input makes up 99% of the tokens, the median run takes 129 s, and the median call emits two checkpoints.

The code-repair execution in Appendix E.1 illustrates the four prediction points and the usage records along a complete run (Figure 9).

Additional trajectories. The generalization experiments use independently released trajectories from LiveClawBench [Long et al., 2026], which records multiple agent LLMs across task domains under a shared agent framework. We retain runs with complete interaction records and per-call input and output token counts, and construct targets with the accounting convention of Section 3.1.

For the reasoning-configuration experiment in Appendix D.4.5, we rerun 69 SWE-bench Verified tasks with GPT-5.4 reasoning effort set to off and compare them with the same tasks under the low setting. The API route reports no separate reasoning-token count. Measured against visible output, the median request bills 0.46 output tokens per byte under the low setting and 0.34 under off.

Table 6: Median tokens in thousands and calls among runs with complete usage, and correct runs as a percentage of all attempted runs, under each harness.
<table><tr><td rowspan="2">Model</td><td colspan="3">DeepSeek Harness</td><td colspan="3">OpenHands</td></tr><tr><td>Tokens</td><td>Calls</td><td>Correct</td><td>Tokens</td><td>Calls</td><td>Correct</td></tr><tr><td>SWE-bench Verified</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-5.4</td><td>245.4</td><td>21</td><td>40.7</td><td>314.5</td><td>20</td><td>36.3</td></tr><tr><td>Claude Opus 4.6</td><td>258.1</td><td>21</td><td>41.7</td><td>470.7</td><td>22</td><td>54.2</td></tr><tr><td>Gemini 3.1 Pro</td><td>333.5</td><td>21</td><td>43.5</td><td>253.5</td><td>21</td><td>47.5</td></tr><tr><td>DeepSeek-V4-Pro</td><td>280.9</td><td>20</td><td>34.3</td><td>336.8</td><td>28</td><td>33.8</td></tr><tr><td>Qwen3.8-27B</td><td>526.0</td><td>22</td><td>29.6</td><td>211.5</td><td>15</td><td>31.2</td></tr><tr><td>Llama-3.2-3B-Instruct</td><td>508.8</td><td>31</td><td>9.5</td><td>619.7</td><td>33</td><td>11.8</td></tr><tr><td>Search-R1</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-5.4</td><td>38.1</td><td>8</td><td>76.3</td><td>37.5</td><td>8</td><td>75.0</td></tr><tr><td>Claude Opus 4.6</td><td>58.3</td><td>9</td><td>59.8</td><td>58.5</td><td>10</td><td>69.6</td></tr><tr><td>Gemini 3.1 Pro</td><td>55.8</td><td>10</td><td>68.8</td><td>48.3</td><td>8</td><td>69.6</td></tr><tr><td>DeepSeek-V4-Pro</td><td>94.8</td><td>12</td><td>65.2</td><td>41.2</td><td>9</td><td>70.5</td></tr><tr><td>Qwen3.8-27B</td><td>60.3</td><td>10</td><td>54.0</td><td>84.0</td><td>9</td><td>58.9</td></tr><tr><td>Llama-3.2-3B-Instruct</td><td>136.4</td><td>16</td><td>27.2</td><td>98.7</td><td>15</td><td>28.6</td></tr><tr><td>MMLU-Pro</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-5.4</td><td>21.6</td><td>4</td><td>81.9</td><td>46.1</td><td>6</td><td>81.9</td></tr><tr><td>Claude Opus 4.6</td><td>27.3</td><td>5</td><td>87.5</td><td>29.3</td><td>4</td><td>88.1</td></tr><tr><td>Gemini 3.1 Pro</td><td>54.6</td><td>9</td><td>75.6</td><td>59.7</td><td>9</td><td>77.5</td></tr><tr><td>DeepSeek-V4-Pro</td><td>35.2</td><td>6</td><td>70.6</td><td>38.6</td><td>5</td><td>76.9</td></tr><tr><td>Qwen3.8-27B</td><td>41.9</td><td>7</td><td>76.9</td><td>52.7</td><td>7</td><td>67.5</td></tr><tr><td>Llama-3.2-3B-Instruct</td><td>30.1</td><td>5</td><td>42.5</td><td>48.7</td><td>5</td><td>43.8</td></tr><tr><td>LongBench-v2</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-5.4</td><td>762.5</td><td>6</td><td>58.8</td><td>608.0</td><td>5</td><td>58.1</td></tr><tr><td>Claude Opus 4.6</td><td>361.7</td><td>4</td><td>66.9</td><td>716.1</td><td>6</td><td>75.0</td></tr><tr><td>Gemini 3.1 Pro</td><td>832.5</td><td>8</td><td>64.4</td><td>610.0</td><td>8</td><td>69.4</td></tr><tr><td>DeepSeek-V4-Pro</td><td>523.7</td><td>4</td><td>60.6</td><td>1,030.7</td><td>9</td><td>52.5</td></tr><tr><td>Qwen3.8-27B</td><td>884.4</td><td>8</td><td>45.0</td><td>581.0</td><td>6</td><td>45.6</td></tr><tr><td>Llama-3.2-3B-Instruct</td><td>645.9</td><td>7</td><td>25.6</td><td>592.1</td><td>4</td><td>13.1</td></tr></table>

## C.2 Data Splits

Tasks are assigned to five folds, balanced over benchmarks and agent LLMs, and all runs of a task remain in their fold. Each fold serves once as the test fold. The next fold calibrates intervals, the one after selects hyperparameters, and the remaining two train. We repeat the partitioning with three random seeds. For each seed, we pool out-of-fold test predictions and compute task-weighted metrics over all evaluated tasks. Tables 1 and 7–10 report means across the three seeds.

## C.3 Baselines

Output-length predictors. TRAIL, EGTP, and TIE are originally designed to predict the length of a single response. We adapt each method to the target associated with a prediction point while preserving its core representation and estimator. For Qwen3.8-27B and Llama-3.2-3B-Instruct, the internal-state methods read the last layer of the agent LLM itself. For the API models, they use Llama-3.2-3B-Instruct as a proxy encoder over the text visible at the prediction point: the task statement at Task Start, the statement and transcript of completed calls at Call Start and Task Update, and the statement and generated prefix at In-call Update. The window keeps the first 384 tokens of the statement and the last 1,536 of the remaining text. TRAIL classifies the last-token state into 8 log-spaced target-length bins and predicts the expected bin median. EGTP fits a ridge regression of log target length on the entropy-weighted mean state and entropy statistics of the window. TIE encodes the window with bge-small-en-v1.5 [Xiao et al., 2024] and fits the location and scale of a log-t distribution with 3.5 degrees of freedom on the concatenated CLS, mean, and max poolings. Its point prediction is the median, and its interval spans the 0.05 and 0.95 quantiles.

Agent usage estimators. Self-Prediction queries the agent LLM for the 5th, 50th, and 95th percentiles of token consumption for the same prediction target. The median serves as the point prediction, and the 5th and 95th percentiles define a nominal 90% prediction interval. We use these predictions directly, without fitting or calibrating them on held-out tasks. At Task Start, the agent first inspects the seed workspace with its full tool set and estimates the tokens of a complete run. At the other prediction points, we adapt the prompt to the observed execution record and corresponding prediction target. At Call Start, the agent estimates the complete consumption of the current call. At In-call Update, it estimates the same complete-call quantity using the observed generation prefix. At Task Update, it estimates the token consumption of the remaining execution. Appendix F provides the complete prompt template for all four points.

Fitting. Each learned baseline uses TokenCast’s partition: it is fitted on each fold’s training tasks, tuned on the validation tasks, and calibrated on the calibration tasks.

## C.4 Metrics

For each prediction setting, let i index the evaluated prediction points, with target $y _ { i } ,$ , point prediction $\hat { y } _ { i }$ , 90% prediction interval $[ \ell _ { i } , u _ { i } ] .$ , and normalized weight $w _ { i }$ . We assign equal weight to each evaluated task, divide that weight equally among its evaluated runs, and divide each run’s weight equally among its evaluated prediction points. Thus, $\textstyle \sum _ { i } w _ { i } = 1$ , and

$$
\mathrm { M A E } = \sum _ { i } w _ { i } | \hat { y } _ { i } - y _ { i } | , \qquad \bar { y } = \sum _ { i } w _ { i } y _ { i } , \qquad \mathrm { W A P E } = \frac { \mathrm { M A E } } { \bar { y } } \times 1 0 0 \% .\tag{5}
$$

MAE and mean target consumption $\bar { y }$ use the same evaluation samples and task–run–checkpoint weights. WAPE expresses the mean absolute error as a percentage of the mean actual target consump tion. For a nominal $1 - \alpha$ prediction interval, we use the interval score

$$
\mathrm { I S } _ { i } = ( u _ { i } - \ell _ { i } ) + \frac { 2 } { \alpha } ( \ell _ { i } - y _ { i } ) { \bf 1 } \{ y _ { i } < \ell _ { i } \} + \frac { 2 } { \alpha } ( y _ { i } - u _ { i } ) { \bf 1 } \{ y _ { i } > u _ { i } \} ,\tag{6}
$$

with α = 0.1. Mean interval score is

$$
{ \mathrm { M I S } } = \sum _ { i } w _ { i } \mathrm { I S } _ { i } .\tag{7}
$$

Empirical coverage and mean interval width are aggregated using the same weights. The Norm. Avg. column of Table 1 is the arithmetic mean over the four prediction points of a method’s MAE divided by the MAE of the corresponding history-median predictor. This predictor outputs the median target of the training tasks in the same fold for the same benchmark, agent LLM, and prediction point, and uses no task features or execution evidence.

## D Extended Evaluation

## D.1 Run-to-Run Variation and Interval Reliability

Figure 6 summarizes repeated GPT-5.4 executions from 48 anchor tasks across four benchmarks and two harnesses. Search-R1 has 16 repetitions per task–harness pair, while the others have eight. Repeated executions of a task show wide consumption spread across all four benchmarks.

Table 3 compares TokenCast’s calibrated intervals with the native intervals returned by Self-Prediction at the four prediction points. MIS combines interval width with a penalty for how far outcomes fall

![](images/b838d45520b4434ab7f0270b0dd270239ea41783567bd29d673b1fb3b68707ba.jpg)  
(a) Run-to-run variation

![](images/76d4914f2f54e1bcaa664ddc5344cfdaa65c8183f07f98b826f0478fda525011.jpg)  
(b) Consumption spread  
Figure 6: Run-to-run variation in token consumption across repeated executions of the same task. The panels show the distribution across repeated runs and the within-task consumption spread as a function of task consumption, over 48 GPT-5.4 anchor tasks.

outside the interval. Lower values indicate better interval forecasts. TokenCast’s wider intervals reduce the missed-outcome penalty enough to improve MIS. This comparison includes TokenCast’s held-out calibration and Self-Prediction’s native, uncalibrated intervals.

## D.2 Results Across Benchmarks and Agent LLMs

Tables 7 to 10 report the MAE for each benchmark and agent LLM on the same runs, with the history median as the normalization reference. MAE is reported in tokens, with k denoting thousands. Norm. Avg. follows the normalization in Appendix C.4.

Table 7: MAE on SWE-bench Verified. Norm. Avg. is the macro-average of MAE normalized at each prediction point by the history-median MAE. k denotes thousands of tokens. Best and second-best results are in bold and underlined, respectively. Marks are assigned using unrounded values.
<table><tr><td>Prediction point</td><td>TokenCast</td><td>TRAIL</td><td>EGTP</td><td>TIE</td><td>Self-Pred.</td><td>History median</td></tr><tr><td>GPT-5.4</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Task Start</td><td>144.0k</td><td>165.3k</td><td>160.9k</td><td>157.0k</td><td>152.0k</td><td>179.0k</td></tr><tr><td>Call Start</td><td>64.5</td><td>71.1</td><td>70.3</td><td>69.5</td><td>70.9</td><td>74.2</td></tr><tr><td>In-call Update</td><td>38.9</td><td>78.9</td><td>74.6</td><td>77.4</td><td>80.2</td><td>80.3</td></tr><tr><td>Task Update</td><td>80.0k</td><td>115.0k</td><td>117.0k</td><td>119.0k</td><td>126.0k</td><td>130.0k</td></tr><tr><td>Norm. Avg.</td><td>0.69</td><td>0.94</td><td>0.92</td><td>0.92</td><td>0.94</td><td>1.00</td></tr><tr><td>Claude Opus 4.6</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Task Start</td><td>150.7k</td><td>157.5k</td><td>153.8k</td><td>146.8k</td><td>138.9k</td><td>171.0k</td></tr><tr><td>Call Start</td><td>64.1</td><td>69.4</td><td>69.9</td><td>66.8</td><td>68.8</td><td>73.5</td></tr><tr><td>In-call Update</td><td>55.0</td><td>80.2</td><td>71.7</td><td>74.4</td><td>78.1</td><td>83.4</td></tr><tr><td>Task Update</td><td>88.7k</td><td>112.7k</td><td>122.4k</td><td>116.4k</td><td>122.0k</td><td>133.0k</td></tr><tr><td>Norm. Avg.</td><td>0.77</td><td>0.92</td><td>0.91</td><td>0.88</td><td>0.90</td><td>1.00</td></tr><tr><td>Gemini 3.1 Pro</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Task Start</td><td>157.8k</td><td>173.5k</td><td>170.2k</td><td>176.4k</td><td>163.6k</td><td>193.0k</td></tr><tr><td>Call Start</td><td>68.3</td><td>74.7</td><td>71.6</td><td>68.4</td><td>79.8</td><td>88.5</td></tr><tr><td>In-call Update</td><td>41.0</td><td>84.3</td><td>79.4</td><td>77.2</td><td>87.2</td><td>96.2</td></tr><tr><td>Task Update</td><td>69.6k</td><td>126.3k</td><td>121.4k</td><td>118.6k</td><td>132.2k</td><td>151.0k</td></tr><tr><td>Norm. Avg.</td><td>0.62</td><td>0.86</td><td>0.83</td><td>0.82</td><td>0.88</td><td>1.00</td></tr><tr><td>DeepSeek-V4-Pro</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Task Start</td><td>212.9k</td><td>204.9k</td><td>211.5k</td><td>198.4k</td><td>194.5k</td><td>232.0k</td></tr><tr><td>Call Start</td><td>80.0</td><td>89.0</td><td>84.6</td><td>85.9</td><td>91.3</td><td>101.5</td></tr><tr><td>In-call Update</td><td>65.5</td><td>96.1</td><td>94.9</td><td>86.3</td><td>104.8</td><td>112.8</td></tr><tr><td>Task Update</td><td>106.2k</td><td>143.4k</td><td>155.6k</td><td>148.1k</td><td>158.4k</td><td>177.0k</td></tr><tr><td>Norm. Avg.</td><td>0.72</td><td>0.86</td><td>0.87</td><td>0.83</td><td>0.89</td><td>1.00</td></tr><tr><td>Qwen3.8-27B</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Task Start</td><td>192.0k</td><td>232.0k</td><td>209.0k</td><td>215.0k</td><td>221.0k</td><td>284.0k</td></tr><tr><td>Call Start</td><td>87.0</td><td>109.0</td><td>102.9</td><td>99.0</td><td>105.5</td><td>119.0</td></tr><tr><td>In-call Update</td><td>81.9</td><td>120.0</td><td>116.0</td><td>111.4</td><td>108.9</td><td>147.0</td></tr><tr><td>Task Update</td><td>124.0k</td><td>176.0k</td><td>171.0k</td><td>164.0k</td><td>173.0k</td><td>198.0k</td></tr><tr><td>Norm. Avg.</td><td>0.65</td><td>0.86</td><td>0.81</td><td>0.79</td><td>0.82</td><td>1.00</td></tr><tr><td>Llama-3.2-3B-Instruct</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Task Start</td><td>272.3k</td><td>277.6k</td><td>269.7k</td><td>258.2k</td><td>264.8k</td><td>302.0k</td></tr><tr><td>Call Start</td><td>133.8</td><td>140.7</td><td>134.6</td><td>140.0</td><td>141.6</td><td>147.0</td></tr><tr><td>In-call Update</td><td>100.3</td><td>141.8</td><td>137.7</td><td>126.4</td><td>150.2</td><td>158.0</td></tr><tr><td>Task Update</td><td>152.2k</td><td>205.3k</td><td>213.1k</td><td>188.2k</td><td>213.7k</td><td>218.0k</td></tr><tr><td>Norm. Avg.</td><td>0.79</td><td>0.93</td><td>0.91</td><td>0.87</td><td>0.94</td><td>1.00</td></tr></table>

## D.3 Error Relative to Consumption

Table 11 reports WAPE and mean target consumption for GPT-5.4 and Qwen3.8-27B across four benchmarks and four prediction points. For each benchmark, agent LLM, and prediction point, we

Table 8: MAE on Search-R1. Norm. Avg. is the macro-average of MAE normalized at each prediction point by the history-median MAE. k denotes thousands of tokens. Best and second-best results are in bold and underlined, respectively. Marks are assigned using unrounded values.
<table><tr><td>Prediction point</td><td>TokenCast</td><td>TRAIL</td><td>EGTP</td><td>TIE</td><td>Self-Pred.</td><td>History median</td></tr><tr><td>GPT-5.4</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Task Start</td><td>34.2k</td><td>37.8k</td><td>37.6k</td><td>36.4k</td><td>36.9k</td><td>44.9k</td></tr><tr><td>Call Start</td><td>34.5</td><td>37.3</td><td>36.5</td><td>34.0</td><td>39.8</td><td>44.0</td></tr><tr><td>In-call Update</td><td>31.3</td><td>39.4</td><td>38.5</td><td>36.8</td><td>39.9</td><td>42.0</td></tr><tr><td>Task Update</td><td>17.6k</td><td>23.1k</td><td>23.7k</td><td>23.3k</td><td>24.2k</td><td>25.0k</td></tr><tr><td>Norm. Avg.</td><td>0.75</td><td>0.89</td><td>0.88</td><td>0.85</td><td>0.91</td><td>1.00</td></tr><tr><td>Claude Opus 4.6</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Task Start</td><td>31.1k</td><td>33.9k</td><td>32.9k</td><td>32.2k</td><td>29.8k</td><td>39.8k</td></tr><tr><td>Call Start</td><td>30.4</td><td>33.6</td><td>32.0</td><td>30.4</td><td>34.9</td><td>40.2</td></tr><tr><td>In-call Update</td><td>22.8</td><td>33.4</td><td>31.6</td><td>29.7</td><td>34.0</td><td>38.5</td></tr><tr><td>Task Update</td><td>12.9k</td><td>20.8k</td><td>18.7k</td><td>19.5k</td><td>21.4k</td><td>23.6k</td></tr><tr><td>Norm. Avg.</td><td>0.67</td><td>0.86</td><td>0.81</td><td>0.79</td><td>0.85</td><td>1.00</td></tr><tr><td>Gemini 3.1 Pro</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Task Start</td><td>38.0k</td><td>42.3k</td><td>41.0k</td><td>40.7k</td><td>37.1k</td><td>49.8k</td></tr><tr><td>Call Start</td><td>40.0</td><td>44.7</td><td>39.8</td><td>42.3</td><td>47.0</td><td>54.5</td></tr><tr><td>In-call Update</td><td>25.5</td><td>46.8</td><td>44.2</td><td>42.6</td><td>47.5</td><td>57.0</td></tr><tr><td>Task Update</td><td>16.1k</td><td>27.8k</td><td>24.6k</td><td>25.2k</td><td>28.6k</td><td>31.5k</td></tr><tr><td>Norm. Avg.</td><td>0.61</td><td>0.84</td><td>0.78</td><td>0.79</td><td>0.84</td><td>1.00</td></tr><tr><td>DeepSeek-V4-Pro</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Task Start</td><td>46.5k</td><td>50.0k</td><td>47.5k</td><td>44.2k</td><td>42.9k</td><td>58.5k</td></tr><tr><td>Call Start</td><td>50.7</td><td>57.3</td><td>54.0</td><td>50.6</td><td>60.6</td><td>68.0</td></tr><tr><td>In-call Update</td><td>35.1</td><td>60.8</td><td>56.2</td><td>50.4</td><td>62.3</td><td>71.5</td></tr><tr><td>Task Update</td><td>19.9k</td><td>33.7k</td><td>31.1k</td><td>29.8k</td><td>34.5k</td><td>37.2k</td></tr><tr><td>Norm. Avg.</td><td>0.64</td><td>0.86</td><td>0.81</td><td>0.75</td><td>0.86</td><td>1.00</td></tr><tr><td>Qwen3.8-27B</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Task Start</td><td>39.0k</td><td>53.5k</td><td>49.8k</td><td>47.6k</td><td>51.0k</td><td>72.9k</td></tr><tr><td>Call Start</td><td>56.7</td><td>65.9</td><td>71.4</td><td>67.7</td><td>76.1</td><td>91.0</td></tr><tr><td>In-call Update</td><td>57.0</td><td>84.6</td><td>80.7</td><td>75.5</td><td>83.2</td><td>93.0</td></tr><tr><td>Task Update</td><td>24.0k</td><td>39.5k</td><td>34.0k</td><td>35.4k</td><td>38.4k</td><td>41.0k</td></tr><tr><td>Norm. Avg.</td><td>0.59</td><td>0.83</td><td>0.79</td><td>0.77</td><td>0.84</td><td>1.00</td></tr><tr><td>Llama-3.2-3B-Instruct</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Task Start</td><td>59.0k</td><td>60.0k</td><td>57.9k</td><td>53.6k</td><td>52.4k</td><td>68.0k</td></tr><tr><td>Call Start</td><td>79.0</td><td>80.9</td><td>70.4</td><td>75.4</td><td>84.6</td><td>95.0</td></tr><tr><td>In-call Update</td><td>64.0</td><td>99.4</td><td>91.2</td><td>84.7</td><td>95.6</td><td>108.0</td></tr><tr><td>Task Update</td><td>31.0k</td><td>43.8k</td><td>40.2k</td><td>38.6k</td><td>45.9k</td><td>49.0k</td></tr><tr><td>Norm. Avg.</td><td>0.73</td><td>0.89</td><td>0.81</td><td>0.79</td><td>0.87</td><td>1.00</td></tr></table>

Table 9: MAE on MMLU-Pro. Norm. Avg. is the macro-average of MAE normalized at each prediction point by the history-median MAE. k denotes thousands of tokens. Best and second-best results are in bold and underlined, respectively. Marks are assigned using unrounded values.
<table><tr><td>Prediction point</td><td>TokenCast</td><td>TRAIL</td><td>EGTP</td><td>TIE</td><td>Self-Pred.</td><td>History median</td></tr><tr><td>GPT-5.4</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Task Start</td><td>8.0k</td><td>8.8k</td><td>8.5k</td><td>8.2k</td><td>7.4k</td><td>9.7k</td></tr><tr><td>Call Start</td><td>14.8</td><td>16.7</td><td>15.5</td><td>15.8</td><td>17.6</td><td>19.6</td></tr><tr><td>In-call Update</td><td>9.2</td><td>19.6</td><td>18.3</td><td>16.7</td><td>17.4</td><td>24.3</td></tr><tr><td>Task Update</td><td>4.3k</td><td>6.2k</td><td>6.5k</td><td>6.0k</td><td>5.8k</td><td>6.9k</td></tr><tr><td>Norm. Avg.</td><td>0.65</td><td>0.87</td><td>0.84</td><td>0.80</td><td>0.80</td><td>1.00</td></tr><tr><td>Claude Opus 4.6</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Task Start</td><td>7.5k</td><td>8.0k</td><td>7.7k</td><td>7.3k</td><td>6.9k</td><td>8.9k</td></tr><tr><td>Call Start</td><td>14.9</td><td>14.8</td><td>15.6</td><td>14.3</td><td>16.1</td><td>18.3</td></tr><tr><td>In-call Update</td><td>11.5</td><td>15.5</td><td>16.8</td><td>16.0</td><td>18.4</td><td>22.0</td></tr><tr><td>Task Update</td><td>4.4k</td><td>6.0k</td><td>6.1k</td><td>5.8k</td><td>5.6k</td><td>6.3k</td></tr><tr><td>Norm. Avg.</td><td>0.72</td><td>0.84</td><td>0.86</td><td>0.81</td><td>0.85</td><td>1.00</td></tr><tr><td>Gemini 3.1 Pro</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Task Start</td><td>8.6k</td><td>10.1k</td><td>9.6k</td><td>9.3k</td><td>8.9k</td><td>11.5k</td></tr><tr><td>Call Start</td><td>18.0</td><td>18.3</td><td>20.1</td><td>19.4</td><td>21.3</td><td>23.9</td></tr><tr><td>In-call Update</td><td>10.5</td><td>24.7</td><td>20.6</td><td>21.8</td><td>23.6</td><td>28.7</td></tr><tr><td>Task Update</td><td>4.7k</td><td>7.2k</td><td>6.8k</td><td>6.4k</td><td>7.5k</td><td>8.0k</td></tr><tr><td>Norm. Avg.</td><td>0.61</td><td>0.85</td><td>0.81</td><td>0.79</td><td>0.86</td><td>1.00</td></tr><tr><td>DeepSeek-V4-Pro</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Task Start</td><td>11.6k</td><td>12.3k</td><td>11.9k</td><td>11.0k</td><td>10.6k</td><td>13.7k</td></tr><tr><td>Call Start</td><td>24.4</td><td>26.2</td><td>24.8</td><td>23.5</td><td>26.9</td><td>29.6</td></tr><tr><td>In-call Update</td><td>19.5</td><td>31.8</td><td>28.7</td><td>25.4</td><td>30.4</td><td>35.9</td></tr><tr><td>Task Update</td><td>6.9k</td><td>8.5k</td><td>9.2k</td><td>8.9k</td><td>9.4k</td><td>9.8k</td></tr><tr><td>Norm. Avg.</td><td>0.73</td><td>0.88</td><td>0.86</td><td>0.80</td><td>0.87</td><td>1.00</td></tr><tr><td>Qwen3.8-27B</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Task Start</td><td>13.0k</td><td>14.3k</td><td>12.5k</td><td>13.7k</td><td>13.3k</td><td>16.2k</td></tr><tr><td>Call Start</td><td>26.5</td><td>30.3</td><td>28.6</td><td>27.9</td><td>31.1</td><td>35.1</td></tr><tr><td>In-call Update</td><td>21.1</td><td>37.7</td><td>35.5</td><td>34.3</td><td>32.8</td><td>42.4</td></tr><tr><td>Task Update</td><td>8.4k</td><td>11.2k</td><td>10.3k</td><td>10.7k</td><td>11.5k</td><td>12.0k</td></tr><tr><td>Norm. Avg.</td><td>0.69</td><td>0.89</td><td>0.82</td><td>0.84</td><td>0.86</td><td>1.00</td></tr><tr><td>Llama-3.2-3B-Instruct</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Task Start</td><td>16.4k</td><td>17.5k</td><td>16.9k</td><td>15.4k</td><td>15.8k</td><td>19.6k</td></tr><tr><td>Call Start</td><td>33.5</td><td>36.5</td><td>33.2</td><td>33.8</td><td>37.9</td><td>41.8</td></tr><tr><td>In-call Update</td><td>30.7</td><td>46.0</td><td>43.7</td><td>42.3</td><td>47.3</td><td>50.2</td></tr><tr><td>Task Update</td><td>10.8k</td><td>13.2k</td><td>13.8k</td><td>13.5k</td><td>12.9k</td><td>14.4k</td></tr><tr><td>Norm. Avg.</td><td>0.75</td><td>0.90</td><td>0.87</td><td>0.84</td><td>0.89</td><td>1.00</td></tr></table>

Table 10: MAE on LongBench-v2. Norm. Avg. is the macro-average of MAE normalized at each prediction point by the history-median MAE. k denotes thousands of tokens. Best and second-best results are in bold and underlined, respectively. Marks are assigned using unrounded values.
<table><tr><td>Prediction point</td><td>TokenCast</td><td>TRAIL</td><td>EGTP</td><td>TIE</td><td>Self-Pred.</td><td>History median</td></tr><tr><td>GPT-5.4</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Task Start</td><td>32.2k</td><td>37.1k</td><td>36.0k</td><td>35.1k</td><td>32.7k</td><td>44.0k</td></tr><tr><td>Call Start</td><td>53.2</td><td>60.6</td><td>56.9</td><td>53.7</td><td>61.2</td><td>74.0</td></tr><tr><td>In-call Update</td><td>35.5</td><td>73.5</td><td>68.8</td><td>62.7</td><td>70.1</td><td>91.0</td></tr><tr><td>Task Update</td><td>13.9k</td><td>26.9k</td><td>24.2k</td><td>25.1k</td><td>28.2k</td><td>33.0k</td></tr><tr><td>Norm. Avg.</td><td>0.57</td><td>0.82</td><td>0.77</td><td>0.74</td><td>0.80</td><td>1.00</td></tr><tr><td>Claude Opus 4.6</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Task Start</td><td>28.7k</td><td>33.4k</td><td>32.0k</td><td>30.7k</td><td>29.4k</td><td>40.5k</td></tr><tr><td>Call Start</td><td>48.1</td><td>57.2</td><td>50.0</td><td>54.5</td><td>56.0</td><td>67.5</td></tr><tr><td>In-call Update</td><td>41.3</td><td>69.5</td><td>57.9</td><td>61.3</td><td>65.4</td><td>82.0</td></tr><tr><td>Task Update</td><td>14.9k</td><td>23.8k</td><td>22.4k</td><td>21.5k</td><td>25.1k</td><td>28.5k</td></tr><tr><td>Norm. Avg.</td><td>0.61</td><td>0.84</td><td>0.76</td><td>0.77</td><td>0.81</td><td>1.00</td></tr><tr><td>Gemini 3.1 Pro</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Task Start</td><td>39.9k</td><td>43.2k</td><td>41.9k</td><td>38.2k</td><td>38.9k</td><td>50.0k</td></tr><tr><td>Call Start</td><td>60.6</td><td>72.3</td><td>66.8</td><td>66.0</td><td>75.0</td><td>85.0</td></tr><tr><td>In-call Update</td><td>43.7</td><td>88.4</td><td>80.6</td><td>75.1</td><td>85.8</td><td>103.0</td></tr><tr><td>Task Update</td><td>15.8k</td><td>31.2k</td><td>27.2k</td><td>28.6k</td><td>33.0k</td><td>38.0k</td></tr><tr><td>Norm. Avg.</td><td>0.59</td><td>0.85</td><td>0.78</td><td>0.76</td><td>0.84</td><td>1.00</td></tr><tr><td>DeepSeek-V4-Pro</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Task Start</td><td>52.3k</td><td>55.0k</td><td>54.9k</td><td>51.2k</td><td>49.6k</td><td>61.5k</td></tr><tr><td>Call Start</td><td>84.5</td><td>96.0</td><td>89.6</td><td>85.6</td><td>97.9</td><td>108.0</td></tr><tr><td>In-call Update</td><td>77.2</td><td>115.6</td><td>110.2</td><td>101.7</td><td>118.4</td><td>132.0</td></tr><tr><td>Task Update</td><td>26.1k</td><td>35.7k</td><td>38.7k</td><td>37.4k</td><td>42.2k</td><td>47.5k</td></tr><tr><td>Norm. Avg.</td><td>0.69</td><td>0.85</td><td>0.84</td><td>0.80</td><td>0.87</td><td>1.00</td></tr><tr><td>Qwen3.8-27B</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Task Start</td><td>50.8k</td><td>60.7k</td><td>55.9k</td><td>54.6k</td><td>52.6k</td><td>70.0k</td></tr><tr><td>Call Start</td><td>94.4</td><td>107.8</td><td>94.1</td><td>99.7</td><td>110.8</td><td>124.0</td></tr><tr><td>In-call Update</td><td>95.0</td><td>136.4</td><td>128.9</td><td>121.2</td><td>118.6</td><td>151.0</td></tr><tr><td>Task Update</td><td>26.9k</td><td>47.6k</td><td>42.7k</td><td>44.5k</td><td>49.0k</td><td>56.0k</td></tr><tr><td>Norm. Avg.</td><td>0.65</td><td>0.87</td><td>0.79</td><td>0.80</td><td>0.83</td><td>1.00</td></tr><tr><td>Llama-3.2-3B-Instruct</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Task Start</td><td>70.6k</td><td>73.6k</td><td>69.2k</td><td>66.3k</td><td>71.4k</td><td>82.0k</td></tr><tr><td>Call Start</td><td>116.8</td><td>128.0</td><td>116.2</td><td>121.8</td><td>133.7</td><td>142.0</td></tr><tr><td>In-call Update</td><td>111.9</td><td>156.2</td><td>141.9</td><td>145.2</td><td>162.7</td><td>171.0</td></tr><tr><td>Task Update</td><td>38.6k</td><td>56.8k</td><td>55.4k</td><td>52.1k</td><td>60.2k</td><td>64.0k</td></tr><tr><td>Norm. Avg.</td><td>0.74</td><td>0.90</td><td>0.84</td><td>0.83</td><td>0.93</td><td>1.00</td></tr></table>

use the same evaluation weights as MAE:

$$
\mathrm { W A P E } ( \mathcal { Y } _ { 0 } ) = 1 0 0 \frac { \sum _ { i } w _ { i } \lvert \widehat { y } _ { i } - y _ { i } \rvert } { \sum _ { i } w _ { i } y _ { i } } = 1 0 0 \frac { \mathrm { M A E } } { \bar { y } } , \qquad \bar { y } = \sum _ { i } w _ { i } y _ { i } , \quad \sum _ { i } w _ { i } = 1 .\tag{8}
$$

Here, $y _ { i }$ is the target consumption: total task consumption T at Task Start, full-call consumption $C _ { k }$ at Call Start and In-call Update, and remaining task consumption $R _ { k }$ at Task Update. The weights follow Appendix C.4. All methods within a row share the same target mean and evaluation weights. A WAPE of 15% means that the weighted MAE is 15% of the weighted mean target.

Table 11: WAPE (%) at four prediction points. Best and second-best results are in bold and underlined, respectively. Marks are assigned using unrounded values.
<table><tr><td>Agent LLM</td><td>Prediction point Mean target (k)</td><td></td><td>TokenCast</td><td>TRAIL</td><td>EGTP</td><td>TIE</td><td>Self-Pred.</td></tr><tr><td colspan="8">SWE-bench Verified</td></tr><tr><td rowspan="4">GPT-5.4</td><td>Task Start</td><td>475.6</td><td>30.3</td><td>34.8</td><td>33.8</td><td>33.0</td><td>32.0</td></tr><tr><td>Call Start</td><td>19.6</td><td>0.329</td><td>0.363</td><td>0.359</td><td>0.355</td><td>0.362</td></tr><tr><td>In-call Update</td><td>20.1</td><td>0.194</td><td>0.393</td><td>0.371</td><td>0.385</td><td>0.399</td></tr><tr><td>Task Update</td><td>286.3</td><td>27.9</td><td>40.2</td><td>40.9</td><td>41.6</td><td>44.0</td></tr><tr><td rowspan="4">Qwen3.8-27B</td><td>Task Start</td><td>611.5</td><td>31.4</td><td>37.9</td><td>34.2</td><td>35.2</td><td>36.1</td></tr><tr><td>Call Start</td><td>25.4</td><td>0.343</td><td>0.429</td><td>0.405</td><td>0.390</td><td>0.415</td></tr><tr><td>In-call Update</td><td>25.9</td><td>0.316</td><td>0.463</td><td>0.448</td><td>0.430</td><td>0.420</td></tr><tr><td>Task Update</td><td>366.3</td><td>33.9</td><td>48.0</td><td>46.7</td><td>44.8</td><td>47.2</td></tr><tr><td colspan="8">Search-R1</td></tr><tr><td rowspan="4">GPT-5.4</td><td>Task Start</td><td>53.5</td><td>63.9</td><td>70.7</td><td>70.3</td><td>68.0</td><td>69.0</td></tr><tr><td>Call Start</td><td>6.0</td><td>0.575</td><td>0.622</td><td>0.608</td><td>0.567</td><td>0.663</td></tr><tr><td>In-call Update</td><td>6.1</td><td>0.513</td><td>0.646</td><td>0.631</td><td>0.603</td><td>0.654</td></tr><tr><td>Task Update</td><td>31.6</td><td>55.7</td><td>73.1</td><td>75.0</td><td>73.7</td><td>76.6</td></tr><tr><td rowspan="4">Qwen3.8-27B</td><td>Task Start</td><td>98.1</td><td>39.8</td><td>54.5</td><td>50.8</td><td>48.5</td><td>52.0</td></tr><tr><td>Call Start</td><td>9.5</td><td>0.597</td><td>0.694</td><td>0.752</td><td>0.713</td><td>0.801</td></tr><tr><td>In-call Update</td><td>9.8 57.6</td><td>0.582</td><td>0.863</td><td>0.823</td><td>0.770</td><td>0.849</td></tr><tr><td>Task Update</td><td></td><td>41.7</td><td>68.6</td><td>59.0</td><td>61.5</td><td>66.7</td></tr><tr><td colspan="8">MMLU-Pro</td></tr><tr><td rowspan="4">GPT-5.4</td><td>Task Start</td><td>50.6</td><td>15.8</td><td>17.4</td><td>16.8</td><td>16.2</td><td>14.6</td></tr><tr><td>Call Start</td><td>8.2</td><td>0.180</td><td>0.204</td><td>0.189</td><td>0.193</td><td>0.215</td></tr><tr><td>In-call Update</td><td>8.4</td><td>0.110</td><td>0.233</td><td>0.218</td><td>0.199</td><td>0.207</td></tr><tr><td>Task Update</td><td>27.9</td><td>15.4</td><td>22.2</td><td>23.3</td><td>21.5</td><td>20.8</td></tr><tr><td rowspan="4">Qwen3.8-27B</td><td>Task Start</td><td>63.8</td><td>20.4</td><td>22.4</td><td>19.6</td><td>21.5</td><td>20.8</td></tr><tr><td>Call Start</td><td>10.0</td><td>0.265</td><td>0.303</td><td>0.286</td><td>0.279</td><td>0.311</td></tr><tr><td>In-call Update</td><td>10.3</td><td>0.205</td><td>0.366</td><td>0.345</td><td>0.333</td><td>0.318</td></tr><tr><td>Task Update</td><td>35.2</td><td>23.9</td><td>31.8</td><td>29.3</td><td>30.4</td><td>32.7</td></tr><tr><td colspan="8">LongBench-v2</td></tr><tr><td rowspan="4">GPT-5.4</td><td>Task Start</td><td>771.2</td><td>4.2</td><td>4.8</td><td>4.7</td><td>4.6</td><td>4.2</td></tr><tr><td>Call Start</td><td>135.7</td><td>0.039</td><td>0.045</td><td>0.042</td><td>0.040</td><td>0.045</td></tr><tr><td>In-call Update</td><td>136.6</td><td>0.026</td><td>0.054</td><td>0.050</td><td>0.046</td><td>0.051</td></tr><tr><td>Task Update</td><td>391.2</td><td>3.6</td><td>6.9</td><td>6.2</td><td>6.4</td><td>7.2</td></tr><tr><td rowspan="4">Qwen3.8-27B</td><td>Task Start</td><td>856.5</td><td>5.9</td><td>7.1</td><td>6.5</td><td>6.4</td><td>6.1</td></tr><tr><td>Call Start</td><td>112.1</td><td>0.084</td><td>0.096</td><td></td><td>0.084 0.089</td><td>0.099</td></tr><tr><td>In-call Update</td><td>112.6</td><td>0.084</td><td>0.121</td><td></td><td>0.114 0.108</td><td>0.105</td></tr><tr><td>Task Update</td><td>434.2</td><td>6.2</td><td>11.0</td><td>9.8</td><td>10.2</td><td>11.3</td></tr></table>

## D.4 Generalization

## D.4.1 Unseen Task Types

We use the publicly released LiveClawBench trajectories [Long et al., 2026], which contain tasks from 10 application domains executed under a shared agent framework. For each agent LLM, we perform leave-one-domain-out evaluation, treating one domain as the unseen task type and training on the remaining domains. Zero-shot transfer uses no trajectories from the held-out domain. For adaptation, we add k held-out-domain tasks to the training set and evaluate on the remaining held-out tasks. The target-only baseline is trained on the same k tasks without source-domain data. All runs of a task remain in one partition. Relative to Self-Prediction, zero-shot MAE is 1.31 at Call Start and 1.47 at Task Update. With 10 target tasks, the ratios fall to 0.86 and 0.91; with 20, they reach 0.82 and 0.85. The trajectory-selection criteria are in Appendix C.1.

## D.4.2 Unseen Agent LLMs

We hold out each agent LLM in turn and train on trajectories from the remaining models under a shared task-collection and agent framework. Adaptation uses k tasks executed by the held-out model, with task partitions shared across models and adaptation tasks disjoint from the test tasks. At Call Start, zero-shot MAE is 1.02 times the target history median; this ratio falls to 0.96 with 10 target-model tasks and 0.95 with 20. With 3–10 tasks, the transferred predictor achieves lower MAE than training on the same target-model tasks alone. Figure 7 also reports an auxiliary Task Update evaluation of remaining output tokens, a different target from total remaining consumption. Its normalized MAE ranges from 1.01 to 1.07 with up to 20 target-model tasks and reaches 0.98 when all target-model tasks are available. The target-only predictor starts at 1.22 with three tasks and approaches the history median as additional target-model trajectories are provided.

![](images/d9cdfe2af2f31cf8ff525523a7d331cf5eaa571619a834563571bfaca3cde96c.jpg)  
(a) Call Start

![](images/21525fdac7f3dfdd340057d03b2faf6ec9c8eb7efd9ec0b04353b582e863b4e1.jpg)  
(b) Task Update, output  
Figure 7: Transfer to unseen agent LLMs with increasing amounts of target-model data. The panels report Call Start and remaining output-token prediction at Task Update. MAE is normalized by the target history median; lower is better.

## D.4.3 Cross-Harness Transfer

Table 12 reports TokenCast’s MAE relative to the history median under each test harness. It also reports transfer from DeepSeek Harness training runs to OpenHands runs of held-out tasks.

## D.4.4 Length Extrapolation

We sort the SWE-bench Verified tasks by their median number of calls into five bands, train on the four shorter bands, and evaluate on the longest band. TokenCast achieves lower MAE than Self-Prediction at all four prediction points on these longer tasks, with reductions of 12.4% at Task

Table 12: TokenCast MAE divided by the history-median MAE on the test harness. Lower is better. The two OpenHands rows use the same test runs.
<table><tr><td>Training harness</td><td>Test harness</td><td>Task Start</td><td>Call t Start</td><td>In-call Update</td><td>Task Update</td></tr><tr><td>DeepSeek Harness</td><td>DeepSeek Harness</td><td>0.90</td><td>0.71</td><td>0.64</td><td>0.82</td></tr><tr><td>OpenHands</td><td>OpenHands</td><td>0.83</td><td>0.77</td><td>0.68</td><td>0.75</td></tr><tr><td>DeepSeek Harness OpenHands</td><td></td><td>1.04</td><td>0.88</td><td>0.79</td><td>0.96</td></tr></table>

Start, 15.7% at Call Start, 44.9% at In-call Update, and 26.8% at Task Update. Table 13 additionally compares this length-based split with random splits of the same fold sizes. Relative to the random splits, MAE differs by +7.8% at Task Start, +1.9% at Call Start, −2.6% at In-call Update, and +7.1% at Task Update, so the longer test band raises MAE by at most 7.8%.

Table 13: Length extrapolation to tasks longer than those observed during training. MAE changes are reported relative to Self-Prediction and matched random splits.
<table><tr><td>Prediction point</td><td>vs. Self-Prediction ↓</td><td>vs. random split ↓</td></tr><tr><td>Task Start</td><td>-12.4%</td><td>+7.8%</td></tr><tr><td>Call Start</td><td>-15.7%</td><td>+1.9%</td></tr><tr><td>In-call Update</td><td>-44.9%</td><td>-2.6%</td></tr><tr><td>Task Update</td><td>-26.8%</td><td>+7.1%</td></tr></table>

## D.4.5 Reasoning Configuration Shift

We evaluate a shift in GPT-5.4’s reasoning configuration by rerunning the same 69 SWE-bench Verified tasks with reasoning effort set to off and comparing them with the corresponding runs under the low setting. Median run consumption decreases from 273k to 238k tokens. When trained and evaluated within the off configuration, TokenCast reduces MAE relative to Self-Prediction by 20.7% at Call Start, 50.8% at In-call Update, and 32.2% at Task Update (Table 14). We also transfer the predictor trained under the low configuration directly to the off configuration and adapt it using 20 target-configuration tasks. After adaptation, MAE decreases to 37.9 at Call Start, 23.6 at In-call Update, and 76.0k at Task Update.

Table 14: MAE under a GPT-5.4 reasoning-configuration shift on SWE-bench Verified at the three online prediction points. Results include evaluation within the off configuration, direct transfer from low to off, and adaptation with 20 target-configuration tasks.
<table><tr><td rowspan="2">Prediction point</td><td colspan="2">Off</td><td colspan="2">Low → Off</td></tr><tr><td>TokenCast</td><td>Self-Pred.</td><td>Direct</td><td>20 tasks</td></tr><tr><td>Call Start</td><td>37.1</td><td>46.8</td><td>50.7</td><td>37.9</td></tr><tr><td>In-call Update</td><td>22.5</td><td>45.7</td><td>33.0</td><td>23.6</td></tr><tr><td>Task Update</td><td>82.0k</td><td>121.0k</td><td>82.0k</td><td>76.0k</td></tr></table>

## D.5 Online Update Frequency

We measure the prediction overhead of refreshing the Task Update forecast at different intervals on SWE-bench Verified with GPT-5.4. Starting from the same initial forecast, the 1-, 3-, and 5-call settings refresh the prediction after every 1, 3, or 5 completed calls, respectively, and End makes only the initial forecast. At skipped checkpoints, the latest forecast of final task consumption is retained, the tokens confirmed so far are subtracted, and the remaining estimate is clipped at zero.

Table 15 reports the mean number of forecasts and cumulative prediction time per run. Counts include the initial Task Start forecast, and times are reported as mean ± standard deviation across runs. Updating every call makes 19.7 forecasts and takes $3 2 . 8 \pm 3 0 . \mathrm { \xi }$ 5 ms per run. The three-call and five-call schedules reduce these to 6.9 forecasts and $1 1 . 9 { \pm } 1 0 . 4 \mathrm { m s } .$ , and 4.3 forecasts and 7.7±6.4 ms, respectively. End makes only the initial forecast and takes 2.1 ± 0.8 ms. Every-call updating therefore remains a small fraction of the 129 s median run time reported in Appendix C.1.

Table 15: Prediction counts and cumulative overhead at online update intervals on SWE-bench Verified. Predictions/run includes the initial Task Start forecast and later Task Update refreshes.
<table><tr><td>Update interval</td><td>Predictions/run ↓</td><td>Time/run (ms) ↓</td></tr><tr><td>1 call</td><td>19.7</td><td> $3 2 . 8 \pm 3 0 . 5$ </td></tr><tr><td>3 calls</td><td>6.9</td><td> $1 1 . 9 \pm 1 0 . 4$ </td></tr><tr><td>5 calls</td><td>4.3</td><td> $7 . 7 \pm 6 . 4$ </td></tr><tr><td>End</td><td>1.0</td><td> $2 . 1 \pm 0 . 8$ </td></tr></table>

## D.6 Ablation Study

Base predictor comparison. Table 16 compares base predictors in the same pipeline. Ridge regression and KNN reach normalized average MAEs of 0.93 and 0.97. Random Forest reaches 0.81, and the MLP reaches 0.78 with 87.2% interval coverage. The three gradient-boosting methods reach 0.75 or lower, and LightGBM gives the lowest error, 0.69, and the highest coverage, 90.6%.

The base-predictor comparison reports 0.8 ms for LightGBM inference, compared with 3.9 ms for XGBoost and 5.2 ms for CatBoost. Under every-call updating, the complete pipeline, including feature extraction and composition, makes 19.7 task-level forecasts and takes 32.8 ms per SWE-bench Verified run on average. We use LightGBM as the base predictor.

Table 16: Base predictor comparison on SWE-bench Verified (GPT-5.4). Norm. Avg. is the normalized MAE averaged over the four prediction points. 90% Cov. is the empirical coverage of the 90% prediction interval. Best and second-best results in each column are in bold and underlined, respectively. Coverage is ranked high to low and other metrics low to high.
<table><tr><td>Predictor</td><td>Norm. Avg. ↓</td><td>90% Cov. (%)</td><td>Model time (ms)</td></tr><tr><td>Ridge Regression</td><td>0.93</td><td>82.4</td><td>0.2</td></tr><tr><td>KNN</td><td>0.97</td><td>79.6</td><td>14.3</td></tr><tr><td>Random Forest</td><td>0.81</td><td>86.8</td><td>2.7</td></tr><tr><td>XGBoost</td><td>0.73</td><td>89.3</td><td>3.9</td></tr><tr><td>CatBoost</td><td>0.75</td><td>88.7</td><td>5.2</td></tr><tr><td>MLP</td><td>0.78</td><td>87.2</td><td>6.1</td></tr><tr><td>LightGBM</td><td>0.69</td><td>90.6</td><td>0.8</td></tr></table>

Prediction strategy. At Task Start, no segment has finished, so the prefix–suffix decomposition has no observed anchor. The compositional path has a normalized MAE of 0.87, compared with 0.82 for direct regression. At Task Update, observed segment boundaries provide information about the input baseline and context growth. The compositional path improves to 0.66, below direct regression’s 0.72. Averaging the two paths reduces MAE to 0.81 at Task Start and 0.64 at Task Update. The correction model uses the direct and compositional forecasts together with the predicted boundary variables, reducing these values to 0.80 and 0.62, with 90.6% coverage (Table 17). It is trained on out-of-fold outputs of the complete pipeline.

Segment representation ablation. Regressing directly on raw execution features raises the normalized average MAE to 0.85, with task-level and call-level MAE of 0.88 and 0.82. Retaining the triple while removing composition gives 0.78. This intervention also changes call-level MAE from 0.68 to

Table 17: Effect of forecasting strategy on SWE-bench Verified (GPT-5.4). Task Start and Task Update columns report the normalized MAE at these two task-level prediction settings. Best and second-best results in each column are in bold and underlined, respectively. Coverage is ranked high to low and other metrics low to high.
<table><tr><td>Strategy</td><td>Task Start ↓</td><td>Task Update ↓</td><td>90% Cov. (%)</td></tr><tr><td>Direct only</td><td>0.82</td><td>0.72</td><td>88.1</td></tr><tr><td>Compositional only</td><td>0.87</td><td>0.66</td><td>89.3</td></tr><tr><td>Direct-compositional average</td><td>0.81</td><td>0.64</td><td>89.8</td></tr><tr><td>Full correction pipeline</td><td>0.80</td><td>0.62</td><td>90.6</td></tr></table>

0.74, so the ablation affects more than the task-level composition readout.

Removing g raises the normalized average MAE to 0.76 and task-level MAE from 0.71 to 0.79. Removing b raises it to 0.73 (Table 18). The larger effect of removing g is consistent with its role in the cross-term $n _ { B } g _ { A } ;$ the ablations also change call-level predictions and do not isolate this term.

Table 18: Ablation of the segment representation on SWE-bench Verified (GPT-5.4). Task-level and Call-level columns report normalized MAE averaged over the two prediction points within each level. Best and second-best results in each column are in bold and underlined, respectively.
<table><tr><td>Variant</td><td>Norm. Avg. ↓</td><td>Task-level ↓</td><td>Call-level ↓</td></tr><tr><td>Raw features</td><td>0.85</td><td>0.88</td><td>0.82</td></tr><tr><td>No composition</td><td>0.78</td><td>0.81</td><td>0.74</td></tr><tr><td>Drop g</td><td>0.76</td><td>0.79</td><td>0.72</td></tr><tr><td>Drop b</td><td>0.73</td><td>0.75</td><td>0.70</td></tr><tr><td>Full</td><td>0.69</td><td>0.71</td><td>0.68</td></tr></table>

Component ablation. The suffix model takes predictions from the prefix model as input. A shift in this input between training and inference can therefore propagate errors through the cascade. Cross-fitting reduces the shift, while the cost-weighted loss scales errors in the coupled variables g and n by their downstream impact. Removing either component raises the normalized average MAE by 0.03–0.04. Removing both raises MAE to 0.78 and lowers coverage to 87.4% (Table 19).

Without the correction model, MAE rises to 0.74 and coverage falls to 89.0%. Removing the boundary variables raises MAE to 0.72, consistent with their role in conditioning the suffix model.

Table 19: Component ablation on SWE-bench Verified (GPT-5.4). Each row removes one component from the full pipeline. Best and second-best results in each column are in bold and underlined, respectively. Coverage is ranked high to low and other metrics low to high.
<table><tr><td>Configuration</td><td>Norm. Avg. ↓</td><td>90% Cov. (%)</td></tr><tr><td>Full</td><td>0.69</td><td>90.6</td></tr><tr><td>– Cross-fitting</td><td>0.73</td><td>88.9</td></tr><tr><td>– Cost weighting</td><td>0.72</td><td>89.4</td></tr><tr><td>– Correction</td><td>0.74</td><td>89.0</td></tr><tr><td>— Boundary vars</td><td>0.72</td><td>90.1</td></tr><tr><td>– Cross-fit. &amp; cost-wt.</td><td>0.78</td><td>87.4</td></tr></table>

## D.7 Budget Control

Replay and stopping rules. We replay 288 GPT-5.4 runs from 144 SWE-bench Verified tasks, with two recorded runs per task. Seven budgets are defined by the 0.3 to 0.9 quantiles of recorded run consumption. At each Task Update checkpoint, the controller adds confirmed consumption to a selected quantile of predicted remaining consumption and stops the run when the sum exceeds the budget. A replay is trace-complete if it reaches its recorded terminal state. For each method, budget, and test fold, the controller selects the 0.05, 0.5, or 0.95 forecast quantile on runs outside the test fold, choosing the one using the fewest tokens while reaching at least the fixed-budget trace-completion rate. Results are pooled over out-of-fold test runs and averaged over three split seeds.

Cost accounting. A controller-stopped run is charged the consumption recorded at its stopping checkpoint. A run that finishes before the fixed cap is charged its recorded total; otherwise it is stopped at the cap, with wall time scaled by the recorded token progress. Prediction processing and time are included in the replay accounting. For encoder-based baselines, prediction tokens count locally processed text; Self-Prediction’s count includes additional provider-recorded LLM usage. Savings are relative to the fixed-budget strategy at the same budget. Table 20 lists the complete results, Figure 8 shows the budget-wise response, and Table 21 compares prediction overhead.

![](images/ae8e4fee69d46fa4b4c2c31b7829cc28cf7863f1ff2d6f7b721abdc4140d3426.jpg)  
(a) Net token saving

![](images/3b3c0735883053d45a5e5986732bcdcbce5951c0a061f96369e3242603719863.jpg)  
(b) Completion-rate change  
Figure 8: Trace completion and token consumption across seven budgets in the offline replay.

Table 20: Budget control on 288 GPT-5.4 runs from 144 SWE-bench Verified tasks. Tokens are in thousands per run, wall time is in seconds per run, and trace completion and savings are in percent. Only TokenCast matches the fixed-budget trace completion at every budget.
<table><tr><td>Budget (k tokens)</td><td>Method</td><td>Trace-complete (%)</td><td>Execution (k/run)</td><td>Prediction (k/run)</td><td>Total (k/run)</td><td>Time (s/run)</td><td>Saving (%)</td></tr><tr><td rowspan="6">174</td><td>Fixed budget</td><td>30.2</td><td>154.7</td><td>0.0</td><td>154.7</td><td>113.7</td><td></td></tr><tr><td>TokenCast</td><td>30.2</td><td>101.1</td><td>0.0</td><td>101.1</td><td>76.5</td><td>34.6</td></tr><tr><td>TRAIL</td><td>29.5</td><td>91.0</td><td>12.5</td><td>103.5</td><td>73.0</td><td>33.1</td></tr><tr><td>EGTP</td><td>13.5</td><td>25.3</td><td>3.6</td><td>28.9</td><td>22.6</td><td>81.3</td></tr><tr><td>TIE</td><td>27.4</td><td>59.8</td><td>8.0</td><td>67.8</td><td>45.6</td><td>56.2</td></tr><tr><td>Self-Pred.</td><td>19.4</td><td>70.0</td><td>92.0</td><td>162.0</td><td>303.5</td><td>-4.7</td></tr><tr><td rowspan="6">203</td><td>Fixed budget</td><td>39.9</td><td>173.5</td><td>0.0</td><td>173.5</td><td>125.3</td><td></td></tr><tr><td>TokenCast</td><td>39.9</td><td>121.2</td><td>0.0</td><td>121.2</td><td>90.0</td><td>30.1</td></tr><tr><td>TRAIL</td><td>39.2</td><td>112.6</td><td>14.5</td><td>127.1</td><td>87.6</td><td>26.7</td></tr><tr><td>EGTP</td><td>17.0</td><td>31.8</td><td>4.8</td><td>36.6</td><td>27.6</td><td>78.9</td></tr><tr><td>TIE</td><td>34.0</td><td>73.5</td><td>9.7</td><td>83.2</td><td>54.9</td><td>52.0</td></tr><tr><td>Self-Pred.</td><td>24.7</td><td>84.5</td><td>109.2</td><td>193.7</td><td>355.6</td><td>-11.6</td></tr><tr><td rowspan="6">234</td><td>Fixed budget</td><td>50.0</td><td>190.7</td><td>0.0</td><td>190.7</td><td>135.8</td><td></td></tr><tr><td>TokenCast</td><td>50.0</td><td>141.2</td><td>0.0</td><td>141.2</td><td>103.1</td><td>26.0</td></tr><tr><td>TRAIL</td><td>49.7</td><td>132.3</td><td>16.7</td><td>149.0</td><td>101.0</td><td>21.9</td></tr><tr><td>EGTP</td><td>24.7</td><td>42.7</td><td>5.6</td><td>48.3</td><td>34.8</td><td>74.7</td></tr><tr><td>TIE</td><td>43.4</td><td>92.3</td><td>11.4</td><td>103.7</td><td>67.1</td><td>45.6</td></tr><tr><td>Self-Pred.</td><td>29.5</td><td>100.2</td><td>127.5</td><td>227.7</td><td>413.3</td><td>-19.4</td></tr><tr><td rowspan="7">277</td><td>Fixed budget</td><td>59.7</td><td>210.4</td><td>0.0</td><td>210.4</td><td>148.5</td><td></td></tr><tr><td>TokenCast TRAIL</td><td>59.7</td><td>165.0</td><td>0.0</td><td>165.0</td><td>118.1</td><td>21.6</td></tr><tr><td></td><td>59.7</td><td>157.3</td><td>19.9</td><td>177.2</td><td>118.7</td><td>15.8</td></tr><tr><td>EGTP</td><td>30.9</td><td>53.4</td><td>6.9</td><td>60.3</td><td>42.1</td><td>71.3</td></tr><tr><td>TIE</td><td>52.8</td><td>111.5</td><td>13.8</td><td>125.3</td><td>80.1</td><td>40.4</td></tr><tr><td>Self-Pred.</td><td>38.2</td><td>120.4</td><td>151.7</td><td>272.1</td><td>481.7</td><td>-29.3</td></tr><tr><td>Fixed budget</td><td>69.8</td><td>237.4</td><td>0.0</td><td>237.4</td><td>165.1</td><td></td></tr><tr><td rowspan="6">357</td><td>TokenCast</td><td>69.8</td><td>197.2</td><td>0.0</td><td>197.2</td><td>138.3</td><td>16.9</td></tr><tr><td>TRAIL</td><td>69.4</td><td>193.9</td><td>23.8</td><td>217.7</td><td>144.2</td><td>8.3</td></tr><tr><td>EGTP</td><td>39.9</td><td>76.0</td><td>9.5</td><td>85.5</td><td>58.9</td><td>64.0</td></tr><tr><td>TIE</td><td>63.9</td><td>141.2</td><td>17.1</td><td>158.3</td><td>98.8</td><td>33.3</td></tr><tr><td>Self-Pred.</td><td>52.1</td><td>154.8</td><td>190.5</td><td>345.3</td><td>593.9</td><td>-45.5</td></tr><tr><td>Fixed budget</td><td>79.9</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="6">440</td><td>TokenCast</td><td>79.9</td><td>258.2 226.5</td><td>0.0</td><td>258.2 226.5</td><td>177.2 156.4</td><td>12.3</td></tr><tr><td>TRAIL</td><td>79.2</td><td>222.3</td><td>0.0 26.9</td><td>249.2</td><td>163.1</td><td>3.5</td></tr><tr><td>EGTP</td><td>51.0</td><td>100.2</td><td>12.7</td><td>112.9</td><td>77.3</td><td>56.3</td></tr><tr><td>TIE</td><td>73.3</td><td>171.5</td><td>20.8</td><td>192.3</td><td>119.6</td><td>25.5</td></tr><tr><td>Self-Pred.</td><td>64.9</td><td>184.3</td><td>222.5</td><td>406.8</td><td>688.3</td><td>-57.6</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="6">604</td><td>Fixed budget</td><td>89.9</td><td>280.7</td><td>0.0</td><td>280.7</td><td>191.3</td><td></td></tr><tr><td>TokenCast</td><td>89.9</td><td>258.8</td><td>0.0</td><td>258.8</td><td>176.4</td><td>7.8</td></tr><tr><td>TRAIL</td><td>89.9</td><td>254.1</td><td>30.4</td><td>284.5</td><td>184.2</td><td>-1.4</td></tr><tr><td>EGTP</td><td>66.3</td><td>140.3</td><td>16.6</td><td>156.9</td><td>104.3</td><td>44.1</td></tr><tr><td>TIE</td><td>86.1</td><td>218.6</td><td>25.6</td><td>244.2</td><td>150.4</td><td>13.0</td></tr><tr><td>Self-Pred.</td><td>79.9</td><td>224.6</td><td>268.3</td><td>492.9</td><td>820.8</td><td>-75.6</td></tr></table>

Table 21: Budget-equal-weighted replay means. Trace completion is shown alongside overhead because lower processing totals can result from stopping more runs. Encoder-based baselines count locally processed input tokens; Self-Prediction counts additional LLM usage. These token counts therefore represent different resources.
<table><tr><td>Method</td><td>Trace-complete (%)</td><td>Prediction tokens</td><td>Total tokens</td><td>Total time</td></tr><tr><td>TokenCast</td><td>59.9</td><td>0.0k</td><td>173.0k</td><td>122.7 s</td></tr><tr><td>TRAIL</td><td>59.5</td><td>20.7k</td><td>186.9k</td><td>124.5 s</td></tr><tr><td>EGTP</td><td>34.8</td><td>8.5k</td><td>75.6k</td><td>52.5 s</td></tr><tr><td>TIE</td><td>54.4</td><td>15.2k</td><td>139.3k</td><td>88.1 s</td></tr><tr><td>Self-Prediction</td><td>44.1</td><td>166.0k</td><td>300.1k</td><td>522.4s</td></tr></table>

## E Case Studies

Token consumption reflects both the actions needed to finish a task and the context carried into each call. Two executions connect these quantities to observable events: verification after an edit and artifact preparation with a long context. Task Update estimates are displayed as final totals, $\widehat { T } _ { k } = S _ { k } + \widehat { R } _ { k } .$ , where $S _ { k }$ is confirmed consumption. Call-level estimates retain the full current-call target $C _ { k }$ . Dashed lines mark retrospective targets. Grouped bars show Call Start forecasts from TokenCast and Self-Prediction for full current-call consumption.

## E.1 Code Repair

GPT-5.4 repairs a swap\_dims() mutation issue in pydata\_\_xarray-6938 with DeepSeek Harness. The run contains 21 calls and 39 checkpoints, consumes 243,371 tokens, and lasts 130 s. The following timeline, detail plots, and table show how forecasts change during source editing, verification failure, retry, and completion.

Execution timeline. Figure 9 plots confirmed consumption over wall time. Task Start is at 0 s and the first Call Start 1 ms later, when the first request is assembled. The In-call Update shown is checkpoint 5 of call 12 at 65.3 s, with 657 bytes committed and 95,591 tokens confirmed. The Task Update after call 12 is at 68.7 s, with 109,011 tokens confirmed and 134,360 still to come.

![](images/b0e029e84c933de98a152c645421f4887a6a57b4cc76eb05bafdf32075e4cf06.jpg)  
Figure 9: Confirmed tokens over wall time for one run. The shaded span is call 12.

Generated prefix. Figure 10 and Table 22 report forecasts at selected prediction points. Call 12 performs the source edit and consumes 13,420 tokens. At 795 generated bytes, the prefix exposes the original source span. At 1,450 bytes, it reveals the replacement text. Across these checkpoints, the 90% interval contracts from 911 tokens at Call Start to 65 tokens at 1,450 bytes, while the forecast remains close to the recorded 13,420-token call cost.

Verification feedback. At call 14, reproduction encounters the removed NumPy unicode\_alias, and the predicted task total rises to 281,746 tokens. After a compatibility workaround allows the non-mutation assertions to pass at call 15, the forecast becomes 252,381 tokens, close to the recorded total of 243,371. The execution still consumes another 92,329 tokens while inspecting the diff, preparing artifacts, and submitting the response, accounting for 37.9% of the final total.

![](images/fc070db4a593cb285eb5d92be66c124c2845ab00446cfe60fd3e6fa80d264cb2.jpg)

![](images/47d22e9b758bd57c255b9bc02bdbeee8d9a8052561ed2e7cf29c55f38c63d5ae.jpg)  
Figure 10: Forecast updates during code repair. Left: task-level forecasts following verification outcomes. Right: current-call forecasts during edit call 12 as generation becomes visible.

Table 22: Forecasts at selected points in the code-repair execution. Token quantities are in thousands. Recorded targets are shown retrospectively.
<table><tr><td>Prediction point Available evidence</td><td></td><td>Target</td><td>Forecast [90% interval]</td><td>Recorded</td></tr><tr><td>Task Start</td><td>Issue and configuration</td><td>T</td><td>194.638 [104.782, 333.912]</td><td>243.371</td></tr><tr><td>Call Start, 12</td><td>Edit request assembled</td><td> $C _ { 1 2 }$ </td><td>13.368 [13.001, 13.912]</td><td>13.420</td></tr><tr><td>In-call, 12</td><td>795 bytes, old source span visible</td><td> $C _ { 1 2 }$ </td><td>13.383 [13.201, 13.694]</td><td>13.420</td></tr><tr><td>In-call, 12</td><td>1,450 bytes, replacement text visible</td><td>C12</td><td>13.431 [13.407, 13.472]</td><td>13.420</td></tr><tr><td></td><td>Update, k = 14 NumPy alias prevents reproduction</td><td>T</td><td>281.746 [191.304, 368.259]</td><td>243.371</td></tr><tr><td></td><td>Update, k = 15 Compatibility workaround, assertions pass</td><td>T</td><td>252.381 [211.623, 298.576]</td><td>243.371</td></tr></table>

## E.2 Long-Context QA

File operations. Qwen3.8-27B answers a LongBench question about OpenLRM and Instant3D using OpenHands. During call 1, it generates option C and its justification, then encounters a file-creation error because answer.md already exists. Inspecting and replacing the placeholder, preparing the patch, and checking the artifacts extend the execution to six calls. Successful replacement at call 3 is followed by another 395,353 tokens of consumption. Figure 11 and Table 23 show the corresponding task-level and call-level forecasts.

![](images/bfb175e4b5cbbb1328b16dee70cf3cf84b3fd9e417fb8fa08d4517b70834354c.jpg)

![](images/0990006cab3c0ab6264cb10f2ff003e3bbae50209324546cddd6a9a429254679.jpg)  
Figure 11: Repeated input makes additional calls expensive. Left: task updates after a failed write and artifact preparation. Right: full-call Call Start forecasts from TokenCast and Self-Prediction.

Repeated input. The first call processes 129,714 input tokens. Subsequent calls process 130,181–

Table 23: Long-context stages. All token quantities are in thousands. The same answer is carried through the subsequent file operations.
<table><tr><td>Point</td><td>Newly available evidence</td><td> $S _ { k }$ </td><td>Forecast T [90% interval]</td></tr><tr><td>Task Start</td><td>Long-context question and output requirements</td><td>0.000</td><td>526.384 [264.731, 919.648]</td></tr><tr><td></td><td>Update, k = 1 Answer generated, file creation fails</td><td></td><td>131.059 834.719 [525.193, 1204.762]</td></tr><tr><td></td><td>Update, k = 2 Existing answer placeholder inspected</td><td></td><td>261.368908.362 [643.881, 1268.997]</td></tr><tr><td></td><td>Update, k = 3 Answer file successfully replaced</td><td></td><td>392.112 762.541 [652.967, 1041.863]</td></tr><tr><td></td><td>Update, k = 5 Answer and patch read back</td><td></td><td>655.063 792.813 [765.208, 839.426]</td></tr><tr><td>Termination</td><td>Finish action recorded</td><td>787.465</td><td></td></tr></table>

132,209 input tokens and emit 128–350 output tokens each. Input contributes 785,146 of 787,465 tokens (99.7%). For the recorded execution, the segment decomposition gives

$$
T = n L _ { 1 } + b = 6 \times 1 2 9 , 7 1 4 + 9 , 1 8 1 = 7 8 7 , 4 6 5 .
$$

The repeated-input term contributes 778,284 tokens. Each additional call therefore carries roughly 130,000 input tokens, so the remaining call count and the input boundary jointly determine the task cost. The failed write at call 1 signals extra calls at this context length, and the Task Update forecast rises from 526k at Task Start to 835k tokens.

## F Self-Prediction Prompt

The following prompt template adapts the zero-shot agent self-prediction protocol of Bai et al. [2026] to the four prediction points used in our evaluation. It retains environment inspection, workload analysis, and phase-wise input/output token estimation. At Task Start, the estimator may inspect the initial environment before predicting complete-run consumption. At the other three prediction points, it receives the observed execution record and the target-specific evidence available at that point without advancing the task. Missing fields are marked as not available.

Self-Prediction prompt template   
Estimate the token consumption of the agent execution described below. Base your   
prediction on the task requirements, the specified model and agent configuration,   
and the evidence available at the supplied prediction point. Your estimate should   
describe the consumption of the agent operating under its existing workflow and   
execution limits. Your deliverable is a token-cost estimate. Do not implement a   
solution, modify task files, or submit an answer to the underlying task.   
Prediction target   
First, determine which part of the execution the estimate must cover. At Task Start,   
estimate all input and output tokens that a complete run would consume from the   
original task state until termination. The estimate includes the exploration,   
solution development, and verification that the run would require. Any inspection   
performed specifically to prepare this estimate belongs to the estimation session   
and is excluded from the predicted consumption. This inspection may improve your   
understanding of the task; however, the predicted run still begins from its   
original state.   
At Task Update, estimate the input and output tokens that will be consumed after the   
latest completed call until the task ends. Completed calls provide evidence of   
progress and consumption, but their recorded usage is outside of this remaining  
task target. Their messages and tool results may still appear in later requests;   
reading that retained content in a later request contributes new input usage within   
the prediction scope.   
At Call Start, estimate the complete input and output consumption of the current call,   
including retries of its request. At In-call Update, estimate the same complete  
call quantity using the generation observed so far. This includes the input,   
supplied output prefix, continuation, and any usage attributable to retries within

the call. A call comprises one initial model request and any retry requests issued before tool feedback is received. A subsequent request made after receiving tool feedback belongs to a later call and is outside the call-level estimate.

A task ends when the agent finishes, fails, or reaches an execution limit. Account for the termination behavior supported by the available evidence. Successful completion is one possible outcome, and repeated failures or exhausted limits may end the run earlier. Execution limits constrain the forecast, but they are not estimates of how much work the agent will perform.

Understanding the task and execution evidence

Read the task statement together with the agent instructions and completion conditions. Determine what the agent must establish, produce, or verify before it can complete the task. Consider how the available tools and workflows organize the work into model calls. Distinguish the number of tool actions from the number of model calls: several actions may be issued in one response, and a single unresolved issue may require multiple responses.

At Task Start, use the available inspection tools to understand the initial environment while leaving task files unchanged. For a coding task, inspect the relevant source files, their dependencies, and the tests associated with the requested behavior. Assess whether the work appears localized or spans several components, whether the expected behavior is clearly specified, and how much investigation is likely to precede a change. For retrieval or document-based tasks, examine the supplied materials and resource descriptions to assess what evidence is already available and what additional information the agent would need to obtain. Focus the inspection on uncertainties that materially affect the expected workload.

At the other prediction points, use the supplied execution record without advancing the task or obtaining additional tool results. Read the actions together with their observed outcomes. A proposed edit, a completed edit, and a successful test provide different evidence of progress. Identify what has been established, what remains unresolved, and which results are still pending. Treat unavailable information as unknown; the absence of a recorded failure does not establish success.

For a task-level forecast, form a plausible continuation from the current state. A coding run may still need to investigate a failure, revise an implementation, run relevant checks, and prepare its submission. A retrieval task may require further searches because the current evidence covers only part of the question. A reasoning task with sufficient information may complete in one response. Use phases that fit the actual task and the observed state. At Task Update, include only phases or portions of phases that remain.

Use the execution history to assess both progress and the cost of further work. Recent calls can indicate typical response lengths, request growth, and the amount of work the agent accomplishes per interaction. Compare calls with similar roles where possible. Repeated searches, recurring test failures, or several calls without resolving an outstanding issue may indicate additional investigation or revision. A sequence of completed checks may indicate that only final verification or submission remains. Relate these observations to the work still required before estimating the remaining call count.

For a call-level forecast, focus on what the current request asks the model to generate . Determine whether the response is likely to contain a short tool invocation, several tool-call arguments, a substantial code fragment, an explanation, or a final answer. At the In-call Update, examine how much of that response has already been produced and what remains unfinished. Use the supplied timing information as supporting evidence when it is informative, while keeping elapsed time distinct from token counts.

## Constructing the token estimate

For task-level predictions, the forecast execution is divided into a small number of non-overlapping phases. For each phase, the likely number of model calls, the input those calls will receive, and the output they will generate should be assessed. Let the phase estimates reflect the actual continuation you expect. Final verification or submission may require only one additional call. For the call-level prediction, a single current\_call phase is sufficient.

Estimate the input consumption from the content submitted for each model request. This can include system and agent instructions, task statements, tool definitions, retained conversation history, content generated from earlier calls, and tool results. Use an exact supplied input token count when one is available. When later requests have not yet been assembled, estimate their size from the current context, expected additions, and specified context retention behavior.

Account for retained content each time it is submitted. A tool result added early in a   
run may appear in several subsequent requests; therefore, its contribution depends   
on both its size and the number of later calls that retain it. Likewise, additional   
investigation can increase the call count and enlarge the context of those calls.   
Reflect both effects in the phase estimates. Apply context pruning, summarization,   
or compaction only when supported by the supplied workflow or observations.   
Estimate output consumption from the responses expected within each phase. Include   
generated text, code, tool-call arguments, and reasoning tokens according to the   
supplied accounting convention. The final answer may be short, even when   
intermediate responses consume substantial tokens. When reasoning tokens are   
already included in the reported output total, count them only once. Tool-generated   
search results, file contents, and test logs contribute model input when submitted   
to a later request; their production by a tool is not itself an LLM output.   
Use confirmed usage as an accounting anchor. Within the selected scope, include each   
confirmed input or output count exactly once. At In-call Update, the visible prefix   
is already part of the complete output being predicted; add only the expected   
continuation to its counted contribution. Do not add the prefix again when a   
supplied cumulative output count already includes it. Character and byte counts are   
observations about text length, not exact token counts; use the supplied token   
measurements or accounting information where available.   
Account for retries within the call according to the request history, observed failures   
, and the configured retry policy. The recorded usage from retries that have   
already occurred within the prediction scope is included. Estimate further retry   
consumption only to the extent supported by the current conditions. The maximum   
retry allowance defines a limit and does not imply that every retry will occur.   
Combine these estimates into the best-supported forecast. Consider whether the evidence   
supports a straightforward continuation, additional revision cycles, or early   
termination, and let this assessment inform the predicted workload. Avoid adding   
unexplained safety margins to each phase.   
Preparing the final output   
Return a non-negative integer for each token estimate. The phase input estimates must   
sum to predicted\_input\_tokens, and the phase output estimates must sum to   
predicted\_output\_tokens. Their combined sum must equal the predicted\_total\_tokens.   
Check that the result covers the requested scope and includes its confirmed usage   
without duplication. A complete call estimate must not fall below the confirmed   
consumption of that call. A Task Update estimate excludes completed call usage and   
is zero when termination is confirmed and no further model calls remain.   
Set predicted\_total\_tokens to the median (50th percentile) of token consumption for the   
specified prediction target. Set lower\_total\_tokens and upper\_total\_tokens to its   
5th and 95th percentiles, respectively, so that the interval targets 90% coverage.   
These quantiles describe uncertainty in token consumption within the same   
prediction scope. Return non-negative integers satisfying lower\_total\_tokens <=   
predicted\_total\_tokens <= upper\_total\_tokens. When confirmed consumption is   
supplied for that scope, all three estimates must be at least that amount. At Task   
Update, the scope includes only future consumption; set all three estimates to zero   
when termination is confirmed and no further model calls remain. Use one   
breakdown\_by\_phase entry for each phase included in the forecast, with names that   
describe the expected work. For either call-level prediction point, use   
current\_call when no further decomposition is needed.   
Submit a JSON object with the following fields:   
"predicted\_input\_tokens": <integer>,   
"predicted\_output\_tokens": <integer>,   
"predicted\_total\_tokens": <integer>,   
"lower\_total\_tokens": <integer>,   
"upper\_total\_tokens": <integer>,   
"breakdown\_by\_phase": [   
"phase": "<phase name>",   
"input\_tokens": <integer>,   
"output\_tokens": <integer>   
Use the completion interface specified below. Submit the JSON estimate without a task   
solution or additional explanatory text. Fields marked not available provide no   
additional evidence and must not be treated as zero-valued observations.

Prediction point:   
{{prediction\_point}}   
Task:   
{{task\_description}}   
Difficulty, when provided:   
{{task\_difficulty}}   
Model and agent configuration:   
{{model\_and\_agent\_configuration}}   
Agent workflow and available tools:   
{{agent\_instructions\_and\_tool\_descriptions}}   
Execution limits and token accounting:   
{{execution\_limits\_and\_token\_accounting}}   
Initial environment:   
{{initial\_environment}}   
Observed execution history and tool feedback:   
{{execution\_history\_and\_tool\_feedback}}   
Current request:   
{{current\_request}}   
Generated prefix and stream timing:   
{{generated\_prefix\_and\_stream\_timing}}   
Confirmed usage within the prediction scope:   
{{confirmed\_usage\_in\_scope}}   
Completion instruction:   
{{completion\_instruction}}   
Estimate the consumption for the specified prediction point and submit the JSON   
estimate.  
The prediction\_point field selects one of the four scopes defined above. When the harness exposes a finish tool, completion\_instruction requests submission of the JSON estimate through that tool. Otherwise, it requests the same object as the final response. Estimation calls and inspections performed solely for estimation are recorded as prediction overhead. The input and output fields refer to the selected scope, and therefore, predicted\_total\_tokens represents complete-task consumption at Task Start, complete-call consumption at both call-level points, and remaining-task consumption at Task Update.