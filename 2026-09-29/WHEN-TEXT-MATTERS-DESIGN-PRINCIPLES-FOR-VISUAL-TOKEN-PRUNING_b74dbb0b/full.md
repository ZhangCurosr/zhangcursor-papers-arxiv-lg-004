# WHEN TEXT MATTERS: DESIGN PRINCIPLES FOR VISUAL TOKEN PRUNING IN VISION-LANGUAGE MODELS

Minchan Kang<sup>1</sup>, Kyeonghye Park<sup>1</sup>, Seoyoung Cho<sup>1</sup>, Daeshik Kim<sup>1∗</sup>, Yucheol Cho<sup>2∗</sup> <sup>1</sup>Korea Advanced Institute of Science and Technology (KAIST), <sup>2</sup>Hanbat National University {mc.kang,pkhpjhs,52tjdud,daeshik}@kaist.ac.kr; yccho@hanbat.ac.kr

## ABSTRACT

Visual token pruning has been widely studied as a practical approach to reducing the computational cost of large vision-language models. However, it struggles to preserve essential visual information, which can lead to substantial performance degradation. In particular, image-based token selection can overlook task-relevant details, while text-guided token selection may fail to capture the text–visual relationships needed for complex reasoning. We find that applying textual guidance too early can limit its ability to identify answer-relevant visual regions, whereas text-to-visual attention becomes more informative at intermediate decoder depths. This finding motivates our training-free method, which separates early visionguided pruning from deferred text-guided reselection. We first prune visual tokens using vision-encoder attention, retain additional candidates until the decoder midpoint, and then use text-to-visual attention to determine the final visual-token set. Across eight benchmarks and three models, our method outperforms the bestperforming baselines by an average of 11.10 and 16.84 percentage points in performance recovery at 80% and 90% pruning, respectively, with comparable or lower LLM-prefill latency than most baselines. The source code is publicly available at https://github.com/kmc3661/DeFT.

## 1 INTRODUCTION

Large vision-language models (VLMs) support diverse tasks, including visual question answering, image captioning, and multimodal reasoning (Liu et al., 2023; Dai et al., 2023). Their capabilities, however, come with substantial computational costs: representing images with numerous visual tokens increases the sequence length processed by the language model, particularly for high-resolution inputs (Chen et al., 2024a; Wang et al., 2024a). Visual token pruning addresses this overhead by shortening the visual sequence. Training-free approaches are especially practical, as they reduce inference costs without updating model parameters or training an additional selector (Chen et al., 2024a; Zhang et al., 2024).

The central challenge is deciding which visual information to preserve. Image-based token selection uses visual importance, redundancy, or sensitivity to identify tokens without conditioning on the prompt (Yang et al., 2025a; Kim et al., 2026). Text-guided token selection instead uses interactions between textual and visual tokens to identify information relevant to the requested task (Xing et al., 2024; Zhang et al., 2024). Despite these differences, both approaches can discard essential visual information, leading to substantial performance degradation.

Figure 1 illustrates representative failure cases of image-based and text-guided visual token pruning. For image-based pruning, ZOO-Prune (Kim et al., 2026) retains prominent visual features of the calculator but discards the region containing the brand name requested by the question. For text-guided pruning, SparseVLM (Zhang et al., 2024) retains support for the phrase “I was the victim” but loses the corresponding value, illustrating a difficulty in capturing the relationships between textual and visual information needed to answer the question. Full-benchmark evaluations further reveal substantial performance losses. At 80% pruning on Qwen3-VL-8B, ZOO-Prune loses 28.8% and 34.7% relative to Dense on TextVQA and TextCaps, respectively, which rely on scene-text understanding (Singh et al., 2019; Sidorov et al., 2020). SparseVLM loses more than 50% on ChartQA and InfoVQA, which require complex reasoning over relationships among textual and visual elements (Masry et al., 2022; Mathew et al., 2022).

![](images/c9dac499cb805166c5bee284b0619b2d7a9540be5f0a0fe55603864457678c37.jpg)  
Figure 1: Failure cases on Qwen3-VL-8B at 80% pruning. Image-based visual token pruning discards the visual region containing the requested text, while text-guided visual token pruning preserves related support but loses the corresponding value. Gray: discarded regions; orange: merged support. Percentages denote full-benchmark performance losses relative to Dense.

These failures highlight the difficulty of identifying the visual information a VLM needs to reason toward a correct answer. This raises a key question: how can textual guidance be used more effectively to identify visual evidence that is relevant to the prompt? To investigate this, we analyze the ability of text-to-visual attention to distinguish answer-relevant regions across decoder depths. We find that this ability becomes substantially stronger at intermediate layers than near the input (Section 3.1), indicating that the timing of textual guidance is critical to effective token selection.

Building on this finding, we introduce a simple training-free method that combines early imagebased pruning with deferred text-guided selection. Specifically, we first prune visual tokens before the LLM using vision-encoder attention, while retaining additional candidates beyond the final token budget. At the decoder midpoint, text-to-visual attention reselects from these candidates to form the final token set. This design deliberately defers textual guidance to the midpoint until it becomes more informative, while relying solely on image information for early pruning. Across eight benchmarks and three VLMs, our method consistently achieves the highest average Dense-relative performance retention across all evaluated pruning ratios, with the margin over the strongest baseline increasing to 11.10 and 16.84 percentage points at 80% and 90% pruning, respectively.

## 2 RELATED WORK

## 2.1 LARGE VISION-LANGUAGE MODELS

VLMs extend large language models (Brown et al., 2020; Grattafiori et al., 2024) with visual encoders and multimodal alignment (Li et al., 2023). CogVLM introduces visual expert modules for deeper vision–language fusion, while Idefics2 studies architecture and training choices for effective multimodal learning (Wang et al., 2024b; Laurenc¸on et al., 2024). Models such as LLaVA-1.5, LLaVA-OneVision, and InternVL advance visual understanding through improved architectures, training, and visual representations (Liu et al., 2024; Li et al., 2024; Chen et al., 2024c), while highresolution processing increases visual-token costs (Guo et al., 2024). We evaluate our token pruning method on Qwen3-VL and LLaVA-OneVision-1.5 (Bai et al., 2025; An et al., 2025).

## 2.2 VISUAL TOKEN PRUNING

Reducing the computational cost of VLMs has been explored through knowledge distillation (Hinton et al., 2015; Cao et al., 2025), quantization (Li et al., 2025; Xiang et al., 2026), and visual token reduction. Token reduction encompasses merging in vision transformers (Bolya et al., 2022; Norouzi et al., 2024) and training-based pruning and merging for VLMs (Cao et al., 2023; 2024). Trainingfree approaches use pretrained signals to select tokens, drawing on visual importance, diversity, and sensitivity (Zhang et al., 2025), text–visual attention and instruction-conditioned relevance (Zhang et al., 2026; Liang et al., 2026), or combinations of visual saliency, textual relevance, and diversity (Liu et al., 2026). Complementary strategies preserve information through merging or recycling (Yang et al., 2025b) and correct token-reduction distortions (Cho et al., 2026). Building on these studies, we examine when textual guidance becomes informative for visual token selection, motivating a separation between early visual pruning and later text-guided reselection.

![](images/0d0ed0c66ecdf1851814e92d4b6a65e948aa801eef74209c418e568a706d826f.jpg)  
Figure 2: Text-guided evidence identification across decoder depth. Top: correlation with maskingbased region importance. Bottom: matching-versus-swapped question gain. Results average four QA benchmarks equally, each with 300 images and two questions per image.

Table 1: Selection-depth ablation at 80% pruning under matched decoder visual-token processing budgets. Observed peak denotes block 22, where the mean question gain in Figure 2 is highest. Values are eight-task Dense-relative recovery (%), with the best results in bold.
<table><tr><td>Boundary</td><td>Qwen3-VL-4B</td><td>Qwen3-VL-8B</td><td>LLaVA-OV-1.5-8B</td><td>Mean</td></tr><tr><td>D/4 (9 blocks)</td><td>82.119</td><td>86.551</td><td>85.898</td><td>84.856</td></tr><tr><td>2D/4 (18 blocks)</td><td>90.860</td><td>94.787</td><td>92.769</td><td>92.806</td></tr><tr><td>Observed peak (22 blocks)</td><td>89.185</td><td>94.014</td><td>92.299</td><td>91.833</td></tr><tr><td>3D/4 (27 blocks)</td><td>87.508</td><td>93.431</td><td>91.162</td><td>90.701</td></tr></table>

## 3 METHOD

## 3.1 MOTIVATION AND DESIGN RATIONALE

To determine when text-to-visual attention provides a reliable signal for token selection, we evaluate its ability to identify answer-relevant visual regions across decoder depth on four multimodal QA benchmarks (Singh et al., 2019; Masry et al., 2022; Mathew et al., 2022; Kembhavi et al., 2016). For each benchmark, we sample 300 images with two distinct questions per image and divide each image’s native visual-token grid into eight spatial regions. Using the unpruned model (Dense), we mask one region at a time throughout all decoder blocks and measure its effect on the correct answer. Let $\mathcal { L } _ { \mathrm { N L L } } ( \mathcal { M } )$ denote the token-averaged negative log-likelihood of the correct answer when regions M are masked. We define the importance of region $\bar { \mathcal { R } } _ { r }$ as

$$
h _ { r } = \mathcal { L } _ { \mathrm { N L L } } ( \mathcal { R } _ { r } ) - \mathcal { L } _ { \mathrm { N L L } } ( \emptyset ) ,\tag{1}
$$

where ∅ denotes the unmasked condition. A larger $h _ { r }$ means that masking region $\textstyle { \mathcal { R } } _ { r }$ has a larger negative effect on answering the question. We next ask whether text-to-visual attention identifies these important regions and whether this ability is specifically conditioned on the question. For this analysis, we use two different questions q and $q ^ { \prime }$ about each image. For both questions, we define

$$
\begin{array} { r } { h ^ { u } = [ h _ { 1 } ^ { u } , \ldots , h _ { 8 } ^ { u } ] , \qquad s _ { \ell } ^ { u } = [ s _ { \ell , 1 } ^ { u } , \ldots , s _ { \ell , 8 } ^ { u } ] , \qquad u \in \{ q , q ^ { \prime } \} , } \end{array}\tag{2}
$$

where $h ^ { u }$ contains the masking-based importance of the eight regions with respect to the correct answer for question $u ,$ and $s _ { \ell } ^ { u }$ contains their region-level text-to-visual attention at decoder block $\ell .$

![](images/9244e691e70646b608cad3ca1cefdddc29beb544482192c39f572bb1477dc4ee.jpg)  
Figure 3: Overview of our two-stage pruning. Before the LLM, vision-encoder attention guides visual-token pruning while preserving an expanded candidate set. At the decoder midpoint, text-tovisual attention selects the final visual-token set. The reserve fraction $\alpha \in [ 0 . 1 , 0 . 3 ]$ controls the number of additional candidates; we use $\alpha = 0 . 2$ by default.

We first test whether each question’s attention ranks the important regions for answering that question. Let $\rho ( \cdot , \cdot )$ denote Spearman rank correlation. Averaging over the two questions, we define

$$
\rho _ { \ell } ^ { \mathrm { m a t c h } } = \frac { 1 } { 2 } \left[ \rho ( s _ { \ell } ^ { q } , h ^ { q } ) + \rho ( s _ { \ell } ^ { q ^ { \prime } } , h ^ { q ^ { \prime } } ) \right] .\tag{3}
$$

A larger $\rho _ { \ell } ^ { \mathrm { m a t c h } }$ indicates better agreement between text-to-visual attention and the corresponding region importance.

We next test whether this alignment is specific to the question by comparing each question’s region importance with attention from the other question while keeping the image fixed:

$$
\rho _ { \ell } ^ { \mathrm { s w a p } } = \frac { 1 } { 2 } \left[ \rho ( s _ { \ell } ^ { q ^ { \prime } } , h ^ { q } ) + \rho ( s _ { \ell } ^ { q } , h ^ { q ^ { \prime } } ) \right] , \qquad \Delta \rho _ { \ell } = \rho _ { \ell } ^ { \mathrm { m a t c h } } - \rho _ { \ell } ^ { \mathrm { s w a p } } .\tag{4}
$$

Here, $\rho _ { \ell } ^ { \mathrm { s w a p } }$ measures how well attention from the other question ranks the important regions for the current question. A positive $\Delta \rho _ { \ell }$ therefore means that the matching question identifies its own important regions better than the other question. Figure 2 reports $\rho _ { \ell } ^ { \mathrm { m a t c h } }$ in the top row and $\Delta \rho _ { \ell }$ in the bottom row. Both are weak in the early decoder layers but become substantially stronger at intermediate depths. This shows that text-to-visual attention provides limited guidance near the decoder input, while becoming much more effective at identifying question-relevant visual evidence after sufficient decoder processing.

These results motivate deferring text-guided selection beyond the early decoder layers. The reselection depth must also account for efficiency: a deeper boundary processes retained visual tokens through more decoder blocks, so fewer can be retained under the same visual-token processing budget. We therefore ablate the reselection depth in our method under matched budgets. Table 1 shows that the decoder midpoint achieves the highest recovery across all three models. Together, the depth-wise analysis and budget-controlled ablation motivate the decoder midpoint as the reselection boundary, where textual guidance is informative while sufficient candidates can be retained at practical cost. We therefore use $L = D / 2$ for text-guided reselection in our method.

## 3.2 PROPOSED METHOD

Early visual pruning with reserve. As illustrated in Figure 3, our method first performs visionguided pruning before the LLM while preserving additional visual candidates for later reselection.

Prior training-free pruning methods have shown that attention within the vision encoder provides an effective signal for identifying visually informative tokens (Yang et al., 2025a; Zhang et al., 2025; Kim et al., 2026). We therefore use this signal for pre-LLM pruning, avoiding decoder computation. Unlike methods that incorporate text–visual interactions into early token selection (Xing et al., 2024; Zhang et al., 2024), we keep this stage purely vision-based, following our observation in Section 3.1 that textual guidance is less informative in early decoder layers.

Let N be the number of prunable visual tokens, D the decoder depth, and $K \approx ( 1 - p ) N$ the final token budget at pruning ratio $p .$ We first obtain the head-averaged vision-encoder attention

$$
\mathbf { A } ^ { \mathrm { i m g } } = \frac { 1 } { H } \sum _ { h = 1 } ^ { H } \mathrm { s o f t m a x } \left( \frac { \mathbf { Q } _ { h } \mathbf { K } _ { h } ^ { \top } } { \sqrt { d _ { h } } } \right) , \qquad \mathbf { \Phi } _ { i } ^ { \mathrm { i m g } } = \left\{ \begin{array} { l l } { A _ { \mathrm { c l a } , i } ^ { \mathrm { i m g } } , } & { \mathrm { w i t h ~ a ~ c l a s s ~ t o k e n } , } \\ { \sum _ { j \in \mathcal { V } } A _ { j , i } ^ { \mathrm { i m g } } , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{5}
$$

where V denotes the visual-token positions. Thus, class-token-based encoders use CLS-to-visual attention, whereas encoders without a class token use the self-attention received by each visual token from other visual queries.

Rather than immediately pruning to K tokens, we preserve additional image-ranked candidates:

$$
M = \mathrm { m i n } \{ N , K + \lceil \alpha ( N - K ) \rceil \} , \qquad \mathcal { C } = \mathrm { T o p K } _ { M } ( s ^ { \mathrm { i m g } } ) ,\tag{6}
$$

where $\alpha \in [ 0 . 1 , 0 . 3 ]$ is a scalar hyperparameter that controls the fraction of otherwise pruned tokens preserved as additional candidates; we use $\alpha = 0 . 2$ by default. The set C contains the image-only top-K together with the next $M - K$ candidates. We process these candidates until the decoder midpoint $L = D / 2 $ , preserving alternatives that may become important once textual guidance becomes sufficiently informative.

Deferred text-guided reselection. At the decoder midpoint, we revise the image-only selection with text-to-visual attention. Let T and C denote the input text tokens and visual candidates, respectively. Using the decoder native queries and keys, we compute the head-averaged attention as

$$
\mathbf { A } ^ { \mathrm { t e x t } } = \frac { 1 } { H } \sum _ { h = 1 } ^ { H } \mathrm { s o f t m a x } \left( \frac { \mathbf { Q } _ { h } ^ { \mathcal { T } } \mathbf { K } _ { h } ^ { \top } } { \sqrt { d _ { h } } } \right) , \qquad s _ { i } ^ { \mathrm { t e x t } } = \sum _ { t \in \mathcal { T } } A _ { t , i } ^ { \mathrm { t e x t } } ,\tag{7}
$$

where the softmax follows the decoder’s native attention mask and $s _ { i } ^ { \mathrm { t e x t } }$ is the attention received by candidate i from the input text tokens. We select the final visual-token set as

$$
\mathcal { S } = \mathrm { T o p K } _ { K } \left( s ^ { \mathrm { t e x t } } \vert _ { \mathcal { C } } \right) .\tag{8}
$$

Overall, image scores are used only to construct the candidate set C, while the final selection at the decoder midpoint depends solely on text-to-visual attention. This allows candidates outside the initial vision-guided top-K to enter the final set when they become more relevant to the input. By separating early vision-guided pruning from deferred text-guided reselection, the method reduces visual-token processing early without committing to the final token set before textual guidance becomes sufficiently informative.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

Models and baselines. We evaluate Qwen3-VL-4B, Qwen3-VL-8B (Bai et al., 2025), and LLaVA-OneVision-1.5-8B (An et al., 2025). We compare with five training-free visual-token reduction methods: FastV (Chen et al., 2024a), SparseVLM (Zhang et al., 2024), VisPruner (Zhang et al., 2025), ZOO-Prune (Kim et al., 2026), and RESTORE (Cho et al., 2026). Since SparseVLM progressively changes the visual-token count across decoder depth, we match its average decoder token usage to ours for a comparable processing budget.

Benchmarks. We evaluate eight full benchmarks: TextVQA (Singh et al., 2019) for scene-text understanding; ChartQA (Masry et al., 2022) and InfoVQA (Mathew et al., 2022) for chart and infographic reasoning; AI2D (Kembhavi et al., 2016), MMMU (Yue et al., 2024), and MMStar (Chen et al., 2024b) for diagram and general multimodal reasoning; and NoCaps (Agrawal et al., 2019) and TextCaps (Sidorov et al., 2020) for open-ended captioning. Together, they test preservation of both global context and localized key evidence.

Table 2: Qwen3-VL-4B results across eight benchmarks. Avg. Rel.: mean performance relative to Dense (%); higher is better. Bold/underline: best/second-best pruned results at each ratio.
<table><tr><td>Method</td><td>TextVQA</td><td>MMMU</td><td>AI2D</td><td>MMStar</td><td>ChartQA</td><td>NoCaps</td><td>TextCaps</td><td>InfoVQA</td><td>Avg. Rel.</td></tr><tr><td>Dense</td><td>81.30</td><td>46.22</td><td>81.87</td><td>56.40</td><td>82.92</td><td>66.51</td><td>81.86</td><td>77.43</td><td>100.00</td></tr><tr><td colspan="10">Prune 70% of vision tokens</td></tr><tr><td>FastV (ECCV2024)</td><td>74.89</td><td>43.33</td><td>72.22</td><td>46.53</td><td>45.64</td><td>63.16</td><td>84.84</td><td>44.51</td><td>83.46</td></tr><tr><td>SparseVLM (ICML2025)</td><td>73.21</td><td>45.44</td><td>73.87</td><td>48.80</td><td>50.72</td><td>65.68</td><td>74.94</td><td>45.78</td><td>84.46</td></tr><tr><td>VisPruner (ICCV2025)</td><td>64.76</td><td>44.33</td><td>72.31</td><td>47.07</td><td>65.00</td><td>66.45</td><td>68.25</td><td>46.48</td><td>83.63</td></tr><tr><td>ZOO-Prune (CVPR2026)</td><td>65.64</td><td>45.11</td><td>76.75</td><td>51.20</td><td>72.84</td><td>66.12</td><td>58.97</td><td>53.92</td><td>86.48</td></tr><tr><td>RESTORE (ICML2026) Ours</td><td>66.17</td><td>43.89</td><td>72.83</td><td>50.67</td><td>55.44</td><td>67.03</td><td>73.34</td><td>37.77</td><td>82.64</td></tr><tr><td></td><td>77.57</td><td>44.78</td><td>78.79</td><td>52.67</td><td>75.92</td><td>66.59</td><td>81.56</td><td>61.32</td><td>94.05</td></tr><tr><td colspan="10">Prune 80% of vision tokens</td></tr><tr><td>FastV (ECCV2024)</td><td>68.57</td><td>42.44</td><td>66.84</td><td>42.00</td><td>28.72</td><td>58.98</td><td>83.07</td><td>33.93</td><td>75.11</td></tr><tr><td>SparseVLM (ICML2025)</td><td>66.59</td><td>44.11</td><td>71.60</td><td>46.93</td><td>34.44</td><td>63.75</td><td>67.98</td><td>38.35</td><td>77.24</td></tr><tr><td>VisPruner (ICCV2025)</td><td>48.16</td><td>42.56</td><td>67.58</td><td>44.93</td><td>49.08</td><td>65.61</td><td>50.76</td><td>36.53</td><td>72.57</td></tr><tr><td>ZOO-Prune (CVPR2026)</td><td>56.14</td><td>44.00</td><td>72.44</td><td>47.73</td><td>63.16</td><td>65.91</td><td>52.04</td><td>43.66</td><td>79.07</td></tr><tr><td>RESTORE (ICML2026)</td><td>55.64</td><td>43.67</td><td>69.11</td><td>48.07</td><td>42.12</td><td>66.58</td><td>63.86</td><td>30.62</td><td>75.13</td></tr><tr><td>Ours</td><td>74.91</td><td>45.11</td><td>76.78</td><td>50.53</td><td>71.20</td><td>66.57</td><td>80.57</td><td>53.73</td><td>90.86</td></tr><tr><td colspan="10">Prune 90% of vision tokens</td></tr><tr><td>FastV (ECCV2024)</td><td>46.46</td><td>39.22</td><td>63.08</td><td>33.00</td><td>15.60</td><td>47.82</td><td>70.71</td><td>25.46</td><td>60.94</td></tr><tr><td>SparseVLM (ICML2025)</td><td>48.95</td><td>43.78</td><td>67.52</td><td>42.07</td><td>19.36</td><td>57.96</td><td>53.31</td><td>28.15</td><td>65.49</td></tr><tr><td>VisPruner (ICCV2025)</td><td>30.69</td><td>41.78</td><td>66.48</td><td>38.27</td><td>32.52</td><td>62.27</td><td>35.74</td><td>27.14</td><td>61.09</td></tr><tr><td>ZOO-Prune (CVPR2026)</td><td>40.33</td><td>42.67</td><td>67.36</td><td>43.67</td><td>44.32</td><td>63.84</td><td>41.14</td><td>30.78</td><td>67.63</td></tr><tr><td>RESTORE (ICML2026)</td><td>37.81</td><td>43.78</td><td>65.77</td><td>43.80</td><td>28.12</td><td>65.08</td><td>45.93</td><td>25.71</td><td>65.04</td></tr><tr><td>Ours</td><td>69.27</td><td>44.11</td><td>72.96</td><td>47.67</td><td>55.68</td><td>63.99</td><td>73.21</td><td>43.54</td><td>82.91</td></tr></table>

Table 3: Qwen3-VL-8B results across eight benchmarks. Avg. Rel.: mean performance relative to Dense (%); higher is better. Bold/underline: best/second-best pruned results at each ratio.
<table><tr><td>Method</td><td>TextVQA</td><td>MMMU</td><td>AI2D</td><td>MMStar</td><td>ChartQA</td><td>NoCaps</td><td>TextCaps</td><td>InfoVQA</td><td>Avg. Rel.</td></tr><tr><td>Dense</td><td>82.95</td><td>51.67</td><td>83.68</td><td>62.73</td><td>83.48</td><td>63.54</td><td>81.62</td><td>81.19</td><td>100.00</td></tr><tr><td colspan="10">Prune 70% of vision tokens</td></tr><tr><td>FastV (ECCV2024)</td><td>74.54</td><td>49.44</td><td>72.73</td><td>50.27</td><td>43.60</td><td>63.11</td><td>84.87</td><td>42.53</td><td>82.56</td></tr><tr><td>SparseVLM (ICML2025)</td><td>76.12</td><td>51.33</td><td>76.68</td><td>54.93</td><td>47.52</td><td>63.54</td><td>78.13</td><td>48.95</td><td>85.41</td></tr><tr><td>VisPruner (ICCV2025)</td><td>75.85</td><td>51.56</td><td>78.47</td><td>52.87</td><td>66.68</td><td>63.13</td><td>79.61</td><td>52.60</td><td>88.85</td></tr><tr><td>ZOO-Prune (CVPR2026)</td><td>69.31</td><td>51.00</td><td>78.14</td><td>56.27</td><td>70.80</td><td>63.39</td><td>63.42</td><td>54.70</td><td>86.87</td></tr><tr><td>RESTORE (ICML2026) Ours</td><td>73.33</td><td>51.00</td><td>78.01</td><td>56.00</td><td>66.24</td><td>64.27</td><td>78.52</td><td>51.00</td><td>88.64</td></tr><tr><td></td><td>81.53</td><td>51.44</td><td>82.74</td><td>59.20</td><td>78.24</td><td>63.77</td><td>84.15</td><td>70.16</td><td>96.84</td></tr><tr><td colspan="10">Prune 80% of vision tokens</td></tr><tr><td>FastV (ECCV2024)</td><td>67.10</td><td>47.11</td><td>69.43</td><td>47.27</td><td>29.12</td><td>61.51</td><td>83.83</td><td>33.96</td><td>75.83</td></tr><tr><td>SparseVLM (ICML2025)</td><td>70.36</td><td>50.22</td><td>74.51</td><td>52.40</td><td>33.96</td><td>62.99</td><td>72.45</td><td>40.36</td><td>79.11</td></tr><tr><td>VisPruner (ICCV2025)</td><td>67.88</td><td>50.22</td><td>74.48</td><td>49.73</td><td>54.40</td><td>62.03</td><td>73.52</td><td>42.00</td><td>81.49</td></tr><tr><td>ZOO-Prune (CVPR2026)</td><td>59.08</td><td>51.67</td><td>74.58</td><td>53.80</td><td>59.92</td><td>62.70</td><td>53.26</td><td>43.53</td><td>79.43</td></tr><tr><td>RESTORE (ICML2026)</td><td>64.80</td><td>50.22</td><td>74.42</td><td>51.87</td><td>53.28</td><td>64.18</td><td>71.00</td><td>40.35</td><td>81.05</td></tr><tr><td>Ours</td><td>80.48</td><td>51.33</td><td>81.96</td><td>57.47</td><td>75.20</td><td>63.49</td><td>83.95</td><td>64.57</td><td>94.79</td></tr><tr><td colspan="10">Prune 90% of vision tokens</td></tr><tr><td>FastV (ECCV2024)</td><td>49.64</td><td>45.89</td><td>66.52</td><td>38.40</td><td>19.40</td><td>55.95</td><td>76.82</td><td>27.99</td><td>66.16</td></tr><tr><td>SparseVLM (ICML2025)</td><td>54.24</td><td>50.00</td><td>70.43</td><td>46.40</td><td>20.40</td><td>58.93</td><td>58.80</td><td>29.73</td><td>68.27</td></tr><tr><td>VisPruner (ICCV2025)</td><td>45.48</td><td>48.33</td><td>68.10</td><td>42.80</td><td>30.88</td><td>59.08</td><td>53.94</td><td>30.43</td><td>66.44</td></tr><tr><td>ZOO-Prune (CVPR2026)</td><td>42.04</td><td>49.67</td><td>68.01</td><td>46.33</td><td>41.00</td><td>60.71</td><td>38.17</td><td>31.75</td><td>66.56</td></tr><tr><td>RESTORE (ICML2026)</td><td>46.70</td><td>49.56</td><td>68.39</td><td>46.00</td><td>30.76</td><td>61.57</td><td>52.27</td><td>30.26</td><td>67.79</td></tr><tr><td>Ours</td><td>76.93</td><td>50.56</td><td>79.86</td><td>55.53</td><td>69.12</td><td>62.88</td><td>80.71</td><td>56.83</td><td>90.65</td></tr></table>

Implementation and metrics. We evaluate 70%, 80%, and 90% visual-token pruning with midpoint reselection. We report each benchmark’s native metric, while Avg. Rel. is the mean task score normalized by the corresponding Dense score; additional details are in Appendix A.

## 4.2 MAIN RESULTS

Tables 2–4 show that our method consistently achieves the highest average relative performance across all nine model–pruning settings. The advantage grows under more aggressive pruning: averaged over the three models, our method exceeds the strongest baseline by 11.10 points at 80% pruning and 16.84 points at 90%. Notably, at 90% pruning on Qwen3-VL-8B, our method retains 90.65% of Dense performance, while all competing methods remain below 70%.

Table 4: LLaVA-OneVision-1.5-8B results across eight benchmarks. Avg. Rel.: mean performance relative to Dense (%); higher is better. Bold/underline: best/second-best pruned results at each ratio.
<table><tr><td>Method</td><td>TextVQA</td><td>MMMU</td><td>AI2D</td><td>MMStar</td><td>ChartQA</td><td>NoCaps</td><td>TextCaps</td><td>InfoVQA</td><td>Avg. Rel.</td></tr><tr><td>Dense</td><td>79.71</td><td>55.78</td><td>84.62</td><td>67.33</td><td>86.64</td><td>110.40</td><td>123.13</td><td>77.41</td><td>100.00</td></tr><tr><td colspan="10">Prune 70% of vision tokens</td></tr><tr><td>FastV (ECCV2024)</td><td>66.67</td><td>54.33</td><td>75.84</td><td>51.80</td><td>48.20</td><td>103.94</td><td>114.19</td><td>43.81</td><td>80.84</td></tr><tr><td>SparseVLM (ICML2025)</td><td>74.68</td><td>55.00</td><td>78.21</td><td>57.33</td><td>62.76</td><td>108.44</td><td>113.21</td><td>48.83</td><td>86.94</td></tr><tr><td>VisPruner (ICCV2025)</td><td>50.50</td><td>52.89</td><td>74.29</td><td>54.20</td><td>53.40</td><td>109.30</td><td>75.89</td><td>37.70</td><td>74.68</td></tr><tr><td>ZOO-Prune (CVPR2026)</td><td>71.09</td><td>55.22</td><td>81.83</td><td>62.87</td><td>77.40</td><td>110.07</td><td>107.16</td><td>54.16</td><td>90.54</td></tr><tr><td>RESTORE (ICML2026) Ours</td><td>63.90</td><td>52.56</td><td>75.58</td><td>54.13</td><td>44.60</td><td>103.38</td><td>104.05</td><td>34.78</td><td>77.33</td></tr><tr><td></td><td>77.56</td><td>56.22</td><td>81.38</td><td>62.67</td><td>83.40</td><td>107.32</td><td>120.73</td><td>64.02</td><td>95.20</td></tr><tr><td colspan="10">Prune 80% of vision tokens</td></tr><tr><td>FastV (ECCV2024)</td><td>57.01</td><td>51.44</td><td>73.22</td><td>47.47</td><td>33.04</td><td>98.38</td><td>102.52</td><td>36.49</td><td>72.30</td></tr><tr><td>SparseVLM (ICML2025)</td><td>67.95</td><td>54.56</td><td>75.97</td><td>53.40</td><td>46.72</td><td>104.13</td><td>100.64</td><td>39.87</td><td>79.21</td></tr><tr><td>VisPruner (ICCV2025)</td><td>41.84</td><td>52.44</td><td>71.60</td><td>50.40</td><td>42.16</td><td>105.42</td><td>67.30</td><td>31.96</td><td>68.26</td></tr><tr><td>ZOO-Prune (CVPR2026)</td><td>65.04</td><td>53.33</td><td>79.95</td><td>60.20</td><td>69.68</td><td>108.73</td><td>97.84</td><td>44.10</td><td>84.56</td></tr><tr><td>RESTORE (ICML2026)</td><td>58.11</td><td>52.78</td><td>72.22</td><td>51.33</td><td>35.32</td><td>101.90</td><td>94.82</td><td>31.02</td><td>72.41</td></tr><tr><td>Ours</td><td>75.97</td><td>54.44</td><td>80.38</td><td>61.13</td><td>81.12</td><td>106.74</td><td>116.32</td><td>60.89</td><td>92.77</td></tr><tr><td colspan="10">Prune 90% of vision tokens</td></tr><tr><td>FastV (ECCV2024)</td><td>40.80</td><td>51.78</td><td>69.04</td><td>41.60</td><td>20.56</td><td>83.65</td><td>79.75</td><td>29.52</td><td>61.22</td></tr><tr><td>SparseVLM (ICML2025)</td><td>56.89</td><td>52.56</td><td>72.73</td><td>49.40</td><td>27.88</td><td>94.84</td><td>86.05</td><td>30.58</td><td>69.05</td></tr><tr><td>VisPruner (ICCV2025)</td><td>30.18</td><td>49.56</td><td>69.82</td><td>44.00</td><td>31.52</td><td>95.50</td><td>53.59</td><td>28.38</td><td>59.70</td></tr><tr><td>ZOO-Prune (CVPR2026)</td><td>52.86</td><td>50.44</td><td>75.23</td><td>52.93</td><td>52.48</td><td>104.93</td><td>80.48</td><td>33.71</td><td>73.60</td></tr><tr><td>RESTORE (ICML2026)</td><td>46.90</td><td>50.89</td><td>70.05</td><td>47.20</td><td>26.20</td><td>98.31</td><td>76.96</td><td>28.29</td><td>65.16</td></tr><tr><td>Ours</td><td>71.66</td><td>52.89</td><td>78.21</td><td>57.80</td><td>72.72</td><td>103.93</td><td>101.99</td><td>52.52</td><td>86.47</td></tr></table>

![](images/71aa8d10ef2a5829f48ebacac7ad205962360b6ea238bdd9b8b699c9f051c11b.jpg)  
Figure 4: TextVQA and ChartQA examples on Qwen3-VL-8B at 80% pruning. Purple: final tokens outside the image-only top-K; gray: discarded regions; orange: merged support.

The gains are especially pronounced on TextVQA, ChartQA, and InfoVQA, where performance depends on preserving localized, prompt-relevant evidence. Averaged across the three backbones, the margins over the strongest competing results increase from 3.66, 5.51, and 10.91 points at 70% pruning to 8.16, 11.59, and 15.97 points at 80%, and further to 19.26, 19.91, and 18.88 points at 90%, respectively. As pruning becomes more aggressive, early removal is more likely to discard the few tokens containing key evidence; retaining additional candidates until textual guidance becomes informative therefore provides greater benefit in these challenging settings.

![](images/2389eecf22311e7f0a4a1e2e26477808a648abcc6e0aabd7f5e0f2523018185c.jpg)

![](images/4c03474359aceddcce3b85698178bf7098d7b81e480a01e51ee2fd2212856eba.jpg)

![](images/52a34a6afb1e1ea80bf8e45a7c13004bf724c5c531a7f90fe2dbb1f5647d0842.jpg)  
Figure 5: Candidate preservation and text-guided reselection at 80% pruning. A: immediate imageonly top-K selection; B: preserve additional candidates until the midpoint, then retain the same initial top-K as A; C: use the same candidates and token-count trajectory as B, but reselect the final K tokens using text-to-visual attention (Ours). Values are Dense-relative recovery (%); the dashed line denotes Dense (100%).

![](images/cba6bb81a35e573523b8ec8f06586514e9ee0322a1384feed057ad16f5823288.jpg)  
(b) Mid-laver final selection

![](images/16ad49d5dda1635e0b04ffb817c2a12a1d27d5a661ce6d23df7e6c5a85931abd.jpg)  
Figure 6: Pruning-score ablations at 80% pruning. (a) Encoder-based candidate selection before the LLM versus after block 1, with or without text attention. (b) Midpoint scoring with fixed candidates and token budgets. Results average Dense-relative recovery across InfoVQA, ChartQA, TextCaps, and MMStar.

Figure 4 further illustrates how deferred selection preserves key evidence. In both examples, answerrelevant regions fall outside the initial image-only top-K but remain in the candidate reserve. Midpoint text-to-visual attention later promotes them into the final set, preserving localized evidence that immediate pruning would discard. This shows how the reserve allows textual guidance to revise the initial visual ranking when relevant evidence is not visually dominant.

## 4.3 ABLATION STUDIES

Figure 5 separates the contributions of candidate preservation and text-guided reselection. Immediate image-only pruning retains the image-only top-K from the beginning. Candidate preservation instead keeps the larger candidate pool until the midpoint but ultimately retains the same image-only top-K, isolating the benefit of delaying the final pruning decision. Text-guided reselection follows the same token-count trajectory as candidate preservation but reselects the final K tokens using midpoint text-to-visual attention. The substantial gain from candidate preservation to text-guided reselection shows that the improvement is not simply due to retaining more tokens early, but also to using midpoint text-to-visual attention to revise the final selection.

Figure 6 examines which information should guide token selection at each stage. In Figure 6(a), using text-to-visual attention in the first decoder block reduces recovery compared with vision-only pre-LLM pruning across all three models, showing that textual guidance can be detrimental when applied too early. In contrast, Figure 6(b) shows that, with the candidate pool, reselection depth, and final budget fixed, midpoint text-to-visual attention alone achieves the highest average recovery on all three models. These results support a clear separation of roles: vision-encoder attention for early pruning and text-to-visual attention for midpoint reselection.

![](images/58b83330d91130eedac323e6d92e76ba8cbf9dc927ff36a61037cd3417080721.jpg)

![](images/29b830798deb9582cdeadcde53634ae6c166f84f70868c3e95e7a73a39f57a11.jpg)

![](images/264f69055077bc7c3bd1dcc934cd01fda5c75e779fffd6d9ea483d26fd7f574e.jpg)  
Figure 7: Performance–latency trade-off as the reserve fraction α varies at 80% pruning. Recovery is averaged equally across eight benchmarks relative to Dense. Latency includes token-selection overhead and is averaged over three timed repetitions on one RTX A6000 GPU, using 100 fixed InfoVQA inputs stratified by image size and aspect ratio.

Table 5: Qwen3-VL-8B inference latency (ms). Measurements average three timed repetitions on one RTX A6000 GPU using 100 fixed InfoVQA inputs stratified by image size and aspect ratio. Prefill includes token-selection overhead. End-to-end spans GPU-ready inputs through the first output token, excluding CPU preprocessing and input transfer.
<table><tr><td rowspan="2">Method</td><td colspan="2">80% pruning</td><td colspan="2">90% pruning</td></tr><tr><td>LLM prefill</td><td>End-to-end</td><td>LLM prefill</td><td>End-to-end</td></tr><tr><td>Dense</td><td>226.66</td><td>364.13</td><td>226.66</td><td>364.13</td></tr><tr><td>FastV</td><td>116.46</td><td>256.15</td><td>106.97</td><td>247.22</td></tr><tr><td>SparseVLM</td><td>134.79</td><td>276.36</td><td>122.16</td><td>263.63</td></tr><tr><td>VisPruner</td><td>131.25</td><td>292.39</td><td>123.80</td><td>284.79</td></tr><tr><td>ZOO-Prune</td><td>137.28</td><td>401.81</td><td>111.69</td><td>375.04</td></tr><tr><td>RESTORE</td><td>128.87</td><td>288.37</td><td>116.46</td><td>276.09</td></tr><tr><td>Ours</td><td>121.20</td><td>282.35</td><td>108.37</td><td>268.50</td></tr></table>

## 4.4 ANALYSIS

The reserve fraction α controls the trade-off between candidates preserved for midpoint reselection and inference cost. Figure 7 shows that increasing α improves average recovery across all three models at the cost of higher LLM-prefill latency. Across $\alpha \in [ 0 . 1 , 0 . 3 ]$ , this trade-off is approximately linear, allowing α to be adjusted to the desired efficiency–accuracy balance. Even $\alpha = 0 . 1$ exceeds the strongest baseline average on every backbone, and we use $\alpha = 0 . 2$ by default.

Table 5 shows that our method maintains practical inference latency despite preserving additional visual candidates in the early decoder layers. Unlike many existing approaches that introduce additional token-merging or learned selection modules, our method simply uses vision-encoder attention for early pruning and text-to-visual attention for midpoint reselection. This targeted use of existing model signals yields substantial performance gains with only modest additional inference cost. Additional analyses in the Appendix quantify the use and performance contribution of reselected candidates and extend the comparison to additional progressive pruning methods.

## 5 CONCLUSION

We show that textual guidance is most effective for visual token pruning when applied at an appropriate decoder depth. Based on this observation, we introduce a simple training-free method that uses vision-encoder attention for early pruning and text-to-visual attention for midpoint reselection, achieving strong performance preservation across three VLMs and eight benchmarks, especially under aggressive pruning. Despite its simplicity, the method provides substantial gains with practical inference efficiency. We hope this approach can serve as a strong baseline for future work toward more effective and general visual token pruning in large vision-language models.

## REFERENCES

Harsh Agrawal, Karan Desai, Yufei Wang, Xinlei Chen, Rishabh Jain, Mark Johnson, Dhruv Batra, Devi Parikh, Stefan Lee, and Peter Anderson. Nocaps: Novel object captioning at scale. In 2019 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 8947–8956. IEEE, 2019.

Xiang An, Yin Xie, Kaicheng Yang, Wenkang Zhang, Xiuwei Zhao, Zheng Cheng, Yirui Wang, Songcen Xu, Changrui Chen, Didi Zhu, et al. Llava-onevision-1.5: Fully open framework for democratized multimodal training. arXiv preprint arXiv:2509.23661, 2025.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your vit but faster. arXiv preprint arXiv:2210.09461, 2022.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Jiajun Cao, Yuan Zhang, Tao Huang, Ming Lu, Qizhe Zhang, Ruichuan An, Ningning Ma, and Shanghang Zhang. Move-kd: Knowledge distillation for vlms with mixture of visual encoders. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 19846– 19856. IEEE, 2025.

Jianjian Cao, Peng Ye, Shengze Li, Chong Yu, Yansong Tang, Jiwen Lu, and Tao Chen. Madtp: Mul timodal alignment-guided dynamic token pruning for accelerating vision-language transformer. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 15710– 15719. IEEE, 2024.

Qingqing Cao, Bhargavi Paranjape, and Hannaneh Hajishirzi. Pumer: Pruning and merging tokens for efficient vision language models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 12890–12903, 2023.

Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large visionlanguage models. In European Conference on Computer Vision, pp. 19–35. Springer, 2024a.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? Advances in Neural Information Processing Systems, 37:27056–27087, 2024b.

Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 24185–24198, 2024c.

Hyeonwoo Cho, Donghyeon Baek, Yewon Kim, and Bumsub Ham. Improving visual token reduction via rectifying distortions for efficient multimodal llm inference. arXiv preprint arXiv:2606.01711, 2026.

Wenliang Dai, Junnan Li, Dongxu Li, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. Advances in neural information processing systems, 36:49250–49267, 2023.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Zonghao Guo, Ruyi Xu, Yuan Yao, Junbo Cui, Zanlin Ni, Chunjiang Ge, Tat-Seng Chua, Zhiyuan Liu, and Gao Huang. Llava-uhd: an lmm perceiving any aspect ratio and high-resolution images. In European Conference on Computer Vision, pp. 390–406. Springer, 2024.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.

Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In European conference on computer vision, pp. 235–251. Springer, 2016.

Youngeun Kim, Youjia Zhang, Huiling Liu, Aecheon Jung, Sunwoo Lee, and Sungeun Hong. Zooprune: training-free token pruning via zeroth-order gradient estimation in vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 39572–39582, 2026.

Hugo Laurenc¸on, Leo Tronchon, Matthieu Cord, and Victor Sanh. What matters when building´ vision-language models? Advances in Neural Information Processing Systems, 37:87874–87907, 2024.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pp. 19730–19742. PmLR, 2023.

Shiyao Li, Yingchun Hu, Xuefei Ning, Xihui Liu, Ke Hong, Xiaotao Jia, Xiuhong Li, Yaqi Yan, Pei Ran, Guohao Dai, et al. Mbq: Modality-balanced quantization for large vision-language models. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4167–4177. IEEE, 2025.

Yuxuan Liang, Xu Li, Xiaolei Chen, Haotian Chen, Yi Zhen, Zhe Liu, Rui Zhu, Bin Li, and Xiangyang Xue. Pyramid token pruning for high-resolution large vision-language models via region, token, and instruction-guided importance. IEEE Transactions on Circuits and Systems for Video Technology, 2026.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 26286–26296. IEEE, 2024.

Ziniu Liu, Shuheng Zhou, Mingqing Liu, Hao Deng, and Huijia Zhu. Crisprune: Combining contextual relevance and intrinsic saliency for efficient visual token pruning in mllms. In Findings of the Association for Computational Linguistics: ACL 2026, pp. 13546–13564, 2026.

Ahmed Masry, Jia Qing Tan, Shafiq Joty, Enamul Hoque, et al. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the association for computational linguistics: ACL 2022, pp. 2263–2279, 2022.

Minesh Mathew, Viraj Bagal, Ruben Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawa- \` har. Infographicvqa. In 2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp. 2582–2591. IEEE, 2022.

Narges Norouzi, Svetlana Orlova, Daan De Geus, and Gijs Dubbelman. Algm: Adaptive local-thenglobal token merging for efficient semantic segmentation with plain vision transformers. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 15773–15782. IEEE, 2024.

Oleksii Sidorov, Ronghang Hu, Marcus Rohrbach, and Amanpreet Singh. Textcaps: a dataset for image captioning with reading comprehension. In European conference on computer vision, pp. 742–758. Springer, 2020.

Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8309–8318. IEEE, 2019.

Jintao Tong, Wenwei Jin, Pengda Qin, Anqi Li, Yixiong Zou, Yuhong Li, Yuhua Li, and Ruixuan Li. Flowcut: Rethinking redundancy via information flow for efficient vision-language models. Advances in Neural Information Processing Systems, 38:94946–94973, 2026.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024a.

Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. Cogvlm: Visual expert for pretrained language models. Advances in Neural Information Processing Systems, 37:121475–121499, 2024b.

Ziwei Xiang, Fanhu Zeng, Hongjian Fang, Rui-Qi Wang, Renxing Chen, Yanan Zhu, Yi Chen, Peipei Yang, and Xu-Yao Zhang. Fine-grained post-training quantization for large vision language models with quantization-aware integrated gradients. arXiv preprint arXiv:2603.17809, 2026.

Long Xing, Qidong Huang, Xiaoyi Dong, Jiajie Lu, Pan Zhang, Yuhang Zang, Yuhang Cao, Conghui He, Jiaqi Wang, Feng Wu, et al. Pyramiddrop: Accelerating your large vision-language models via pyramid visual redundancy reduction. arXiv preprint arXiv:2410.17247, 2024.

Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, and Jiaya Jia. Visionzip: Longer is better but not necessary in vision language models. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 19792–19802. IEEE, 2025a.

Sihan Yang, Runsen Xu, Chenhang Cui, Tai Wang, Dahua Lin, and Jiangmiao Pang. Vflowopt: A token pruning framework for lmms with visual information flow-guided optimization. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 23924–23934. IEEE, 2025b.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multi modal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9556–9567, 2024.

Qizhe Zhang, Aosong Cheng, Ming Lu, Renrui Zhang, Zhiyong Zhuo, Jiajun Cao, Shaobo Guo, Qi She, and Shanghang Zhang. Beyond text-visual attention: Exploiting visual cues for effective token pruning in vlms. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 20857–20867. IEEE, 2025.

Qizhe Zhang, Mengzhen Liu, Lichen Li, Ming Lu, Yuan Zhang, Junwen Pan, Qi She, and Shanghang Zhang. Beyond attention or similarity: Maximizing conditional diversity for token pruning in mllms. Advances in Neural Information Processing Systems, 38:25438–25468, 2026.

Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, et al. Sparsevlm: Visual token sparsifi cation for efficient vision-language model inference. arXiv preprint arXiv:2410.04417, 2024.

## A ADDITIONAL PROTOCOL DETAILS

Benchmarks and metrics. The full evaluation sets contain 5,000 TextVQA (Singh et al., 2019), 900 MMMU (Yue et al., 2024), 3,088 AI2D (Kembhavi et al., 2016), 1,500 MMStar (Chen et al., 2024b), 2,500 ChartQA (Masry et al., 2022), 4,500 NoCaps (Agrawal et al., 2019), 3,166 TextCaps (Sidorov et al., 2020), and 2,801 InfoVQA (Mathew et al., 2022) examples. TextVQA uses VQA scoring without additional OCR text in the prompt; MMMU, AI2D, and MMStar use accuracy; ChartQA uses relaxed correctness; NoCaps and TextCaps use CIDEr; and InfoVQA uses ANLS. CIDEr and ANLS are displayed after multiplication by 100. Avg. Rel. is the equal-weighted mean of each task score normalized by the corresponding Dense score and can exceed 100%. Four-task pruning ablations average InfoVQA, ChartQA, TextCaps, and MMStar equally, whereas the depth diagnostic in Figure 2 uses TextVQA, ChartQA, InfoVQA, and AI2D.

Evaluation and baseline adaptation. For task evaluation, Qwen models use FlashAttention-2 except for RESTORE, whose implementation requires SDPA to apply additive attention biases. All methods on LLaVA-OneVision-1.5 use SDPA. KV caching is disabled for all methods to maintain a consistent. evaluation configuration. Within each backbone, all methods share inputs, preprocessing, generation settings, and scorers. Baselines are adapted from their official implementations. FastV retains two initial decoder blocks before pruning. Since SparseVLM progressively changes the number of visual tokens across decoder depth, we match its average decoder visual-token usage to ours rather than its final token count. VisPruner, ZOO-Prune, and RESTORE perform their initial visual-token reduction before the decoder; RESTORE additionally applies position and attention correction inside the language model. Appendix C provides additional comparisons with progressive pruning methods.

## A.1 SELECTION IMPLEMENTATION

Encoder scores. Qwen aggregates vision-encoder attention received by each visual key across visual queries and heads. LLaVA-OneVision uses head-averaged CLS-to-patch attention from the penultimate vision-encoder block. Scores are aligned with each backbone’s visual-token geometry before top-M selection.

Midpoint scores. Midpoint scores are computed from the decoder queries and keys using the input prompt tokens as queries. Correct answers and generated answer tokens are not used for token selection. When attention probabilities are not directly exposed by the attention kernel, the required scores are recomputed from the corresponding queries and keys.

Computational budget. For reselection after block L, we measure visual-token processing using the token–block count

$$
B _ { \mathrm { v i s } } = L M + ( D - L ) K \approx D K + \alpha L ( N - K ) .\tag{9}
$$

This measure is used to match decoder visual-token workloads across selection depths and progressive pruning baselines. For the analytical FLOPs results in Figure 8, decoder cost is computed from the sequence length processed before and after reselection and averaged over individual inputs.

## A.2 RESERVE-SIZE TRADE-OFF

Figure 8 examines the effect of the reserve fraction α using two measures of computational cost: analytical decoder-prefill FLOPs and GPU-input-to-first-token latency. Each point corresponds to $\alpha \in \{ 0 . 1 0 , 0 . 1 5 , 0 . 2 0 , 0 . 2 5 , 0 . 3 0 \}$ at 80% pruning, with recovery averaged equally over all eight benchmarks.

Across all three backbones, larger reserves improve recovery while increasing computation, with diminishing gains at larger reserve fractions. The same trend under analytical FLOPs confirms that this trade-off is not specific to measured latency. The relative increase in first-token latency is smaller because it includes vision processing and other reserve-independent costs. Overall, α directly controls the trade-off between candidate preservation and inference cost.

## B ADDITIONAL ABLATION ANALYSIS

Effect of candidate preservation and reselection. Table 6 reports the fraction of final tokens selected from outside the initial vision-guided top-K, together with the recovery gain from candidate preservation to text-guided reselection. Across tasks and backbones, approximately 39–45% of the final visual tokens come from the reserved candidates, showing that midpoint text-to-visual attention substantially revises the initial visual ranking. The largest recovery gains appear on InfoVQA and ChartQA.

## C COMPARISON WITH PROGRESSIVE PRUNING

Table 7 extends our evaluation to two additional progressive pruning methods, PyramidDrop (Xing et al., 2024) and FlowCut (Tong et al., 2026). To account for their different pruning schedules, we match each baseline to Ours’ per-input visual token–block budget at α = 0.2, up to integer rounding. The 80% and 90% settings denote Ours’ final pruning ratios; progressive baselines may therefore use different final token counts under the matched workloads. Ours achieves the highest average relative performance at both budgets. At 80% pruning, Ours exceeds FlowCut and PyramidDrop by 9.51 and 19.26 percentage points, respectively; at 90%, the margins increase to 9.47 and 27.82 points.

![](images/472f7e5c5ff5c7361bbac3a9b20b3db0c7291c43f507a7bb5b1f593840c29b83.jpg)  
Figure 8: Reserve-size trade-offs at 80% pruning across the three backbones. Recovery is averaged equally across eight benchmarks relative to Dense. Top: analytical decoder-prefill FLOPs, excluding the vision frontend and token-selection overhead. Bottom: measured GPU-input-to-first-token latency, including the vision frontend and token-selection overhead but excluding CPU preprocessing and input transfer. Labels indicate α ∈ {0.10, 0.15, 0.20, 0.25, 0.30}, the fraction of otherwise pruned tokens preserved as additional candidates until midpoint reselection.

Table 6: Effect of candidate preservation and midpoint reselection at 80% pruning. Ret. is the input-averaged percentage of final tokens selected from outside the initial vision-guided top-K. ∆ is the Dense-relative recovery gain (percentage points) from candidate preservation to text-guided reselection under the same token-count trajectory.
<table><tr><td rowspan="2">Task</td><td colspan="2">Qwen3-VL-4B</td><td colspan="2">Qwen3-VL-8B</td><td colspan="2">LLaVA-OV-1.5-8B</td></tr><tr><td>Ret. (%)</td><td>∆(pp)</td><td>Ret. (%)</td><td>∆(pp)</td><td>Ret. (%)</td><td>∆(pp)</td></tr><tr><td>InfoVQA</td><td>42.09</td><td>+18.37</td><td>42.96</td><td>+14.09</td><td>44.28</td><td>+9.28</td></tr><tr><td>ChartQA</td><td>39.03</td><td>+13.27</td><td>42.00</td><td>+6.04</td><td>41.18</td><td>+4.57</td></tr><tr><td>TextCaps</td><td>40.65</td><td>+11.15</td><td>41.47</td><td>+6.47</td><td>44.74</td><td>+5.75</td></tr><tr><td>MMStar</td><td>43.84</td><td>+3.78</td><td>41.55</td><td>+0.11</td><td>44.38</td><td>+3.27</td></tr></table>

## D QUALITATIVE RESULTS

Figures 9–16 provide qualitative comparisons across all eight benchmarks at 80% pruning on three VLMs. The examples highlight a recurring limitation of existing pruning strategies: visually small but task-relevant evidence can be removed even when surrounding or semantically related regions are preserved. This is particularly visible on TextVQA, ChartQA, and InfoVQA, where answering correctly often depends on retaining localized text, numbers, or chart elements. ZOO-Prune can preserve visually salient regions while missing such details, whereas SparseVLM can retain related visual support yet lose the precise evidence or relationships required for the answer.

Table 7: Comparison with two additional progressive pruning methods on Qwen3-VL-8B under matched per-input visual token–block budgets. The 80% and 90% settings denote Ours’ final pruning ratios; baseline final token counts may differ under the matched workloads. Avg. Rel. is the eight-task mean performance relative to Dense. Bold/underline indicate the best/second-best pruned results.
<table><tr><td>Method</td><td>TextVQA</td><td>MMMU</td><td>AI2D</td><td>MMStar</td><td>ChartQA</td><td>NoCaps</td><td>TextCaps</td><td>InfoVQA</td><td>Avg. Rel.</td></tr><tr><td>Dense</td><td>82.95</td><td>51.67</td><td>83.68</td><td>62.73</td><td>83.48</td><td>63.54</td><td>81.62</td><td>81.19</td><td>100.00</td></tr><tr><td colspan="10">Matched to Ours at 80% pruning</td></tr><tr><td>PyramidDrop</td><td>61.77</td><td>48.22</td><td>70.76</td><td>47.33</td><td>45.80</td><td>62.41</td><td>66.92</td><td>33.58</td><td>75.53</td></tr><tr><td>FlowCut</td><td>77.23</td><td>50.44</td><td>75.71</td><td>51.07</td><td>50.40</td><td>62.47</td><td>83.83</td><td>47.29</td><td>85.28</td></tr><tr><td>Ours</td><td>80.48</td><td>51.33</td><td>81.96</td><td>57.47</td><td>75.20</td><td>63.49</td><td>83.95</td><td>64.57</td><td>94.79</td></tr><tr><td colspan="10">Matched to Ours at 90% pruning</td></tr><tr><td>PyramidDrop</td><td>43.18</td><td>47.78</td><td>67.78</td><td>41.33</td><td>25.28</td><td>57.12</td><td>48.34</td><td>25.87</td><td>62.83</td></tr><tr><td>FlowCut</td><td>72.50</td><td>49.89</td><td>71.99</td><td>49.87</td><td>44.32</td><td>61.64</td><td>79.10</td><td>42.97</td><td>81.18</td></tr><tr><td>Ours</td><td>76.93</td><td>50.56</td><td>79.86</td><td>55.53</td><td>69.12</td><td>62.88</td><td>80.71</td><td>56.83</td><td>90.65</td></tr></table>

In contrast, our deferred reselection can preserve these initially lower-ranked candidates until textto-visual attention becomes more informative, allowing the final token set to better retain questionrelevant evidence. The AI2D, MMMU, and MMStar examples further show that this behavior extends beyond scene text to diagrams and general multimodal reasoning, where relevant evidence may be spatially sparse or distributed across multiple regions. Finally, the TextCaps and NoCaps examples indicate that the benefit is not limited to question-conditioned tasks: our method also preserves visual information needed for more faithful open-ended descriptions. Together, these examples qualitatively support the main finding that delaying text-guided selection helps preserve task-relevant evidence that early pruning may otherwise discard.

![](images/4b505e82b062f84e6625a99d09ca461275f15937c2fd03ca6ffead8a3b8ff03a.jpg)  
Figure 9: TextVQA examples across three backbones at 80% pruning. Columns compare the original image, ZOO-Prune, SparseVLM, and Ours.

![](images/b64ce1f5ea0fc5c9155d96e41f40f5d057642b3807b26c4a92927215f3b0be4a.jpg)  
Figure 10: ChartQA examples across three backbones at 80% pruning. Columns compare the original image, ZOO-Prune, SparseVLM, and Ours.

![](images/a380fb3fbb418c94b1833bcedae272f1e41daf645f8f765c7b656e48c42a121e.jpg)  
Figure 11: AI2D examples across three backbones at 80% pruning. Columns compare the original image, ZOO-Prune, SparseVLM, and Ours.

## MMMU | Qwen3-VL-4B

Q: [image] is not just limited to political decisions.

A True

![](images/a6eb6fb5237d15b2c4c5773e607ce1519748dabedf6705cca8414353a6df7782.jpg)

B False

![](images/aa1175b0cf5e42976a788d1a7c063d82eb343586e647c07a07eccbbc22b8a99b.jpg)

![](images/fafdee2ac56125b97ffccd4c91eee1d09da4aaf8b7d57ba88cbaa403334f36b5.jpg)

D None of the above

![](images/4987e49f031453f752083846f0215f94e6819f04ee519ce1d6ba23e9eb1a503c.jpg)

## MMMU | Qwen3-VL-8B

Q: [image] The graph above shows the velocity versus time for an object moving in a straight line. At what time after t = 0 does the object again pass through its initial position? A 1 s B Between 1 and 2 s C 2 s D Between 2 and 3 s

![](images/6420f04ee71de0b0e1a93b27c4067794f3d99b0bb1d58f3b99024e1caec5c61a.jpg)

![](images/0e834b33cb2e5b42618f8acb9886b138f9ae421b0f6247575aca120cfe385f7d.jpg)

![](images/720c137552cf68ca859dfb4106f594b770b9b9750ec8df573fb5b7552409736f.jpg)

![](images/7aef35ec0dde5345f88ea355e136ee4ec616185d374daeb80811e500a147a0a1.jpg)

## MMMU | LLaVA-OV-1.5-8B

Q: What type of sleeve is this? [image] A Bell Sleeve B Puff Sleeve

GT: A  
![](images/d81c36f3b07321b646519fe0b60a2cb32d2bac1a284b895c3c15bfd7e90f0472.jpg)

![](images/4227d36fd1b97fae1c5830719198214b4620d68121b74d9a979c73adc09632d4.jpg)  
C Cap Sleeve

![](images/64a5a309fe9ed8fed8498fc1ade9c7469fb6757977705f37e2b1d32a5ec8d59b.jpg)

D Raglan Sleeve  
![](images/a671e5398eb5ffbcfc3c7af51beefd2bbbeaf5085da9016bbb13b3f529acb576.jpg)  
Figure 12: MMMU examples across three backbones at 80% pruning. Columns compare the original image, ZOO-Prune, SparseVLM, and Ours.

![](images/eaaaf3f868b9f878f4cf9011f23d786739818c03cbdaaa08186391e91ba66c9d.jpg)  
Figure 13: InfoVQA examples across three backbones at 80% pruning. Columns compare the original image, ZOO-Prune, SparseVLM, and Ours.

## MMStar | Qwen3-VL-4B

Q: How many jewelry items are present in the image? A 1 B 3 C 2 D 4

GT: B  
GT: B  
D 4  
![](images/616fd3a5083df0c7f0d4e1928802a4813dec0a45bc2b1c99ececbb5565127408.jpg)

![](images/364c50ab79c16726d438c208da7ce4c3a2aa2b01fc08f88bd7b76884f5c48bcd.jpg)

![](images/e323a38973a2f9e6e2084878d740dcf1ddbe3b475a821067c15f447bba62a62d.jpg)

![](images/bea6a37aff39f34232a3665025127986ef59a0156aec98761de33200ea1e6d77.jpg)

## MMStar | Qwen3-VL-8B

Q: How many ties can be seen in the image? A 3 B 2 C 1

![](images/208dbd527778bf4e4a8b1c8fe853659de08fca2e2d31879263ba96d807c2f7a3.jpg)

![](images/782b21f393becbff743a32e53944a8e39b107da900661d6ce08e410846995d3b.jpg)

![](images/d9048f42cbe997299894e06b03a216cef5ff386d3e0c6f37f23360fdbee1d8d2.jpg)

![](images/9aa8dcca0527401f20d47a1a03683553892dc83e34693165606d595d08a1c52c.jpg)

## MMStar | LLaVA-OV-1.5-8B

Q: How many types of fruits are in the bowl? A 4 B 2 C 3

GT: A  
![](images/96378d33558be084f4ec393629efa8f664929b3a73a9b2820e75d978d602eee8.jpg)

![](images/07065399e3e05b7fa333b22bd29ef5a98e1073f063d3faf9f1d30ec88705c485.jpg)

![](images/811c17638446821bee10d490e1339625a14322b79ff945f795be2871dc377e3f.jpg)

![](images/ef8b1979ca16446bca64589daa77e50463a0bfc44f20d51f36831173d8d7c0db.jpg)  
Figure 14: MMStar examples across three backbones at 80% pruning. Columns compare the original image, ZOO-Prune, SparseVLM, and Ours.

## TextCaps | Qwen3-VL-4B

Q: Output only a concise caption of at most 12 words for the image.   
GT: An interior designer is named Lang and Potter.

![](images/2659ea0bc9a4cd6963b37cc9264d3d733828302efd2ff2228ee5597e9699b916.jpg)

![](images/4ff55b5c10558ec5ef228a7b9661610c53b2f33cb022959483dfc50459377b34.jpg)  
Answer: White van with "Lan & PLYMOUTH" branding on side.

![](images/c7cb205cbb1de2a3ad5d20128d3156cde08344d7eab12d0569b38150cbd1be3c.jpg)  
Answer: White van with company branding and driver visible.

![](images/257b12a3f1287dc5174d697f06c355a619410cd22669f2c92a485a59a5276a76.jpg)  
Answer: White van with "Lang & Potter" branding on side.

## TextCaps | Qwen3-VL-8B

Q: Output only a concise caption of at most 12 words for the image.

GT: A screen is shown displaying several fighters and the words "LaRusso's gonna fight!"

![](images/a94dffecb06e408098af8a755080fc2cc22934b64699fa88735028362b5ed540.jpg)

![](images/6c02d30d285866bf058f397268213970956c935ba6a47ea01eeefec5025e9c6e.jpg)  
Answer: Projected wrestling match scene on wall.

![](images/b1b45b342620cbf102b2fbd6f07f3ddadd8637c90b25d3059de621e08c410e83.jpg)  
Answer: Movie scene with subtitle: "La fight!"

![](images/df9ab5584f6469e4e33140c76725995c826b18ed5884bcafa3003d6b41fbcb87.jpg)  
Answer: Projected scene with text: "LaRusso's gonna fight!"

## TextCaps | LLaVA-OV-1.5-8B

Q: Output only a concise caption of at most 12 words for the image. GT: An ad for the Louvre Museum is displayed on an LG computer screen

![](images/6d7109f3133d454d82fe2d6a7b14a4ba9e4d18ea093733aa26ed4933267144b4.jpg)

![](images/9c50fa7d14d1fc94b3a3cc510dc6b5293507ac398eb9111a74c76153e89b4a3e.jpg)

Answer: A computer monitor with an advertisement for The Museum.

![](images/3a9119ca480119c2cbd246df21268221ce77d242401ca514202e9616653e9351.jpg)  
Answer: A computer monitor with The Museum on it.

![](images/b5e5bca8d5e3372e0d5966c5f703b9f989e3ec52192b4d9f141c348c60166bee.jpg)  
Answer: A computer monitor with The Louvre Museum on it.

Figure 15: TextCaps examples across three backbones at 80% pruning. Columns compare the original image, ZOO-Prune, SparseVLM, and Ours.

## Dropped SparseVLM: merged Ours: text-guided retention

## NoCaps | Qwen3-VL-4B

Q: Output only a concise caption of at most 12 words for the image.   
GT: Two zebras stand side by side facing in different directions.

![](images/729ab56025823c1753cf1903f22dcd0b4358ea6d6b6f5f7ba5719d40f0624b6f.jpg)

![](images/1c1b65a3889667b0df069031c1a5789698ca3fbdb607dacf39c73b428139ffbc.jpg)  
Answer: Zebra with distinctive black and white stripes.

![](images/19db8096ae0382aef51bb5b040f49dede228074268240930e18a7364437bdccb.jpg)  
Answer: Zebra with distinctive black and white stripes.

![](images/909a4518b779cd31e844593d216cb427af33d7f2878d70ba3a47c5d31a52b1d2.jpg)  
Answer: Two zebras standing close together in a natural setting.

## NoCaps | Qwen3-VL-8B

Q: Output only a concise caption of at most 12 words for the image.   
GT: Three different kinds of knives sit on a black cutting board.

![](images/4b1d7227aba426ce2a2ed62126d24e04faec011b405e9ec46e61f701836acb7b.jpg)

![](images/d451d0ba071b7acfd5c7730ce5849683e02e5550ab7574981cc3f5dbfc02a81f.jpg)  
Answer: Two kitchen knives on dark surface with green garnish.

![](images/9ccfb7579da286e8c70219f700d92cf610ff90f033e5174f6fa99acb3813e1b3.jpg)  
Answer: Two kitchen knives resting on dark surface.

![](images/812017d9a329f33cbce154a0651d2e99b82f1d04fcfe4f9edcff4c7e5b63d7e6.jpg)  
Answer: Three kitchen knives with black handles on dark surface.

## NoCaps | LLaVA-OV-1.5-8B

Q: Output only a concise caption of at most 12 words for the image.   
GT: Five balls rest on a green billiards table.

![](images/3f44b7715ba1138536d1f221a0f07f0902f6a1010dbb9753f4bbb02b6c16e080.jpg)

![](images/e20d027b6eddc4f9827d9075cb0236f23caa0a9bba9ab09917143fac85e2f880.jpg)  
Answer: A green surface with four balls on it, one white and three brown.

![](images/c41b5b6d3b42080f6749a011f43996239ad635249b94767b4a528b3b8ff60a5f.jpg)  
Answer: Four balls on a green surface, one with a red top and another with a blue top.

![](images/37b2009959d2c9a2779c27ce4b83e4913a0e24ddf594df792d01901832e9f6fa.jpg)  
Answer: A pool table with five balls on it, one white and four brown.

Figure 16: NoCaps examples across three backbones at 80% pruning. Columns compare the original image, ZOO-Prune, SparseVLM, and Ours.