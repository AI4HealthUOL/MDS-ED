# MDS-ED: Multimodal Decision Support in the Emergency Department -- a Benchmark Dataset for Diagnoses and Deterioration Prediction in Emergency Medicine

This is the official repository for the paper MDS-ED: Multimodal Decision Support in the Emergency Department -- a Benchmark Dataset for Diagnoses and Deterioration Prediction in Emergency Medicine

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2405.15871)

## Clinical Setting

In this study, conducted within the context of an emergency department, we introduce a state-of-the-art biomedical multimodal benchmark. This benchmark is evaluated in two comprehensive settings:

1. **Patient Discharge Diagnoses**: A dataset consisting of 1,428 patient discharge diagnoses.
2. **Patient Decompensation Events**: A dataset consisting of 15 patient decompensation events.

The datasets include various patient data collected within a 90-minute interval upon arrival, such as:
- Demographics
- Biometrics
- Vital parameter trends
- Laboratory value trends
- ECG waveforms

![alt text](https://github.com/AI4HealthUOL/MDS-ED/blob/main/reports/abstract_img.png?style=centerme)


## Comparison to Prior Benchmarks

1. **Comprehensive Size**: MDS-ED ranks first in terms of the number of patients and second in the number of visits in the open-source domain, despite focusing only on the first 1.5 hours of ED arrival.

2. **Features Diversity**: MDS-ED leads in feature modalities, including demographics, biometrics, vital parameter trends, laboratory value trends, and ECG waveforms, making it more extensive than most datasets. Chief complaints and previous medications were excluded due to their unstructured nature and potential bias.

3. **Extensive Range of Target Labels**: MDS-ED offers 1,443 target labels, significantly more than other datasets, which usually have fewer and narrower scope tasks.

4. **Accessibility**: MDS-ED is open-source, promoting further research and collaboration.

![alt text](https://github.com/AI4HealthUOL/MDS-ED/blob/main/reports/related_work.png?style=centerme)


## Proposed Baseline Benchmark

Overall, we can draw several conclusions:

1. **Superior Performance of Multimodal Models**: The results demonstrate that multimodal models, which integrate diverse data types, offer superior performance in both diagnostic and deterioration tasks (row 1/2 vs. row 3/4).
2. **Improved Performance with Raw ECG Waveforms**: In the diagnoses task, the use of ECG raw waveforms instead of ECG features improves the performance in a statistically significant manner (row 3 vs. row 4), whereas for the deterioration task, we performed a direct comparison via bootstrapping the score difference for statistical significance, and we did not find a statistically significant difference. To the best of our knowledge, this is the first statistically robust demonstration of the added value of raw waveform input against features for clinically relevant prediction tasks such as diagnoses prediction.
3. **ECG Waveform Superiority in Diagnostic Setting**: The model building on ECG waveforms as only input outperforms the tabular-only model in the diagnostic setting, but not in the deterioration setting (row 1 vs. row 2). We hypothesize that this is due to the inclusion of tabular trends over time which aligns with the task definition of deterioration. We believe that the inclusion of multiple raw ECGs over time instead of just a single snapshot would allow us to capture more meaningful deterioration trends also from raw waveform data.

![alt text](https://github.com/AI4HealthUOL/MDS-ED/blob/main/reports/benchmark.png?style=centerme)




## Reference
```bibtex
@misc{alcaraz2024causalconceptts,
      title={CausalConceptTS: Causal Attributions for Time Series Classification using High Fidelity Diffusion Models}, 
      author={Juan Miguel Lopez Alcaraz and Nils Strodthoff},
      year={2024},
      eprint={2405.15871},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
