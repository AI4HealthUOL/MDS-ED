# MDS-ED: Multimodal Decision Support in the Emergency Department – a benchmark dataset based on MIMIC-IV

This is the official repository for the paper MDS-ED: Multimodal Decision Support in the Emergency Department – a benchmark dataset based on MIMIC-IV

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

![alt text](https://github.com/AI4HealthUOL/MDS-ED/blob/main/reports/abstract_image.png?style=centerme)


## Comparison to Prior Benchmarks

1. **Comprehensive Size**: MDS-ED ranks first in terms of the number of patients and second in the number of visits in the open-source domain, despite focusing only on the first 1.5 hours of ED arrival.

2. **Features Diversity**: MDS-ED leads in feature modalities, including demographics, biometrics, vital parameter trends, laboratory value trends, and ECG waveforms, making it more extensive than most datasets. Chief complaints and previous medications were excluded due to their unstructured nature and potential bias.

3. **Extensive Range of Target Labels**: MDS-ED offers 1,443 target labels, significantly more than other datasets, which usually have fewer and narrower scope tasks.

4. **Accessibility**: MDS-ED is open-source, promoting further research and collaboration.

![alt text](https://github.com/AI4HealthUOL/MDS-ED/blob/main/reports/related_work.png?style=centerme)


## Proposed Baseline Benchmark

Overall, we can draw several conclusions:

1. **Superior Performance of Multimodal Models**: The results demonstrate that multimodal models, which integrate diverse data types, offer superior performance in both diagnostic and decompensation tasks.
2. **Improved Performance with Raw ECG Waveforms**: Using ECG raw waveforms instead of ECG features improves performance. This is the first demonstration of the added value of raw waveform input for clinically relevant prediction tasks.
3. **ECG Waveform Superiority in Diagnostic Setting**: ECG waveforms only show superiority over a robust set of tabular features for the diagnostic setting, but not for decompensation. This may be due to the inclusion of tabular trends over time, which correlate with the task definition of decompensation. Including two raw ECGs over time instead of a single snapshot might potentially capture meaningful decompensation trends.

![alt text](https://github.com/AI4HealthUOL/MDS-ED/blob/main/reports/benchmark.png?style=centerme)


## Clinical Significance






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
