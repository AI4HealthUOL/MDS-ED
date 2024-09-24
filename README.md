# MDS-ED: Multimodal Decision Support in the Emergency Department -- a Benchmark Dataset for Diagnoses and Deterioration Prediction in Emergency Medicine

This is the official repository for the paper MDS-ED: Multimodal Decision Support in the Emergency Department -- a Benchmark Dataset for Diagnoses and Deterioration Prediction in Emergency Medicine

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2407.17856)

## Clinical Setting

In this study, conducted within the context of an emergency department, we introduce a state-of-the-art biomedical multimodal benchmark. This benchmark is evaluated in two comprehensive settings:

1. **Patient Discharge Diagnoses**: A dataset consisting of 1,428 patient discharge diagnoses.
2. **Patient Deterioration Events**: A dataset consisting of 15 patient deterioration events.

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

![alt text](https://github.com/AI4HealthUOL/MDS-ED/blob/main/reports/bench.png?style=centerme)


## Conclusions

Overall, we can draw several conclusions:

1. Firstly, the results demonstrate that multimodal models, which integrate diverse data types, offer superior performance in both diagnostic and deterioration tasks (row 4&5 vs. the rest). 

2. Secondly, in the diagnoses task as well as in the deterioration task, the use of ECG raw waveforms instead of ECG features improves the performance in a statistically significant manner (row 4 vs. row 5), finding which is not in line with [ 14 ]. To the best of our knowledge, this is the first statistically robust demonstration of the added value of raw ECG waveform input against ECG features
for clinically relevant prediction tasks such as diagnoses and deterioration prediction. 

3. Thirdly, for the unimodal models, in the deterioration task, the clinical routine data model outperforms ECG features only and ECG waveforms only, however, in the diagnoses task, the ECG waveforms only outperforms the other 2 settings, we hypothesize that for the deterioration task, the clinical routine data apart of including a rich set of clinical features (demographics, biometrics, vital parameters trends, and laboratory values trends) against only an single ECG either in features or waveform, it also includes trends over time which aligns with the task definition of deterioration. Despite this, we believe that a single ECG snapshot can achieve high performances for both tasks, but also we believe that the inclusion of multiple ECGs over time instead of just a single snapshot would allow us to capture more meaningful deterioration and potentially diagnoses trends.


## Reference
```bibtex
@misc{alcaraz2024mdsedmultimodaldecisionsupport,
      title={MDS-ED: Multimodal Decision Support in the Emergency Department -- a Benchmark Dataset for Diagnoses and Deterioration Prediction in Emergency Medicine}, 
      author={Juan Miguel Lopez Alcaraz and Hjalmar Bouma and Nils Strodthoff},
      year={2024},
      eprint={2407.17856},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.17856}, 
}
```
