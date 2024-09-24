## Data 

1.1 - For convenience, we have located the MDS-ED dataset in [PhysioNet](https://physionet.org/content/multimodal-emergency-benchmark).

1.2 - Alternatively, you can preprocess the MDS-ED dataset on your own, first, download from the following databases the following files and place them under data/, then run full_preprocessing.py

- From MIMIC-IV-ECG-ICD: records_w_diag_icd10.csv
- From MIMIC-IV-ED: edstays.csv.gz, diagnosis.csv.gz, pyxis.csv.gz, vitalsign.csv.gz, triage.csv.gz
- From MIMIC-IV: admissions.csv.gz, diagnoses_icd.csv.gz, d_labitems.csv.gz, labevents.csv.gz, icustays.csv.gz, procedures_icd.csv.gz, omr.csv.gz
- From MIMIC-IV-ECG: mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip

2.0 - Please see under demo.ipynb an example of how to acess and manipulate the dataset.


## Experiments

We provide a convenient pipeline for our multimodal (ECG waveforms + tabular) experiments replication. The next commands would train and test the diagnoses and deterioration models respectively:

```
python main_all.py --config config/config_supervised_multimodal_mdsed_diagnoses_s4.yaml
```

```
python main_all.py --config config/config_supervised_multimodal_mdsed_deterioration_s4.yaml
```
