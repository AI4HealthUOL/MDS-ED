## Data 

For convenience, we have located the MDS-ED dataset in physionet, follow this link

Alternatively, you can preprocess the MDS-ED dataset on your own, first, download from the following databased the following files and place them under this directory, then run full_preprocessing.py

- From MIMIC-IV-ECG-ICD: records_w_diag_icd10.csv
- From MIMIC-IV-ED: edstays.csv.gz, diagnosis.csv.gz, pyxis.csv.gz, vitalsign.csv.gz, triage.csv.gz
- From MIMIC-IV: admissions.csv.gz, diagnoses_icd.csv.gz, d_labitems.csv.gz, labevents.csv.gz, icustays.csv.gz, procedures_icd.csv.gz, omr.csv.gz



## Experiments

If you wish to replicate our experiments, 

waveforms

https://github.com/AI4HealthUOL/ECG-MIMIC 



```
python full_preprocessing.py --mimic-path <path to mimic-iv directory ended in 'mimiciv/2.2/'> --zip-path <path to ecgs zip file> --target-path <desired output for preprocessed data default='./'>
```


