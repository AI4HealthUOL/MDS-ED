## Data 

For convenience, we have located the MDS-ED dataset in physionet. **under revision**

Alternatively, you can preprocess the MDS-ED dataset on your own, first, download from the following databased the following files and place them under data/, then run full_preprocessing.py

- From MIMIC-IV-ECG-ICD: records_w_diag_icd10.csv
- From MIMIC-IV-ED: edstays.csv.gz, diagnosis.csv.gz, pyxis.csv.gz, vitalsign.csv.gz, triage.csv.gz
- From MIMIC-IV: admissions.csv.gz, diagnoses_icd.csv.gz, d_labitems.csv.gz, labevents.csv.gz, icustays.csv.gz, procedures_icd.csv.gz, omr.csv.gz

**Since MIMIC-IV-ECG-ICD is under revision, please contact me at juan.lopez.alcaraz@uol.de so I can provide a numpy file for the stratified splits and the diagnostic columns, which are the only variables requiered from MIMIC-IV-ECG-ICD**

Please see under demo.ipybn an example of how to acess and manipulate the dataset.


## Experiments

We provide a convenient pipeline for our multimodal (ECG waveforms + tabular) experiments replication. The next commands would train and test the diagnoses and deterioration models respectively:

```
python main_ecg.py --data data/memmap --input-size 250 --finetune-dataset mds_diags --architecture s4mm --precision 32 --s4-n 8 --s4-h 512 --batch-size 32 --epochs 20 --export-predictions-path multi_diags/ > multi_diags/multi_diags.txt
```
```
python main_ecg.py --data data/memmap --input-size 250 --finetune-dataset mds_decomp --architecture s4mm --precision 32 --s4-n 8 --s4-h 512 --batch-size 32 --epochs 20 --export-predictions-path multi_det/ > multi_det/multi_det.txt
```

