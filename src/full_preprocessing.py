import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import datetime
from datetime import timedelta
from tqdm import tqdm
import glob
from sklearn.linear_model import LinearRegression

from ecg_utils import prepare_mimicecg
from timeseries_utils import reformat_as_memmap

import argparse
from pathlib import Path
import icd10

!pip install dtype_diet
from dtype_diet import report_on_dataframe, optimize_dtypes

pd.options.mode.chained_assignment = None


# Convert signals into numpy 
zip_file_path = Path('data/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip') # path to mimic-ecg zip
target_path = Path('data/memmap/') # desired output path
df,_,_,_=prepare_mimicecg(zip_file_path, target_folder=target_path)


# Reformat as memmap for fast access
reformat_as_memmap(df, 
                   target_path/"memmap.npy", 
                   data_folder=target_path, 
                   annotation=False, 
                   max_len=0, 
                   delete_npys=True, 
                   col_data="data", 
                   col_lbl=None, 
                   batch_length=0, 
                   skip_export_signals=False)


# Preprocess the tabular 

df_diags = pd.read_csv('data/records_w_diag_icd10.csv')
df_diags['all_diag_all']=df_diags['all_diag_all'].apply(lambda x:eval(x))


df_edstays = pd.read_csv('data/edstays.csv.gz') 
df_diags = df_diags[df_diags['ed_stay_id'].isin(df_edstays['stay_id'].unique())]
df_diags['ecg_time'] = pd.to_datetime(df_diags['ecg_time'])
df_edstays['intime'] = pd.to_datetime(df_edstays['intime'])
df_edstays['outtime'] = pd.to_datetime(df_edstays['outtime'])
df_diags['dod'] = pd.to_datetime(df_diags['dod'])

rows_to_keep = []

# Iterate over each row in the df_diags DataFrame (unique MIMIC-ICD ECG rows)
for _, row in df_diags.iterrows():
    patient = row['subject_id']
    ecg_time = row['ecg_time']
    studys_id.append(row['study_id'])

    df_patient = df_edstays[df_edstays['subject_id'] == patient]
  
    for _, patient_row in df_patient.iterrows(): 
        intime = patient_row['intime']
        intime_90min = intime + pd.Timedelta(minutes=90)

        if intime <= ecg_time <= intime_90min:
            row_dict = row.to_dict()
            row_dict['stay_id'] = patient_row['stay_id']
            row_dict['intime'] = patient_row['intime']
            row_dict['outtime'] = patient_row['outtime']
            row_dict['gender'] = patient_row['gender']
            row_dict['race'] = patient_row['race']

            rows_to_keep.append(row_dict)
            break  

df_filtered_diags = pd.DataFrame(rows_to_keep)
df_filtered_diags.reset_index(drop=True, inplace=True)
df_filtered_diags['90min'] = df_filtered_diags['intime'] + pd.Timedelta(minutes=90)

# Mortality
df_filtered_diags['mortality_hours'] = (df_filtered_diags['dod'] - df_filtered_diags['intime']).dt.total_seconds() / 3600
df_filtered_diags['mortality_days'] = (df_filtered_diags['dod'] - df_filtered_diags['intime']).dt.days


mortality_cols = ['mortality_1D','mortality_7D','mortality_28D','mortality_90D','mortality_180D','mortality_365D']

for col in mortality_cols:
    binary_values = []
    for _, row in df_filtered_diags.iterrows():
        
        if pd.notnull(row['mortality_hours']) and row['mortality_hours'] < 1.5:
            binary_values.append(-999)
        else:
            mortality_days = row['mortality_days']
            ecg_time = row['ecg_time']
        
            if pd.notnull(mortality_days):
                binary_values.append(1 if mortality_days <= int(col.split('_')[-1][:-1]) else 0)

            else:
                binary_values.append(0)
                
    df_filtered_diags[col] = binary_values
    

df_hosp = pd.read_csv('data/admissions.csv.gz')
dischtime_dict = df_hosp.set_index('hadm_id')['dischtime'].to_dict()
df_filtered_diags['hosp_dischtime'] = df_filtered_diags['hosp_hadm_id'].map(dischtime_dict)
df_filtered_diags['hosp_dischtime'] = pd.to_datetime(df_filtered_diags['hosp_dischtime'])

condition = []

for _, row in df_filtered_diags.iterrows():
    
    if pd.notnull(row['mortality_hours']) and row['mortality_hours'] < 1.5: 
        condition.append(-999)
        
    elif not pd.notnull(row['dod']):
        condition.append(0)
        
    else:
        if row['dod']<row['hosp_dischtime']:
            condition.append(1)
        elif row['dod']<row['outtime']:
            condition.append(1)
        elif row['dod']>row['hosp_dischtime'] or row['dod']>row['outtime']:
            condition.append(0)


            
df_filtered_diags['mortality_stay'] = condition
mortality_cols.append('mortality_stay')

df_filtered_diags[mortality_cols] = df_filtered_diags[mortality_cols].astype(float)


# Diagnoses
df_filtered_diags = df_filtered_diags[df_filtered_diags['all_diag_all'].apply(lambda x: len(x) > 0)]
df_filtered_diags["all_diag_all"]=df_filtered_diags["all_diag_all"].apply(lambda x: list(set([y.strip()[:5] for y in x])))
df_filtered_diags["all_diag_all"]=df_filtered_diags["all_diag_all"].apply(lambda x: list(set([y.rstrip("X") for y in x])))
df_filtered_diags['all_diag_all'] = df_filtered_diags['all_diag_all'].apply(lambda code: code[:-1] if code[-1].isalpha() else code)

def label_propagation(df, column_name, propagate_all=True):
    def flatten(l):
        return [item for sublist in l for item in sublist]
    
    col_flat = flatten(np.array(df[column_name]))
    
    def prepare_consistency_mapping_internal(codes_unique, codes_unique_all):
        res={}
        for c in codes_unique:
            if(propagate_all):
                res[c]=[c[:i] for i in range(3,len(c)+1)]
            else: 
                res[c]=np.intersect1d([c[:i] for i in range(3,len(c)+1)],codes_unique_all)
        return res
    
    cons_map = prepare_consistency_mapping_internal(np.unique(col_flat), np.unique(col_flat))
    df[column_name] = df[column_name].apply(lambda x: list(set(flatten([cons_map[y] for y in x]))))
    return df

df_filtered_diags = label_propagation(df_filtered_diags, "all_diag_all", propagate_all=True)


def process_multilabel(df, column_name, threshold, output_column_name):

    counts = {}
    
    for row in df[column_name]:
        for item in row:
            counts[item] = counts.get(item, 0) + 1
    
    filtered_counts = {item: count for item, count in counts.items() if count >= threshold}
    
    unique_strings = sorted(filtered_counts, key=filtered_counts.get, reverse=True)
    
    df[output_column_name] = df[column_name].apply(lambda row: [1 if item in row else 0 for item in unique_strings])

    return df, np.array(unique_strings)

df_filtered_diags, lbls_diags = process_multilabel(df_filtered_diags, 'all_diag_all', 230, 'Diagnoses_labels')



# ICU

df_icu = pd.read_csv('data/icustays.csv.gz')
df_icu['intime'] = pd.to_datetime(df_icu['intime'])

icus = []

for i, row in df_filtered_diags.iterrows():
    
    ed_hadm_id = row['ed_hadm_id']
    if ed_hadm_id in df_icu['hadm_id'].values:
        icu_intime = df_icu[df_icu['hadm_id']==ed_hadm_id]['intime'].values[0]
        patient_intime = row['intime']
        time_diff_hours = (icu_intime - patient_intime).total_seconds() / 3600.0
        icus.append(time_diff_hours)
    else:
        icus.append(np.nan)


df_filtered_diags['icu_time_hours'] = icus
icu_24 = []
icu_stay = []

for i, row in df_filtered_diags.iterrows():
    
    if not pd.isna(row['icu_time_hours']) and row['icu_time_hours']<1.5:
        icu_24.append(-999)
        icu_stay.append(-999)
    
    elif row['icu_time_hours']>=1.5 and row['icu_time_hours']<=24:
        icu_24.append(1)
        icu_stay.append(1)
        
    elif row['icu_time_hours']>24:
        icu_24.append(0)
        icu_stay.append(1)
        
    elif pd.isna(row['icu_time_hours']):
        icu_24.append(0)
        icu_stay.append(0)
        
df_filtered_diags['ICU_24H'] = icu_24
df_filtered_diags['ICU_stay'] = icu_stay

icu_cols = ['ICU_24H', 'ICU_stay']
df_filtered_diags[icu_cols] = df_filtered_diags[icu_cols].astype(float)


# targets with proedures

df_proc = pd.read_csv('data/procedures_icd.csv.gz')
df_proc['chartdate'] = pd.to_datetime(df_proc['chartdate'])
mechanical_ventilation_codes = ['9670', '9671', '9672', '5A1935Z', '5A1945Z', '5A1955Z'] 
extracorporeal_membrane_oxygenation = ['3961', '3965', '3966', '5A1221Z', '5A1522G','5A1522H','5A15223','5A1522F','5A15A2F','5A15A2G','5A15A2H']  # ECMO codes

new_col_mechanical_v = []
new_col_ecmo = []

for i, row in df_filtered_diags.iterrows():
    patient = row['subject_id']
    ed_hadm_id = row['ed_hadm_id']
    intime = row['intime']
    
    df_patient = df_proc[(df_proc['subject_id'] == patient) & (df_proc['hadm_id'] == ed_hadm_id)]
    
    df_patient_mechanical = df_patient[df_patient['icd_code'].isin(mechanical_ventilation_codes)]
    df_patient_mechanical = df_patient_mechanical[df_patient_mechanical['chartdate'].dt.date <= intime.date() + pd.Timedelta(days=1)]
    if not df_patient_mechanical.empty:
        new_col_mechanical_v.append(1)
    else:
        new_col_mechanical_v.append(0)
        
    df_patient_ecmo = df_patient[df_patient['icd_code'].isin(extracorporeal_membrane_oxygenation)]
    df_patient_ecmo = df_patient_ecmo[df_patient_ecmo['chartdate'].dt.date <= intime.date() + pd.Timedelta(days=1)]
    if not df_patient_ecmo.empty:
        new_col_ecmo.append(1)
    else:
        new_col_ecmo.append(0)
    

df_filtered_diags['mechanical_ventilation'] = new_col_mechanical_v
df_filtered_diags['ecmo'] = new_col_ecmo


d = pd.read_csv('data/diagnosis.csv.gz')
d_cardiac_ed = d[d['icd_title'].str.lower().str.contains('cardiac arrest')]

cardiac_arrests = ['I469', '4275', 'I462', 'V1253', 'I468']

d = pd.read_csv('data/diagnoses_icd.csv.gz')
d_cardiac_hosp = d[d['icd_code'].isin(cardiac_arrests)]
d_cardiac_hosp = d_cardiac_hosp[d_cardiac_hosp['hadm_id'].isin(df_filtered_diags['hosp_hadm_id'])]

cardiac_arrest = []

for i, row in df_filtered_diags.iterrows():

    patient = row['subject_id']
    ed_stay_id = row['ed_stay_id']
    hosp_id = row['hosp_hadm_id']
    
    intime = row['intime']
    
    df_ed = d_cardiac_ed[(d_cardiac_ed['subject_id'] == patient) & (d_cardiac_ed['stay_id'] == ed_stay_id)]
    
    if len(df_ed)>0:
        ed_outtime = row['outtime']
        if ed_outtime <= intime+pd.Timedelta(days=1):
            cardiac_arrest.append(1)
        else:
            df_hosp = d_cardiac_hosp[(d_cardiac_hosp['subject_id'] == patient) & (d_cardiac_hosp['hadm_id'] == hosp_id)]
        
            if len(df_hosp)>0:
                hosp_outtime = row['hosp_dischtime']
                if hosp_outtime <= intime+pd.Timedelta(days=1):
                    cardiac_arrest.append(1)
            
                else:
                    cardiac_arrest.append(0)
                    
            else:
                cardiac_arrest.append(0)
        
    else:
        cardiac_arrest.append(0)
        
df_filtered_diags['cardiac_arrest'] = cardiac_arrest

vasopressors='epinephrine, norepinephrine, vasopressin, dobutamine, dopamine, or phenylephrine'
inotropes='epinephrine, dobutamine, or dopamine'


df_pyxis = pd.read_csv('data/pyxis.csv.gz')
df_pyxis = df_pyxis[df_pyxis['subject_id'].isin(df_filtered_diags['subject_id'].unique())]
df_pyxis = df_pyxis.dropna(subset='gsn')
df_pyxis['charttime'] = pd.to_datetime(df_pyxis['charttime'])

total_vasopressors = ['Norepinephrine','NORepinephrine','EPINEPHrine (for dilution)',
                      'NORepinephrine (in NS)','EPINEPHrine Auto Injector','EPINEPHrine (for dil 2mg/2mL 2mL',
                      'EPINEPHrine','EPINEPHrine 1mg/1mL 1mL AMP','EPINEPHrine (B 1mg/10mL 10mL SYR',
                      'EPINEPHrine (Bristojet)','EPINEPHrine 1mg/10mL 10mL SYR','NORepinephrine ((for dil 8mg KIT',
                      'Norepinephrine','NORepinephrine','NORepinephrine (in NS)',
                      'NORepinephrine ((for dil 8mg KIT','Vasopressin','Vasopressin (for 40units/2mL 2mL',
                      'Vasopressin (for dilution)','DOPamine','DOPamine 400mg BAG',
                      'PHENYLEPHrine','PHENYLEPHrine (for dilution)','Phenylephrine',
                      'PHENYLEPHrine (for dilu 60mg/6mL','PHENYLEPHrine (for 50mg/5mL VIAL','PHENYLEPHrine (QuVa)',
                      'DOBUTamine 250mg BAG','DOBUTamine']

total_inotropes = ['EPINEPHrine (for dilution)','EPINEPHrine Auto Injector','EPINEPHrine (for dil 2mg/2mL 2mL',
                   'EPINEPHrine','EPINEPHrine 1mg/1mL 1mL AMP','EPINEPHrine (B 1mg/10mL 10mL SYR',
                   'EPINEPHrine (Bristojet)','EPINEPHrine 1mg/10mL 10mL SYR','DOPamine',
                   'DOPamine 400mg BAG','DOBUTamine 250mg BAG','DOBUTamine']

vasopressors = []
inotropes = []

for i, row in df_filtered_diags.iterrows():
    
    stay_id = row['ed_stay_id']
    intime = row['intime']
    
    df_meds = df_pyxis[df_pyxis['stay_id'] == stay_id]
    
    cutoff_time = row['90min']
    
    meds = df_meds[df_meds['charttime'] >= cutoff_time]['name'].tolist()
    
    if any(med in total_vasopressors for med in meds):
        vasopressors.append(1)
    else:
        vasopressors.append(0)

    if any(med in total_inotropes for med in meds):
        inotropes.append(1)
    else:
        inotropes.append(0)
        
df_filtered_diags['vasopressors'] = vasopressors
df_filtered_diags['inotropes'] = inotropes


df_vitalsign = pd.read_csv('data/vitalsign.csv.gz')
df_vitalsign = df_vitalsign[df_vitalsign['subject_id'].isin(df_filtered_diags['subject_id'].unique())]
df_vitalsign['charttime'] = pd.to_datetime(df_vitalsign['charttime'])

df_vitalsign.loc[df_vitalsign['temperature'] < 50, 'temperature'] = np.nan # 53 minimum recorded
df_vitalsign.loc[df_vitalsign['temperature'] > 150, 'temperature'] = np.nan # 115.7 maximum recorded
df_vitalsign.loc[df_vitalsign['heartrate'] > 700, 'heartrate'] = np.nan # 600 maximum recorded
df_vitalsign.loc[df_vitalsign['resprate'] > 300, 'resprate'] = np.nan # normal 20, athletes 50
df_vitalsign.loc[df_vitalsign['o2sat'] > 100, 'o2sat'] = np.nan # can't be negative nor more than 100
df_vitalsign.loc[df_vitalsign['o2sat'] < 0, 'o2sat'] = np.nan # can't be negative nor more than 100
df_vitalsign.loc[df_vitalsign['dbp'] > 500, 'dbp'] = np.nan # max recorded 370
df_vitalsign.loc[df_vitalsign['sbp'] > 500, 'sbp'] = np.nan # max recorded 360


severe_hypoxemia = []

for i, row in df_filtered_diags.iterrows():
    
    df_patient_vital_before = df_vitalsign[(df_vitalsign['stay_id'] == row['ed_stay_id']) & 
                                    (df_vitalsign['charttime'] <= row['90min'])]
    df_patient_vital_before = df_patient_vital_before.dropna(subset=['o2sat'])
    
    
    df_patient_vital_after = df_vitalsign[(df_vitalsign['stay_id'] == row['ed_stay_id']) & 
                                    (df_vitalsign['charttime'] > row['90min'])]
    df_patient_vital_after = df_patient_vital_after.dropna(subset=['o2sat'])
    
    if len(df_patient_vital_after)==0: 
        severe_hypoxemia.append(-999)
    
    elif len(df_patient_vital_before)==0: 
        severe_hypoxemia.append(-999)
    
    elif any(df_patient_vital_before['o2sat']<=85): 
        severe_hypoxemia.append(-999)
    
    else:
        
        df_patient_vital_within = df_patient_vital_after[df_patient_vital_after['charttime']<= row['intime']+pd.Timedelta(days=1)]
        df_patient_vital_after = df_patient_vital_after[df_patient_vital_after['charttime']> row['intime']+pd.Timedelta(days=1)]
        
        if any(df_patient_vital_within['o2sat']<=85):
            severe_hypoxemia.append(1)

        elif not any(df_patient_vital_within['o2sat']<=85) and not any(df_patient_vital_after['o2sat']<=85):
            severe_hypoxemia.append(0)
            
        elif len(df_patient_vital_within)>0 and not any(df_patient_vital_within['o2sat']<=85) and any(df_patient_vital_after['o2sat']<=85): 
            severe_hypoxemia.append(0)
        
        elif len(df_patient_vital_within)==0 and not any(df_patient_vital_within['o2sat']<=85) and any(df_patient_vital_after['o2sat']<=85): 
            severe_hypoxemia.append(-999)
            
df_filtered_diags['severe_hypoxemia'] = severe_hypoxemia
decompensation_cols = ['severe_hypoxemia','ecmo','vasopressors','inotropes', 'mechanical_ventilation','cardiac_arrest']
df_features = df_filtered_diags.copy()

df_features.rename(columns={'Unnamed: 0': 'data'}, inplace=True)
decompensation_model_cols = decompensation_cols + icu_cols + mortality_cols
df_features['Deterioration_labels'] = df_features[decompensation_model_cols].values.tolist()
df_features.drop(columns=decompensation_model_cols, inplace=True)
df_features.drop(columns=['fold'], inplace=True)
labels_cols = ['Diagnoses_labels','Deterioration_labels']


# features

df_features['gender'] = df_features['gender'].apply(lambda x: 1 if x=='M' else 0)
def map_race(row):
    if 'WHITE' in row:
        return 'white'
    elif 'BLACK' in row:
        return 'black/african'
    elif 'HISPANIC' in row:
        return 'hispanic/latino'
    elif 'ASIAN' in row:
        return 'asian'
    else:
        return 'other'
    
df_features['race_mapped'] = df_features['race'].apply(map_race)

race_dummies = pd.get_dummies(df_features['race_mapped'], prefix='demographics_ethnicity')

df_features = pd.concat([df_features, race_dummies], axis=1)


# biometrics

omr = pd.read_csv('data/omr.csv.gz')
omr['result_value'] = pd.to_numeric(omr['result_value'], errors='coerce')
omr['chartdate'] = pd.to_datetime(omr['chartdate'])
omr = omr[omr['result_name'].isin(['BMI (kg/m2)','Height (Inches)','Weight (Lbs)'])]

omr = omr[omr['subject_id'].isin(df_features['subject_id'].unique())]
omr.dropna(subset=['result_value'], inplace=True)
omr.loc[omr['result_name'] == 'Height (Inches)', 'result_value'] *= 2.54
omr.loc[omr['result_name'] == 'Height (Inches)', 'result_name'] = 'Height (cm)'
omr.loc[omr['result_name'] == 'Weight (Lbs)', 'result_value'] *= 0.453592
omr.loc[omr['result_name'] == 'Weight (Lbs)', 'result_name'] = 'Weight (kg)'

conditions = [
    (omr['result_name'] == 'BMI (kg/m2)') & (omr['result_value'] > 100),
    (omr['result_name'] == 'Weight (kg)') & (omr['result_value'] > 400),
    (omr['result_name'] == 'Height (cm)') & (omr['result_value'] > 400),
    (omr['result_name'] == 'Weight (kg)') & (omr['result_value'] < 20),
    (omr['result_name'] == 'Height (cm)') & (omr['result_value'] < 60)
]

for condition in conditions:
    omr.loc[condition, 'result_value'] = np.nan
    
omr.dropna(subset=['result_value'], inplace=True)


out_bmi = []
out_weight = []
out_height = []

for _, row in df_features.iterrows(), total=len(df_features):
    patient = row['subject_id']
    intime = row['ecg_time']
    
    df_patient = omr[omr['subject_id'] == patient]
    df_patient_within = df_patient.loc[(df_patient['chartdate'] >= (intime - pd.Timedelta(days=30))) & 
                                       (df_patient['chartdate'] <= (intime + pd.Timedelta(days=30)))]
    
    if df_patient_within.empty:
        out_bmi.append(np.nan)
        out_weight.append(np.nan)
        out_height.append(np.nan)
    else:
        # Find the closest BMI to ecg_time
        bmi_rows = df_patient_within[df_patient_within['result_name'] == 'BMI (kg/m2)']
        if not bmi_rows.empty:
            closest_bmi = bmi_rows.iloc[(bmi_rows['chartdate'] - intime).abs().argsort()[:1]]['result_value'].values[0]
            out_bmi.append(closest_bmi)
        else:
            out_bmi.append(np.nan)
        
        # Find the closest Weight to ecg_time
        weight_rows = df_patient_within[df_patient_within['result_name'] == 'Weight (kg)']
        if not weight_rows.empty:
            closest_weight = weight_rows.iloc[(weight_rows['chartdate'] - intime).abs().argsort()[:1]]['result_value'].values[0]
            out_weight.append(closest_weight)
        else:
            out_weight.append(np.nan)
        
        # Find the closest Height to ecg_time
        height_rows = df_patient_within[df_patient_within['result_name'] == 'Height (cm)']
        if not height_rows.empty:
            closest_height = height_rows.iloc[(height_rows['chartdate'] - intime).abs().argsort()[:1]]['result_value'].values[0]
            out_height.append(closest_height)
        else:
            out_height.append(np.nan)
            
df_features['biometrics_bmi'] = out_bmi
df_features['biometrics_weight'] = out_weight
df_features['biometrics_height'] = out_height


# vital signs

df_vitalsign = pd.read_csv('data/vitalsign.csv.gz')
df_vitalsign = df_vitalsign[df_vitalsign['subject_id'].isin(df_features['subject_id'].unique())]
df_vitalsign['charttime'] = pd.to_datetime(df_vitalsign['charttime'])

df_vitalsign.loc[df_vitalsign['temperature'] < 50, 'temperature'] = np.nan # 53 minimum recorded
df_vitalsign.loc[df_vitalsign['temperature'] > 150, 'temperature'] = np.nan # 115.7 maximum recorded
df_vitalsign.loc[df_vitalsign['heartrate'] > 700, 'heartrate'] = np.nan # 600 maximum recorded
df_vitalsign.loc[df_vitalsign['resprate'] > 300, 'resprate'] = np.nan # normal 20, athletes 50
df_vitalsign.loc[df_vitalsign['o2sat'] > 100, 'o2sat'] = np.nan # can't be negative nor more than 100
df_vitalsign.loc[df_vitalsign['o2sat'] < 0, 'o2sat'] = np.nan # can't be negative nor more than 100
df_vitalsign.loc[df_vitalsign['dbp'] > 500, 'dbp'] = np.nan # max recorded 370
df_vitalsign.loc[df_vitalsign['sbp'] > 500, 'sbp'] = np.nan # max recorded 360

def fahrenheit_to_celsius(f):
    return (f - 32) * 5.0/9.0

df_vitalsign['temperature'] = df_vitalsign['temperature'].apply(fahrenheit_to_celsius)

new_df_vital = []

for stay_id in df_vitalsign['stay_id'].unique():

    df_vital_stay = df_vitalsign[df_vitalsign['stay_id'] == stay_id]
    df_intime = df_features[df_features['ed_stay_id'] == stay_id]
    
    if len(df_intime)>0:
    
        intime = df_intime['intime'].values[0]
        
        df_vital_stay_filtered = df_vital_stay[
        (df_vital_stay['charttime'] >= intime) &
        (df_vital_stay['charttime'] <= intime + pd.Timedelta(minutes=90))]
        
        if len(df_vital_stay_filtered)>0:
            new_df_vital.append(df_vital_stay_filtered)
            
df_vital = pd.concat(new_df_vital)

intime_dict = df_features.set_index('ed_stay_id')['intime'].to_dict()
df_vital['intime'] = df_vital['stay_id'].map(intime_dict)
df_vital['minutes_since_intime'] = (df_vital['charttime']-df_vital['intime']).dt.total_seconds() / 60

vp_cols = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp']

vital_features = []

for i, row in df_features.iterrows():
    df_vital_p = df_vital[df_vital['stay_id'] == row['ed_stay_id']]
    
    if len(df_vital_p) > 0:
        features = {}
        for col in vp_cols:
            vals = df_vital_p[col].dropna().values  # Drop NaNs for computation
            df_vital_pv = df_vital_p.dropna(subset=col)
            if len(vals) > 0:
                mean_val = np.mean(vals)
                median_val = np.median(vals)
                min_val = np.min(vals)
                max_val = np.max(vals)
                std_val = np.std(vals)
                first_val = vals[0]
                last_val = vals[-1]
                
                rate_change = (last_val - first_val) / first_val if first_val != 0 else np.nan
                
                X = df_vital_pv['minutes_since_intime'].values.reshape(-1, 1)
                y = vals.reshape(-1, 1)
                model = LinearRegression().fit(X, y)
                coeff = model.coef_[0][0]
                
                # Store computed features
                features[f"vitals_{col}_mean"] = mean_val
                features[f"vitals_{col}_median"] = median_val
                features[f"vitals_{col}_min"] = min_val
                features[f"vitals_{col}_max"] = max_val
                features[f"vitals_{col}_std"] = std_val
                features[f"vitals_{col}_first"] = first_val
                features[f"vitals_{col}_last"] = last_val
                features[f"vitals_{col}_rate_change"] = rate_change
                features[f"vitals_{col}_coeff"] = coeff
            else:
                # If no values, set all features to NaN
                features[f"vitals_{col}_mean"] = np.nan
                features[f"vitals_{col}_median"] = np.nan
                features[f"vitals_{col}_min"] = np.nan
                features[f"vitals_{col}_max"] = np.nan
                features[f"vitals_{col}_std"] = np.nan
                features[f"vitals_{col}_first"] = np.nan
                features[f"vitals_{col}_last"] = np.nan
                features[f"vitals_{col}_rate_change"] = np.nan
                features[f"vitals_{col}_coeff"] = np.nan
        
        # Append features to vital_features
        vital_features.append(features)
    
    else:
        # If no vital sign data for the stay_id, append NaN for all features
        features = {}
        for col in vp_cols:
            features[f"vitals_{col}_mean"] = np.nan
            features[f"vitals_{col}_median"] = np.nan
            features[f"vitals_{col}_min"] = np.nan
            features[f"vitals_{col}_max"] = np.nan
            features[f"vitals_{col}_std"] = np.nan
            features[f"vitals_{col}_first"] = np.nan
            features[f"vitals_{col}_last"] = np.nan
            features[f"vitals_{col}_rate_change"] = np.nan
            features[f"vitals_{col}_coeff"] = np.nan
            
        vital_features.append(features)
        
vital_features_df = pd.DataFrame(vital_features)
df_features.reset_index(inplace=True, drop=True)

df_features = pd.concat([df_features, vital_features_df], axis=1)

acuities = []

df_triage = pd.read_csv('data/triage.csv.gz')

for i,row in df_features.iterrows():
    row_acuity = df_triage[df_triage['stay_id']==row['ed_stay_id']]
    if len(row_acuity)>0:
        acu = row_acuity['acuity'].values[0]
        acuities.append(acu)
    else:
        acuities.append(np.nan)
        
df_features['vitals_acuity'] = acuities


df_labtitles = pd.read_csv('data/d_labitems.csv.gz', compression='gzip')
df_labevents = pd.read_csv('data/labevents.csv.gz', compression='gzip', low_memory=False)

df_labs = pd.merge(df_labevents, df_labtitles, on='itemid')
df_labevents = None

df_labs = df_labs[df_labs['subject_id'].isin(df_features['subject_id'].unique())]
df_labs['charttime'] = pd.to_datetime(df_labs['charttime'])

labs_to_keep = ['Absolute Basophil Count', 'Absolute Eosinophil Count',
       'Absolute Lymphocyte Count', 'Alanine Aminotransferase (ALT)',
       'Albumin', 'Alkaline Phosphatase',
       'Asparate Aminotransferase (AST)', 'Bands', 'Base Excess',
       'Basophils', 'Bicarbonate', 'Bilirubin, Direct',
       'Bilirubin, Total', 'C-Reactive Protein', 'Calcium, Total',
       'Carboxyhemoglobin', 'Chloride', 'Creatine Kinase (CK)',
       'Creatine Kinase, MB Isoenzyme', 'Creatinine', 'Eosinophils',
       'Fibrinogen, Functional', 'Free Calcium', 'Glucose', 'Hematocrit',
       'Hemoglobin', 'INR(PT)', 'Lactate', 'Lymphocytes', 'Magnesium',
       'Neutrophils', 'Oxygen Saturation',
       'PT', 'PTT', 'Phosphate', 'Platelet Count', 'Potassium', 'RDW',
       'Red Blood Cells', 'Sodium', 
        'Troponin T', 'Urea Nitrogen','White Blood Cells', 'pCO2', 'pH']

df_labs = df_labs[df_labs['label'].isin(labs_to_keep)]

fluid_counts = df_labs.groupby(['label', 'fluid']).size().reset_index(name='count')
idx = fluid_counts.groupby('label')['count'].idxmax()
top_fluid_counts = fluid_counts.loc[idx]
df_labs = df_labs.merge(top_fluid_counts[['label', 'fluid']], on=['label', 'fluid'])

df_labs.loc[(df_labs['label'] == 'Absolute Basophil Count') & (df_labs['valuenum'] > 20), 'valuenum'] = np.nan
df_labs.loc[(df_labs['label'] == 'Absolute Eosinophil Count') & (df_labs['valuenum'] > 20), 'valuenum'] = np.nan
df_labs.loc[(df_labs['label'] == 'Absolute Lymphocyte Count') & (df_labs['valuenum'] > 100), 'valuenum'] = np.nan

df_labs.loc[(df_labs['label'] == 'Alanine Aminotransferase (ALT)') & (df_labs['valuenum'] > 2000), 'valuenum'] = np.nan
df_labs.loc[(df_labs['label'] == 'Alkaline Phosphatase') & (df_labs['valuenum'] > 2000), 'valuenum'] = np.nan
df_labs.loc[(df_labs['label'] == 'Asparate Aminotransferase (AST)') & (df_labs['valuenum'] > 2000), 'valuenum'] = np.nan

df_labs.loc[(df_labs['label'] == 'Creatine Kinase (CK)') & (df_labs['valuenum'] > 2000), 'valuenum'] = np.nan
df_labs.loc[(df_labs['label'] == 'Glucose') & (df_labs['valuenum'] > 2000), 'valuenum'] = np.nan
df_labs.loc[(df_labs['label'] == 'Lactate') & (df_labs['valuenum'] > 2000), 'valuenum'] = np.nan

df_labs.loc[(df_labs['label'] == 'Platelet Count') & (df_labs['valuenum'] > 2000), 'valuenum'] = np.nan

fluid_counts = df_labs.groupby(['label', 'fluid']).size().reset_index(name='count')

idx = fluid_counts.groupby('label')['count'].idxmax()
top_fluid_counts = fluid_counts.loc[idx]
df_labs = df_labs.merge(top_fluid_counts[['label', 'fluid']], on=['label', 'fluid'])


lab = []

for i, row in df_features.iterrows():
    
    df_lb = df_labs[df_labs['subject_id'] == row['subject_id']]
    df_lb = df_lb[(df_lb['charttime'] >= row['intime']) & (df_lb['charttime'] <= row['90min'])]
    lab.append(df_lb)
    
df_labs = pd.concat(lab)
df_labs = df_labs[['subject_id','hadm_id','charttime','valuenum','valueuom','label','fluid']]

df_labs = df_labs.drop_duplicates()

lab_features = []

for i, row in df_features.iterrows():
    
    df_lb = df_labs[df_labs['subject_id'] == row['subject_id']]
    df_lb = df_lb[(df_lb['charttime'] >= row['intime']) & (df_lb['charttime'] <= row['90min'])]
    df_lb['minutes_since_intime'] = (df_lb['charttime']-row['intime']).dt.total_seconds() / 60
    
    assert not any(df_lb['minutes_since_intime']<0)
    assert not any(df_lb['minutes_since_intime']>90)
        
    if len(df_lb) > 0:
        features = {}
        for col in labs_to_keep:
            df_lab = df_lb[df_lb['label']==col]
            df_lab = df_lab.dropna(subset='valuenum') 
            
            
            vals = df_lab['valuenum'].values
            X = df_lab['minutes_since_intime'].values.reshape(-1, 1)
            y = vals.reshape(-1, 1)
            
            if len(vals) > 0:
                mean_val = np.mean(vals)
                median_val = np.median(vals)
                min_val = np.min(vals)
                max_val = np.max(vals)
                std_val = np.std(vals)
                first_val = vals[0]
                last_val = vals[-1]
                
                rate_change = (last_val - first_val) / first_val if first_val != 0 else np.nan
                
                model = LinearRegression().fit(X, y)
                coeff = model.coef_[0][0]
                
                # Store computed features
                features[f"labvalues_{col}_mean"] = mean_val
                features[f"labvalues_{col}_median"] = median_val
                features[f"labvalues_{col}_min"] = min_val
                features[f"labvalues_{col}_max"] = max_val
                features[f"labvalues_{col}_std"] = std_val
                features[f"labvalues_{col}_first"] = first_val
                features[f"labvalues_{col}_last"] = last_val
                features[f"labvalues_{col}_rate_change"] = rate_change
                features[f"labvalues_{col}_coeff"] = coeff
            else:
                # If no values, set all features to NaN
                features[f"labvalues_{col}_mean"] = np.nan
                features[f"labvalues_{col}_median"] = np.nan
                features[f"labvalues_{col}_min"] = np.nan
                features[f"labvalues_{col}_max"] = np.nan
                features[f"labvalues_{col}_std"] = np.nan
                features[f"labvalues_{col}_first"] = np.nan
                features[f"labvalues_{col}_last"] = np.nan
                features[f"labvalues_{col}_rate_change"] = np.nan
                features[f"labvalues_{col}_coeff"] = np.nan
        
        lab_features.append(features)
    
    else:
        features = {}
        for col in labs_to_keep:
            features[f"labvalues_{col}_mean"] = np.nan
            features[f"labvalues_{col}_median"] = np.nan
            features[f"labvalues_{col}_min"] = np.nan
            features[f"labvalues_{col}_max"] = np.nan
            features[f"labvalues_{col}_std"] = np.nan
            features[f"labvalues_{col}_first"] = np.nan
            features[f"labvalues_{col}_last"] = np.nan
            features[f"labvalues_{col}_rate_change"] = np.nan
            features[f"labvalues_{col}_coeff"] = np.nan
            
        lab_features.append(features)
        
lab_features_df = pd.DataFrame(lab_features)

df_features = pd.concat([df_features, lab_features_df], axis=1)
df_features.drop(columns='race_mapped', inplace=True)
df_features.rename(columns={'age':'demographics_age','gender':'demographics_gender'}, inplace=True)

features_cols = ['demographics_age','demographics_gender'] + df_features.columns[34:].tolist() # 470

columns_to_rename = [
    'data', 'file_name', 'study_id', 'subject_id', 'ecg_time', 
    'ed_stay_id', 'ed_hadm_id', 'hosp_hadm_id', 'ed_diag_ed', 
    'ed_diag_hosp', 'hosp_diag_hosp', 'all_diag_hosp', 'all_diag_all', 
    'anchor_year', 'anchor_age', 'dod', 'ecg_no_within_stay', 
    'ecg_taken_in_ed', 'ecg_taken_in_hosp', 'ecg_taken_in_ed_or_hosp', 
    'strat_fold', 'stay_id', 'intime', 'outtime', 'race', '90min', 
    'mortality_hours', 'mortality_days', 'hosp_dischtime', 'icu_time_hours'
]
renaming_dict = {col: f'general_{col}' for col in columns_to_rename}
df_features.rename(columns=renaming_dict, inplace=True)

lbls_diags = lbls_diags # 1414
lbls_det = decompensation_model_cols # 15

for i, label in enumerate(lbls_diags):
    df_features[f'diagnoses_{label}'] = df_features['Diagnoses_labels'].apply(lambda x: x[i])

for i, label in enumerate(lbls_det):
    df_features[f'deterioration_{label}'] = df_features['Deterioration_labels'].apply(lambda x: x[i])

df_features.drop(['Diagnoses_labels', 'Deterioration_labels'], axis=1, inplace=True)

columns_to_drop = [
    'general_hosp_hadm_id', 
    'general_hosp_diag_hosp',
    'general_all_diag_hosp',
    'general_all_diag_all',
    'general_ecg_taken_in_ed',
    'general_ecg_taken_in_hosp',
    'general_ecg_taken_in_ed_or_hosp',
    'general_stay_id'
]

df_diags = pd.read_csv('data/records_w_diag_icd10.csv')
df_diags['general_data'] = df_diags.index
df_diags['general_study_id'] = df_diags['study_id']

df_features['general_data'] = df_features['general_study_id'].map(df_diags.set_index('general_study_id')['general_data'])
df_features.columns = df_features.columns.str.replace(' ', '_', regex=False)
df_features.columns = df_features.columns.str.lower()
df_features.to_csv('data/memmap/mds_ed.csv', index=False)
