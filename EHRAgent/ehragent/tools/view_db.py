from tabtools import *

def load_json(filename):
    with open(filename) as f:
        return json.load(f)

def view_csv_line_data(csv_filename):
    data = db_loader(csv_filename)
    data_lines = data.to_dict(orient='records')
    return data_lines

if __name__ == '__main__':

    # view data in csv
    # data = db_loader('admissions')
    # data_lines = data.to_dict(orient='records')
    # tmp = []

    # We can find the basic information of patient 366 in the patients database.
    patients_db = db_loader('patients')
    filtered_patients_db = data_filter(patients_db, 'SUBJECT_ID=366')
    patient_info = get_value(filtered_patients_db, 'ROW_ID, SUBJECT_ID, GENDER, DOB, DOD')

    # We can find the visiting information of patient 366 in the admissions database.
    admissions_db = LoadDB('admissions')
    filtered_admissions_db = FilterDB(admissions_db, 'SUBJECT_ID=366')
    admissions_info = GetValue(filtered_admissions_db, '*')

    # We can find the intensive care unit stay information of patient 366 in the icustays database.
    icustays_db = LoadDB('icustays')
    filtered_icustays_db = FilterDB(icustays_db, 'SUBJECT_ID=366')
    icustays_info = GetValue(filtered_icustays_db, '*')

    # We can find the transfer information of patient 366 in the transfers database.
    transfers_db = LoadDB('transfers')
    filtered_transfers_db = FilterDB(transfers_db, 'SUBJECT_ID=366')
    transfers_info = GetValue(filtered_transfers_db, '*')

    # We can find the cost information of patient 366 in the cost database.
    cost_db = LoadDB('cost')
    filtered_cost_db = FilterDB(cost_db, 'SUBJECT_ID=366')
    cost_info = GetValue(filtered_cost_db, '*')

    # We can find the charted events information of patient 366 in the chartevents database.
    chartevents_db = LoadDB('chartevents')
    filtered_chartevents_db = FilterDB(chartevents_db, 'SUBJECT_ID=366')
    chartevents_info = GetValue(filtered_chartevents_db, '*')

    # We can find the laboratory test results of patient 366 in the labevents database.
    labevents_db = LoadDB('labevents')
    filtered_labevents_db = FilterDB(labevents_db, 'SUBJECT_ID=366')
    labevents_info = GetValue(filtered_labevents_db, '*')

    # We can find the output measurements of patient 366 in the outputevents database.
    outputevents_db = LoadDB('outputevents')
    filtered_outputevents_db = FilterDB(outputevents_db, 'SUBJECT_ID=366')
    outputevents_info = GetValue(filtered_outputevents_db, '*')

    # We can find the prescription information of patient 366 in the prescriptions database.
    prescriptions_db = LoadDB('prescriptions')
    filtered_prescriptions_db = FilterDB(prescriptions_db, 'SUBJECT_ID=366')
    prescriptions_info = GetValue(filtered_prescriptions_db, '*')

    # We can find the procedure information of patient 366 in the procedures_icd database.
    procedures_icd_db = LoadDB('procedures_icd')
    filtered_procedures_icd_db = FilterDB(procedures_icd_db, 'SUBJECT_ID=366')
    procedures_icd_info = GetValue(filtered_procedures_icd_db, '*')

    # We can find the diagnosis information of patient 366 in the diagnoses_icd database.
    diagnoses_icd_db = LoadDB('diagnoses_icd')
    filtered_diagnoses_icd_db = FilterDB(diagnoses_icd_db, 'SUBJECT_ID=366')
    diagnoses_icd_info = GetValue(filtered_diagnoses_icd_db, '*')

    # We can find the microbiology test results of patient 366 in the microbiologyevents database.
    microbiologyevents_db = LoadDB('microbiologyevents')
    filtered_microbiologyevents_db = FilterDB(microbiologyevents_db, 'SUBJECT_ID=366')
    microbiologyevents_info = GetValue(filtered_microbiologyevents_db, '*')

    # We can find the input events of patient 366 in the inputevents_cv database.
    inputevents_cv_db = LoadDB('inputevents_cv')
    filtered_inputevents_cv_db = FilterDB(inputevents_cv_db, 'SUBJECT_ID=366')
    inputevents_cv_info = GetValue(filtered_inputevents_cv_db, '*')

    # Combine all the information
    answer = {
        'patient_info': patient_info,
        'admissions_info': admissions_info,
        'icustays_info': icustays_info,
        'transfers_info': transfers_info,
        'cost_info': cost_info,
        'chartevents_info': chartevents_info,
        'labevents_info': labevents_info,
        'outputevents_info': outputevents_info,
        'prescriptions_info': prescriptions_info,
        'procedures_icd_info': procedures_icd_info,
        'diagnoses_icd_info': diagnoses_icd_info,
        'microbiologyevents_info': microbiologyevents_info,
        'inputevents_cv_info': inputevents_cv_info
    }