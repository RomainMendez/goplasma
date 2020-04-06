import numpy as np
import pandas as pd
import os
import time
import sys
import pickle

MODEL_FILE = 'RF_model_trained.sav'
MODEL      = pickle.load(open(MODEL_FILE, 'rb'))

SCORE_MAX_NO_SYMPTOMS = 50
SCORE_WORST_CASE      = 10

work_env_mapping = { 'healthcare' : 0, 'close_contact' : 1, 'regular_contact' : 2, 'no_contact' : 3 }

worst_case = pd.Series( {
    'generally_ill' : 1, # Q1
    'resp_cough'    : 1, # Q2.1
    'resp_throat'   : 1, # Q2.2
    'resp_breath'   : 1, # Q2.3
    'taste_loss'    : 1, # Q3
    'sympt_fever'   : 1, # Q4.1
    'sympt_sens_fev': 1, # Q4.2
    'sympt_musc'    : 1, # Q4.3
    'sympt_head'    : 1, # Q4.4
    'work_env'      : 3, # Q6
    'proxim_sympt'  : 1, # Q8.1
    'proxim_case'   : 1, # Q8.2
    'contact_sympt' : 1, # Q9.1
    'contact_case'  : 1, # Q9.2
} ) 

no_symptoms = pd.Series( {
    'resp_cough'    : 1, # Q2.1
    'resp_throat'   : 1, # Q2.2
    'resp_breath'   : 1, # Q2.3
    'taste_loss'    : 1, # Q3
    'sympt_fever'   : 1, # Q4.1
    'sympt_sens_fev': 1, # Q4.2
    'sympt_musc'    : 1, # Q4.3
    'sympt_head'    : 1, # Q4.4
} ) 

def validate_data(sample):
# adjust hidden answers to defaults:
    if sample.generally_ill == 1:
        sample['illness_level'] = -1
    if sample.work_env == 3:
        sample['work_protec'] = -1
        sample['work_exposure'] = -1
        
    return sample

def convert_input(args, result_dict):
    for key, arg in args.items():
        if arg == 'yes':
            result_dict[key] = 0
        elif arg == 'no':
            result_dict[key] = 1
        elif arg == 'unknown':
            result_dict[key] = -1
        elif key == 'work_env':
            result_dict[key] = work_env_mapping[arg]
        elif key == 'illness_level':
            result_dict[key] = int(arg)
        else:
            print('DATA UNKNOWN:', key, arg)
        print(key, arg)
        
    return validate_data( pd.Series(result_dict) )

def initialize_dict():
    return {
        'generally_ill' : 1, # Q1
        'illness_level' : 0, # NEW
        'resp_cough'    : 1, # Q2.1
        'resp_throat'   : 1, # Q2.2
        'resp_breath'   : 1, # Q2.3
        'taste_loss'    : 1, # Q3
        'sympt_fever'   : 1, # Q4.1
        'sympt_sens_fev': 1, # Q4.2
        'sympt_musc'    : 1, # Q4.3
        'sympt_head'    : 1, # Q4.4
        'date_symptoms' : 0, # Q5
        'work_env'      : 3, # Q6
        'work_protec'   : 1, # Q7.a
        'work_exposure' : 1, # Q7.b
        'proxim_sympt'  : 1, # Q8.1
        'proxim_case'   : 1, # Q8.2
        'contact_sympt' : 1, # Q9.1
        'contact_case'  : 1, # Q9.2
    } 

def predict(in_data):
# Function for predicting one sample at a time (passed as pandas series)    
    features = in_data.values.reshape(1, -1)[:,2:]
    
    print('Predicting for: ', features)
    prediction = MODEL.predict(features)
    pred_proba = MODEL.predict_proba(features)
    
    return prediction[0], pred_proba[0]

def score_to_percentage(score, probabilities = None):
    p0    = [20, 40, 60, 80]
    p_cls = [35, 50, 55, 60]
    
    if probabilities is None:
        return score * 20
    
    cls_idx = int(score) - 1
    highest_proba = probabilities[cls_idx] * 100
    cls_2nd = np.argsort(probabilities)[-2]
    
    if highest_proba >= p_cls[cls_idx]:
        return p0[cls_idx]
    
    else:
        return (p0[cls_idx] + p0[cls_2nd]) / 2

def scorer(args):
    print('ENTERING SCORER')
        
    results = convert_input( args, initialize_dict() )
    
    if (results[worst_case.index] == worst_case).all(): # hard-code the absolute worst case
        print('Detected no symptoms; returning')
        return int(SCORE_WORST_CASE / 10)
    
    prediction, probabilities = predict(results)
    score   = score_to_percentage(prediction, probabilities)
    
    if (results[no_symptoms.index] == no_symptoms).all():
        print('Capped at %d' %SCORE_MAX_NO_SYMPTOMS)
        score = min(score, SCORE_MAX_NO_SYMPTOMS)
        
    print('LEAVING SCORER')
    print('Predicted score: %d' %score)
    return int(score/10)









