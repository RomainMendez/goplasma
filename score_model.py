import numpy as np
import pandas as pd
import os
import time
import sys
import pickle

MODEL_FILE = 'RF_model_trained.sav'
MODEL      = pickle.load(open(MODEL_FILE, 'rb'))

work_env_mapping = { 'healthcare' : 0, 'close_contact' : 1, 'regular_contact' : 2, 'no_contact' : 3 }

def convert_input(args, result_dict):
    for key, arg in args.items():
        if arg == 'yes':
            result_dict[key] = 0
        elif arg == 'no':
            result_dict[key] = 1
        elif arg == 'unknown':
            result_dict[key] = 2
        elif key == 'work_env':
            result_dict[key] = work_env_mapping[arg]
        elif key == 'illness_level':
            result_dict[key] = int(arg)
        else:
            print('DATA UNKNOWN:', key, arg)
        print(key, arg)
        
    return pd.Series(result_dict)

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

def predict(features):
# Function for predicting one sample at a time (passed as pandas series)

    print('Predicting for: ', features)
    prediction = MODEL.predict(features.values.reshape(1, -1))
    print(prediction)
    
    return prediction[0]

def score_to_percentage(score):
    return score * 20

def scorer(args):
    print('ENTERING SCORER')
        
    results = convert_input( args, initialize_dict() )
    score   = predict(results)
        
    print('LEAVING SCORER')
    print('Predicted score: %d' %score)
    return score_to_percentage(score)










