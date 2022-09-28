


import os.path as op
import pandas as pd
import pickle
import argparse

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--nameParams', help='path to json file with input params', required=True)
args = parser.parse_args()

PATH_DATA = op.join(op.dirname(op.realpath('__file__')), 'input', 'pdm')


telemetry_df = pd.read_csv(op.join(PATH_DATA, 'telemetry.csv'))
telemetry_df['datetime'] = pd.to_datetime(telemetry_df['datetime'], format='%m/%d/%Y %I:%M:%S %p')
failures_df = pd.read_csv(op.join(PATH_DATA, 'failures.csv'))
failures_df['datetime'] = pd.to_datetime(failures_df['datetime'], format='%m/%d/%Y %I:%M:%S %p')
errors_df = pd.read_csv(op.join(PATH_DATA, 'errors.csv') )
errors_df['datetime'] = pd.to_datetime(errors_df['datetime'], format='%m/%d/%Y %I:%M:%S %p')



failures = [False]*len(telemetry_df)
errors = {} 
is_error = [False]*len(telemetry_df)

for error in errors_df.errorID.unique():
    errors[error] = [False]*len(telemetry_df)

for machine_id in telemetry_df.machineID.unique():
    machine_telemetry = telemetry_df[telemetry_df.machineID==machine_id]
    
    for date_time in failures_df[failures_df.machineID == machine_id].datetime:
        for date_time_index in machine_telemetry.index[machine_telemetry.datetime==date_time]:                
            failures[date_time_index] = True
    
    for _, error_row  in errors_df[errors_df.machineID == machine_id].iterrows():
        date_time = error_row.datetime
        error_id = error_row.errorID
        for date_time_index in machine_telemetry.index[machine_telemetry.datetime==date_time]:                
            errors[error_id][date_time_index] = True
            is_error[date_time_index] = True


telemetry_df['failures'] = failures
telemetry_df['is_error'] = is_error

for error_col_key in errors.keys():
    telemetry_df[error_col_key] = errors[error_col_key]
    

with open(op.join(PATH_DATA,'data.pkl'),'wb') as outp:
    pickle.dump(telemetry_df, outp)
    
    
name_params = args.nameParams
  
PATH_SAVE_DATA = op.join(op.dirname(op.realpath('__file__')), 'input', 'stream_generated')
for i in range(1,101):
    
    df = telemetry_df[telemetry_df['machineID'] == i]
    
    
    y = df['failures'].values
    with open(op.join(PATH_SAVE_DATA,'y_'+str(i)+'_'+ name_params+'.pkl'),'wb') as outp:
        pickle.dump(y, outp)
    df = df.drop(['datetime', 'machineID', 'failures'], axis=1)
    with open(op.join(PATH_SAVE_DATA,'x_'+str(i)+'_'+ name_params+'.pkl'),'wb') as outp:
        pickle.dump(df.values, outp)

    
