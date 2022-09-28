
import argparse
import pickle
import os
import os.path as op 
import importlib
from .utils import baselines_competitors_utils_pdm as bcu
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--nameParams', help='path to json file with input params', required=True)
parser.add_argument('--extractParams', help='path to json file with extraction params', required=True)
parser.add_argument('--expeco', help='path to json file with extraction params', required=True)
parser.add_argument('--nbCores', help='path to json file with extraction params', required=True)
args = parser.parse_args()
name_params = args.nameParams
extractParams = args.extractParams
exp = args.expeco

### Notice that we are going to score from 0 to n-w (economy and classifier -w issues)

# load params 
NB_CORES = int(args.nbCores)



GLOBAL_PARAMS_EXTRACTION = importlib.import_module('input.params_extraction.'+extractParams)
idx_subjects_train_classifiers = GLOBAL_PARAMS_EXTRACTION.idx_subjects_train_classifiers
idx_subjects_val_score = GLOBAL_PARAMS_EXTRACTION.idx_subjects_val_score
idx_subjects_ep = GLOBAL_PARAMS_EXTRACTION.idx_subjects_ep
idx_subjects_test = GLOBAL_PARAMS_EXTRACTION.idx_subjects_test
# load teain and ep
path_windows = op.join(op.dirname(op.realpath('__file__')), 'input', 'windows_h_generated')
PATH_FLOW = op.join(op.dirname(op.realpath('__file__')), 'models', 'preds_probas_flow', name_params, extractParams)

with open(op.join(PATH_FLOW, 'idx_hor.pkl'), 'rb') as outp:
    i_hor = pickle.load(outp)

# load test data

path_stream = op.join(op.dirname(op.realpath('__file__')), 'input', 'stream_generated')






path_extraction = op.join(op.dirname(op.realpath('__file__')), 'input', 'params_extraction')
path_params_exp = op.join(op.dirname(op.realpath('__file__')), 'models', 'eco_h', exp, 'params.pkl')
with open(path_params_exp, 'rb') as inp:
    params_exp = pickle.load(inp)

#GLOBAL_PARAMS_EXTRACTION  = getattr(__import__(input.params_extraction, fromlist=[extractParams]), extractParams) 
# from os import path as imported | package = "os" name = "path" | imported = getattr(__import__(package, fromlist=[name]), name)

path_params = op.join(op.dirname(op.realpath('__file__')), 'models', 'eco_h', exp, 'params.pkl')
with open(path_params, 'rb') as dtfile:
    params = pickle.load(dtfile)

s = params["s"]
w_s = params["w"]
H_s = params["H"]
misClassificationCosts = params_exp["misClassificationCosts"]
p = 0
timeCosts = params_exp["horizonCost"]
alphas = params_exp["alphas"]
extract_params = params_exp["extract"]

thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.40, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]



#-----------------------------------------------------------------------------------------------------------------------------------
# Baseline 1 and 0: use only one classifier (-w) for predicting when it exceeds a threshold-----------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------
path_preds = op.join(op.dirname(op.realpath('__file__')), 'models', 'preds_probas_flow', name_params, extract_params)
path_save =  op.join(op.dirname(op.realpath('__file__')), 'models', 'baselines')
try:
    os.mkdir(path_save)
except OSError:
    print ("Creation of the directory %s failed" % path_save)
else:
    print ("Successfully created the directory %s " % path_save)

path_save =  op.join(op.dirname(op.realpath('__file__')), 'models', 'baselines', name_params)
try:
    os.mkdir(path_save)
except OSError:
    print ("Creation of the directory %s failed" % path_save)
else:
    print ("Successfully created the directory %s " % path_save)

path_save =  op.join(op.dirname(op.realpath('__file__')), 'models', 'baselines', name_params, extract_params)



try:
    os.mkdir(path_save)
except OSError:
    print ("Creation of the directory %s failed" % path_save)
else:
    print ("Successfully created the directory %s " % path_save)

    
print('baseline most accurate and late')
for idx in idx_subjects_test:
    print(idx)
    use_type = 'test'
    with open(op.join(path_stream,'y_'+str(idx)+'_'+ name_params+'.pkl'),'rb') as outp:
        y_test = pickle.load(outp)
   
    results_baseline1_test = bcu.run_avgCost_computation(idx, 'baselinea', p, w_s, H_s, s, use_type, path_preds, name_params, misClassificationCosts, timeCosts, alphas, thresholds, NB_CORES, y_test, path_save, "opts",i_hor[0])
    results_baseline1_test_dict = bcu.from_parallel_result_to_dict(idx, 'baselinea', results_baseline1_test, w_s, misClassificationCosts, timeCosts, alphas, thresholds, path_save, use_type, name_params, extractParams, exp)


"""
print('baseline1')
for idx in idx_subjects_test:
    print(idx)
    use_type = 'test'
    with open(op.join(path_stream,'y_'+str(idx)+'_'+ name_params+'.pkl'),'rb') as outp:
        y_test = pickle.load(outp)
   
    results_baseline1_test = bcu.run_avgCost_computation(idx, 'baseline1', p, w_s, H_s, s, use_type, path_preds, name_params, misClassificationCosts, timeCosts, alphas, thresholds, NB_CORES, y_test, path_save, "opts")
    results_baseline1_test_dict = bcu.from_parallel_result_to_dict(idx, 'baseline1', results_baseline1_test, w_s, misClassificationCosts, timeCosts, alphas, thresholds, path_save, use_type, name_params, extractParams, exp)



print('baseline 0')
for idx in idx_subjects_test:
    print(idx)
    with open(op.join(path_stream,'y_'+str(idx)+'_'+ name_params+'.pkl'),'rb') as outp:
        y_test = pickle.load(outp)
    use_type = 'test'
    results_baseline1_test = bcu.run_avgCost_computation(idx,'baseline0',p,  w_s, H_s, s, use_type, path_preds, name_params, misClassificationCosts, timeCosts, alphas, thresholds, NB_CORES, y_test, path_save, "opts")
    results_baseline1_test_dict = bcu.from_parallel_result_to_dict(idx,'baseline0', results_baseline1_test, w_s, misClassificationCosts, timeCosts, alphas, thresholds, path_save, use_type, name_params, extractParams, exp)
    



#-----------------------------------------------------------------------------------------------------------------------------------
print("Baseline 2: use the collection of classifiers when it exceeds a threshold --------------------------------------------------------")
#-----------------------------------------------------------------------------------------------------------------------------------
path_preds = op.join(op.dirname(op.realpath('__file__')), 'models', 'preds_probas_flow', name_params, extract_params)

RESO = []
for idx in idx_subjects_val_score:
    print(idx)
    use_type = 'val'
    with open(op.join(path_stream,'y_'+str(idx)+'_'+ name_params+'.pkl'),'rb') as outp:
        y_val = pickle.load(outp)
    results_baseline2_val = bcu.run_avgCost_computation(idx, 'baseline2', p, w_s, H_s, s, use_type, path_preds, name_params, misClassificationCosts, timeCosts, alphas, thresholds, NB_CORES, y_val, path_save)
    results_baseline2_val_dict = bcu.from_parallel_result_to_dict(idx, 'baseline2', results_baseline2_val, w_s, misClassificationCosts, timeCosts, alphas, thresholds, path_save, use_type, name_params, extractParams, exp)
    RESO.append(results_baseline2_val_dict)





opts = bcu.optimal_threshold(RESO, thresholds, w_s, misClassificationCosts, alphas, timeCosts)
with open(op.join(path_save, 'results_optimalThreshold_baseline_2'+use_type+name_params+extractParams+exp+'.pkl'), 'wb') as outp:
    pickle.dump(opts, outp)
with open(op.join(path_save, 'results_optimalThreshold_baseline_2'+use_type+name_params+extractParams+exp+'.pkl'), 'rb') as outp:
    opts = pickle.load(outp)

for idx in idx_subjects_test:
    print('idx:',idx)
    use_type = 'test'
    with open(op.join(path_stream,'y_'+str(idx)+'_'+ name_params+'.pkl'),'rb') as outp:
        y_test = pickle.load(outp)
    results_baseline2_test = bcu.run_avgCost_computation(idx, 'baseline2', p, w_s, H_s, s, use_type, path_preds, name_params, misClassificationCosts, timeCosts, alphas, thresholds, NB_CORES, y_test, path_save, opts)
    results_baseline2_test_dict = bcu.from_parallel_result_to_dict(idx, 'baseline2', results_baseline2_test, w_s, misClassificationCosts, timeCosts, alphas, thresholds, path_save, use_type, name_params, extractParams, exp)

#-----------------------------------------------------------------------------------------------------------------------------------
print("Competitor 1: Mori decision function ---------------------------------------------------------------------------------------------")
#-----------------------------------------------------------------------------------------------------------------------------------

#thresholds = [-1,  -0.2, -0.4, -0.6,  -0.8, 0, 0.2, 0.4,  0.6,  0.8, 1]
thresholds = [-1, -0.5, 0, 0.5, 1]
thresholds_SR = []
for threshold_i in thresholds:
    for threshold_j in thresholds:
        for threshold_k in thresholds:
            thresholds_SR.append((threshold_i, threshold_j, threshold_k))
    

path_save =  op.join(op.dirname(op.realpath('__file__')), 'models', 'competitors')
try:
    os.mkdir(path_save)
except OSError:
    print ("Creation of the directory %s failed" % path_save)
else:
    print ("Successfully created the directory %s " % path_save)

path_save =  op.join(op.dirname(op.realpath('__file__')), 'models', 'competitors', name_params)
try:
    os.mkdir(path_save)
except OSError:
    print ("Creation of the directory %s failed" % path_save)
else:
    print ("Successfully created the directory %s " % path_save)
path_save =  op.join(op.dirname(op.realpath('__file__')), 'models', 'competitors', name_params, extract_params)
try:
    os.mkdir(path_save)
except OSError:
    print ("Creation of the directory %s failed" % path_save)
else:
    print ("Successfully created the directory %s " % path_save)


path_preds = op.join(op.dirname(op.realpath('__file__')), 'models', 'preds_probas_flow', name_params, extract_params)



RESO = []
for idx in idx_subjects_val_score:
    print(idx)
    use_type = 'val'
    with open(op.join(path_stream,'y_'+str(idx)+'_'+ name_params+'.pkl'),'rb') as outp:
        y_val = pickle.load(outp)
    results_competitor1_val = bcu.run_avgCost_computation(idx,'competitor1', p, w_s, H_s, s, use_type, path_preds, name_params, misClassificationCosts, timeCosts, alphas, thresholds_SR, NB_CORES, y_val, path_save)
    results_competitor1_val_dict = bcu.from_parallel_result_to_dict(idx,'competitor1', results_competitor1_val, w_s, misClassificationCosts, timeCosts, alphas, thresholds_SR, path_save, use_type, name_params, extractParams, exp)
    RESO.append(results_competitor1_val_dict)


opts = bcu.optimal_threshold(RESO, thresholds_SR, w_s, misClassificationCosts, alphas, timeCosts)
with open(op.join(path_save, 'results_optimalThreshold_competitor_1'+use_type+name_params+extractParams+exp+'.pkl'), 'wb') as outp:
    pickle.dump(opts, outp)

for idx in idx_subjects_test:
    print(idx)
    use_type = 'test'
    with open(op.join(path_stream,'y_'+str(idx)+'_'+ name_params+'.pkl'),'rb') as outp:
        y_test = pickle.load(outp)
    results_competitor1_test = bcu.run_avgCost_computation(idx,'competitor1', p, w_s, H_s, s, use_type, path_preds, name_params, misClassificationCosts, timeCosts, alphas, thresholds_SR, NB_CORES, y_test, path_save, opts)
    results_competitor1_test_dict = bcu.from_parallel_result_to_dict(idx,'competitor1', results_competitor1_test, w_s, misClassificationCosts, timeCosts, alphas, thresholds_SR, path_save, use_type, name_params, extractParams, exp)
"""
