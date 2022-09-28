

import argparse
from .utils import economyhUtils as ecoUtils
import pickle
import os.path as op 
from .utils import evaluationUtils as evalUtils
import numpy as np
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--exp', help='path to json file with input params', required=True)
parser.add_argument('--nbCores', help='path to json file with input params', required=True)

args = parser.parse_args()
experiment = args.exp


# load params 
NB_CORES = int(args.nbCores)
path_params = op.join(op.dirname(op.realpath('__file__')), 'models', 'eco_h', experiment, 'params.pkl')
with open(path_params, 'rb') as dtfile:
    params = pickle.load(dtfile)
misClassificationCosts = params["misClassificationCosts"]
timeCost = params["horizonCost"]
nbGroups = params["nbGroups"]
s = params["s"]
w_s = params["w"]
H_s = params["H"]
alphas = params["alphas"]
data = params["data"]
metric = params["metric"]
param_extract = params["extract"]


# load teain and ep
extraction_type = 'multiple_horizons'
use_type = 'train'
name_params = data 
path_windows = op.join(op.dirname(op.realpath('__file__')), 'input', 'windows_h_generated')
path_metrics = op.join(op.dirname(op.realpath('__file__')), 'models', 'eco_h', experiment, 'metrics')



 
ecoUtils.train_eco_models(params, NB_CORES, experiment)
print('----------------train_eco_model----------------')


optimal_nbGroups = ecoUtils.computeOptimal_nbGroups(params, NB_CORES, experiment, metric)
with open(path_metrics + 'optimal_nbGroups_'+experiment+'.pkl','wb') as outp:
    pickle.dump(optimal_nbGroups, outp)
#print('----------------computeOptimal_nbGroups----------------')
#with open(path_metrics + 'optimal_nbGroups_'+experiment+'.pkl','rb') as outp:
#    optimal_nbGroups = pickle.load(outp)

ys_preds = ecoUtils.test_eco_h_models(params, optimal_nbGroups, NB_CORES, experiment)
with open(path_metrics + 'predictions_stream_test_'+experiment+'.pkl','wb') as outp:
    pickle.dump(ys_preds, outp)
#print('----------------test_eco_h_models----------------')
#with open(path_metrics + 'predictions_stream_test_'+experiment+'.pkl','rb') as outp:
#    ys_preds = pickle.load(outp)

evaluation = ecoUtils.evaluateModels(params, ys_preds, NB_CORES)
with open(path_metrics + 'evaluation_stream_test_'+experiment+'.pkl','wb') as outp:
    pickle.dump(evaluation, outp)
print('----------------Evaluation----------------')
#with open(path_metrics + 'evaluation_stream_test_'+experiment+'.pkl','rb') as outp:
#    evaluation = pickle.load(outp)

ecoUtils.viz_results(params, evaluation, ys_preds, experiment)
print('----------------Vizu----------------')


