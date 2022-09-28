import argparse
import pickle 
import os.path as op
import pandas as pd
from joblib import Parallel, delayed
import importlib
import json
import os
import random
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import metrics as met
from .utils import pdmUtils as pdm
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--nameParams', help='path to json file with input params', required=True)
parser.add_argument('--extractParams', help='path to json file with input params', required=True)
parser.add_argument('--nbCores', help='path to json file with input params', required=True)
args = parser.parse_args()
GLOBAL_PARAMS_EXTRACTION = importlib.import_module('input.params_extraction.'+args.extractParams)




name_params = args.nameParams
extract_params = args.extractParams
nbCores = int(args.nbCores)
PATH_DATA = op.join(op.dirname(op.realpath('__file__')), 'input', 'stream_generated')
PATH_SAVE = op.join(op.dirname(op.realpath('__file__')), 'input', 'windows_h_generated')


idx_subjects_train_classifiers = GLOBAL_PARAMS_EXTRACTION.idx_subjects_train_classifiers
idx_subjects_val_score = GLOBAL_PARAMS_EXTRACTION.idx_subjects_val_score
idx_subjects_ep = GLOBAL_PARAMS_EXTRACTION.idx_subjects_ep
idx_subjects_test = GLOBAL_PARAMS_EXTRACTION.idx_subjects_test
w_s = GLOBAL_PARAMS_EXTRACTION.W
H_s = GLOBAL_PARAMS_EXTRACTION.H
s = GLOBAL_PARAMS_EXTRACTION.S


mis = GLOBAL_PARAMS_EXTRACTION.mis






PATH_PREDS_CLASSIFIERS = op.join(op.dirname(op.realpath('__file__')), 'models', 'preds_probas_classifiers_h', name_params)
try:
    os.mkdir(PATH_PREDS_CLASSIFIERS)
except OSError:
    print ("Creation of the directory %s failed" % PATH_PREDS_CLASSIFIERS)
else:
    print ("Successfully created the directory %s " % PATH_PREDS_CLASSIFIERS)

PATH_PREDS_CLASSIFIERS = op.join(op.dirname(op.realpath('__file__')), 'models', 'preds_probas_classifiers_h', name_params, extract_params)
print('hey params')
try:
    os.mkdir(PATH_PREDS_CLASSIFIERS)
except OSError:
    print ("Creation of the directory %s failed" % PATH_PREDS_CLASSIFIERS)
else:
    print ("Successfully created the directory %s " % PATH_PREDS_CLASSIFIERS)



mds = Parallel(n_jobs=nbCores)(delayed(pdm.train_xgb)((w, H, h, PATH_SAVE, name_params, extract_params, mis)) for w in w_s for H in H_s for h in range(-w, H, s))
models = {}
iso = {}
i=0
for w in w_s:
    for H in H_s:
        for t in range(-w, H, s):
            md, isso = mds[i]
            models[str(t)+str(w)+str(H)] = md
            iso[str(t)+str(w)+str(H)] = isso
            i+=1
  
with open(op.join(PATH_PREDS_CLASSIFIERS, 'models_' + extract_params  + '_' + name_params + '.pkl'), 'wb') as data_file:
    pickle.dump(models, data_file)            
with open(op.join(PATH_PREDS_CLASSIFIERS, 'iso_' + extract_params  + '_' + name_params + '.pkl'), 'wb') as data_file:
    pickle.dump(iso, data_file)   
   

with open(op.join(PATH_PREDS_CLASSIFIERS, 'models_' + extract_params  + '_' + name_params + '.pkl'), 'rb') as data_file:
    models = pickle.load(data_file)            
with open(op.join(PATH_PREDS_CLASSIFIERS, 'iso_' + extract_params  + '_' + name_params + '.pkl'), 'rb') as data_file:
    iso = pickle.load(data_file)  


Parallel(n_jobs=nbCores)(delayed(pdm.predict_lab_prob)((PATH_SAVE, h, w, H, name_params, extract_params, models[str(h)+str(w)+str(H)], PATH_PREDS_CLASSIFIERS)) for w in w_s for H in H_s for h in range(-w, H, s) )
   



















# preds for test val                        
PATH_FLOW = op.join(op.dirname(op.realpath('__file__')), 'models', 'preds_probas_flow', name_params)
try:
    os.mkdir(PATH_FLOW)
except OSError:
    print ("Creation of the directory %s failed" % PATH_FLOW)
else:
    print ("Successfully created the directory %s " % PATH_FLOW)

PATH_FLOW = op.join(op.dirname(op.realpath('__file__')), 'models', 'preds_probas_flow', name_params, extract_params)
try:
    os.mkdir(PATH_FLOW)
except OSError:
    print ("Creation of the directory %s failed" % PATH_FLOW)
else:
    print ("Successfully created the directory %s " % PATH_FLOW)



use_type = 'test'
Parallel(n_jobs=nbCores)(delayed(pdm.compute_preds_probas_flowpreds)((i, w, H, s, h, name_params, extract_params, models[str(h)+str(w)+str(H)], iso[str(h)+str(w)+str(H)], PATH_SAVE, PATH_DATA, PATH_FLOW, use_type)) for w in w_s for H in H_s for h in range(-w, H, s) for i in idx_subjects_test)



for w in w_s:
    for H in H_s:
        for i in idx_subjects_test:
            preds = []
            probas = []
            isoo = []
            for h in range(-w, H, s):
                with open(op.join(PATH_FLOW,'preds_'+str(h)+name_params+'_'+use_type + str(i) + str(w) + '_' + str(H) + '.pkl'),'rb') as outp:
                    pr = pickle.load(outp)
                preds.append(pr[0])
                
                with open(op.join(PATH_FLOW,'probas_'+str(h)+name_params+'_'+use_type + str(i) + str(w) + '_' + str(H) + '.pkl'),'rb') as outp:
                    pr = pickle.load(outp)
                probas.append(pr[0])
                
                with open(op.join(PATH_FLOW,'iso_'+str(h)+name_params+'_'+use_type + str(i) + str(w) + '_' + str(H) + '.pkl'),'rb') as outp:
                    pr = pickle.load(outp)
                isoo.append(pr[0])

            with open(op.join(PATH_FLOW,'preds_'+name_params+'_'+use_type + str(i) + str(w) + '_' + str(H) + '.pkl'),'wb') as outp:
                pickle.dump(preds, outp)
                
            with open(op.join(PATH_FLOW,'probas_'+name_params+'_'+use_type + str(i) + str(w) + '_' + str(H) + '.pkl'),'wb') as outp:
                pickle.dump(probas, outp)
                
            with open(op.join(PATH_FLOW,'iso_'+name_params+'_'+use_type + str(i) + str(w) + '_' + str(H) + '.pkl'),'wb') as outp:
                pickle.dump(isoo, outp)

use_type = 'val'
Parallel(n_jobs=nbCores)(delayed(pdm.compute_preds_probas_flowpreds)((i, w, H, s, h, name_params, extract_params, models[str(h)+str(w)+str(H)], iso[str(h)+str(w)+str(H)], PATH_SAVE, PATH_DATA, PATH_FLOW, use_type)) for w in w_s for H in H_s for h in range(-w, H, s) for i in idx_subjects_val_score)



for w in w_s:
    for H in H_s:
        for i in idx_subjects_val_score:
            preds = []
            probas = []
            isoo = []
            for h in range(-w, H, s):
                with open(op.join(PATH_FLOW,'preds_'+str(h)+name_params+'_'+use_type + str(i) + str(w) + '_' + str(H) + '.pkl'),'rb') as outp:
                    pr = pickle.load(outp)
                preds.append(pr[0])
                
                
                with open(op.join(PATH_FLOW,'probas_'+str(h)+name_params+'_'+use_type + str(i) + str(w) + '_' + str(H) + '.pkl'),'rb') as outp:
                    pr = pickle.load(outp)
                probas.append(pr[0])
                
                with open(op.join(PATH_FLOW,'iso_'+str(h)+name_params+'_'+use_type + str(i) + str(w) + '_' + str(H) + '.pkl'),'rb') as outp:
                    pr = pickle.load(outp)
                isoo.append(pr[0])
                
            with open(op.join(PATH_FLOW,'preds_'+name_params+'_'+use_type + str(i) + str(w) + '_' + str(H) + '.pkl'),'wb') as outp:
                pickle.dump(preds, outp)

            with open(op.join(PATH_FLOW,'probas_'+name_params+'_'+use_type + str(i) + str(w) + '_' + str(H) + '.pkl'),'wb') as outp:
                pickle.dump(probas, outp)
                
            with open(op.join(PATH_FLOW,'iso_'+name_params+'_'+use_type + str(i) + str(w) + '_' + str(H) + '.pkl'),'wb') as outp:
                pickle.dump(isoo, outp)