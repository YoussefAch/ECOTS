
import argparse
import pickle 
import os.path as op
from joblib import Parallel, delayed
import importlib
from .utils import pdmUtils as pdm
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


Parallel(n_jobs=nbCores)(delayed(pdm.build_X_y)((i, PATH_DATA, PATH_SAVE, w, H, s, name_params, extract_params)) for w in w_s for H in H_s for i in idx_subjects_train_classifiers + idx_subjects_ep)

for w in w_s:
    for H in H_s:
        for h in range(-w, H, s):
            x_h = []
            y_h = []
            for i in idx_subjects_train_classifiers:
                
                with open(op.join(PATH_SAVE, 'X_h_subject_'+str(i)+'_'+name_params+str(h)+extract_params+str(w)+str(H)+'.pkl'),'rb') as outp:
                    feats = pickle.load(outp)
                with open(op.join(PATH_SAVE, 'y_h_subject_'+str(i)+'_'+name_params+str(h)+extract_params+str(w)+str(H)+'.pkl'),'rb') as outp:
                    labels = pickle.load(outp)
                    
                for e1,e2 in zip(feats,labels):
                    x_h.append(e1)
                    y_h.append(e2)
                
            with open(op.join(PATH_SAVE, 'X_h_subject_train'+name_params+str(h)+extract_params+str(w)+str(H)+'.pkl'),'wb') as outp:
                pickle.dump(x_h, outp)
            with open(op.join(PATH_SAVE, 'y_h_subject_train'+name_params+str(h)+extract_params+str(w)+str(H)+'.pkl'),'wb') as outp:
                pickle.dump(y_h, outp)
                
for w in w_s:
    for H in H_s:
        for h in range(-w, H, s):
            x_h = []
            y_h = []
            for i in idx_subjects_ep:
                
                with open(op.join(PATH_SAVE, 'X_h_subject_'+str(i)+'_'+name_params+str(h)+extract_params+str(w)+str(H)+'.pkl'),'rb') as outp:
                    feats = pickle.load(outp)
                with open(op.join(PATH_SAVE, 'y_h_subject_'+str(i)+'_'+name_params+str(h)+extract_params+str(w)+str(H)+'.pkl'),'rb') as outp:
                    labels = pickle.load(outp)
                    
                for e1,e2 in zip(feats,labels):
                    x_h.append(e1)
                    y_h.append(e2)
                
            with open(op.join(PATH_SAVE, 'X_h_subject_ep'+name_params+str(h)+extract_params+str(w)+str(H)+'.pkl'),'wb') as outp:
                pickle.dump(x_h, outp)
            with open(op.join(PATH_SAVE, 'y_h_subject_ep'+name_params+str(h)+extract_params+str(w)+str(H)+'.pkl'),'wb') as outp:
                pickle.dump(y_h, outp)              
                

Parallel(n_jobs=nbCores)(delayed(pdm.build_X_whole_stream)((idx, w, PATH_DATA, PATH_SAVE, name_params, extract_params)) for w in w_s for H in H_s for idx in idx_subjects_test + idx_subjects_val_score)