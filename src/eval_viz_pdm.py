import random 
import os.path as op
import pickle
import matplotlib.pyplot as plt
import argparse
import importlib
import os 
import os.path as op 
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

import sys 

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--nameParams', help='path to json file with input params', required=True)
parser.add_argument('--extractParams', help='path to json file with input params', required=True)
parser.add_argument('--nbCores', help='path to json file with input params', required=True)
args = parser.parse_args()
GLOBAL_PARAMS_EXTRACTION = importlib.import_module('input.params_extraction.'+args.extractParams)

import numpy as np


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


PATH_FLOW = op.join(op.dirname(op.realpath('__file__')), 'models', 'preds_probas_flow', name_params, extract_params)
import collections
ths = np.arange(0, 1, 0.1)


ths=[0.51]



def process(idx_subjects_test, PATH_FLOW, name_params, extract_params, w_s, H_s, s, th):
    
    
    print('hey', th)
    pres = []
    recs = []
    aucs = []
    for i in idx_subjects_test:
        print(i)
        metrics_test, random_values, preds_true_w_H_test = pdm.save_metrics_classifiers(i, PATH_FLOW, name_params, extract_params, 'test', w_s, H_s, s, th)
        #metrics_train, preds_true_w_H_train = save_metrics_classifiers(y_train, path_to_save, name_params, name_params_ex, 'train', w_s, H_s, s)
        
        
        
        ######################################## AUC curve ########################################
        
        metrics_auc = {}
        
        for w in w_s:
            for H in H_s:
                    for t in range(-w, H, s):
                        y_preds, y_true, y_rd = preds_true_w_H_test[str(w)+str(H)+str(t)+str(th)]      
                        fpr_ro, tpr_ro, thresholds = met.roc_curve(y_true, y_preds)
                        #print('t = ',t, collections.Counter(y_preds), collections.Counter(y_true))
                        metrics_auc[str(w)+str(H)+str(t)+'auc'+str(th)] = met.auc(fpr_ro, tpr_ro)
                        metrics_auc[str(w)+str(H)+str(t)+'pre'+str(th)] = precision_score(y_true, y_preds)
                        metrics_auc[str(w)+str(H)+str(t)+'rec'+str(th)] = recall_score(y_true, y_preds)
                        #print('t=', t, 'th=', th, 'auc=', metrics_auc[str(w)+str(H)+str(t)+'auc'], ' precision=',metrics_auc[str(w)+str(H)+str(t)+'pre'], ' recall=', metrics_auc[str(w)+str(H)+str(t)+'rec'])
                        fpr_ro, tpr_ro, thresholds = met.roc_curve(y_true, y_rd)
                        metrics_auc[str(w)+str(H)+'random'] = met.auc(fpr_ro, tpr_ro)
            
        path = op.join(op.dirname(op.realpath('__file__')), 'models', 'preds_probas_flow', name_params, 'metrics_auc_classifiers_'+str(i)+extract_params+'.pkl')
        with open(path, 'wb') as inp:
            pickle.dump(metrics_auc, inp)
            
        aucs.append([metrics_auc[str(w)+str(H)+str(h)+'auc'+str(th)] for h in range(-w, H, s)])
        pres.append([metrics_auc[str(w)+str(H)+str(h)+'pre'+str(th)] for h in range(-w, H, s)])
        recs.append([metrics_auc[str(w)+str(H)+str(h)+'rec'+str(th)] for h in range(-w, H, s)])
    
        """plt.figure(figsize=(20,10))
        c=0
        for w in w_s:
            plt.plot([str(h) for h in range(-w, H, s)],
                        [metrics_auc[str(w)+str(H)+str(h)+'auc'+str(th)] for h in range(-w, H, s)],
                        marker='o', label='auc', color='blue')
            plt.plot([str(h) for h in range(-w, H, s)],
                        [metrics_auc[str(w)+str(H)+str(h)+'pre'+str(th)] for h in range(-w, H, s)],
                        marker='o', label='precision', color='yellow')
            plt.plot([str(h) for h in range(-w, H, s)],
                        [metrics_auc[str(w)+str(H)+str(h)+'rec'+str(th)] for h in range(-w, H, s)],
                        marker='o', label='recall', color='red')
        c+=1"
        
        
            
            
        plt.plot([str(h) for h in range(-w, H, s)],
                     [0.5 for _ in range(-w, H, s)], 
                    color='green', marker='o', label='Random')
        plt.ylim([-0.1, 1.1])
        plt.xticks(rotation=45)
        plt.title('AUC ')
        plt.grid()
        plt.xlabel('horizon')
        plt.legend()
        path = op.join(op.dirname(op.realpath('__file__')), 'plots', 'pdm', 'Metrics_classifiers_auc'+name_params+extract_params+'H'+str(H)+str(w)+str(i)+'.png')
        plt.savefig(path)"""
    
    
    # avg auc
    ac = []
    for j in range(len(aucs[0])):
        e = 0
        for i in range(len(aucs)):
            e+= aucs[i][j]
        e = e/len(aucs)
        ac.append(e)
        
    pr = []
    for j in range(len(pres[0])):
        e = 0
        for i in range(len(pres)):
            e+= pres[i][j]
        e = e/len(pres)
        pr.append(e)
    
        
    rec = []
    for j in range(len(recs[0])):
        e = 0
        for i in range(len(recs)):
            e+= recs[i][j]
        e = e/len(recs)
        rec.append(e)
        
    plt.figure(figsize=(10,5))
    c=0
    xt = []
    inn = True
    for i in range(-w, H, 1):
        if inn:
            xt.append(str(i))
            inn = False
        else:
            xt.append(" ")
            inn = True
        
    plt.xticks(range(-w, H, 1), xt, rotation=90, fontsize=12)
    for w in w_s:
        plt.plot([h for h in range(-w, H, s)],
                 ac,
                 marker='o', label='auc', color='blue')
        """plt.plot([str(h) for h in range(-w, H, s)],
                 pr,
                 marker='o', label='precision', color='yellow')
        plt.plot([str(h) for h in range(-w, H, s)],
                 rec,
                 marker='o', label='recall', color='red')"""
    c+=1
    """plt.plot([str(h) for h in range(-w, H, s)],
             [0.5 for _ in range(-w, H, s)], 
              color='green', marker='o', label='Random')"""
    plt.ylim([0.4, 1])
    

    #plt.grid()
    plt.xlabel('Horizon', fontsize=16)
    plt.ylabel('AUC',fontsize=16)
    #plt.legend()
    try:
        os.mkdir(op.join(op.dirname(op.realpath('__file__')), 'plots', 'pdm', 'auc_'+extract_params))
    except OSError:
        print ("Creation of the directoryfailed")
    else:
        print ("Successfully created the directory ")

    path = op.join(op.dirname(op.realpath('__file__')), 'plots', 'pdm', 'auc_'+extract_params, str(th)+'.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    
    mx = -sys.maxsize
    hor = 0
    for ii,e in enumerate(ac):
        if e > mx:
            mx = e
            hor = ii
    
    print(hor)
    return hor

i_hor = Parallel(n_jobs=nbCores)(delayed(process)(idx_subjects_test, PATH_FLOW, name_params, extract_params, w_s, H_s, s, th) for th in ths)

with open(op.join(PATH_FLOW, 'idx_hor.pkl'), 'wb') as outp:
    pickle.dump(i_hor, outp)
    
    


