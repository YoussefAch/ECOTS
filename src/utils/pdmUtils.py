import statistics
import pickle 
import os.path as op
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedShuffleSplit 
from collections import Counter
import random
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import metrics as met
from sklearn.calibration import CalibratedClassifierCV



def extract_all_windows_from_stream(flow_x, w):
    
    # extracted windows (real values and class)
    extracted_w = []
    
    n = len(flow_x)
    for target_y in range(n-w+1):
        extracted_w.append(flow_x[target_y:target_y+w])
    

    return extracted_w   
    
import collections
def extract_windows_different_horizons_targetFocus_MTS(idxs_failures, flow_x, flow_y, w, H, s):
    
    
    
    extracted_w = {h:[] for h in range(-w, H, s)}
    labels_w =  {h:[] for h in range(-w, H, s)}
    n = len(flow_x)
    
    # first extract all failures
    for failure_idx in idxs_failures:
        e_x = []
        e_y = []
        for h in range(-w, H, s):
            if len(flow_x[failure_idx-h-w:failure_idx-h]) == w:
                e_x.append(flow_x[failure_idx-h-w:failure_idx-h])
                e_y.append(flow_y[failure_idx])       
            else:
                break
        if len(e_y) == len(range(-w, H, s)):
            for a,b,h in zip(e_x,e_y,range(-w, H, s)):
                extracted_w[h].append(a)
                labels_w[h].append(b)
    
    # extract normal windows
    
    
    
    # extracted windows (real values and class)
    print('HEY: ', len(idxs_failures))
    for target_y in range(w+H-1, n-w, w+H):
        
        if len(idxs_failures) > 0:
            if min(target_y - np.array(idxs_failures)) >= w+H+1:
                for h in range(-w, H, s):
                    extracted_w[h].append(flow_x[target_y-h-w:target_y-h])
                    labels_w[h].append(flow_y[target_y])
        else:
            for h in range(-w, H, s):
                extracted_w[h].append(flow_x[target_y-h-w:target_y-h])
                labels_w[h].append(flow_y[target_y])
    
    """fails_in = []
    for target_y in range(w+H-1, n-w, w+H):
            if flow_y[target_y]==1:
                fails_in.append(target_y)
            for h in range(-w, H, s):
                extracted_w[h].append(flow_x[target_y-h-w:target_y-h])
                labels_w[h].append(flow_y[target_y])
                
    
               
    initial_dist = collections.Counter(flow_y)[1]/collections.Counter(flow_y)[0]
    
    i=0
    while collections.Counter(labels_w[0])[1]/collections.Counter(labels_w[0])[0] < initial_dist:
        if idxs_failures[i] not in fails_in:
            failure_idx = idxs_failures[i]
            e_x = []
            e_y = []
            for h in range(-w, H, s):
                if len(flow_x[failure_idx-h-w:failure_idx-h]) == w:
                    e_x.append(flow_x[failure_idx-h-w:failure_idx-h])
                    e_y.append(flow_y[failure_idx])       
                else:
                    break
            if len(e_y) == len(range(-w, H, s)):
                for a,b,h in zip(e_x,e_y,range(-w, H, s)):
                    extracted_w[h].append(a)
                    labels_w[h].append(b)
        i += 1
        if len(idxs_failures)==i:
            break"""
            
            
    
    return extracted_w, labels_w

def build_X_whole_stream(arguments):
    
    i, w, PATH_DATA, PATH_SAVE, name_params, extract_params = arguments
    
    with open(op.join(PATH_DATA,'x_'+str(i)+'_'+ name_params+'.pkl'),'rb') as outp:
        flow_x = pickle.load(outp)
        

    n = len(flow_x)
    extracted_w = extract_all_windows_from_stream(flow_x, w)
    assert len(extracted_w) == n-w+1
    
    

    feats = []
    for l,window in enumerate(extracted_w):
        features = []
        for k in range(4):
            col = window[:,k]
            features.append(max(col))
            features.append(min(col))
            features.append(statistics.mean(col))
            features.append(statistics.pstdev(col))
        for k in range(4,10):
            col = window[:,k]
            features.append(sum(col))
        feats.append(features)
        
        
    with open(op.join(PATH_SAVE, 'X_whole_subject_'+str(i)+'_'+name_params+extract_params+str(w)+'.pkl'),'wb') as outp:
        pickle.dump(feats, outp)

def build_X_y(argss):
    
    
    idx, PATH_DATA, PATH_SAVE, w, H, s, name_params, extract_params=argss
    
    with open(op.join(PATH_DATA,'x_'+str(idx)+'_'+ name_params+'.pkl'),'rb') as outp:
        flow_x = pickle.load(outp)
    with open(op.join(PATH_DATA,'y_'+str(idx)+'_'+ name_params+'.pkl'),'rb') as outp:
        flow_y = pickle.load(outp)
        
    n = len(flow_x)
    idxs_failures = []
    for i in range(n):
        if flow_y[i]:
            idxs_failures.append(i)
            
    extracted_w, labels_w = extract_windows_different_horizons_targetFocus_MTS(idxs_failures, flow_x, flow_y, w, H, s)

    
    with open(op.join(PATH_SAVE,'windows_'+str(idx)+str(w)+str(H)+'.pkl'),'wb') as outp:
        pickle.dump(extracted_w, outp)
            
    with open(op.join(PATH_SAVE,'labels_'+str(idx)+str(w)+str(H)+'.pkl'),'wb') as outp:
        pickle.dump(labels_w, outp)
        
        
    
    
    # extract features
    for h in range(-w, H, s):
        feats = []
        for l,window in enumerate(extracted_w[h]):

            features = []
            for i in range(4):
                col = window[:,i]
                features.append(max(col))
                features.append(min(col))
                features.append(statistics.mean(col))
                features.append(statistics.pstdev(col))
            for i in range(4,10):
                col = window[:,i]
                features.append(sum(col))
            feats.append(features)
        
        with open(op.join(PATH_SAVE, 'X_h_subject_'+str(idx)+'_'+name_params+str(h)+extract_params+str(w)+str(H)+'.pkl'),'wb') as outp:
            pickle.dump(feats, outp)
        with open(op.join(PATH_SAVE, 'y_h_subject_'+str(idx)+'_'+name_params+str(h)+extract_params+str(w)+str(H)+'.pkl'),'wb') as outp:
            pickle.dump(labels_w[h], outp)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

def save_metrics_classifiers(i, path_to_save, name_params, name_params_ex, use_type, w_s, H_s, s, th):

    preds_true_w_H = {}
    for w in w_s:
        for H in H_s:
            # zone à prédire par tous les classifieurs 
            
            
            
                preds = []
                for h in range(-w, H, s):
                    with open(op.join(path_to_save,'probas_'+str(h)+name_params+'_'+ 'test'+str(i) + str(w) + '_' + str(H) + '.pkl'),'rb') as outp:
                        prob = pickle.load(outp)
                    
                    prd = list(map(lambda x: 1 if x>=th else 0, prob[0]))
                    preds.append(prd)
                    
            
            
            
            

                
                PATH_DATA = op.join(op.dirname(op.realpath('__file__')), 'input', 'stream_generated')
                with open(op.join(PATH_DATA,'y_'+str(i)+'_'+ name_params+'.pkl'),'rb') as outp:
                    y_test = pickle.load(outp)
                
                n = len(y_test)
                y_true = y_test[w+H-1:n-w+1]
                    
    
                for k,t in enumerate(range(-w, H, s)): ##########################
                    
                    
                    y_preds = preds[k]
                    y_preds = list(map(int, y_preds))
                    y_pred_random_model=[]
                    for _ in range(len(y_true)):
                        num = random.randint(0,1)
                        y_pred_random_model.append(num)
                    
                    assert len(y_preds)==len(y_true)
                    preds_true_w_H[str(w)+str(H)+str(t)+str(th)] = [y_preds, y_true, y_pred_random_model]

    metrics = {}
    metrics_random = {}
    for w in w_s:
        for H in H_s:

                for t in range(-w, H, s):
                    y_preds, y_true, y_random = preds_true_w_H[str(w)+str(H)+str(t)+str(th)]        
                    metrics[str(w)+str(H)+str(t)+str(th)]  = compute_metrics(np.array(y_true), np.array(y_preds))   
                    metrics_random[str(w)+str(H)+str(t)+str(th)]  = compute_metrics(np.array(y_true), np.array(y_random))   

    path = op.join(op.dirname(op.realpath('__file__')), 'models', 'preds_probas_flow', name_params, 'metrics_classifiers_'+str(i)+use_type+name_params_ex+'.pkl')
    with open(path, 'wb') as inp:
        pickle.dump(metrics, inp)
        
    path = op.join(op.dirname(op.realpath('__file__')), 'models', 'preds_probas_flow', name_params, 'random_classifier_metrics'+str(i)+name_params_ex+'.pkl')
    with open(path, 'wb') as inp:
        pickle.dump(metrics_random, inp)

    path = op.join(op.dirname(op.realpath('__file__')), 'models', 'preds_probas_flow', name_params, 'preds_true_w_H_'+str(i)+use_type+name_params_ex+'.pkl')
    with open(path, 'wb') as inp:
        pickle.dump(preds_true_w_H, inp)
    
    return metrics, metrics_random, preds_true_w_H










def save_metrics_random(i,y_test, name_params, name_params_ex, w_s, H_s):
    # random model
    n = len(y_test)
    metrics={}
    random_values = {}
    for w in w_s:
        for H in H_s:
            y_true = y_test[w+H-1:n-w+1]
            taille = len(y_test[w+H-1:n-w+1])
            y_pred_random_model = []
            for i in range(taille):
                num = random.randint(0,1)
                y_pred_random_model.append(num)
            random_values[str(w)+str(H)] = y_pred_random_model
            metrics[str(w)+str(H)]  = compute_metrics(np.array(y_true), np.array(y_pred_random_model))   

    
    path = op.join(op.dirname(op.realpath('__file__')), 'models', 'preds_probas_flow', name_params, 'random_classifier_'+str(i)+name_params_ex+'.pkl')
    with open(path, 'wb') as inp:
        pickle.dump(random_values, inp)

    path = op.join(op.dirname(op.realpath('__file__')), 'models', 'preds_probas_flow', name_params, 'random_classifier_metrics'+str(i)+name_params_ex+'.pkl')
    with open(path, 'wb') as inp:
        pickle.dump(metrics, inp)

    return metrics, random_values





def compute_metrics(values_real, values_pred):
    recall_classic = recall_score(values_real, values_pred)
    precision_classic = precision_score(values_real, values_pred)
    f1_classic = f1_score(values_real, values_pred)

    """flat_metric = TSMetric(metric_option="time-series", alpha_r=0.1, cardinality="reciprocal", bias_p="flat", bias_r="flat")
    precision_flat, recall_flat, f1_flat = flat_metric.score(values_real, values_pred)

    front_metric = TSMetric(metric_option="time-series", alpha_r=0.1, cardinality="reciprocal", bias_p="flat", bias_r="front")
    precision_front, recall_front, f1_front = front_metric.score(values_real, values_pred)

    middle_metric = TSMetric(metric_option="time-series", alpha_r=0.1, cardinality="reciprocal", bias_p="flat", bias_r="middle")
    precision_middle, recall_middle, f1_middle = middle_metric.score(values_real, values_pred)

    back_metric = TSMetric(metric_option="time-series", alpha_r=0.1, cardinality="reciprocal", bias_p="flat", bias_r="back")
    precision_back, recall_back, f1_back = back_metric.score(values_real, values_pred)

    results = {
        "recall_classic":recall_classic,
        "precision_classic":precision_classic,
        "f1_classic":f1_classic,
        "precision_flat":precision_flat,
        "recall_flat":recall_flat,
        "f1_flat":f1_flat,
        "precision_front":precision_front,
        "recall_front":recall_front,
        "f1_front":f1_front,
        "precision_middle":precision_middle,
        "recall_middle":recall_middle,
        "f1_middle":f1_middle,
        "precision_back":precision_back,
        "recall_back":recall_back,
        "f1_back":f1_back
    }"""
    
    return recall_classic, precision_classic, f1_classic











import warnings

warnings.filterwarnings("ignore")


from sklearn.linear_model import LogisticRegression 

import sys
    

import collections

def train_xgb(argss):
    w, H, h, PATH_SAVE, name_params, extract_params, mis = argss
    
    with open(op.join(PATH_SAVE, 'X_h_subject_train'+name_params+str(h)+extract_params+str(w)+str(H)+'.pkl'),'rb') as outp:
        x_h = pickle.load(outp)
    with open(op.join(PATH_SAVE, 'y_h_subject_train'+name_params+str(h)+extract_params+str(w)+str(H)+'.pkl'),'rb') as outp:
        y_h = pickle.load(outp)
    
    X = pd.DataFrame([x_h[k] for k in range(len(x_h))])
    y = pd.DataFrame([y_h[k] for k in range(len(x_h))])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=0)
    for train_idx, val_idx in sss.split(x_h, y_h):
        x_train = [x_h[k] for k in train_idx]
        df_train = pd.DataFrame(x_train)
        y_train = [y_h[k] for k in train_idx]
        x_val = [x_h[k] for k in val_idx]
        df_val = pd.DataFrame(x_val)
        y_val = [y_h[k] for k in val_idx]
             
    """print('YVAAAAAAAAAAAAAAAL::::', collections.Counter(y_val)[1]/collections.Counter(y_val)[0])
    FP_c = mis[0][1]
    FN_c = mis[1][0]
    def total_cost_eval(y_pred, y_true):
        labels = np.array(y_true)
        
        t = np.arange(0, 1, 0.005)
        f = np.repeat(0, 200)
        results = np.vstack([t, f]).T
        real_neg = (labels == 0)
        real_pos = (labels == 1)
        # assuming labels only containing 0's and 1's
        n_pos_examples = sum(labels)
        if n_pos_examples == 0:
            raise ValueError("labels not containing positive examples")
    
        for i in range(200):
            
            
            pred_indexes = (y_pred >= results[i, 0])
            pred_not_indexes = (y_pred < results[i, 0])
            
            
            
            FP = sum(real_neg & pred_indexes)
            FN = sum(real_pos & pred_not_indexes)
            
            
            results[i, 1] = (FP * FP_c + FN * FN_c) / len(labels)
        
        return min(results[:, 1])
    
    params_grid = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5, 10] 
    }

    best_params = []
    bestcost = sys.maxsize 
    import math
    for min_child_weight in params_grid['min_child_weight']:
        
        for gamma in params_grid['gamma']:
            for subsample in params_grid['subsample']:
                for colsample_bytree in params_grid['colsample_bytree']:
                    for max_depth in params_grid['max_depth']:
                        
                        counter = Counter(y_h)
                        estimate = math.sqrt(counter[0] / counter[1])
                        model = XGBClassifier(max_depth=max_depth, n_estimators=100, scale_pos_weight=estimate, seed=0,nthread=3, objective='binary:logistic', min_child_weight=min_child_weight, gamma=gamma, subsample=subsample, colsample_bytree=colsample_bytree)
                        model.fit(
                            df_train, 
                            y_train
                            )

                        totcost = total_cost_eval(model.predict(df_val), y_val)
                        print('totcost', totcost, min_child_weight, gamma, subsample, colsample_bytree, max_depth)
                        if totcost < bestcost:
                            bestcost = totcost
                            best_params = [min_child_weight,gamma,subsample,colsample_bytree,max_depth]
                        print('BEST', bestcost)
    model = XGBClassifier(max_depth=best_params[-1], n_estimators=100, scale_pos_weight=estimate, seed=0,nthread=3, objective='binary:logistic', min_child_weight=best_params[0], gamma=best_params[1], subsample=best_params[2], colsample_bytree=best_params[3])
    model.fit(
                            X, 
                            y
                            )  

    """
    
    counter = Counter(y_h)
    estimate = counter[0] / counter[1]
    model = XGBClassifier(max_depth=10, n_estimators=100, scale_pos_weight=estimate, seed=0,nthread=3, objective='binary:logistic')
    model.fit(
        df_train, 
        y_train, 
        eval_set=[(df_val, y_val)], 
        early_stopping_rounds=10, 
        eval_metric='aucpr'
        )

                     
    clf_isotonic = CalibratedClassifierCV(model, cv=2, method='isotonic')
    clf_isotonic.fit(pd.DataFrame(x_h), y_h)
    return model, clf_isotonic
                   




def predict_lab_prob(arguments):   
    
    PATH_SAVE, h, w, H, name_params, extract_params, md, PATH_PREDS_CLASSIFIERS  = arguments
    print(op.join(PATH_SAVE, 'X_h_subject_ep'+name_params+str(h)+extract_params+str(w)+str(H)+'.pkl'))
    
    
    with open(op.join(PATH_SAVE, 'X_h_subject_ep'+name_params+str(h)+extract_params+str(w)+str(H)+'.pkl'),'rb') as outp:
        X = pickle.load(outp)

        
    
            
    
    preds = md.predict(pd.DataFrame(X))
    probas = md.predict_proba(pd.DataFrame(X))[:,1]
    
    
        
    with open(op.join(PATH_PREDS_CLASSIFIERS, 'ep_probas_'+str(h)+'.pkl'),'wb') as outp:
        pickle.dump(probas, outp)
    with open(op.join(PATH_PREDS_CLASSIFIERS, 'ep_preds_'+str(h)+'.pkl'),'wb') as outp:
        pickle.dump(preds, outp)
    
    










def compute_preds_probas_flowpreds(argss):
    
        i,  w, H, s, h, name_params, extract_params, md, iso, PATH_SAVE, PATH_DATA, PATH_FLOW, use_type = argss
        
        with open(op.join(PATH_SAVE, 'X_whole_subject_'+str(i)+'_'+name_params+extract_params+str(w)+'.pkl'),'rb') as outp:
            X = pickle.load(outp)
    
            
        with open(op.join(PATH_DATA,'y_'+str(i)+'_'+ name_params+'.pkl'),'rb') as outp:
            flow_y = pickle.load(outp)
        n = len(flow_y)
        
        

        size_M = len(flow_y[w+H-1:n-w+1])
        size = n - (w+H-1) - (w-1)
        assert size_M == size
                
        dico_h = {k:v for v,k in enumerate(range(H-1, -w-1, -1))}
                
                

        preds = []
        probas = []
        probas_iso = []

        prd = md.predict(pd.DataFrame(X))
        prd = prd[dico_h[h]:dico_h[h]+size]
        
        preds.append(prd)
        with open(op.join(PATH_FLOW,'preds_'+str(h)+name_params+'_'+use_type + str(i) + str(w) + '_' + str(H) + '.pkl'),'wb') as outp:
            pickle.dump(preds, outp)
            
            
        probs = md.predict_proba(pd.DataFrame(X))[:,1]
                
        probs = probs[dico_h[h]:dico_h[h]+size]
        probas.append(probs)
                    
        with open(op.join(PATH_FLOW,'probas_'+str(h)+name_params+'_'+use_type + str(i) + str(w) + '_' + str(H) + '.pkl'),'wb') as outp:
            pickle.dump(probas, outp)   
            
            
        isoos = iso.predict_proba(pd.DataFrame(X))
        isoos = isoos[dico_h[h]:dico_h[h]+size]
        probas_iso.append(isoos)
        
                
                
        with open(op.join(PATH_FLOW,'iso_'+str(h)+name_params+'_'+use_type + str(i) + str(w) + '_' + str(H) + '.pkl'),'wb') as outp:
            pickle.dump(probas_iso, outp)  
        
    


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        