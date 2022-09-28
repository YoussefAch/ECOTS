from .economyhUtils_pdm import compute_avgcost_stream
import numpy as np
import pickle
import os.path as op
from joblib import Parallel, delayed

def getIdxs(y, p):
    n = len(y)
    idxs = []
    last_class = y[0]
    for i in range(len(y)):
        if y[i] != last_class:
            for k in range(i-1, i-p-1, -1):
                if k >= 0 and y[k] == last_class:
                    idxs.append(k)
                else:
                    last_class = y[i]
                    break
            
    last_class = y[-1]
    cnt = 0
    for i in range(len(y)):
        if last_class == y[-(i+1)] and cnt<p:
            idxs.append(n-1-i)
            cnt +=1
        else:
            break
    return idxs


def compute_Avg_Cost_baseline_0(arguments):
    p, w, H, predictions_w_H, predicted_labels_base, y_trues, misClassificationCost, timeCost, path_save = arguments
    n = len(y_trues)
    y_preds = predicted_labels_base[-1]

    
    
    y_trues = y_trues[w+H-1:n-w+1]
  
    with open(path_save, 'wb') as outp:
        pickle.dump(y_preds, outp)


    h_preds_horizons = len(y_trues)*[H-1]
    
    
    
    avgCost = compute_avgcost_stream(y_preds, h_preds_horizons, y_trues, misClassificationCost, timeCost)

    return avgCost

def compute_Avg_Cost_baseline_a(arguments):
    p, w, H, predictions_w_H, predicted_labels_base, y_trues, misClassificationCost, timeCost, path_save, i_hor = arguments
    n = len(y_trues)
    print('hooo',i_hor)
    y_preds = predicted_labels_base[i_hor]
    
    hrz = list(range(-w,H,1))
    
    
    with open(path_save, 'wb') as outp:
        pickle.dump(y_preds, outp)

    y_trues = y_trues[w+H-1:n-w+1]
    
    h_preds_horizons = len(y_trues)*[hrz[i_hor]]
    
    
    avgCost = compute_avgcost_stream(y_preds, h_preds_horizons, y_trues, misClassificationCost, timeCost)

    return avgCost
def compute_Avg_Cost_baseline_1(arguments):
    p, w, H, predictions_w_H, predicted_labels_base, y_trues, misClassificationCost, timeCost, path_save = arguments
    n = len(y_trues)
    y_preds = predicted_labels_base[0]

    
    
    with open(path_save, 'wb') as outp:
        pickle.dump(y_preds, outp)

    y_trues = y_trues[w+H-1:n-w+1]
    
    h_preds_horizons = len(y_trues)*[-w]
    
    
    avgCost = compute_avgcost_stream(y_preds, h_preds_horizons, y_trues, misClassificationCost, timeCost)

    return avgCost


def optimal_threshold(results, thresholds, w_s, misClassificationCosts, alphas, timeCosts):
    opt = {}
    
    
    for w in w_s:
        for misClassificationCost in misClassificationCosts:
            for alpha,timeCost in zip(alphas, timeCosts):
                for i in range(len(results)):
                    if i==0:
                        acosts = np.array([results[i][str(w)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(alpha)+str(threshold)] for threshold in thresholds])
                    else:
                        acosts += np.array([results[i][str(w)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(alpha)+str(threshold)] for threshold in thresholds])
                idx = np.argmin(acosts)
                opt[str(w)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(alpha)] = thresholds[idx]
    return opt



def compute_Avg_Cost_baseline_2(arguments):
    p, w, H, s, predictions_w_H, predicted_labels_base, threshold, y_trues, misClassificationCost, timeCost, name = arguments

    n = len(y_trues)

    targets = np.arange(n)
    prediced_labels = n*[None]

    horizon_used_for_prediction = n*[None]
    horizons = [h for h in range(-w, H, s)]

    for i in range(n-w+1):

        current_targets = targets[i: i + w + H]    

        # I can use a heap data structure here :-), absolutely in a job interview
        for idx,target in enumerate(current_targets):

            if idx == 0:

                if prediced_labels[target] == None: # we fore the system to trigger a decision
                    prediced_labels[target] = predicted_labels_base[idx][i]
                    horizon_used_for_prediction[target] = horizons[idx]
            else:
                
                if prediced_labels[target] == None:
                    
                    if predictions_w_H[idx][i][1] > threshold:
                        prediced_labels[target] = predicted_labels_base[idx][i]
                        horizon_used_for_prediction[target] = horizons[idx]
    
    prediced_labels = prediced_labels[w+H-1:n-w+1]
    horizon_used_for_prediction = horizon_used_for_prediction[w+H-1:n-w+1]
    with open(name, 'wb') as outp:
        pickle.dump([prediced_labels, horizon_used_for_prediction], outp)

    avgCost = compute_avgcost_stream(prediced_labels, horizon_used_for_prediction, y_trues[w+H-1:n-w+1], misClassificationCost, timeCost)
    return avgCost



def compute_Avg_Cost_competitor_1(arguments):

    p, w, H, s, predictions_w_H, predicted_labels_base, threshold, y_trues, misClassificationCost, timeCost, name = arguments


    n = len(y_trues)

    targets = np.arange(n)
    prediced_labels = n*[None]

    horizon_used_for_prediction = n*[None]
    horizons = [h for h in range(-w, H, s)]

    for i in range(n-w+1):

        current_targets = targets[i: i + w + H]

        # I can use a heap data structure here :-), absolutely in a job interview
        for idx,target in enumerate(current_targets):

            if idx == 0:

                if prediced_labels[target] == None: # we fore the system to trigger a decision
                    prediced_labels[target] = predicted_labels_base[idx][i]
                    horizon_used_for_prediction[target] = horizons[idx]
            else:
                
                if prediced_labels[target] == None:

                    maxproba = np.max(predictions_w_H[idx][i])
                    ecart = abs(predictions_w_H[idx][i][0] - predictions_w_H[idx][i][1])
                    decision_rule = threshold[0] * maxproba + threshold[1] * ecart + threshold[2] * ((w+H-idx) / (w+H))

                    if decision_rule > 0:
                        prediced_labels[target] = predicted_labels_base[idx][i]
                        horizon_used_for_prediction[target] = horizons[idx]
                    
    
    prediced_labels = prediced_labels[w+H-1:n-w+1]
    horizon_used_for_prediction = horizon_used_for_prediction[w+H-1:n-w+1]
    with open(name, 'wb') as outp:
        pickle.dump([prediced_labels, horizon_used_for_prediction], outp)

    avgCost = compute_avgcost_stream(prediced_labels, horizon_used_for_prediction, y_trues[w+H-1:n-w+1], misClassificationCost, timeCost)
    return avgCost



def run_avgCost_computation(idx, method, p, w_s, H_s, s, use_type, path_preds, name_params, misClassificationCosts, timeCosts, alphas, thresholds, NB_CORES, y, path_save, opts=None, i_hor=None):

    
    if method == 'baselinea':
        params = []
        for H in H_s:
            for w in w_s:

                with open(op.join(path_preds,'iso_'+name_params+'_'+use_type + str(idx) + str(w) + '_' + str(H) + '.pkl'),'rb') as inp:
                    predictions_w_H = pickle.load(inp)
                    
                    
                with open(op.join(path_preds,'preds_'+name_params+'_'+use_type + str(idx) + str(w) + '_' + str(H) + '.pkl'),'rb') as inp:
                    predicted_labels = pickle.load(inp)
                for misClassificationCost in misClassificationCosts:
                    for alpha in alphas:
                        name = op.join(path_save, 'preds_baseline_a' + str(idx) + str(w) + str(H) + use_type+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+ str(alpha)  + '.pkl')
                        params.append((p, w, H, predictions_w_H, predicted_labels, y, misClassificationCost, timeCosts[str(w)+str(H)+str(alpha)], name, i_hor))
                        
                        
        results = Parallel(n_jobs=NB_CORES)(delayed(compute_Avg_Cost_baseline_a)(param) for param in params)
        
        
    elif method == 'baseline1':
        params = []
        for H in H_s:
            for w in w_s:

                with open(op.join(path_preds,'iso_'+name_params+'_'+use_type + str(idx) + str(w) + '_' + str(H) + '.pkl'),'rb') as inp:
                    predictions_w_H = pickle.load(inp)
                    
                    
                with open(op.join(path_preds,'preds_'+name_params+'_'+use_type + str(idx) + str(w) + '_' + str(H) + '.pkl'),'rb') as inp:
                    predicted_labels = pickle.load(inp)
                for misClassificationCost in misClassificationCosts:
                    for alpha in alphas:
                        name = op.join(path_save, 'preds_baseline_1' + str(idx) + str(w) + str(H) + use_type+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+ str(alpha)  + '.pkl')
                        params.append((p, w, H, predictions_w_H, predicted_labels, y, misClassificationCost, timeCosts[str(w)+str(H)+str(alpha)], name))
                                        


        results = Parallel(n_jobs=NB_CORES)(delayed(compute_Avg_Cost_baseline_1)(param) for param in params)

    elif method == 'baseline0':
        params = []
        for H in H_s:
            for w in w_s:
                with open(op.join(path_preds,'iso_'+name_params+'_'+use_type + str(idx) + str(w) + '_' + str(H) + '.pkl'),'rb') as inp:
                    predictions_w_H = pickle.load(inp)
                    
                    
                with open(op.join(path_preds,'preds_'+name_params+'_'+use_type + str(idx) + str(w) + '_' + str(H) + '.pkl'),'rb') as inp:
                    predicted_labels = pickle.load(inp)
                    
                for misClassificationCost in misClassificationCosts:
                    for alpha in alphas:
                        name = op.join(path_save, 'preds_baseline_0' + str(idx) + str(w) + str(H) + use_type +str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+ str(alpha)  + '.pkl')
                        params.append((p, w, H, predictions_w_H, predicted_labels, y, misClassificationCost, timeCosts[str(w)+str(H)+str(alpha)], name))
                                        


        results = Parallel(n_jobs=NB_CORES)(delayed(compute_Avg_Cost_baseline_0)(param) for param in params)
        
    elif method == 'baseline2':

        params = []
        for H in H_s:
            for w in w_s:
                with open(op.join(path_preds,'iso_'+name_params+'_'+use_type + str(idx) + str(w) + '_' + str(H) + '.pkl'),'rb') as inp:
                    predictions_w_H = pickle.load(inp)
                for i,h in enumerate(range(-w, H, s)):
                    predictions_w_H[h] = [[1,0]] * (w+H-1) +  list(predictions_w_H[h]) + [[1,0]] * (w-1)
                
                
              
                        
                with open(op.join(path_preds,'preds_'+name_params+'_'+use_type + str(idx) + str(w) + '_' + str(H) + '.pkl'),'rb') as inp:
                    predicted_labels = pickle.load(inp)
                for i,h in enumerate(range(-w, H, s)):
                    predicted_labels[h] = [1] * (w+H-1) +  list(map(int, predicted_labels[h])) + [1] * (w-1)
                for misClassificationCost in misClassificationCosts:
                    for alpha in alphas:
                        if use_type == 'val':
                            for threshold in thresholds:
                                name = op.join(path_save, 'preds_baseline_2'+ str(idx) + str(w) + str(H) + use_type + str(threshold)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+ str(alpha)  + '.pkl')
                                params.append((p, w, H, s, predictions_w_H, predicted_labels, threshold, y, misClassificationCost, timeCosts[str(w)+str(H)+str(alpha)], name))
                        else:
                            name = op.join(path_save, 'preds_baseline_2'+ str(idx) + str(w) + str(H) + use_type + str(opts[str(w)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(alpha)])+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+ str(alpha)  + '.pkl')
                            params.append((p, w, H, s, predictions_w_H, predicted_labels, opts[str(w)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(alpha)], y, misClassificationCost, timeCosts[str(w)+str(H)+str(alpha)], name))

        results = Parallel(n_jobs=NB_CORES)(delayed(compute_Avg_Cost_baseline_2)(param) for param in params)


    else:
        params = []
        for H in H_s:
            for w in w_s:
                with open(op.join(path_preds,'iso_'+name_params+'_'+use_type + str(idx) + str(w) + '_' + str(H) + '.pkl'),'rb') as inp:
                    predictions_w_H = pickle.load(inp)
                    for i,h in enumerate(range(-w, H, s)):
                        predictions_w_H[h] = [[1,0]] * (w+H-1) +  list(predictions_w_H[h]) + [[1,0]] * (w-1)
                with open(op.join(path_preds,'preds_'+name_params+'_'+use_type + str(idx) + str(w) + '_' + str(H) + '.pkl'),'rb') as inp:
                    predicted_labels = pickle.load(inp)
                    for i,h in enumerate(range(-w, H, s)):
                        predicted_labels[h] = [1] * (w+H-1) +  list(map(int, predicted_labels[h])) + [1] * (w-1)
                for misClassificationCost in misClassificationCosts:
                    for alpha in alphas:
                        if use_type == 'val':
                            for threshold in thresholds:
                                name = op.join(path_save, 'preds_competitor_1' + str(idx)+ str(w) + str(H) + use_type + str(threshold[0]) + str(threshold[1])+ str(threshold[2])+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+ str(alpha) +'.pkl')
                                params.append((p, w, H, s, predictions_w_H, predicted_labels, threshold, y, misClassificationCost, timeCosts[str(w)+str(H)+str(alpha)], name))
                        else:
                            name = op.join(path_save, 'preds_competitor_1' + str(idx) + str(w) + str(H) + use_type + str(opts[str(w)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(alpha)])[0] + str(opts[str(w)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(alpha)])[1]+ str(opts[str(w)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(alpha)])[2]+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+ str(alpha) + '.pkl')
                            params.append((p, w, H, s, predictions_w_H, predicted_labels, opts[str(w)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(alpha)], y, misClassificationCost, timeCosts[str(w)+str(H)+str(alpha)], name))
                        
        results = Parallel(n_jobs=NB_CORES)(delayed(compute_Avg_Cost_competitor_1)(param) for param in params)

    return results



def from_parallel_result_to_dict(idx, method, parallel_result, w_s, misClassificationCosts, timeCosts, alphas, thresholds, path_save, use_type, name_params, extractParams, exp):
    
    if method == 'baselinea':

        results = {}
        cnt=0
        for w in w_s:
            for misClassificationCost in misClassificationCosts:
                for alpha,timeCost in zip(alphas, timeCosts):
                    if use_type == 'val':
                        for threshold in thresholds:
                            results[str(w)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(alpha)+str(threshold)] = parallel_result[cnt]
                            cnt+=1
                    else:
                        results[str(w)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(alpha)] = parallel_result[cnt]
                        cnt+=1

        with open(op.join(path_save, 'results_avgCost_baseline_a'+str(idx)+use_type+name_params+extractParams+exp+'.pkl'), 'wb') as outp:
            pickle.dump(results, outp)
            
    if method == 'baseline1':

        results = {}
        cnt=0
        for w in w_s:
            for misClassificationCost in misClassificationCosts:
                for alpha,timeCost in zip(alphas, timeCosts):
                    if use_type == 'val':
                        for threshold in thresholds:
                            results[str(w)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(alpha)+str(threshold)] = parallel_result[cnt]
                            cnt+=1
                    else:
                        results[str(w)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(alpha)] = parallel_result[cnt]
                        cnt+=1

        with open(op.join(path_save, 'results_avgCost_baseline_1'+str(idx)+use_type+name_params+extractParams+exp+'.pkl'), 'wb') as outp:
            pickle.dump(results, outp)

        
    elif method == 'baseline0':

        results = {}
        cnt=0
        for w in w_s:
            for misClassificationCost in misClassificationCosts:
                for alpha,timeCost in zip(alphas, timeCosts):
                    if use_type == 'val':
                        for threshold in thresholds:
                            results[str(w)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(alpha)+str(threshold)] = parallel_result[cnt]
                            cnt+=1
                    else:
                        results[str(w)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(alpha)] = parallel_result[cnt]
                        cnt+=1

        with open(op.join(path_save, 'results_avgCost_baseline0'+str(idx)+use_type+name_params+extractParams+exp+'.pkl'), 'wb') as outp:
            pickle.dump(results, outp)

    elif method == 'baseline2':

        results = {}
        cnt=0
        for w in w_s:
            for misClassificationCost in misClassificationCosts:
                for alpha,timeCost in zip(alphas, timeCosts):
                    if use_type == 'val':
                        for threshold in thresholds:
                            results[str(w)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(alpha)+str(threshold)] = parallel_result[cnt]
                            cnt+=1
                    else:
                        results[str(w)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(alpha)] = parallel_result[cnt]
                        cnt+=1
        with open(op.join(path_save, 'results_avgCost_baseline2'+str(idx)+use_type+name_params+extractParams+exp+'.pkl'), 'wb') as outp:
            pickle.dump(results, outp)



    
    else:

        results = {}
        cnt=0
        for w in w_s:
            for misClassificationCost in misClassificationCosts:
                for alpha,timeCost in zip(alphas, timeCosts):
                    if use_type == 'val':
                        for threshold in thresholds:
                            results[str(w)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(alpha)+str(threshold)] = parallel_result[cnt]
                            cnt+=1
                    else:
                        results[str(w)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(alpha)] = parallel_result[cnt]
                        cnt+=1
        with open(op.join(path_save, 'results_avgCost_competitor1'+str(idx)+use_type+name_params+extractParams+exp+'.pkl'), 'wb') as outp:
            pickle.dump(results, outp)

    return results
