
from . import DataGenerationUtils as dgutils
import pandas as pd 
import pickle
import os.path as op
from joblib import Parallel, delayed

def most_frequent(List):
    return max(set(List), key = List.count)


def flatten_list_tuples(ls):
    return list(sum(ls, ()))


def clear_overlaps(windows):
    intersections = []
    for i in range(len(windows)):
        for j in range(i+1, len(windows)):
            if len(set(windows[i][:-1]).intersection(windows[j][:-1])) > 0:
                intersections.append((i,j))
    return intersections


def is_window_overlaps_windows(l1, l):
    for e in l:
        if len(set(e[:-1]).intersection(l1)) > 0:
            return True
    return False



def extract_windows_different_horizons(flow_x, flow_y, w, H, s):
    
    # extracted windows (real values and class)
    extracted_w = {i:[] for i in range(-w, H, s)}
    
    #extracted windows (indices and class)
    indices = {i:[] for i in range(-w, H, s)}
    
    # flow indices
    flow_x_i = [i for i in range(len(flow_x))]
    
    # compute counters
    counters = dgutils.portions_to_counters(flow_y)
    # extract windows for anomalies to train classifiers
    index = 0
    
    # iterate over portions
    for nbPoints, current_class in counters:
        
        # extract windows around anomaly except the first one
        # h_0 : pas d'anomalie dans la fenetre
        # h_-1: un seul point
        # h_1 : 1 point de plus avant l'anomalie 
        if current_class == 1 and index != 0:
            for h in range(-w, H, s):
                # we don't want the windows before anomaly that contains some anomaly points
                if not (h >= 0 and flow_y[index-h-w:index-h] != w*[0]):
                    extracted_w[h].append(flow_x[index-h-w:index-h]+["1"])
                    indices[h].append(flow_x_i[index-h-w:index-h]+["1"])
                    
        index += nbPoints
    
    
    # clear overlaps of class 1
    for h in range(-w, H, s):
        cleared = clear_overlaps(indices[h])
        while cleared:
            freq_element = most_frequent(flatten_list_tuples(cleared))
            indices[h].pop(freq_element)
            cleared = clear_overlaps(indices[h])
                
        
    
    # extract normal windows
    for i in range(len(flow_x)-w+1):
        for h in range(-w, H, s):
            ## pas avec overlap avec les exemples de classe 1
            ## pas doverlap avec une anomalie dans l'horizon h
            if not is_window_overlaps_windows(flow_x_i[i:i+w], indices[h]):
                if i+w+h < len(flow_y) and flow_y[i+w+h] != 1:
                    extracted_w[h].append(flow_x[i:i+w]+["0"])
                    indices[h].append(flow_x_i[i:i+w]+["0"])
    
    for h in range(-w, H, s):
        for i,e in enumerate(extracted_w[h]):
            if len(e) != w+1:
                extracted_w[h].pop(i)

    for k,v in extracted_w.items():
        extracted_w[k] = pd.DataFrame.from_records(v)
    return [extracted_w, indices]





def extract_windows_different_horizons_targetFocus(flow_x, flow_y, w, H, s):
    
    # extracted windows (real values and class)
    extracted_w = {i:[] for i in range(-w, H, s)}
    
    #extracted windows (indices and class)
    indices = {i:[] for i in range(-w, H, s)}
    
    # flow indices
    flow_x_i = [i for i in range(len(flow_x))]
    
    n = len(flow_x)
    
    for target_y in range(w+H-1, n-w, w+H):
        for h in range(-w, H, s):
            extracted_w[h].append(flow_x[target_y-h-w:target_y-h]+[str(flow_y[target_y])])
            indices[h].append(flow_x_i[target_y-h-w:target_y-h]+[str(flow_y[target_y])])
    
    for k,v in extracted_w.items():
        extracted_w[k] = pd.DataFrame.from_records(v)

    print('finish extraction for w=',w,' and H=',H)
    return extracted_w, indices





















def extract_windows_for_economy_classifiers(flow_x, flow_y, T, skip, threshold, sampling_ratio=0.2):
    
    assert len(flow_x)==len(flow_y)
    
    step = int(T*sampling_ratio) if int(T*sampling_ratio)>0 else 1
    n=len(flow_x)
    
    # windows to train rocket classifier
    extracted_w = {t:[] for t in range(int(T*sampling_ratio), T+1, int(T*sampling_ratio))}
    indices_w = {t:[] for t in range(int(T*sampling_ratio), T+1, int(T*sampling_ratio))}
    flow_x_i = [i for i in range(len(flow_x))]
    
    #T/2 CHOIX
    for i in range(0, n-T+1, skip):

        # classical full-length time series
        full_ts = flow_x[i:i+T]
        
        # choice if more than half is considered anormal
        nb_ones = flow_y[i:i+T].count(1)
        
        # extract all windows
        for t in range(step, T+1, step):
            
            if nb_ones > threshold:
                extracted_w[t].append(full_ts[:t]+["1"])
                indices_w[t].append(flow_x_i[i:i+t]+["1"])
            else:
                extracted_w[t].append(full_ts[:t]+["0"])
                indices_w[t].append(flow_x_i[i:i+t]+["0"])
                
        #if t!= T-1:
        #    if nb_ones > T//2:
        #        extracted_w[t].append(full_ts[:T]+["1"])
        #        indices_w[t].append(flow_x_i[i:i+T]+["1"])
        #    else:
        #       extracted_w[t].append(full_ts[:T]+["0"])
        #        indices_w[t].append(flow_x_i[i:i+T]+["0"])
                
        
    # clear overlaps
    for t in range(step, T+1, step):

        cleared = clear_overlaps(indices_w[t])
        while cleared:
            freq_element = most_frequent(flatten_list_tuples(cleared))
            indices_w[t].pop(freq_element)
            cleared = clear_overlaps(indices_w[t])
    for k,v in extracted_w.items():
        extracted_w[k] = pd.DataFrame.from_records(v)
    return extracted_w, indices_w


def extract_windows(path_data, path_windows_h, use_type, name_params, extraction_type, extraction_params, nb_cores):
    with open(op.join(path_data, 'x_' + use_type + '_' + name_params + '.pkl'), 'rb') as data_file:
        flow_x = pickle.load(data_file)
    with open(op.join(path_data, 'y_' + use_type + '_' + name_params + '.pkl'), 'rb') as data_file:
        flow_y = pickle.load(data_file)
    if extraction_type=='simple_economy':
        extracted_w, indices = extract_windows_for_economy_classifiers(flow_x, flow_y, extraction_params['T'], extraction_params['skip'], extraction_params['threshold'], extraction_params['sampling_ratio'])
    else:
        results = Parallel(n_jobs=nb_cores)(delayed(extract_windows_different_horizons_targetFocus)(flow_x, flow_y, w, H, extraction_params['s']) for w in extraction_params['w'] for H in extraction_params['H'])
        assert len(results) == len(extraction_params['w']) * len(extraction_params['H'])
    cnt=-1
    for w in extraction_params['w']:
        for H in extraction_params['H']:
            cnt+=1
            with open(op.join(path_windows_h, 'extracted_w_indices_' + extraction_type + use_type + '_' + name_params + str(w) + '_' + str(H) + '.pkl'), 'wb') as data_file:
                pickle.dump(results[cnt], data_file)