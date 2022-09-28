
from joblib import Parallel, delayed
import os.path as op 
import os
import pickle 
from src.economy_pdm.Economy_Gamma_h import Economy_Gamma_h
from .evaluationUtils_pdm import compute_metrics
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sns
import pandas as pd

rcParams['figure.figsize'] = 18, 12

def train_eco_model(method, misClassificationCost, timeCost, nbGroup, folderRealData, w, H, s, train_classifs, estimate_probas, path_save, alpha):


    name = method + '_' + str(alpha) + '_' + str(w) + '_' + str(s) + '_' +  str(H) + '_' + str(nbGroup) + '_' + str(misClassificationCost[0][1]) +  str(misClassificationCost[1][0])
    if not (op.exists(op.join(path_save, name + '.pkl'))):
        if method == 'Gamma_h_useMC_myopic':
            myopic = True
            useMC = True
            modelEco_h = Economy_Gamma_h(myopic, useMC, misClassificationCost, timeCost, nbGroup, folderRealData, w, H, s)
        elif method == 'Gamma_h_useMC_non_myopic':
            myopic = False
            useMC = True
            modelEco_h = Economy_Gamma_h(myopic, useMC, misClassificationCost, timeCost, nbGroup, folderRealData, w, H, s)
        elif method == 'Gamma_h_notuseMC':
            useMC = False
            modelEco_h = Economy_Gamma_h(useMC, misClassificationCost, timeCost, nbGroup, folderRealData, w, H, s)
        else:
            print('model not found')
        modelEco_h.fit(train_classifs, estimate_probas)
        
        with open(op.join(path_save, name + '.pkl'),'wb') as outfile:
            pickle.dump(modelEco_h, outfile)


def train_eco_models(params, nb_cores, exp):

    methods = params["methods"] 
    misClassificationCosts = params["misClassificationCosts"]
    timeCosts = params["horizonCost"]
    nbGroups = params["nbGroups"]
    s = params["s"]
    w_s = params["w"]
    H_s = params["H"]
    name_params = params["data"]
    extract_params = params["extract"]
    folderRealData = op.join(op.dirname(op.realpath('__file__')), 'models', 'preds_probas_classifiers_h', name_params, extract_params)
    alphas = params["alphas"]

    extraction_type = 'multiple_horizons'
    use_type = 'train'
    path_windows = op.join(op.dirname(op.realpath('__file__')), 'input', 'windows_h_generated')
    use_type = 'train'
    train_classifs = {}
    for w in w_s:
        for H in H_s:
            train = {}
            
            for h in range(-w, H, s):
                with open(op.join(op.dirname(op.realpath('__file__')), 'input', 'windows_h_generated', 'X_h_subject_train'+name_params+str(h)+extract_params+str(w)+str(H)+'.pkl'),'rb') as outp:
                    x_h = pickle.load(outp)
                train[h] = pd.DataFrame(x_h)
            
            train_classifs[str(w)+str(H)] = train

    use_type = 'ep'
    estimate_probas = {}
    for w in w_s:
        for H in H_s:
            ep = {}
            for h in range(-w, H, s):
                with open(op.join(op.dirname(op.realpath('__file__')), 'input', 'windows_h_generated', 'X_h_subject_ep'+name_params+str(h)+extract_params+str(w)+str(H)+'.pkl'),'rb') as outp:
                    x_h = pickle.load(outp)
                ep[h] = pd.DataFrame(x_h)
            
            estimate_probas[str(w)+str(H)] = ep




    path_save = op.join(op.dirname(op.realpath('__file__')), 'models', 'eco_h', exp, 'ecoModels')
    try:
        os.mkdir(path_save)
    except OSError:
        print ("Creation of the directory %s failed" % path_save)
    else:
        print ("Successfully created the directory %s " % path_save)
    

    Parallel(n_jobs=nb_cores)(delayed(train_eco_model)(method, misClassificationCost, timeCosts[str(w)+str(H)+str(alpha)], nbGroup, folderRealData, w, H, s, train_classifs[str(w)+str(H)], estimate_probas[str(w)+str(H)], path_save, alpha) for method in methods for nbGroup in nbGroups for alpha in alphas for misClassificationCost in misClassificationCosts for w in w_s for H in H_s)


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
    

def evaluateModel(parameter):
    
    y_preds, h_preds_horizons, y_test, misClassificationCost, horizonCost, n, w, H, method = parameter
    

    y_test = y_test[w+H-1:n-w+1]
    
    
    metrics = compute_avgcost_stream(y_preds, h_preds_horizons, y_test, misClassificationCost, horizonCost)
    
    metrics_ts = {}#compute_metrics(np.array(y_test),np.array(y_preds))

    return metrics, metrics_ts


def evaluateModels(params, ys_preds, nb_cores, idx_subjects_test):
    methods = params["methods"] 
    misClassificationCosts = params["misClassificationCosts"]
    w_s = params["w"]
    H_s = params["H"]
    horizonCosts = params["horizonCost"]
    name_params = params["data"]
    alphas = params["alphas"]

    use_type = 'test'
    path_stream = op.join(op.dirname(op.realpath('__file__')), 'input', 'stream_generated')
    
    RESULTS = []
    for i,idx in enumerate(idx_subjects_test):
        
        with open(op.join(path_stream,'y_'+str(idx)+'_'+ name_params+'.pkl'),'rb') as outp:
            y_test = pickle.load(outp)
        
        parameters = []
        counter = 0
        for method in methods:
            for alpha in alphas:
                for misClassificationCost in misClassificationCosts:
                    for w in w_s:
                        for H in H_s:
                            y_preds, h_preds_horizons = ys_preds[i][counter]
                            counter += 1
                            n = len(y_test)
                            parameters.append((y_preds, h_preds_horizons, y_test, misClassificationCost, horizonCosts[str(w)+str(H)+str(alpha)], n, w, H, method))
    
    
        results = Parallel(n_jobs=nb_cores)(delayed(evaluateModel)(parameter) for parameter in parameters)
        RESULTS.append(results)
    return RESULTS


    

    







def computeMetric(idx, method, metric, misClassificationCost, horizonCost, alpha, w, s, H, nbGroup, exp, preds, probas, x_val, y_val):


    name = method + '_' + str(alpha) + '_' + str(w) + '_' + str(s) + '_' +  str(H) + '_' + str(nbGroup) + '_' + str(misClassificationCost[0][1]) +  str(misClassificationCost[1][0])
    path_model = op.join(op.dirname(op.realpath('__file__')), 'models', 'eco_h', exp, 'ecoModels')
    path_save = op.join(op.dirname(op.realpath('__file__')), 'models', 'eco_h', exp, 'metrics')

    with open(op.join(path_model, name + '.pkl'),'rb') as outfile:
        modelEco_h = pickle.load(outfile)
        
        
    y_preds, h_preds_horizons = modelEco_h.predict_horizon_flow(x_val, preds, probas)
    n = len(y_val)
    if metric == 'AvgCost':
        print('HEEEEEEEEEEEEEEY')
        metrics = compute_avgcost_stream(y_preds, h_preds_horizons, y_val[w+H-1:n-w+1], misClassificationCost, horizonCost)
    else:
        metrics = compute_metrics(np.array(y_val[w+H-1:n-w+1]),np.array(y_preds))

    path_save = op.join(op.dirname(op.realpath('__file__')), 'models', 'eco_h', exp, 'metrics')
    with open(op.join(path_save, name+ '_' +metric+str(idx)+'.pkl'), 'wb') as outfile:
        pickle.dump(metrics, outfile)

    return (method, metrics, nbGroup, alpha, w, H, misClassificationCost)

    
def compute_avgcost_stream(y_preds, h_preds_horizons, y_trues, misClassificationCost, timeCost):
    n = len(y_trues)
    AvgCost = 0
    for y_pred, horizon, y_true in zip(y_preds, h_preds_horizons, y_trues):
        AvgCost += misClassificationCost[int(y_pred)][int(y_true)] + timeCost[horizon]
    AvgCost /= n
    return AvgCost


def computeOptimal_nbGroups(params, nb_cores, exp, metric, idx_subjects_val_score):

    methods = params["methods"] 
    misClassificationCosts = params["misClassificationCosts"] 
    horizonCosts = params["horizonCost"]
    nbGroups = params["nbGroups"]
    s = params["s"]
    w_s = params["w"]
    H_s = params["H"]
    name_params = params["data"]
    alphas = params["alphas"]
    extract_params = params["extract"]

    path_m = op.join(op.dirname(op.realpath('__file__')), 'models', 'eco_h', exp, 'metrics')
    try:
        os.mkdir(path_m)
    except OSError:
        print ("Creation of the directory %s failed" % path_m)
    else:
        print ("Successfully created the directory %s " % path_m)


    # load data
    METRICS = []
    for idx in idx_subjects_val_score:
        print('IDDDDDDDDX : ', idx)
        with open(op.join(op.join(op.dirname(op.realpath('__file__')), 'input', 'stream_generated'),'x_'+str(idx)+'_'+ name_params+'.pkl'),'rb') as outp:
            x_val = pickle.load(outp)
        with open(op.join(op.join(op.dirname(op.realpath('__file__')), 'input', 'stream_generated'),'y_'+str(idx)+'_'+ name_params+'.pkl'),'rb') as outp:
            y_val = pickle.load(outp)
            
        print('g1')
        path_to_save = op.join(op.dirname(op.realpath('__file__')), 'models', 'preds_probas_flow', name_params, extract_params)
        use_type = 'val'
    
        preds={}
        probas={}
        for w in w_s:
            for H in H_s:
                with open(op.join(path_to_save,'preds_'+name_params+'_'+use_type + str(idx) + str(w) + '_' + str(H) + '.pkl'),'rb') as outp:
                    preds[str(w)+str(H)] = pickle.load(outp)
                
                print('g2') 
                for i,h in enumerate(range(-w, H, s)):
                        
                    preds[str(w)+str(H)][i] = [None] * (w+H-1) + list(map(int, preds[str(w)+str(H)][i])) + [None] * (w-1)
                        
                    assert len(preds[str(w)+str(H)][i]) == len(x_val)
                with open(op.join(path_to_save,'probas_'+name_params+'_'+use_type + str(idx) + str(w) + '_' + str(H) + '.pkl'),'rb') as outp:
                    probas[str(w)+str(H)] = pickle.load(outp)
                print('g3')
                for i,h in enumerate(range(-w, H, s)):
                    probas[str(w)+str(H)][i] = [None] * (w+H-1) + list(map(int, probas[str(w)+str(H)][i])) + [None] * (w-1)
                    assert len(probas[str(w)+str(H)][i]) == len(x_val)
        print('g4')
        metrics = Parallel(n_jobs=nb_cores)(delayed(computeMetric)(idx, method, metric, misClassificationCost, horizonCosts[str(w)+str(H)+str(alpha)], alpha, w, s, H, nbGroup, exp, preds[str(w)+str(H)], probas[str(w)+str(H)], x_val, y_val) for method in methods for nbGroup in nbGroups for alpha in alphas for misClassificationCost in misClassificationCosts for w in w_s for H in H_s)
        with open(op.join(path_m, 'metrics_val'+str(idx)+'.pkl'), 'wb') as outp:
            pickle.dump(metrics, outp)
        
        METRICS.append(metrics)
        
    metrics = []
    for i in range(len(METRICS[0])):
        avvg = 0
        for k in range(len(METRICS)):
            method, metrics_e, nbGroup, alpha, w, H, misClassificationCost = METRICS[k][i]
            avvg += metrics_e
        avvg /= len(METRICS)
        metrics.append((method, avvg, nbGroup, alpha, w, H, misClassificationCost))
    
    optimal_nbGroups = {}
    dict_metrics = {}
    for e in metrics:
        method, metrics_e, nbGroup, alpha, w, H, misClassificationCost = e
        dict_metrics[method+str(nbGroup)+str(alpha)+str(w)+str(H)+ str(misClassificationCost[0][1]) +  str(misClassificationCost[1][0])] = metrics_e


    for method in methods:
        for alpha in alphas:
            for w in w_s:
                for H in H_s:
                    for misClassificationCost in misClassificationCosts:
                        if metric == 'AvgCost':
                            optimal_nbGroups[method+str(alpha)+str(w)+str(H)+ str(misClassificationCost[0][1]) +  str(misClassificationCost[1][0])] = np.argmin(np.array([dict_metrics[method+str(nbGroup)+str(alpha)+str(w)+str(H)+ str(misClassificationCost[0][1]) +  str(misClassificationCost[1][0])] for nbGroup in nbGroups])) + 1
                        else:
                            optimal_nbGroups[method+str(alpha)+str(w)+str(H)+ str(misClassificationCost[0][1]) +  str(misClassificationCost[1][0])] = np.argmin(np.array([dict_metrics[method+str(nbGroup)+str(alpha)+str(w)+str(H)+ str(misClassificationCost[0][1]) +  str(misClassificationCost[1][0])][metric] for nbGroup in nbGroups])) + 1


    return optimal_nbGroups



def predict_test(method, misClassificationCost, exp, alpha, w, s, H, nbGroup, preds, probas, x_test):

    name = method + '_' + str(alpha) + '_' + str(w) + '_' + str(s) + '_' +  str(H) + '_' + str(nbGroup) + '_' +str(misClassificationCost[0][1]) +  str(misClassificationCost[1][0])
    path_model = op.join(op.dirname(op.realpath('__file__')), 'models', 'eco_h', exp, 'ecoModels')

    with open(op.join(path_model, name + '.pkl'),'rb') as outfile:
        modelEco_h = pickle.load(outfile)
    y_preds, horizon_used_for_prediction = modelEco_h.predict_horizon_flow(x_test, preds, probas)
    
    return [y_preds, horizon_used_for_prediction]



def test_eco_h_models(params, optimal_nbGroups, nb_cores, exp, idx_subjects_test):

    methods = params["methods"]
    misClassificationCosts = params["misClassificationCosts"] 
    s = params["s"]
    w_s = params["w"]
    H_s = params["H"]
    name_params = params["data"]
    alphas = params["alphas"]
    extract_params = params["extract"]
    use_type = 'test'
    path_stream = op.join(op.dirname(op.realpath('__file__')), 'input', 'stream_generated')

    YS_PREDS = []
    for idx in idx_subjects_test:
        print('IDX test : ', idx)
        
        with open(op.join(path_stream,'x_'+str(idx)+'_'+ name_params+'.pkl'),'rb') as outp:
            x_test = pickle.load(outp)
    
        path_to_save = op.join(op.dirname(op.realpath('__file__')), 'models', 'preds_probas_flow', name_params, extract_params)
        use_type = 'test'
        preds={}
        probas={}
        for w in w_s:
            for H in H_s:
                
                with open(op.join(path_to_save,'preds_'+name_params+'_'+use_type + str(idx) + str(w) + '_' + str(H) + '.pkl'),'rb') as outp:
                    preds[str(w)+str(H)] = pickle.load(outp)
                
                    # add dummy thinks on both sides for preds and proba
                    for i,h in enumerate(range(-w, H, s)):
                        preds[str(w)+str(H)][i] = [None] * (w+H-1) + list(map(int, preds[str(w)+str(H)][i])) + [None] * (w-1)
                        assert len(preds[str(w)+str(H)][i]) == len(x_test)
                        
                with open(op.join(path_to_save,'probas_'+name_params+'_'+use_type + str(idx) + str(w) + '_' + str(H) + '.pkl'),'rb') as outp:
                    probas[str(w)+str(H)] = pickle.load(outp)
                    for i,h in enumerate(range(-w, H, s)):
                        probas[str(w)+str(H)][i] = [None] * (w+H-1) + list(map(int, probas[str(w)+str(H)][i])) + [None] * (w-1)
                        assert len(probas[str(w)+str(H)][i]) == len(x_test)
        
        
        
        
        ys_preds = Parallel(n_jobs=nb_cores)(delayed(predict_test)(method, misClassificationCost, exp, alpha, w, s, H, optimal_nbGroups[method+str(alpha)+str(w)+str(H)+ str(misClassificationCost[0][1]) +  str(misClassificationCost[1][0])], preds[str(w)+str(H)], probas[str(w)+str(H)], x_test) for method in methods for alpha in alphas for misClassificationCost in misClassificationCosts for w in w_s for H in H_s)
        YS_PREDS.append(ys_preds)
    return YS_PREDS


def viz_results(params, evaluation, ys_preds, exp, idx_subjects_test):

    methods = params["methods"]
    misClassificationCosts = params["misClassificationCosts"] 
    s = params["s"]
    w_s = params["w"]
    H_s = params["H"]
    name_params = params["data"]
    alphas = params["alphas"]
    horizonCosts = params["horizonCost"]
    nbGroups = params["nbGroups"]

    # create directory
    path_save_plots = op.join(op.dirname(op.realpath('__file__')), 'models', 'eco_h', exp, 'plots')
    try:
        os.mkdir(path_save_plots)
    except OSError:
        print ("Creation of the directory %s failed" % path_save_plots)
    else:
        print ("Successfully created the directory %s " % path_save_plots)

    # -----------------------------------------------------------------------------------------------------------------------------
    # AvgCost in Validation versus nbGroups ---------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------------------
    markers = ["o", "s", "^", "*"]
    linestyles = ["solid", "dashed"]

    for w in w_s:
        for H in H_s:
            for alpha in alphas:
                fig, ax = plt.subplots()
                for m,method in enumerate(methods):
                    for mis,misClassificationCost in enumerate(misClassificationCosts):
                        y_values_plot = []
                        for nbGroup in nbGroups:

                            name = method + '_' + str(alpha) + '_' + str(w) + '_' + str(s) + '_' +  str(H) + '_' + str(nbGroup) + '_' + str(misClassificationCost[0][1]) +  str(misClassificationCost[1][0])
                            path_save = op.join(op.dirname(op.realpath('__file__')), 'models', 'eco_h', exp, 'metrics')
                            with open(op.join(path_save, name+ '_' +'AvgCost'+'.pkl'), 'rb') as outfile:
                                metrics = pickle.load(outfile)
                            y_values_plot.append(metrics)
                        
                        ax.plot([str(nbGroup) for nbGroup in nbGroups], y_values_plot, marker=markers[mis], linestyle=linestyles[m], label=method+ str(misClassificationCost[0][1]) +  str(misClassificationCost[1][0]))
                plt.title('AvgCost vs nbGroups on validation set')
                plt.xticks(rotation=45)
                plt.legend()
                plt.savefig(op.join(path_save_plots,  'AvgCost_vs_nbGroups_on_validation_set'+name_params+'w='+str(w)+'H='+str(H)+'alpha='+str(alpha)+'.png'))
                plt.close()

    # -----------------------------------------------------------------------------------------------------------------------------
    # AvgCost in Test versus horizon Cost -----------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------------------
    #### extract infos 
    to_plot = {}
    cnt = 0
    for method in methods:
        for alpha in alphas:
            for misClassificationCost in misClassificationCosts:
                for w in w_s:
                    for H in H_s:
                        metrics_avgcost, metrics_ts = evaluation[cnt]
                        cnt += 1
                        to_plot[method+str(alpha)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(w)+str(H)+'metrics_avgcost'] = metrics_avgcost
                        to_plot[method+str(alpha)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(w)+str(H)+'metrics_ts'] = metrics_ts
    
    
    for w in w_s:
        for H in H_s:
            fig, ax = plt.subplots()
            for m,method in enumerate(methods):
                for mis,misClassificationCost in enumerate(misClassificationCosts):
                    y_values_plot = []
                    for alpha in alphas:
                        y_values_plot.append(to_plot[method+str(alpha)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(w)+str(H)+'metrics_avgcost'])

                    ax.plot([str(alpha) for alpha in alphas], y_values_plot, marker=markers[mis], linestyle=linestyles[m], label=method+ str(misClassificationCost[0][1]) +  str(misClassificationCost[1][0]))
            plt.title('AvgCost vs horizon cost on test set')
            plt.xticks(rotation=45)
            plt.legend()
            plt.savefig(op.join(path_save_plots,  'AvgCost_vs_horizonCost_on_test_set'+name_params+'w='+str(w)+'H='+str(H)+'alpha='+str(alpha)+'.png'))
            plt.close()

    # -----------------------------------------------------------------------------------------------------------------------------
    # Ts metrics ------------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------------------
    for w in w_s:
        for H in H_s:
            fig, ax = plt.subplots(4, 4, figsize=(18, 18))
            for i,metric in enumerate(list(to_plot[methods[0]+str(alphas[0])+str(misClassificationCosts[0][0][1])+str(misClassificationCosts[0][1][0])+str(w_s[0])+str(H_s[0])+'metrics_ts'].keys())):
                plt.subplot(4, 4, i+1)

                for m,method in enumerate(methods):
                    for mis,misClassificationCost in enumerate(misClassificationCosts):
                        plt.plot([str(alpha) for alpha in alphas],
                        [to_plot[method+str(alpha)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(w)+str(H)+'metrics_ts'][metric] for alpha in alphas],
                        marker=markers[mis], linestyle=linestyles[m], label=method+ str(misClassificationCost[0][1]) +  str(misClassificationCost[1][0]))
                plt.ylim([0, 1])
                plt.xticks(rotation=45)
                plt.title(metric)
                plt.legend()            
            plt.savefig(op.join(path_save_plots,  'Metrics_ts_versus_horizon_cost_on_test_set'+name_params+'w='+str(w)+'H='+str(H)+'alpha='+str(alpha)+'.png'))
            plt.close()


    # -----------------------------------------------------------------------------------------------------------------------------
    # Distribution-----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------------------
    ### extract decision horizons
    colors = ['orange', 'blue', 'black', 'yellow', 'gray', 'purple']
    c=0
    horizons_plot = {}
    for method in methods:
        for alpha in alphas:
            for misClassificationCost in misClassificationCosts:
                for w in w_s:
                    for H in H_s:
                        y_preds, horizon_used_for_prediction = ys_preds[c]
                        horizons_plot[method+str(alpha)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(w)+str(H)] = horizon_used_for_prediction
                        c+=1


    
    for w in w_s:
        for H in H_s:
            for m,method in enumerate(methods):
                for mis,misClassificationCost in enumerate(misClassificationCosts):
                    fig, axes = plt.subplots(4, 4, figsize=(18, 18), dpi=100)
                    for i,alpha in enumerate(alphas):
                        
                        plt.subplot(5, 5, i+1)
                        plt.ylim([0, 5000])
                        plt.hist(horizons_plot[method+str(alpha)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(w)+str(H)], color = 'purple', label=method+ str(misClassificationCost[0][1]) +  str(misClassificationCost[1][0]))
                        plt.xticks(rotation=45)
                        plt.title('alpha:'+str(alpha))
                        plt.legend()  

                    plt.savefig(op.join(path_save_plots,  'Distribution_of_decision_moment'+name_params+'w='+str(w)+'H='+str(H)+'alpha='+str(alpha)+'method='+method+'misclassif='+str(misClassificationCost[0][1]) +  str(misClassificationCost[1][0])+'.png'))
                    
"""
for j,method in enumerate(methods):
    fig, axes = plt.subplots(3,4,figsize=(30, 15), dpi=100)
    for i,dataset in enumerate(datasets[24:36]):
        modelName = method + ',' + dataset + ',' + str(PARAMTIME)
        sns.distplot(data[modelName][0], color=colors[j], ax=axes[i//4][i%4]).set_title(method, fontsize=2)
        axes[i//4][i%4].set_xlim([0.0,1.0])
"""




