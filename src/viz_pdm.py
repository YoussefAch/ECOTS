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
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='test')
parser.add_argument('--nameParams', help='path to json file with input params', required=True)
parser.add_argument('--extractParams', help='path to json file with input params', required=True)
parser.add_argument('--exp', help='path to json file with input params', required=True)
args = parser.parse_args()
GLOBAL_PARAMS_EXTRACTION = importlib.import_module('input.params_extraction.'+args.extractParams)




name_params = args.nameParams
extract_params = args.extractParams
exp = args.exp
PATH_DATA = op.join(op.dirname(op.realpath('__file__')), 'input', 'stream_generated')
PATH_SAVE = op.join(op.dirname(op.realpath('__file__')), 'input', 'windows_h_generated')


path_params = op.join(op.dirname(op.realpath('__file__')), 'models', 'eco_h', exp, 'params.pkl')
with open(path_params, 'rb') as dtfile:
    params = pickle.load(dtfile)
misClassificationCosts = params["misClassificationCosts"]
timeCost = params["horizonCost"]

s = params["s"]
w_s = params["w"]
H_s = params["H"]
alphas = params["alphas"]
methods = params["methods"]



idx_subjects_train_classifiers = GLOBAL_PARAMS_EXTRACTION.idx_subjects_train_classifiers
idx_subjects_val_score = GLOBAL_PARAMS_EXTRACTION.idx_subjects_val_score
idx_subjects_ep = GLOBAL_PARAMS_EXTRACTION.idx_subjects_ep
idx_subjects_test = GLOBAL_PARAMS_EXTRACTION.idx_subjects_test
w_s = GLOBAL_PARAMS_EXTRACTION.W
H_s = GLOBAL_PARAMS_EXTRACTION.H
s = GLOBAL_PARAMS_EXTRACTION.S


##############################################################################################################################
##############################################################################################################################
########################################## plot avgcost for each subject in test ! ##########################################
##############################################################################################################################
##############################################################################################################################

path_baselines = op.join(op.dirname(op.realpath('__file__')), 'models', 'baselines', name_params, extract_params)
path_competitor = op.join(op.dirname(op.realpath('__file__')), 'models', 'competitors', name_params, extract_params)
path_eco = op.join(op.dirname(op.realpath('__file__')), 'models', 'eco_h', exp, 'metrics')


res_b_1_all = []
for idx in idx_subjects_test:
    with open(op.join(path_baselines, 'results_avgCost_baseline_1'+str(idx)+'test'+name_params+extract_params+exp+'.pkl'), 'rb') as outp:
        res_b_1 = pickle.load(outp)
    res_b_1_all.append(res_b_1)
   
    
res_b_a_all = []
for idx in idx_subjects_test:
    with open(op.join(path_baselines, 'results_avgCost_baseline_a'+str(idx)+'test'+name_params+extract_params+exp+'.pkl'), 'rb') as outp:
        res_b_a = pickle.load(outp)
    res_b_a_all.append(res_b_a)
    

res_b_2_all = []
for idx in idx_subjects_test:
    with open(op.join(path_baselines, 'results_avgCost_baseline2'+str(idx)+'test'+name_params+extract_params+exp+'.pkl'), 'rb') as outp:
        res_b_2 = pickle.load(outp)
    res_b_2_all.append(res_b_2)

res_b_0_all = []   
for idx in idx_subjects_test: 
    with open(op.join(path_baselines, 'results_avgCost_baseline0'+str(idx)+'test'+name_params+extract_params+exp+'.pkl'), 'rb') as outp:
        res_b_0 = pickle.load(outp)
    res_b_0_all.append(res_b_0)
   
res_c_1_all = []
for idx in idx_subjects_test: 
    with open(op.join(path_competitor, 'results_avgCost_competitor1'+str(idx)+'test'+name_params+extract_params+exp+'.pkl'), 'rb') as outp:
        res_c_1 = pickle.load(outp)
    res_c_1_all.append(res_c_1)

with open(path_eco + 'evaluation_stream_test_'+exp+'.pkl','rb') as outp:
    evaluation = pickle.load(outp)

res_eco_all = []
for idx in range(len(idx_subjects_test)):
    res_eco={}
    counter = 0
    for method in methods:
        for alpha in alphas:
            for misClassificationCost in misClassificationCosts:
                for w in w_s:
                    for H in H_s:
                        metrics_avg, metrics_ts = evaluation[idx][counter]
                        counter +=1
                        res_eco[method+str(w)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(alpha)] = metrics_avg
    res_eco_all.append(res_eco)

import numpy as np

for idx in range(len(idx_subjects_test)):
        if idx == 0:
            avg_b0 = np.array([res_b_0_all[0][str(w)+str(misClassificationCosts[0][0][1])+str(misClassificationCosts[0][1][0])+str(alpha)] for alpha in alphas])
            avg_b1 = np.array([res_b_1_all[0][str(w)+str(misClassificationCosts[0][0][1])+str(misClassificationCosts[0][1][0])+str(alpha)] for alpha in alphas])
            avg_ba = np.array([res_b_a_all[0][str(w)+str(misClassificationCosts[0][0][1])+str(misClassificationCosts[0][1][0])+str(alpha)] for alpha in alphas])
            avg_b2 = np.array([res_b_2_all[0][str(w)+str(misClassificationCosts[0][0][1])+str(misClassificationCosts[0][1][0])+str(alpha)] for alpha in alphas])
            avg_c1 = np.array([res_c_1_all[0][str(w)+str(misClassificationCosts[0][0][1])+str(misClassificationCosts[0][1][0])+str(alpha)] for alpha in alphas])
            avg_res_eco = np.array([res_eco_all[0]['Gamma_h_useMC_non_myopic'+str(w)+str(misClassificationCosts[0][0][1])+str(misClassificationCosts[0][1][0])+str(alpha)] for alpha in alphas])
        else:
            avg_b0 += np.array([res_b_0_all[idx][str(w)+str(misClassificationCosts[0][0][1])+str(misClassificationCosts[0][1][0])+str(alpha)] for alpha in alphas])
            avg_b1 += np.array([res_b_1_all[idx][str(w)+str(misClassificationCosts[0][0][1])+str(misClassificationCosts[0][1][0])+str(alpha)] for alpha in alphas])
            avg_ba += np.array([res_b_a_all[idx][str(w)+str(misClassificationCosts[0][0][1])+str(misClassificationCosts[0][1][0])+str(alpha)] for alpha in alphas])
            avg_b2 += np.array([res_b_2_all[idx][str(w)+str(misClassificationCosts[0][0][1])+str(misClassificationCosts[0][1][0])+str(alpha)] for alpha in alphas])
            avg_c1 += np.array([res_c_1_all[idx][str(w)+str(misClassificationCosts[0][0][1])+str(misClassificationCosts[0][1][0])+str(alpha)] for alpha in alphas])
            avg_res_eco += np.array([res_eco_all[idx]['Gamma_h_useMC_non_myopic'+str(w)+str(misClassificationCosts[0][0][1])+str(misClassificationCosts[0][1][0])+str(alpha)] for alpha in alphas])       
    
        """ plt.figure()
        plt.plot([str(alpha) for alpha in alphas],
                 [res_b_1_all[idx][str(w)+str(misClassificationCosts[0][0][1])+str(misClassificationCosts[0][1][0])+str(alpha)] for alpha in alphas],
                            marker='x', label='Late baseline', color='grey')
        
        plt.plot([str(alpha) for alpha in alphas],
                 [res_b_2_all[idx][str(w)+str(misClassificationCosts[0][0][1])+str(misClassificationCosts[0][1][0])+str(alpha)] for alpha in alphas],
                            marker='x', label='CC', color='black')
        plt.plot([str(alpha) for alpha in alphas],
                 [res_c_1_all[idx][str(w)+str(misClassificationCosts[0][0][1])+str(misClassificationCosts[0][1][0])+str(alpha)] for alpha in alphas],
                            marker='x', label='SR', color='orange')

        plt.plot([str(alpha) for alpha in alphas],
                 [res_eco_all[idx]['Gamma_h_useMC_non_myopic'+str(w)+str(misClassificationCosts[0][0][1])+str(misClassificationCosts[0][1][0])+str(alpha)] for alpha in alphas],
                            marker='o', label=r'ECO-$\gamma$-non-myopic', color='red')
        plt.plot([str(alpha) for alpha in alphas],
                 [res_b_0_all[idx][str(w)+str(misClassificationCosts[0][0][1])+str(misClassificationCosts[0][1][0])+str(alpha)] for alpha in alphas],
                            marker='x', label='Early baseline', color='blue')
        plt.xticks(rotation=90)
        plt.ylim([0,25])
        plt.grid()
        plt.ylabel('AvgCost')
        plt.xlabel(r'$\alpha$')
        plt.legend()
        plt.savefig(op.join(op.dirname(op.realpath('__file__')), 'plots', 'pdm', extract_params, exp+'cm_'+str(idx_subjects_test[idx])+'_avg_cost.png'), bbox_inches='tight')"""

plt.figure()
print("yes,", avg_res_eco/len(idx_subjects_test))
"""n=6
alphas = alphas[:n]
avg_b0 = avg_b0[:n]
avg_b1 = avg_b1[:n]
avg_ba = avg_ba[:n]
avg_b2 = avg_b2[:n]
avg_c1 = avg_c1[:n]
avg_res_eco = avg_res_eco[:n]"""

print("yes,sr", avg_c1/len(idx_subjects_test))
"""plt.plot([str(alpha) for alpha in alphas],
                 avg_ba/len(idx_subjects_test),
                            marker='x', label='Late baseline', color='grey')"""
plt.plot([str(alpha) for alpha in alphas],
                 avg_b1/len(idx_subjects_test),
                            marker='x', label='Late baseline', color='grey')
plt.plot([str(alpha) for alpha in alphas],
                 avg_b2/len(idx_subjects_test),
                            marker='x', label='CC', color='black')


plt.plot([str(alpha) for alpha in alphas],
                 avg_res_eco/len(idx_subjects_test),
                            marker='o', label=r'Economy-$\gamma$', color='red')
plt.plot([str(alpha) for alpha in alphas],
                 avg_b0/len(idx_subjects_test),
                            marker='x', label='Early baseline', color='blue')
plt.plot([str(alpha) for alpha in alphas],
         avg_c1/len(idx_subjects_test),
         marker='x', label='SR', color='orange')

plt.xticks(rotation=90, fontsize=15)
plt.yticks(fontsize=15)
plt.ylim([0,2])
plt.ylabel('AvgCost', fontsize=16)
plt.xlabel(r'$\alpha$', fontsize=20)
plt.legend(fontsize=14)
plt.savefig(op.join(op.dirname(op.realpath('__file__')), 'plots', 'pdm', extract_params, 'AVVVG'+exp+'cm_'+str(idx_subjects_test[idx])+'_avg_cost.png'),  bbox_inches='tight')


plt.close()








##############################################################################################################################
##############################################################################################################################
####################################################### DECISION MOMENT ######################################################
##############################################################################################################################
##############################################################################################################################
with open(path_eco + 'predictions_stream_test_'+exp+'.pkl','rb') as outp:
    ys_preds = pickle.load(outp)
import collections

horizons = []
preds = []
for idx in range(len(idx_subjects_test)):

    # eco
    c = 0
    horizons_plot = {}
    preds_plot = {}
    for method in methods:
        for alpha in alphas:
            for misClassificationCost in misClassificationCosts:
                for w in w_s:
                    for H in H_s:
                        y_preds, horizon_used_for_prediction = ys_preds[idx][c]
                        if alpha == alphas[-1]:
                            xx = collections.Counter(horizon_used_for_prediction)
                            print(xx)
                        horizons_plot[method+str(alpha)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(w)+str(H)] = horizon_used_for_prediction
                        preds_plot[method+str(alpha)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(w)+str(H)] = y_preds
                        c += 1


    
    
    # b2
    with open(op.join(path_baselines, 'results_optimalThreshold_baseline_2'+'val'+name_params+extract_params+exp+'.pkl'), 'rb') as outp:
        opts = pickle.load(outp)
    
    for alpha in alphas:
        for w in w_s:
            for H in H_s:
                name = op.join(path_baselines, 'preds_baseline_2' + str(idx_subjects_test[idx])+ str(w) + str(H) + 'test' + str(opts[str(w)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(alpha)])+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+ str(alpha)  + '.pkl')
                with open(name, 'rb') as inp:
                    res = pickle.load(inp)
                                
                pr, horizonsCC = res
                horizons_plot['CC'+str(alpha)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(w)+str(H)] = horizonsCC
                preds_plot['CC'+str(alpha)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(w)+str(H)] = pr
            
    # SR
    with open(op.join(path_competitor, 'results_optimalThreshold_competitor_1'+'val'+name_params+extract_params+exp+'.pkl'), 'rb') as outp:
        opts = pickle.load(outp)
    for alpha in alphas:
        for w in w_s:
            for H in H_s:                   
                name = op.join(path_competitor, 'preds_competitor_1'  + str(idx_subjects_test[idx])+ str(w) + str(H) + 'test' + str(opts[str(w)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(alpha)])[0] + str(opts[str(w)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(alpha)])[1]+ str(opts[str(w)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(alpha)])[2]+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+ str(alpha) + '.pkl')
                with open(name, 'rb') as inp:
                    res = pickle.load(inp)
                                
                _, horizonsSR = res
                horizons_plot['SR'+str(alpha)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(w)+str(H)] = horizonsSR      
    
    for alpha in alphas:
        for w in w_s:
            for H in H_s:
                
                path_preds = op.join(op.dirname(op.realpath('__file__')), 'models', 'preds_probas_flow', name_params, extract_params)
                with open(op.join(path_preds,'preds_'+name_params+'_'+'test' + str(idx_subjects_test[idx]) + str(w) + '_' + str(H) + '.pkl'),'rb') as inp:
                    predicted_labels = pickle.load(inp)
                    
                
                
                horizons_plot['Late'+str(alpha)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(w)+str(H)] = -w
                preds_plot['Late'+str(alpha)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(w)+str(H)] = predicted_labels[0]
                
                horizons_plot['Early'+str(alpha)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(w)+str(H)] = H-1
                preds_plot['Early'+str(alpha)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(w)+str(H)] = predicted_labels[-1]
    
    horizons.append(horizons_plot)
    preds.append(preds_plot)





methods = ['Gamma_h_useMC_non_myopic', 'SR', 'CC', 'Late', 'Early']
mmethods = ['ECONOMY', 'SR', 'CC', 'Late baseline', 'Early baseline']
#methods = ['Gamma_h_useMC_non_myopic', 'CC', 'Late', 'Early']
savefigs = op.join(op.dirname(op.realpath('__file__')), 'plots', 'pdm', 'dist', extract_params)
import seaborn as sns

#sns.set_theme()
fig, axes = plt.subplots(2,1, figsize=(10,7))
sns.distplot(horizons[0]['Gamma_h_useMC_non_myopic'+str(0.001)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(w)+str(H)], ax=axes[0], color='red', label=r'Economy-$\gamma$', kde=False)
sns.distplot(horizons[0]['SR'+str(0.001)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(w)+str(H)], ax=axes[0], color='orange', bins=1, label='SR', kde=False)
print('ana',collections.Counter(horizons[0]['SR'+str(0.001)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(w)+str(H)]))
sns.distplot(horizons[0]['CC'+str(0.001)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(w)+str(H)], ax=axes[0], color='black', label='CC', kde=False)
axes[0].set_xlim(-w, H+1)
axes[0].set_ylim(0, 1000)
axes[0].set_xlabel("Horizon", fontsize=16)
axes[0].set_ylabel("Frequency", fontsize=16)

axes[0].set_xticklabels(axes[0].get_xticks(),fontsize=14)
axes[0].set_title(r'Distribution of the decision moments when $\alpha$ = ' + str(0.001), fontsize=16)

axes[0].legend( fontsize=16)
sns.distplot(horizons[0]['Gamma_h_useMC_non_myopic'+str(0.1)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(w)+str(H)], ax=axes[1], color='red', label=r'Economy-$\gamma$', kde=False)
sns.distplot(horizons[0]['SR'+str(0.1)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(w)+str(H)], ax=axes[1], color='orange', label='SR', kde=False)
sns.distplot(horizons[0]['CC'+str(0.1)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(w)+str(H)], ax=axes[1], color='black', label='CC', kde=False)
axes[1].set_xlim(-w, H+1)
axes[1].set_ylim(0, 2000)
axes[1].set_xlabel("Horizon", fontsize=16)
axes[1].set_ylabel("Frequency", fontsize=16)
axes[1].set_xticklabels(axes[1].get_xticks(),fontsize=14)
axes[1].set_title(r'Distribution of the decision moments when $\alpha$ = ' + str(0.1), fontsize=16)
axes[1].legend(fontsize=16)
fig.tight_layout()
plt.savefig(op.join(savefigs,'dist_paper.png'), bbox_inches='tight')
plt.close()
import collections
for idx in range(1):
    print(idx)
    horizons[idx]["SR"+str(10)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(w)+str(H)] = 50
    horizons[idx]["SR"+str(100)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(w)+str(H)] = 50
    horizons[idx]["SR"+str(0.0001)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(w)+str(H)] = 20
    horizons[idx]["SR"+str(0.001)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(w)+str(H)] = 20
    horizons[idx]["SR"+str(0.01)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(w)+str(H)] = 20
    """print(collections.Counter(horizons[idx]["SR"+str(0.0001)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(w)+str(H)]))
    print(collections.Counter(horizons[idx]["SR"+str(0.001)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(w)+str(H)]))
    print(collections.Counter(horizons[idx]["SR"+str(0.01)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(w)+str(H)]))
    print(collections.Counter(horizons[idx]["SR"+str(10)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(w)+str(H)]))
    print(collections.Counter(horizons[idx]["SR"+str(100)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(w)+str(H)]))"""
    fig, axes = plt.subplots(7,5, figsize=(18,10))
    for p in range(7):
        for m in range(5):
            #if methods[m] == "Late" or methods[m] == "Early" or (methods[m]=="SR" and (alphas[p]==0.0001 or alpha==0.001 or alpha==0.01 or alpha ==10)):
            """if (methods[m] == "SR" and  alphas[p]==0.0001) or  (methods[m] == "SR" and  alphas[p]==0.001) or (methods[m] == "SR" and  alphas[p]==0.01) or   (methods[m] == "SR" and  alphas[p]==10) or  (methods[m] == "SR" and  alphas[p]==100) or methods[m] == "Late" or methods[m] == "Early" : 
                axes[p,m].hist(horizons[idx][methods[m]+str(alphas[p])+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(w)+str(H)], bins=1) 
            else:"""
            sns.distplot(horizons[idx][methods[m]+str(alphas[p])+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(w)+str(H)], ax=axes[p,m])

            axes[p,m].set_xlim(-w-5, H+1)
            axes[p,m].set_ylim(0, 1)
            axes[p,m].set_title(mmethods[m]+' and alpha = ' + str(alphas[p]))
    fig.tight_layout()
    plt.savefig(op.join(savefigs, str(idx_subjects_test[idx])+'.png'))
    plt.close()


"""
##### STATS about when the failures have been detected
PATH_SAVE_DATA = op.join(op.dirname(op.realpath('__file__')), 'input', 'stream_generated')


stats = []
for i, idx in enumerate(idx_subjects_test):
    idx_failures = []
    with open(op.join(PATH_SAVE_DATA,'y_'+str(idx)+'_'+ name_params+'.pkl'),'rb') as outp:
        y = pickle.load(outp)
    n = len(y)
    idx_failures = [k for k in range(len(y)) if y[k] == 1]
    print('number of failures: ', len(idx_failures))
    # failure in economy should be between w+H-1 and n-w
    
    preds_horizons = []
    for alpha in alphas:
        p_h = []
        for w in w_s:
            for H in H_s:
                nbF = 0
                for f in idx_failures:
                    if f < w+H-1 or f > n-w:
                        p_h.append((None,None))
                    else:
                        if preds[i]['Gamma_h_useMC_non_myopic'+str(alpha)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(w)+str(H)][f-(w+H-1)] == y[f]:
                            nbF += 1
                        p_h.append((preds[i]['Gamma_h_useMC_non_myopic'+str(alpha)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(w)+str(H)][f-(w+H-1)], horizons[i]['Gamma_h_useMC_non_myopic'+str(alpha)+str(misClassificationCost[0][1])+str(misClassificationCost[1][0])+str(w)+str(H)][f-(w+H-1)]))
                nbF /= len(idx_failures)
                print('alpha = ', alpha, ', percentage of true positives: ', nbF)
        preds_horizons.append(p_h)
    
    print(preds_horizons)
    




    
print('###### number of groups')

with open(path_eco + 'optimal_nbGroups_'+exp+'.pkl','rb') as outp:
    optimal_nbGroups = pickle.load(outp)

print(optimal_nbGroups)
   

"""
















