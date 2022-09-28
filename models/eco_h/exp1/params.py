from input.params_extraction import ex_pdm_1 as GLOBAL_PARAMS_EXTRACTION 
import numpy as np 
import pickle 
import os.path as op


exp = 'exp_pddm_1'
data = 'pdm'
extract = "ex_pdm_1"
Cm = 1
s = GLOBAL_PARAMS_EXTRACTION.S
w_s = [10]
H_s = [51]
nbCLasses = 2

misClassificationCosts = [[[0,1],[1,0]]]



alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
horizonCost = {str(w)+str(H)+str(alpha):{i: alpha*t/(H+w) for t,i in zip(range(H+w, 0, -1), range(-w, H+w, 1))} for alpha in alphas for w in w_s for H in H_s}
nbGroups = [i for i in range(1,21)]
metric = "AvgCost"

methods = ['Gamma_h_useMC_non_myopic']

dictionary = {
    "extract":extract,
    "data":data,
    "Cm":Cm,
    "s":s,
    "w":w_s,
    "H":H_s,
    "nbClasses":nbCLasses,
    "misClassificationCosts":misClassificationCosts,
    "alphas":alphas,
    "horizonCost":horizonCost,
    "nbGroups":nbGroups,
    "metric":metric,
    "methods":methods
}
path = op.join(op.dirname(op.realpath('__file__')), 'models', 'eco_h', exp, 'params.pkl')
with open(path, 'wb') as dtfile:
    pickle.dump(dictionary, dtfile)
