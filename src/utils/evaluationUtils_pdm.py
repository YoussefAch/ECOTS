
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import random 
import pickle 
import os.path as op 

def save_metrics_random(y_test, name_params, name_params_ex, w_s, H_s):
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

    
    path = op.join(op.dirname(op.realpath('__file__')), 'models', 'preds_probas_flow', name_params, 'random_classifier_'+name_params_ex+'.pkl')
    with open(path, 'wb') as inp:
        pickle.dump(random_values, inp)

    path = op.join(op.dirname(op.realpath('__file__')), 'models', 'preds_probas_flow', name_params, 'random_classifier_metrics'+name_params_ex+'.pkl')
    with open(path, 'wb') as inp:
        pickle.dump(metrics, inp)

    return metrics, random_values


def save_metrics_classifiers(y_test, path_to_save, name_params, name_params_ex, use_type, w_s, H_s, s):

    
    n = len(y_test)
    preds_true_w_H = {}
    for w in w_s:
        for H in H_s:
            # zone à prédire par tous les classifieurs 
            y_true = y_test[w+H-1:n-w+1]
            with open(op.join(path_to_save,name_params_ex,'preds_'+name_params+'_'+use_type + str(w) + '_' + str(H) + '.pkl'),'rb') as outp:
                preds = pickle.load(outp)

            for k,t in enumerate(range(-w, H, s)):
                y_preds = preds[k]
                
                y_preds = list(map(int, y_preds))
                
                assert len(y_preds)==len(y_true)
                preds_true_w_H[str(w)+str(H)+str(t)] = [y_preds, y_true]

    metrics = {}
    for w in w_s:
        for H in H_s:
            for t in range(-w, H, s):
                y_preds, y_true = preds_true_w_H[str(w)+str(H)+str(t)]        
                metrics[str(w)+str(H)+str(t)]  = compute_metrics(np.array(y_true), np.array(y_preds))   

    path = op.join(op.dirname(op.realpath('__file__')), 'models', 'preds_probas_flow', name_params, 'metrics_classifiers_'+use_type+name_params_ex+'.pkl')
    with open(path, 'wb') as inp:
        pickle.dump(metrics, inp)

    path = op.join(op.dirname(op.realpath('__file__')), 'models', 'preds_probas_flow', name_params, 'preds_true_w_H_'+use_type+name_params_ex+'.pkl')
    with open(path, 'wb') as inp:
        pickle.dump(preds_true_w_H, inp)
    
    return metrics, preds_true_w_H



def compute_metrics(values_real, values_pred):
    recall_classic = recall_score(values_real, values_pred)
    precision_classic = precision_score(values_real, values_pred)
    f1_classic = f1_score(values_real, values_pred)

    flat_metric = TSMetric(metric_option="time-series", alpha_r=0.1, cardinality="reciprocal", bias_p="flat", bias_r="flat")
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
    }
    
    return results



# this class was taken from github : put the like (todo)
class TSMetric:
    def __init__(self, metric_option="classic", beta=1.0, alpha_r=0.0, cardinality="one", bias_p="flat", bias_r="flat"):
        
        assert (alpha_r >= 0)&(alpha_r <= 1)
        assert metric_option in ["classic", "time-series", "numenta"]
        assert beta > 0
        assert cardinality in ["one", "reciprocal", "udf_gamma"]
        assert bias_p in ["flat", "front", "middle", "back"]
        assert bias_r in ["flat", "front", "middle", "back"]
        
        self.metric_option = metric_option
        self.beta = beta
        self.alpha_r = alpha_r
        self.alpha_p = 0
        self.cardinality = cardinality
        self.bias_p = bias_p
        self.bias_r = bias_r
    
    def _udf_gamma(self, overlap, task_type):
        """
        user defined gamma
        """
        return 1.0
    
    def _gamma_select(self, gamma, overlap, task_type):
        if gamma == "one":
            return 1.0
        elif gamma == "reciprocal":
            if overlap > 1:
                return 1.0/overlap
            else:
                return 1.0
        elif gamma == "udf_gamma_def":
            if overlap > 1:
                return 1.0/self._udf_gamma(overlap, task_type)
            else:
                return 1.0
    
    def _gamma_function(self, overlap_count, task_type):
        overlap = overlap_count[0]
        if task_type == 0:
            return self._gamma_select(self.cardinality, overlap, task_type)
        elif task_type == 1:
            return self._gamma_select(self.cardinality, overlap, task_type)
        else:
            raise Exception("invalid argument in gamma function")
    
    
    def _compute_omega_reward(self, r1, r2, overlap_count, task_type):
        if r1[1] < r2[0] or r1[0] > r2[1]:
            return 0
        else:
            overlap_count[0] += 1
            overlap = np.zeros(r1.shape)
            overlap[0] = max(r1[0], r2[0])
            overlap[1] = min(r1[1], r2[1])
            return self._omega_function(r1, overlap, task_type)
    
    def _omega_function(self, rrange, overlap, task_type):
        anomaly_length = rrange[1] - rrange[0] + 1
        my_positional_bias = 0
        max_positional_bias = 0
        temp_bias = 0
        for i in range(1, anomaly_length+1):
            temp_bias = self._delta_function(i, anomaly_length, task_type)
            max_positional_bias += temp_bias
            j = rrange[0] + i -1
            if j >= overlap[0] and j <= overlap[1]:
                my_positional_bias += temp_bias
        if max_positional_bias > 0:
            res = my_positional_bias / max_positional_bias
            return res
        else:
            return 0
    
    def _delta_function(self, t, anomaly_length, task_type):
        if task_type == 0:
            return self._delta_select(self.bias_p, t, anomaly_length, task_type)
        elif task_type == 1:
            return self._delta_select(self.bias_r, t, anomaly_length, task_type)
        else:
            raise Exception("Invalid task type in delta function")
    
    def _delta_select(self, delta, t, anomaly_length, task_type):
        if delta == "flat":
            return 1.0
        elif delta == "front":
            return float(anomaly_length - t + 1.0)
        elif delta == "middle":
            if t <= anomaly_length/2.0 :
                return float(t)
            else:
                return float(anomaly_length - t + 1.0)
        elif delta == "back":
            return float(t)
        elif delta == "udf_delta":
            return self._udf_delta(t, anomaly_length, task_type)
        else:
            raise Exception("Invalid positional bias value")
    
    def _udf_delta(self, t, anomaly_length, task_type):
        """
        user defined delta function
        """
        return 1.0
    
    def _update_precision(self, real_anomalies, predicted_anomalies):
        precision = 0
        if len(predicted_anomalies) == 0:
            return 0
        for i in range(len(predicted_anomalies)):
            range_p = predicted_anomalies[i, :]
            omega_reward = 0
            overlap_count = [0]
            for j in range(len(real_anomalies)):
                range_r = real_anomalies[j, :]
                omega_reward += self._compute_omega_reward(range_p, range_r, overlap_count, 0)
            overlap_reward = self._gamma_function(overlap_count, 0)*omega_reward
            if overlap_count[0] > 0:
                existence_reward = 1
            else:
                existence_reward = 0

            precision += self.alpha_p*existence_reward + (1 - self.alpha_p)*overlap_reward
        precision /= len(predicted_anomalies)
        return precision
    
    def _update_recall(self, real_anomalies, predicted_anomalies):
        recall = 0
        if len(real_anomalies) == 0:
            return 0
        for i in range(len(real_anomalies)):
            omega_reward = 0
            overlap_count = [0]
            range_r = real_anomalies[i, :]
            for j in range(len(predicted_anomalies)):
                range_p = predicted_anomalies[j, :]
                omega_reward += self._compute_omega_reward(range_r, range_p, overlap_count, 1)
            overlap_reward = self._gamma_function(overlap_count, 1)*omega_reward
            
            if overlap_count[0] > 0:
                existence_reward = 1
            else:
                existence_reward = 0

            recall += self.alpha_r*existence_reward + (1 - self.alpha_r)*overlap_reward
        recall /= len(real_anomalies)
        return recall
    
    def _shift(self, arr, num, fill_value=np.nan):
        arr = np.roll(arr,num)
        if num < 0:
            arr[num:] = fill_value
        elif num > 0:
            arr[:num] = fill_value
        return arr
    
    def _prepare_data(self, values_real, values_pred):
        
        assert len(values_real) == len(values_pred)
        assert np.allclose(np.unique(values_real), np.array([0, 1]))
        assert np.allclose(np.unique(values_pred), np.array([0, 1]))
        
        if self.metric_option == "classic":
            real_anomalies = np.argwhere(values_real == 1).repeat(2, axis=1)
            predicted_anomalies = np.argwhere(values_pred == 1).repeat(2, axis=1)
            
        elif self.metric_option == "time-series":
            predicted_anomalies_ = np.argwhere(values_pred == 1).ravel()
            predicted_anomalies_shift_forward = self._shift(predicted_anomalies_, 1, fill_value=predicted_anomalies_[0])
            predicted_anomalies_shift_backward = self._shift(predicted_anomalies_, -1, fill_value=predicted_anomalies_[-1])
            predicted_anomalies_start = np.argwhere((predicted_anomalies_shift_forward - predicted_anomalies_) != -1).ravel()
            predicted_anomalies_finish = np.argwhere((predicted_anomalies_ - predicted_anomalies_shift_backward) != -1).ravel()
            predicted_anomalies = np.hstack([predicted_anomalies_[predicted_anomalies_start].reshape(-1, 1), \
                                             predicted_anomalies_[predicted_anomalies_finish].reshape(-1, 1)])
            
            real_anomalies_ = np.argwhere(values_real == 1).ravel()
            real_anomalies_shift_forward = self._shift(real_anomalies_, 1, fill_value=real_anomalies_[0])
            real_anomalies_shift_backward = self._shift(real_anomalies_, -1, fill_value=real_anomalies_[-1])
            real_anomalies_start = np.argwhere((real_anomalies_shift_forward - real_anomalies_) != -1).ravel()
            real_anomalies_finish = np.argwhere((real_anomalies_ - real_anomalies_shift_backward) != -1).ravel()
            real_anomalies = np.hstack([real_anomalies_[real_anomalies_start].reshape(-1, 1), \
                                             real_anomalies_[real_anomalies_finish].reshape(-1, 1)])
            
        elif self.metric_option == "numenta":
            predicted_anomalies = np.argwhere(values_pred == 1).repeat(2, axis=1)
            real_anomalies_ = np.argwhere(values_real == 1).ravel()
            real_anomalies_shift_forward = self._shift(real_anomalies_, 1, fill_value=real_anomalies_[0])
            real_anomalies_shift_backward = self._shift(real_anomalies_, -1, fill_value=real_anomalies_[-1])
            real_anomalies_start = np.argwhere((real_anomalies_shift_forward - real_anomalies_) != -1).ravel()
            real_anomalies_finish = np.argwhere((real_anomalies_ - real_anomalies_shift_backward) != -1).ravel()
            real_anomalies = np.hstack([real_anomalies_[real_anomalies_start].reshape(-1, 1), \
                                             real_anomalies_[real_anomalies_finish].reshape(-1, 1)])
        return real_anomalies, predicted_anomalies
    
    def score(self, values_real, values_predicted):
        assert isinstance(values_real, np.ndarray)
        assert isinstance(values_predicted, np.ndarray)
        if len(np.unique(values_predicted))==1 and np.unique(values_predicted)[0] == 0:
            precision=0
            recall=0
            Fbeta=0
        else:
            real_anomalies, predicted_anomalies = self._prepare_data(values_real, values_predicted)
            precision = self._update_precision(real_anomalies, predicted_anomalies)
            recall = self._update_recall(real_anomalies, predicted_anomalies)
            if precision + recall != 0:
                Fbeta = (1 + self.beta**2)*precision*recall/(self.beta**2*precision + recall)
            else:
                Fbeta = 0
        
        return precision, recall, Fbeta


