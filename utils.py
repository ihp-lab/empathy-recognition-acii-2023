import numpy as np
import torch, logging
from sklearn.metrics import f1_score, r2_score
import pandas as pd
import random
from collections import Counter

def CCC_score(y_true, y_pred, sample_weight=None, multioutput='uniform_average'):
    cor = np.corrcoef(y_true, y_pred)[0][1]

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)

    numerator = 2 * cor * sd_true * sd_pred

    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    return numerator / denominator

class CCC_loss(torch.nn.Module): 
    def __init__(self):
        super(CCC_loss, self).__init__()

    def forward(self, x, y):

        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        rho = torch.sum(vx * vy) / ((torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))) + 1e-10)
        x_m = torch.mean(x)
        y_m = torch.mean(y)
        x_s = torch.std(x)
        y_s = torch.std(y)
        ccc = 2 * rho * x_s * y_s / ((x_s ** 2 + y_s ** 2 + (x_m - y_m) ** 2) + 1e-10)

        return 1 - ccc

def get_score(labels, predictions, opt): 
                    
    score = CCC_score(labels, predictions)
    return score

def get_predictions(logits, opt):
 
    logits = logits.squeeze(axis=1)
    predictions = torch.round(logits.detach().cpu(), decimals=5).tolist()

    return predictions 

def get_loss(logits, labels, loss_func, opt):


    logits = logits.squeeze(axis=1)
    loss = loss_func(logits, labels.float())

    return loss


def set_logger(log_path):

	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	if not logger.handlers:
		file_handler = logging.FileHandler(log_path)
		file_handler.setFormatter(logging.Formatter('%(message)s'))
		logger.addHandler(file_handler)

		stream_handler = logging.StreamHandler()
		stream_handler.setFormatter(logging.Formatter('%(message)s'))
		logger.addHandler(stream_handler)
                
def standardize_attributes(df):


    datasets = list(set(df['dataset']))
    _precision = 5

    if 'PRIME' in datasets: 
        df_prime = df[df['dataset']=='PRIME']
        prime_empathy = df_prime['Empathy'].values
        
        _min, _max = 1, 7
        prime_empathy = np.round((prime_empathy-_min) / (_max-_min), _precision)
        
        df_prime['Empathy_norm'] = prime_empathy
        
        new_df = df_prime


    if 'NEXT' in datasets: 
    
        df_next = df[df['dataset']=='NEXT']
        
        next_empathy = df_next['Empathy'].values
        
        _min, _max = 1, 5
        next_empathy = np.round((next_empathy-_min) / (_max-_min), _precision)
        
        df_next['Empathy_norm'] = next_empathy
        
        new_df = df_next

    if len(datasets) == 2:   
        new_df = pd.concat([df_prime, df_next], axis=0)
        new_df.sort_index(inplace=True)
    
    return new_df


def get_discrete_labels_for_crossval(labels):

    labels_crossval = []

    tmp = list(set(labels))
    tmp.sort()
    label_to_crossval_dict = {}
    for i, val in enumerate(tmp):
        label_to_crossval_dict[val] = i

    print(label_to_crossval_dict)
    for l in labels:
        labels_crossval.append(label_to_crossval_dict[l])

    labels_crossval_df = pd.DataFrame({'id':labels.index.tolist(), 'label':labels_crossval})

    labels_crossval_df.set_index('id', inplace=True)

    return labels_crossval_df


def combine_single_sess_therapists(therapist_ids):

    counter = Counter(therapist_ids)

    single_sess_thers = []
    for ther in counter: 
        if counter[ther] < 5:
            single_sess_thers.append(ther)

    combined_ther_list = []
    for ther in therapist_ids: 
        if ther in single_sess_thers: 
            #map all the therapists with small sessions to the smallest ther_id
            combined_ther_list.append(int(single_sess_thers[0]))
        else: 
            combined_ther_list.append(int(ther))

    return np.array(combined_ther_list)

