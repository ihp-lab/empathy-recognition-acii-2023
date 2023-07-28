import pandas as pd 
import numpy as np
import os 
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
from collections import Counter
import utils

def get_eval_metrics(path): 

    df = pd.read_csv(path, header=0, index_col=0)
    print(Counter(df['labels']))

    accuracy = accuracy_score(df['labels'], df['predictions'])
    f1 = f1_score(df['labels'], df['predictions'], average='macro')
    #conf_mat = confusion_matrix(df['labels'], df['predictions'])
    #tn, fp, fn, tp = conf_mat.ravel()

    specificity = recall_score(df['labels'], df['predictions'], pos_label=0)
    sensitivity = recall_score(df['labels'], df['predictions'], pos_label=1)


    print('accuracy\t', accuracy)
    print('f1 macro\t', f1)
    print('sensitivity\t', sensitivity)
    print('specificity\t', specificity)
    

def get_average_fold_scores(path):

    files = os.listdir(path)
    files.sort()

    q = 0
    for file in files: 
        f_path = os.path.join(path, file)

        if not os.path.exists(os.path.join(f_path, 'predictions.csv')):
            q += 1
            continue

        df = pd.read_csv(os.path.join(f_path, 'predictions.csv'), header=0, index_col=0)

        scores = []
        for fold in range(5):
            df_fold = df[df['fold']==fold]
            fold_score = utils.CCC_score(df_fold['labels'], df_fold['predictions'])
            scores.append(fold_score)

        print('Q'+str(q), np.mean(scores), np.std(scores))
        q += 1


def main(): 
    
    
    path = 'results/combined/Empathy/EXPERIMENT_FOLDER'
    get_average_fold_scores(path)

if __name__ == '__main__':
    main()