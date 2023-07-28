import os
import pandas as pd
import utils
import numpy as np

#####################################################################################
#This file is focused on evaluating the results on the session level across quartiles
#####################################################################################

def get_full_session_results(path): 

    folds = os.listdir(path)
    folds.sort()

    session_df = {}
    for i, f in enumerate(folds): 
        f_path = os.path.join(path, f, 'predictions.csv')
        df = pd.read_csv(f_path, header=0, index_col=0)
        if i == 0:
            session_df['labels'] = df['labels'].tolist()
            session_df['fold'] = df['fold'].tolist()
            session_df['dataset'] = df['dataset'].tolist()
            session_df['ID'] = df.index.tolist()

        session_df['Q'+str(i)] = df['predictions'].tolist()

    session_df = pd.DataFrame(session_df)
    session_df = session_df.set_index('ID')

    session_df['session_pred'] = session_df[['Q0', 'Q1', 'Q2', 'Q3']].mean(axis=1)
    sess_ccc_score = utils.CCC_score(session_df['labels'].values, session_df['session_pred'].values)
    print('full score', sess_ccc_score)


    fold_scores = []
    for i in range(5):
        fold_df = session_df[session_df['fold']==i]
        fold_ccc_score = utils.CCC_score(fold_df['labels'].values, fold_df['session_pred'].values)
        fold_scores.append(fold_ccc_score)

    print('Score across folds', np.mean(fold_scores), np.std(fold_scores))

def get_full_sess_cross_corpus_per_val(path):

    for q_val in range(4):
        predictions = []
        for q_train in range(4):
            sub_path = os.path.join(path, 'train_Q'+str(q_train)+'_val_Q'+str(q_val), 'predictions.csv')

            df = pd.read_csv(sub_path, header=0, index_col=0)
            predictions.append(df['predictions'].tolist())

            if q_val == 0: 
                labels = df['labels'].values

        predictions = np.array(predictions)
        predictions = predictions.mean(axis=0)
        ccc_score = utils.CCC_score(labels, predictions)
        print('Quartile'+str(q_val), ccc_score)
        print('------------------------------------')



def main(): 

    #This folder includes separate directories for results from individual quartiles
    #Named Q0, Q1, Q2, Q3

    path = 'results/combined/Empathy/EXPERIMENT_FOLDER'
    
    if 'cross_corpus' in path: 
        get_full_sess_cross_corpus_per_val(path)
    else: 
        get_full_session_results(path)

if __name__ == "__main__":
    main()