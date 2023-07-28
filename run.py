import os 

import pandas as pd 
import numpy as np
import torch, pickle
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, train_test_split, GroupShuffleSplit
from collections import Counter
import logging, random

from Args import Args
from data_loader import SessionEmbeddingDatasetFromText, collate_fn_text
from model import BERT_RNN_fromText
from train_validate import train_model, validate_model

from functools import partial

from dataclasses import asdict

from simple_parsing import ArgumentParser
import utils


seed = 110
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

def run_model(data_path, outcome_path):
    
    dataset_df = pd.read_csv(data_path, header=0, sep='\t', index_col='global_id')[["normed_text", "speaker", "session", "dataset", "quantile"]]
    outcome_df = pd.read_csv(outcome_path, header=0, index_col='ID')


    #simple_parsing: https://github.com/lebrice/SimpleParsing
    parser = ArgumentParser()
    parser.add_arguments(Args, dest='args')
    opt = parser.parse_args().args


    #https://huggingface.co/j-hartmann/emotion-english-distilroberta-base (emo-roberta)

    model_dict = {"roberta":"roberta-base", 
                  'distil-roberta':'distilroberta-base',
                  'emo-roberta':"j-hartmann/emotion-english-distilroberta-base"}

    opt.encoder = model_dict[opt.model]
    
    if opt.dataset != "combined":
        dataset_df = dataset_df[dataset_df['dataset']==opt.dataset]
        outcome_df = outcome_df[outcome_df['dataset']==opt.dataset]

    #Remove nan values 
    dataset_df = dataset_df.dropna()

    session_ids = list(set(dataset_df['session'].values.tolist()))
    session_ids.sort()
    
    outcome = outcome_df[['Empathy', 'dataset', 'therapist_id']]
    outcome = outcome[outcome.index.isin(session_ids)] 
    
    out_path = os.path.join('results', opt.dataset, opt.global_type)

    output_dir = os.path.join(out_path, opt.output_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    utils.set_logger(os.path.join(output_dir, 'logging.log'))
    device = torch.device(f"cuda:{opt.device_id}" if opt.device_id >= 0 else "cpu")

    #Regression
    outcome = utils.standardize_attributes(outcome)
    opt.global_type += '_norm'
    labels = outcome[opt.global_type]
    labels_crossval = utils.get_discrete_labels_for_crossval(labels)


    if opt.cross_corpus: 

        train_ids = list(set(dataset_df[dataset_df['dataset']=='PRIME']['session']))
        val_ids = list(set(dataset_df[dataset_df['dataset']=='NEXT']['session']))

        val_labels = labels.loc[val_ids]
        train_labels = labels.loc[train_ids]

        trial_ids = {'train':train_ids, 'val':val_ids}
        model = BERT_RNN_fromText(opt).to(device)

        val = SessionEmbeddingDatasetFromText(trial_ids['val'], dataset_df, val_labels, opt, 'val', opt.cross_corpus_val_q)
        train = SessionEmbeddingDatasetFromText(trial_ids['train'], dataset_df, train_labels, opt, 'train', opt.cross_corpus_train_q)
            
        val_dl = torch.utils.data.DataLoader(val, batch_size=opt.batch_size, collate_fn=collate_fn_text, shuffle=True, pin_memory=True)
        train_dl = torch.utils.data.DataLoader(train, batch_size=opt.batch_size, collate_fn=collate_fn_text, shuffle=True, pin_memory=True, drop_last=True)

        loss_func = utils.CCC_loss()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate)

        last_best_epoch = -1
        best_score = float('-inf')
        for epoch in range(opt.epochs_num):
            
            loss, train_score = train_model(model, train_dl, optimizer, loss_func, device, opt)
            _ , val_score, results_df = validate_model(model, val_dl, loss_func, device, opt)

            
            best_score_str =  ""

            if val_score > best_score: 
                best_score = val_score
                best_score_str = 'New found best!'
                last_best_epoch = epoch
                all_results_df = results_df
                
            
            log_str = 'Epoch: ' + '{0:02d}'.format(epoch) + '/{0:02d}'.format(opt.epochs_num) + " | Loss: {0:.5f}".format(loss) + " | Train Score: {0:.5f}".format(train_score) +  " | Val Score: {0:.5f}".format(val_score) + " | " + best_score_str
            logging.info(log_str)


            if epoch - last_best_epoch >= opt.patience_factor:
                stop_log_str = 'Training stopped due to no progress...'
                logging.info(stop_log_str)
                break

        logging.info('Final Val Score: '+ str(best_score))
        


    else: 

        cv_fold_num = 5
        all_results_df = pd.DataFrame([])
        
        if not opt.therapist_independent: 

            cv = StratifiedKFold(n_splits=cv_fold_num, random_state=seed, shuffle=True)

            train_folds, val_folds, test_folds = [], [], []
            for train_val_fold_id, test_fold_id in cv.split(session_ids, labels_crossval):

                train_val = np.array(session_ids)[train_val_fold_id]
                test = np.array(session_ids)[test_fold_id] 

                train, val, _, _ = train_test_split(train_val, labels_crossval.loc[train_val], test_size=0.25, random_state=seed, stratify=labels_crossval.loc[train_val].values) 
                
                test_folds.append(test)
                val_folds.append(val)
                train_folds.append(train)
    
        else: 

            kf = StratifiedGroupKFold(n_splits=cv_fold_num)
            train_folds, val_folds, test_folds = [], [], []
            therapist_ids = outcome['therapist_id'].values

            therapist_ids = utils.combine_single_sess_therapists(therapist_ids)

            for train_val_fold_id, test_fold_id in kf.split(session_ids, labels_crossval['label'].values, therapist_ids):

                train_val = np.array(session_ids)[train_val_fold_id]
                test = np.array(session_ids)[test_fold_id] 

                train_val_ther_id = therapist_ids[train_val_fold_id]

                splitter = GroupShuffleSplit(test_size=0.30, n_splits=1, random_state = seed)
                split = splitter.split(train_val, labels_crossval.loc[train_val]['label'].values, train_val_ther_id)
                train, val = next(split) 

                assert len( set(train_val_ther_id[train]) & set(train_val_ther_id[val]) & set(therapist_ids[test_fold_id]) ) == 0

                train = train_val[train]
                val = train_val[val]

                
                test_folds.append(test)
                val_folds.append(val)
                train_folds.append(train)


        fold_count = 0
        for train_fold_id, val_fold_id,test_fold_id in zip(train_folds, val_folds, test_folds):
            
            print('------------------- split', fold_count, ' -------------------')
            
            test_labels = labels.loc[test_fold_id]
            val_labels = labels.loc[val_fold_id]
            train_labels = labels.loc[train_fold_id]

            print('train/val/test size', len(train_fold_id), len(val_fold_id), len(test_fold_id))
            print('Class Dist-- train:', Counter(train_labels), 'val:', Counter(val_labels), 'test', Counter(test_labels))
                
            
            trial_ids = {'train':train_fold_id, 'val':val_fold_id, 'test':test_fold_id}


            model = BERT_RNN_fromText(opt)
            model = model.to(device)

            
            test = SessionEmbeddingDatasetFromText(trial_ids['test'], dataset_df, test_labels, opt, 'test', opt.quantile)
            val = SessionEmbeddingDatasetFromText(trial_ids['val'], dataset_df, val_labels, opt, 'val', opt.quantile)
            train = SessionEmbeddingDatasetFromText(trial_ids['train'], dataset_df, train_labels, opt, 'train', opt.quantile)
        
            test_dl = torch.utils.data.DataLoader(test, batch_size=opt.batch_size, collate_fn=collate_fn_text, shuffle=False, pin_memory=True)
            val_dl = torch.utils.data.DataLoader(val, batch_size=opt.batch_size, collate_fn=collate_fn_text, shuffle=True, pin_memory=True)
            train_dl = torch.utils.data.DataLoader(train, batch_size=opt.batch_size, collate_fn=collate_fn_text, shuffle=True, pin_memory=True, drop_last=True)
                        
            loss_func = utils.CCC_loss()
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate)
            
            last_best_epoch = -1
            best_score = float('-inf')
            for epoch in range(opt.epochs_num):
                
                loss, train_score = train_model(model, train_dl, optimizer, loss_func, device, opt)
                _ , val_score, _ = validate_model(model, val_dl, loss_func, device, opt)
                
                best_score_str =  ""

                if val_score > best_score: 
                    best_score = val_score
                    _, test_score, result_df = validate_model(model, test_dl, loss_func, device, opt)
                    result_df['fold'] = [fold_count] * result_df.shape[0]
                    best_score_str = 'New found best! | test score: {0:.5f}'.format(test_score)
                    last_best_epoch = epoch
                                
                log_str = 'Epoch: ' + '{0:02d}'.format(epoch) + '/{0:02d}'.format(opt.epochs_num) + " | Loss: {0:.5f}".format(loss) + " | Train Score: {0:.5f}".format(train_score) +  " | Val Score: {0:.5f}".format(val_score) + " | " + best_score_str
                logging.info(log_str)


                if epoch - last_best_epoch >= opt.patience_factor:
                    stop_log_str = 'Training stopped due to no progress...'
                    logging.info(stop_log_str)
                    break
            

            all_results_df = pd.concat([all_results_df, result_df], axis=0)

            final_log_str = 'Final Test Score for fold {0:01d} :'.format(fold_count) + ' {0:.3f}'.format(test_score) + '\n ******************************************** \n'
            logging.info(final_log_str)
            
            fold_count += 1
            

    all_results_df.set_index('ID', inplace=True)
    if not opt.cross_corpus: 
        all_results_df = pd.concat([all_results_df, outcome_df['dataset']], axis=1)
    all_results_df.sort_index(inplace=True)
    all_results_df.to_csv(os.path.join(output_dir, 'predictions.csv'), header=True, index=True)

    args_dict = {k: str(v) for k, v in asdict(opt).items()}
    with open(os.path.join(output_dir, 'args.pkl'), 'wb') as f:
        pickle.dump(args_dict, f)
        
    full_f1_score = utils.get_score(all_results_df['labels'], all_results_df['predictions'], opt)
    logging.info('Final Score {0:.3f}'.format(full_f1_score))


data_path = 'data/combined_prime_next_withTime_manualRevision.tsv'
outcome_path = 'data/combined_prime_next_outcomes.csv'

run_model(data_path, outcome_path)
    



