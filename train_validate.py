import torch 
import numpy as np
from sklearn.metrics import f1_score
import pandas as pd
from sklearn.metrics import r2_score
import utils

def train_model(model, train_dl, optimizer, loss_func, device, opt):
    
    all_labels, all_predictions = [], []

    model.train()
    total_train_loss = 0.0 
    for batch_idx, batch in enumerate(train_dl):
        
        optimizer.zero_grad()
        
        input_ids, attn_mask, lengths, labels = batch['input_ids'], batch['attn_mask'], batch['lengths'], batch['labels']

        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)

        logits = model.forward(input_ids, attn_mask, lengths)
        labels = labels.to(device)

        loss = utils.get_loss(logits, labels, loss_func, opt)                        

        loss.backward()
        optimizer.step()


        with torch.no_grad(): 

            predictions = utils.get_predictions(logits, opt)

            all_predictions += predictions
            
            labels = labels.detach().cpu().numpy()
            labels = labels.reshape(1,-1).tolist()[0]
            all_labels += labels
                
            total_train_loss += loss.item()

    score = utils.get_score(all_labels, all_predictions, opt)                

    return total_train_loss/(batch_idx+1), score

def validate_model(model, dl, loss_func, device, opt):
    
    all_labels, all_predictions, all_sess_ids = [], [], []
    
    total_loss = 0.0
    model.eval()
    with torch.no_grad(): 
        for batch_idx, batch in enumerate(dl):

            input_ids, attn_mask, lengths, labels, sess_ids = batch['input_ids'], batch['attn_mask'], batch['lengths'], batch['labels'], batch['session_ids']

            labels = labels.to(device)
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)

            logits = model.forward(input_ids, attn_mask, lengths)


            loss = utils.get_loss(logits, labels, loss_func, opt)

            total_loss += loss.item()

            predictions = utils.get_predictions(logits, opt)
            labels = labels.detach().cpu().numpy()

            labels = labels.reshape(1,-1).tolist()[0]
            all_labels += labels
            all_predictions += predictions
            all_sess_ids += sess_ids


    score = utils.get_score(all_labels, all_predictions, opt)
    results_df = pd.DataFrame({'labels':all_labels, 'predictions':all_predictions, 'ID':all_sess_ids})
        
    return total_loss/(batch_idx+1), score, results_df