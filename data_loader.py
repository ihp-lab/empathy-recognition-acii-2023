from datasets import Dataset 
from transformers import AutoTokenizer
import torch, random
import numpy as np

class SessionEmbeddingDatasetFromText(Dataset):

    def __init__(self, ids, dataset, labels, opt, split_type, quantile):

        assert np.array_equal(ids, labels.index)

        self.ids = ids
        self.dataset = dataset[dataset['session'].isin(ids)]
        self.dataset = self.dataset[self.dataset['speaker']=='I']


        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(opt.encoder) 
        self.input_ids, self.attn_mask, self.lengths= {}, {}, {}

        for sess_id in self.ids: 

            sess_df = self.dataset[self.dataset['session']==sess_id]

            if not quantile: 
                #extracting window from the session end
                sess_df = sess_df.iloc[-opt.max_sess_length:][['normed_text', 'session']]
                assert sess_df.shape[0] <= opt.max_sess_length

            else: 

                sess_df = sess_df[sess_df['quantile']==quantile]
                sess_df = sess_df.iloc[:opt.max_sess_length][['normed_text', 'session']]

            
            utts = sess_df['normed_text'].tolist()
            encodings = self.tokenizer(utts, truncation=True, padding=True)

            _input_ids = encodings['input_ids']
            _attn_mask = encodings['attention_mask']
            _max_seq_length = len(_input_ids[0]) # all the sequences are padded to the longest seq length
            
            input_ids_tensor = torch.zeros((len(_input_ids), _max_seq_length)).long()
            attn_mask_tensor = torch.zeros((len(_attn_mask), _max_seq_length)).long()
            
            assert len(_input_ids) == len(_attn_mask) 
            
            for idx, (_input, _attn) in enumerate(zip(_input_ids, _attn_mask)):

                input_ids_tensor[idx] = torch.tensor(_input)
                attn_mask_tensor[idx] = torch.tensor(_attn)
            
            self.input_ids[sess_id] = input_ids_tensor
            self.attn_mask[sess_id] = attn_mask_tensor
            self.lengths[sess_id] = _max_seq_length
            
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        
        sess_id = self.ids[idx]
        label = torch.tensor(self.labels[sess_id])
        

        item = {
            'input_ids': self.input_ids[sess_id],
            'attn_mask': self.attn_mask[sess_id],
            'labels': label,
            'session_ids': sess_id
        }

        return item

def collate_fn_text(data): 
    """
    data: is a list of tuples with (example, label, length)
        where 'example' is a tensor of arbitrary shape
        and label/length are scalars
    """    
    
    lengths_dim1 = list(map( lambda n: n['input_ids'].shape[0] , data)) # length of the sequence (session utt count)
    lengths_dim2 = list(map( lambda n: n['input_ids'].shape[1] , data)) # length of the longest utterance (token count)
    
    batch_size = len(data)
    
    labels = torch.tensor([item['labels'] for item in data])
    
    session_ids = [item['session_ids'] for item in data]

    input_ids_tensor = torch.zeros(batch_size, max(lengths_dim1), max(lengths_dim2)).long()
    attn_mask_tensor = torch.zeros(batch_size, max(lengths_dim1), max(lengths_dim2)).long()


    for idx, (item, length1, length2) in enumerate(zip(data, lengths_dim1, lengths_dim2)):

        input_ids_tensor[idx,:length1, :length2] = item['input_ids']
        attn_mask_tensor[idx,:length1, :length2] = item['attn_mask']
            

    return {'input_ids': input_ids_tensor, 
            'attn_mask': attn_mask_tensor, 
            'lengths': lengths_dim1, 
            'labels': labels, 
            'session_ids': session_ids}


