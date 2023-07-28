import torch 
from transformers import AutoModel, AutoModelForMaskedLM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BERT_RNN_fromText(torch.nn.Module):
    def __init__(self, opt):
        super(BERT_RNN_fromText, self).__init__()
 

        self.bert = AutoModel.from_pretrained(opt.encoder)
        input_dim = self.bert.config.hidden_size

        for param in self.bert.parameters():
            param.requires_grad = False

        if opt.finetune:
            for param in self.bert.pooler.parameters():
                param.requires_grad = True
            for param in self.bert.encoder.layer[-1].parameters():
                param.requires_grad = True

            if opt.finetune == 2: 
                for param in self.bert.encoder.layer[-2].parameters():
                    param.requires_grad = True
        
        self.pooling_type = opt.encoder_pooling
        self.selfattn = opt.selfattn
        
        self.downsize_linear = torch.nn.Linear(input_dim, opt.hidden_dim_1)
        self.GRU = torch.nn.GRU(opt.hidden_dim_1, opt.hidden_dim_2, num_layers=opt.nlayer, bidirectional=opt.bidirectional, batch_first=True)

        classifier_hidden_dim = opt.hidden_dim_2 * 2 if opt.bidirectional else opt.hidden_dim_2
        classifier_hidden_dim = classifier_hidden_dim * 2 if (self.selfattn and opt.attn_pooling == 'mean_max') else classifier_hidden_dim

        self.output_dim = 1 
        self.linear = torch.nn.Linear(classifier_hidden_dim, self.output_dim)
        
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=opt.dropout)


        if self.selfattn:
            self.attn_pooling = opt.attn_pooling
            self.attnfunc = torch.nn.MultiheadAttention(opt.hidden_dim_2 * 2 if opt.bidirectional else opt.hidden_dim_2, num_heads=opt.attn_heads, batch_first=True)


    def forward(self, input_ids, attn_mask, lengths): 
        

        batch_size, utt_count, token_count = input_ids.shape        
        input_ids = input_ids.reshape(-1, token_count)  # (bs*chunk_size, emb_dim)
        attn_mask = attn_mask.reshape(-1, token_count)
        
        #https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions
        # shape (180, 512, 768) ~ (seq_length, token_length, embedding_dim)

        embeddings = self.bert(input_ids, attention_mask=attn_mask, output_hidden_states=False)


        if self.pooling_type == 'mean':
            embeddings = embeddings.pooler_output

        elif self.pooling_type == 'cls':
            embeddings = embeddings.last_hidden_state
            embeddings = embeddings[:,0,:]

        embeddings = self.downsize_linear(embeddings)
        
        nhid = embeddings.shape[-1]
        embeddings = embeddings.reshape(batch_size, -1, nhid)        
           
        embeddings_packed = pack_padded_sequence(embeddings, lengths, enforce_sorted=False, batch_first=True)
        self.GRU.flatten_parameters()
        outputs, _ = self.GRU(embeddings_packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)  # (bs,chunk_size/chunk_len, emb_dim)

        if self.selfattn: 
            outputs, _ = self.attnfunc(outputs, outputs, outputs)
            if self.attn_pooling == 'mean':
                outputs = torch.mean(outputs, dim=1)
            elif self.attn_pooling == 'max':
                outputs, _ = torch.max(outputs, dim=1)
            elif self.attn_pooling == 'mean_max': 
                outputs_max, _ = torch.max(outputs, dim=1)
                outputs_mean = torch.mean(outputs, dim=1)
                outputs = torch.cat((outputs_mean, outputs_max), dim=1)
        else: 

            outputs = outputs[:, -1, :]


        outputs = self.dropout(outputs)
        outputs_1 = self.linear(outputs) 
        
        
        return outputs_1
