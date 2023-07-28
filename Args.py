from dataclasses import dataclass
from typing import Optional

@dataclass
class Args:

    hidden_dim_1: Optional[int] = 512
    hidden_dim_2: Optional[int] = 256

    learning_rate: Optional[float] = 5e-4

    epochs_num: Optional[int] = 50
    patience_factor: Optional[int] = 10
    batch_size: Optional[int] = 8

    max_sess_length: Optional[int] = 64
    quantile: Optional[int] = None  #0|1|2|3
        
    nlayer: Optional[int] = 1
    dropout: Optional[float] = 0.5
    bidirectional: Optional[bool] = False
    finetune: Optional[int] = 1 # 0 | 1 | 2
        

    device_id: Optional[int] = 0

    model: Optional[str] = 'emo-roberta' # roberta | distil-roberta | emo-roberta

    encoder_pooling: Optional[str] = 'cls'
    
    selfattn: Optional[bool] = True
    attn_pooling: Optional[str] = 'mean_max'
    attn_heads: Optional[int] = 2   
    
    output_filename: Optional[str] = 'EXPERIMENT_FOLDER'

    global_type: Optional[str] = 'Empathy' 
    dataset: Optional[str] = 'combined' #[PRIME|NEXT|combined]
    therapist_independent: Optional[bool] = False

    cross_corpus: Optional[bool] = False
    cross_corpus_train_q: Optional[int] = None
    cross_corpus_val_q: Optional[int] = None


    