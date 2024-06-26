from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import numpy as np
from torch import nn
from transformers.file_utils import ModelOutput


@dataclass
class SimpleOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    logits:torch.FloatTensor = None

def sinusoidal_init(num_embeddings: int, embedding_dim: int):
    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * i / embedding_dim) for i in range(embedding_dim)]
        if pos != 0 else np.zeros(embedding_dim) for pos in range(num_embeddings)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)



class HierarchicalConvolutionalBert(nn.Module):
    def __init__(self, encoder, max_segments=[32, 16, 8], max_segment_length=[128, 256, 512],num_labels=None):
        super(HierarchicalConvolutionalBert, self).__init__()

        supported_models = ['bert','distilbert','roberta']
        assert encoder.config.model_type in supported_models 

        self.encoder = encoder

        self.hidden_size = encoder.config.hidden_size
        self.max_segments = max_segments
        self.max_segment_length = max_segment_length
        self.num_labels=num_labels
        # Init sinusoidal positional embeddings
        self.seg_pos_embeddings = nn.Embedding(max_segments[0] + 1, self.hidden_size,
                                               padding_idx=0,
                                               _weight=sinusoidal_init(max_segments[0] + 1, self.hidden_size))


        if encoder.config.model_type == 'distilbert':
            self.seg_encoder = nn.Transformer(d_model=encoder.config.dim,
                                  nhead=encoder.config.n_heads,
                                  batch_first=True, dim_feedforward=4*encoder.config.dim,
                                  dropout=encoder.config.dropout,
                                  activation="gelu",
                                  num_encoder_layers=2, num_decoder_layers=0).encoder
        else:
            self.seg_encoder = nn.Transformer(d_model=self.hidden_size,
                                            nhead=encoder.config.num_attention_heads,
                                            batch_first=True, dim_feedforward=encoder.config.intermediate_size,
                                            activation=encoder.config.hidden_act,
                                            dropout=encoder.config.hidden_dropout_prob,
                                            layer_norm_eps=encoder.config.layer_norm_eps,
                                            num_encoder_layers=2, num_decoder_layers=0).encoder
        self.vector_transformer=nn.Transformer(d_model=self.hidden_size,
                                            nhead=encoder.config.num_attention_heads,
                                            batch_first=True, dim_feedforward=encoder.config.intermediate_size,
                                            activation=encoder.config.hidden_act,
                                            dropout=encoder.config.hidden_dropout_prob,
                                            layer_norm_eps=encoder.config.layer_norm_eps,
                                            num_encoder_layers=1, num_decoder_layers=0).encoder


        if encoder.config.model_type == 'bert':
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_size, self.num_labels)
            )
        elif encoder.config.model_type == 'distilbert':
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Linear(self.hidden_size, self.num_labels),
                nn.Dropout(0.2)
            )
        elif encoder.config.model_type == 'roberta':
            self.classifier = nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.Dropout(0.1),
                    nn.Linear(self.hidden_size, self.num_labels)
            )

    def forward(self, input_ids_1=None, attention_mask_1=None, token_type_ids_1=None, 
                input_ids_2=None, attention_mask_2=None, token_type_ids_2=None,
                input_ids_3=None, attention_mask_3=None, token_type_ids_3=None,labels=None,**kwargs):

        output_list = []
        for i in range(1,4):
            input_ids_i = eval("input_ids_" + str(i))
            seg_mask = (torch.sum(input_ids_i, 2) != 0).to(input_ids_i.dtype)
            seg_positions = torch.arange(1, self.max_segments[i-1] + 1).to(input_ids_i.device) * seg_mask

            #batch=input_ids_i.size(0)
            attention_mask_i = eval("attention_mask_" + str(i))
            token_type_ids_i = eval("token_type_ids_" + str(i))          
            input_ids_i = input_ids_i.contiguous().view(-1, input_ids_i.size(-1))
            attention_mask_i = attention_mask_i.contiguous().view(-1, attention_mask_i.size(-1))
            if token_type_ids_i is not None:
                token_type_ids_i = token_type_ids_i.contiguous().view(-1, token_type_ids_i.size(-1))
            else:
                token_type_ids_i=None
            
            if self.encoder.config.model_type == 'bert':
                encoder_outputs = self.encoder(input_ids=input_ids_i,
                                            attention_mask=attention_mask_i,
                                            token_type_ids=token_type_ids_i)[0]
            else:
                encoder_outputs = self.encoder(input_ids=input_ids_i,
                                            attention_mask=attention_mask_i)[0]
            encoder_outputs = encoder_outputs.contiguous().view(-1, self.max_segments[i-1],
                                                                self.max_segment_length[i-1],
                                                                self.hidden_size)#batch*seg*seg_len*hidden
            segment_vector = encoder_outputs.mean(dim=2)#batch*seg*hidden
            transformed_segment_vector = self.vector_transformer(segment_vector)
            encoder_outputs = encoder_outputs[:, :, 0]#batch*seg*hidden

            encoder_outputs = encoder_outputs+self.seg_pos_embeddings(seg_positions)
    
            seg_encoder_outputs = self.seg_encoder(encoder_outputs)#batch*seg*hidden
            seg_encoder_outputs = seg_encoder_outputs + transformed_segment_vector
            outputs,_= torch.max(seg_encoder_outputs, 1)#batch*hidden
            output_list.append(outputs)#3*batch*hidden

        output = torch.mean(torch.stack(output_list), dim=0) #batch*(hidden+10)
        logits = self.classifier(output)#batch*num_labels
        return SimpleOutput(logits=logits, last_hidden_state=logits)


