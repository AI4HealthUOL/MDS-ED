__all__ = ['PoolingHead', 'PoolingHeadConfig']

import torch
import torch.nn as nn
import numpy as np

from ..template_modules import HeadBase, HeadBaseConfig, _string_to_class
import dataclasses
from dataclasses import dataclass, field
from typing import List
    
class PoolingHead(HeadBase):
    def __init__(self, hparams_head, hparams_input_shape, target_dim):
        super().__init__(hparams_head, hparams_input_shape, target_dim)
        #assert(target_dim is None or hparams_head.output_layer is True)
        if(target_dim is not None and hparams_head.output_layer is False):
            print("Warning: target_dim",target_dim,"is passed to PoolingHead but output_layer is False. target_dim will be ignored.")
        self.local_pool = hparams_head.multi_prediction
        self.output_dim = hparams_input_shape.channels if not hparams_head.output_layer else target_dim
        
        if(self.local_pool):#local pool
            self.local_pool_padding = (hparams_head.local_pool_kernel_size-1)//2
            self.local_pool_kernel_size = hparams_head.local_pool_kernel_size
            self.local_pool_stride = hparams_head.local_pool_kernel_size if hparams_head.local_pool_stride==0 else hparams_head.local_pool_stride
            if(hparams_head.local_pool_max):
                self.pool = torch.nn.MaxPool1d(kernel_size=hparams_head.local_pool_kernel_size,stride=hparams_head.local_pool_stride if hparams_head.local_pool_stride!=0 else hparams_head.local_pool_kernel_size,padding=(hparams_head.local_pool_kernel_size-1)//2)
            else:
                self.pool = torch.nn.AvgPool1d(kernel_size=hparams_head.local_pool_kernel_size,stride=hparams_head.local_pool_stride if hparams_head.local_pool_stride!=0 else hparams_head.local_pool_kernel_size,padding=(hparams_head.local_pool_kernel_size-1)//2)        
        else:#global pool
            if(hparams_head.local_pool_max):
                self.pool = torch.nn.AdaptiveMaxPool1d(1)
            else:
                self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(hparams_input_shape.channels, target_dim) if hparams_head.output_layer else nn.Identity()

        self.output_shape = dataclasses.replace(hparams_input_shape)
        self.output_shape.channels = self.output_dim
        #assert(hparams.predictor._target_!="clinical_ts.ts.transformer.TransformerPredictor" or (hparams.predictor.cls_token is True or (hparams.predictor.cls_token is False and (hparams_head.head_pooling_type!="cls" and hparams_head.head_pooling_type!="meanmax-cls"))))

        self.output_shape.length = int(np.floor((hparams_input_shape.length + 2*self.local_pool_padding- self.local_pool_kernel_size)/self.local_pool_stride+1)) if self.local_pool else 0

    def forward(self, **kwargs):
        seq = kwargs["seq"]
        #input has shape B,S,E
        seq = seq.transpose(1,2) 
        seq = self.pool(seq)
        return {"seq": self.linear(seq.transpose(1,2))}#return B,S,E
    
    def get_output_shape(self):
        return self.output_shape

@dataclass
class PoolingHeadConfig(HeadBaseConfig):
    _target_:str = "clinical_ts.ts.head.PoolingHead"
    
    multi_prediction:bool = False #local pool vs. global pool
    local_pool_max:bool = False #max pool vs. avg pool
    local_pool_kernel_size: int = 0 #kernel size for local pooling
    local_pool_stride: int = 0 #kernel_size if 0
    #local_pool_padding=(kernel_size-1)//2
    output_layer: bool = False
