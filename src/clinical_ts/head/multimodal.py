__all__ = ['ConcatFusionHead', 'ConcatFusionHeadConfig']

import torch
import torch.nn as nn

from ..template_modules import HeadBase, HeadBaseConfig
import dataclasses
from dataclasses import dataclass

class ConcatFusionHead(HeadBase):
    def __init__(self, hparams_head, hparams_input_shape, target_dim):
        '''Simple concatenation plus linear head'''
        super().__init__(hparams_head, hparams_input_shape, target_dim)

        input_size = (hparams_input_shape.length*hparams_input_shape.channels if hparams_input_shape.length>0 else hparams_input_shape.channels)+hparams_input_shape.static_dim+hparams_input_shape.static_dim_cat
        self.linear = nn.Linear(input_size,target_dim)
        self.output_shape = dataclasses.replace(hparams_input_shape)
        self.output_shape.channels = target_dim
        self.output_shape.length = 0
        self.output_shape.static_dim = 0
        self.output_shape.static_dim_cat = 0
        
    def forward(self, **kwargs):
        static = kwargs["static"]
        seq = kwargs["seq"]
        return {"seq": self.linear(torch.cat((seq.view(seq.shape[0],-1),static),dim=1))}
    
    def get_output_shape(self):
        return self.output_shape

@dataclass
class ConcatFusionHeadConfig(HeadBaseConfig):
    _target_:str = "clinical_ts.head.multimodal.ConcatFusionHead"
    multi_prediction:bool=False
