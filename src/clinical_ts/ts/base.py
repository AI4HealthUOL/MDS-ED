__all__ = ['NoPredictor', 'NoPredictorConfig', 'CNNPredictor', 'CNNPredictorConfig']

import dataclasses
from dataclasses import dataclass, field
from typing import List

import torch.nn as nn
import numpy as np
from .basic_conv1d_modules.basic_conv1d import _conv1d

from ..template_modules import PredictorBase, PredictorBaseConfig

class NoPredictor(PredictorBase):
    def __init__(self, hparams, hparams_input_shape):
        '''
        no predictor e.g. for pretraining purposes
        '''
        super().__init__(hparams, hparams_input_shape)
        
    def forward(self, **kwargs):    
        return {}
    

@dataclass
class NoPredictorConfig(PredictorBaseConfig):
    _target_:str = "clinical_ts.ts.base.NoPredictor"


class CNNPredictor(PredictorBase):
    def __init__(self, hparams_encoder, hparams_input_shape):
        '''this is a reduced version of the RNNEncoder'''
        super().__init__(hparams_encoder, hparams_input_shape)
        assert(not hparams_input_shape.sequence_last)
        assert(len(hparams_encoder.strides)==len(hparams_encoder.kss) and len(hparams_encoder.strides)==len(hparams_encoder.features) and len(hparams_encoder.strides)==len(hparams_encoder.dilations))
        lst = []
        for i,(s,k,f,d) in enumerate(zip(hparams_encoder.strides,hparams_encoder.kss,hparams_encoder.features,hparams_encoder.dilations)):
            lst.append(_conv1d(hparams_input_shape.channels if i==0 else hparams_encoder.features[i-1],f,kernel_size=k,stride=s,dilation=d,bn=hparams_encoder.normalization,layer_norm=hparams_encoder.layer_norm))
           
        self.layers = nn.Sequential(*lst)
        self.downsampling_factor = np.prod(hparams_encoder.strides)
        
        self.output_dim = hparams_encoder.features[-1]

        self.output_shape = dataclasses.replace(hparams_input_shape)
        self.output_shape.channels = self.output_dim
        self.output_shape.length = int(hparams_input_shape.length//self.downsampling_factor+ (1 if hparams_input_shape.length%self.downsampling_factor>0 else 0))

    def get_output_shape(self):
        return self.output_shape

    def forward(self, **kwargs):
        seq =  kwargs["seq"].transpose(1,2)
        return {"seq": self.layers(seq).transpose(1,2)}#bs,seq,feat
    
@dataclass
class CNNPredictorConfig(PredictorBaseConfig):
    _target_:str = "clinical_ts.ts.base.CNNPredictor"

    strides:List[int]=field(default_factory=lambda: [1,1,1,1]) #help="encoder strides (space-separated)")
    kss:List[int]=field(default_factory=lambda: [1,1,1,1]) #help="encoder kernel sizes (space-separated)")
    features:List[int]=field(default_factory=lambda: [512,512,512,512]) #help="encoder features (space-separated)")
    dilations:List[int]=field(default_factory=lambda: [1,1,1,1]) #help="encoder dilations (space-separated)")
    normalization:bool=True #help="disable encoder batch/layer normalization")
    layer_norm:bool=False#", action="store_true", help="encoder layer normalization")


    