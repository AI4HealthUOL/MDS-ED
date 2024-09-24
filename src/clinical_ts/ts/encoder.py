__all__ = ['NoEncoder', 'NoEncoderConfig']

import torch
from torch import nn
import numpy as np

import dataclasses
from dataclasses import dataclass, field
from typing import List

from ..template_modules import EncoderBase, EncoderBaseConfig
from .basic_conv1d_modules.basic_conv1d import _conv1d
from .transformer_modules.transformer import TransformerConvStemTokenizer

class NoEncoder(EncoderBase):
    def __init__(self, hparams_encoder, hparams_input_shape):
        '''
        no encoder- flattens by default if multiple channels are passed
        '''
        super().__init__(hparams_encoder, hparams_input_shape)
        self.timesteps_per_token = hparams_encoder.timesteps_per_token
        self.sequence_last = hparams_input_shape.sequence_last
        self.input_channels = hparams_input_shape.channels if hparams_input_shape.channels2==0 else hparams_input_shape.channels*hparams_input_shape.channels2
        
        self.output_shape = dataclasses.replace(hparams_input_shape)
        self.output_shape.channels = self.input_channels*self.timesteps_per_token
        self.output_shape.channels2 = 0
        self.output_shape.length = hparams_input_shape.length//self.timesteps_per_token
        self.output_shape.sequence_last = False
    
    def forward(self, **kwargs):
        seq = kwargs["seq"] #bs,channels,freq,seq
        if(not self.sequence_last):
            seq = torch.movedim(seq,1,-1)
        if(len(seq.size())==4):#spectrogram input
            seq = seq.view(seq.size(0),-1,seq.size(-1))#flatten  
          
        if(self.timesteps_per_token==1):
            return {"seq": seq.transpose(1,2)}
        else:
            assert(seq.size(-1)%self.timesteps_per_token==0)
            size = seq.size()
            return {"seq": seq.view(size[0],-1,seq.shape[-1]).transpose(1,2).reshape(size[0],size[2]//self.timesteps_per_token,-1).transpose(1,2)}

    def get_output_shape(self):
        return self.output_shape


@dataclass
class NoEncoderConfig(EncoderBaseConfig):
    _target_:str = "clinical_ts.ts.encoder.NoEncoder"
