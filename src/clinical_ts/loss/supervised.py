__all__ = ['SupervisedLossConfig', 'BCELossConfig', 'BinaryCrossEntropyFocalLoss', 'BCEFLossConfig']

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from dataclasses import dataclass, field
from typing import List

from ..template_modules import LossConfig

####################################################################################
# BASIC supervised losses
###################################################################################
@dataclass
class SupervisedLossConfig(LossConfig):
    _target_:str = "" #insert appropriate loss class
    loss_type:str ="supervised"
    supervised_type:str="classification_single"#"classification_multi","regression_quantile"                    

@dataclass
class BCELossConfig(SupervisedLossConfig):
    _target_:str= "clinical_ts.loss.supervised.BinaryCrossEntropyLoss"
    loss_type:str="supervised"
    supervised_type:str="classification_multi"
    pos_weight:List[float]=field(default_factory=lambda: [])#class weights e.g. inverse class prevalences
    ignore_nans:bool=False #ignore nans- requires separate BCEs for each label

class BinaryCrossEntropyFocalLoss(nn.Module):
    """
    Focal BCE loss for binary classification with labels of 0 and 1
    """
    def __init__(self, hparams_loss):
        super().__init__()
        self.gamma = hparams_loss.gamma
        
        self.ignore_nans = hparams_loss.ignore_nans
        self.pos_weight_set = len(hparams_loss.pos_weight)>0

        if(not self.ignore_nans):
            self.bce = torch.nn.BCEWithLogitsLoss(reduction="none",pos_weight=torch.from_numpy(np.array(hparams_loss.pos_weight,dtype=np.float32)) if len(hparams_loss.pos_weight)>0 else None)
        else:
            if(self.pos_weight_set):
                self.bce = torch.nn.ModuleList([torch.nn.BCEWithLogitsLoss(reduction="none",pos_weight=torch.from_numpy(np.array([hparams_loss.pos_weight[i]],dtype=np.float32))) for i in range(len(self.pos_weight_set))])
            else:
                self.bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, preds, targs):
        if(not(self.ignore_nans)):
            probs = torch.sigmoid(preds)
            p_t = probs * targs + (1 - probs) * (1 - targs)
            focal_modulation = torch.pow((1 - p_t), self.gamma)
            # mean aggregation
            return (focal_modulation * self.bce(input=preds, target=targs.float())).sum(-1).mean()
        else:
            losses = []
            for i in range(preds.size(1)):
                predsi = preds[:,i]
                targsi = targs[:,i]
                maski = ~torch.isnan(targsi)
                predsi = predsi[maski]
                targsi = targsi[maski]
                if(len(predsi)>0):
                    probsi = torch.sigmoid(predsi)
                    p_ti = probsi * targsi + (1 - probsi) * (1 - targsi)
                    focal_modulationi = torch.pow((1 - p_ti), self.gamma)
                    if(self.pos_weight_set):
                        losses.append(torch.mean(focal_modulationi*self.bce[i](predsi,targsi)))
                    else:
                        losses.append(torch.mean(focal_modulationi*self.bce(predsi,targsi)))
                
            return torch.sum(torch.stack(losses)) if(len(losses)>0) else 0.
        
@dataclass
class BCEFLossConfig(BCELossConfig):
    _target_:str= "clinical_ts.loss.supervised.BinaryCrossEntropyFocalLoss"
    gamma:float=2.