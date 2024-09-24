__all__ = ['BaseModel','SSLModel']

###############
#generic
import lightning.pytorch as lp
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn

from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
from itertools import chain
from operator import attrgetter
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
try:
    import pickle5 as pickle
except ImportError as e:
    import pickle
import os

import dataclasses
#################
#specific

from .data.time_series_dataset import TimeSeriesDataset, TimeSeriesDatasetConfig, ConcatTimeSeriesDataset
from .data.time_series_dataset_transforms import SequenceToSampleLabelTransform, Transform, Normalize, ChannelFilter, ToTensor
from .data.time_series_dataset_utils import load_dataset,reformat_as_memmap

from .template_modules import _string_to_class, ShapeConfig, SequentialTimeSeriesEncoder
from .utils.callbacks import ForwardHook, freeze_bn_stats, sanity_check
from .utils.schedulers import get_constant_schedule, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_invsqrt_decay_schedule_with_warmup, get_linear_schedule_with_warmup

class BaseModel(lp.LightningModule):
    '''Model base class provides basic functionality: data loading, score evaluation etc'''

    def __init__(self, hparams):
        #fix default chunk_length and stride hyperparameters
        if(hparams.base.input_size_dl==0):
            hparams.base.input_size_dl = hparams.base.input_size
        if(hparams.base.chunk_length_train==-1):
            hparams.base.chunk_length_train = hparams.base.input_size_dl
        if(hparams.base.stride_train==-1):
            hparams.base.stride_train = hparams.base.chunk_length_train
        if(hparams.base.chunk_length_val==-1):
            hparams.base.chunk_length_val = hparams.base.input_size_dl
        if(hparams.base.stride_val==-1):
            hparams.base.stride_val = hparams.base.chunk_length_val
        if(hparams.base.chunk_length_valtest==-1):
            hparams.base.chunk_length_valtest = hparams.base.input_size_dl
        if(hparams.base.stride_valtest==-1):
            hparams.base.stride_valtest = hparams.base.chunk_length_valtest
        if(hparams.base.stride_export==-1):
            hparams.base.stride_export = hparams.base.input_size_dl

        super(BaseModel, self).__init__()

        #identify dataset keys
        self.dataset_keys = [d for d in hparams.keys() if ("_target_" in hparams[d].keys() and hparams[d]["_target_"]=="clinical_ts.config.BaseConfigData")]
        assert(len(self.dataset_keys)>0)

        if(not(len(hparams[self.dataset_keys[0]].input_channels_filter)==0 or hparams.base.input_channels==len(hparams[self.dataset_keys[0]].input_channels_filter))):
            print("WARNING: input_channels do not match number of filtered channels")
        assert(hparams.loss.loss_type=="supervised" or (hparams.trainer.precision==32 or hparams.loss.pretraining_targets==0))#gumbel softmax does not work in fp16
        assert(hparams.loss.loss_type=="supervised" or not(hparams.loss.loss_type=="masked" and hparams.loss.pretraining_targets==0))#masked requires quantizer
        
        if(hparams.loss.loss_type=="supervised" and hparams.trainer.pretrained=="" and hparams.trainer.resume=="" and hparams.base.discriminative_lr_factor!=1):
            print("INFO: Setting discriminative-lr-factor=1 for training from scratch.")
            hparams.base.discriminative_lr_factor = 1

        self.lr = hparams.base.lr[0] if len(hparams.base.lr)==1 else list(hparams.base.lr)
        self.save_hyperparameters(hparams)
        
        if(hparams.loss._target_==""):
            self.loss_type = ""
            self.criterion = None
        else:
            self.loss_type = hparams.loss.loss_type
        
            if(hparams.loss._target_.startswith("torch.nn.functional.")):#binary_cross_entropy_with_logits or cross_entropy or mse_loss
                self.criterion = _string_to_class(hparams.loss._target_)
            else:
                self.criterion = _string_to_class(hparams.loss._target_)(hparams.loss)
        
        #changed behavior in lightning 2.0 https://github.com/Lightning-AI/lightning/pull/16520 https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#hooks 
        self.val_preds = [[]]
        self.val_targs = [[]]
        self.test_preds = [[],[]] if self.loss_type == "supervised" else []
        self.test_targs = [[],[]] if self.loss_type == "supervised" else []

                
        ##################################
        #set target_dim
        ##################################
        self.lbl_itos = None
        if(self.loss_type == "supervised"):
            def _prefetch_lbl_itos():#load lbl_itos of the first dataset
                _, lbl_itos, _, _ = self.preprocess_dataset(hparams[self.dataset_keys[0]])
                return lbl_itos[hparams[self.dataset_keys[0]].label_filter] if len(hparams[self.dataset_keys[0]].label_filter)>0 else lbl_itos
            self.lbl_itos = _prefetch_lbl_itos()
            self.target_dim = len(hparams.loss.quantiles)*len(self.lbl_itos) if (self.loss_type=="supervised" and hparams.loss.supervised_type == "regression_quantile") else len(self.lbl_itos)
        elif(self.loss_type.startswith("instance_contrastive")):#clip or infonce
            self.target_dim = hparams.loss.target_dim
        else:# other pretraining
            self.target_dim = None

        ###################################
        #prepare metrics
        ###################################
        self.metrics_train_val = [_string_to_class(self.hparams[d]["_target_"])(self.hparams[d],lbl_itos=self.lbl_itos,key_postfix="0",test=False) for d in self.hparams if ("_target_" in self.hparams[d].keys() and self.hparams[d]["_target_"].startswith("clinical_ts.metric."))]
        self.metrics_test_val = [_string_to_class(self.hparams[d]["_target_"])(self.hparams[d],lbl_itos=self.lbl_itos,key_prefix="test",key_postfix="0",test=True) for d in self.hparams if ("_target_" in self.hparams[d].keys() and self.hparams[d]["_target_"].startswith("clinical_ts.metric."))]
        self.metrics_test_test = [_string_to_class(self.hparams[d]["_target_"])(self.hparams[d],lbl_itos=self.lbl_itos,key_prefix="test",key_postfix="1",test=True) for d in self.hparams if ("_target_" in self.hparams[d].keys() and self.hparams[d]["_target_"].startswith("clinical_ts.metric."))]
        
    def forward(self, **kwargs):
        raise NotImplementedError
    
    def is_multi_prediction(self):
        raise NotImplementedError
    
    def _step(self,data_batch, batch_idx, train, test=False, dataloader_idx=0, freeze_bn=False): 
        if(self.loss_type=="supervised"):
            output_dict = self.forward(**data_batch._asdict())
            preds_all = output_dict["input_predicted"]
            #reshape sequence level predictions for loss computation
            preds = preds_all.view(-1,preds_all.shape[-1]) #if self.is_multi_prediction() else preds_all #B*S, Nc (workaround to fix shapes)
            if(self.hparams.loss.supervised_type=="classification_single"):
                targs = data_batch.label.long().view(-1)#casting to long in case labels have another integer type
                if(len(self.hparams[self.dataset_keys[0]].label_filter)>0):#filter out undesired labels
                    preds=preds[targs>=0]
                    targs=targs[targs>=0]
            elif(self.hparams.loss.supervised_type=="classification_multi" or self.hparams.loss.supervised_type.startswith("regression")):
                targs = data_batch.label.float().view(-1,len(self.lbl_itos))#casting to float in case labels have another type
            loss_dict = {k:v for k,v in output_dict.items() if k.startswith("loss")}
            if(len(loss_dict)>0):
                self.log_dict(loss_dict)
            metric_dict =  {k:v for k,v in output_dict.items() if k.startswith("metric")}
            if(len(metric_dict)>0):
                self.log_dict(metric_dict)

            loss = self.criterion(preds,targs)
            self.log("train_loss" if train else ("val_loss"+str(dataloader_idx) if not test else "test_loss"+str(dataloader_idx)), loss)
            if(not(train)):
                if(test):
                    self.test_preds[dataloader_idx].append(preds_all.detach())
                    self.test_targs[dataloader_idx].append(data_batch.label)
                else:
                    self.val_preds[dataloader_idx].append(preds_all.detach())
                    self.val_targs[dataloader_idx].append(data_batch.label)
            return loss
        else:
            if(self.loss_type=="instance_contrastive_sequence"):#instance level loss
                db1 = data_batch[0]._asdict()
                db2 = data_batch[1]._asdict()
                db12 = {k:torch.cat((db1[k],db2[k])) for k in db1.keys()}
                output_dict = self.forward(**db12)
            else:
                output_dict = self.forward(**data_batch._asdict())
            loss_dict = {k:v for k,v in output_dict.items() if k.startswith("loss")}

            if(self.criterion):
                loss_dict.update(self.criterion(**output_dict))
            self.log_dict(loss_dict)

            metric_dict =  {k:v for k,v in output_dict.items() if k.startswith("metric")}
            if(len(metric_dict)>0):
                self.log_dict(metric_dict)
            
            loss = 0
            
            for x in [v for k,v in loss_dict.items() if k.startswith("loss")]:
                loss += x
             
            #if(train):
            #    self.log("loss_quantizer", loss_quantizer)
            #if(self.loss_type == "cpc" or self.loss_type == "masked_rec"):
            #    loss_pretraining, acc = self.criterion(*outputs, eval_acc=True)
            #    self.log("acc_"+self.loss_type if train else "val_acc_"+self.loss_type, acc)
            #elif(self.loss_type=="masked_pred"):
            #    loss_pretraining, acc = self.criterion(outputs[0],data_batch.label,*outputs[2:], eval_acc=True)
            #    self.log("acc_"+self.loss_type if train else "val_acc_"+self.loss_type, acc)
            #elif(self.loss_type=="lm"):
            #    loss_pretraining = self.criterion(*outputs)
            #elif(self.loss_type=="clip"):
            #    loss_pretraining = self.criterion(outputs[0],self.encoder_static_hook.stored)

            #self.log("loss_pretraining" if train else "val_loss_pretraining", loss_pretraining)
            #weighting factor only set if quantizer is enabled
            #loss = loss_pretraining if self.quantizer is None else loss_pretraining + self.quantizer.#loss_weight*loss_quantizer
            self.log("train_loss" if train else "val_loss", loss)
            
            return loss
      
    def training_step(self, train_batch, batch_idx):
        if(self.hparams.base.linear_eval):
            freeze_bn_stats(self)
        return self._step(train_batch,batch_idx,train=True,test=False,freeze_bn=self.hparams.base.linear_eval)
        
    def validation_step(self, val_batch, batch_idx, dataloader_idx=0):
        return self._step(val_batch,batch_idx,train=False,test=False,dataloader_idx=dataloader_idx)

    def test_step(self, val_batch, batch_idx, dataloader_idx=0):
        return self._step(val_batch,batch_idx,train=False,test=True,dataloader_idx=dataloader_idx)

    def on_fit_start(self):
        if(self.hparams.trainer.pretrained!=""):
            print("Loading pretrained weights from",self.hparams.trainer.pretrained,"pretrained_keys_filter:",self.hparams.trainer.pretrained_keys_filter)
            self.load_weights_from_checkpoint(self.hparams.trainer.pretrained,self.hparams.trainer.pretrained_keys_filter)
                
        if(self.hparams.base.linear_eval):
            print("copying state dict before training for sanity check after training")   
            self.state_dict_pre = copy.deepcopy(self.state_dict().copy())

    def on_fit_end(self):
        if(self.hparams.base.linear_eval):
            sanity_check(self,self.state_dict_pre)

    def on_validation_epoch_end(self):
        self.on_valtest_epoch_end({"preds":self.val_preds[0], "targs":self.val_targs[0]}, test=False)
        self.val_preds[0].clear()
        self.val_targs[0].clear()

    def on_test_epoch_end(self):
        for i in range(len(self.test_preds)):
            self.on_valtest_epoch_end({"preds":self.test_preds[i], "targs":self.test_targs[i]}, test=True, dataloader_idx=i)
            self.test_preds[i].clear()
            self.test_targs[i].clear()

    def on_valtest_epoch_end(self, outputs, test, dataloader_idx=0):
        if(self.hparams.loss.loss_type=="supervised"):
            
            results_dict = {}
            
            ds = self.val_dataset if test is False else (self.test_dataset_val if dataloader_idx==0 else self.test_dataset_test)
            if(self.is_multi_prediction()):
                if(test and self.hparams.base.aggregate_strided_multi_predictions and not self.trainer.sanity_checking):#aggregate multi-predictions
                    
                    preds = torch.cat(outputs["preds"]).cpu().float()# preds have shape bs,seq,classes
                    if(self.hparams.loss.supervised_type == "classification_single"):#apply softmax/sigmoid before aggregation
                        preds = F.softmax(preds.float(),dim=-1)
                    elif(self.hparams.loss.supervised_type == "classification_multi"):
                        preds = torch.sigmoid(preds.float())
                    targs = torch.cat(outputs["targs"]).cpu()
                    targs_all = []
                    preds_all = []
                    stridetmp = int(self.hparams.base.stride_valtest/self.hparams.base.input_size*preds.shape[1]) if test is True else int(self.hparams.base.stride_val/self.hparams.base.input_size*preds.shape[1])

                    id_map = ds.get_id_mapping()
                    for x in np.unique(id_map):
                        idtmp = np.where(id_map==x)[0]
                        predstmp = torch.zeros((preds.shape[1]+(len(idtmp)-1)*stridetmp,preds.shape[2]),dtype=torch.float32)
                        predstmp_weight = torch.zeros(predstmp.shape[0],dtype=torch.int64)
                        targstmp = torch.zeros(predstmp.shape[0] if len(targs.shape)==2 else (predstmp.shape[0],targs.shape[-1]),dtype=torch.int64)
                        for i,(p,t) in enumerate(zip(preds[idtmp],targs[idtmp])):
                            start_idx = i*stridetmp
                            predstmp[start_idx:start_idx+preds.shape[1]]+=p
                            predstmp_weight[start_idx:start_idx+preds.shape[1]]+=1
                            targstmp[start_idx:start_idx+preds.shape[1]]=t
                        predstmp=predstmp/predstmp_weight.unsqueeze(-1)#take the weighted mean of all predictions
                        preds_all.append(predstmp)
                        targs_all.append(targstmp)
                    preds_all=torch.cat(preds_all,dim=0)
                    targs_all=torch.cat(targs_all,dim=0)

                else:#naive approach: just concatenate everything
                    preds_all = torch.cat(outputs["preds"]).cpu()
                    preds_all = preds_all.view(-1,preds_all.shape[-1])
                    if(self.hparams.loss.supervised_type == "classification_single"):
                        preds_all = F.softmax(preds_all,dim=-1)
                    elif(self.hparams.loss.supervised_type == "classification_multi"):
                        preds_all = torch.sigmoid(preds_all)
                    targs_all = torch.cat(outputs["targs"])
            else:#no multi-prediction
                preds_all = torch.cat(outputs["preds"]).cpu().float()
                if(self.hparams.loss.supervised_type == "classification_single"):
                    preds_all = F.softmax(preds_all,dim=-1)
                elif(self.hparams.loss.supervised_type == "classification_multi"):
                    preds_all = torch.sigmoid(preds_all)
                targs_all = torch.cat(outputs["targs"]).cpu()
            #export predictions
            if(test and self.hparams.trainer.export_predictions):
                np.savez(Path(self.hparams.trainer.output_path)/("preds_val.npz" if dataloader_idx==0 else "preds_test.npz"),preds=preds_all.cpu().numpy(),targs=targs_all.cpu().numpy(),ids=ds.get_id_mapping())

            #bring preds and targs into an appropriate shape
            if(self.hparams.loss.supervised_type == "classification_single"):
                if(np.any([len(d.label_filter)>0 for d in [self.hparams[d] for d in self.dataset_keys]])): #filter out labels we don't care about
                    preds_all=preds_all.view(-1,len(self.lbl_itos))[(targs_all.view(-1)>=0).to(preds_all.device)]
                    targs_all=targs_all.view(-1)[(targs_all.view(-1)>=0).to(preds_all.device)]
                targs_all = torch.eye(len(self.lbl_itos)).to(preds_all.device)[targs_all.view(-1).to(preds_all.device)]#flatten targets, tw: add .to(preds_all.device)
            elif(self.hparams.loss.supervised_type == "classification_multi"):
                if(len(targs_all.shape)==1):#single binary prediction
                    targs_all = targs_all.unsqueeze(dim=1)
                targs_all = targs_all.view(-1,targs_all.shape[-1])#flatten targets
                preds_all = preds_all.view(-1,targs_all.shape[-1])
            preds_all = preds_all.cpu().numpy()
            targs_all = targs_all.cpu().numpy()
            
        if(self.hparams.loss.loss_type=="supervised"): 
            self.test_idmaps = [self.test_dataset_val.get_id_mapping(), self.test_dataset_test.get_id_mapping()]

            if("mean" in [x.aggregation for x in self.metrics_train_val] and not self.trainer.sanity_checking):#checking metrics_train_val is sufficient as all of them are the same up to prefix/postfix
                preds_all_agg_mean,targs_all_agg_mean = ds.aggregate_predictions(preds_all,targs_all,aggregate_fn=np.mean)
            if("max" in [x.aggregation for x in self.metrics_train_val] and not self.trainer.sanity_checking):
                preds_all_agg_max,targs_all_agg_max = ds.aggregate_predictions(preds_all,targs_all,aggregate_fn=np.max)
            
            metrics = self.metrics_test_val if (test and dataloader_idx==0) else (self.metrics_test_test if (test and dataloader_idx==1) else self.metrics_train_val)
            for m in metrics if not self.trainer.sanity_checking else [m for m in metrics if m.aggregation==""]:
                if(m.aggregation==""):
                    res = m(targs_all,preds_all)
                elif(m.aggregation=="mean"):
                    res = m(targs_all_agg_mean,preds_all_agg_mean)
                elif(m.aggregation=="max"):
                    res = m(targs_all_agg_max,preds_all_agg_max)
                else:
                    assert(False)
                results_dict.update(res)

            if(not self.trainer.sanity_checking):
                self.log_dict(results_dict)
                if(test):
                    with open(Path(self.hparams.trainer.output_path)/("scores_val.pkl" if dataloader_idx==1 else "scores_test.pkl"), 'wb') as handle:
                        results_dict["epoch"]=self.current_epoch
                        pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    def modify_dataset_config(self,dataset_config,stage="train"):
        '''possibility modify dataset config parameters passed to dataset constructor in derived classes'''
        return dataset_config

    def preprocess_dataset(self,dataset_kwargs):
        '''override preprocessing in derived classes if desired'''
        return load_dataset(Path(dataset_kwargs.path))

    #override in derived classes to modify
    def get_custom_transforms(self,dataset_kwargs,lst_default_transforms, train=False):
        '''override transforms in derived classes if desired'''
        return lst_default_transforms
    
    def setup(self, stage):
        
        train_datasets = []
        val_datasets = []
        test_datasets_val = []
        test_datasets_test = []
        self.export_datasets = []

        ds_mean = None
        ds_std = None

        for d in [self.hparams[d] for d in self.dataset_keys]:
            
            df_mapped, lbl_itos, mean, std = self.preprocess_dataset(d)
            if(ds_mean is None):
                ds_mean, ds_std = mean, std

            #build up transforms
            assert(self.hparams.base.fs==d.fs)#resampling on the fly coming soon
            
            #aggregate sequence labels if desired
            if(self.hparams.loss.loss_type=="supervised" and self.hparams.base.label_aggregation_epoch_length>=0):
                tfms_train =[SequenceToSampleLabelTransform(num_classes=len(lbl_itos),majority_vote=self.hparams.base.label_aggregation_majority_vote,binary=self.hparams.base.label_aggregation_binary,epoch_length=self.hparams.base.label_aggregation_epoch_length)]
                tfms_valtest =[SequenceToSampleLabelTransform(num_classes=len(lbl_itos),majority_vote=self.hparams.base.label_aggregation_majority_vote,binary=True,epoch_length=self.hparams.base.label_aggregation_epoch_length)]#always use binary labels during test
            else:
                tfms_train = []
                tfms_valtest = []

            tfms=[]
            if(self.hparams.loss.loss_type=="supervised" and len(d.label_filter)>0):
                if(self.hparams.loss.supervised_type == "classification_single"):
                    #map labels to consecutive range (remaining labels to -1)
                    exttoint_dict = {e:i for i,e in enumerate(d.label_filter)}
                    exttoint = np.array([exttoint_dict[i] if i in exttoint_dict.keys() else -1 for i in range(len(lbl_itos))])
                    tfms+=[Transform(lambda x:exttoint[x])]
                else: #regression or multi-label classification
                    tfms+=[Transform(lambda x:x[:,d.label_filter] if len(x.shape)==2 else x[d.label_filter])]#multi-hot encoded: just select appropriate rows
                lbl_itos=lbl_itos[d.label_filter]
                
            if(self.hparams.base.normalize):
                tfms+=[Normalize(ds_mean,ds_std)]
            if(len(d.input_channels_filter)>0):#spectrograms have ts,ch,freq other ts,ch
                tfms+=[ChannelFilter(channels=d.input_channels_filter,axis=1 if self.hparams.base.freq_bins>0 else -1)]
            
            #obligatory ToTensor
            tfms_train+=tfms+[ToTensor()]
            tfms_valtest+=tfms+[ToTensor()]

            #apply custom transforms if desired
            tfms_train = self.get_custom_transforms(d,tfms_train,train=True)
            tfms_valtest = self.get_custom_transforms(d,tfms_valtest,train=False)

            assert(self.lbl_itos is None or np.all(self.lbl_itos == lbl_itos))#make sure all lbl_itos are identical
            self.lbl_itos = lbl_itos

            def get_folds_ids(fold_ids, all_ids, stage="train_supervised"):#stage: train_supervised, train_unsupervised, val_supervised, val_unsupervised, test_supervised
                if(len(fold_ids)==0):#use default assignments
                    max_fold_id=max(all_ids)
                    if(stage.startswith("train")):#train
                        res=[x for x in all_ids if (x<max_fold_id-1 if "train_supervised" else x<max_fold_id)]
                    elif(stage.startswith("val")):#val
                        res=[max_fold_id-1 if "val_supervised" else max_fold_id]
                    else:#test
                        res=[max_fold_id]
                else:
                    pos_ids = [x for x in fold_ids if x>=0] 
                    neg_ids = [-x for x in fold_ids if x<0]
                    assert(len(pos_ids)==0 or len(neg_ids)==0)#either only negative or only positive ids
                    if(len(neg_ids)>0):
                        res = [x for x in fold_ids if not x in neg_ids]
                    else:
                        res = fold_ids
                return res   

            #determine fold ids
            assert((len(d.train_fold_ids)>0 and len(d.val_fold_ids)>0) or d.col_train_fold==d.col_val_fold)
            train_ids = get_folds_ids(d.train_fold_ids,np.unique(df_mapped[d.col_train_fold]),"train_supervised" if self.hparams.loss.loss_type=="supervised" else "train_unsupervised")
            val_ids = get_folds_ids(d.val_fold_ids,np.unique(df_mapped[d.col_val_fold]),"val_supervised" if self.hparams.loss.loss_type=="supervised" else "val_unsupervised")
            df_train = df_mapped[df_mapped[d.col_train_fold].apply(lambda x: x in train_ids)]
            df_val = df_mapped[df_mapped[d.col_val_fold].apply(lambda x: x in val_ids)]
            
            if(self.hparams.loss.loss_type=="supervised"):
                test_ids = get_folds_ids(d.test_fold_ids,np.unique(df_mapped[d.col_test_fold]),"test")
                df_test = df_mapped[df_mapped[d.col_test_fold].apply(lambda x: x in test_ids)]
            
            #prepare default kwargs
            target_folder = Path(d.path)
            dataset_config_train = TimeSeriesDatasetConfig(
                df=df_train,
                output_size=self.hparams.base.input_size_dl,  
                data_folder=target_folder,
                chunk_length=self.hparams.base.chunk_length_train,
                min_chunk_length=self.hparams.base.input_size,
                stride=self.hparams.base.stride_train,
                transforms=tfms_train,
                sample_items_per_record=self.hparams.base.sample_items_per_record,# if self.hparams.loss.loss_type=="instance_contrastive_sequence" else 1,
                annotation=d.annotation,
                col_lbl=d.col_lbl if self.hparams.loss.loss_type=="supervised" else None,
                cols_static=d.cols_static if len(d.cols_static)>0 else None,
                cols_static_cat=d.cols_static_cat if len(d.cols_static_cat)>0 else None,
                memmap_filename=target_folder/("memmap.npy"),
                memmap_label_filename=None if d.path_label=="" else Path(d.path_label)/("memmap_label.npy"),
                fs_annotation_over_fs_data=d.fs_annotation/d.fs if d.fs_annotation!=-1 else 1.,
                return_idxs=self.hparams.base.return_idxs)# or self.hparams.loss.loss_type=="instance_contrastive_sequence")
            
            dataset_config_val = dataclasses.replace(dataset_config_train)
            dataset_config_val.df = df_val
            dataset_config_val.chunk_length= self.hparams.base.chunk_length_val
            dataset_config_val.stride= self.hparams.base.stride_val
            dataset_config_val.transforms= tfms_valtest
            
            if(self.hparams.loss.loss_type=="supervised"):
                dataset_config_valtest_val = dataclasses.replace(dataset_config_val)
                dataset_config_valtest_val.df = df_val
                dataset_config_valtest_val.chunk_length = self.hparams.base.chunk_length_valtest
                dataset_config_valtest_val.stride = self.hparams.base.stride_valtest
                dataset_config_valtest_val = self.modify_dataset_config(dataset_config_valtest_val,stage="test")

                dataset_config_valtest_test = dataclasses.replace(dataset_config_val)
                dataset_config_valtest_test.df = df_test
                dataset_config_valtest_test.chunk_length = self.hparams.base.chunk_length_valtest
                dataset_config_valtest_test.stride = self.hparams.base.stride_valtest
                dataset_config_valtest_test = self.modify_dataset_config(dataset_config_valtest_test,stage="test")


            if(self.hparams.loss.loss_type=="supervised"):
                test_datasets_val.append(TimeSeriesDataset(dataset_config_valtest_val))
                test_datasets_test.append(TimeSeriesDataset(dataset_config_valtest_test))
            if(self.hparams.trainer.export_features!=""):
                dataset_config_export = dataclasses.replace(dataset_config_val)
                dataset_config_export.df = df_mapped
                dataset_config_export.chunk_length = self.hparams.base.input_size
                dataset_config_export.stride = self.hparams.base.stride_export
                dataset_config_export = self.modify_dataset_config(dataset_config_export,stage="val")
                self.export_datasets.append(TimeSeriesDataset(dataset_config_export))
                print("export dataset:",len(self.export_datasets[-1]),"samples")
            
            #finally also create train and val datasets
            dataset_config_train = self.modify_dataset_config(dataset_config_train,stage="train") 
            dataset_config_val = self.modify_dataset_config(dataset_config_val,stage="val") 
            train_datasets.append(TimeSeriesDataset(dataset_config_train))
            val_datasets.append(TimeSeriesDataset(dataset_config_val))
            
            print("\n",d.path)
            print("train dataset:",len(train_datasets[-1]),"samples")
            print("val dataset:",len(val_datasets[-1]),"samples")
            if(self.hparams.loss.loss_type=="supervised"):
                print("test dataset(val):",len(test_datasets_val[-1]),"samples")
                print("test dataset(test):",len(test_datasets_test[-1]),"samples")

        if(len(train_datasets)>1): #multiple data folders
            print("\nCombined:")
            self.train_dataset = ConcatTimeSeriesDataset(train_datasets)
            self.val_dataset = ConcatTimeSeriesDataset(val_datasets)
            print("train dataset:",len(self.train_dataset),"samples")
            print("val dataset:",len(self.val_dataset),"samples")
            if(self.hparams.loss.loss_type=="supervised"):
                self.test_dataset_val = ConcatTimeSeriesDataset(test_datasets_val)
                self.test_dataset_test = ConcatTimeSeriesDataset(test_datasets_test)
                print("test dataset(val):",len(self.test_dataset_val),"samples")
                print("test dataset(test):",len(self.test_dataset_test),"samples")
        else: #just a single data folder
            self.train_dataset = train_datasets[0]
            self.val_dataset = val_datasets[0]
            if(self.hparams.loss.loss_type=="supervised"):
                self.test_dataset_val = test_datasets_val[0]
                self.test_dataset_test = test_datasets_test[0]
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.base.batch_size, num_workers=4, shuffle=True, drop_last = True)
        
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.base.batch_size, num_workers=self.hparams.trainer.num_workers)
    
    def test_dataloader(self):
        if(self.hparams.loss.loss_type=="supervised"):#multiple val dataloaders
            return [DataLoader(self.test_dataset_val, batch_size=self.hparams.base.batch_size, num_workers=self.hparams.trainer.num_workers),DataLoader(self.test_dataset_test, batch_size=self.hparams.base.batch_size, num_workers=self.hparams.trainer.num_workers)]
        else:
            return DataLoader(self.val_dataset, batch_size=self.hparams.base.batch_size, num_workers=self.hparams.trainer.num_workers)

    def get_params(self, modules=False):
        raise NotImplementedError
    
    def configure_optimizers(self):
        
        if(self.hparams.base.optimizer == "sgd"):
            opt = torch.optim.SGD
        elif(self.hparams.base.optimizer == "adam"):
            opt = torch.optim.AdamW
        else:
            raise NotImplementedError("Unknown Optimizer.")
        params = self.get_params()
        optimizer = opt(params, weight_decay=self.hparams.base.weight_decay)

        if(self.hparams.base.lr_schedule=="const"):
            scheduler = get_constant_schedule(optimizer)
        elif(self.hparams.base.lr_schedule=="const-plateau"):
            scheduler = ReduceLROnPlateau(optimizer)
        elif(self.hparams.base.lr_schedule=="warmup-const"):
            scheduler = get_constant_schedule_with_warmup(optimizer,self.hparams.base.lr_num_warmup_steps)
        elif(self.hparams.base.lr_schedule=="warmup-cos"):
            scheduler = get_cosine_schedule_with_warmup(optimizer,self.hparams.base.lr_num_warmup_steps,self.hparams.base.epochs*len(self.train_dataloader()),num_cycles=0.5)
        elif(self.hparams.base.lr_schedule=="warmup-cos-restart"):
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,self.hparams.base.lr_num_warmup_steps,self.hparams.base.epochs*len(self.train_dataloader()),num_cycles=self.hparams.base.epochs-1)
        elif(self.hparams.base.lr_schedule=="warmup-poly"):
            scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,self.hparams.base.lr_num_warmup_steps,self.hparams.base.epochs*len(self.train_dataloader()),num_cycles=self.hparams.base.epochs-1)   
        elif(self.hparams.base.lr_schedule=="warmup-invsqrt"):
            scheduler = get_invsqrt_decay_schedule_with_warmup(optimizer,self.hparams.base.lr_num_warmup_steps)
        elif(self.hparams.base.lr_schedule=="linear"): #linear decay to be combined with warmup-invsqrt c.f. https://arxiv.org/abs/2106.04560
            scheduler = get_linear_schedule_with_warmup(optimizer, 0, self.hparams.base.epochs*len(self.train_dataloader()))
        else:
            assert(False)

        return (
        [optimizer],
        [
            {
                'scheduler': scheduler,
                'interval': 'epoch' if self.hparams.base.lr_schedule == "const-plateau" else 'step',
                'frequency': 1,
                'monitor': 'val_loss/dataloader_idx_0' if len(self.val_dataloader())>1 else 'val_loss' #for plateau
            }
        ])

    def load_weights_from_checkpoint(self, checkpoint, filter=[]):
        """ Function that loads the weights from a given checkpoint file. 
        based on https://github.com/PyTorchLightning/pytorch-lightning/issues/525
        """
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage,)
        pretrained_dict = checkpoint["state_dict"]
        print("pretrained modules:",np.unique([".".join(k.split(".")[:2]) for k in  pretrained_dict.keys()]))
        
        if(len(filter)>0):
            def check_key(k):
                return any([k.startswith(f) for f in filter])
            pretrained_dict={k:v for k,v in pretrained_dict.items() if check_key(k)}
        model_dict = self.state_dict()

        print("model modules:",np.unique([".".join(k.split(".")[:2]) for k in model_dict.keys()]))

        pretrained_minus_model = [k for k in pretrained_dict.keys() if not k in model_dict.keys()]
        pretrained_minus_model.sort()
        model_minus_pretrained = [k for k in model_dict.keys() if not k in pretrained_dict.keys()]
        model_minus_pretrained.sort()

        if(len(pretrained_minus_model)>0):
            print("Warning: The following parameter were only present in the (filtered) state_dict (not in the model):",pretrained_minus_model)
        if(len(model_minus_pretrained)>0):
            print("Warning: The following parameter were only present in the model (not in the (filtered) state_dict):",model_minus_pretrained)
        
        #update only keys that are actually present in the model
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def load_state_dict(self, state_dict):
        '''S4-compatible load_state_dict'''
        for name, param in self.named_parameters():
            param.data = state_dict[name].data.to(param.device)
        for name, param in self.named_buffers():
            param.data = state_dict[name].data.to(param.device)

    def export_features(self, output_path, module="ts_encoder.predictor", as_memmap=False, aggregate_strides=True):
        
        getter = attrgetter(module)
        hook = ForwardHook(getter(self), store_output=True)

        for di,d in enumerate([self.hparams[d] for d in self.dataset_keys]):

            output_path_ds = Path(output_path)/d.name
            if not os.path.exists(output_path_ds):
                os.makedirs(output_path_ds)
    
            print("Exporting features for dataset",d.name,"to",output_path_ds,"...")

            input_size = self.hparams.base.input_size
            stride = self.hparams.base.stride_export

            #load df_mapped again in order to be able to store the features in standard format
            df_mapped, _, _, _ = self.preprocess_dataset(d)
            ds_export = self.export_datasets[di]
            dl_export = DataLoader(ds_export, batch_size=self.hparams.base.batch_size, num_workers=0)
            
            data_tmp = {}
            idx = 0
            id_map = np.array(ds_export.get_id_mapping())

            metadata = []

            self.eval()
            for i,data_batch in tqdm(enumerate(iter(dl_export)),total=len(dl_export),leave=False):
                input_data = data_batch[0].to(self.device)
                self.forward(seq=input_data)
                
                hidden_reps = hook.stored.detach().cpu().numpy()#bs,seq,feat
                ids = id_map[idx:idx+input_data.shape[0]]

                for x in np.unique(ids):
                    #prepare data
                    idtmp = np.where(ids==x)[0]
                    datatmp = hidden_reps[idtmp]#bs,seq,feat
                    
                    #store temporarily as bs,seq,feat (across multiple batches)
                    data_tmp[x]= np.concatenate((data_tmp[x],datatmp),axis=0) if x in data_tmp.keys() else datatmp
                    
                #write to file
                for x in list(data_tmp.keys()):
                    if(x != max(ids) or i==len(dl_export)-1):#sample is complete
                        filename_feat = str(df_mapped.iloc[x]["data_original"]).split("/")[-1]
                        if(aggregate_strides and stride!=input_size):
                            stride_pred = stride//input_size*data_tmp[x].shape[1]#stride in predictor units
                            datatmp=np.zeros((data_tmp[x].shape[1]+(data_tmp[x].shape[0]-1)*stride_pred,data_tmp[x].shape[1]),dtype=np.float32)
                            datatmp_weights = np.zeros(data_tmp[x].shape[1]+data_tmp[x].shape[0]*stride_pred,dtype=np.int64)
                            for j,y in enumerate(data_tmp[x]):
                                start_idx = j*stride_pred
                                datatmp[start_idx:start_idx+data_tmp[x].shape[1]]+=y
                                datatmp_weights[start_idx:start_idx+data_tmp[x].shape[1]]+=1
                            datatmp = datatmp/np.expand_dims(datatmp_weights,axis=-1)
                        else:
                            datatmp= np.concatenate([y for y in data_tmp[x]])
                        tmp_dict = {"id":x,"data_feat":filename_feat, "data_feat_length":len(datatmp)}
                        np.save(output_path_ds/filename_feat,datatmp)
                        del data_tmp[x]

                        metadata.append(tmp_dict)

                idx += input_data.shape[0]
            df_feat = pd.DataFrame(metadata).set_index("id")
            df_mapped["df_idx"]=range(len(df_mapped))
            df_mapped = df_mapped.join(df_feat)
            df_mapped.to_pickle(output_path_ds/("df_mapped.pkl"))
            
            if(as_memmap):
                print("Reformating as memmap...")
                reformat_as_memmap(df_mapped, output_path_ds/"memmap.npy", data_folder=output_path_ds, col_data="data_feat", delete_npys=True)
                                                                       
        

class SSLModel(BaseModel):

    def __init__(self, hparams):
        super(SSLModel, self).__init__(hparams)
        
        hparams_input_shape = ShapeConfig(hparams.base.input_channels,hparams.base.input_size,True,hparams.base.input_channels_cont,hparams.base.freq_bins,hparams.base.input_channels_cat)
        #input_size = hparams.base.input_size if isinstance(hparams.base.input_size,int) else int(np.round(hparams.base.input_size*hparams.base.fs))
        
        ##################################
        # create static encoder
        ##################################
        self.static_encoder = _string_to_class(hparams.static._target_)(hparams.static,hparams_input_shape,self.target_dim if self.loss_type.startswith("instance_contrastive") else None) if hparams.static._target_!="" else None
        if(self.static_encoder):
            hparams_output_shape_static_encoder = self.static_encoder.get_output_shape()

        ##################################
        # create ts encoder
        ##################################
        ts_encoders_keys = [d for d in hparams if ("_target_" in self.hparams[d].keys() and self.hparams[d]["_target_"]=="clinical_ts.template_modules.TimeSeriesEncoder")]

        for t in ts_encoders_keys:#set unique names (to differentiate losses)
            self.hparams[t].name = t+"_"+self.hparams[t].name
            if(self.hparams[t].enc["_target_"]=="clinical_ts.template_modules.TimeSeriesEncoder"):#also adjust potentially nested tsencoder
                self.hparams[t].enc.name = t+"_"+self.hparams[t].enc.name

        target_dim_ts_encoder = self.target_dim if (self.loss_type.startswith("instance_contrastive") or hparams.head._target_=="") else None
        if(len(ts_encoders_keys)==1):
            tsek = ts_encoders_keys[0]
            self.ts_encoder = _string_to_class(self.hparams[tsek]._target_)(self.hparams[tsek], hparams_input_shape, target_dim_ts_encoder)
        else:
            self.ts_encoder = SequentialTimeSeriesEncoder([self.hparams[tsek] for tsek in ts_encoders_keys], hparams_input_shape, target_dim_ts_encoder)
        hparams_output_shape_ts_encoder = self.ts_encoder.get_output_shape()

        #################################
        # create head
        #################################
        hparams_output_shape_ts_encoder.static_dim = hparams_output_shape_static_encoder.static_dim if self.static_encoder else 0
        hparams_output_shape_ts_encoder.static_dim_cat = 0
        
        self.head = _string_to_class(hparams.head._target_)(hparams.head, hparams_output_shape_ts_encoder, self.target_dim) if hparams.head._target_!="" else None #static_encoder_output_dim if self.loss_type!="clip" else 0

        self.multi_prediction = (self.hparams.head.multi_prediction or (self.hparams.head._target_=="" and self.hparams[ts_encoders_keys[-1]].head.multi_prediction))

        ################################
        # basic output
        ################################
        print("MODEL ARCHITECTURE:")
        print("Input shape:",hparams_input_shape)
        if(self.ts_encoder is not None):
            print("\n\nTS ENCODER:")
            print(self.ts_encoder)
        if(self.static_encoder is not None):
            print("\n\nSTATIC ENCODER:")
            print(self.static_encoder)
        if(self.head is not None):
            print("\n\nHEAD:")
            print(self.head)
        print("\n\n\n")

    def is_multi_prediction(self):
        return self.multi_prediction
    
    def forward(self, **kwargs):
        kwargs["seq"][torch.isnan(kwargs["seq"])]=0 # QUICK FIX FOR NANS IN INPUT
        if("static" in kwargs.keys()):
            if(kwargs["static"].dtype==torch.float64):
                kwargs["static"]=kwargs["static"].float() # QUICK FIX FOR FP32 STATIC INPUT
                
        if(self.static_encoder is not None):
            stat_enc_res = self.static_encoder(**kwargs) if (self.static_encoder is not None) else None

        seq_enc_res = self.ts_encoder(**kwargs)
        res = kwargs.copy()
        res.update(seq_enc_res)
        if(self.static_encoder is not None):
            res.update(stat_enc_res)
        
        if(self.head is not None):
            head_res = self.head(**res)
            res.update(head_res)
        
        res["input_predicted"]=res["seq"]
        if(self.static_encoder is not None):
            res["static_encoded"]=res["static"]
        
        return res
    
    def get_params(self, modules=False):
        encoder_modules = []
        predictor_modules = []
        head_modules = []

        if(self.static_encoder is not None):
            encoder_modules.append(self.static_encoder)
        if(self.ts_encoder is not None):
            enc_mod, pred_mod, head_mod = self.ts_encoder.get_modules()
            encoder_modules+=enc_mod
            predictor_modules+=pred_mod
            head_modules+=head_mod        
        if(self.head is not None):
            head_modules.append(self.head)
        if(isinstance(self.criterion,nn.Module)):#some loss functions also carry parameters
            head_modules.append(self.criterion)

        encoder_params = chain(*[e.parameters() for e in encoder_modules])
        predictor_params = chain(*[p.parameters() for p in predictor_modules])
        head_params = chain(*[h.parameters() for h in head_modules])

        if(self.loss_type=="supervised" and (self.hparams.base.linear_eval or self.hparams.base.train_head_only)):
            params = [{"params":head_modules if modules else self.ts_encoder.head.parameters(), "lr":self.lr}]
        else:#self.hparams.base.discriminative_lr_factor might also be one
            if(isinstance(self.lr,list) and len(self.lr)==3):
                params = [{"params":head_modules if modules else head_params, "lr":self.lr[0]},{"params":predictor_modules if modules else predictor_params, "lr":self.lr[1]},{"params":encoder_modules if modules else encoder_params, "lr":self.lr[2]}]
            else:
                params = [{"params":head_modules if modules else head_params, "lr":self.lr},{"params":predictor_modules if modules else predictor_params, "lr":self.lr*self.hparams.base.discriminative_lr_factor},{"params":encoder_modules if modules else encoder_params, "lr":self.lr*self.hparams.base.discriminative_lr_factor*self.hparams.base.discriminative_lr_factor}]
        return params
