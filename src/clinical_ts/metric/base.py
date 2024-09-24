__all__ = ['MetricConfig', 'MetricBase', 'MetricAUROC', 'MetricAUROCConfig', 'MetricAUROCAggConfig']

import numpy as np

from dataclasses import dataclass

from ..utils.eval_utils_cafa import multiclass_roc_curve
from ..utils.bootstrap_utils import empirical_bootstrap

import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Filter out the warnings due to not enough positive/negative samples during bootstrapping
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

@dataclass
class MetricConfig:
    _target_:str = ""
    
    name:str = ""#name of the metric e.g. auroc

    aggregation:str = "" #"" means no aggregation across segments of the same sequence, other options: "mean", "max"
    
    key_summary_metric:str = "" #key into the output dict that can serve as summary metric for early stopping etc e.g. (without key_prefix and key_postfix and aggregation type)
    mode_summary_metric:str ="max" #used to determine if key_summary_metric is supposed to be maximized or minimized
    
    verbose:str = "" # comma-separated list of keys to be printed after metric evaluation (without key_prefix and key_postfix and aggregation type)
    
    bootstrap_report_nans:bool = False #report nans during bootstrapping (due to not enough labels of a certain type in certain bootstrap iterations etc)
    bootstrap_iterations:int = 0 #0: no bootstrap
    bootstrap_alpha:float= 0.95 # bootstrap alpha

def _reformat_lbl_itos(k):
    #    return re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower() #openclip
    return k.replace(" ","_").replace("|","_").replace("(","").replace(")","")
    
class MetricBase:
    def __init__(self, hparams_metric, lbl_itos, key_prefix, key_postfix, test=True):
        self.key_prefix = (key_prefix+"_" if len(key_prefix)>0 else "")+hparams_metric.name+"_"
        self.key_postfix = key_postfix
        self.aggregation = hparams_metric.aggregation
        self.aggregation_txt = ("_agg" if hparams_metric.aggregation=="mean" else "_agg"+hparams_metric.aggregation) if hparams_metric.aggregation!="" else ""
        self.key_summary_metric = self.key_prefix+hparams_metric.key_summary_metric+self.aggregation_txt+self.key_postfix #data loader id added by default
        self.mode_summary_metric = hparams_metric.mode_summary_metric
        self.verbose = [x for x in hparams_metric.verbose.split(",") if x!=""]

        self.bootstrap_iterations = hparams_metric.bootstrap_iterations if test else 0 #disable bootstrap during training
        self.bootstrap_alpha = hparams_metric.bootstrap_alpha
        self.bootstrap_report_nans = hparams_metric.bootstrap_report_nans

        self.lbl_itos = [_reformat_lbl_itos(l) for l in lbl_itos]
        self.keys = self.get_keys(self.lbl_itos)


    def get_keys(self, lbl_itos):
        '''returns metrics keys in the order they will later be returned by _eval'''
        raise NotImplementedError
    
    def __call__(self,targs,preds):
        
        if(self.bootstrap_iterations==0):
            point = self._eval(targs,preds)
        else:
            point,low,high,nans = empirical_bootstrap((targs,preds), self._eval, n_iterations=self.bootstrap_iterations , alpha=self.bootstrap_alpha,ignore_nans=True)#score_fn_kwargs={"classes":self.lbl_itos}
        res = {self.key_prefix+k+self.aggregation_txt+self.key_postfix:v for v,k in zip(point,self.keys)}
        if(self.bootstrap_iterations>0):
            res_low = {self.key_prefix+k+self.aggregation_txt+self.key_postfix+"_low":v for v,k in zip(low,self.keys)}
            res_high = {self.key_prefix+k+self.aggregation_txt+self.key_postfix+"_high":v for v,k in zip(high,self.keys)}
            res_nans = {self.key_prefix+k+self.aggregation_txt+self.key_postfix+"_nans":v for v,k in zip(nans,self.keys)}
            res.update(res_low)
            res.update(res_high)
            if(self.bootstrap_report_nans):
                res.update(res_nans)

        if(len(self.verbose)>0):
            for k in self.verbose:
                print("\n"+self.key_prefix+k+self.aggregation_txt+self.key_postfix+":"+str(res[self.key_prefix+k+self.aggregation_txt+self.key_postfix]))
        
        return res

    def _eval(self,targs,preds):
        # should return an array of results ordered according to the entries returned by get_keys()
        raise NotImplementedError


    
class MetricAUROC(MetricBase):
    '''provides class-wise+macro+micro AUROC/AUPR scores'''
    def __init__(self, hparams_metric, lbl_itos, key_prefix="", key_postfix="0", test=True):
        super().__init__(hparams_metric, lbl_itos=lbl_itos, key_prefix=key_prefix, key_postfix=key_postfix, test=test)
        self.precision_recall = hparams_metric.precision_recall

    def get_keys(self, lbl_itos):
        return list(lbl_itos)+["micro","macro"]
    
    def _eval(self,targs,preds):
        if(self.precision_recall):
            _,_,res = multiclass_roc_curve(targs,preds,classes=self.lbl_itos,precision_recall=True)
            return np.array(list(res.values()))
        else:
            _,_,res = multiclass_roc_curve(targs,preds,classes=self.lbl_itos)
            return np.array(list(res.values()))
        

@dataclass
class MetricAUROCConfig(MetricConfig):
    _target_:str = "clinical_ts.metric.base.MetricAUROC"
    key_summary_metric:str = "macro"
    verbose:str="macro" #by default print out macro auc
    precision_recall:bool = False #calculate the area under the precision recall curve instead of the ROC curve
    name:str = "auroc"
    bootstrap_report_nans:bool = True #by default report number of bootstrap iterations where the score was nan (due to insufficient number of labels etc)

#shorthand for mean aggregation
@dataclass
class MetricAUROCAggConfig(MetricAUROCConfig):
    aggregation:str="mean"
