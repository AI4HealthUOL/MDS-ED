from hydra.core.config_store import ConfigStore
from dataclasses import dataclass

#for base configs
from .template_modules import * 

#for specific configs
from .ts.encoder import *
from .ts.head import *
from .ts.base import *

##################################################################
# try to import (optional) modules with special dependencies
S4_AVAILABLE = True
try:
    from .ts.s4 import *
except:
    print("WARNING: Could not import s4 module")
    S4_AVAILABLE = False

from .tabular.base import *

from .head.multimodal import *

from .loss.supervised import *

from .metric.base import *

from .task.multimodal import *

###########################################################################################################
# https://hydra.cc/docs/tutorials/structured_config/config_groups/
@dataclass
class FullConfig:

    base: BaseConfig
    data: BaseConfigData
    loss: LossConfig
    metric: MetricConfig
    trainer: TrainerConfig
    task: TaskConfig

    ts: TimeSeriesEncoderConfig
    static: EncoderStaticBaseConfig
    head: HeadBaseConfig
    

def create_default_config():
    cs = ConfigStore.instance()
    cs.store(name="config", node=FullConfig)

    ######################################################################
    # base
    ######################################################################
    cs.store(group="base", name="base", node=BaseConfig)

    ######################################################################
    # input data
    ######################################################################
    cs.store(group="data", name="base", node=BaseConfigData)
    
    ######################################################################
    # time series encoder
    ######################################################################
    cs.store(group="ts", name="tsenc",  node=TimeSeriesEncoderConfig)
    
    #ENCODER
    cs.store(group="ts/enc", name="none", node=NoEncoderConfig)
    
    #PREDICTOR
    cs.store(group="ts/pred", name="none", node=NoPredictorConfig)#no predictor
    if(S4_AVAILABLE):
        cs.store(group="ts/pred", name="s4", node=S4PredictorConfig)#S4 model
    
    #HEADS
    cs.store(group="ts/head", name="none", node=HeadBaseConfig)
    cs.store(group="ts/head", name="pool", node=PoolingHeadConfig)

    #SSL HEADS
    cs.store(group="ts/head_ssl", name="none", node=HeadBaseConfig)

    #QUANTIZER
    cs.store(group="ts/quant", name="none", node=QuantizerBaseConfig)

    #MASK
    cs.store(group="ts/mask", name="none", node=MaskingBaseConfig)

    #LOSS
    cs.store(group="ts/loss", name="none", node=SSLLossConfig)

    #PRE
    cs.store(group="ts/pre", name="none", node=PrePostBaseConfig)  
    
    #POST
    cs.store(group="ts/pre", name="none", node=PrePostBaseConfig)
    
    ######################################################################
    # static encoder
    ######################################################################
    for g in ["static", "ts/static"]:
        cs.store(group=g, name="none", node=EncoderStaticBaseConfig)
        cs.store(group=g, name="mlp", node=BasicEncoderStaticMLPConfig)
        
    ######################################################################
    # optional multimodal head
    ######################################################################
    cs.store(group="head", name="none", node=HeadBaseConfig)
    cs.store(group="head", name="concat", node=ConcatFusionHeadConfig)

    ######################################################################
    # loss function
    ######################################################################
    #no global loss
    cs.store(group="loss", name="none", node=LossConfig)
    #supervised losses
    cs.store(group="loss", name="bce", node=BCELossConfig)
    cs.store(group="loss", name="bcef", node=BCEFLossConfig) 
    
    ######################################################################
    # metrics
    ######################################################################
    cs.store(group="metric", name="none", node=MetricConfig)
    cs.store(group="metric", name="auroc", node=MetricAUROCConfig)
    cs.store(group="metric", name="aurocagg", node=MetricAUROCAggConfig)
    
    ######################################################################
    # trainer
    ######################################################################
    cs.store(group="trainer", name="trainer", node=TrainerConfig)
    
    ######################################################################
    # task
    ######################################################################
    cs.store(group="task", name="none", node=TaskConfig)
    cs.store(group="task", name="multi", node=TaskConfigMultimodal)
    
    return cs
