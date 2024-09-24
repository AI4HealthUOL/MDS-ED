__all__ = ['TransformerPatchTokenizer','TransformerSingleConvTokenizer','TransformerConvStemTokenizer','AttentionPool1d', 'LearnableQueryAttentionPool1d','TransformerPooling', 'TransformerHead', 'TransformerModule']

import torch
from torch.nn import Module, Linear, Dropout, LayerNorm, Identity
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

################################################
#POSITIONAL ENCODINGS
################################################
#https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs/blob/main/lab4/text_recognizer/models/line_cnn_transformer.py

class PositionalEncoding(torch.nn.Module):
    """Classic Attention-is-all-you-need positional encoding."""

    def __init__(self, dim_model: int, dropout: float = 0.1, max_len: int = 1000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout) if dropout>0 else Identity()

        pe = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, ind=None) -> torch.Tensor:
        '''allows to optionally pass an index tensor of shape bs,seq otherwise just range(len(seq)) will be assumed'''
        if(ind is None):
            x = x + self.pe[: x.size(0), :]
        else:
            #ind has shape bs,seq and x has shape seq,bs,e
            x = x + self.pe[ind.view(-1)].view(ind.size(0),ind.size(1),-1).transpose(0,1)
        return self.dropout(x)

class LearnedPositionalEncoding(torch.nn.Module):
    """Learned positional encoding."""

    def __init__(self, dim_model: int, dropout: float = 0.1, max_len: int = 1000) -> None:
        super(LearnedPositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout) if dropout>0 else Identity()
        self.emb = nn.Embedding(max_len, dim_model)
        nn.init.trunc_normal_(self.emb, std=0.2)

    def forward(self, x: torch.Tensor, ind=None) -> torch.Tensor:
        '''allows to optionally pass an index tensor of shape bs,seq otherwise just range(len(seq)) will be assumed'''
        if(ind is None):
            x = x + self.emb(torch.arange(0,x.size(0))).unsqueeze(1).to(x.device)
        else:
            #ind has shape bs,seq and x has shape seq,bs,e
            x = x + self.emb(ind).transpose(0,1)
        return self.dropout(x)

################################################
#TOKENIZER
################################################

def TransformerPatchTokenizer(dim_model, patch_size=16, input_channels=3, dim_data=2, return_batch_first=True):
    '''standard ViT Tokenizer'''
    return TransformerTokenizer(strides=[patch_size],kss=[patch_size],features=[dim_model],paddings=[0],max_pool=False,activation=None,input_channels=input_channels,dim_data=dim_data,return_batch_first=return_batch_first)

def TransformerSingleConvTokenizer(dim_model, input_channels=3, kernel_size=7, stride=2, padding=3, pooling_kernel_size=3, pooling_stride=2, pooling_padding=1, max_pool=True, activation="relu",normalization=False,dim_data=2,return_batch_first=True):
    '''CompactTransformer Tokenizer'''
    return TransformerTokenizer(strides=[stride],kss=[kernel_size],features=[dim_model],paddings=[padding],pooling_ks=pooling_kernel_size,pooling_stride=pooling_stride,pooling_padding=pooling_padding,max_pool=max_pool,activation=activation,input_channels=input_channels,normalization=normalization,dim_data=dim_data,return_batch_first=return_batch_first)

#Vision Transformer with conv stem https://arxiv.org/abs/2106.14881; dx.doi.org/10.22489%2FCinC.2020.107 for ecg
def TransformerConvStemTokenizer(features=[64,128,256,512,768], input_channels=3, kernel_sizes=[3,3,3,3,1], strides=[2,2,2,2,1], paddings=[1,1,1,1,0], pooling_kernel_size=3, pooling_stride=2, pooling_padding=1, max_pool=False, activation="relu",normalization=True,dim_data=2,return_batch_first=True):
    '''CompactTransformer Tokenizer'''
    return TransformerTokenizer(strides=strides,kss=kernel_sizes,features=features,paddings=paddings,pooling_ks=pooling_kernel_size,pooling_stride=pooling_stride,pooling_padding=pooling_padding,max_pool=max_pool,activation=activation,input_channels=input_channels,normalization=normalization,dim_data=dim_data,return_batch_first=return_batch_first)

class TransformerTokenizer(nn.Module):
    '''adapted from CompactTransformer'''
    def __init__(self,
                 strides, kss, features, paddings=None,
                 pooling_ks=3, pooling_stride=2, pooling_padding=1, max_pool=False,
                 input_channels=3,
                 activation="gelu",normalization=False,dim_data=2,return_batch_first=True):
        super(TransformerTokenizer, self).__init__()

        self.dim_data = dim_data
        self.input_channels = input_channels
        self.return_batch_first = return_batch_first

        in_filters = [input_channels]+list(features[:-1])
        out_filters = list(features)
        paddings = [(ks-1)//2 for ks in kss] if paddings is None else paddings

        if(activation=="gelu"):
            act_fn = nn.GELU
        elif(activation=="relu"):
            act_fn = nn.ReLU
        else:
            act_fn = None
        conv_fn = nn.Conv2d if dim_data==2 else nn.Conv1d
        pooling_fn =nn.MaxPool2d if dim_data==2 else nn.MaxPool1d
        normalization_fn = (nn.BatchNorm2d if dim_data==2 else nn.BatchNorm1d) if normalization is True else None
        layers = []

        for inf,outf,ks,stride,padding in zip(in_filters,out_filters,kss,strides,paddings):
            layers.append(conv_fn(inf,outf,kernel_size=ks,stride=stride,padding=padding,bias=not(normalization)))
            if(normalization_fn is not None):
                layers.append(normalization_fn(outf))
            if(act_fn is not None):
                layers.append(act_fn())
        if(max_pool):
            layers.append(pooling_fn(kernel_size=pooling_ks, stride=pooling_stride, padding=pooling_padding))
        if(dim_data==2):
            layers.append(nn.Flatten(2, 3)) #B,E,S
        self.layers = nn.Sequential(*layers)
        
        self.kwargs = {"dim_data":dim_data, "kss":kss, "strides": strides, "paddings": paddings, "max_pool": max_pool, "pooling_ks": pooling_ks, "pooling_stride": pooling_stride, "pooling_padding": pooling_padding, "normalization": normalization} 

        self.apply(self.init_weight)

    def get_output_length(self,length):
        for ks, stride, padding in zip(self.kwargs["kss"],self.kwargs["strides"],self.kwargs["paddings"]):
            length = np.floor((length+2*padding-(ks-1)-1)/stride+1)
        if(self.kwargs["max_pool"]):
            length = np.floor((length+2*self.kwargs["pooling_padding"]-(self.kwargs["pooling_ks"]-1)-1)/self.kwargs["pooling_stride"]+1)
        length = int(length)
        return length*length if self.kwargs["dim_data"]==2 else length

    def forward(self, x):
        x = self.layers(x).transpose(1,2) #B,S,E
        return x if self.return_batch_first else x.transpose(0,1)#S,B,E in the latter case

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d) or isinstance(m,nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)

################################################
#CLASSIFICATION HEAD
################################################
# adapted from https://github.com/openai/CLIP/blob/main/clip/model.py#L56
class AttentionPool1d(nn.Module):
    def __init__(self, length: int, embed_dim: int, num_heads: int, output_dim: int = None, batch_first: bool=True, bias=False, pos_embedding=True):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(length+1, embed_dim) / embed_dim ** 0.5) if pos_embedding else None
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim, bias=bias)
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.bias = bias


    def forward(self, x):
        if(self.batch_first): # (B, Sx, E) otherwise (Sx, B, E) 
            x = x.permute(1, 0, 2)  # (Sx, B, E)
        #add cls token
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # TNC
        if(self.positional_embedding is not None):
            x = x + self.positional_embedding[:, None, :].to(x.dtype)  # TNC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]) if self.bias else None,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias if self.bias else None,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]

class LearnableQueryAttentionPool1d(nn.Module):
    '''V-JEPA (https://openreview.net/forum?id=WFYbBOEOtv) Learnable Query Attention Pooling '''
    def __init__(self, embed_dim: int, num_heads: int, output_dim: int = None, batch_first: bool = True, bias: bool = False):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim, bias=bias)
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.bias = bias

    def forward(self, x):
        if(self.batch_first): # (B, Sx, E) otherwise (Sx, B, E) 
            x = x.permute(1, 0, 2)  # (Sx, B, E)
        
        x, _ = F.multi_head_attention_forward(
            query=self.query.repeat(1,x.shape[1],1), key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]) if self.bias else None,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias if self.bias else None,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]

class TransformerPooling(nn.Module):
    def __init__(self,dim_model,pooling_type="cls",cls_token_first=False,batch_first=True,n_heads_seq_pool=1,layer_norm=True):
        super().__init__()
        self.pooling_type = pooling_type
        self.batch_first = batch_first
        self.cls_token_first = cls_token_first
        self.norm = nn.LayerNorm(dim_model) if layer_norm else nn.Identity()
        
        if(pooling_type=="cls"):
            n_dim = 1
        elif(pooling_type=="mean"):
            n_dim = 1 
        elif(pooling_type=="max"):
            n_dim = 1    
        elif(pooling_type=="meanmax"):
            n_dim = 2
        elif(pooling_type=="meanmax-cls"):
            n_dim = 3
        elif(pooling_type.startswith("seq")):
            if(pooling_type=="seq"):# generalization of seq_pool from Compact-Transformers (multi-head attention)
                n_dim = n_heads_seq_pool
            elif(pooling_type=="seq-meanmax"):
                n_dim = 2+n_heads_seq_pool
            elif(pooling_type=="seq-meanmax-cls"):
                n_dim = 3+n_heads_seq_pool
            else:
                assert(False)
            self.attention_pool = nn.Linear(dim_model, n_heads_seq_pool)
        else:
            assert(False)
        self.apply(_init_weight)
        self.output_dim = n_dim*dim_model

    def forward(self,x):
        if(self.batch_first): # (B, Sx, E) otherwise (Sx, B, E) 
            x = x.permute(1, 0, 2)  # (Sx, B, E)

        x = self.norm(x)

        if(self.pooling_type=="cls"):
            x = x[0] if self.cls_token_first else x[-1]
        elif(self.pooling_type=="mean"):
            x = torch.mean(x,dim=0)
        elif(self.pooling_type=="max"):
            x = torch.max(x,dim=0)[0]
        elif(self.pooling_type=="meanmax"):
            x = torch.cat([torch.mean(x,dim=0),torch.max(x,dim=0)[0]],dim=1)
        elif(self.pooling_type=="meanmax-cls"):
            x = torch.cat((x[0] if self.cls_token_first else x[-1],torch.mean(x,dim=0),torch.max(x,dim=0)[0]),dim=1)
        elif(self.pooling_type=="seq"):
            x=torch.einsum('sbn,sbd->bnd',F.softmax(self.attention_pool(x),dim=0),x).view(x.size(1),-1)
        elif(self.pooling_type=="seq-meanmax"):
            x=torch.cat((torch.mean(x,dim=0),torch.max(x,dim=0)[0],torch.einsum('sbn,sbd->bnd',F.softmax(self.attention_pool(x),dim=0),x).view(x.size(1),-1)),dim=1)
        elif(self.pooling_type=="seq-meanmax-cls"):
            x=torch.cat((x[0] if self.cls_token_first else x[-1],torch.mean(x,dim=0),torch.max(x,dim=0)[0],torch.einsum('sbn,sbd->bnd',F.softmax(self.attention_pool(x),dim=0),x).view(x.size(1),-1)),dim=1)
        
        return x


class TransformerHead(nn.Module):
    def __init__(self,dim_model,num_classes,pooling_type="cls",cls_token_first=False,batch_first=True,n_heads_seq_pool=1):
        super().__init__()
        self.pool = TransformerPooling(dim_model=dim_model,pooling_type=pooling_type,cls_token_first=cls_token_first,batch_first=batch_first,n_heads_seq_pool=n_heads_seq_pool,layer_norm=True)
        
        self.fc = nn.Linear(self.pool.output_dim,num_classes)
        self.apply(_init_weight)

    def forward(self,x):
        x = self.pool(x)
        return self.fc(x)
        
################################################
#WEIGHT INIT
################################################

def _init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


################################################
#TRANSFORMER MODULE (AS RNN REPLACEMENT)
################################################

class TransformerModule(nn.Module):
    """ Transformer module that can be used as a replacement for a RNN module"""

    def __init__(self, dim_model=768, mlp_ratio = 4.0, dropout=0.1, dropout_attention=0.1, stochastic_depth_rate=0.1, num_layers=12, num_heads=12, masked=False, max_length=1024, batch_first=True, input_size=None, output_size=None, activation='gelu', norm_first=True, pos_enc="sine", cls_token=False, native=True):
        super().__init__()
        self.dim_model = dim_model
        dim_feedforward = int(dim_model * mlp_ratio)

        if(cls_token):
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim_model), requires_grad=True)
        else:
            self.cls_token = None

        if(pos_enc == "sine"):
            self.pos_encoder = PositionalEncoding(dim_model=dim_model,dropout=dropout,max_len=max_length+1 if cls_token else max_length)
        elif(pos_enc == "learned"):
            self.pos_encoder = LearnedPositionalEncoding(dim_model=dim_model,dropout=dropout,max_len=max_length+1 if cls_token else max_length)
        else:
            self.pos_encoder = None
            
        self.masked = masked
        if(masked):
            self.mask = self.generate_square_subsequent_mask(max_length+1 if cls_token else max_length)

        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_rate, num_layers)]

        self.transformer_encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, norm_first=norm_first) if native else TransformerEncoderLayer(d_model=dim_model, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout,
                 attention_dropout=dropout_attention, drop_path_rate=dpr[i], activation=activation, norm_first=norm_first)
            
            for i in range(num_layers)])

        self.num_layers = num_layers
        self.batch_first = batch_first

        self.fc_in = nn.Linear(input_size,dim_model) if input_size is not None and input_size!=dim_model else None
        self.fc_out = nn.Linear(dim_model,output_size) if output_size is not None and output_size!=dim_model else None
        
        self.apply(_init_weight)

    @staticmethod
    def generate_square_subsequent_mask(size: int) -> torch.Tensor:
        """Generate a triangular (size, size) mask."""
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x: torch.Tensor, src_key_padding_mask=None) -> torch.Tensor:
        if(self.batch_first): # (B, Sx, E) otherwise (Sx, B, E) 
            x = x.permute(1, 0, 2)  # (Sx, B, E)

        if(self.fc_in is not None):
            x = self.fc_in(x)
        
        #add dummy cls token at the end (for consistency with lstm)
        if(self.cls_token is not None):
            x=torch.cat([x,self.cls_token.expand(-1,x.shape[1], -1)],dim=0)
        
        if(self.pos_encoder is not None):
            x = self.pos_encoder(x)  # (Sx, B, E)

        Sx = x.shape[0]

        for blk in self.transformer_encoder:
            x = blk(x, self.mask[:Sx, :Sx].type_as(x) if self.masked else None, src_key_padding_mask=src_key_padding_mask) # (Sx, B, E)
        
        output = self.fc_out(x) if self.fc_out is not None else x # (Sx, B, C)

        return output.permute(1, 0, 2) if self.batch_first else output


################################################
#MULTIHEAD ATTENTION
################################################
# based on rwightman's timm package and CompactTransformers
# https://github.com/rwightman/pytorch-image-models and https://github.com/SHI-Labs/Compact-Transformer
class Attention(Module):
    """
    Obtained from: https://github.com/rwightman/pytorch-image-models
    with added support for masks, RNN channel ordering
    """
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        assert(dim%num_heads==0)
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = Linear(dim, dim * 3, bias=False)
        self.attn_drop = Dropout(attention_dropout) if attention_dropout>0 else Identity()
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(projection_dropout) if projection_dropout>0 else Identity()

    def forward(self, x, attn_mask=None):
        N, B, C = x.shape
        qkv = self.qkv(x).reshape(N, B, 3, self.num_heads, C // self.num_heads).permute(2, 1, 3, 0, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] #shapes B H N C//H

        attn = (q @ k.transpose(-2, -1)) * self.scale #B H N N
        if(attn_mask is not None):
            attn = attn + attn_mask

        attn = attn.softmax(dim=-1) #B H N N * B H N C//H = B H N C//H
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(0, 2).reshape(N, B, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

################################################
#SINGLE TRANSFORMER LAYER
################################################

class TransformerEncoderLayer(Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and
    rwightman's timm package.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1, activation= "gelu", norm_first=True):
        super(TransformerEncoderLayer, self).__init__()
        self.pre_norm = LayerNorm(d_model) if norm_first else Identity()#was always active originally
        self.self_attn = Attention(dim=d_model, num_heads=nhead,
                                   attention_dropout=attention_dropout, projection_dropout=dropout)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout1 = Dropout(dropout) if dropout>0 else Identity()
        self.post_norm = LayerNorm(d_model) if not(norm_first) else Identity()#was always active originally
        self.linear2 = Linear(dim_feedforward, d_model)
        self.dropout2 = Dropout(dropout) if dropout>0 else Identity()

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else Identity()
        if(activation=="gelu"):
            self.activation = F.gelu
        elif(activation=="relu"):
            self.activation = F.relu
        else:
            assert(False)

    def forward(self, src: torch.Tensor, attn_mask=None, *args, **kwargs) -> torch.Tensor:
        src = src + self.drop_path(self.self_attn(self.pre_norm(src), attn_mask))
        src = self.post_norm(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src

################################################
#STOCHASTIC DEPTH
################################################

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(Module):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
