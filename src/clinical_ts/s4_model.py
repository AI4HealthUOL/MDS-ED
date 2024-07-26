__all__ = ['S4Model', 'S4ModelMM']
#adapted from https://github.com/HazyResearch/state-spaces/blob/main/example.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from clinical_ts.s42 import S4 as S42


class S4Model(nn.Module):

    def __init__(
        self, 
        d_input, # None to disable encoder
        d_output, # None to disable decoder
        d_state=64, #MODIFIED: N
        d_model=512, #MODIFIED: H
        n_layers=4, 
        dropout=0.2,
        prenorm=False,
        l_max=1024,
        transposed_input=True, # behaves like 1d CNN if True else like a RNN with batch_first=True
        bidirectional=True, #MODIFIED
        layer_norm = True, # MODIFIED
        pooling = True, # MODIFIED
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.transposed_input = transposed_input
        
        # MODIFIED TO ALLOW FOR MODELS WITHOUT ENCODER
        if(d_input is None):
            self.encoder = nn.Identity()
        else:
            self.encoder = nn.Conv1d(d_input, d_model, 1) if transposed_input else nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S42(
                    d_state=d_state,
                    l_max=l_max,
                    d_model=d_model, 
                    bidirectional=bidirectional,
                    postact='glu',
                    dropout=dropout, 
                    transposed=True,
                ))
            #MODIFIED TO ALLOW BATCH NORM MODELS
            self.layer_norm = layer_norm
            if(layer_norm):
                self.norms.append(nn.LayerNorm(d_model))
            else: #MODIFIED
                self.norms.append(nn.BatchNorm1d(d_model))
            self.dropouts.append(nn.Dropout2d(dropout))

        self.pooling = pooling
        # Linear decoder
        # MODIFIED TO ALLOW FOR MODELS WITHOUT DECODER
        if(d_output is None):
            self.decoder = None
        else:
            self.decoder = nn.Linear(d_model, d_output)

    #MODIFIED
    def forward(self, x, rate=1.0):
        """
        Input x is shape (B, d_input, L) if transposed_input else (B, L, d_input)
        """
        x = self.encoder(x)  # (B, d_input, L) -> (B, d_model, L) if transposed_input else (B, L, d_input) -> (B, L, d_model)

        if(self.transposed_input is False):
            x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                # MODIFIED
                z = norm(z.transpose(-1, -2)).transpose(-1, -2) if self.layer_norm else norm(z)
            
            # Apply S4 block: we ignore the state input and output
            # MODIFIED
            z, _ = layer(z, rate=rate)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                # MODIFIED
                x = norm(x.transpose(-1, -2)).transpose(-1, -2) if self.layer_norm else norm(z)

        x = x.transpose(-1, -2) # (B, d_model, L) -> (B, L, d_model)

        # MODIFIED ALLOW TO DISABLE POOLING
        if(self.pooling):
            # Pooling: average pooling over the sequence length
            x = x.mean(dim=1)

        # Decode the outputs
        if(self.decoder is not None):
            x = self.decoder(x)  # (B, d_model) -> (B, d_output) if pooling else (B, L, d_model) -> (B, L, d_output)
            
        if(not self.pooling and self.transposed_input is True):
            x = x.transpose(-1, -2) # (B, L, d_output) -> (B, d_output, L)
        return x
    
    
    
def bn_drop_lin(n_in, n_out, bn=True, p=0., actn=None, layer_norm=False):
    "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
    layers = [nn.BatchNorm1d(n_in) if layer_norm is False else nn.LayerNorm(n_in)] if bn else []
    if p != 0: layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None: layers.append(actn)
    return layers


class S4ModelMM(nn.Module):

    def __init__(
        self, 
        d_input, # None to disable encoder
        d_output, # None to disable decoder
        d_state=64, #MODIFIED: N
        d_model=512, #MODIFIED: H
        tab_features=None,
        n_layers=4, 
        dropout=0.2,
        prenorm=False,
        l_max=1024,
        transposed_input=True, # behaves like 1d CNN if True else like a RNN with batch_first=True
        bidirectional=True, #MODIFIED
        layer_norm = True, # MODIFIED
        pooling = True, # MODIFIED
):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.transposed_input = transposed_input
        
        # MODIFIED TO ALLOW FOR MODELS WITHOUT ENCODER
        if(d_input is None):
            self.encoder = nn.Identity()
        else:
            self.encoder = nn.Conv1d(d_input, d_model, 1) if transposed_input else nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S42(
                    d_state=d_state,
                    l_max=l_max,
                    d_model=d_model, 
                    bidirectional=bidirectional,
                    postact='glu',
                    dropout=dropout, 
                    transposed=True,
                ))
            #MODIFIED TO ALLOW BATCH NORM MODELS
            self.layer_norm = layer_norm
            if(layer_norm):
                self.norms.append(nn.LayerNorm(d_model))
            else: #MODIFIED
                self.norms.append(nn.BatchNorm1d(d_model))
            self.dropouts.append(nn.Dropout2d(dropout))

        self.pooling = pooling
            
        self.meta_encoder = nn.Linear(tab_features, d_model)
        
        self.prelu  = nn.PReLU()
                
       #meta_modules = bn_drop_lin(tab_features, d_model, bn=False, actn=nn.PReLU()) +\
       #                 bn_drop_lin(d_model, d_model, bn=True, p=0.5, actn=nn.PReLU()) +\
       #                 bn_drop_lin(d_model, d_model, bn=True, p=0.5, actn=nn.PReLU())
        
        
        
        #self.meta_head = nn.Sequential(*meta_modules)
        
        #self.query_token = nn.Parameter(torch.randn(d_model))  # Learnable query token
        #self.key_matrix = nn.Parameter(torch.randn(d_model, d_model))  # Key matrix
        #self.value_matrix = nn.Parameter(torch.randn(d_model, d_model))  # Value matrix
        #self.num_heads = 12
        
        
        self.decoder = nn.Linear((d_model+1)*(d_model+1), d_output) # d_model*2, (d_model+1)*(d_model+1), d_model
        

    #MODIFIED
    def forward(self, x, meta_feats, rate=1.0):
        """
        Input x is shape (B, d_input, L) if transposed_input else (B, L, d_input)
        """
        x = self.encoder(x)  # (B, d_input, L) -> (B, d_model, L) if transposed_input else (B, L, d_input) -> (B, L, d_model)

        if(self.transposed_input is False):
            x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                # MODIFIED
                z = norm(z.transpose(-1, -2)).transpose(-1, -2) if self.layer_norm else norm(z)
            
            # Apply S4 block: we ignore the state input and output
            # MODIFIED
            z, _ = layer(z, rate=rate)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                # MODIFIED
                x = norm(x.transpose(-1, -2)).transpose(-1, -2) if self.layer_norm else norm(z)

        x = x.transpose(-1, -2) # (B, d_model, L) -> (B, L, d_model)
        
        x = x.mean(dim=1) # (B,d_model)
        
        
        # 0.86
        #meta_feats = self.meta_head(meta_feats.float()) # (B,features) -> (B,d_model)
        #x = torch.cat([x, meta_feats], axis=1) # (B,d_model*2) 
        
        
        meta_feats = self.prelu(self.meta_encoder(meta_feats.float()))
        
        x = torch.cat((x, torch.ones(x.shape[0],1, device=x.device)), dim=1)
        meta_feats = torch.cat((meta_feats, torch.ones(x.shape[0], 1, device=meta_feats.device)), dim=1)
        
        x = (x.unsqueeze(dim=1) * meta_feats.unsqueeze(dim=2)).view(x.shape[0], -1)
        
        # 0. 
        #meta_feats = self.prelu(self.meta_encoder(meta_feats.float()))
        #attention_x = F.softmax(torch.matmul(x, self.key_matrix) @ self.query_token.unsqueeze(-1) / self.num_heads, dim=0)
        #attention_meta = F.softmax(torch.matmul(meta_feats, self.key_matrix) @ self.query_token.unsqueeze(-1) / self.num_heads, dim=0)
        #weighted_x = torch.matmul(attention_x.unsqueeze(-1), x.unsqueeze(1)).squeeze(1)
        #weighted_meta = torch.matmul(attention_meta.unsqueeze(-1), meta_feats.unsqueeze(1)).squeeze(1)
        
        #x = weighted_x + weighted_meta
        
        
        x = self.decoder(x) # (B,d_model) -> (B,d_output)
            
        return x