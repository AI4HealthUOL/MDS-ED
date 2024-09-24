__all__ = ['tsdata_seq','tsdata_seq_static','tsdata_seq_idxs','tsdata_seq_static_idxs','ConcatTimeSeriesDataset','TimeSeriesDataset','TimeSeriesDatasetConfig']

import numpy as np
import torch
import torch.utils.data
from .time_series_dataset_transforms import Compose

#Note: due to issues with the numpy rng for multiprocessing (https://github.com/pytorch/pytorch/issues/5059) that could be fixed by a custom worker_init_fn we use random throught for convenience
import random

#Note: multiprocessing issues with python lists and dicts (https://github.com/pytorch/pytorch/issues/13246) and pandas dfs (https://github.com/pytorch/pytorch/issues/5902)
#import multiprocessing as mp

from dataclasses import dataclass
import pandas as pd
from typing import Union, Any
import pathlib


from collections import namedtuple

tsdata_seq = namedtuple("tsdata_seq",("seq","label"))
tsdata_seq_static = namedtuple("tsdata_seq_static",("seq","label","static"))
tsdata_seq_cat = namedtuple("tsdata_seq_cat",("seq","label","static_cat"))
tsdata_seq_static_cat = namedtuple("tsdata_seq_static_cat",("seq","label","static","static_cat"))

tsdata_seq_idxs = namedtuple("tsdata_seq_idxs",("seq","label","seq_idxs"))
tsdata_seq_cat_idxs = namedtuple("tsdata_seq_cat_idxs",("seq","label","static_cat","seq_idxs"))
tsdata_seq_static_idxs = namedtuple("tsdata_seq_static_idxs",("seq","label","static","seq_idxs"))
tsdata_seq_static_cat_idxs = namedtuple("tsdata_seq_static_cat_idxs",("seq","label","static","static_cat","seq_idxs"))

def arrays_equal_with_nans(arr1, arr2):
    '''helper function to compare arrays with nans'''
    return ((arr1 == arr2) | (np.isnan(arr1) & np.isnan(arr2))).all()

class ConcatTimeSeriesDataset(torch.utils.data.ConcatDataset):
    '''ConcatDataset that handles id mapping correctly (to allow to aggregate predictions)'''
    def __init__(self, datasets):
        super().__init__(datasets)
        idmaps = []
        for dataset_idx,ds in enumerate(self.datasets):
            idmap = ds.get_id_mapping()
            remap_dict = {x:j+(self.cumulative_sizes[dataset_idx-1] if dataset_idx>0 else 0) for j,x in enumerate(np.unique(idmap))}
            idmaps.append(np.array([remap_dict[x] for x in idmap]))
        self.df_idx_mapping = np.concatenate(idmaps)

    def get_id_mapping(self):
        return self.df_idx_mapping

    def aggregate_predictions(self, preds,targs,idmap=None,aggregate_fn = np.mean,verbose=False):
        return self.datasets[0].aggregate_predictions(preds, targs, self.df_idx_mapping if idmap is None else idmap, aggregate_fn, verbose)


class TimeSeriesDataset(torch.utils.data.Dataset):
    """timeseries dataset with partial crops."""

    def __init__(self, hparams):
        """
        accepts three kinds of input:
        1) filenames pointing to aligned numpy arrays [timesteps,channels,...] for data and either integer labels or filename pointing to numpy arrays[timesteps,...] e.g. for annotations
        2) memmap_filename to memmap file (same argument that was passed to reformat_as_memmap) for data [concatenated,...] and labels- data column in df corresponds to index in this memmap; memmap_label_filename can normally kept as None (will use the memmap_label file in the same directory in this case)
        3) npy_data [samples,ts,...] (either path or np.array directly- also supporting variable length input) - data column in df corresponds to sampleid

        transforms: list of callables (transformations) or single instance e.g. from torchvision.transforms.Compose (applied in the specified order i.e. leftmost element first)
        
        col_lbl = None: return dummy label 0 (e.g. for unsupervised pretraining)
        cols_static: (optional) list of cols with extra static information (continuous-valued)
        cols_static_cat: (optional) list of cols with extra static information (categorical)
        fs_annotation_over_fs_data over ratio of sampling frequencies
        return_idxs: returns sample_idx from the underlying dataframe and start_idx and end_idx within the sequence (for aligned sequences e.g. spectra or for certain contrastive approaches such as ts2vec)
        """
        super().__init__()
        assert not((hparams.memmap_filename is not None) and (hparams.npy_data is not None))
        # require integer entries if using memmap or npy
        assert (hparams.memmap_filename is None and hparams.npy_data is None) or (hparams.df[hparams.col_data].dtype==np.int64 or hparams.df[hparams.col_data].dtype==np.int32 or hparams.df[hparams.col_data].dtype==np.int16)
        # keys (in column data) have to be unique
        assert(hparams.allow_multiple_keys or len(hparams.df[hparams.col_data].unique())==len(hparams.df))

        self.timeseries_df_data = np.array(hparams.df[hparams.col_data])
        if(self.timeseries_df_data.dtype not in [np.int16, np.int32, np.int64]):
            assert(hparams.memmap_filename is None and hparams.npy_data is None) #only for filenames in mode files
            self.timeseries_df_data = np.array(hparams.df[hparams.col_data].astype(str)).astype(np.string_)

        if(hparams.col_lbl is None):# use dummy labels
            self.timeseries_df_label = np.zeros(len(hparams.df))
        else: # use actual labels
            if(isinstance(hparams.df[hparams.col_lbl].iloc[0],list) or isinstance(hparams.df[hparams.col_lbl].iloc[0],np.ndarray)):#stack arrays/lists for proper batching
                self.timeseries_df_label = np.stack(hparams.df[hparams.col_lbl])
            else: # single integers/floats
                self.timeseries_df_label = np.array(hparams.df[hparams.col_lbl])

            if(not(hparams.annotation and hparams.memmap_filename is not None)):#skip if memmap and annotation        
                if(self.timeseries_df_label.dtype not in [np.int16, np.int32, np.int64, np.float32, np.float64]): #everything else cannot be batched anyway mp.Manager().list(self.timeseries_df_label)
                    assert(hparams.annotation and hparams.memmap_filename is None and hparams.npy_data is None)#only for filenames in mode files
                    self.timeseries_df_label = np.array(hparams.df[hparams.col_lbl].apply(lambda x:str(x))).astype(np.string_)

        def concat_columns(row):
            return [item for col in row for item in (col if isinstance(col, np.ndarray) else [col])]

        if(hparams.cols_static is not None):
            self.timeseries_df_static = np.array(hparams.df[hparams.cols_static].apply(concat_columns, axis=1).to_list())
            self.timeseries_df_static = np.squeeze(self.timeseries_df_static,axis=1)#remove unit axis
            self.static = True
        else:
            self.static = False

        if(hparams.cols_static_cat is not None):
            self.timeseries_df_static_cat = np.array(hparams.df[hparams.cols_static_cat].apply(concat_columns, axis=1).to_list())
            self.timeseries_df_static_cat = np.squeeze(self.timeseries_df_static_cat,axis=1)#remove unit axis
            self.static_cat = True
        else:
            self.static_cat = False

        self.output_size = hparams.output_size
        self.data_folder = hparams.data_folder
        self.transforms = Compose(hparams.transforms) if isinstance(hparams.transforms,list) else hparams.transforms
        #if(isinstance(self.transforms,list) or isinstance(self.transforms,np.ndarray)):
        #    print("Warning: the use of lists as arguments for transforms is discouraged")
        self.annotation = hparams.annotation
        self.col_lbl = hparams.col_lbl

        self.mode="files"
        self.fs_annotation_over_fs_data = hparams.fs_annotation_over_fs_data
        self.return_idxs = hparams.return_idxs
        
        if(hparams.memmap_filename is not None):
            self.memmap_meta_filename = hparams.memmap_filename.parent/(hparams.memmap_filename.stem+"_meta.npz")
            self.mode="memmap"
            memmap_meta = np.load(self.memmap_meta_filename, allow_pickle=True)
            self.memmap_start = memmap_meta["start"].astype(np.int64)# cast as integers to be on the safe side
            self.memmap_shape = memmap_meta["shape"].astype(np.int64)
            self.memmap_length = memmap_meta["length"].astype(np.int64)
            self.memmap_file_idx = memmap_meta["file_idx"].astype(np.int64)
            self.memmap_dtype = np.dtype(str(memmap_meta["dtype"]))
            self.memmap_filenames = np.array(memmap_meta["filenames"]).astype(np.string_)#save as byte to avoid issue with mp
            if(hparams.annotation):
                #by default use the memmap_label.npy in the same directory as the signal memmap file
                memmap_label_filename=hparams.memmap_label_filename if hparams.memmap_label_filename is not None else self.memmap_meta_filename.parent/("_".join(self.memmap_meta_filename.stem.split("_")[:-1])+"_label.npy")
                self.memmap_meta_filename_label = hparams.memmap_filename.parent/(memmap_label_filename.stem+"_meta.npz")
                memmap_meta_label =np.load(self.memmap_meta_filename_label, allow_pickle=True)
                self.memmap_start_label = memmap_meta_label["start"].astype(np.int64)
                self.memmap_shape_label = memmap_meta_label["shape"].astype(np.int64)
                self.memmap_length_label = memmap_meta_label["length"].astype(np.int64)
                self.memmap_file_idx_label = memmap_meta_label["file_idx"].astype(np.int64)
                self.memmap_dtype_label = np.dtype(str(memmap_meta_label["dtype"]))
                self.memmap_filenames_label = np.array(memmap_meta_label["filenames"]).astype(np.string_)
        elif(hparams.npy_data is not None):
            self.mode="npy"
            if(isinstance(hparams.npy_data,np.ndarray) or isinstance(hparams.npy_data,list)):
                self.npy_data = np.array(hparams.npy_data)
                assert(hparams.annotation is False)
            else:
                self.npy_data = np.load(hparams.npy_data, allow_pickle=True)
            if(hparams.annotation):
                self.npy_data_label = np.load(hparams.npy_data.parent/(hparams.npy_data.stem+"_label.npy"), allow_pickle=True)

        self.random_crop = hparams.random_crop
        self.sample_items_per_record = hparams.sample_items_per_record

        self.df_idx_mapping=[]
        self.start_idx_mapping=[]
        self.end_idx_mapping=[]

        for df_idx,(id,row) in enumerate(hparams.df.iterrows()):
            if(self.mode=="files"):
                data_length = row["data_length"]
            elif(self.mode=="memmap"):
                data_length= self.memmap_length[row[hparams.col_data]]
            else: #npy
                data_length = len(self.npy_data[row[hparams.col_data]])

            if(hparams.chunk_length == 0):#do not split
                idx_start = [hparams.start_idx]
                idx_end = [data_length]
            else:
                idx_start = list(range(hparams.start_idx,data_length,hparams.chunk_length if hparams.stride is None else hparams.stride))
                idx_end = [min(l+hparams.chunk_length, data_length) for l in idx_start]

            #remove final chunk(s) if too short
            for i in range(len(idx_start)):
                if(idx_end[i]-idx_start[i]< hparams.min_chunk_length):
                    del idx_start[i:]
                    del idx_end[i:]
                    break
            #append to lists
            for _ in range(hparams.copies+1):
                for i_s,i_e in zip(idx_start,idx_end):
                    self.df_idx_mapping.append(df_idx)
                    self.start_idx_mapping.append(i_s)
                    self.end_idx_mapping.append(i_e)
        #convert to np.array to avoid mp issues with python lists
        self.df_idx_mapping = np.array(self.df_idx_mapping)
        self.start_idx_mapping = np.array(self.start_idx_mapping)
        self.end_idx_mapping = np.array(self.end_idx_mapping)
            
    def __len__(self):
        return len(self.df_idx_mapping)

    @property
    def is_empty(self):
        return len(self.df_idx_mapping)==0

    def __getitem__(self, idx):
        lst=[]
        for _ in range(self.sample_items_per_record):
            #determine crop idxs
            timesteps= self.get_sample_length(idx)

            if(self.random_crop):#random crop
                if(timesteps==self.output_size):
                    start_idx_rel = 0
                else:
                    start_idx_rel = random.randint(0, timesteps - self.output_size -1)#np.random.randint(0, timesteps - self.output_size)
            else:
                start_idx_rel =  (timesteps - self.output_size)//2
            if(self.sample_items_per_record==1):
                return self._getitem(idx,start_idx_rel)
            else:
                lst.append(self._getitem(idx,start_idx_rel))
        return tuple(lst)

    def _getitem(self, idx,start_idx_rel):
        #low-level function that actually fetches the data
        df_idx = self.df_idx_mapping[idx]
        start_idx = self.start_idx_mapping[idx]
        end_idx = self.end_idx_mapping[idx]
        #determine crop idxs
        timesteps= end_idx - start_idx
        assert(timesteps>=self.output_size)
        start_idx_crop = start_idx + start_idx_rel
        end_idx_crop = start_idx_crop+self.output_size
        if(self.annotation):
            start_idx_crop_label = int(np.round(start_idx_crop*self.fs_annotation_over_fs_data))
            end_idx_crop_label = start_idx_crop_label+int(np.round(self.output_size*self.fs_annotation_over_fs_data))

        #print(idx,start_idx,end_idx,start_idx_crop,end_idx_crop)
        #load the actual data
        if(self.mode=="files"):#from separate files
            data_filename = str(self.timeseries_df_data[df_idx],encoding='utf-8') #todo: fix potential issues here
            if self.data_folder is not None:
                data_filename = self.data_folder/data_filename
            data = np.load(data_filename, allow_pickle=True)[start_idx_crop:end_idx_crop] #data type has to be adjusted when saving to npy

            ID = data_filename.stem

            if(self.annotation is True):
                label_filename = str(self.timeseries_df_label[df_idx],encoding='utf-8')
                if self.data_folder is not None:
                    label_filename = self.data_folder/label_filename
                label = np.load(label_filename, allow_pickle=True)[start_idx_crop_label:end_idx_crop_label] #data type has to be adjusted when saving to npy
            else:
                label = self.timeseries_df_label[df_idx] #input type has to be adjusted in the dataframe
        elif(self.mode=="memmap"): #from one memmap file
            memmap_idx = self.timeseries_df_data[df_idx] #grab the actual index (Note the df to create the ds might be a subset of the original df used to create the memmap)
            memmap_file_idx = self.memmap_file_idx[memmap_idx]
            idx_offset = self.memmap_start[memmap_idx]

            #wi = torch.utils.data.get_worker_info()
            #pid = 0 if wi is None else wi.id#os.getpid()
            #print("idx",idx,"ID",ID,"idx_offset",idx_offset,"start_idx_crop",start_idx_crop,"df_idx", self.df_idx_mapping[idx],"pid",pid)
            mem_filename = str(self.memmap_filenames[memmap_file_idx],encoding='utf-8')
            mem_file = np.memmap(self.memmap_meta_filename.parent/mem_filename, self.memmap_dtype, mode='r', shape=tuple(self.memmap_shape[memmap_file_idx]))
            data = np.copy(mem_file[idx_offset + start_idx_crop: idx_offset + end_idx_crop])
            del mem_file
            #print(mem_file[idx_offset + start_idx_crop: idx_offset + end_idx_crop])
            if(self.annotation):
                memmap_file_idx_label = self.memmap_file_idx_label[memmap_idx]
                idx_offset_label = self.memmap_start_label[memmap_idx]

                mem_filename_label = str(self.memmap_filenames_label[memmap_file_idx_label],encoding='utf-8')
                mem_file_label = np.memmap(self.memmap_meta_filename_label.parent/mem_filename_label, self.memmap_dtype_label, mode='r', shape=tuple(self.memmap_shape_label[memmap_file_idx]))
                
                label = np.copy(mem_file_label[idx_offset_label + start_idx_crop_label: idx_offset_label + end_idx_crop_label])
                del mem_file_label
            else:
                label = self.timeseries_df_label[df_idx]
        else:#single npy array
            ID = self.timeseries_df_data[df_idx]

            data = self.npy_data[ID][start_idx_crop:end_idx_crop]

            if(self.annotation):
                label = self.npy_data_label[ID][start_idx_crop:end_idx_crop]
            else:
                label = self.timeseries_df_label[df_idx]

        sample = (data, label, self.timeseries_df_static[df_idx] if self.static else None, self.timeseries_df_static_cat[df_idx] if self.static_cat else None,np.array([df_idx,start_idx_crop,end_idx_crop]))
        
        # consistency check: make sure that data and annotation lengths match (check here because transforms might change the shape of the annotation)
        assert(self.annotation is False or len(sample[1])==int(np.round(self.fs_annotation_over_fs_data*len(sample[0]))))
        sample = self.transforms(sample)

        if(self.return_idxs):
            if(self.static is True and self.static_cat is True):
                return tsdata_seq_static_cat_idxs(sample[0],sample[1], sample[2], sample[3], sample[4])
            elif(self.static is True):
                return tsdata_seq_static_idxs(sample[0],sample[1], sample[2], sample[4])
            elif(self.static_cat is True):
                return tsdata_seq_cat_idxs(sample[0],sample[1], sample[3], sample[4])
            else:
                return tsdata_seq_idxs(sample[0], sample[1], sample[4])
        else:
            if(self.static is True and self.static_cat is True):
                return tsdata_seq_static_cat(sample[0],sample[1], sample[2], sample[3])
            elif(self.static is True):
                return tsdata_seq_static(sample[0],sample[1], sample[2])
            elif(self.static_cat is True):
                return tsdata_seq_cat(sample[0],sample[1], sample[3])
            else:
                return tsdata_seq(sample[0], sample[1])
        

    def get_sampling_weights(self, class_weight_dict,length_weighting=False, timeseries_df_group_by_col=None):
        '''
        class_weight_dict: dictionary of class weights
        length_weighting: weigh samples by length
        timeseries_df_group_by_col: column of the pandas df used to create the object'''
        assert(self.annotation is False)
        assert(length_weighting is False or timeseries_df_group_by_col is None)
        weights = np.zeros(len(self.df_idx_mapping),dtype=np.float32)
        length_per_class = {}
        length_per_group = {}
        for iw,(i,s,e) in enumerate(zip(self.df_idx_mapping,self.start_idx_mapping,self.end_idx_mapping)):
            label = self.timeseries_df_label[i]
            weight = class_weight_dict[label]
            if(length_weighting):
                if label in length_per_class.keys():
                    length_per_class[label] += e-s
                else:
                    length_per_class[label] = e-s
            if(timeseries_df_group_by_col is not None):
                group = timeseries_df_group_by_col[i]
                if group in length_per_group.keys():
                    length_per_group[group] += e-s
                else:
                    length_per_group[group] = e-s
            weights[iw] = weight

        if(length_weighting):#need second pass to properly take into account the total length per class
            for iw,(i,s,e) in enumerate(zip(self.df_idx_mapping,self.start_idx_mapping,self.end_idx_mapping)):
                label = self.timeseries_df_label[i]
                weights[iw]= (e-s)/length_per_class[label]*weights[iw]
        if(timeseries_df_group_by_col is not None):
            for iw,(i,s,e) in enumerate(zip(self.df_idx_mapping,self.start_idx_mapping,self.end_idx_mapping)):
                group = timeseries_df_group_by_col[i]
                weights[iw]= (e-s)/length_per_group[group]*weights[iw]

        weights = weights/np.min(weights)#normalize smallest weight to 1
        return weights

    def get_id_mapping(self):
        return self.df_idx_mapping

    def get_sample_id(self,idx):
        return self.df_idx_mapping[idx]

    def get_sample_length(self,idx):
        return self.end_idx_mapping[idx]-self.start_idx_mapping[idx]

    def get_sample_start(self,idx):
        return self.start_idx_mapping[idx]

    def aggregate_predictions(self, preds,targs=None,idmap=None,aggregate_fn = np.mean,verbose=False):
        '''
        aggregates potentially multiple predictions per sample (can also pass targs for convenience)
        idmap: idmap as returned by TimeSeriesCropsDataset's get_id_mapping (uses self.get_id_mapping by default)
        preds: ordered predictions as returned by learn.get_preds()
        aggregate_fn: function that is used to aggregate multiple predictions per sample (most commonly np.amax or np.mean)
        '''
        idmap = self.get_id_mapping() if idmap is None else idmap
        if(idmap is not None and len(idmap)!=len(np.unique(idmap))):
            if(verbose):
                print("aggregating predictions...")
            preds_aggregated = []
            targs_aggregated = []
            for i in np.unique(idmap):
                preds_local = preds[np.where(idmap==i)[0]]
                preds_aggregated.append(aggregate_fn(preds_local,axis=0))
                if targs is not None:
                    targs_local = targs[np.where(idmap==i)[0]]
                    #assert(np.all(targs_local==targs_local[0])) #all labels have to agree
                    assert(np.all([np.array_equal(t, targs_local[0], equal_nan=True) for t in targs_local])) #all labels have to agree (including nans)
                    targs_aggregated.append(targs_local[0])
            if(targs is None):
                return np.array(preds_aggregated)
            else:
                return np.array(preds_aggregated),np.array(targs_aggregated)
        else:
            if(targs is None):
                return preds
            else:
                return preds,targs

@dataclass
class TimeSeriesDatasetConfig:
    df:pd.DataFrame
    output_size:int
    chunk_length:int
    min_chunk_length:int
    memmap_filename:Union[str,pathlib.PosixPath,None]=None
    memmap_label_filename:Union[str,pathlib.PosixPath,None]=None
    npy_data:Union[np.ndarray,list,None]=None
    random_crop:bool=True
    data_folder:Union[str,pathlib.PosixPath,None]=None
    copies:int=0
    col_data:str="data"
    col_lbl:Union[str,None]="label"
    cols_static:Union[str,None]=None
    cols_static_cat:Union[str,None]=None
    stride:Union[int,None]=None
    start_idx:int=0
    annotation:bool=False
    transforms:Any=None
    sample_items_per_record:int=1
    fs_annotation_over_fs_data:float=1.
    return_idxs:bool=False
    allow_multiple_keys:bool=False #in the df allow multiple rows with identical IDs