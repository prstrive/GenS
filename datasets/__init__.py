import torch
import torch.distributed as dist
from torch.utils.data import RandomSampler, SequentialSampler, DistributedSampler, DataLoader

from .dtu import DTUDataset
from .dtu_finetune import DTUDatasetFinetune
from .bmvs import BMVSDataset
from .bmvs_finetune import BMVSDatasetFinetune


def collect_fn(data):
    return data[0]


def get_loader(conf, mode, distributed):

    batch_size = 1
    dataset_name = conf.get_string("dataset_name")
    
    if dataset_name == "DTUDataset":
        dataset = DTUDataset(conf, mode)
    elif dataset_name == "BMVSDataset":
        dataset = BMVSDataset(conf, mode)
    elif dataset_name == "DTUDatasetFinetune":
        dataset = DTUDatasetFinetune(conf, mode)
    elif dataset_name == "BMVSDatasetFinetune":
        dataset = BMVSDatasetFinetune(conf, mode)
        
    if mode == "finetune":
        return dataset

    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    else:
        sampler = RandomSampler(dataset) if (mode == "train") else SequentialSampler(dataset)
    
    data_loader = DataLoader(dataset, batch_size, sampler=sampler, num_workers=8, drop_last=(mode == "train"), pin_memory=False, collate_fn=collect_fn)

    return data_loader, sampler, dataset