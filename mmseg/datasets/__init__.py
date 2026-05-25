# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

from .builder import DATASETS, PIPELINES, build_dataloader, build_dataset
from .cityscapes import CityscapesDataset
from .custom import CustomDataset
from .dataset_wrappers import ConcatDataset, RepeatDataset
from .dark_zurich import DarkZurichDataset
from .gta import GTADataset
from .uda_dataset import UDADataset
from .A2C import A2CDataset
from .A4C import A4CDataset

__all__ = [
    'CustomDataset',
    'build_dataloader',
    'ConcatDataset',
    'RepeatDataset',
    'DATASETS',
    'build_dataset',
    'PIPELINES',
    'CityscapesDataset',
    'DarkZurichDataset',
    'GTADataset',
    'UDADataset',
    'A2CDataset',
    'A4CDataset'
]
