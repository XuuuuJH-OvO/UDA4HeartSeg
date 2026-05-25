from .A4C import A4CDataset
from .builder import DATASETS


@DATASETS.register_module()
class A2CDataset(A4CDataset):  #A4CDataset
    CLASSES = A4CDataset.CLASSES
    PALETTE = A4CDataset.PALETTE

    def __init__(self, **kwargs):
        assert kwargs.get('split') in [None, 'train']
        if 'split' in kwargs:
            kwargs.pop('split')
        super(A2CDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_labelTrainIds.png',
            split=None,
            **kwargs)
