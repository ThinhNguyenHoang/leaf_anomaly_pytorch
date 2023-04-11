from .stc import StcDataset
from .mvtec import MVTecDataset
from .plant_village import PlantVillageDataset
import torch

def prepare_dataset(c):
    # data
    kwargs = {'num_workers': c.workers, 'pin_memory': True} if c.use_cuda else {}

    if c.dataset == 'mvtec':
        train_dataset = MVTecDataset(c, is_train=True)
        test_dataset  = MVTecDataset(c, is_train=False)
    elif c.dataset == 'stc':
        train_dataset = StcDataset(c, is_train=True)
        test_dataset  = StcDataset(c, is_train=False)
    elif c.dataset == 'plant_village':
        train_dataset = PlantVillageDataset(c, is_train=True)
        test_dataset = PlantVillageDataset(c, is_train=False)
    else:
        raise NotImplementedError('{} is not supported dataset!'.format(c.dataset))
    #
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True, drop_last=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=c.batch_size, shuffle=False, drop_last=False, **kwargs)
    print('train/test loader length', len(train_loader.dataset), len(test_loader.dataset))
    print('train/test loader batches', len(train_loader), len(test_loader))
    return train_loader, test_loader