import os, random, time, math
import numpy as np
import torch
import torchvision
# import timm
# from timm.data import resolve_data_config
from config import get_args
from train import train
import utils.cloud_utils as cloud_utils

print(f"Notebook runtime: {'GPU' if torch.cuda.is_available() else 'CPU'}")
print(f"PyTorch version : {torch.__version__}")
print(f"PyTorch Vision version : {torchvision.__version__}")
def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def handle_dataset_path(c):
    # DATASET SPECIFIC ARGUMENTS ADJUSTMENT
    if c.dataset == 'mvtec':
        c.data_path = './data/MVTec-AD'
        raise NotImplementedError('{} is not supported for running on cloud!'.format(c.dataset))
        # if c.gcp:
        #     c.data_path = os.path.join(cloud_utils.get_bucket_prefix(), 'datasets','MVTec-AD')
    elif c.dataset == 'plant_village':
        c.data_path = './data/PlantVillage'
        c.no_mask = True
        if c.gcp:
            c.data_path = os.path.join(cloud_utils.get_bucket_prefix(), 'datasets','PlantVillage')
    else:
        raise NotImplementedError('{} is not supported dataset!'.format(c.dataset))
def handle_weight_dir_path(c):
    c.weight_dir = './weights'
    c.result_dir = './results'
    if c.gcp:
        c.weight_dir = os.path.join(cloud_utils.get_bucket_prefix(), 'weights')
        c.result_dir = os.path.join(cloud_utils.get_bucket_prefix(), 'results')

# Make sure the submodels (resnet, u2net) load the correct saved page for evalation
def handle_submodel_weight_paths(c):
    if c.use_saliency:
        c.u2net_weight_path = 'custom_models/u2net/saved_models/u2net/u2net.pth'
    if c.gcp:
        # load from buckets
        c.wide_resnet50_weight_path = os.path.join(cloud_utils.get_bucket_prefix(), 'resnet','weights', 'wide_resnet50_2-95faca4d.pth')
        c.u2net_weight_path = os.path.join(cloud_utils.get_bucket_prefix(), 'u2net', 'weights','u2net.pth')
def main(c):
    # model
    if c.action_type in ['norm-train', 'norm-test']:
        c.model = f"{c.dataset}{'_saliency' if c.use_saliency else ''}_{c.enc_arch}_{c.dec_arch}_pl{c.pool_layers}_cb{c.coupling_blocks}_inp{c.input_size}_run{c.run_name}_{c.class_name}"
    else:
        raise NotImplementedError('{} is not supported action-type!'.format(c.action_type))
    # image
    c.img_size = (c.input_size, c.input_size)  # HxW format
    c.crp_size = (c.input_size, c.input_size)  # HxW format
    if c.dataset == 'stc':
        c.norm_mean, c.norm_std = 3*[0.5], 3*[0.225]
    else:
        c.norm_mean, c.norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    #
    c.img_dims = [3] + list(c.img_size)
    # network hyperparameters
    c.clamp_alpha = 1.9  # see paper equation 2 for explanation
    c.condition_vec = 128
    c.dropout = 0.0  # dropout in s-t-networks
    # output settings
    c.verbose = True
    c.hide_tqdm_bar = True
    c.save_results = True
    # unsup-train
    c.print_freq = 2
    c.temp = 0.5
    c.lr_decay_epochs = [i*c.meta_epochs//100 for i in [50,75,90]]
    print('LR schedule: {}'.format(c.lr_decay_epochs))
    c.lr_decay_rate = 0.1
    c.lr_warm_epochs = 2
    c.lr_warm = True
    c.lr_cosine = True
    if c.lr_warm:
        c.lr_warmup_from = c.lr/10.0
        if c.lr_cosine:
            eta_min = c.lr * (c.lr_decay_rate ** 3)
            c.lr_warmup_to = eta_min + (c.lr - eta_min) * (
                    1 + math.cos(math.pi * c.lr_warm_epochs / c.meta_epochs)) / 2
        else:
            c.lr_warmup_to = c.lr
    # HANLDE DATA AND WEIGHT DIR PATHS:
    # Load images data from the cloud storage bucket
    handle_dataset_path(c)
    # Load submodels' saved weight from the cloud storage
    handle_submodel_weight_paths(c)
    # Save the model weight to the cloud storage
    handle_weight_dir_path(c)
    ########
    os.environ['CUDA_VISIBLE_DEVICES'] = c.gpu
    c.use_cuda = not c.no_cuda and torch.cuda.is_available()
    init_seeds(seed=int(time.time()))
    if c.use_cuda:
        print("=================== PREPARING TO TRAIN WITH GPU ==============================")
    c.device = torch.device("cuda" if c.use_cuda else "cpu")
    # selected function:
    if c.action_type in ['norm-train', 'norm-test']:
        train(c)
    else:
        raise NotImplementedError('{} is not supported action-type!'.format(c.action_type))
    print("TRAINING WITH CONFIG:", str(c))

if __name__ == '__main__':
    c = get_args()
    try:
        main(c)
    except Exception as e:
        print(str(e))

