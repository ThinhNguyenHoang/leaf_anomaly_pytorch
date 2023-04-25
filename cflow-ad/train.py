import os, time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm
from visualize import *
from model import load_decoder_arch, load_encoder_arch, positionalencoding2d, activation, load_saliency_detector_arch, get_saliency_map
from utils import get_logp, rescale, Score_Observer, t2np, calculate_seg_pro_auc
from custom_datasets import *
from custom_models import *
from torchvision import transforms

OUT_DIR = './viz/'

gamma = 0.0
theta = torch.nn.Sigmoid()
log_theta = torch.nn.LogSigmoid()


def train_meta_epoch(c, epoch, loader,saliency_detector, encoder, decoders, optimizer, pool_layers, N):
    P = c.condition_vec
    L = c.pool_layers
    decoders = [decoder.train() for decoder in decoders]
    adjust_learning_rate(c, optimizer, epoch)
    I = len(loader)
    iterator = iter(loader)
    for sub_epoch in range(c.sub_epochs):
        train_loss = 0.0
        train_count = 0
        for i in range(I):
            # warm-up learning rate
            lr = warmup_learning_rate(c, epoch, i+sub_epoch*I, I*c.sub_epochs, optimizer)
            # sample batch
            try:
                image, _, _ = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                image, _, _ = next(iterator)
            # encoder prediction
            image = image.to(c.device)  # single scale
            if c.use_saliency:
                saliency_map = get_saliency_map(saliency_detector, image) # Bx1xHxW
            with torch.no_grad():
                _ = encoder(image)
            # train decoder
            e_list = list()
            c_list = list()
            for l, layer in enumerate(pool_layers):
                e = activation[layer].detach()  # BxCxHxW
                #
                B, C, H, W = e.size()
                S = H*W
                E = B*S
                # POSITIONAL ENCODING
                positional_encoding = positionalencoding2d(P, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1) # BxPxHxW
                condition_vector = positional_encoding
                if c.use_saliency:
                    saliency_resized = transforms.Resize([H, W])(saliency_map).unsqueeze(1)
                    condition_vector = torch.mul(saliency_resized, positional_encoding)
                # BxPxHxW -----> BxPx(HW)---------> Bx(HW)xP ----> BHWxP
                condition_vector_reshaped = condition_vector.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # BHWxP
                # BxCxHxW -----> BxCx(HW)---------> Bx(HW)xC ----> BHWxC
                # feature map --- encoding
                feature_map_reshaped = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC
                perm = torch.randperm(E).to(c.device)  # BHW
                decoder = decoders[l]
                #
                FIB = E//N  # number of fiber batches
                assert FIB > 0, 'MAKE SURE WE HAVE ENOUGH FIBERS, otherwise decrease N or batch-size!'
                for f in range(FIB):  # per-fiber processing
                    idx = torch.arange(f*N, (f+1)*N)
                    condition_patch = condition_vector_reshaped[perm[idx]]  # NxP
                    feature_patch = feature_map_reshaped[perm[idx]]  # NxC
                    if 'cflow' in c.dec_arch:
                        z, log_jac_det = decoder(feature_patch, [condition_patch,])
                    else:
                        z, log_jac_det = decoder(feature_patch)
                    #
                    decoder_log_prob = get_logp(C, z, log_jac_det)
                    log_prob = decoder_log_prob / C  # likelihood per dim
                    loss = -log_theta(log_prob)
                    optimizer.zero_grad()
                    loss.mean().backward()
                    optimizer.step()
                    train_loss += t2np(loss.sum())
                    train_count += len(loss)
        #
        mean_train_loss = train_loss / train_count
        if c.verbose:
            print('Epoch: {:d}.{:d} \t train loss: {:.4f}, lr={:.6f}'.format(epoch, sub_epoch, mean_train_loss, lr))
    #


def test_meta_epoch(c, epoch,loader, encoder, decoders, pool_layers, N, saliency_detector=None):
    # test
    if c.verbose:
        print('\nCompute loss and scores on test set:')
    #
    P = c.condition_vec
    decoders = [decoder.eval() for decoder in decoders]
    height = list()
    width = list()
    image_list = list()
    gt_label_list = list()
    gt_mask_list = list()
    test_dist = [list() for layer in pool_layers]
    test_loss = 0.0
    test_count = 0
    start = time.time()
    with torch.no_grad():
        for i, (image, label, mask) in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
            # save
            if c.viz:
                image_list.extend(t2np(image))
            # ground_truth label list
            gt_label_list.extend(t2np(label))
            gt_mask_list.extend(t2np(mask))
            # data
            image = image.to(c.device) # single scale
            _ = encoder(image)  # BxCxHxW
            if c.use_saliency:
                saliency_map = get_saliency_map(saliency_detector, image)
            for l, layer in enumerate(pool_layers):
                e = activation[layer]  # BxCxHxW
                #
                B, C, H, W = e.size()
                S = H*W
                E = B*S
                #
                if i == 0:  # get stats
                    height.append(H)
                    width.append(W)
                #
                # SALIENCY MAP
                # Positional encoding
                p = positionalencoding2d(P, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
                condition_vector = p
                if c.use_saliency:
                    saliency_resized = transforms.Resize([H, W])(saliency_map).unsqueeze(1)
                    condition_vector = torch.mul(condition_vector, saliency_resized)
                condition_vector_reshaped = condition_vector.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # BHWxP
                feature_map_reshaped = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC
                #
                m = F.interpolate(mask, size=(H, W), mode='nearest')
                m_r = m.reshape(B, 1, S).transpose(1, 2).reshape(E, 1)  # BHWx1
                #
                decoder = decoders[l]
                FIB = E//N + int(E%N > 0)  # number of fiber batches
                for f in range(FIB):
                    if f < (FIB-1):
                        idx = torch.arange(f*N, (f+1)*N)
                    else:
                        idx = torch.arange(f*N, E)
                    #
                    condition_patch = condition_vector_reshaped[idx]  # NxP
                    feature_patch = feature_map_reshaped[idx]  # NxC
                    m_p = m_r[idx] > 0.5  # Nx1
                    #
                    if 'cflow' in c.dec_arch:
                        z, log_jac_det = decoder(feature_patch, [condition_patch,])
                    else:
                        z, log_jac_det = decoder(feature_patch)
                    #
                    decoder_log_prob = get_logp(C, z, log_jac_det)
                    log_prob = decoder_log_prob / C  # likelihood per dim
                    loss = -log_theta(log_prob)
                    test_loss += t2np(loss.sum())
                    test_count += len(loss)
                    test_dist[l] = test_dist[l] + log_prob.detach().cpu().tolist()
    #
    fps = len(loader.dataset) / (time.time() - start)
    mean_test_loss = test_loss / test_count
    if c.verbose:
        print('Epoch: {:d} \t test_loss: {:.4f} and {:.2f} fps'.format(epoch, mean_test_loss, fps))
    #
    return height, width, image_list, test_dist, gt_label_list, gt_mask_list



def train(c):
    run_date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    L = c.pool_layers # number of pooled layers
    print('Number of pool layers =', L)
    encoder, pool_layers, pool_dims = load_encoder_arch(c, L)
    encoder = encoder.to(c.device).eval()
    # NF decoder
    saliency_detector = load_saliency_detector_arch(c)
    decoders = [load_decoder_arch(c, pool_dim) for pool_dim in pool_dims]
    decoders = [decoder.to(c.device) for decoder in decoders]

    # optimizer
    params = list(decoders[0].parameters())
    for l in range(1, L):
        params += list(decoders[l].parameters())
    optimizer = torch.optim.Adam(params, lr=c.lr)

    # data preparation
    train_loader, test_loader, val_loader = prepare_dataset(c)
    N = 256  # hyperparameter that increases batch size for the decoder model by N

    # stats
    det_roc_obs = Score_Observer('DET_AUROC')
    seg_roc_obs = Score_Observer('SEG_AUROC')
    seg_pro_obs = Score_Observer('SEG_AUPRO')
    accuracy_obs = Score_Observer('ACCURACY')
    precision_obs = Score_Observer('PRECISION')
    recall_obs = Score_Observer('RECALL')
    if c.action_type == 'norm-test':
        c.meta_epochs = 1
    for epoch in range(c.meta_epochs):
        if c.action_type == 'norm-test' and c.checkpoint:
            load_weights(encoder, decoders, c.checkpoint)
        elif c.action_type == 'norm-train':
            print('Train meta epoch: {}'.format(epoch))
            train_meta_epoch(c, epoch,train_loader, saliency_detector,  encoder, decoders, optimizer, pool_layers, N)
        else:
            raise NotImplementedError('{} is not supported action type!'.format(c.action_type))
        height, width, test_image_list, test_dist, gt_label_list, gt_mask_list = test_meta_epoch(
            c, epoch, test_loader, encoder, decoders, pool_layers, N, saliency_detector=saliency_detector)

        # PxEHW
        print('Heights/Widths', height, width)
        test_map = [list() for p in pool_layers]
        for l, p in enumerate(pool_layers):
            test_norm = torch.tensor(test_dist[l], dtype=torch.double)  # EHWx1
            test_norm-= torch.max(test_norm) # normalize likelihoods to (-Inf:0] by subtracting a constant
            test_prob = torch.exp(test_norm) # convert to probs in range [0:1]
            test_mask = test_prob.reshape(-1, height[l], width[l])
            test_mask = test_prob.reshape(-1, height[l], width[l])
            # upsample
            test_map[l] = F.interpolate(test_mask.unsqueeze(1),
                size=c.crp_size, mode='bilinear', align_corners=True).squeeze().numpy()

        # EVALUATION METRICS
        # score aggregation
        score_map = np.zeros_like(test_map[0]) # BxHxW
        for l, p in enumerate(pool_layers):
            score_map += test_map[l]
        score_mask = score_map
        # MASK | M
        # invert probs to anomaly scores
        # --> Lower likelihood --> Higher score --> Abnormal points
        # --> Max likelihood (or near max) --> Normal points
        super_mask = score_mask.max() - score_mask # scalar - BxHxW
        # det_aur_roc
        # SEG_AUROC
        gt_mask = np.squeeze(np.asarray(gt_mask_list, dtype=bool), axis=1)
        if not c.no_mask: # If have mask in dataset, best weights if seg score is best
            seg_roc_auc = roc_auc_score(gt_mask.flatten(), super_mask.flatten())
            save_best_seg_weights = seg_roc_obs.update(100.0*seg_roc_auc, epoch)
            if save_best_seg_weights and c.action_type != 'norm-test':
                save_weights(c, encoder, decoders, c.model, run_date)  # avoid unnecessary saves
            # calculate segmentation AUPRO
            # from https://github.com/YoungGod/DFR:
            if c.pro:  # and (epoch % 4 == 0):  # AUPRO is expensive to compute
                calculate_seg_pro_auc(super_mask, gt_mask, epoch, seg_pro_obs)

        # LABEL | Y
        score_label = np.max(super_mask, axis=(1, 2)) # score_label (B,) <-- max([B, H, W], axis=(1,2))
        gt_label = np.asarray(gt_label_list, dtype=bool)
        # auc_roc
        det_roc_auc = roc_auc_score(gt_label, score_label)
        save_weights_best_det_auc_roc = det_roc_obs.update(100.0*det_roc_auc, epoch)

        # DET_AUROC
        if c.no_mask and c.action_type != 'norm-test':
            # precision | accuracy | recall
            binary_score_label = rescale_and_score(score_label)
            accuracy = accuracy_score(gt_label, binary_score_label)
            _ = accuracy_obs.update(accuracy *100, epoch)
            precision = precision_score(gt_label, binary_score_label)
            precision_best_weight = precision_obs.update(precision *100, epoch)
            recall = recall_score(gt_label, binary_score_label)
            _ = recall_obs.update(recall *100, epoch)
            if save_weights_best_det_auc_roc:
                save_weights(c,encoder, decoders, c.model, run_date)

    #
    # save_results(det_roc_obs, seg_roc_obs, seg_pro_obs, c.model, c.class_name, run_date)
    save_model_metrics(c, [accuracy_obs, precision_obs, recall_obs, det_roc_obs, seg_roc_obs],c.model, c.class_name, run_date)
    # export visualuzations
    if c.viz:
        save_visualization(c, test_image_list, super_mask, gt_mask, gt_label, score_label)