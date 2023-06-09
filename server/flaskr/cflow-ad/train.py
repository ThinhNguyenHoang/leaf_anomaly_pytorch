import os, time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix, det_curve, auc, f1_score
from tqdm import tqdm
from custom_models.utils import parse_checkpoint_filename
from utils.viz_utils import *
from model import load_decoder_arch, load_encoder_arch, positionalencoding2d, activation, load_saliency_detector_arch, get_saliency_map, ClassificationHead, Wide, train_class_head
from utils.score_utils import get_logp, rescale, Score_Observer, t2np, calculate_seg_pro_auc, get_anomaly_score, rescale_and_score, score_sigmoid, find_best_thresh_hold_sig, weight_precision_recall
from custom_datasets import *
from custom_models import *
from torchvision import transforms
import utils.cv_utils as cv_utils
OUT_DIR = './viz/'


gamma = 0.0
theta = torch.nn.Sigmoid()
log_sigmoid = torch.nn.LogSigmoid()

DET_AUC_ROC = 'DET_AUCROC'
PREC_REC_AUC = 'PREC_REC_AUC'
F1_SCORE = 'F1_SCORE'
SEG_AUROC = 'SEG_AUROC'
SEG_AUPRO = 'SEG_AUPRO'
ACCURACY = 'ACCURACY'
PRECISION = 'PRECISION'
RECALL = 'RECALL'
CF_MATRIX = 'CF_MATRIX'
STAT_DICT = {
    DET_AUC_ROC: 0,
    PREC_REC_AUC: 0,
    F1_SCORE: 0,
    SEG_AUPRO: 0,
    SEG_AUROC: 0,
    ACCURACY: 0,
    PRECISION: 0,
    RECALL: 0,
    CF_MATRIX:[0,0,0,0]
}

def update_stat_dict(accuracy, precision, recall, cf_matrix, seg_pro_auc, seg_roc_auc, det_roc_auc, prec_rec_auc, f1):
    STAT_DICT[RECALL] = recall
    STAT_DICT[ACCURACY] = accuracy
    STAT_DICT[PRECISION] = precision
    STAT_DICT[CF_MATRIX] = cf_matrix
    STAT_DICT[SEG_AUPRO] = seg_pro_auc
    STAT_DICT[SEG_AUROC] = seg_roc_auc
    STAT_DICT[DET_AUC_ROC] = det_roc_auc
    STAT_DICT[PREC_REC_AUC] = prec_rec_auc
    STAT_DICT[F1_SCORE] = f1

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
            if 'saliency' in c.sub_arch:
                saliency_map = get_saliency_map(saliency_detector, image) # Bx1xHxW
            with torch.no_grad():
                _ = encoder(image)
            for l, layer in enumerate(pool_layers):
                e = activation[layer].detach()  # BxCxHxW
                #
                B, C, H, W = e.size()
                S = H*W
                E = B*S
                # POSITIONAL ENCODING
                positional_encoding = positionalencoding2d(P, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1) # BxPxHxW
                condition_vector = positional_encoding
                if 'saliency' in c.sub_arch:
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
                    loss = -log_sigmoid(log_prob)
                    optimizer.zero_grad()
                    loss.mean().backward()
                    optimizer.step()
                    train_loss += t2np(loss.sum())
                    train_count += len(loss)
        #
        mean_train_loss = train_loss / train_count
        if c.verbose:
            print('Epoch: {:d}.{:d} \t train loss: {:.4f}, lr={:.6f}'.format(epoch, sub_epoch, mean_train_loss, lr))


def test_meta_epoch(c, epoch, loader, encoder, decoders, pool_layers, N, saliency_detector=None, class_head=None, should_train_class_head=True):
    #
    P = c.condition_vec
    decoders = [decoder.eval() for decoder in decoders]
    height = list()
    width = list()
    image_list = list()
    gt_label_list = list()
    gt_mask_list = list()
    saliency_image_list = list()
    test_dist = [list() for layer in pool_layers]
    test_loss = 0.0
    test_count = 0
    detection_loss = None
    start = time.time()
    with torch.no_grad():
        for i, (image, label, mask) in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
            # save
            if c.viz:
                image_list.extend(t2np(image))
            # ground_truth label list
            gt_label_list.extend(t2np(label))
            gt_mask_list.extend(t2np(mask))
            # encoder prediction
            image = image.to(c.device)  # single scale
            _ = encoder(image)  # BxCxHxW
            if 'saliency' in c.sub_arch:
                saliency_map = get_saliency_map(saliency_detector, image)
                saliency_image_list += saliency_map.detach().cpu().tolist()
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
                if 'saliency' in c.sub_arch:
                    saliency_resized = transforms.Resize([H, W])(saliency_map).unsqueeze(1)
                    condition_vector = torch.mul(condition_vector, saliency_resized)
                condition_vector_reshaped = condition_vector.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # BHWxP
                feature_map_reshaped = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC
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
                    #
                    if 'cflow' in c.dec_arch:
                        z, log_jac_det = decoder(feature_patch, [condition_patch,])
                    else:
                        z, log_jac_det = decoder(feature_patch)
                    #
                    decoder_log_prob = get_logp(C, z, log_jac_det)
                    log_prob = decoder_log_prob / C  # likelihood per dim
                    loss = -log_sigmoid(log_prob)
                    test_loss += t2np(loss.sum())
                    test_count += len(loss)
                    test_dist[l] = test_dist[l] + log_prob.detach().cpu().tolist()
    # Measuring performance
    fps = len(loader.dataset) / (time.time() - start)
    mean_test_loss = test_loss / test_count
    test_stat = {'mean_test_loss': mean_test_loss, 'fps': fps}
    if c.verbose:
        print(f"Test Epoch {epoch}: {test_stat}")
    #
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

    # score aggregation
    score_map = np.zeros_like(test_map[0]) # BxHxW
    for l, p in enumerate(pool_layers):
        score_map += test_map[l]
    score_mask = score_map
    # MASK | M
    # invert probs to anomaly scores
    # --> Lower likelihood --> Higher score --> Abnormal points
    # --> Max likelihood (or near max) --> Normal points
    super_mask = score_mask.max() - score_mask # scalar - BxHxW --> Shape (Bx3, H,W)
    saliency_added = super_mask
    
    abnormaly_score_dict = {'sum_prop_map': score_mask, 'ano_heat_map': super_mask} 
    # Train classification head from super_mask
    if class_head and should_train_class_head:
        # Remove the leaky abnormaly points
        saliency_added = super_mask * saliency_image_list
        train_class_head(c, class_head,saliency_added, gt_label_list, start_lr=0.001*(1 / (epoch + 1) ** 2))
    # else:
    #     train_class_head(c, class_head,saliency_added, gt_label_list, start_lr=0.001*(1 / (epoch + 1) ** 2))
    return image_list, gt_label_list, gt_mask_list, detection_loss, saliency_added, saliency_image_list, abnormaly_score_dict

def eval_batch(c, epoch, test_loader, encoder, decoders, pool_layers, N, saliency_detector, class_head, is_test_run=False, should_track_stats=True, pre_threshold=None, run_id=''):
    should_train_class_head = not is_test_run and epoch < c.class_head_stop_epoch
    test_image_list,gt_label_list, gt_mask_list, detection_score, super_mask, saliency_image_list, abnormaly_score_dict = test_meta_epoch(
        c, epoch, test_loader, encoder, decoders, pool_layers, N, saliency_detector=saliency_detector, class_head=class_head, should_train_class_head=should_train_class_head)
    accuracy, precision, recall, cf_matrix, seg_pro_auc, seg_roc_auc, det_roc_auc, f1, prec_rec_auc = (0,0,0,0,0,0,0, 0,0)

    # det_aur_roc
    # SEG_AUROC
    gt_mask = np.squeeze(np.asarray(gt_mask_list, dtype=bool), axis=1)
    if not c.no_mask: # If have mask in dataset, best weights if seg score is best
        seg_roc_auc = roc_auc_score(gt_mask.flatten(), super_mask.flatten())
        # calculate segmentation AUPRO
        if c.pro:  # and (epoch % 4 == 0):  # AUPRO is expensive to compute
            seg_pro_auc = calculate_seg_pro_auc(super_mask, gt_mask)

    # LABEL | Y
    gt_label = np.asarray(gt_label_list, dtype=bool)
    score_label = np.max(super_mask, axis=(1, 2)) # score_label (B,) <-- max([B, H, W], axis=(1,2))
    lr_precision, lr_recall, _ = precision_recall_curve(gt_label, score_label)
    if class_head:
        with torch.no_grad():
            outputs = class_head(super_mask)
            _, score_label = torch.max(outputs, 1)
    elif detection_score:
        score_label = get_anomaly_score(score_label, detection_score)
    # auc_roc
    if should_track_stats:
        det_roc_auc = roc_auc_score(gt_label, score_label)
    # calculate scores
    if class_head:
        f1, prec_rec_auc = f1_score(gt_label, score_label), auc(lr_recall, lr_precision)

    # BINARY
    if c.no_mask and c.action_type != 'norm-test' and should_track_stats:
        # precision | accuracy | recall
        if class_head:
            binary_score_label = score_label
        else:
            thresh_hold = pre_threshold
            # calculate precision and recall for each threshold
            if not thresh_hold:
                thresh_hold = find_best_thresh_hold_sig(gt_label, score_label)
            scaled_probs = score_sigmoid(score_label)
            binary_score_label = np.where(scaled_probs > thresh_hold, True, False)
        cf_matrix = confusion_matrix(gt_label, binary_score_label).ravel()
        accuracy = accuracy_score(gt_label, binary_score_label)
        precision = precision_score(gt_label, binary_score_label)
        recall = recall_score(gt_label, binary_score_label)
        update_stat_dict(accuracy, precision, recall, cf_matrix, seg_pro_auc, seg_roc_auc, det_roc_auc, prec_rec_auc, f1)

    # export visualuzations
    should_save_viz = is_test_run
    if c.viz and should_save_viz:
        save_visualization(c, test_image_list, super_mask, gt_mask, gt_label, score_label, saliency_list=saliency_image_list, run_id=run_id, score_mask_dict=abnormaly_score_dict)

def prepare_architecture(c):
    L = c.pool_layers # number of pooled layers
    print('Number of pool layers =', L)
    encoder, pool_layers, pool_dims = load_encoder_arch(c, L)
    encoder = encoder.to(c.device).eval()
    # NF decoder
    decoders = [load_decoder_arch(c, pool_dim) for pool_dim in pool_dims]
    decoders = [decoder.to(c.device) for decoder in decoders]

    # Test classification decoder
    saliency_detector = None
    class_head = None
    if 'class_head' in c.sub_arch:
        print("======Using classification head ========")
        # class_head = ClassificationHead(input_dims=c.img_size)
        class_head = Wide(input_dims=c.img_size)
    if 'saliency' in c.sub_arch:
        print("====== Using saliency detector========")
        saliency_detector = load_saliency_detector_arch(c)
    return encoder, pool_layers, decoders, saliency_detector, class_head

def train(c):
    run_date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    # Prepare model architecture
    encoder, pool_layers, decoders, saliency_detector, class_head = prepare_architecture(c)
    # optimizer
    params = list(decoders[0].parameters())
    L = c.pool_layers
    for l in range(1, L):
        params += list(decoders[l].parameters())
    optimizer = torch.optim.Adam(params, lr=c.lr)

    # data preparation
    train_loader, test_loader, val_loader = prepare_dataset(c)
    N = c.N if c.N else 256  # hyperparameter that increases batch size for the decoder model by N

    # stats
    det_roc_obs = Score_Observer(DET_AUC_ROC)
    prec_rec_auc_obs = Score_Observer(PREC_REC_AUC)
    f1_score_obs = Score_Observer(F1_SCORE)
    seg_roc_obs = Score_Observer(SEG_AUROC)
    seg_pro_obs = Score_Observer(SEG_AUPRO)
    accuracy_obs = Score_Observer(ACCURACY)
    precision_obs = Score_Observer(PRECISION)
    recall_obs = Score_Observer(RECALL)
    #
    meta_score = 0.0
    # How many unimproved epoch before stop training
    PATIENCE = c.patience
    early_stopping_patience = PATIENCE

    if c.action_type == 'norm-test':
        c.meta_epochs = 1
    for epoch in range(c.meta_epochs):
        print('Train meta epoch: {}'.format(epoch))
        train_meta_epoch(c, epoch,train_loader, saliency_detector,  encoder, decoders, optimizer, pool_layers, N)
        # Validation BATCH
        eval_batch(c, epoch, val_loader ,encoder, decoders, pool_layers, N, saliency_detector, class_head)
        # STATICTICS
        best_acc = accuracy_obs.update(STAT_DICT[ACCURACY], epoch)
        best_prec = precision_obs.update(STAT_DICT[PRECISION], epoch, False)
        best_rec = recall_obs.update(STAT_DICT[RECALL], epoch, False)
        # AUC
        if c.pro:
            seg_pro_obs.update(STAT_DICT[SEG_AUPRO], epoch, False)
        best_det_auc_roc = det_roc_obs.update(STAT_DICT[DET_AUC_ROC], epoch)
        best_seg_auc_roc = seg_roc_obs.update(STAT_DICT[SEG_AUROC], epoch, False if c.no_mask else True)
        best_prec_rec_auc = prec_rec_auc_obs.update(STAT_DICT[PREC_REC_AUC], epoch)
        best_f1_score = f1_score_obs.update(STAT_DICT[F1_SCORE], epoch)
        score = weight_precision_recall(STAT_DICT[PRECISION], STAT_DICT[RECALL])
        class_head_condition = ('class_head' in c.sub_arch) and (best_f1_score or best_det_auc_roc or best_prec_rec_auc or best_acc)
        normal_condition = ('class_head' not in c.sub_arch) and (score > meta_score) or best_acc
        if class_head_condition or normal_condition:
            print(f'Saving weight at stats: {STAT_DICT}')
            meta_score = score
            save_weights(c,encoder, decoders, c.model, run_date, class_head=class_head)
            early_stopping_patience = PATIENCE
        else:
            early_stopping_patience = early_stopping_patience - 1
            if early_stopping_patience == 0:
                print(f"NO IMPROVEMENT AFTER {PATIENCE} epochs. EARLY STOPPING NOW")
                break
    # Run on unseen test set
    eval_batch(c, 9999, test_loader ,encoder, decoders, pool_layers, N, saliency_detector, class_head, is_test_run=True)
    # save_results(det_roc_obs, seg_roc_obs, seg_pro_obs, c.model, c.class_name, run_date)
    save_model_metrics(c, [accuracy_obs, precision_obs, recall_obs, det_roc_obs, seg_roc_obs, f1_score_obs, prec_rec_auc_obs, accuracy_obs],c.model, c.class_name, run_date, confusion_dict=None,test_metrics=STAT_DICT)

def test(c, loader, run_id='EVAL'):
    if not c.checkpoint:
        raise ValueError("test run must have a checkpoint filepath provided")
    N = c.N if c.N else 256
    # Prepare model architecture
    encoder, pool_layers, decoders, saliency_detector, class_head = prepare_architecture(c)
    load_weights(c, encoder, decoders, class_head, c.checkpoint)
    eval_batch(c, 9999, loader ,encoder, decoders, pool_layers, N, saliency_detector, class_head, is_test_run=True,run_id=run_id, should_track_stats=False)

def test_one_shot(c, img_filenames, run_id="EVAL"):
    kwargs = {'num_workers': c.workers, 'pin_memory': True} if c.use_cuda else {}
    val_dataset = OneOffDataset(c,img_filenames=img_filenames)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size= 4, shuffle=False, drop_last=False, **kwargs)
    test(c, loader=val_loader, run_id=run_id)