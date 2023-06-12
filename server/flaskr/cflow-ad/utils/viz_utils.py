import os
import datetime
import numpy as np
import pandas as pd
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib
# from score_utils import *
from sklearn.metrics import precision_recall_curve, roc_curve
from utils import cloud_utils
import torch
import math
OUT_DIR = './viz/'

norm = matplotlib.colors.Normalize(vmin=0.0, vmax=255.0)
cm = 1/2.54
dpi = 300

def denormalization(x, norm_mean, norm_std):
    mean = np.array(norm_mean)
    std = np.array(norm_std)
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


def export_hist(c, gts, scores, threshold, out_dir=OUT_DIR):
    print('Exporting histogram...')
    plt.rcParams.update({'font.size': 4})
    image_dirs = os.path.join(out_dir, c.model)
    os.makedirs(image_dirs, exist_ok=True)
    Y = scores.flatten()
    Y_label = gts.flatten()
    fig = plt.figure(figsize=(4*cm, 4*cm), dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)
    plt.hist([Y[Y_label==1], Y[Y_label==0]], 500, density=True, color=['r', 'g'], label=['ANO', 'TYP'], alpha=0.75, histtype='barstacked')
    image_file = os.path.join(image_dirs, 'hist_images_' + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
    fig.savefig(image_file, dpi=dpi, format='svg', bbox_inches = 'tight', pad_inches = 0.0)
    plt.close()

def export_groundtruth(c, test_img, gts, out_dir=OUT_DIR):
    image_dirs = os.path.join(out_dir, c.model, 'gt_images_' + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
    # images
    if not os.path.isdir(image_dirs):
        print('Exporting grountruth...')
        os.makedirs(image_dirs, exist_ok=True)
        num = len(test_img)
        kernel = morphology.disk(4)
        for i in range(num):
            img = test_img[i]
            img = denormalization(img, c.norm_mean, c.norm_std)
            # gts
            gt_mask = gts[i].astype(np.float64)
            gt_mask = morphology.opening(gt_mask, kernel)
            gt_mask = (255.0*gt_mask).astype(np.uint8)
            gt_img = mark_boundaries(img, gt_mask, color=(1, 0, 0), mode='thick')
            #
            fig = plt.figure(figsize=(2*cm, 2*cm), dpi=dpi)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(gt_img)
            image_file = os.path.join(image_dirs, '{:08d}'.format(i))
            fig.savefig(image_file, dpi=dpi, format='svg', bbox_inches = 'tight', pad_inches = 0.0)
            plt.close()


EVAL_PREFIX = '$EVALUATION$'
def export_scores(c, test_img, scores, threshold, saliency_list=None ,out_dir=OUT_DIR, score_labels=None, run_id=''):
    # images
    if not os.path.isdir(out_dir):
        print('Exporting scores...')
        os.makedirs(out_dir, exist_ok=True)
        num = len(test_img)
        kernel = morphology.disk(4)
        scores_norm = 1.0/scores.max()
        for i in range(num):
            img = test_img[i]
            img = denormalization(img, c.norm_mean, c.norm_std)
            if saliency_list:
                saliency_mask = saliency_list[i]
                saliency_mask = np.multiply(255.0, saliency_mask)
            # scores
            score_mask = np.zeros_like(scores[i])
            score_mask[scores[i] >  threshold] = 1.0
            score_mask = morphology.opening(score_mask, kernel)
            score_mask = (255.0*score_mask).astype(np.uint8)
            score_img = mark_boundaries(img, score_mask, color=(1, 0, 0), mode='thick')
            score_map = (255.0*scores[i]*scores_norm).astype(np.uint8)
            label = ''
            #
            rows = 3 if saliency_list else 2
            fig_img, ax_img = plt.subplots(rows, 1, figsize=(2*cm, 4*cm))
            for ax_i in ax_img:
                ax_i.axes.xaxis.set_visible(False)
                ax_i.axes.yaxis.set_visible(False)
                ax_i.spines['top'].set_visible(False)
                ax_i.spines['right'].set_visible(False)
                ax_i.spines['bottom'].set_visible(False)
                ax_i.spines['left'].set_visible(False)
            #
            plt.subplots_adjust(hspace = 0.1, wspace = 0.1)
            # Plot the original and heatmap overlay
            ax_img[0].imshow(img, cmap='gray', interpolation='none')
            ax_img[0].imshow(score_map, cmap='jet', norm=norm, alpha=0.5, interpolation='none')
            # Plot Saliency Map Image And The Original
            if saliency_list:
                ax_img[1].imshow(saliency_mask)
                ax_img[2].imshow(score_img)
            else:
                ax_img[1].imshow(score_img)

            if score_labels is not None:
                scored_label = score_labels[i]
                label = 'anomaly' if scored_label else 'normal'
            image_file = os.path.join(out_dir, '{:08d}_{}'.format(i, label))
            fig_img.savefig(image_file, dpi=dpi, format='svg', bbox_inches = 'tight', pad_inches = 0.0)
            plt.close()


def export_test_images(c, test_images, gts, scores, threshold, out_dir = OUT_DIR):
    image_dirs = os.path.join(out_dir, c.model, 'images_' + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
    cm = 1/2.54
    # images
    if not os.path.isdir(image_dirs):
        print('Exporting images...')
        os.makedirs(image_dirs, exist_ok=True)
        num = len(test_images)
        font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 8}
        kernel = morphology.disk(4)
        scores_norm = 1.0/scores.max() # More abnormal --> Higher scores --> Inverted --> Lower score_norm (blue, green area)
        for i in range(num):
            img = test_images[i]
            img = denormalization(img, c.norm_mean, c.norm_std)
            # gts
            gt_mask = gts[i].astype(np.float64)
            # print('GT:', i, gt_mask.sum())
            gt_mask = morphology.opening(gt_mask, kernel)
            gt_mask = (255.0*gt_mask).astype(np.uint8)
            gt_img = mark_boundaries(img, gt_mask, color=(1, 0, 0), mode='thick')
            # scores
            score_mask = np.zeros_like(scores[i])
            score_mask[scores[i] >  threshold] = 1.0
            # print('SC:', i, score_mask.sum())
            score_mask = morphology.opening(score_mask, kernel)
            score_mask = (255.0*score_mask).astype(np.uint8)
            score_img = mark_boundaries(img, score_mask, color=(1, 0, 0), mode='thick')
            score_map = (255.0*scores[i]*scores_norm).astype(np.uint8)
            #
            fig_img, ax_img = plt.subplots(3, 1, figsize=(2*cm, 6*cm))
            for ax_i in ax_img:
                ax_i.axes.xaxis.set_visible(False)
                ax_i.axes.yaxis.set_visible(False)
                ax_i.spines['top'].set_visible(False)
                ax_i.spines['right'].set_visible(False)
                ax_i.spines['bottom'].set_visible(False)
                ax_i.spines['left'].set_visible(False)
            #
            plt.subplots_adjust(hspace = 0.1, wspace = 0.1)
            ax_img[0].imshow(gt_img)
            ax_img[1].imshow(score_map, cmap='jet', norm=norm)
            ax_img[2].imshow(score_img)
            image_file = os.path.join(image_dirs, '{:08d}.svg'.format(i))
            fig_img.savefig(image_file, dpi=dpi, format='svg', bbox_inches = 'tight', pad_inches = 0.0)
            plt.close()

def plot_hist_with_score_label(mean_data, labels, out_dir=''):
    plt.clf()
    data = {'mean_score':mean_data, 'label': labels}
    df = pd.DataFrame(data)
    normal = df.query("label==False")['mean_score']
    anomaly = df.query("label==True")['mean_score']
    # An "interface" to matplotlib.axes.Axes.hist() method
    n, bins, patches = plt.hist(x=normal, bins='auto', color='green',
                                alpha=0.7, rwidth=0.85)
    n, bins, patches = plt.hist(x=anomaly, bins='auto', color='red',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('score')
    plt.ylabel('Frequency')
    plt.title('My Very Own Histogram')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        file_path = os.path.join(out_dir,'hist.png')
        plt.savefig(file_path)

def plot_tpr_fpr(tpr,fpr, out_dir='', best_thresh=''):
    # plot the roc curve for the model
    plt.clf()
    plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label='model')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    # show the plot
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    file_path = os.path.join(out_dir,f'auc_roc_{best_thresh}.png')
    plt.savefig(file_path)
def save_visualization(c, test_image_list, super_masks, gt_masks, gt_labels, score_labels, saliency_list=None, run_id='', score_mask_dict=None):
    # precision, recall, thresholds = precision_recall_curve(gt_labels, score_labels)
    # a = 2 * precision * recall
    # b = precision + recall
    # f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    # det_threshold = thresholds[np.argmax(f1)]
    # print('Optimal DET Threshold: {:.2f}'.format(det_threshold))
    precision, recall, thresholds = precision_recall_curve(gt_masks.flatten(), super_masks.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    seg_threshold = thresholds[np.argmax(f1)]
    # Handle location
    out_dir = OUT_DIR
    if c.gcp:
        cloud_bucket_prefix = cloud_utils.get_bucket_prefix()
        out_dir = os.path.join(cloud_bucket_prefix,'viz')
    # print('Optimal SEG Threshold: {:.2f}'.format(seg_threshold))
    image_dirs = os.path.join(out_dir, c.model, 'scores_images_' + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
    # Handle one-off batch evaluation
    if run_id != '':
        image_dirs = os.path.join(out_dir, f'run_id:{run_id}@{c.model}', 'scores_images_' + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
    plot_dirs = os.path.join(out_dir, f'run_id:{run_id}@plots')
    if score_mask_dict:
        # Calcualte det_threshold
        sum_prop_map, abnomaly_heat_map = score_mask_dict['sum_prop_map'], score_mask_dict['ano_heat_map']
        mean_map = abnomaly_heat_map.mean((1,2))
        B, H, W = abnomaly_heat_map.shape
        flat_heat_map = abnomaly_heat_map.reshape(B, H * W) # BxHxW -> Bx(HW)
        values, indexes = torch.topk(torch.from_numpy(flat_heat_map), 30)
        max_k_sum = torch.sum(values, 1)
        abnomaly_score = mean_map + (4* max_k_sum.numpy())
        if 'TEST' in run_id:
            # abnomaly_score = sum_prop_map.mean((1,2))
            fpr, tpr, thresholds = roc_curve(gt_labels, abnomaly_score)
            # get the best threshold
            # calculate the g-mean for each threshold
            gmeans = np.sqrt(tpr * (1-fpr))
            # locate the index of the largest g-mean
            ix = np.argmax(gmeans)
            best_thresh = thresholds[ix]
            score_labels = np.where(abnomaly_score > best_thresh, True, False)
            plot_hist_with_score_label(abnomaly_score, gt_labels,plot_dirs)
            plot_tpr_fpr(tpr, fpr,plot_dirs, best_thresh=best_thresh)
            if c.verbose:
                print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
                print('Best DET Threshold=%f' % (best_thresh))
        if 'EVAL' in run_id:
            # recalculate the score_labels with provided threshold 
            if c.det_thresh:
                score_labels = np.where(abnomaly_score > c.det_thresh, True, False)
    export_scores(c, test_image_list, super_masks, seg_threshold, out_dir=image_dirs, saliency_list= saliency_list, score_labels=score_labels, run_id=run_id)