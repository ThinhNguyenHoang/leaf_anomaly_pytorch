import os, math
import numpy as np
import torch
import json

from utils.score_utils import Score_Observer



__all__ = ('save_results', 'save_weights', 'load_weights', 'adjust_learning_rate', 'warmup_learning_rate', 'save_model_metrics')

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


def save_model_metrics(c, metric_obs_list, model_name, class_name, run_date, confusion_dict=None, test_metrics=None):
    if not os.path.exists(c.result_dir):
        os.makedirs(c.result_dir)
    result = ''
    if confusion_dict:
        result += str(confusion_dict)
    if test_metrics:
        result += f"TEST_STATS: {test_metrics}"
    run_history_dict = {}
    for obs in metric_obs_list:
        if isinstance(obs, Score_Observer):
            result += f'{obs.name}: {obs.max_score} at epoch {obs.max_epoch}\n'
            run_history_dict[f'{obs.name}'] = obs.history
    # Dump this run metrics data
    result += f'\n=============HISTORY============\n'
    result += json.dumps(run_history_dict)
    fp = open(os.path.join(c.result_dir, f'{model_name}_{class_name}_{run_date}.txt'), "w")
    fp.write(result)
    fp.close()

def save_results(det_roc_obs, seg_roc_obs, seg_pro_obs, model_name, class_name, run_date):
    if not os.path.exists(c.result_dir):
        os.makedirs(c.result_dir)
    result = '{:.2f},{:.2f},{:.2f} \t\tfor {:s}/{:s}/{:s} at epoch {:d}/{:d}/{:d} for {:s}\n'.format(
        det_roc_obs.max_score, seg_roc_obs.max_score, seg_pro_obs.max_score,
        det_roc_obs.name, seg_roc_obs.name, seg_pro_obs.name,
        det_roc_obs.max_epoch, seg_roc_obs.max_epoch, seg_pro_obs.max_epoch, class_name)
    fp = open(os.path.join(c.result_dir, '{}_{}.txt'.format(model_name, run_date)), "w")
    fp.write(result)
    fp.close()


def save_weights(c, encoder, decoders, model_name, run_date, class_head=None):
    if not os.path.exists(c.weight_dir):
        os.makedirs(c.weight_dir)
    if class_head:
        state = {'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': [decoder.state_dict() for decoder in decoders],
                'class_head_state_dict':class_head.state_dict()}
    else:
        state = {'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': [decoder.state_dict() for decoder in decoders]}

    filename = '{}_{}.pt'.format(model_name, run_date)
    path = os.path.join(c.weight_dir, filename)
    torch.save(state, path)
    print('Saving weights to {}'.format(filename))

def parse_checkpoint_filename(filename):
    #
    date_idx = filename.find("date") # strip from _date...
    checkpoint_name = filename[:date_idx]
    used_args = checkpoint_name.split("&")
    args_dict = {}
    for args in used_args:
        if args:
            if args.split(":")[0] == "cv":
                k = "cv"
                cv_idx = args.find("cv")
                v = args[cv_idx + 3:]
            else:
                k,v = args.split(":")
        args_dict[k] = v
    return args_dict

def load_weights(c,encoder, decoders, class_head, checkpoint_filename):
    path = os.path.join(checkpoint_filename)
    state = torch.load(path)
    used_args = parse_checkpoint_filename(checkpoint_filename)
    sub_arch = used_args['sa'] if 'sa' in used_args else ''
    if ('class_head' in sub_arch):
        encoder.load_state_dict(state['class_head_state_dict'], strict=False)
        class_head.load_state_dict(class_head.state_dict())
    encoder.load_state_dict(state['encoder_state_dict'], strict=False)
    decoders = [decoder.load_state_dict(state, strict=False) for decoder, state in zip(decoders, state['decoder_state_dict'])]
    print('Loaded weights from {}'.format(checkpoint_filename))


def adjust_learning_rate(c, optimizer, epoch):
    lr = c.lr
    if c.lr_cosine:
        eta_min = lr * (c.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / c.meta_epochs)) / 2
    else:
        steps = np.sum(epoch >= np.asarray(c.lr_decay_epochs))
        if steps > 0:
            lr = lr * (c.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(c, epoch, batch_id, total_batches, optimizer):
    if c.lr_warm and epoch < c.lr_warm_epochs:
        p = (batch_id + epoch * total_batches) / \
            (c.lr_warm_epochs * total_batches)
        lr = c.lr_warmup_from + p * (c.lr_warmup_to - c.lr_warmup_from)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    #
    for param_group in optimizer.param_groups:
        lrate = param_group['lr']
    return lrate
