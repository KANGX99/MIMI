import torch
import numpy as np
#from torch.utils.data import DataLoader
import clip
import random
import scipy.stats
from utils import set_dataset, _preprocess2
import torch.nn.functional as F
from itertools import product
import os
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from tabulate import tabulate
from CLIP_MINN import CLIP_MINN

preprocess2 = _preprocess2()

dists = ['jpeg2000 compression', 'jpeg compression', 'white noise', 'gaussian blur', 'fastfading', 'fnoise', 'contrast', 'lens', 'motion', 'diffusion', 'shifting',
         'color quantization', 'oversaturation', 'desaturation', 'white with color', 'impulse', 'multiplicative',
         'white noise with denoise', 'brighten', 'darken', 'shifting the mean', 'jitter', 'noneccentricity patch',
         'pixelate', 'quantization', 'color blocking', 'sharpness', 'realistic blur', 'realistic noise',
         'underexposure', 'overexposure', 'realistic contrast change', 'other realistic']

scenes = ['animal', 'cityscape', 'human', 'indoor', 'landscape', 'night', 'plant', 'still_life', 'others']
qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']

dists_map = ['jpeg2000 compression', 'jpeg compression', 'noise', 'blur', 'color', 'contrast', 'overexposure',
            'underexposure', 'spatial', 'quantization', 'other']

live_set = '../databaserelease2/'
csiq_set = '../CSIQ/'
bid_set = '../BID/'
clive_set = '../ChallengeDB_release/'
koniq10k_set = '../koniq-10k/'
kadid10k_set = '../kadid10k/'

seed = 20200626

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

joint_texts = torch.cat([clip.tokenize(f"a photo of a {c} with {d} artifacts, which is of {q} quality") for q, c, d
                         in product(qualitys, scenes, dists_map)]).to(device)

def do_batch(x, text):
    batch_size = x.size(0)
    num_patch = x.size(1)
    x = x.view(-1, x.size(2), x.size(3), x.size(4))
    logits_per_image, _ = model.clip_model.forward(x, text)
    logits_per_image = logits_per_image.view(batch_size, num_patch, -1)
    logits_per_image = logits_per_image.mean(1)
    logits_per_image = F.softmax(logits_per_image, dim=1)
    return logits_per_image


def eval(loader, phase, dataset):
    model.eval()
    q_mos = []
    q_hat = []
    for step, sample_batched in enumerate(loader, 0):

        x, gmos, _, _, _, _, _ = sample_batched['I'], sample_batched['mos'], sample_batched[
            'dist_type'], sample_batched['scene_content1'], sample_batched['scene_content2'], \
                                                       sample_batched['scene_content3'], sample_batched['valid']

        x = x.to(device)
        q_mos = q_mos + gmos.cpu().tolist()

        with torch.no_grad():
            logits_per_image = do_batch(x, joint_texts)
            logits_per_image = logits_per_image.view(-1, len(qualitys), len(scenes), len(dists_map))
            logits_quality = logits_per_image.sum(3).sum(2)
            logits_quality = 1 * logits_quality[:, 0] + 2 * logits_quality[:, 1] + 3 * logits_quality[:, 2] + \
                            4 * logits_quality[:, 3] + 5 * logits_quality[:, 4]

            _, _, logits_quality = model(logits_quality, isTest = True, nn_id = dataset_map[dataset])

        q_hat = q_hat + logits_quality.cpu().tolist()

    print_text = dataset + ' ' + phase + ' finished'
    print(print_text)
    return q_mos, q_hat


def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    # 4-parameter logistic function
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat


def compute_metrics(y_pred, y):
    '''
    compute metrics btw predictions & labels
    '''
    # compute SRCC & KRCC
    SRCC = scipy.stats.spearmanr(y, y_pred)[0]
    try:
        KRCC = scipy.stats.kendalltau(y, y_pred)[0]
    except:
        KRCC = scipy.stats.kendalltau(y, y_pred, method='asymptotic')[0]

    # logistic regression btw y_pred & y
    beta_init = [np.max(y), np.min(y), np.mean(y_pred), np.std(y_pred)]
    popt, _ = curve_fit(logistic_func, y_pred, y, p0=beta_init, maxfev=int(1e8))
    y_pred_logistic = logistic_func(y_pred, *popt)

    # compute PLCC RMSE
    PLCC = scipy.stats.pearsonr(y, y_pred_logistic)[0]
    RMSE = np.sqrt(mean_squared_error(y, y_pred_logistic))
    return SRCC, KRCC, PLCC, RMSE

num_workers = 8
all_srcc = {'live': [], 'csiq': [], 'bid': [], 'clive': [], 'koniq10k': [], 'kadid10k': []}
all_plcc = {'live': [], 'csiq': [], 'bid': [], 'clive': [], 'koniq10k': [], 'kadid10k': []}
dataset_map = {'live':1, 'csiq':2, 'bid':3, 'clive':4, 'koniq10k':5, 'kadid10k':6}


for session in range(0,10):
    print('session {}'.format(session+1))
    model = CLIP_MINN(device=device, block_num=1)
    model.to(device)
    ckpt = os.path.join('checkpoints', str(session + 1), 'quality_best_ckpt.pt')
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint['model_state_dict'])

    live_test_csv = os.path.join('./IQA_Database/databaserelease2/splits2', str(session+1), 'live_test_clip.txt')
    csiq_test_csv = os.path.join('./IQA_Database/CSIQ/splits2', str(session+1), 'csiq_test_clip.txt')
    bid_test_csv = os.path.join('./IQA_Database/BID/splits2', str(session+1), 'bid_test_clip.txt')
    clive_test_csv = os.path.join('./IQA_Database/ChallengeDB_release/splits2', str(session+1), 'clive_test_clip.txt')
    koniq10k_test_csv = os.path.join('./IQA_Database/koniq-10k/splits2', str(session+1), 'koniq10k_test_clip.txt')
    kadid10k_test_csv = os.path.join('./IQA_Database/kadid10k/splits2', str(session+1), 'kadid10k_test_clip.txt')

    live_test_loader = set_dataset(live_test_csv, 16, live_set, num_workers, preprocess2, 15, True)
    csiq_test_loader = set_dataset(csiq_test_csv, 16, csiq_set, num_workers, preprocess2, 15, True)
    bid_test_loader = set_dataset(bid_test_csv, 16, bid_set, num_workers, preprocess2, 15, True)
    clive_test_loader = set_dataset(clive_test_csv, 16, clive_set, num_workers, preprocess2, 15, True)
    koniq10k_test_loader = set_dataset(koniq10k_test_csv, 16, koniq10k_set, num_workers, preprocess2, 15, True)
    kadid10k_test_loader = set_dataset(kadid10k_test_csv, 16, kadid10k_set, num_workers, preprocess2, 15, True)

    q_mos1, q_hat1 = eval(live_test_loader, 'test', 'live')
    q_mos2, q_hat2 = eval(csiq_test_loader, 'test', 'csiq')
    q_mos3, q_hat3 = eval(bid_test_loader, 'test', 'bid')
    q_mos4, q_hat4 = eval(clive_test_loader, 'test', 'clive')
    q_mos5, q_hat5 = eval(koniq10k_test_loader, 'test', 'koniq10k')
    q_mos6, q_hat6 = eval(kadid10k_test_loader, 'test', 'kadid10k')

    srcc1, _, plcc1, _ = compute_metrics(q_hat1, q_mos1)
    srcc2, _, plcc2, _ = compute_metrics(q_hat2, q_mos2)
    srcc3, _, plcc3, _ = compute_metrics(q_hat3, q_mos3)
    srcc4, _, plcc4, _ = compute_metrics(q_hat4, q_mos4)
    srcc5, _, plcc5, _ = compute_metrics(q_hat5, q_mos5)
    srcc6, _, plcc6, _ = compute_metrics(q_hat6, q_mos6)

    all_srcc['live'].append(srcc1)
    all_srcc['csiq'].append(srcc2)
    all_srcc['bid'].append(srcc3)
    all_srcc['clive'].append(srcc4)
    all_srcc['koniq10k'].append(srcc5)
    all_srcc['kadid10k'].append(srcc6)

    all_plcc['live'].append(plcc1)
    all_plcc['csiq'].append(plcc2)
    all_plcc['bid'].append(plcc3)
    all_plcc['clive'].append(plcc4)
    all_plcc['koniq10k'].append(plcc5)
    all_plcc['kadid10k'].append(plcc6)


def final_median(all_srcc, all_plcc):
    median_srcc = np.median(np.array(all_srcc))
    median_plcc = np.median(np.array(all_plcc))

    return [median_srcc, median_plcc]


def calculate_weighted_metrics(live_srcc, csiq_srcc, bid_srcc, clive_srcc, koniq10k_srcc, kadid10k_srcc,
                                live_plcc, csiq_plcc, bid_plcc, clive_plcc, koniq10k_plcc, kadid10k_plcc):
  
  weighted_srcc = live_srcc * 779 + csiq_srcc * 866 + kadid10k_srcc * 10125 + bid_srcc * 586 + clive_srcc * 1162 + koniq10k_srcc * 10073
  weighted_plcc = live_plcc * 779 + csiq_plcc * 866 + kadid10k_plcc * 10125 + bid_plcc * 586 + clive_plcc * 1162 + koniq10k_plcc * 10073

  total_weight = 779 + 866 + 10125 + 586 + 1162 + 10073
  weighted_srcc = weighted_srcc / total_weight
  weighted_plcc = weighted_plcc / total_weight

  return weighted_srcc, weighted_plcc


live_results = final_median(all_srcc['live'], all_plcc['live'])
csiq_results = final_median(all_srcc['csiq'], all_plcc['csiq'])
bid_results = final_median(all_srcc['bid'], all_plcc['bid'])
clive_results = final_median(all_srcc['clive'], all_plcc['clive'])
koniq10k_results = final_median(all_srcc['koniq10k'], all_plcc['koniq10k'])
kadid10k_results = final_median(all_srcc['kadid10k'], all_plcc['kadid10k'])

w_srcc, w_plcc = calculate_weighted_metrics(live_results[0], csiq_results[0], bid_results[0], clive_results[0],
                           koniq10k_results[0], kadid10k_results[0],
                           live_results[1], csiq_results[1], bid_results[1], clive_results[1],
                           koniq10k_results[1], kadid10k_results[1])
data = [
    ["live", live_results[0], live_results[1]],
    ["csiq", csiq_results[0], csiq_results[1]],
    ["bid", bid_results[0], bid_results[1]],
    ["clive", clive_results[0], clive_results[1]],
    ["koniq10k", koniq10k_results[0], koniq10k_results[1]],
    ["kadid10k", kadid10k_results[0], kadid10k_results[1]],
    ["Weight", w_srcc, w_plcc],
]
headers = ["Dataset", "SRCC", "PLCC"]
print(tabulate(data, headers=headers, tablefmt="grid"))
