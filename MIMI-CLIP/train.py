import torch
import torch.nn as nn
import numpy as np
#from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import clip
import random
import time
from MNL_Loss import Fidelity_Loss, loss_m4, Multi_Fidelity_Loss, Fidelity_Loss_distortion
import scipy.stats
from utils import set_dataset, _preprocess2, _preprocess3, convert_models_to_fp32
import torch.nn.functional as F
from itertools import product
import os
import pickle
from weight_methods import WeightMethods
from CLIP_MINN import CLIP_MINN

##############################textual template####################################
dists = ['jpeg2000 compression', 'jpeg compression', 'white noise', 'gaussian blur', 'fastfading', 'fnoise', 'contrast', 'lens', 'motion', 'diffusion', 'shifting',
         'color quantization', 'oversaturation', 'desaturation', 'white with color', 'impulse', 'multiplicative',
         'white noise with denoise', 'brighten', 'darken', 'shifting the mean', 'jitter', 'noneccentricity patch',
         'pixelate', 'quantization', 'color blocking', 'sharpness', 'realistic blur', 'realistic noise',
         'underexposure', 'overexposure', 'realistic contrast change', 'other realistic']

scenes = ['animal', 'cityscape', 'human', 'indoor', 'landscape', 'night', 'plant', 'still_life', 'others']
qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']

type2label = {'jpeg2000 compression':0, 'jpeg compression':1, 'white noise':2, 'gaussian blur':3, 'fastfading':4, 'fnoise':5, 'contrast':6, 'lens':7, 'motion':8,
              'diffusion':9, 'shifting':10, 'color quantization':11, 'oversaturation':12, 'desaturation':13,
              'white with color':14, 'impulse':15, 'multiplicative':16, 'white noise with denoise':17, 'brighten':18,
              'darken':19, 'shifting the mean':20, 'jitter':21, 'noneccentricity patch':22, 'pixelate':23,
              'quantization':24, 'color blocking':25, 'sharpness':26, 'realistic blur':27, 'realistic noise':28,
              'underexposure':29, 'overexposure':30, 'realistic contrast change':31, 'other realistic':32}

dist_map = {'jpeg2000 compression':'jpeg2000 compression', 'jpeg compression':'jpeg compression',
                   'white noise':'noise', 'gaussian blur':'blur', 'fastfading': 'jpeg2000 compression', 'fnoise':'noise',
                   'contrast':'contrast', 'lens':'blur', 'motion':'blur', 'diffusion':'color', 'shifting':'blur',
                   'color quantization':'quantization', 'oversaturation':'color', 'desaturation':'color',
                   'white with color':'noise', 'impulse':'noise', 'multiplicative':'noise',
                   'white noise with denoise':'noise', 'brighten':'overexposure', 'darken':'underexposure', 'shifting the mean':'other',
                   'jitter':'spatial', 'noneccentricity patch':'spatial', 'pixelate':'spatial', 'quantization':'quantization',
                   'color blocking':'spatial', 'sharpness':'contrast', 'realistic blur':'blur', 'realistic noise':'noise',
                   'underexposure':'underexposure', 'overexposure':'overexposure', 'realistic contrast change':'contrast', 'other realistic':'other'}

map2label = {'jpeg2000 compression':0, 'jpeg compression':1, 'noise':2, 'blur':3, 'color':4,
             'contrast':5, 'overexposure':6, 'underexposure':7, 'spatial':8, 'quantization':9, 'other':10}

dists_map = ['jpeg2000 compression', 'jpeg compression', 'noise', 'blur', 'color', 'contrast', 'overexposure',
            'underexposure', 'spatial', 'quantization', 'other']

scene2label = {'animal':0, 'cityscape':1, 'human':2, 'indoor':3, 'landscape':4, 'night':5, 'plant':6, 'still_life':7,
               'others':8}
##############################textual template####################################

##############################general setup####################################
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

initial_lr = 5e-6
num_epoch = 80
bs = 32

train_patch = 3

loss_img2 = Fidelity_Loss_distortion()
loss_scene = Multi_Fidelity_Loss()

scene_texts = torch.cat([clip.tokenize(f"a photo of a {c}") for c in scenes]).to(device)
dist_texts = torch.cat([clip.tokenize(f"a photo with {c} artifacts") for c in dists]).to(device)
distmap_texts = torch.cat([clip.tokenize(f"a photo with {c} artifacts") for c in dists_map]).to(device)
quality_texts = torch.cat([clip.tokenize(f"a photo with {c} quality") for c in qualitys]).to(device) 

joint_texts = torch.cat([clip.tokenize(f"a photo of a {c} with {d} artifacts, which is of {q} quality") for q, c, d
                         in product(qualitys, scenes, dists_map)]).to(device) 
##############################general setup####################################

preprocess2 = _preprocess2()
preprocess3 = _preprocess3()

def do_batch(x, text):
    batch_size = x.size(0)
    num_patch = x.size(1)
    x = x.view(-1, x.size(2), x.size(3), x.size(4))
    logits_per_image, _ = model.clip_model.forward(x, text)
    logits_per_image = logits_per_image.view(batch_size, num_patch, -1)
    logits_per_image = logits_per_image.mean(1) 
    logits_per_image = F.softmax(logits_per_image, dim=1)

    return logits_per_image


def train(model):
    start_time = time.time()
    beta = 0.9
    running_loss = 0 if epoch == 0 else train_loss[-1]
    running_duration = 0.0
    num_steps_per_epoch = 200
    local_counter = epoch * num_steps_per_epoch + 1
    model.train()
    loaders = []
    for loader in train_loaders:
        loaders.append(iter(loader))

    print(optimizer.state_dict()['param_groups'][0]['lr'])
    if optimizer.state_dict()['param_groups'][0]['lr'] == 0:
        scheduler.step()
        print(optimizer.state_dict()['param_groups'][0]['lr'])
    for step in range(num_steps_per_epoch):
        all_batch = []
        scene_gt_batch = []
        dist_gt_batch = []
        gmos_batch = []
        samples_per_ds = []

        for dataset_idx, loader in enumerate(loaders, 0):
            try:
                sample_batched = next(loader)
            except StopIteration: 
                loader = iter(train_loaders[dataset_idx])
                sample_batched = next(loader)
                loaders[dataset_idx] = loader

            x, gmos, dist, scene1, scene2, scene3, valid = sample_batched['I'], sample_batched['mos'], sample_batched[
                'dist_type'], sample_batched['scene_content1'], sample_batched['scene_content2'], \
                                                    sample_batched['scene_content3'], sample_batched['valid']
            x = x.to(device)
            gmos = gmos.to(device)
            gmos_batch.append(gmos)
            samples_per_ds.append(x.size(0))

            scene_gt = np.zeros((len(scene1), len(scenes)), dtype=float)
            dist_gt = np.zeros((len(dist), len(dists_map)), dtype=float)

            for i in range(len(scene1)):
                if valid[i] == 1:
                    scene_gt[i, scene2label[scene1[i]]] = 1.0
                elif valid[i] == 2:
                    scene_gt[i, scene2label[scene1[i]]] = 1.0
                    scene_gt[i, scene2label[scene2[i]]] = 1.0
                elif valid[i] == 3:
                    scene_gt[i, scene2label[scene1[i]]] = 1.0
                    scene_gt[i, scene2label[scene2[i]]] = 1.0
                    scene_gt[i, scene2label[scene3[i]]] = 1.0
                dist_gt[i, map2label[dist_map[dist[i]]]] = 1.0
            scene_gt = torch.from_numpy(scene_gt).to(device)
            dist_gt = torch.from_numpy(dist_gt).to(device)

            all_batch.append(x)
            scene_gt_batch.append(scene_gt)
            dist_gt_batch.append(dist_gt)

        all_batch = torch.cat(all_batch, dim=0)
        scene_gt_batch = torch.cat(scene_gt_batch, dim=0)
        dist_gt_batch = torch.cat(dist_gt_batch, dim=0)

        optimizer.zero_grad()
        logits_per_image = do_batch(all_batch, joint_texts)
        logits_per_image = logits_per_image.view(-1, len(qualitys), len(scenes), len(dists_map)) 

        logits_quality = logits_per_image.sum(3).sum(2)
        logits_scene = logits_per_image.sum(3).sum(1)
        logits_distortion = logits_per_image.sum(1).sum(1)

        logits_quality = 1 * logits_quality[:, 0] + 2 * logits_quality[:, 1] + 3 * logits_quality[:, 2] + \
                            4 * logits_quality[:, 3] + 5 * logits_quality[:, 4]
        
        y1_batch, y2_batch, _, multi_range_min, multi_range_max = model(logits_quality, isTest = False, samples_per_ds = samples_per_ds)
        y2_batch_s = torch.split(y2_batch, samples_per_ds)

        loss_1 = loss_fn(y2_batch_s[0], gmos_batch[0])
        loss_2 = loss_fn(y2_batch_s[1], gmos_batch[1])
        loss_3 = loss_fn(y2_batch_s[2], gmos_batch[2])
        loss_4 = loss_fn(y2_batch_s[3], gmos_batch[3])
        loss_5 = loss_fn(y2_batch_s[4], gmos_batch[4])
        loss_6 = loss_fn(y2_batch_s[5], gmos_batch[5])
        loss_7 = loss_fn(y1_batch,y2_batch)

        MIN, MAX  = torch.min(multi_range_min), torch.max(multi_range_max)
        loss_8 = torch.zeros(1).to(device)
        for k in range(len(multi_range_min)):
            loss_8 += torch.abs(multi_range_min[k].squeeze()-MIN)+ torch.abs(multi_range_max[k].squeeze()-MAX)

        quality_loss = ((loss_1+loss_2+loss_3+loss_4+loss_5+loss_6) / 6 + 2*loss_7 + loss_8 / 6).mean()
        dist_loss = loss_img2(logits_distortion, dist_gt_batch.detach()).mean()
        scene_loss = loss_scene(logits_scene, scene_gt_batch.detach()).mean()
        
        total_loss =  quality_loss + dist_loss + scene_loss
        all_loss = [quality_loss, dist_loss, scene_loss]

        shared_parameters = None
        last_shared_layer = None
        if not torch.isnan(total_loss):
            # weight losses and backward
            total_loss = weighting_method.backwards(
                all_loss,
                epoch=epoch,
                logsigmas=None,
                shared_parameters=shared_parameters,
                last_shared_params=last_shared_layer,
                returns=True
            )
        else:
            total_loss.backward()
            continue

        if device == "cpu":
            optimizer.step()
        else:
            convert_models_to_fp32(model.clip_model)
            optimizer.step()
            clip.model.convert_weights(model.clip_model)

        # statistics
        running_loss = beta * running_loss + (1 - beta) * total_loss.data.item()
        loss_corrected = running_loss / (1 - beta ** local_counter)

        current_time = time.time()
        duration = current_time - start_time
        running_duration = beta * running_duration + (1 - beta) * duration
        duration_corrected = running_duration / (1 - beta ** local_counter)

        format_str = ('(E:%d, S:%d / %d) [Loss = %.4f] (%.3f sec/batch) '
                      'Quality Loss: %.4f (avg(loss_1-6): %.4f, loss_7: %.4f, loss_8/6: %.4f) '
                      'Distortion Loss: %.4f Scene Loss: %.4f')
        print(format_str % (epoch, step + 1, num_steps_per_epoch, loss_corrected,
                            duration_corrected, 
                            quality_loss.item(), 
                            ((loss_1+loss_2+loss_3+loss_4+loss_5+loss_6) / 6).mean().item(),
                            loss_7.mean().item(), 
                            (loss_8 / 6).mean().item(),
                            dist_loss.item(), 
                            scene_loss.item()))

        local_counter += 1
        start_time = time.time()

        train_loss.append(loss_corrected)
    quality_result_val = {}
    scene_result_val = {}
    distortion_result_val = {}
    all_result_val = {}
    if (epoch >= 0):
        scene_acc1, dist_acc1, srcc1 = eval(live_val_loader, phase='val', dataset='live')
        scene_acc2, dist_acc2, srcc2 = eval(csiq_val_loader, phase='val', dataset='csiq')
        scene_acc3, dist_acc3, srcc3 = eval(bid_val_loader, phase='val', dataset='bid')
        scene_acc4, dist_acc4, srcc4 = eval(clive_val_loader, phase='val', dataset='clive')
        scene_acc5, dist_acc5, srcc5 = eval(koniq10k_val_loader, phase='val', dataset='koniq10k')
        scene_acc6, dist_acc6, srcc6 = eval(kadid10k_val_loader, phase='val', dataset='kadid10k')

        quality_result_val = {'live':srcc1, 'csiq':srcc2, 'bid':srcc3, 'clive':srcc4, 'koniq10k':srcc5,
                                 'kadid10k':srcc6}
        scene_result_val = {'live':scene_acc1, 'csiq':scene_acc2, 'bid':scene_acc3, 'clive':scene_acc4, 'koniq10k':scene_acc5,
                                 'kadid10k':scene_acc6}
        distortion_result_val = {'live':dist_acc1, 'csiq':dist_acc2, 'bid':dist_acc3, 'clive':dist_acc4, 'koniq10k':dist_acc5,
                                 'kadid10k':dist_acc6}
        all_result_val = {'quality':quality_result_val, 'scene':scene_result_val,
                             'distortion':distortion_result_val}

        srcc_avg = (srcc1 + srcc2 + srcc3 + srcc4 + srcc5 + srcc6) / 6
        scene_avg = (scene_acc1 + scene_acc2 + scene_acc3 + scene_acc4 + scene_acc5 + scene_acc6) / 6
        dist_avg = (dist_acc1 + dist_acc2 + dist_acc3 + dist_acc4 + dist_acc5 + dist_acc6) / 6

        if srcc_avg > best_result['quality']:
            print('**********New quality best!**********')
            best_epoch['quality'] = epoch
            best_result['quality'] = srcc_avg
            srcc_dict['live'] = srcc1
            srcc_dict['csiq'] = srcc2
            srcc_dict['bid'] = srcc3
            srcc_dict['clive'] = srcc4
            srcc_dict['koniq10k'] = srcc5
            srcc_dict['kadid10k'] = srcc6

            if not os.path.exists(os.path.join('checkpoints', str(session+1))):
                os.makedirs(os.path.join('checkpoints', str(session+1)))
            ckpt_name = os.path.join('checkpoints', str(session + 1), 'quality_best_ckpt.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'all_results': all_result_val
            }, ckpt_name)  # just change to your preferred folder/filename

        if dist_avg > best_result['distortion']:
            print('**********New distortion best!**********')
            best_epoch['distortion'] = epoch
            best_result['distortion'] = dist_avg

        if scene_avg > best_result['scene']:
            print('**********New scene best!**********')
            best_epoch['scene'] = epoch
            best_result['scene'] = scene_avg

    return all_result_val


def eval(loader, phase, dataset):
    model.eval()
    correct_scene = 0.0
    correct_dist = 0.0
    q_mos = []
    q_hat = []
    num_scene = 0
    num_dist = 0
    for step, sample_batched in enumerate(loader, 0):

        x, gmos, dist, scene1, scene2, scene3, valid = sample_batched['I'], sample_batched['mos'], sample_batched[
            'dist_type'], sample_batched['scene_content1'], sample_batched['scene_content2'], \
                                                       sample_batched['scene_content3'], sample_batched['valid']

        x = x.to(device)
        q_mos = q_mos + gmos.cpu().tolist()

        with torch.no_grad():
            logits_per_image = do_batch(x, joint_texts)

            logits_per_image = logits_per_image.view(-1, len(qualitys), len(scenes), len(dists_map))

            logits_quality = logits_per_image.sum(3).sum(2)
            similarity_scene = logits_per_image.sum(3).sum(1)
            similarity_distortion = logits_per_image.sum(1).sum(1)

            logits_quality = 1 * logits_quality[:, 0] + 2 * logits_quality[:, 1] + 3 * logits_quality[:, 2] + \
                            4 * logits_quality[:, 3] + 5 * logits_quality[:, 4]
            
            _, y2_batch, _ = model(logits_quality, isTest = True, nn_id = dataset_map[dataset])

        q_hat = q_hat + y2_batch.cpu().tolist()

        indice2 = similarity_distortion.argmax(dim=1)

        for i in range(len(dist)):
            if dist_map[dist[i]] == dists_map[indice2[i]]:
                correct_dist += 1
            num_dist += 1

        for i in range(len(valid)):
            if valid[i] == 1:
                indice = similarity_scene.argmax(dim=1)
                # indice = indice.squeeze()
                if scene1[i] == scenes[indice[i]]:
                    correct_scene += 1
                num_scene += 1
            elif valid[i] == 2:
                _, indices = similarity_scene.topk(k=2, dim=1)
                # indices = indices.squeeze()
                if (scene1[i] == scenes[indices[i, 0]]) | (scene1[i] == scenes[indices[i, 1]]):
                    correct_scene += 1
                if (scene2[i] == scenes[indices[i, 0]]) | (scene2[i] == scenes[indices[i, 1]]):
                    correct_scene += 1
                num_scene += 2
            elif valid[i] == 3:
                _, indices = similarity_scene.topk(k=3, dim=1)
                indices = indices.squeeze()
                if (scene1[i] == scenes[indices[i, 0]]) | (scene1[i] == scenes[indices[i, 1]]) | (
                        scene1[i] == scenes[indices[i, 2]]):
                    correct_scene += 1
                if (scene2[i] == scenes[indices[i, 0]]) | (scene2[i] == scenes[indices[i, 1]]) | (
                        scene2[i] == scenes[indices[i, 2]]):
                    correct_scene += 1
                if (scene3[i] == scenes[indices[i, 0]]) | (scene3[i] == scenes[indices[i, 1]]) | (
                        scene3[i] == scenes[indices[i, 2]]):
                    correct_scene += 1
                num_scene += 3

    scene_acc = correct_scene / num_scene
    dist_acc = correct_dist / num_dist
    srcc = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]

    print_text = dataset + ' ' + phase + ' finished'
    print(print_text)

    return scene_acc, dist_acc, srcc

num_workers = 8
dataset_map = {'live':1, 'csiq':2, 'bid':3, 'clive':4, 'koniq10k':5, 'kadid10k':6}
for session in range(0,10):

    weighting_method = WeightMethods(
        method='dwa',
        n_tasks=3,
        alpha=1.5,
        temp=2.0,
        n_train_batch=200,
        n_epochs=num_epoch,
        main_task=0,
        device=device
    )

    model = CLIP_MINN(device=device, block_num=1)
    model.to(device)
    loss_fn = nn.SmoothL1Loss()
    loss_fn.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=initial_lr,
        weight_decay=0.001)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    train_loss = []
    start_epoch = 0

    model.clip_model.logit_scale.requires_grad = False

    best_result = {'avg': 0.0, 'quality': 0.0, 'scene': 0.0, 'distortion': 0.0}
    best_epoch = {'avg': 0, 'quality': 0, 'scene': 0, 'distortion': 0}

    srcc_dict = {'live': 0.0, 'csiq': 0.0, 'bid': 0.0, 'clive': 0.0, 'koniq10k': 0.0, 'kadid10k': 0.0}

    live_train_csv = os.path.join('./IQA_Database/databaserelease2/splits2', str(session+1), 'live_train_clip.txt')
    live_val_csv = os.path.join('./IQA_Database/databaserelease2/splits2', str(session+1), 'live_val_clip.txt')

    csiq_train_csv = os.path.join('./IQA_Database/CSIQ/splits2', str(session+1), 'csiq_train_clip.txt')
    csiq_val_csv = os.path.join('./IQA_Database/CSIQ/splits2', str(session+1), 'csiq_val_clip.txt')

    bid_train_csv = os.path.join('./IQA_Database/BID/splits2', str(session+1), 'bid_train_clip.txt')
    bid_val_csv = os.path.join('./IQA_Database/BID/splits2', str(session+1), 'bid_val_clip.txt')

    clive_train_csv = os.path.join('./IQA_Database/ChallengeDB_release/splits2', str(session+1), 'clive_train_clip.txt')
    clive_val_csv = os.path.join('./IQA_Database/ChallengeDB_release/splits2', str(session+1), 'clive_val_clip.txt')

    koniq10k_train_csv = os.path.join('./IQA_Database/koniq-10k/splits2', str(session+1), 'koniq10k_train_clip.txt')
    koniq10k_val_csv = os.path.join('./IQA_Database/koniq-10k/splits2', str(session+1), 'koniq10k_val_clip.txt')

    kadid10k_train_csv = os.path.join('./IQA_Database/kadid10k/splits2', str(session+1), 'kadid10k_train_clip.txt')
    kadid10k_val_csv = os.path.join('./IQA_Database/kadid10k/splits2', str(session+1), 'kadid10k_val_clip.txt')

    live_train_loader = set_dataset(live_train_csv, 4, live_set, num_workers, preprocess3, train_patch, False)
    live_val_loader = set_dataset(live_val_csv, 16, live_set, num_workers, preprocess2, 15, True)
    live_max_mos = (114.4147 - live_train_loader.dataset.min_mos)*10 / (live_train_loader.dataset.max_mos - live_train_loader.dataset.min_mos)
    live_min_mos = (0 - live_train_loader.dataset.min_mos)*10 / (live_train_loader.dataset.max_mos - live_train_loader.dataset.min_mos)

    csiq_train_loader = set_dataset(csiq_train_csv, 4, csiq_set, num_workers, preprocess3, train_patch, False)
    csiq_val_loader = set_dataset(csiq_val_csv, 16, csiq_set, num_workers, preprocess2, 15, True)
    csiq_max_mos = (1 - csiq_train_loader.dataset.min_mos)*10 / (csiq_train_loader.dataset.max_mos - csiq_train_loader.dataset.min_mos)
    csiq_min_mos = (0 - csiq_train_loader.dataset.min_mos)*10 / (csiq_train_loader.dataset.max_mos - csiq_train_loader.dataset.min_mos)

    bid_train_loader = set_dataset(bid_train_csv, 4, bid_set, num_workers, preprocess3, train_patch, False)
    bid_val_loader = set_dataset(bid_val_csv, 16, bid_set, num_workers, preprocess2, 15, True)
    bid_max_mos = (4.9198 - bid_train_loader.dataset.min_mos)*10 / (bid_train_loader.dataset.max_mos - bid_train_loader.dataset.min_mos)
    bid_min_mos = (0.1638  - bid_train_loader.dataset.min_mos)*10 / (bid_train_loader.dataset.max_mos - bid_train_loader.dataset.min_mos)

    clive_train_loader = set_dataset(clive_train_csv, 4, clive_set, num_workers, preprocess3, train_patch, False)
    clive_val_loader = set_dataset(clive_val_csv, 16, clive_set, num_workers, preprocess2, 15, True)
    clive_max_mos = (92.4320 - clive_train_loader.dataset.min_mos)*10 / (clive_train_loader.dataset.max_mos - clive_train_loader.dataset.min_mos)
    clive_min_mos = (3.4200  - clive_train_loader.dataset.min_mos)*10 / (clive_train_loader.dataset.max_mos - clive_train_loader.dataset.min_mos)

    koniq10k_train_loader = set_dataset(koniq10k_train_csv, 16, koniq10k_set, num_workers, preprocess3, train_patch, False)
    koniq10k_val_loader = set_dataset(koniq10k_val_csv, 16, koniq10k_set, num_workers, preprocess2, 15, True)
    koniq10k_max_mos = (4.3100 - koniq10k_train_loader.dataset.min_mos)*10 / (koniq10k_train_loader.dataset.max_mos - koniq10k_train_loader.dataset.min_mos)
    koniq10k_min_mos = (1.0962  - koniq10k_train_loader.dataset.min_mos)*10 / (koniq10k_train_loader.dataset.max_mos - koniq10k_train_loader.dataset.min_mos)

    kadid10k_train_loader = set_dataset(kadid10k_train_csv, 16, kadid10k_set, num_workers, preprocess3, train_patch, False)
    kadid10k_val_loader = set_dataset(kadid10k_val_csv, 16, kadid10k_set, num_workers, preprocess2, 15, True)
    kadid10k_max_mos = (4.93 - kadid10k_train_loader.dataset.min_mos)*10 / (kadid10k_train_loader.dataset.max_mos - kadid10k_train_loader.dataset.min_mos)
    kadid10k_min_mos = (1  - kadid10k_train_loader.dataset.min_mos)*10 / (kadid10k_train_loader.dataset.max_mos - kadid10k_train_loader.dataset.min_mos)

    # used for range constraint loss
    min_mos = [live_min_mos, csiq_min_mos, bid_min_mos, clive_min_mos, koniq10k_min_mos, kadid10k_min_mos]
    max_mos = [live_max_mos, csiq_max_mos, bid_max_mos, clive_max_mos, koniq10k_max_mos, kadid10k_max_mos]
    model.min_mos = min_mos
    model.max_mos = max_mos

    train_loaders = [live_train_loader, csiq_train_loader, bid_train_loader, clive_train_loader,
                     koniq10k_train_loader, kadid10k_train_loader]

    result_pkl = {}
    for epoch in range(0, num_epoch):
        all_result = train(model)
        scheduler.step()

        result_pkl[str(epoch)] = all_result

        print(weighting_method.method.lambda_weight[:, epoch])

        print('...............current quality best...............')
        print('best quality epoch:{}'.format(best_epoch['quality']))
        print('best quality result:{}'.format(best_result['quality']))
        for dataset in srcc_dict.keys():
            print_text = dataset + ':' + 'srcc:{}'.format(srcc_dict[dataset])
            print(print_text)

        print('...............current scene best...............')
        print('best scene epoch:{}'.format(best_epoch['scene']))
        print('best scene result:{}'.format(best_result['scene']))

        print('...............current distortion best...............')
        print('best distortion epoch:{}'.format(best_epoch['distortion']))
        print('best distortion result:{}'.format(best_result['distortion']))
            
    pkl_name = os.path.join('checkpoints', str(session+1), 'all_results.pkl')
    with open(pkl_name, 'wb') as f:
        pickle.dump(result_pkl, f)

    lambdas = weighting_method.method.lambda_weight
    pkl_name = os.path.join('checkpoints', str(session+1), 'lambdas.pkl')
    with open(pkl_name, 'wb') as f:
        pickle.dump(lambdas, f)








