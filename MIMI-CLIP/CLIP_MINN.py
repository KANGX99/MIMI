import torch
import torch.nn as nn
import clip
import MRNN
import torch.nn.functional as F


class CLIP_MINN(nn.Module):
    def __init__(self, clip_model_name="ViT-B/32", device="cpu", block_num=1):
        super().__init__()
        self.device = device
        self.clip_model, _ = clip.load(clip_model_name, device=self.device, jit=False)
        # Monotonic Invertible Neural Network
        self.MINN1 = MRNN.MonotonicInvNet(block_num)
        self.MINN2 = MRNN.MonotonicInvNet(block_num)
        self.MINN3 = MRNN.MonotonicInvNet(block_num)
        self.MINN4 = MRNN.MonotonicInvNet(block_num)
        self.MINN5 = MRNN.MonotonicInvNet(block_num)
        self.MINN6 = MRNN.MonotonicInvNet(block_num)
        # Monotonic Function
        self.M1 = MRNN.MonotonicFunc(3,[100,100,100], nb_steps=100)
        self.M2 = MRNN.MonotonicFunc(3,[100,100,100], nb_steps=100)
        self.M3 = MRNN.MonotonicFunc(3,[100,100,100], nb_steps=100)
        self.M4 = MRNN.MonotonicFunc(3,[100,100,100], nb_steps=100)
        self.M5 = MRNN.MonotonicFunc(3,[100,100,100], nb_steps=100)
        self.M6 = MRNN.MonotonicFunc(3,[100,100,100], nb_steps=100)

        self.min_mos = []
        self.max_mos = []

    def forward(self, logits_quality, isTest, nn_id = None, samples_per_ds = None):
        if not isTest:
            y1_batch,y2_batch = [], []
            split_pred = torch.split(logits_quality, samples_per_ds)
            multi_range_min, multi_range_max = [], []
            for index in range(len(split_pred)):
                mos_min = torch.tensor([[self.min_mos[index]]]).to(self.device)
                mos_max = torch.tensor([[self.max_mos[index]]]).to(self.device)
                h1 = torch.zeros(1, 2).to(self.device)

                x1 = split_pred[index].unsqueeze(1)
                h = torch.zeros(x1.shape[0], 2).to(self.device)
                x2 = getattr(self,'M'+str(index+1),'None')(x1,h)
                y1,y2 = getattr(self, 'MINN'+str(index+1), 'None')(x1,x2,h)

                range_min, _ = getattr(self, 'MINN'+str(index+1), 'None')(mos_min,mos_min,h1,rev = True)
                range_max, _ = getattr(self, 'MINN'+str(index+1), 'None')(mos_max,mos_max,h1,rev = True)
                multi_range_min.append(range_min)
                multi_range_max.append(range_max)

                y1_batch.append(y1)
                y2_batch.append(y2)
            y1_batch = torch.cat(y1_batch, dim = 0).squeeze(1)
            y2_batch = torch.cat(y2_batch, dim = 0).squeeze(1)
            multi_range_min = torch.cat(multi_range_min, dim = 0).squeeze(1)
            multi_range_max = torch.cat(multi_range_max, dim = 0).squeeze(1)
            return y1_batch, y2_batch, logits_quality, multi_range_min, multi_range_max
        else:
            x1 = logits_quality.unsqueeze(1)
            h = torch.zeros(x1.shape[0], 2).to(self.device)
            x2 = getattr(self,'M'+str(nn_id),'None')(x1,h)
            y1,y2 = getattr(self, 'MINN'+str(nn_id), 'None')(x1,x2,h)
            return y1.squeeze(1), y2.squeeze(1), logits_quality