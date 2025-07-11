import os
import scipy.stats
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms
import torch.nn as nn
from ImageDataset import ImageDataset

from BaseCNN import BaseCNN

from Transformers import AdaptiveResize


class Trainer(object):
    def __init__(self, config):
        torch.manual_seed(config.seed)
        
        self.config = config

        if config.get_scores:
            live_str = 'live_test.txt'
            csiq_str = 'csiq_test.txt'
            kadid10k_str = 'kadid10k_test.txt'
            bid_str = 'bid_test.txt'
            clive_str = 'clive_test.txt'
            koniq10k_str = 'koniq10k_test.txt'
        else:
            live_str = 'live_valid.txt'
            csiq_str = 'csiq_valid.txt'
            kadid10k_str = 'kadid10k_valid.txt'
            bid_str = 'bid_valid.txt'
            clive_str = 'clive_valid.txt'
            koniq10k_str = 'koniq10k_valid.txt'

        self.min_mos = []
        self.max_mos = []

        self.train_transform = transforms.Compose([
            AdaptiveResize(512),
            transforms.RandomCrop(config.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])
        self.test_transform = transforms.Compose([
            AdaptiveResize(768),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])

        self.train_batch_size = config.batch_size
        self.test_batch_size = 1
        

        self.train_data = ImageDataset(csv_file=os.path.join(config.trainset, 'splits2', str(config.split), config.train_txt),
                                       img_dir="../",
                                       transform=self.train_transform,
                                       test= False)
        self.train_loader = DataLoader(self.train_data,
                                       batch_size=self.train_batch_size,
                                       shuffle=True,
                                       pin_memory=True,
                                       num_workers=16)

        # validation set configuration
        self.live_data = ImageDataset(csv_file=os.path.join(config.live_set, 'splits2', str(config.split), live_str),
                                      img_dir=os.path.join("../databaserelease2/"),
                                      transform=self.test_transform,
                                      test=True)


        self.live_loader = DataLoader(self.live_data,
                                      batch_size=self.test_batch_size,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=1)

        self.csiq_data = ImageDataset(csv_file=os.path.join(config.csiq_set, 'splits2', str(config.split), csiq_str),
                                      img_dir=os.path.join("../CSIQ/"),
                                      transform=self.test_transform,
                                      test=True)

        self.csiq_loader = DataLoader(self.csiq_data,
                                      batch_size=self.test_batch_size,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=1)

        self.kadid10k_data = ImageDataset(csv_file=os.path.join(config.kadid10k_set, 'splits2', str(config.split), kadid10k_str),
                                         img_dir=os.path.join("../kadid10k/"),
                                         transform=self.test_transform,
                                         test=True)

        self.kadid10k_loader = DataLoader(self.kadid10k_data,
                                         batch_size=self.test_batch_size,
                                         shuffle=False,
                                         pin_memory=True,
                                         num_workers=4)

        self.bid_data = ImageDataset(csv_file=os.path.join(config.bid_set, 'splits2', str(config.split), bid_str),
                                     img_dir=os.path.join("../BID/"),
                                     transform=self.test_transform,
                                     test=True)

        self.bid_loader = DataLoader(self.bid_data,
                                     batch_size=self.test_batch_size,
                                     shuffle=False,
                                     pin_memory=True,
                                     num_workers=1)

        self.clive_data = ImageDataset(csv_file=os.path.join(config.clive_set, 'splits2', str(config.split), clive_str),
                                       img_dir=os.path.join("../ChallengeDB_release/"),
                                       transform=self.test_transform,
                                       test=True)

        self.clive_loader = DataLoader(self.clive_data,
                                       batch_size=self.test_batch_size,
                                       shuffle=False,
                                       pin_memory=True,
                                       num_workers=1)

        self.koniq10k_data = ImageDataset(csv_file=os.path.join(config.koniq10k_set, 'splits2', str(config.split),koniq10k_str),
                                       img_dir=os.path.join("../koniq-10k/"),
                                       transform=self.test_transform,
                                       test=True)

        self.koniq10k_loader = DataLoader(self.koniq10k_data,
                                       batch_size=self.test_batch_size,
                                       shuffle=False,
                                       pin_memory=True,
                                       num_workers=4)
        self.min_mos.append(self.train_data.live_min),  self.max_mos.append(self.train_data.live_max)
        self.min_mos.append(self.train_data.csiq_min), self.max_mos.append(self.train_data.csiq_max)
        self.min_mos.append(self.train_data.kadid_min), self.max_mos.append(self.train_data.kadid_max)
        self.min_mos.append(self.train_data.bid_min), self.max_mos.append(self.train_data.bid_max)
        self.min_mos.append(self.train_data.clive_min), self.max_mos.append(self.train_data.clive_max)
        self.min_mos.append(self.train_data.koniq_min),self.max_mos.append(self.train_data.koniq_max)

        self.device = torch.device("cuda" if torch.cuda.is_available() and config.use_cuda else "cpu")

        self.model = BaseCNN(config)

        self.model = self.model.to(self.device)


        self.model_name = type(self.model).__name__

        # loss function
        self.loss_fn = nn.SmoothL1Loss()

        self.loss_fn.to(self.device)
        
        #set optimizer
        self.lr = config.lr

        params_dict = [{'params':self.model.backbone.parameters(),'lr':self.lr * 0.1},
                {'params':self.model.fc.parameters(),'lr':self.lr},
                {'params':self.model.nlm1.parameters(),'lr':self.lr},
                {'params':self.model.nlm2.parameters(),'lr':self.lr},
                {'params':self.model.nlm3.parameters(),'lr':self.lr},
                {'params':self.model.nlm4.parameters(),'lr':self.lr},
                {'params':self.model.nlm5.parameters(),'lr':self.lr},
                {'params':self.model.nlm6.parameters(),'lr':self.lr},
                {'params': self.model.trans1.parameters(), 'lr': self.lr},
                {'params': self.model.trans2.parameters(), 'lr': self.lr},
                {'params': self.model.trans3.parameters(), 'lr': self.lr},
                {'params': self.model.trans4.parameters(), 'lr': self.lr},
                {'params': self.model.trans5.parameters(), 'lr': self.lr},
                {'params': self.model.trans6.parameters(), 'lr': self.lr},]


        self.optimizer = torch.optim.Adam(params_dict)  


        # some states
        self.start_epoch = 0
        self.start_step = 0
        self.test_results_srcc = {'live': [], 'csiq': [],  'kadid10k': [], 'bid': [], 'clive': [], 'koniq10k': []}
        self.srcc_imp_prop = {}
        self.nlm_weight = {}
        self.ckpt_path = config.ckpt_path
        self.max_epochs = config.max_epochs
        self.epochs_per_eval = config.epochs_per_eval
        self.epochs_per_save = config.epochs_per_save
        self.val_srcc_best = 0.0


        # try load the model
        if config.resume or not config.train:
            if config.ckpt:
                ckpt = os.path.join(config.ckpt_path, config.ckpt)
            else:
                ckpt = self._get_latest_checkpoint(path=config.ckpt_path)
            self._load_checkpoint(ckpt=ckpt)

        self.scheduler = lr_scheduler.StepLR(self.optimizer,
                                             last_epoch=self.start_epoch-1,
                                             step_size=config.decay_interval,
                                             gamma=config.decay_ratio)
        

    def fit(self):
        for epoch in range(self.start_epoch, self.max_epochs):
            _ = self._train_single_epoch_regression(epoch)
            self.scheduler.step()


    def _train_single_epoch_regression(self, epoch):
        # initialize logging system
        num_steps_per_epoch = len(self.train_loader)

        # start training
        print('Adam learning rate backbone: {:.8f}, other layers: {:.8f}'.format(self.optimizer.param_groups[0]['lr'],self.optimizer.param_groups[1]['lr']))
        self.model.train()
        for step, sample_batched in enumerate(self.train_loader, 0):
            x, g ,std, tag = sample_batched['I'], sample_batched['mos'],sample_batched['std'],sample_batched['tag']
            x, g, tag = x.to(self.device), g.to(self.device), tag.to(self.device)
            self.optimizer.zero_grad()
        
            y1, y2, DNN_x, dnn_min, dnn_max= self.model(x,tag,minmos = self.min_mos, maxmos = self.max_mos, device = self.device, istest=False)
            self.loss_1 = self.loss_fn(y2[:,0], g[:,0])
            self.loss_2 = self.loss_fn(y2[:,1], g[:,1])
            self.loss_3 = self.loss_fn(y2[:,2], g[:,2])
            self.loss_4 = self.loss_fn(y2[:,3], g[:,3])
            self.loss_5 = self.loss_fn(y2[:,4], g[:,4])
            self.loss_6 = self.loss_fn(y2[:,5], g[:,5])
            self.loss_7 = self.loss_fn(y1,y2)
            MIN, MAX  = torch.min(dnn_min), torch.max(dnn_max)
            self.loss_8 = torch.zeros(1).to(self.device)
            for k in range(len(dnn_min)):
                self.loss_8 += torch.abs(dnn_min[k].squeeze()-MIN)+ torch.abs(dnn_max[k].squeeze()-MAX)

            self.loss = (self.loss_1+self.loss_2+4*self.loss_3+self.loss_4+self.loss_5+4*self.loss_6)/6 + self.loss_8/6 + 2*self.loss_7
            loss_str_1 = ('loss1:%.3f\t loss2:%.3f\t loss3:%.3f\t loss4:%.3f\t loss5:%.3f\t loss6:%.3f\t loss7:%.3f\t loss8:%.3f\t')
            print(loss_str_1%(self.loss_1.data.item(),self.loss_2.data.item(),self.loss_3.data.item(),
            self.loss_4.data.item(),self.loss_5.data.item(),self.loss_6.data.item(),self.loss_7.data.item(),self.loss_8.data.item()))
            self.loss.backward()
            self.optimizer.step()

            format_str = ('(E:%d, it:%d / %d) [Loss = %.4f]')
            print(format_str % (epoch, step + 1, num_steps_per_epoch, self.loss.item()))
            self.start_step = 0

        if (epoch % self.epochs_per_eval == 0):
            # evaluate after every other epoch
            test_results_srcc,PredLoss_avg, y1y2loss = self.eval()  # srcc and plcc across all validate sets.

            srcc_temp = 0
            self.test_results_srcc['live'].append(test_results_srcc['live'])
            self.test_results_srcc['csiq'].append(test_results_srcc['csiq'])
            self.test_results_srcc['kadid10k'].append(test_results_srcc['kadid10k'])
            self.test_results_srcc['bid'].append(test_results_srcc['bid'])
            self.test_results_srcc['clive'].append(test_results_srcc['clive'])
            self.test_results_srcc['koniq10k'].append(test_results_srcc['koniq10k'])

            srcc_temp = test_results_srcc['live'] * 779  + test_results_srcc['csiq'] * 866  + test_results_srcc['kadid10k'] * 10125 + \
                test_results_srcc['bid'] * 586 + test_results_srcc['clive'] * 1162 + test_results_srcc['koniq10k'] * 10073
            srcc_temp = srcc_temp / (779+866+10125+586+1162+10073)
            out_str = 'Testing: LIVE SRCC: {:.4f}  CSIQ SRCC: {:.4f}  KADID10K SRCC: {:.4f} ' \
                      'BID SRCC: {:.4f} CLIVE SRCC: {:.4f}  KONIQ10K SRCC: {:.4f} Weight_SRCC: {:.3f} '.format(
                test_results_srcc['live'],
                test_results_srcc['csiq'],
                test_results_srcc['kadid10k'],
                test_results_srcc['bid'],
                test_results_srcc['clive'],
                test_results_srcc['koniq10k'],
                srcc_temp)

            PredLoss_avg_str = '(pred_loss_avg) loss_avg loss1:{:.3f} loss2:{:.3f} loss3:{:.3f} loss4:{:.3f} loss5:{:.3f} loss6:{:.3f}'.format(
                PredLoss_avg['live'],
                PredLoss_avg['csiq'],
                PredLoss_avg['kadid10k'],
                PredLoss_avg['bid'],
                PredLoss_avg['clive'],
                PredLoss_avg['koniq10k'])

            y1y2loss_str = '(y1y2loss_avg) loss_avg loss1:{:.3f} loss2:{:.3f} loss3:{:.3f} loss4:{:.3f} loss5:{:.3f} loss6:{:.3f}'.format(
                y1y2loss['live'],
                y1y2loss['csiq'],
                y1y2loss['kadid10k'],
                y1y2loss['bid'],
                y1y2loss['clive'],
                y1y2loss['koniq10k'])


            print(out_str)
            print(PredLoss_avg_str)
            print(y1y2loss_str)

            record_name = '{}_{}.txt'.format(str(self.config.split),'srcc_plcc')
            record_name = os.path.join('./srcc',record_name)
            with open(record_name,'a') as f:
                f.write(out_str+'\n')
                f.write(PredLoss_avg_str+'\n')
                f.write(y1y2loss_str + '\n')
            
            # save the best model(max(sum(SRCC)))
            if self.val_srcc_best < srcc_temp:
                self.val_srcc_best = srcc_temp
                model_name = '{}-{}.pt'.format(self.model_name,'best')
                model_name = os.path.join(self.ckpt_path, model_name)
                self.sd = self.model.state_dict()
                self._save_checkpoint({
                    'epoch': epoch,
                    'state_dict': self.sd,
                    'optimizer': self.optimizer.state_dict(),
                    'test_results_srcc': self.test_results_srcc,
                }, model_name)

            elif self.val_srcc_best>=srcc_temp:
                pass
        return self.loss.data.item()


    def eval(self):

        # summary
        srcc = {}
        plcc = {}
        loss_ave = {}
        y1y2loss_ave = {}
        self.model.eval()

        # LIVE
        q_mos = []
        q_hat = []
        q_loss = []
        y1y2_loss = []
        for _, sample_batched in enumerate(self.live_loader, 0):
            x, y, tag = sample_batched['I'], sample_batched['mos'], sample_batched['tag']
            q_mos.append(y.data.numpy())
            x = x.to(self.device)
            y1,y2, _ = self.model(x,tag,minmos = self.min_mos, maxmos = self.max_mos,device=self.device, istest=True)
            y = y.to(self.device)
            loss_temp = self.loss_fn(y2.squeeze(0), y)
            y1y2loss = self.loss_fn(y1,y2)
            y1y2_loss.append(y1y2loss.item())
            q_loss.append(loss_temp.item())
            q_hat.append(y2.item())
        srcc['live'] = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
        loss_ave['live'] = np.mean(q_loss)
        y1y2loss_ave['live'] = np.mean(y1y2_loss) 
        print('LIVE: ','SRCC: ',srcc['live'],'PredLoss_avg: ',loss_ave['live'],'y1y2Loss: ',y1y2loss_ave['live'])

        # CSIQ
        q_mos = []
        q_hat = []
        q_loss = []
        y1y2_loss = []
        for _, sample_batched in enumerate(self.csiq_loader, 0):
            x, y, tag = sample_batched['I'], sample_batched['mos'], sample_batched['tag']
            q_mos.append(y.data.numpy())
            x = x.to(self.device)
            y1,y2, _ = self.model(x,tag,minmos = self.min_mos, maxmos = self.max_mos,device=self.device, istest=True)
            y = y.to(self.device)
            loss_temp = self.loss_fn(y2.squeeze(0), y)
            y1y2loss = self.loss_fn(y1,y2)
            y1y2_loss.append(y1y2loss.item())
            q_loss.append(loss_temp.item())
            q_hat.append(y2.item())
        srcc['csiq'] = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
        loss_ave['csiq'] = np.mean(q_loss)
        y1y2loss_ave['csiq'] = np.mean(y1y2_loss) 
        print('CSIQ: ','SRCC: ',srcc['csiq'],'PredLoss_avg: ',loss_ave['csiq'],'y1y2Loss: ',y1y2loss_ave['csiq'])

        # KADID-10k
        q_mos = []
        q_hat = []
        q_loss = []
        y1y2_loss = []
        for _, sample_batched in enumerate(self.kadid10k_loader, 0):
            x, y, tag = sample_batched['I'], sample_batched['mos'], sample_batched['tag']
            q_mos.append(y.data.numpy())
            x = x.to(self.device)
            y1,y2, _ = self.model(x,tag,minmos = self.min_mos, maxmos = self.max_mos,device=self.device, istest=True)
            y = y.to(self.device)
            loss_temp = self.loss_fn(y2.squeeze(0), y)
            y1y2loss = self.loss_fn(y1,y2)
            y1y2_loss.append(y1y2loss.item())
            q_loss.append(loss_temp.item())
            q_hat.append(y2.item())
        srcc['kadid10k'] = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
        loss_ave['kadid10k'] = np.mean(q_loss)
        y1y2loss_ave['kadid10k'] = np.mean(y1y2_loss) 
        print('KADID-10k: ','SRCC: ',srcc['kadid10k'],'PredLoss_avg: ',loss_ave['kadid10k'],'y1y2Loss: ',y1y2loss_ave['kadid10k'])

        # BID
        q_mos = []
        q_hat = []
        q_loss = []
        y1y2_loss = []
        for _, sample_batched in enumerate(self.bid_loader, 0):
            x, y, tag = sample_batched['I'], sample_batched['mos'], sample_batched['tag']
            q_mos.append(y.data.numpy())
            x = x.to(self.device)
            y1,y2, _ = self.model(x,tag,minmos = self.min_mos, maxmos = self.max_mos,device=self.device, istest=True)
            y = y.to(self.device)
            loss_temp = self.loss_fn(y2.squeeze(0), y)
            y1y2loss = self.loss_fn(y1,y2)
            y1y2_loss.append(y1y2loss.item())
            q_loss.append(loss_temp.item())
            q_hat.append(y2.item())
        srcc['bid'] = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
        loss_ave['bid'] = np.mean(q_loss)
        y1y2loss_ave['bid'] = np.mean(y1y2_loss) 
        print('BID: ','SRCC: ',srcc['bid'],'PredLoss_avg: ',loss_ave['bid'],'y1y2Loss: ',y1y2loss_ave['bid'])

        # CLIVE
        q_mos = []
        q_hat = []
        q_loss = []
        y1y2_loss = []
        for _, sample_batched in enumerate(self.clive_loader, 0):
            x, y, tag = sample_batched['I'], sample_batched['mos'], sample_batched['tag']
            q_mos.append(y.data.numpy())
            x = x.to(self.device)
            y1,y2, _ = self.model(x,tag,minmos = self.min_mos, maxmos = self.max_mos,device=self.device, istest=True)
            y = y.to(self.device)
            loss_temp = self.loss_fn(y2.squeeze(0), y)
            y1y2loss = self.loss_fn(y1,y2)
            y1y2_loss.append(y1y2loss.item())
            q_loss.append(loss_temp.item())
            q_hat.append(y2.item())
        srcc['clive'] = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
        loss_ave['clive'] = np.mean(q_loss)
        y1y2loss_ave['clive'] = np.mean(y1y2_loss) 
        print('CLIVE: ','SRCC: ',srcc['clive'],'PredLoss_avg: ',loss_ave['clive'],'y1y2Loss: ',y1y2loss_ave['clive'])

        # KONIQ-10K
        q_mos = []
        q_hat = []
        q_loss = []
        y1y2_loss = []
        for _, sample_batched in enumerate(self.koniq10k_loader, 0):
            x, y, tag = sample_batched['I'], sample_batched['mos'], sample_batched['tag']
            q_mos.append(y.data.numpy())
            x = x.to(self.device)
            y1,y2, _ = self.model(x,tag,minmos = self.min_mos, maxmos = self.max_mos,device=self.device, istest=True)
            y = y.to(self.device)
            loss_temp = self.loss_fn(y2.squeeze(0), y)
            y1y2loss = self.loss_fn(y1,y2)
            y1y2_loss.append(y1y2loss.item())
            q_loss.append(loss_temp.item())
            q_hat.append(y2.item())
        srcc['koniq10k'] = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
        loss_ave['koniq10k'] = np.mean(q_loss)
        y1y2loss_ave['koniq10k'] = np.mean(y1y2_loss) 
        print('KONIQ-10k: ','SRCC: ',srcc['koniq10k'],'PredLoss_avg: ',loss_ave['koniq10k'],'y1y2Loss: ',y1y2loss_ave['koniq10k'])

        return srcc, loss_ave, y1y2loss_ave
 
    def get_scores(self):
        all_mos = {}
        all_hat = {}
        all_DNN_mos = {}

        self.model.eval()
        q_mos = []
        q_hat = []
        DNN_mos = []
        for _, sample_batched in enumerate(self.live_loader, 0):
            x, y, tag = sample_batched['I'], sample_batched['mos'], sample_batched['tag']
            x = Variable(x)
            x = x.to(self.device)

            y1,y2, DNN_x = self.model(x,tag,minmos = self.min_mos, maxmos = self.max_mos,device=self.device, istest=True)
            q_mos.append(y.item())
            q_hat.append(y2.item())
            DNN_mos.append(DNN_x.item())
        all_mos['live'] = q_mos
        all_DNN_mos['live'] = DNN_mos
        all_hat['live'] = q_hat

        q_mos = []
        q_hat = []
        DNN_mos = []
        for _, sample_batched in enumerate(self.csiq_loader, 0):
            x, y, tag = sample_batched['I'], sample_batched['mos'], sample_batched['tag']
            x = Variable(x)
            x = x.to(self.device)
            y1,y2, DNN_x = self.model(x,tag,minmos = self.min_mos, maxmos = self.max_mos,device=self.device, istest=True)
            q_mos.append(y.item())
            q_hat.append(y2.item())
            DNN_mos.append(DNN_x.item())
        all_mos['csiq'] = q_mos
        all_DNN_mos['csiq'] = DNN_mos
        all_hat['csiq'] = q_hat


        q_mos = []
        q_hat = []
        DNN_mos = []
        for _, sample_batched in enumerate(self.kadid10k_loader, 0):
            x, y, tag = sample_batched['I'], sample_batched['mos'], sample_batched['tag']
            x = Variable(x)
            x = x.to(self.device)
            y1,y2, DNN_x = self.model(x,tag,minmos = self.min_mos, maxmos = self.max_mos,device=self.device, istest=True)
            q_mos.append(y.item())
            q_hat.append(y2.item())
            DNN_mos.append(DNN_x.item())
        all_mos['kadid10k'] = q_mos
        all_DNN_mos['kadid10k'] = DNN_mos
        all_hat['kadid10k'] = q_hat

        q_mos = []
        q_hat = []
        DNN_mos = []
        for _, sample_batched in enumerate(self.bid_loader, 0):
            x, y, tag = sample_batched['I'], sample_batched['mos'], sample_batched['tag']
            x = Variable(x)
            x = x.to(self.device)
            y1,y2, DNN_x = self.model(x,tag,minmos = self.min_mos, maxmos = self.max_mos,device=self.device, istest=True)
            q_mos.append(y.item())
            q_hat.append(y2.item())
            DNN_mos.append(DNN_x.item())
        all_mos['bid'] = q_mos
        all_DNN_mos['bid'] = DNN_mos
        all_hat['bid'] = q_hat

        q_mos = []
        q_hat = []
        DNN_mos = []
        for _, sample_batched in enumerate(self.clive_loader, 0):
            x, y, tag = sample_batched['I'], sample_batched['mos'], sample_batched['tag']
            x = Variable(x)
            x = x.to(self.device)
            y1,y2, DNN_x = self.model(x,tag,minmos = self.min_mos, maxmos = self.max_mos,device=self.device, istest=True)
            q_mos.append(y.item())
            q_hat.append(y2.item())
            DNN_mos.append(DNN_x.item())
        all_mos['clive'] = q_mos
        all_DNN_mos['clive'] = DNN_mos
        all_hat['clive'] = q_hat

        q_mos = []
        q_hat = []
        DNN_mos = []
        for _, sample_batched in enumerate(self.koniq10k_loader, 0):
            x, y,tag  = sample_batched['I'], sample_batched['mos'], sample_batched['tag']
            x = Variable(x)
            x = x.to(self.device)
            y1,y2, DNN_x = self.model(x,tag,minmos = self.min_mos, maxmos = self.max_mos,device=self.device, istest=True)
            q_mos.append(y.item())
            q_hat.append(y2.item())
            DNN_mos.append(DNN_x.item())

        all_mos['koniq10k'] = q_mos
        all_DNN_mos['koniq10k'] = DNN_mos
        all_hat['koniq10k'] = q_hat
        return all_mos, all_hat, all_DNN_mos

    def _load_checkpoint(self, ckpt):
        if os.path.isfile(ckpt):
            print("[*] loading checkpoint '{}'".format(ckpt))
            checkpoint = torch.load(ckpt)
            self.start_epoch = checkpoint['epoch']+1
            self.test_results_srcc = checkpoint['test_results_srcc']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("[*] loaded checkpoint '{}' (epoch {})"
                  .format(ckpt, checkpoint['epoch']))
        else:
            print("[!] no checkpoint found at '{}'".format(ckpt))

    @staticmethod
    def _get_latest_checkpoint(path):
        ckpts = os.listdir(path)
        ckpts = [ckpt for ckpt in ckpts if not os.path.isdir(os.path.join(path, ckpt))]
        all_times = sorted(ckpts, reverse=True)
        return os.path.join(path, all_times[0])

    # save checkpoint
    @staticmethod
    def _save_checkpoint(state, filename='checkpoint.pth.tar'):
        torch.save(state, filename)

