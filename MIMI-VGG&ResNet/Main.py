import argparse
import TrainModel
import scipy.io as sio
import os


def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument('--get_scores', type=bool, default=False)
    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=19901116)

    parser.add_argument("--backbone", type=str, default='vgg16')

    parser.add_argument("--split", type=int, default=1)

    parser.add_argument("--trainset", type=str, default="./SampledDataSplit10/") #combine 6 databases
    parser.add_argument("--live_set", type=str, default="./SampledDataSplit10/databaserelease2/")
    parser.add_argument("--csiq_set", type=str, default="./SampledDataSplit10/CSIQ/")
    parser.add_argument("--bid_set", type=str, default="./SampledDataSplit10/BID/")
    parser.add_argument("--clive_set", type=str, default="./SampledDataSplit10/ChallengeDB_release/")
    parser.add_argument("--koniq10k_set", type=str, default="./SampledDataSplit10/koniq-10k/")
    parser.add_argument("--kadid10k_set", type=str, default="./SampledDataSplit10/kadid10k/")

    parser.add_argument('--ckpt_path', default='./checkpoint', type=str,
                        metavar='PATH', help='path to checkpoints')
    parser.add_argument('--ckpt', default=None, type=str, help='name of the checkpoint to load')

    parser.add_argument("--train_txt", type=str, default='train.txt')

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=384, help='None means random resolution')
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--decay_interval", type=int, default=5)
    parser.add_argument("--decay_ratio", type=float, default=0.8)
    parser.add_argument("--epochs_per_eval", type=int, default=1)
    parser.add_argument("--epochs_per_save", type=int, default=1)
    
    parser.add_argument("--continue_train",type = bool, default=False)

    return parser.parse_args()


def main(cfg):
    t = TrainModel.Trainer(cfg)
    if cfg.train: 
        t.fit() 
    elif cfg.get_scores: 
        all_mos, all_hat, all_DNN_mos = t.get_scores()
        scores_path = os.path.join('./scores/', ('scores' + str(cfg.split) + '.mat'))
        sio.savemat(scores_path, {'mos': all_mos, 'hat': all_hat, 'DNN_mos':all_DNN_mos})


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES']='2'
    config = parse_config()
    if config.get_scores:
        for i in range(1, 11): # set 10 splits
            config = parse_config()
            config.split = i
            config.ckpt_path = os.path.join(config.ckpt_path, str(config.split))
            config.resume = True 
            config.ckpt = 'BaseCNN-best' + '.pt'
            main(config)
    else:
        for i in range(1, 11):  # set 10 splits
            config = parse_config()
            config.split = i
            config.ckpt_path = os.path.join(config.ckpt_path, str(config.split))
            if not os.path.exists(config.ckpt_path):
                os.makedirs(config.ckpt_path)

            if config.continue_train:
                config.resume = True
                config.ckpt = 'BaseCNN-best' + '.pt'

            main(config)








