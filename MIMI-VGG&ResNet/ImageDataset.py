import os
import torch
import functools
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def image_loader(image_name):
    if has_file_allowed_extension(image_name, IMG_EXTENSIONS):
        I = Image.open(image_name)
    return I


def get_default_img_loader():
    return functools.partial(image_loader)


class ImageDataset(Dataset):
    def __init__(self, csv_file,
                 img_dir,
                 transform=None,
                 test=False,
                 get_loader=get_default_img_loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory of the images.
            transform (callable, optional): transform to be applied on a sample.
        """
        print('start loading csv data...')
        print("Split path:",csv_file)
        self.data = pd.read_csv(csv_file, sep='\t', header=None)
        self.test = test
        
        if not test:
            tags = self.data[3]
            self.data_tag1 = self.data[:][tags==1]
            self.data_tag2 = self.data[:][tags==2]
            self.data_tag3 = self.data[:][tags==3]
            self.data_tag4 = self.data[:][tags==4]
            self.data_tag5 = self.data[:][tags==5]
            self.data_tag6 = self.data[:][tags==6]
            
            self.data_mos1 = self.data_tag1.iloc[:,1]
            max_mos = max(self.data_mos1)
            min_mos = min(self.data_mos1)
            self.live_max = (114.4147 - min_mos)/(max_mos - min_mos)*10
            self.live_min = (0 - min_mos)/(max_mos - min_mos)*10
            self.data_tag1.iloc[:,1] = (self.data_mos1 - min_mos)/(max_mos - min_mos) *10

            max_mos = max(self.data_tag2.iloc[:,1])
            min_mos = min(self.data_tag2.iloc[:,1])
            self.csiq_max = (1 - min_mos)/(max_mos - min_mos)*10
            self.csiq_min = (0 - min_mos)/(max_mos - min_mos)*10
            self.data_tag2.iloc[:,1] = (self.data_tag2.iloc[:,1] - min_mos)/(max_mos - min_mos) *10

            max_mos = max(self.data_tag3.iloc[:,1])
            min_mos = min(self.data_tag3.iloc[:,1])
            self.kadid_max = (4.93 - min_mos)/(max_mos - min_mos)*10
            self.kadid_min = (1 - min_mos)/(max_mos - min_mos)*10
            self.data_tag3.iloc[:,1] = (self.data_tag3.iloc[:,1] - min_mos)/(max_mos - min_mos) *10

            max_mos = max(self.data_tag4.iloc[:,1])
            min_mos = min(self.data_tag4.iloc[:,1])
            self.bid_max = (4.9198 - min_mos)/(max_mos - min_mos)*10
            self.bid_min = (0.1638 - min_mos)/(max_mos - min_mos)*10
            self.data_tag4.iloc[:,1] = (self.data_tag4.iloc[:,1] - min_mos)/(max_mos - min_mos) *10

            max_mos = max(self.data_tag5.iloc[:,1])
            min_mos = min(self.data_tag5.iloc[:,1])
            self.clive_max = (92.4320 - min_mos)/(max_mos - min_mos)*10
            self.clive_min = (3.4200 - min_mos)/(max_mos - min_mos)*10
            self.data_tag5.iloc[:,1] = (self.data_tag5.iloc[:,1] - min_mos)/(max_mos - min_mos) *10
            
            max_mos = max(self.data_tag6.iloc[:,1])
            min_mos = min(self.data_tag6.iloc[:,1])
            self.koniq_max = (4.3100 - min_mos)/(max_mos - min_mos)*10
            self.koniq_min = (1.0962 - min_mos)/(max_mos - min_mos)*10
            self.data_tag6.iloc[:,1] = (self.data_tag6.iloc[:,1] - min_mos)/(max_mos - min_mos) *10
            


            self.len_tag1 = len(self.data_tag1)
            self.len_tag2 = len(self.data_tag2)
            self.len_tag3 = len(self.data_tag3)
            self.len_tag4 = len(self.data_tag4)
            self.len_tag5 = len(self.data_tag5)
            self.len_tag6 = len(self.data_tag6)
            self.max_len = max([self.len_tag1,self.len_tag2,self.len_tag3,self.len_tag4,self.len_tag5,self.len_tag6])
        
        print('%d csv data successfully loaded!' % self.__len__())
        self.img_dir = img_dir
        self.transform = transform
        self.loader = get_loader()
   
    def __getitem__(self, index):
        
        if not self.test:
            '''数据量少的数据集重复采样'''
            img_info = []
            img_info.append(self.data_tag1.iloc[index%self.len_tag1, :])
            img_info.append(self.data_tag2.iloc[index%self.len_tag2, :])
            img_info.append(self.data_tag3.iloc[index%self.len_tag3, :])
            img_info.append(self.data_tag4.iloc[index%self.len_tag4, :])
            img_info.append(self.data_tag5.iloc[index%self.len_tag5, :])
            img_info.append(self.data_tag6.iloc[index%self.len_tag6, :])


    
            img_data = []
            img_mos = torch.zeros(6)
            img_std = torch.zeros(6)
            img_tag = torch.zeros(6)
            for sap_id in range(len(img_info)):
                I = self.loader(os.path.join(self.img_dir,img_info[sap_id][0]))
                if self.transform is not None:
                    I = self.transform(I)
                img_data.append(I)
                img_mos[sap_id] = img_info[sap_id][1]
                img_std[sap_id] = img_info[sap_id][2]
                img_tag[sap_id] = img_info[sap_id][3]
            
            img_data = torch.stack(img_data)
            sample = {'I': img_data, 'mos': img_mos, 'std': img_std,'tag':img_tag}
        else:
            image_name1 = os.path.join(self.img_dir, self.data.iloc[index, 0])
            I1 = self.loader(image_name1)
            if self.transform is not None:
                I1 = self.transform(I1)
            mos = self.data.iloc[index, 1]
            std = self.data.iloc[index, 2]
            tag = self.data.iloc[index, 3]
            sample = {'I': I1, 'mos': mos, 'std': std,'tag':tag}
        return sample

    def __len__(self):
        if self.test:
            return len(self.data.index)
        else:
            return self.max_len
 
