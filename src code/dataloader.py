from builtins import NotImplementedError
import pandas as pd
from numpy import dtype, zeros

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import json
from PIL import Image

def denormalize(tensor,mean=0.5,std=0.5):
    for b in range(tensor.shape[0]):
        tensor[b].mul_(std).add_(mean)
    return tensor


class ICLEVR_dataset(Dataset):
    def __init__(self, args,mode='train'):
        assert mode == 'train' or mode == 'test' or mode=='new_test'
        self.mode=mode
        self.data_root=args.data_root
        self.img_root=args.img_root
        f = open(f'{self.data_root}/objects.json')
        self.num_class=args.num_class
        self.label_map=json.load(f)
        self.tf=transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.image_size=args.image_size

        if self.mode=='train':
            f = open(f'{self.data_root}/train.json')
            
            # change to df
            self.data=pd.DataFrame(list(json.load(f).items()),columns=['img_file', 'label'])
        elif self.mode=="test":
            f = open(f'{self.data_root}/test.json')
            self.data=json.load(f)
        elif self.mode=="new_test":
            f = open(f'{self.data_root}/new_test.json')
            self.data=json.load(f)
        else:
            raise NotImplementedError

            
    def __len__(self):
        return len(self.data)
        

    
    def __getitem__(self, index):
        if self.mode=='train':
            file_name,obj_label=self.data.iloc[index,:]
            img= Image.open(f'{self.img_root}/{file_name}').convert("RGB") 
            img=self.tf(img)

            # change label into id
            label_id=[]
            for l in obj_label:
                label_id.append(self.label_map[l])
            
            # fill to 3 labels
            while len(label_id)<3:
                label_id.append(self.num_class) # null is 24
            label_id=torch.Tensor(label_id).to(dtype=int)

            return img,label_id
        else:
            obj_label=self.data[index]
            
            # change label into id
            label_id=[]
            for l in obj_label:
                label_id.append(self.label_map[l])
            
            one_hot=torch.zeros(self.num_class).to(dtype=int)
            for i in label_id:
                one_hot[i]=1
            
            # fill to 3 labels
            while len(label_id)<3:
                label_id.append(self.num_class) # null is 24
            label_id=torch.Tensor(label_id).to(dtype=int)

            return label_id,one_hot 



if __name__=="__main__":

    f = open('./train.json')
    data=pd.DataFrame(list(json.load(f).items()),columns=['img_file', 'label'])
    print(data)
    file_name,label=data.iloc[0,:]
    print(file_name,label)
