import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os 
import sklearn

import librosa
from nnAudio import Spectrogram
import albumentations as A
from torch.utils.data import Dataset
#import torchvision.transforms as transforms
from torch.utils.data import Dataset,WeightedRandomSampler,TensorDataset
import torch.nn as nn
import pretrainedmodels
from torch.nn import functional as F
from torch.cuda import amp

from torchvision.utils import save_image
import torch.nn.functional as nnf


from PIL import Image
from PIL import ImageFile
#from torch_lr_finder import LRFinder




torch.backends.cudnn.benchmark = False



ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.manual_seed(0)
np.random.seed(0)




def wav_to_img(path,save_path,signal_length=None):
    audio_list = os.listdir(path)
    for idx,aud in tqdm(enumerate(audio_list),leave=False):
      #print(aud)
      #sr, song = wavfile.read(f'drive/My Drive/train_audio/{aud}')
      song,sr = librosa.load(f'{path}{aud}')
      song = sklearn.preprocessing.minmax_scale(song, axis=0)
      #np.amax(np.abs(samples))
      #print(x.shape)
      x = song
      #print("before",x.shape)
      #x = np.pad(x,(0,signal_length-x.shape[0]))
      #print(x.shape)
      x = torch.tensor(x).to("cuda",dtype=torch.float) 
      spec0 = Spectrogram.MelSpectrogram(n_fft=1024, n_mels=128, trainable_mel=False, trainable_STFT=False,
                                          fmin=15,fmax=11000,sr=sr,pad_mode='reflect')  
      spec1 = Spectrogram.STFT(n_fft=1024, hop_length=512, trainable=False)
      spec2 = Spectrogram.MFCC(sr=22050, n_mfcc=200, norm='ortho', device='cuda:0', verbose=True)

      x0 = spec0(x)
      x0 = x0.unsqueeze(1)

      x1 = spec1(x)
      x1 = x1[:,:,:,0].unsqueeze(1)

      x2 = spec2(x)
      x2 = x2.unsqueeze(1)

      x0 = nnf.interpolate(x0, size=(224, 224), mode='bicubic', align_corners=False)
      x1 = nnf.interpolate(x1, size=(224, 224), mode='bicubic', align_corners=False)
      x2 = nnf.interpolate(x2, size=(224, 224), mode='bicubic', align_corners=False)

      img = torch.cat((x0,x1,x2), dim=1)

      aud1 = aud.replace('.wav','')
      save_image(img,f'{save_path}{aud1}.jpeg')





class GIZ_Dataset(Dataset):
    def __init__(self,audio_path,data,targets,device='cuda',transform=None):
        self.audio_path = audio_path
        self.data = data
        self.targets = targets
        self.audio_id = self.data['fn'].unique()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self,item):
        aud = self.audio_id[item]
        aud1 = aud.replace('.wav','.jpeg')
        image = Image.open(f"{self.audio_path}/{aud1}")
        image = np.array(image)

        if self.transform is not None:
          image = self.transform(image=image)["image"]
           #image = self.transform.augment(image)

        image = np.transpose(image,(2,0,1))
        label = int(self.targets[item]) 
        
        return {
            'x': torch.tensor(image,dtype=torch.float),
            'y': torch.tensor(label)
            }
        
class GIZ_Test_Dataset(Dataset):
    def __init__(self,audio_path,data,transform=None):
        self.audio_path = audio_path
        self.data = data
        self.audio_id = self.data['fn'].unique()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self,item):
        aud = self.audio_id[item]
        aud1 = aud.replace('.wav','.jpeg')
        image = Image.open(f"{self.audio_path}/{aud1}")
        image = np.array(image)
        if self.transform is not None:
          image = self.transform(image=image)["image"]

        image = np.transpose(image,(2,0,1))

        return {
            'x': torch.tensor(image),
            'aud':aud
            }       

# Build Engine 

class Engine:
    def __init__(self,model,optimizer,device):
        self.model =model
        self.optimizer =optimizer
        self.device = device

    def loss_fn(self,outputs,targets):
        return nn.CrossEntropyLoss()(outputs,targets)
    
    def train(self,data_loader,scaler):
        self.model.train()
        final_loss = 0
        for data in data_loader:
            self.optimizer.zero_grad()
            with amp.autocast(): 
                inputs = data['x'].to(self.device,dtype=torch.float)
                targets = data['y'].to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs,targets)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update( )
            final_loss += loss.item()
        return final_loss/len(data_loader)
    
    
    def mix_up_train(self,data_loader,scaler,alpha=0.2):
        self.model.train()
        final_loss = 0
        for data in data_loader:
            bz = data['x'].size()[0]
            #print(bz)
            lam = np.random.beta(alpha,alpha)
            #lam_x = torch.reshape(lam,(bz,1,1,1))
            #lam_y = torch.reshape(lam,(bz,1))

            input_ = data['x'].to(self.device,dtype=torch.float)
            target_ = data['y'].to(self.device)

            rand_index = torch.randperm(bz).to(self.device)

            target_a = target_
            target_b = target_[rand_index]

            self.optimizer.zero_grad()
            with amp.autocast():
                inputs = lam * input_[:,:,:,:] + (1-lam) * input_[rand_index,:,:,:]
                #targets = lam_y*y1 + (1-lam_y)*y2
                #*ww.to(self.device,dtype=torch.float)
                outputs = self.model(inputs)
                loss = lam * self.loss_fn(outputs,target_a) + (1-lam) * self.loss_fn(outputs,target_b)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update( )
            final_loss += loss.item()
        return final_loss/len(data_loader) 

    def cut_mix_train(self,data_loader,scaler,alpha=1.0):
        self.model.train()
        final_loss = 0

        for data in data_loader:
            bz = data['x'].size()[0]
            W = data['x'].size()[2]
            H = data['x'].size()[3]

            input_ = data['x'].to(self.device,dtype=torch.float)
            target_ = data['y'].to(self.device)
            rand_index = torch.randperm(bz).to(self.device)

            target_a = target_
            target_b = target_[rand_index]


            lam = np.random.beta(alpha,alpha)

            r_x = np.random.randint(W)
            r_y = np.random.randint(H)

            r_w = np.int(W * np.sqrt(1-lam))
            r_h = np.int(H * np.sqrt(1-lam))

            x1 = np.clip((r_x - r_w) // 2,0,W)
            x2 = np.clip((r_x + r_w) // 2,0,W)
            y1 = np.clip((r_y - r_h) // 2,0,H)
            y2 = np.clip((r_x + r_w) // 2,0,H)

            input_[:, :, x1:x2, y1:y2] = input_[rand_index, :, x1:x2, y1:y2]

            lambd = (1 - (x2-x1)*(y2-y1)/(W*H))

            #target = lambd * target_a + (1 - lambd) * target_b

            self.optimizer.zero_grad()
            with amp.autocast():
                inputs = input_
                #targets = target
                outputs = self.model(inputs)
                loss = lambd * self.loss_fn(outputs,target_a) + (1 - lambd) * self.loss_fn(outputs,target_b)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update( )
            final_loss += loss.item()
        return final_loss/len(data_loader) 
    
    def validate(self,data_loader):
        self.model.eval()
        final_loss = 0
        for data in data_loader:
            inputs = data['x'].to(self.device,dtype=torch.float)        
            targets = data['y'].to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs,targets)
            final_loss += loss.item()
        return final_loss/len(data_loader)
     
# Define Model
class resnet34(nn.Module):
    def __init__(self,pretrained):
        super(resnet34,self).__init__()
        if pretrained is True:
          self.model = pretrainedmodels.__dict__['resnet34'](pretrained="imagenet")
        else:
          self.model = pretrainedmodels.__dict__['resnet34'](pretrained=None)

        self.last_linear = nn.Linear(512, 193)
        self.bn = nn.BatchNorm2d(512)
        

    def forward(self, x):
        bsize, c , h, w = x.shape
        x = self.model.features(x)
        #x = self.bn(x)
        x = F.adaptive_avg_pool2d(x,1).reshape(x.data.size(0),-1)
        fn = self.last_linear(x)
        return fn    
   


class GIZ_Dataset_lr(Dataset):
    def __init__(self,audio_path,data,targets,device='cuda',transform=None):
        self.audio_path = audio_path
        self.data = data
        self.targets = targets
        self.audio_id = self.data['fn'].unique()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self,item):
        aud = self.audio_id[item]
        aud1 = aud.replace('.wav','.jpeg')
        image = Image.open(f"{self.audio_path}/{aud1}")
        image = np.array(image)

        if self.transform is not None:
          image = self.transform(image=image)["image"]
           #image = self.transform.augment(image)

        image = np.transpose(image,(2,0,1))
        label = int(self.targets[item]) 
        
        return torch.tensor(image,dtype=torch.float),torch.tensor(label)                                                                        
   
               
                








