import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch
import os
from natsort import natsorted
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from pathlib import Path
from torchvision.transforms import Resize
import numpy as np
import matplotlib.pyplot as plt
import dcp_improved

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import os

def find_lr(gt,hazy):
    epsilon = 1e-8
    gt_inverse = 1.0 / (gt + epsilon)
    l_r = gt_inverse * hazy
    return l_r

def find_lr_inverse(lr):
    epsilon = 1e-8
    l_r_inverse=1.0/(lr+epsilon)
    return l_r_inverse
    
def apply_color_map(tensor):
    # Convert tensor to numpy array
    np_tensor = tensor.cpu().numpy()
    
    # Normalize to [0, 1] range
    normalized_tensor = (np_tensor - np_tensor.min()) / (np_tensor.max() - np_tensor.min())
    
    # Apply color map
    colormap = plt.get_cmap('jet')  # You can use other colormaps if needed
    
    # Assuming tensor is [C, H, W] (3 channels, height, width)
    color_mapped_image = np.zeros((normalized_tensor.shape[1], normalized_tensor.shape[2], 3))  # [H, W, 3]
    
    for i in range(3):  # For each channel
        channel = normalized_tensor[i]
        color_mapped_channel = colormap(channel)[:, :, :3]  # Apply colormap and drop alpha channel
        color_mapped_image += color_mapped_channel
    
    # Convert to [0, 1] range as a float tensor (to maintain consistency)
    color_mapped_image = torch.tensor(color_mapped_image).permute(2, 0, 1)  # Convert to tensor and permute to [3, H, W]
    
    return color_mapped_image

def dehaze_image_lr(lr, hazy):
    epsilon = 1e-8
    lr_inverse = 1.0 / (lr + epsilon)
    recreate_image= hazy*lr_inverse
    return recreate_image
    return recreate_image

class MyData(Dataset):
    def __init__(self, path, image_size=(512,512)):
        self.filename_original =sorted(os.listdir(path+'//Hazy'), key=len) 
        self.filename_target = sorted(os.listdir(path+'//GT'), key=len)
        
        self.filename_original=natsorted(self.filename_original)
        self.filename_target=natsorted(self.filename_target)
        
        i=0
        while i<len(self.filename_original):
            self.filename_original[i]=path+'/Hazy/'+self.filename_original[i]
            self.filename_target[i]=path+'/GT/'+self.filename_target[i]
            i+=1
        
        self.image_size=image_size
    
    def __len__(self):
        return len(self.filename_original)
    
    def __getitem__(self,idx):
        filename_o=self.filename_original[idx]
        filename_t=self.filename_target[idx]
        
        
        resize=transforms.Resize(self.image_size)
        #norm=transforms.Normalize([0.5], [0.5])
        
        real=Image.open(filename_o)
        real=resize(real)
        
        condition=Image.open(filename_t)
        condition=resize(condition)
        
        real=transforms.functional.to_tensor(real) #Real=hazy
        #real=norm(real)
        condition=transforms.functional.to_tensor(condition)
        #condition=norm(condition)
        
        #l_r=find_lr(condition, real)
        #l_r_inverse=find_lr_inverse(l_r)
        #map=apply_color_map(l_r)
        
        return real, condition




class MyData_Test(Dataset):
    def __init__(self, path, image_size=(1024, 1024)):
        self.filename_original = natsorted(sorted(os.listdir(path + '/Hazy'), key=len))
        self.filename_target = natsorted(sorted(os.listdir(path + '/GT'), key=len))

        self.filename_original = [os.path.join(path, 'Hazy', fname) for fname in self.filename_original]
        self.filename_target = [os.path.join(path, 'GT', fname) for fname in self.filename_target]

        self.resize = Resize(image_size)

    def __len__(self):
        return len(self.filename_original)

    def __getitem__(self, idx):
        filename_o = self.filename_original[idx]
        filename_t = self.filename_target[idx]

        real = Image.open(filename_o).convert('RGB')  # Ensure RGB format
        condition = Image.open(filename_t).convert('RGB')  # Ensure RGB format

        real = self.resize(real)
        condition = self.resize(condition)

        real = transforms.functional.to_tensor(real)
        condition = transforms.functional.to_tensor(condition)

        return real, condition



class MyData_Whole(Dataset):
    def __init__(self, path, image_size=(512, 512)):
        self.filename_original = sorted(os.listdir(path + '//hazy'), key=len)
        self.filename_target = sorted(os.listdir(path + '//clear'), key=len)
        self.filename_transmission = sorted(os.listdir(path + '//trans'), key=len)
        
        self.filename_original = natsorted(self.filename_original)
        self.filename_target = natsorted(self.filename_target)
        self.filename_transmission = natsorted(self.filename_transmission)
        
        i = 0
        while i < len(self.filename_original):
            self.filename_original[i] = path + '/hazy/' + self.filename_original[i]
            self.filename_target[i] = path + '/clear/' + self.filename_target[i]
            self.filename_transmission[i] = path + '/trans/' + self.filename_transmission[i]
            i += 1
        
        self.image_size = image_size
    
    def __len__(self):
        return len(self.filename_original)
    
    def __getitem__(self, idx):
        filename_o = self.filename_original[idx]
        filename_t = self.filename_target[idx]
        filename_tm = self.filename_transmission[idx]
        
        resize = transforms.Resize(self.image_size)
        
        real = Image.open(filename_o)
        real = resize(real)
        
        condition = Image.open(filename_t)
        condition = resize(condition)
        
        transmission = Image.open(filename_tm)
        transmission = resize(transmission)
        
        real = transforms.functional.to_tensor(real)  # Real = hazy
        condition = transforms.functional.to_tensor(condition)
        transmission = transforms.functional.to_tensor(transmission)
        
        l_r = find_lr(condition, real)
        l_r_inverse = find_lr_inverse(l_r)
        map = apply_color_map(l_r)
        
        # Extract Airlight 'A' from the filename
        filename_wo_ext = os.path.splitext(filename_o)[0]  # Remove extension
        A_val = float(filename_wo_ext.rsplit('_', 1)[-1])  # Extract last part
        
        return real, condition, l_r, map, l_r_inverse, transmission, A_val
    



class MyData_Unsupervised(Dataset):
    def __init__(self, path, image_size=(512,512)):
        self.filename_original = sorted(os.listdir(path), key=len)
        self.filename_original = natsorted(self.filename_original)
        
        i = 0
        while i < len(self.filename_original):
            self.filename_original[i] = Path(path) / self.filename_original[i]
            i += 1
        
        self.image_size = image_size
        print(f"Initialized dataset with {len(self.filename_original)} images.")

    def __len__(self):
        return len(self.filename_original)
    
    def __getitem__(self, idx):
        filename_o = self.filename_original[idx]
        
        
        resize = transforms.Resize(self.image_size)
        
        real = Image.open(filename_o)
        # Check the number of channels
        if real.mode == 'RGBA' or real.mode == 'CMYK' or real.mode == 'LA':
            real = real.convert('RGB')
        elif real.mode == 'L':
            real = real.convert('RGB')
        real = resize(real)
        
        real = transforms.functional.to_tensor(real)  # Real = hazy
        real=real.unsqueeze(0)
        a,t=dcp_improved.get_atmospheric_light_and_transmission(real)
        real=real.squeeze(0)
        return real,a ,t

class MyData_Unsupervised1(Dataset):
    def __init__(self, path, image_size=(512,512)):
        self.filename_original = sorted(os.listdir(path), key=len)
        self.filename_original = natsorted(self.filename_original)
        
        i = 0
        while i < len(self.filename_original):
            self.filename_original[i] = Path(path) / self.filename_original[i]
            i += 1
        
        self.image_size = image_size
        print(f"Initialized dataset with {len(self.filename_original)} images.")

    def __len__(self):
        return len(self.filename_original)
    
    def __getitem__(self, idx):
        filename_o = self.filename_original[idx]
        
        
        resize = transforms.Resize(self.image_size)
        
        real = Image.open(filename_o)
        # Check the number of channels
        if real.mode == 'RGBA' or real.mode == 'CMYK' or real.mode == 'LA':
            real = real.convert('RGB')
        elif real.mode == 'L':
            real = real.convert('RGB')
        real = resize(real)
        
        real = transforms.functional.to_tensor(real)  # Real = hazy
        
        return real

class MyData_Test1(Dataset):
    def __init__(self, path):
        self.filename_original =sorted(os.listdir(path+'//Hazy'), key=len) 
        self.filename_target = sorted(os.listdir(path+'//GT'), key=len)
        
        self.filename_original=natsorted(self.filename_original)
        self.filename_target=natsorted(self.filename_target)
        
        i=0
        while i<len(self.filename_original):
            self.filename_original[i]=path+'/Hazy/'+self.filename_original[i]
            self.filename_target[i]=path+'/GT/'+self.filename_target[i]
            i+=1
        
        
    
    def __len__(self):
        return len(self.filename_original)
    
    def __getitem__(self,idx):
        filename_o=self.filename_original[idx]
        filename_t=self.filename_target[idx]
        
        
        
        
        real=Image.open(filename_o)
        
        
        condition=Image.open(filename_t)
        
        
        real=transforms.functional.to_tensor(real) #Real=hazy
        #real=norm(real)
        condition=transforms.functional.to_tensor(condition)
        #condition=norm(condition)
        
        
        return real, condition


class MyData_subset(Dataset):
    def __init__(self, path, image_size=(512, 512)):
        self.filename_original = natsorted(sorted(os.listdir(path + '/Hazy'), key=len))
        self.filename_target = natsorted(sorted(os.listdir(path + '/GT'), key=len))

        self.filename_original = [os.path.join(path, 'Hazy', fname) for fname in self.filename_original]
        self.filename_target = [os.path.join(path, 'GT', fname) for fname in self.filename_target]

        self.resize = Resize(image_size)

    def __len__(self):
        return len(self.filename_original)

    def __getitem__(self, idx):
        filename_o = self.filename_original[idx]
        filename_t = self.filename_target[idx]

        real = Image.open(filename_o).convert('RGB')  # Ensure RGB format
        condition = Image.open(filename_t).convert('RGB')  # Ensure RGB format

        real = self.resize(real)
        condition = self.resize(condition)

        real = transforms.functional.to_tensor(real)
        condition = transforms.functional.to_tensor(condition)
        
        real=real.unsqueeze(0)
        a,t=dcp_improved.get_atmospheric_light_and_transmission(real)
        real=real.squeeze(0)

        return real, condition, a,t


class MyData_Test_Single(Dataset):
    def __init__(self, path, resize_dimen=(1024,1024)):
        # Initialize the filename list first
        self.filename_original = sorted(os.listdir(path))
        self.filename_original = natsorted(self.filename_original)
        
        # Update paths
        self.filename_original = [os.path.join(path, filename) 
                                for filename in self.filename_original]
        
        # Initialize the resize transform
        self.resize = Resize(resize_dimen)
    
    def __len__(self):
        return len(self.filename_original)
    
    def __getitem__(self, idx):
        filename_o = self.filename_original[idx]
        
        # Open and process the image
        real = Image.open(filename_o).convert('RGB')
        real = self.resize(real)
        real = transforms.functional.to_tensor(real)
        
        filename_with_ext = os.path.basename(filename_o)

        
        return real, filename_with_ext  

'''
#Enter the path for input images
script_dir = Path(__file__).resolve().parent
input_path = script_dir  / '..' / 'test_input'
input_path = os.path.normpath(input_path)
dataset = MyData_Test_Single(input_path)
img, name = dataset[0]
print(img.shape)  # torch.Size([3, 1024, 1024])
print(name)       # e.g., "image1.jpg"
'''