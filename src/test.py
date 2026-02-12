import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import os
import torch
import matplotlib.pyplot as plt
import torchvision
import time
import torch.nn.functional as F
import unsupervised_modules
import torch.nn as nn
from retinexmodule import dehazing_module
from dataloader import MyData_Test_Single


# Get the directory where the script is located
script_dir = Path(__file__).resolve().parent

#Select size of the input images. Default is 1024x1024
Image_size=(1024,1024)

#Enter the path for input images
input_path = script_dir  / '..' / 'test_input'
input_path = os.path.normpath(input_path) 

#Enter path for output images
folder_path="./output/"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

#Enter path for output images
folder_path="./output/J_RtxICE-Net"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

#Enter path for output images
folder_path="./output/J_ASMICE-Net"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

#Enter path for output images
folder_path="./output/trm_ASMICE-Net"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

J_asm_path = "./output/J_ASMICE-Net"
J_ret_path = "./output/J_RtxICE-Net"
t_map_path = "./output/trm_ASMICE-Net"

#Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path='./models/final_dehazing_model_10.pth'  
asm_model=unsupervised_modules.ASM_Module(in_ch=3)
retinex_model=dehazing_module(3)
combined_model=unsupervised_modules.CombinedModel(asm_model, retinex_model)
#combined_model.load_state_dict(torch.load(model_path))
# Load weights mapped to the current device (CPU or GPU)
combined_model.load_state_dict(torch.load(model_path, map_location=device))
combined_model.to(device)
combined_model.eval()

#Load input images
test_data=MyData_Test_Single(input_path, Image_size)
test_data_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

# Process images
for hazy_img, filename in test_data_dataloader:
    with torch.no_grad():
        hazy_img = hazy_img.to(device)
        J_asm, t_map, a, lr_inv = combined_model(hazy_img)
        J_ret = hazy_img * lr_inv
        
        # Clamp the images for saving
        J_ret = J_ret.clamp(0, 1)
        J_asm = J_asm.clamp(0, 1)
        t_map_3ch_clamped = t_map.repeat(1, 3, 1, 1).clamp(0, 1)  # Repeat the single channel to create a 3-channel image
        
        fname = filename[0] 
        
        torchvision.utils.save_image(J_ret, os.path.join(J_ret_path, fname))
        torchvision.utils.save_image(J_asm, os.path.join(J_asm_path, fname))
        torchvision.utils.save_image(t_map_3ch_clamped, os.path.join(t_map_path, fname))

        print(f'Image Processed:{fname}')