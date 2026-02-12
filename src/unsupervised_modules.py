import torch
import torch.nn as nn
import dehazing_module
import transmission_map
import retinexmodule


class retinex_module(nn.Module):
    def __init__(self, in_ch=3):
        super(retinex_module, self).__init__()
        
        self.retinex=retinexmodule.dehazing_module(in_channels=in_ch)

    def forward(self, hazy):
        dehazed=self.retinex(hazy)
        return dehazed


class transmission_module(nn.Module):
    def __init__(self, in_ch=3):
        super(transmission_module, self).__init__()
        self.transmission=dehazing_module.autonext_module_sigmoid(in_channels=in_ch)
        #self.transmission=dehazing_module.autonext_module_sigmoid(in_channels=in_ch)
        #self.transmission=transmission_map.Generator_tmap(in_channels=in_ch)
    def forward(self, hazy):
        transmission=self.transmission(hazy)
        return transmission


class airlight_module(nn.Module):
    def __init__(self, in_ch=3):
        super(airlight_module, self).__init__()
        
        
        #self.airlight=transmission_map.Generator_tmap(in_channels=in_ch, out_channels=3)
        self.airlight=dehazing_module.autonext_module_sigmoid(in_channels=in_ch, out_ch=3)
        
    def forward(self, hazy):
        airlight=self.airlight(hazy)
        return airlight

class ASM_Module(nn.Module):
    def __init__(self, in_ch=3):
        super(ASM_Module, self).__init__()
        
        self.transmission = transmission_module(in_ch=3)
        self.airlight = airlight_module(in_ch=3)
        self.t_min = 0.1
        self.t_min_tensor = torch.tensor(self.t_min, dtype=torch.float32)  # Store tensor once

    def forward(self, hazy):
        t_min_tensor = self.t_min_tensor.to(hazy.device)
        t_map = self.transmission(hazy)  # (B, 1, W, H)
        a = self.airlight(hazy)  # (B, 1)
        
        step1=hazy-a
        step2=step1/t_map.clamp(min=t_min_tensor)
        dehazed_image=step2+a
        
        return dehazed_image, t_map, a

class CombinedModel(nn.Module):
    def __init__(self, asm_model, retinex_model):
        super(CombinedModel, self).__init__()
        self.asm_model = asm_model
        self.retinex_model = retinex_model

    def forward(self, x):
        asm_dehazed, t_map, a = self.asm_model(x)
        lr_inv = self.retinex_model(x)
        return asm_dehazed, t_map, a, lr_inv


