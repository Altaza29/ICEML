import os
import torch
import unsupervised_modules
from retinexmodule import dehazing_module

# Create models
asm_model = unsupervised_modules.ASM_Module(in_ch=3)
retinex_model = dehazing_module(3)
combined_model = unsupervised_modules.CombinedModel(asm_model, retinex_model)

# Output folder
folder_path = "./output/"
os.makedirs(folder_path, exist_ok=True)

def model_info(model):
    # Number of parameters (in millions)
    num_params_million = sum(p.numel() for p in model.parameters()) / 1e6
    
    # Model size (in MB)
    param_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model_size_mb = param_size_bytes / (1024 ** 2)
    
    return num_params_million, model_size_mb

# Get info for each model
asm_params, asm_size = model_info(asm_model)
retinex_params, retinex_size = model_info(retinex_model)
combined_params, combined_size = model_info(combined_model)

# Prepare output string with 3 decimal places
output_str = (
    f"ASMICE-Net: {asm_params:.3f} Million params, {asm_size:.3f} MB\n"
    f"RtxICE-Net: {retinex_params:.3f} Million params, {retinex_size:.3f} MB\n"
)

# Save to text file
output_file = os.path.join(folder_path, "model_info.txt")
with open(output_file, "w") as f:
    f.write(output_str)

print(f"Model info saved to {output_file}")
print(output_str)
