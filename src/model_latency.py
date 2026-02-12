import torch 
import time
import os
import unsupervised_modules
from retinexmodule import dehazing_module


warmup_iterations = 50 # Set the number of warmup iterations
num_iterations = 200 # Set the number of iterations to average over



device='cuda'
Images=[(128,128), (256,256), (512,512), (1024,1024)]
torch.cuda.empty_cache()
fps_asm=[]
latency_asm=[]
fps_rtx=[]
latency_rtx=[]


#Calculate ASMICE-Net Latency
asm_model = unsupervised_modules.ASM_Module(in_ch=3)
for i in range(len(Images)):
    t1_shape=(1,3,  *Images[i])
    # Initialize the model
    model = asm_model.to(device)
    model.eval()


    #Create Loop from  here
    # Create a random input tensor

    with torch.no_grad():
        input = torch.randn(t1_shape).to(device)

        # Warm-up iterations
        for _ in range(warmup_iterations):
            model(input)

        # Number of iterations to average over
        num_iterations = num_iterations

        # Record the start time
        start_time = time.time()
        # Run the model multiple times
        for _ in range(num_iterations):
            model(input)

    # Record the end time
    end_time = time.time()
    input=input.to('cpu')
    torch.cuda.empty_cache()
    # Calculate and print the average latency
    average_latency = (end_time - start_time) / num_iterations


    # Calculate and print the FPS
    fps = 1 / average_latency
    
    latency_asm.append(average_latency)
    fps_asm .append(fps)
    del model

torch.cuda.empty_cache()
#Calculate RtxICE-Net Latency
retinex_model = dehazing_module(3)
for i in range(len(Images)):
    t1_shape=(1,3,  *Images[i])
    # Initialize the model
    model = retinex_model.to(device)
    model.eval()


    #Create Loop from  here
    # Create a random input tensor

    with torch.no_grad():
        input = torch.randn(t1_shape).to(device)

        # Warm-up iterations
        for _ in range(warmup_iterations):
            model(input)

        # Number of iterations to average over
        num_iterations = num_iterations

        # Record the start time
        start_time = time.time()
        # Run the model multiple times
        for _ in range(num_iterations):
            model(input)

    # Record the end time
    end_time = time.time()
    input=input.to('cpu')
    torch.cuda.empty_cache()
    # Calculate and print the average latency
    average_latency = (end_time - start_time) / num_iterations


    # Calculate and print the FPS
    fps = 1 / average_latency
    
    latency_rtx.append(average_latency)
    fps_rtx .append(fps)
    del model

folder_path="./output/"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    

with open('./output/latency_results.txt', 'w') as f:
    for i in range(len(Images)):
        f.write(f"Image size: {Images[i]}\n")
        f.write(f"ASMICE-Net Average Latency: {latency_asm[i]: .3f} seconds\n")
        f.write(f"ASMICE-Net FPS: {fps_asm[i]: .1f}\n")
        f.write(f"RtxICE-Net Average Latency: {latency_rtx[i]: .3f} seconds\n")
        f.write(f"RtxICE-Net FPS: {fps_rtx[i]: .1f}\n\n")
    f.close()