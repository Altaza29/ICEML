import torch
from torchvision import transforms
from PIL import Image
import piq
import os
import illumination_module  # Make sure this is correctly defined elsewhere
import transmission_map
import dehazing_module

# Initialize the model
#model = illumination_module.ill_gen(64, 5)  # Replace with your model class
#model=transmission_map.Generator(3,3)
model=dehazing_module.dehazing_module(in_channels=3)
model_path = './models/dehazing_model_1_dehazing_module.pth'
model.load_state_dict(torch.load(model_path))  # Replace with your model path
model.to('cuda')
model.eval()

# Define a transform to resize the image to 512x512 and convert it to a tensor
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize the image to 512x512
    transforms.ToTensor()           # Convert the image to a PyTorch tensor
])

# Path to the directory containing images
input_dir = './Test/'
output_dir = './Test_Result_Dehazing_mod/'

# Create output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Process each image in the directory
for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
        # Construct the full file path
        image_path = os.path.join(input_dir, filename)
        
        # Load the image using PIL and convert it to RGB format
        image = Image.open(image_path).convert('RGB')
        
        # Apply the transform to the image to convert it into a tensor
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0).to('cuda')
        
        # Perform dehazing
        with torch.no_grad():
            lr_inverse = model(image_tensor)
            dehazed_image = image_tensor * lr_inverse
        
        # Squeeze the batch dimension
        dehazed_image = dehazed_image.squeeze(0)
        
        # Clamp the values of the dehazed image to the range [0, 1]
        dehazed_image_clamped = torch.clamp(dehazed_image, 0.0, 1.0)
        
        # Move the tensor to the CPU and convert it to a PIL image for saving
        dehazed_image_clamped_cpu = dehazed_image_clamped.cpu()
        dehazed_image_pil = transforms.ToPILImage()(dehazed_image_clamped_cpu)
        
        # Save the dehazed image
        output_path = os.path.join(output_dir, filename)
        dehazed_image_pil.save(output_path)
        
        print(f"Processed and saved dehazed image: {output_path}")