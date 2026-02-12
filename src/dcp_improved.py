import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    """Load an image and normalize it to [0,1]."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # Convert to Bx3xHxW
    return image

def show_images(original, dehazed, transmission):
    """Display original, dehazed, and transmission map."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(original)
    axs[0].set_title("Hazy Image")
    axs[0].axis("off")
    
    axs[1].imshow(dehazed)
    axs[1].set_title("Dehazed Image")
    axs[1].axis("off")
    
    axs[2].imshow(transmission, cmap='gray')
    axs[2].set_title("Transmission Map")
    axs[2].axis("off")
    plt.show()

def dark_channel(image, kernel_size=15):
    """Differentiable dark channel computation."""
    r, g, b = image[:, 0:1, :, :], image[:, 1:2, :, :], image[:, 2:3, :, :]
    dark = torch.min(torch.min(r, g), b)
    dark = -F.max_pool2d(-dark, kernel_size, stride=1, padding=kernel_size // 2)
    return dark

def estimate_atmospheric_light(image, dark_channel, top_percent=0.001, threshold=0.75):
    """Estimate atmospheric light A while removing sky regions."""
    b, c, h, w = image.shape
    num_pixels = max(int(h * w * top_percent), 1)
    dark_flat = dark_channel.view(b, -1)
    image_flat = image.permute(0, 2, 3, 1).view(b, -1, 3)
    
    # Filter out pixels where the dark channel value is > 0.7 (sky region)
    valid_indices = dark_flat < threshold
    dark_filtered = dark_flat * valid_indices  # Zero out invalid pixels
    
    top_indices = torch.argsort(dark_filtered, dim=1, descending=True)[:, :num_pixels]
    A = torch.gather(image_flat, 1, top_indices.unsqueeze(-1).expand(-1, -1, 3)).mean(dim=1)
    return A.view(b, 3, 1, 1)

def transmission_map(image, A, kernel_size=15, omega=0.95):
    """Compute transmission map."""
    norm_image = image / A
    t = 1 - omega * dark_channel(norm_image, kernel_size)
    return t

def recover_scene_radiance(image, t, A, t_min=0.1):
    """Recover dehazed image."""
    t = torch.clamp(t, min=t_min)
    J = (image - A) / t + A
    return torch.clamp(J, 0, 1)

def dehaze_image(image_path):
    image = load_image(image_path)
    dark = dark_channel(image)
    A = estimate_atmospheric_light(image, dark)
    t = transmission_map(image, A)
    dehazed = recover_scene_radiance(image, t, A)
    
    original = image.squeeze(0).permute(1, 2, 0).numpy()
    dehazed = dehazed.squeeze(0).permute(1, 2, 0).detach().numpy()
    transmission = t.squeeze(0).squeeze(0).detach().numpy()
    
    show_images(original, dehazed, transmission)

def get_atmospheric_light_and_transmission(image):
    """Compute atmospheric light (A) and transmission map (t) from a hazy image tensor."""
    dark = dark_channel(image)
    A = estimate_atmospheric_light(image, dark)
    t = transmission_map(image, A)
    b, c, h, w = image.shape
    A_expanded = A.expand(b, c, h, w)  # Expand A to the shape Bx3xWxH
    A_expanded=A_expanded.squeeze(0)
    t=t.squeeze(0)
    return A_expanded, t

def get_transmission(image):
    """Compute atmospheric light (A) and transmission map (t) from a hazy image tensor."""
    dark = dark_channel(image)
    A = estimate_atmospheric_light(image, dark)
    t = transmission_map(image, A)
    return t
