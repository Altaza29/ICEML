import torch

def dehaze_image(a, t_map, hazy, t_min_tensor=0.1):
    """
    Dehazes the input hazy image using the provided airlight (a) and transmission map (t_map).

    Parameters:
    a (torch.Tensor): The airlight tensor.
    t_map (torch.Tensor): The transmission map tensor.
    hazy (torch.Tensor): The hazy image tensor.
    t_min_tensor (float): The minimum value for clamping the transmission map. Default is 0.1.

    Returns:
    torch.Tensor: The dehazed image tensor.
    """
    step1 = hazy - a
    step2 = step1 / t_map.clamp(min=t_min_tensor)
    dehazed_image = step2 + a
    return dehazed_image

def haze_image(clean, a, t_map, t_min_tensor=0.1):
    """
    Generates a hazy image from the clean image using the provided airlight (a) and transmission map (t_map).

    Parameters:
    clean (torch.Tensor): The clean image tensor.
    a (torch.Tensor): The airlight tensor.
    t_map (torch.Tensor): The transmission map tensor.
    t_min_tensor (float): The minimum value for clamping the transmission map. Default is 0.1.

    Returns:
    torch.Tensor: The hazy image tensor.
    """
    t_map_clamped = t_map.clamp(min=t_min_tensor)
    hazy_image = clean * t_map_clamped + a * (1 - t_map_clamped)
    return hazy_image