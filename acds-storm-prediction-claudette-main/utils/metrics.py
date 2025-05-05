import torch
import torch.nn.functional as F
import torch.nn as nn

def mse(pred, target):
    return F.mse_loss(pred, target)

# If you need structural similarity, you can use 
# a library like piqa or pytorch-msssim, or define your own function.
from torchmetrics import StructuralSimilarityIndexMeasure

ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)

def compute_ssim(pred, target):
    return ssim_metric(pred, target)

def probabilistic_loss(p_zero, mu_nonzero, sigma_nonzero, targets):
    is_zero = targets[:, 0].float()
    is_nonzero = 1 - is_zero

    # Binary Cross-Entropy Loss for p_zero
    bce_loss = nn.functional.binary_cross_entropy(p_zero, is_zero)

    # Gaussian Negative Log-Likelihood for non-zero values
    var = sigma_nonzero ** 2
    nll = 0.5 * (torch.log(2 * torch.pi * var) + ((targets[:, 1] - mu_nonzero) ** 2) / var)
    nll_loss = (is_nonzero * nll).mean()

    return bce_loss + nll_loss

def chamfer_loss(array1, array2):
    """
    Compute the bi-directional Chamfer loss between two point clouds (array1 and array2).

    Parameters:
        array1 (torch.Tensor): Tensor of shape [n1, 3], the first set of 3D vectors.
        array2 (torch.Tensor): Tensor of shape [n2, 3], the second set of 3D vectors.

    Returns:
        torch.Tensor: The bi-directional Chamfer loss (a scalar).
    """
    # Compute pairwise distances
    diff_1_to_2 = torch.cdist(array1, array2, p=2)  # Shape: [n1, n2]
    diff_2_to_1 = torch.cdist(array2, array1, p=2)  # Shape: [n2, n1]

    # Compute the forward Chamfer distance
    forward_loss = torch.mean(torch.min(diff_1_to_2, dim=1).values)  # Min over array2, mean over array1

    # Compute the backward Chamfer distance
    backward_loss = torch.mean(torch.min(diff_2_to_1, dim=1).values)  # Min over array1, mean over array2

    # Total Chamfer loss (bi-directional)
    total_loss = forward_loss + backward_loss

    return total_loss

def mae(preds, targets):
    """Calculate Mean Absolute Error (MAE)"""
    return torch.mean(torch.abs(preds - targets))

def ssim(preds, targets, data_range=1.0, window_size=11, size_average=True):
    """
    Updated SSIM implementation that handles multiple channels
    """
    if len(preds.size()) != 4:
        raise ValueError("Input tensors must be 4D (batch, channels, height, width)")
        

    total_ssim = 0
    num_channels = preds.size(1)
    

    window = _create_window(window_size, preds.device, preds.dtype)
    pad = window_size // 2


    for channel in range(num_channels):
        pred_chan = preds[:, channel:channel+1, :, :]
        target_chan = targets[:, channel:channel+1, :, :]

        mu1 = F.conv2d(pred_chan, window, padding=pad, groups=1)
        mu2 = F.conv2d(target_chan, window, padding=pad, groups=1)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred_chan * pred_chan, window, padding=pad, groups=1) - mu1_sq
        sigma2_sq = F.conv2d(target_chan * target_chan, window, padding=pad, groups=1) - mu2_sq
        sigma12 = F.conv2d(pred_chan * target_chan, window, padding=pad, groups=1) - mu1_mu2

        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                  ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        total_ssim += ssim_map.mean()
    
    return total_ssim / num_channels

def _create_window(window_size, device, dtype):
    """Create 1D Gaussian window for SSIM calculation"""
    def gaussian(window_size, sigma):
        gauss = torch.exp(-(torch.arange(window_size) - window_size//2)**2 / float(2*sigma**2))
        return gauss/gauss.sum()
    
    sigma = 1.5  
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(1, 1, window_size, window_size).to(device=device, dtype=dtype)
    return window


ssim_index_measure = ssim
