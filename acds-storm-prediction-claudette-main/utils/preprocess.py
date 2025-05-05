import numpy as np
import cv2  
def normalize_channel(channel_data, min_val=None, max_val=None):
    """
    Normalize channel data to [0, 1] (or any custom range).
    channel_data: np.array of shape (H, W, T)
    """
    if min_val is None:
        min_val = channel_data.min()
    if max_val is None:
        max_val = channel_data.max()
    if max_val - min_val < 1e-5:
        return np.zeros_like(channel_data)
    normed = (channel_data - min_val) / (max_val - min_val)
    return normed

def resize_channel(channel_data, target_shape=(384, 384)):
    """
    Resize each frame in time dimension to target_shape (H, W).
    channel_data: np.array of shape (H, W, T)
    target_shape: desired (H, W)
    Returns: np.array of shape (target_H, target_W, T)
    """
    T = channel_data.shape[2]
    resized_frames = []
    for t in range(T):
        frame_resized = cv2.resize(channel_data[..., t], target_shape, interpolation=cv2.INTER_LINEAR)
        resized_frames.append(frame_resized[..., np.newaxis])
    return np.concatenate(resized_frames, axis=2)

def preprocess_channels(vis, ir069, ir107, vil, 
                        resize_to=(384, 384), 
                        do_normalize=True):
    """
    Example function that resizes all channels to 192x192 
    and normalizes them. 
    Returns a single NumPy array of shape (4, T, 192, 192), 
    i.e. (channels, time, height, width).
    """
    ir069_resized= resize_channel(ir069,resize_to)
    ir107_resized= resize_channel(ir107,resize_to)

    if do_normalize:
        vis_norm   = normalize_channel(vis)
        ir069_norm = normalize_channel(ir069_resized)
        ir107_norm = normalize_channel(ir107_resized)
        vil_norm   = normalize_channel(vil)
    else:
        vis_norm, ir069_norm, ir107_norm, vil_norm = (vis, ir069_resized, 
                                                      ir107_resized, vil)

    vis_final   = np.transpose(vis_norm,   (2,0,1))
    ir069_final = np.transpose(ir069_norm, (2,0,1))
    ir107_final = np.transpose(ir107_norm, (2,0,1))
    vil_final   = np.transpose(vil_norm,   (2,0,1))

    combined = np.stack([vis_final, ir069_final, ir107_final, vil_final], axis=0)
    return combined
def preprocess_vil(vil, resize_to=(384, 384), do_normalize=True):
    """
    Processes the VIL channel by resizing, normalizing, and reshaping it.
    Returns a NumPy array of shape (1, T, H, W), i.e., (channels, time, height, width).
    """

    
    if do_normalize:
        vil_norm = normalize_channel(vil)
    else:
        vil_norm = vil
    
    vil_final = np.transpose(vil_norm, (2, 0, 1))

    combined = np.expand_dims(vil_final, axis=0)
    
    return combined

def denormalize(predictions, mins, maxs):
    """Convert normalized predictions back to original scale."""
    return predictions * (maxs - mins) + mins

