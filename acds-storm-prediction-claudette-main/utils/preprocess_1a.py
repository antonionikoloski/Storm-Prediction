import numpy as np
import cv2

def preprocess_channels(vil_data, resize_to=(384, 384), do_normalize=True):
    """
    Preprocess the 'vil' channel data.

    Args:
        vil_data (np.ndarray): Array of shape (T, H, W).
        resize_to (tuple): Desired (height, width).
        do_normalize (bool): Whether to normalize the data.

    Returns:
        np.ndarray: Preprocessed 'vil' data of shape (T, H_resized, W_resized).
    """
    T, H, W = vil_data.shape
    processed = []

    for t in range(T):
        frame = vil_data[t]
        if do_normalize:
            frame_resized = frame_resized.astype(np.float32)
            frame_resized = (frame_resized - np.mean(frame_resized)) / (np.std(frame_resized) + 1e-5)

        processed.append(frame_resized)

    processed = np.stack(processed, axis=0) 
    return processed


def preprocess_vil_no_resize(vil_data, do_normalize=True):
    """
    Preprocess the 'vil' channel data without resizing.

    Args:
        vil_data (np.ndarray): Array of shape (T, H, W).
        do_normalize (bool): Whether to normalize the data.

    Returns:
        np.ndarray: Preprocessed 'vil' data of shape (T, H, W).
    """
    T, H, W = vil_data.shape
    processed = []

    for t in range(T):
        frame = vil_data[t]

        # No resizing
        frame_resized = frame  # Keep original size

        if do_normalize:
            frame_resized = frame_resized.astype(np.float32)
            frame_resized = (frame_resized - np.mean(frame_resized)) / (np.std(frame_resized) + 1e-5)

        processed.append(frame_resized)

    processed = np.stack(processed, axis=0)  # (T, H, W)
    return processed

