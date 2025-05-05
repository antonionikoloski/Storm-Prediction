import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils.data_loader_1a import Task2Dataset
from utils.data_loader_1a_1 import Task2Dataset_Production
from models.task1a_model import EnhancedVILPredictor
from utils.metrics import mse  # Ensure you have this function defined

def read_ids(txt_path):
    """
    Reads a list of IDs from a text file.

    Args:
        txt_path (str): Path to the text file containing IDs.

    Returns:
        list: List of IDs as strings.
    """
    with open(txt_path, 'r') as f:
        ids = [line.strip() for line in f.readlines()]
    return ids

def visualize_prediction(pred, gt, out_dir, sample_idx=0, storm_id=None):
    """
    Visualizes and saves a comparison between prediction and ground truth.

    Args:
        pred (np.ndarray): Predicted 'vil' frames of shape (out_frames, H, W)
        gt (np.ndarray): Ground truth 'vil' frames of shape (out_frames, H, W)
        out_dir (str): Directory to save the visualization
        sample_idx (int): Index of the sample in the dataset
        storm_id (str, optional): Identifier for the storm
    """
    os.makedirs(out_dir, exist_ok=True)
    T, H, W = pred.shape

    fig, axs = plt.subplots(2, T, figsize=(3*T, 6))

    for t in range(T):
        axs[0, t].imshow(gt[t], cmap='viridis', origin='upper')
        axs[0, t].set_title(f"GT (t={t})")
        axs[0, t].axis("off")


        axs[1, t].imshow(pred[t], cmap='viridis', origin='upper')
        axs[1, t].set_title(f"Pred (t={t})")
        axs[1, t].axis("off")

    title_str = f"Storm: {storm_id}" if storm_id else "Comparison of GT and Prediction"
    plt.suptitle(title_str)

  
    plot_path = os.path.join(out_dir, f"sample_{sample_idx}_prediction.png")
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved comparison plot at: {plot_path}")

def predict_task2(cfg_path='configs/task2.yaml'):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = cfg['training']['device']

    test_ids = read_ids(cfg['data']['test_ids_file'])
    print(f"Running inference on {len(test_ids)} storms.")
    print(f"First storm ID: {test_ids[0]}")

    test_dataset = Task2Dataset_Production(
        csv_path=cfg['data']['csv_path'],
        h5_path=cfg['data']['h5_path'],
        storm_ids=test_ids,
        in_frames=cfg['data']['in_frames'],
        out_frames=cfg['data']['out_frames'],
        stride=1, 
        resize_to=tuple(cfg['data']['resize_to']),
        do_normalize=cfg['data']['do_normalize']
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


    model = EnhancedVILPredictor(
        input_channels=cfg['model']['in_channels'],
        hidden_dim=cfg['model']['hidden_dim'],
        out_frames=cfg['data']['out_frames']
    ).to(device)


    ckpt_path = os.path.join(cfg['training']['ckpt_dir'], 'best_model.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"[INFO] Loaded model from {ckpt_path}")


    predictions = []
    gts = []
    storm_ids_list = [] 

    with torch.no_grad():
        for idx, (inputs, targets, mins, maxs) in enumerate(test_loader):
            inputs = inputs.to(device)  
            targets = targets.to(device)  
            mins = mins.to(device)       
            maxs = maxs.to(device)       
            outputs = model(inputs)      

            outputs = outputs.squeeze(2)  
            outputs_denorm = outputs * (maxs - mins).view(-1, 1, 1, 1) + mins.view(-1, 1, 1, 1)
            targets_denorm = targets * (maxs - mins).view(-1, 1, 1, 1) + mins.view(-1, 1, 1, 1)
            pred = outputs_denorm.cpu().numpy().squeeze()      
            gt = targets_denorm.cpu().numpy().squeeze()      
            predictions.append(pred)
            gts.append(gt)
            storm_id = test_ids[idx]
            
            pred =np.transpose(pred, (1, 2, 0))
            gt = np.transpose(gt, (1, 2, 0))
            out_dir = cfg['inference']['output_dir']
            pred_filename = f"{storm_id}_pred.npy"
            gt_filename = f"{storm_id}_gt.npy"
            pred_path = os.path.join(out_dir, pred_filename)
            gt_path = os.path.join(out_dir, gt_filename)


            np.save(pred_path, pred)
            np.save(gt_path, gt)
            print("ovde sum")
            print("shape of the prediction",np.array(pred).shape)
            print(f"[INFO] Saved prediction to {pred_path}")
            print(f"[INFO] Saved ground truth to {gt_path}")

    out_dir = cfg['inference']['output_dir']
    os.makedirs(out_dir, exist_ok=True)
    print("shape of the predictions",np.array(predictions).shape)
    print("shape of the ground truths",np.array(gts).shape)
    preds_path = os.path.join(out_dir, 'predictions.npy')
    gts_path   = os.path.join(out_dir, 'groundtruths.npy')
    np.save(preds_path, predictions)
    np.save(gts_path, gts)
    print(f"[INFO] Saved predictions to {preds_path}")
    print(f"[INFO] Saved ground truths to {gts_path}")

if __name__ == "__main__":
    import sys

    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/task2.yaml'
    predict_task2(cfg_path)
