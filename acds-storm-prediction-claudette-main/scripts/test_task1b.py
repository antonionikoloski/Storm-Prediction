import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.data_loader import StormDatasetTask1BPreload 
from utils.data_loader_1b_1 import StormDatasetTask1BPreload_Production 
from models.task1b_model import Task1bConvLSTM  


def visualize_batch(batch_x, batch_y, out_dir, batch_idx=0, num_samples=4):
    """
    Visualizes and saves a batch of input and ground truth images.

    Parameters:
    - batch_x (torch.Tensor): Input tensor of shape (B, C, T_in, H, W)
    - batch_y (torch.Tensor): Ground truth tensor of shape (B, T_out, H, W)
    - out_dir (str): Directory to save the visualization
    - batch_idx (int): Index of the batch (used in the filename)
    - num_samples (int): Number of samples from the batch to visualize
    """
    os.makedirs(out_dir, exist_ok=True)
    

    batch_x = batch_x.cpu().numpy()
    batch_y = batch_y.cpu().numpy()
    
    B, C, T_in, H, W = batch_x.shape
    _, T_out, _, _ = batch_y.shape
    

    num_samples = min(num_samples, B)
    
    for i in range(num_samples):
        vil_channel_idx = 3
        if C <= vil_channel_idx:
            print(f"[ERROR] Expected at least {vil_channel_idx + 1} channels, but got {C}.")
            continue
        

        num_cols = T_in + T_out
        fig, axs = plt.subplots(2, num_cols, figsize=(3*num_cols, 6))
        fig.suptitle(f"Batch {batch_idx}, Sample {i} - VIL Channel", fontsize=16)
        

        for t in range(T_in):
            axs[0, t].imshow(batch_x[i, vil_channel_idx, t], cmap='viridis', origin='upper')
            axs[0, t].set_title(f"Input T{t}")
            axs[0, t].axis("off")
        

        for t in range(T_out):
            axs[1, t].imshow(batch_y[i, t], cmap='viridis', origin='upper')
            axs[1, t].set_title(f"GT T{t}")
            axs[1, t].axis("off")
        

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
        plot_path = os.path.join(out_dir, f"batch_{batch_idx}_sample_{i}_vil.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[INFO] Saved VIL visualization at: {plot_path}")




def read_ids(txt_path):
    with open(txt_path, 'r') as f:
        ids = [line.strip() for line in f.readlines()]
    return ids
def visualize_prediction(pred, gt, out_dir, sample_idx=0, storm_id=None):
    """
    pred, gt: shape (T, H, W) each 
    out_dir: directory to save the plot
    sample_idx: index of the sample in the dataset
    storm_id: optional label to show in the title
    """
    os.makedirs(out_dir, exist_ok=True)
    T, H, W = pred.shape

    fig, axs = plt.subplots(2, T, figsize=(3*T, 6))

    for t in range(T):

        axs[0, t].imshow(gt[t], cmap='gray', origin='upper')
        axs[0, t].set_title(f"GT (t={t})")
        axs[0, t].axis("off")


        axs[1, t].imshow(pred[t], cmap='gray', origin='upper')
        axs[1, t].set_title(f"Pred (t={t})")
        axs[1, t].axis("off")

    title_str = f"Storm: {storm_id}" if storm_id else "Comparison of GT and Prediction"
    plt.suptitle(title_str)


    plot_path = os.path.join(out_dir, f"sample_{sample_idx}_prediction.png")
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved comparison plot at: {plot_path}")


def predict_task1b(cfg_path='configs/task1b.yaml'):

    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = cfg['training']['device']
    test_ids = read_ids(cfg['data']['test_ids_file'])
    test_dataset = StormDatasetTask1BPreload_Production(
        csv_path=cfg['data']['csv_path'],
        h5_path=cfg['data']['h5_path'],
        storm_ids=test_ids,  # pass the list of IDs
        in_frames=cfg['data']['in_frames'],
        out_frames=cfg['data']['out_frames'],
        stride=cfg['data']['stride'],
        resize_to=tuple(cfg['data']['resize_to']),
        do_normalize=cfg['data']['do_normalize']
    )
       
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    visualization_dir = 'visualizations/train'
    try:
        batch_x, batch_y = next(iter(test_loader))
        visualize_batch(batch_x, batch_y, out_dir=visualization_dir, batch_idx=0, num_samples=4)
    except Exception as e:
        print(f"[ERROR] Failed to visualize training batch: {e}")

    print(f"[INFO] Test dataset loaded with {len(test_dataset)} samples")
    model = Task1bConvLSTM(
        in_channels=cfg['model']['in_channels'],   
        in_time=cfg['data']['in_frames'],        
        out_frames=cfg['data']['out_frames']      
    ).to(device)


    print(f"[INFO] Loading model from {cfg['training']['ckpt_dir']}")
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
        for idx, (batch_x, batch_y) in enumerate(test_loader):
            print("shape of batch_x", batch_x.shape)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            pred = model(batch_x)  
            predictions.append(pred.squeeze(0).cpu().numpy()) 
            gts.append(batch_y.squeeze(0).cpu().numpy())
            storm_id = test_ids[idx]
            pred =pred.squeeze(0).cpu().numpy()
            gt = batch_y.squeeze(0).cpu().numpy()
            pred =np.transpose(pred, (1, 2, 0))
            gt = np.transpose(gt, (1, 2, 0))
            out_dir = cfg['inference']['output_dir']
            pred_filename = f"{storm_id}_pred.npy"
            gt_filename = f"{storm_id}_gt.npy"
            pred_path = os.path.join(out_dir, pred_filename)
            gt_path = os.path.join(out_dir, gt_filename)
            np.save(pred_path, pred)
            print(f"[INFO] Saved prediction to {pred_path}")


    print(len(predictions), len(gts))
    out_dir = cfg['inference']['output_dir']
    os.makedirs(out_dir, exist_ok=True)

    preds_path = os.path.join(out_dir, 'predictions.npy')
    gts_path   = os.path.join(out_dir, 'groundtruths.npy')
    np.save(preds_path, predictions)
    np.save(gts_path,   gts)
    print(f"[INFO] Saved predictions to {preds_path}")
    print(f"[INFO] Saved ground truths to {gts_path}")


if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/task1b.yaml'
    predict_task1b(cfg_path)