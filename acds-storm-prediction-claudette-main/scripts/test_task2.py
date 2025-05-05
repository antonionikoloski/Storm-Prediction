import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.metrics import mse, mae 
from models.task2_model import ConvLSTMForecast
from utils.data_loader_2 import StormDataset
from utils.data_loader_2_1 import StormDataset_Production

def read_ids(txt_path):
    """Read storm IDs from text file"""
    with open(txt_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def visualize_predictions(batch_x, batch_y, predictions, out_dir, batch_idx=0, num_samples=4):
    """Visualization function remains unchanged"""


def test_task2(cfg_path='configs/task2.yaml'):
    """Main testing function"""
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    test_ids = read_ids(cfg['data']['test_ids_file'])
    print(f"Testing on {len(test_ids)} storms")

    test_dataset = StormDataset_Production(
        csv_path=cfg['data']['csv_path'],
        h5_path=cfg['data']['h5_path'],
        storm_ids=test_ids,
        in_frames=cfg['data']['in_frames'],
        out_frames=cfg['data']['out_frames'],
        stride=cfg['data']['stride'],
        resize_to=tuple(cfg['data']['resize_to']),
        do_normalize=cfg['data']['do_normalize']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=False
    )

    model = ConvLSTMForecast(
        in_channels=cfg['model']['in_channels'],
        hidden_dim=cfg['model']['hidden_dim'],
        height=cfg['model']['height'],
        width=cfg['model']['width']
    ).to(cfg['training']['device'])

    ckpt_path = os.path.join(cfg['training']['ckpt_dir'], 'best_model.pt')
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))
    model.eval()

    all_metrics = {'mse': [], 'mae': []}
    visualization_dir = os.path.join(cfg['inference']['output_dir'], 'visualizations')
    all_predictions = []
    all_gts = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader)):
            inputs = inputs.to(cfg['training']['device'])
            targets = targets.to(cfg['training']['device'])
            
            predictions = model(inputs)
            

            batch_size = inputs.size(0)
            for i in range(batch_size):
                sample_idx = batch_idx * cfg['training']['batch_size'] + i
                if sample_idx >= len(test_ids):
                    break
                
                storm_id = test_ids[sample_idx]
                
                
                pred = predictions[i].cpu().numpy() 
                pred = np.transpose(pred, (1, 2, 0))
                
                gt = targets[i].cpu().numpy()       
                gt = np.transpose(gt, (1, 2, 0))    
                

                out_dir = cfg['inference']['output_dir']
                pred_path = os.path.join(out_dir, f"{storm_id}_pred222.npy")
                gt_path = os.path.join(out_dir, f"{storm_id}_gt.npy")
                np.save(pred_path, pred)
                np.save(gt_path, gt)

            all_predictions.append(predictions.cpu().numpy())
            all_gts.append(targets.cpu().numpy())
            batch_mse = mse(predictions, targets)
            batch_mae = mae(predictions, targets)
            all_metrics['mse'].append(batch_mse.item())
            all_metrics['mae'].append(batch_mae.item())

            if batch_idx == 0:
                visualize_predictions(
                    inputs, targets, predictions,
                    out_dir=visualization_dir,
                    batch_idx=batch_idx,
                    num_samples=4
                )

    predictions = np.concatenate(all_predictions, axis=0)
    gts = np.concatenate(all_gts, axis=0)
    out_dir = cfg['inference']['output_dir']
    os.makedirs(out_dir, exist_ok=True)
    
    np.save(os.path.join(out_dir, 'predictions.npy'), predictions)
    np.save(os.path.join(out_dir, 'groundtruths.npy'), gts)

    final_metrics = {metric: np.mean(values) for metric, values in all_metrics.items()}
    metrics_path = os.path.join(out_dir, 'test_metrics.txt')
    with open(metrics_path, 'w') as f:
        for metric, value in final_metrics.items():
            f.write(f"{metric.upper()}: {value:.4f}\n")

    print("\nTest Results:")
    for metric, value in final_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/task2.yaml'
    test_task2(config_path)