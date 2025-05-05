# test.py
import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.metrics import probabilistic_loss
from models.task3_model import ProbabilisticCNN_3
from utils.data_loader_3 import CustomImageDataset

def read_ids(txt_path):
    """Read storm IDs from text file"""
    with open(txt_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

# def visualize_predictions(batch_x, batch_y, predictions, out_dir, batch_idx=0, num_samples=4):
#     """
#     Visualizes and saves predictions alongside inputs and ground truth
#     """
#     os.makedirs(out_dir, exist_ok=True)
    
#     # Convert tensors to numpy arrays
#     batch_x = batch_x.cpu().numpy()
#     batch_y = batch_y.cpu().numpy()
#     predictions = predictions.cpu().numpy()
    
#     B, C, T_in, H, W = batch_x.shape
#     _, T_out, _, _ = batch_y.shape
    
#     num_samples = min(num_samples, B)
    
#     for i in range(num_samples):
#         fig, axs = plt.subplots(3, max(T_in, T_out), figsize=(3*max(T_in,T_out), 9))
#         fig.suptitle(f"Test Sample {i}", fontsize=16)
        
#         # Plot input frames
#         for t in range(T_in):
#             axs[0, t].imshow(batch_x[i, 0, t], cmap='viridis', origin='upper')
#             axs[0, t].set_title(f"Input T{t}")
#             axs[0, t].axis("off")
        
#         # Plot ground truth
#         for t in range(T_out):
#             axs[1, t].imshow(batch_y[i, t], cmap='viridis', origin='upper')
#             axs[1, t].set_title(f"GT T{t}")
#             axs[1, t].axis("off")
        
#         # Plot predictions
#         for t in range(T_out):
#             axs[2, t].imshow(predictions[i, t], cmap='viridis', origin='upper')
#             axs[2, t].set_title(f"Pred T{t}")
#             axs[2, t].axis("off")
        
#         plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#         plot_path = os.path.join(out_dir, f"test_sample_{i}.png")
#         plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#         plt.close(fig)

def predicting_distributions(model, images, device):
    model.eval()
    test_preds = []

    for image in images:
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():  # Disable gradient computation
        p_zero_pred, mu_nonzero_pred, sigma_nonzero_pred = model(image)
        p_zero_pred = p_zero_pred.cpu().numpy()[0]  # Extract scalar
        mu_nonzero_pred = mu_nonzero_pred.cpu().numpy()[0]
        sigma_nonzero_pred = sigma_nonzero_pred.cpu().numpy()[0]
        test_vec = np.array([p_zero_pred, mu_nonzero_pred, sigma_nonzero_pred])
        test_preds.append(test_vec)
    return test_preds

def flashes_per_second(test_preds):
    event_preds = []
    for h in range(len(test_preds)//36):
    output_num_flashes = []
    output_long = []
    for i in range(35*5*60):
        frame = int(i//(5*60) + i%(5*60)//(2.5*60)) + h*36 #Determining which frame
        try:
        preds = test_preds[frame] #getting the relevant distribution
        except:
        print(i)
        probs_zero = [preds[0], 1 - preds[0]]
        result = np.random.choice([0, 1], p=probs_zero)
        if result == 1:
        mean = preds[1]
        std = preds[2]
        num_flashes = np.random.normal(mean, std)
        num_flashes = round(num_flashes)
        output_num_flashes.append((i,num_flashes))
        for j in range(num_flashes):
            output_long.append((i,1))

    storm_preds = pd.DataFrame(output_long, columns=['t','num'])
    del storm_preds['num']

    event_preds.append(storm_preds)

    return event_preds

def mvn_from_file(cfg_path='configs/task3.yaml'):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    mean = np.loadtxt(os.path.join(cfg['inference']['mvn_mean_dir'], 'mvn_mean.csv'), delimiter=',')
    covariance = np.loadtxt(os.path.join(cfg['inference']['mvn_cov_dir'], 'mvn_covariance.csv'), delimiter=',')
    return mean, covariance

def find_closest_vector(vector: torch.Tensor, array: torch.Tensor):
    """
    Finds the closest (4,) vector in the (4, H, W) array to the given (4,) vector using Euclidean distance.

    Args:
        vector (torch.Tensor): A 1D tensor of shape (4,).
        array (torch.Tensor): A 3D tensor of shape (4, H, W) representing multiple (4,) vectors at each (H, W) position.

    Returns:
        closest_vector (torch.Tensor): The closest (4,) vector from the array.
        position (tuple): The (H, W) position of the closest vector in the array.
    """
    # Compute Squared Euclidean Distance (Avoids Square Root for Efficiency)
    distances = ((array - vector.view(4, 1, 1)) ** 2).sum(dim=0)  # Shape: (H, W)

    # Find the minimum distance index
    min_index = torch.argmin(distances)  # Flattened index
    min_index_2d = torch.unravel_index(min_index, distances.shape)  # Convert to (H, W)

    # Extract the closest vector
    closest_vector = array[:, min_index_2d[0], min_index_2d[1]]  # Shape: (4,)

    return closest_vector, min_index_2d

# # Example Usage
# vector = torch.rand(4)  # Shape: (4,)
# array = torch.rand(4, 384, 384)  # Shape: (4, 384, 384)

# closest_vec, position = find_closest_vector(vector, array)

# print("Closest Vector:", closest_vec)
# print("Position in (H, W):", position)

# event_preds = flashes_per_second(test_preds)

def test_task3(cfg_path='configs/task3.yaml'):
    """Main testing function"""
    # Load config
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Load test IDs
    test_ids = read_ids(cfg['data']['test_ids_file'])
    print(f"Testing on {len(test_ids)} storms")

    # Initialize test dataset
    test_dataset = CustomImageDataset(
        csv_path=cfg['data']['csv_path'],
        h5_path=cfg['data']['h5_path'],
        storm_ids=test_ids
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=True
    )

    # Initialize model
    model = ProbabilisticCNN_3().to(cfg['training']['device'])

    # Load trained weights
    ckpt_path = os.path.join(cfg['training']['ckpt_dir'], 'best_model.pt')
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    # Testing loop
    # all_metrics = {'mse': [], 'mae': [], 'ssim': []}
    # visualization_dir = os.path.join(cfg['inference']['output_dir'], 'visualizations')
    all_predictions = []
    all_gts = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader)):
            inputs = inputs.to(cfg['training']['device'])
            targets = targets.to(cfg['training']['device'])

            test_preds = predicting_distributions(model, inputs, cfg['training']['device'])
            event_preds = flashes_per_second(test_preds)
            mean, covariance = mvn_from_file(cfg_path)

            for h in range(len(event_preds)):
                xs = []
                ys = []
                j=0

                for i in event_preds[h]['t']:

                    frame = int(i//(5*60) + i%(5*60)//(2.5*60)) + h*36
                    # print('ok1')
                    draw = np.random.multivariate_normal(mean, cov=covariance, size=1)
                    # print('ok2')
                    draw = torch.tensor(draw)
                    # print('ok3')
                    array = images[frame]
                    # print('ok4')
                    closest_vec, position = find_closest_vector(draw, array)
                    # print('ok5')
                    xs.append(int(position[1]))
                    ys.append(int(position[0]))
                    # which frame it belongs to
                    # distance to frame
                    # x and y
                    j+=1
                    if j%500==0:
                        print(j)

                event_preds[h]['x_forecast']=xs
                event_preds[h]['y_forecast']=ys

            all_predictions.append(np.array(event_preds))
            all_gts.append(targets.cpu().numpy())

    # Save predictions and ground truths
    out_dir = cfg['inference']['output_dir']
    os.makedirs(out_dir, exist_ok=True)
    
    preds_path = os.path.join(out_dir, 'predictions.npy')
    gts_path = os.path.join(out_dir, 'groundtruths.npy')

    predictions = np.concatenate(all_predictions, axis=0)
    gts = np.concatenate(all_gts, axis=0)
    
    np.save(preds_path, predictions)
    np.save(gts_path, gts)
    
    print(f"\n[INFO] Saved predictions to {preds_path}")
    print(f"[INFO] Saved ground truths to {gts_path}")
    # # Calculate final metrics
    # final_metrics = {
    #     metric: np.mean(values) for metric, values in all_metrics.items()
    # }
    
    # # Save metrics
    # metrics_path = os.path.join(cfg['inference']['output_dir'], 'test_metrics.txt')
    # with open(metrics_path, 'w') as f:
    #     for metric, value in final_metrics.items():
    #         f.write(f"{metric.upper()}: {value:.4f}\n")
    
    # # Print results
    # print("\nTest Results:")
    # for metric, value in final_metrics.items():
    #     print(f"{metric.upper()}: {value:.4f}")

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/task3.yaml'
    test_task3(config_path)