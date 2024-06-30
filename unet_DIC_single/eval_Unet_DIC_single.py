import torch
import torch.optim 
from torch.utils.data import DataLoader
import h5py
from Networks.unet2d import UNet
from Utils.UnetLoss import GenLoss
import matplotlib
matplotlib.use('Agg')
import numpy as np
from numpy import save
from torch.utils.tensorboard import SummaryWriter
import time
import math
import tables
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from PIL import Image
from Utils.trainUtils import LoadingDatasetTrain, LoadingDatasetTest, calculate_metrics, average_list, extract_random_section
from scipy import ndimage

# Define paths and parameters
modality= "DIC"
data_type= "single" 
pytable_valid = "study_" + modality + "_valid_" + data_type
data_name = "study_" + modality + "_" + data_type
state_save_path = 'checkpoints/Unet/'
ignore_index = 0
gpuid = 0

# Unet params
n_classes = 4
in_channels = 1
padding = True
depth = 6
wf = 5
up_mode = 'upconv'
batch_norm = False

# Training params
batch_size = 1  # Use batch size of 1 for validation
num_epochs = 50
phases = ["train","valid","test"] 
organelles = ["Nucleus", "Mitochondria", "Actin", "Tubulin"]
metric_names = ["MAE", "SSIM", "PCC", "ECD", "COS"]
path = data_type + "/" + modality + "/"

# Check if GPU is available
if(torch.cuda.is_available()):
    print(torch.cuda.get_device_properties(gpuid))
    torch.cuda.set_device(gpuid)
    device = torch.device(f'cuda:{gpuid}')
else:
    device = torch.device(f'cpu')

# Initialize U-Net model
Gen = UNet(n_classes=n_classes, in_channels=in_channels, padding=padding,depth=depth,wf=wf, up_mode=up_mode, batch_norm=batch_norm).to(device)

# Load validation dataset
dataset = {}
dataLoader = {}
dataset['valid'] = LoadingDatasetTrain(f"../pytables/{path}/{pytable_valid}.pytable")     
dataLoader['valid'] = DataLoader(dataset['valid'], batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

# Define the evaluation function
def evaluate_model(epoch, model, dataloader, device):
    model.eval()
    metrics_val = {metric: [] for metric in metric_names}
    with torch.no_grad():
        for jj, (X, y) in enumerate(dataloader):    
            y = y.to(device)
            X = X.to(device)
            prediction1 = model(X)
            for channel in range(n_classes):
                prediction = prediction1[0, channel, :, :].cpu().numpy()
                ground_truth = y[0, channel, :, :].cpu().numpy()
                if channel < 2:
                    metrics = calculate_metrics(ground_truth, prediction, all=True)
                    for i, metric in enumerate(metrics):
                        if ~np.isnan(metric):
                            metrics_val[metric_names[i]].append(metric)
                else:
                    metrics = calculate_metrics(ground_truth, prediction, all=False)
                    for i, metric in enumerate(metrics):
                        if ~np.isnan(metric):
                            metrics_val[metric_names[i+1]].append(metric)
    metric_means_val = {key: sum(values) / len(values) for key, values in metrics_val.items()}
    return metric_means_val

# Dictionary to store metrics for all epochs
all_metrics = {}

# Evaluate each epoch's saved model
for epoch in range(num_epochs):
    epoch_path = f"{state_save_path}{data_name}_epoch_{epoch}_Unet.pth"
    checkpoint = torch.load(epoch_path, map_location=device)
    Gen.load_state_dict(checkpoint['model_dict'])

    metrics = evaluate_model(epoch, Gen, dataLoader['valid'], device)
    print(f"Epoch {epoch} Validation Metrics: {metrics}")

    all_metrics[epoch] = metrics

# Save all metrics to a single .npy file
save_path = f'eval/all_validation_metrics_{modality}_{data_type}.npy'
np.save(save_path, all_metrics)

print("Validation complete for all epochs.")
