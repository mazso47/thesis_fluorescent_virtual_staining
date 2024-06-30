#import sys
#sys.path.append('/home/local-admin/lightmycells/Label_free_cell_painting')

import os
from Networks.unet2d import UNet
import torch
import torch.optim 
import matplotlib
matplotlib.use('Agg')
import numpy as np
from PIL import Image
import sys, glob
import re
from aicsimageio import AICSImage
from torch.utils.data import DataLoader
import tables
from Utils.trainUtils import LoadingDatasetTest, calculate_metrics
from scipy import ndimage
import json
import math

# Load the saved model weights e.g. 28th epoch of trained WGAN model
epoch_num = 942 # what epoch to load

modality= "DIC"
#mode= "train"
data_type= "stack" 
#pytable_train = "study_" + modality + "_train_" + data_type
pytable_test = "study_" + modality + "_test_" + data_type
path = data_type + "/" + modality + "_6_cubic/"
data_name = "study_" + modality + "_" + data_type
state_save_path = 'checkpoints/Unet/'

checkpoint = torch.load(f'{state_save_path}/study_{modality}_{data_type}_epoch_{epoch_num}_Unet.pth') 
output_path = "result/Unet"

# Network/Training Parameters (copied from training)
ignore_index = 0 
gpuid=0
n_classes= 4
in_channels= 6
padding= True
depth= 6
wf= 5 
up_mode= 'upconv' 
batch_norm = False
batch_size=1
patch_size=256
edge_weight = 1.1 
phases = ["train","val"] 
validation_phases= ["val"] 

# Specify if we should use a GPU (cuda) or only the CPU
if(torch.cuda.is_available()):
    print(torch.cuda.get_device_properties(gpuid))
    torch.cuda.set_device(gpuid)
    device = torch.device(f'cuda:{gpuid}')
else:
    device = torch.device(f'cpu')

# Define the network
Gen = UNet(n_classes=n_classes, in_channels=in_channels, padding=padding,depth=depth,wf=wf, up_mode=up_mode, batch_norm=batch_norm).to(device)
print(f"total params: \t{sum([np.prod(p.size()) for p in Gen.parameters()])}")
Gen.load_state_dict(checkpoint['model_dict'])
Gen.eval()

# Define empty arrays
checkfull = {}
    
batch_size=1
dataset={} 
dataLoader={}
#dataset["test"]=LoadingDatasetTest("/media/local-admin/lightmycells/thesis/code/temp_DIC_single_small/pytables/study_DIC_test_single_full.pytable")
dataset["test"]=LoadingDatasetTest(f"../pytables/{path}/{pytable_test}.pytable")                  # test !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
dataLoader["test"]=DataLoader(dataset["test"], batch_size=batch_size, 
                                    shuffle=False, num_workers=0, pin_memory=False)


# Initialize a dictionary to store metrics
all_metrics = []

for ii, (X, y) in enumerate(dataLoader["test"]): #for each of the batches
    x_in = X
    x_in = x_in.to(device)
    prediction1 = Gen(x_in)
    batch_metrics = {"batch": ii, "metrics": []}
    
    for channel in range(n_classes): 
        checkfull = prediction1[0,channel,:,:]
        checkfull_cpu = checkfull.cpu()
        prediction = checkfull_cpu.detach().numpy()
        gt = y[0][channel].numpy()

        gt_height, gt_width = gt.shape[-2:]  # prediction images are 1024x1024, so resize to gt size for best comparison
        if (prediction.shape[-1] < gt_width) or (prediction.shape[-2] < gt_height):
            pred = Image.fromarray(prediction)
            prediction = np.array(pred.resize((gt_width, gt_height)))
        
        directory_im = output_path
        os.makedirs(directory_im, exist_ok=True)
        img = Image.fromarray(prediction)
        img.save(f"{directory_im}/Unet_epoch_{epoch_num}_channel_{channel}_img_{ii}_{data_name}.tif")

        directory_gt = output_path
        os.makedirs(directory_gt, exist_ok=True)
        gtim = Image.fromarray(gt)
        gtim.save(f"{directory_gt}/Unet_gt_epoch_{epoch_num}_channel_{channel}_img_{ii}_{data_name}.tif")
        
        # Calculate metrics
        metrics = calculate_metrics(gt, prediction, all=True if channel < 2 else False)
        
        if channel < 2:
            batch_metrics["metrics"].append({
                "channel": channel,
                "MAE": metrics[0] ,
                "SSIM": metrics[1],
                "PCC": metrics[2], 
                "ECD": metrics[3] ,
                "COS": metrics[4]
            })

        else:
            batch_metrics["metrics"].append({
                "channel": channel,
                "SSIM": metrics[0],
                "PCC": metrics[1]
            })
    
    all_metrics.append(batch_metrics)
    #print(batch_metrics)

# Calculate the mean of each metric across all batches for each channel
mean_metrics = {channel: {metric: [] for metric in ["MAE", "SSIM", "PCC", "ECD", "COS"]} for channel in range(n_classes)}

for batch in all_metrics:
    for channel_metrics in batch["metrics"]:
        channel = channel_metrics["channel"]
        for metric, value in channel_metrics.items():
            if metric != "channel" and value is not math.isnan(value):
                mean_metrics[channel][metric].append(value)

# Compute the mean for each metric for each channel
mean_metrics = {channel: {metric: np.nanmean(values) if values else None for metric, values in channel_metrics.items()} for channel, channel_metrics in mean_metrics.items()}

# Save all metrics to a single file
with open(f'eval/Unet_epoch_{epoch_num}_{modality}_{data_type}_metrics.json', 'w') as f:
    json.dump(mean_metrics, f)

print("Testing complete and metrics saved.")
