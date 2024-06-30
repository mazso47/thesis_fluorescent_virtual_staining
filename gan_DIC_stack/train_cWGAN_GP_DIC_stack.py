import torch
import torch.optim 
from torch.utils.data import DataLoader
from Utils.trainUtils import LoadingDatasetTrain, LoadingDatasetTest, calculate_metrics
import h5py
from Networks.unet2d import UNet
from Networks.patchdiscrim2d import Discriminator
from Utils.GANLoss import GenLoss
import matplotlib
matplotlib.use('Agg')
import numpy as np
from numpy import save
from torch.utils.tensorboard import SummaryWriter
import time
import math
import tables
from torch.autograd import Variable
import torch.autograd as autograd
from skimage.metrics import structural_similarity as ssim
from PIL import Image

# Set up your variables and paths
fresh_train = True  # Set this to False if continuing training from pre-trained weights
unet_pretrained_epoch = 942
modality = "DIC"
data_type = "stack"
pytable_train = "study_" + modality + "_train_" + data_type
path = data_type + "/" + modality + "/"
data_name = "study_" + modality + "_" + data_type
state_save_path = 'checkpoints/GAN/'

# Paths to pre-trained U-Net weights
unet_pretrained_path = f"{state_save_path}{data_name}_epoch_{unet_pretrained_epoch}_Unet.pth"
epoch_path_gen = f"{state_save_path}{data_name}_epoch_{unet_pretrained_epoch}_GEN.pth"
epoch_path_disc = f"{state_save_path}{data_name}_epoch_{unet_pretrained_epoch}_DISC.pth"

ignore_index = 0  # Value won't be included in the loss calculation
gpuid = 0

# Unet params
n_classes = 4  # Output channels (fluorescent)
in_channels = 6  # Input channels (brightfield)
padding = True  # Should levels be padded
depth = 6  # Depth of the network
wf = 5  # Number of filters in the first layer is 2**wf
up_mode = 'upconv'  # Upsample or interpolation
batch_norm = False  # Batch normalization between the layers

# Training params
batch_size = 2
patch_size = 1024
num_epochs = 500
edge_weight = 1.1
phases = ["train", "valid", "test"]
validation_phases = ["valid"]
metric_names = ["MAE", "SSIM", "PCC", "ECD", "COS"]

# Specify if we should use a GPU (cuda) or only the CPU
if torch.cuda.is_available():
    print(torch.cuda.get_device_properties(gpuid))
    torch.cuda.set_device(gpuid)
    device = torch.device(f'cuda:{gpuid}')
else:
    device = torch.device('cpu')

Gen = UNet(n_classes=n_classes, in_channels=in_channels, padding=padding, depth=depth, wf=wf, up_mode=up_mode, batch_norm=batch_norm).to(device)
Disc = Discriminator().to(device)
gen_criterion = GenLoss()

tables.file._open_files.close_all()

dataset = {}
dataLoader = {}
dataset["train"] = LoadingDatasetTrain(f"../pytables/{path}/{pytable_train}.pytable")
dataLoader["train"] = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
tables.file._open_files.close_all()

optimizerG = torch.optim.Adam(Gen.parameters(), lr=0.0002, betas=(0., 0.9))
optimizerD = torch.optim.Adam(Disc.parameters(), lr=0.0002, betas=(0., 0.9), weight_decay=0.001)

writer = SummaryWriter()
best_loss_on_test = np.Infinity
edge_weight = torch.tensor(edge_weight).to(device)
start_time = time.time()

# Load pre-trained U-Net weights into the generator
if fresh_train:
    start_epoch = 0
    Gen.load_state_dict(torch.load(unet_pretrained_path)['model_dict'])
    print(f'Starting GAN training from pre-trained U-Net epoch {unet_pretrained_epoch}')
else:
    checkpoint = torch.load(epoch_path_gen)
    checkpoint2 = torch.load(epoch_path_disc)
    Gen.load_state_dict(checkpoint['model_dict'])
    Disc.load_state_dict(checkpoint2['model_dict'])
    start_epoch = checkpoint['epoch']
    print('Start Epoch:', start_epoch)

def calculate_gradient_penalty(real_images, fake_images):
    eta = torch.FloatTensor(batch_size, 1, 1, 1).uniform_(0, 1)
    eta = eta.expand(batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
    eta = eta.to(device)
    interpolated = eta * real_images + ((1 - eta) * fake_images)
    
    interpolated = Variable(interpolated, requires_grad=True)
    interpolated = interpolated.to(device)

    prob_interpolated = Disc(interpolated)
    prob_interpolated = prob_interpolated.to(device)

    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                              grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                              create_graph=True, retain_graph=True)[0]
    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return grad_penalty

def replace_blank_channels(fake_imgs, real_imgs):
    for i in range(fake_imgs.size(0)):  # iterate over the batch
        for channel in range(fake_imgs.size(1)):  # iterate over channels
            if torch.all(real_imgs[i, channel, :, :] == 0):  # if the real image channel is blank
                fake_imgs[i, channel, :, :] = 0  # set the fake image channel to blank
    return fake_imgs

loss_values, loss_values_val = [], []
running_loss, running_loss_val = 0.0, 0.0
all_metrics = {}
for epoch in range(start_epoch, num_epochs):
    print(f"Starting new epoch: {epoch}")

    metrics_val = {}
    metrics_tr = {}

    for metric_name in metric_names:
        metrics_tr[metric_name] = []
        metrics_val[metric_name] = []

    all_acc = {key: 0 for key in phases}
    all_loss = {key: torch.zeros(0).to(device) for key in phases}
    cmatrix = {key: np.zeros((2, 2)) for key in phases}

    for ii, (X, y) in enumerate(dataLoader["train"]):
        optimizerD.zero_grad()
        y = y.to(device)
        X = X.to(device)

        fake_imgs = Gen(X)
        fake_imgs = replace_blank_channels(fake_imgs, y)  # replace blank channels in generated images

        real_concat_with_input = torch.cat((y, X), 1)
        fake_concat_with_input = torch.cat((fake_imgs, X), 1)

        real_out = Disc(real_concat_with_input).mean()
        fake_out = Disc(fake_concat_with_input).mean()
        gradient_penalty = calculate_gradient_penalty(real_concat_with_input, fake_concat_with_input)

        was_loss = (fake_out - real_out) + 10 * gradient_penalty
        was_loss.create_graph = True
        was_loss.backward(retain_graph=True)
        optimizerD.step()

        optimizerG.zero_grad()

        if ii % 5 == 0:
            fake_imgs = Gen(X)
            fake_imgs = replace_blank_channels(fake_imgs, y)  # replace blank channels in generated images

            fake_concat_with_input = torch.cat((fake_imgs, X), 1)
            fake_out = Disc(fake_concat_with_input).mean()
            g_loss = gen_criterion(fake_out, fake_imgs, y, epoch)
            g_loss.backward()
            optimizerG.step()

            # #print(X.dtype)
            # #print(X.min())
            # #print(X.max())
            # #print(y.dtype)
            # #print(y.min())
            # #print(y.max())
            # #print("Predicting:")
            # #print(fake_imgs.dtype)

            # directory_im = "./temp"
            # pred = np.squeeze(fake_imgs[0,0,:,:].cpu().detach().numpy())
            # #print(fake_imgs[0,0,:,:])
            # #print(pred)
            # pred = Image.fromarray(pred)
            # pred.save(f"{directory_im}/Unet_pred_epoch_{epoch}_img_{ii}_{data_name}.tif")

            # gtim = Image.fromarray(np.squeeze(y[0,0,:,:].cpu().detach().numpy()))
            # gtim.save(f"{directory_im}/Unet_gt_epoch_{epoch}_img_{ii}_{data_name}.tif")

            # img = Image.fromarray(np.squeeze(X[0,0,:,:].cpu().detach().numpy()))
            # img.save(f"{directory_im}/Unet_epoch_{epoch}_channel_img_{ii}_{data_name}.tif")            

            for channel in range(n_classes):
                hold_pred = fake_imgs[0,channel,:,:]
                hold_pred_cpu = hold_pred.cpu()
                prediction_tr = hold_pred_cpu.detach().numpy()
                hold_gt = y[0,channel,:,:]
                hold_gt_cpu = hold_gt.cpu()
                ground_truth = hold_gt_cpu.detach().numpy()  

                # gt_height, gt_width = ground_truth.shape[-2:]                          # max output size is 1183x1183, if the gt is larger, resize pred to same size
                # if (prediction_tr.shape[-1] < gt_width) or (prediction_tr.shape[-2] < gt_height):      #this is not needed as training is on 256x256
                #     pred_ch = Image.fromarray(prediction_tr)
                #     prediction_tr = pred_ch.resize((gt_width, gt_height))             

                if channel<2:
                    metrics = calculate_metrics(ground_truth, prediction_tr, all=True) # Is cosine even needed?
                    print('TRAINING: MAE: ',metrics[0],  'SSIM: ',metrics[1], ' PCC: ',metrics[2],  ' ECD: ', metrics[3], ' COD: ', metrics[4])
                    for i,metric in enumerate(metrics):
                        if ~np.isnan(metric):
                            metrics_tr[metric_names[i]].append(metric)
                else:
                    metrics = calculate_metrics(ground_truth, prediction_tr, all=False) # Is cosine even needed?
                    print('TRAINING: SSIM: ',metrics[0], ' PCC: ',metrics[1])
                    for i,metric in enumerate(metrics):
                        if ~np.isnan(metric):
                            metrics_tr[metric_names[i+1]].append(metric) 

           

        if ii % 10 == 0:
            state = {
                'epoch': epoch + 1,
                'model_dict': Gen.state_dict(),
                'optim_dict': optimizerG.state_dict(),
                'best_loss_on_test': all_loss,
                'n_classes': n_classes,
                'in_channels': in_channels,
                'padding': padding,
                'depth': depth,
                'wf': wf,
                'up_mode': up_mode,
                'batch_norm': batch_norm
            }

            state2 = {
                'epoch': epoch + 1,
                'model_dict': Disc.state_dict(),
                'optim_dict': optimizerD.state_dict(),
                'best_loss_on_test': all_loss,
                'n_classes': n_classes,
                'in_channels': in_channels,
                'padding': padding,
                'depth': depth,
                'wf': wf,
                'up_mode': up_mode,
                'batch_norm': batch_norm
            }

    # metric_means_tr = {key: sum(values) / len(values) for key, values in metrics_tr.items()} 
    # all_metrics[epoch] = metric_means_tr

    if epoch % 2 == 0:
        torch.save(state2, f"{state_save_path}{data_name}_epoch_{epoch}_DISC.pth")
        torch.save(state, f"{state_save_path}{data_name}_epoch_{epoch}_GEN.pth")

# Save all metrics to a single .npy file
#save_path = f'eval/all_train_metrics_{modality}_{data_type}.npy'
#np.save(save_path, all_metrics)

print("Training complete and metrics saved for all epochs.")

