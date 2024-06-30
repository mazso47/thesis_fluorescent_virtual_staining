#import sys
#sys.path.append('/home/local-admin/lightmycells/Label_free_cell_painting')

import torch
import torch.optim 
from torch.utils.data import DataLoader, Dataset
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
from Utils.trainUtils import  LoadingDatasetTrain, LoadingDatasetTest, calculate_metrics, average_list, extract_random_section
from scipy import ndimage


fresh_train = False
epoch_num = 1000

modality= "DIC"
#mode= "train"
data_type= "stack" 
pytable_train = "study_" + modality + "_train_" + data_type
pytable_valid = "study_" + modality + "_valid_" + data_type
path = data_type + "/" + modality + "_6_cubic/"


data_name = "study_" + modality + "_" + data_type

state_save_path = 'checkpoints/Unet/'
#pytable_path = "study_BF_1024_nuc_train.pytable"
#train_progress_images_path = "/media/local-admin/Extra_space_for_/Study_PC_mito_nuc/tools/training_progress/"
epoch_path = f"{state_save_path}{data_name}_epoch_{epoch_num}_Unet.pth"

ignore_index = 0 # his value won't be included in the loss calculation (output image value)- e.g. 0 is good for this data.
gpuid=0

#Unet params
n_classes= 4    #output channels (fluorescent)
in_channels= 6  #input channels (brightfield)
padding= True   #should levels be padded
depth= 6     #depth of the network 
wf= 5           #wf (int): number of filters in the first layer is 2**wf, was 6
up_mode= 'upconv' #upsample or interpolation 
batch_norm = False #sbatch normalization between the layers

#Training params
batch_size=2
patch_size=256
num_epochs = 1100
edge_weight = 1.1 
phases = ["train","valid","test"] 
organelles = ["Nucleus", "Mitochondria", "Actin", "Tubulin"]
metric_names = ["MAE", "SSIM_train", "PCC", "ECD", "COS"]

#specify if we should use a GPU (cuda) or only the CPU
if(torch.cuda.is_available()):
    print(torch.cuda.get_device_properties(gpuid))
    torch.cuda.set_device(gpuid)
    device = torch.device(f'cuda:{gpuid}')
else:
    device = torch.device(f'cpu')

Gen = UNet(n_classes=n_classes, in_channels=in_channels, padding=padding,depth=depth,wf=wf, up_mode=up_mode, batch_norm=batch_norm).to(device)
#print(f"total params: \t{sum([np.prod(p.size()) for p in Gen.parameters()])}")

tables.file._open_files.close_all()

dataset={} 
dataLoader={}
# for phase in phases: #now for each of the phases, we're creating the dataloader
#                       #interestingly, given the batch size, i've not seen any improvements from using a num_workers>0
#      #f = h5py.File("./Data/study_5_pytable_file.pytable")
#      #f.close()
#      dataset[phase]=Dataset(f"{pytable_name}_{phase}.pytable")
#      dataLoader[phase]=DataLoader(dataset[phase], batch_size=batch_size, 
#                                  shuffle=True, num_workers=0, pin_memory=False)
#      tables.file._open_files.close_all()         

dataset['train']=LoadingDatasetTrain(f"../pytables/{path}/{pytable_train}.pytable")         
dataLoader['train']=DataLoader(dataset['train'], batch_size=batch_size, 
                             shuffle=True, num_workers=0, pin_memory=False)

#dataset['valid']=LoadingDatasetTest(f"../pytables/{path}/{pytable_valid}.pytable")     
#dataLoader['valid']=DataLoader(dataset['valid'], batch_size=1, 
#                             shuffle=True, num_workers=0, pin_memory=False)

#tables.file._open_files.close_all() 


optimizerG = torch.optim.Adam(Gen.parameters(),lr=.0002, weight_decay=0.0002)
#optim = torch.optim.Adam(Gen.parameters(), 
#                           lr=.0002,
#                           weight_decay=0.0002)

#nclasses = dataset["train"].numpixels.shape[1]                  #not used for anything?
gen_criterion = GenLoss()

writer=SummaryWriter() 
best_loss_on_test = np.Infinity
edge_weight=torch.tensor(edge_weight).to(device)
start_time = time.time()

if fresh_train:
    start_epoch = 0
else:
    checkpoint = torch.load(epoch_path)
    start_epoch = checkpoint['epoch']
    Gen.load_state_dict(checkpoint['model_dict'])

#Save some variables e.g. MAE, SSIM etc
#The blank arrays are defined in file called 'make_variable_table.py'
MAE_train, SSIM_train, PCC_train, ECD_train, COS_train = {}, {}, {}, {}, {}
MAE_val, SSIM_val, PCC_val, ECD_val, COS_val = {}, {}, {}, {}, {}


for epoch in range(num_epochs):
    MAE_train[epoch] = []
    SSIM_train[epoch] = []
    PCC_train[epoch] = []
    ECD_train[epoch] = []
    COS_train[epoch] = []

    MAE_val[epoch] = []
    SSIM_val[epoch] = []
    PCC_val[epoch] = []
    ECD_val[epoch] = []
    COS_val[epoch] = []


def PSNR(im1, im2):
    im1 = im1.astype(np.float64) / 255
    im2 = im2.astype(float) / 255
    mse = np.mean((im1 - im2)**2)
    return 10*math.log10(1. / mse)

loss_values, loss_values_val = [], []
running_loss, running_loss_val = 0.0, 0.0
#early_stopper = EarlyStopper(patience=20, min_delta=0.05)
all_metrics = {}
for epoch in range(start_epoch, num_epochs):
    print(f"New epoch: {epoch}")

    metrics_val = {}
    metrics_tr = {}

    for metric_name in metric_names:
        metrics_tr[metric_name] = []
        metrics_val[metric_name] = []
    
    all_acc = {key: 0 for key in phases} 
    all_loss = {key: torch.zeros(0).to(device) for key in phases}
    cmatrix = {key: np.zeros((2,2)) for key in phases}

    for ii , (X, y) in enumerate(dataLoader["train"]): #for each of the batches
        Gen.train()
        y = y.to(device)
        X = X.to(device)                 
        prediction = Gen(X)
        Gen.zero_grad()
        g_loss = gen_criterion(prediction, y, epoch) 
        g_loss.backward(retain_graph=True)
        prediction = Gen(X)
        optimizerG.step()

        #print(X.dtype)
        #print(X.min())
        #print(X.max())
        #print(y.dtype)
        #print(y.min())
        #print(y.max())
        #print("Predicting:")
        #print(prediction.dtype)

        #directory_im = "./temp"
        #pred = np.squeeze(prediction[0,0,:,:].cpu().detach().numpy())
        # print(prediction[0,0,:,:])
        # print(pred)
        #pred = Image.fromarray(pred)
        #pred.save(f"{directory_im}/Unet_pred_epoch_{epoch}_img_{ii}_{data_name}.tif")

        #gtim = Image.fromarray(np.squeeze(y[0,0,:,:].cpu().detach().numpy()))
        #gtim.save(f"{directory_im}/Unet_gt_epoch_{epoch}_img_{ii}_{data_name}.tif")

        #img = Image.fromarray(np.squeeze(X[0,0,:,:].cpu().detach().numpy()))
        #img.save(f"{directory_im}/Unet_epoch_{epoch}_channel_img_{ii}_{data_name}.tif")
        
        if ii % 10 == 0:
            # save metrics to their arrays every 10 updates (batches)
           
            state = {'epoch' : epoch +1,
            'model_dict': Gen.state_dict(),
            'optim_dict': optimizerG.state_dict(),
            'best_loss_on_test': all_loss,
            'n_classes': n_classes,
            'in_channels': in_channels,
            'padding': padding,
            'depth': depth,
            'wf': wf,
            'up_mode': up_mode, 'batch_norm': batch_norm}
        
            
#            loss_train = g_loss.cpu().detach().numpy()#np.asarray(g_np)
#            LOSS_train = np.append(LOSS_train, loss_train)
#            save(f'LOSS_train.npy', LOSS_train)
#            
            for channel in range(n_classes):
                hold_pred = prediction[0,channel,:,:]
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
                    # for i,metric in enumerate(metrics):
                    #     if ~np.isnan(metric):
                    #         metrics_tr[metric_names[i]].append(metric)
                else:
                    metrics = calculate_metrics(ground_truth, prediction_tr, all=False) # Is cosine even needed?
                    print('TRAINING: SSIM: ',metrics[0], ' PCC: ',metrics[1])
                    # for i,metric in enumerate(metrics):
                    #     if ~np.isnan(metric):
                    #         metrics_tr[metric_names[i+1]].append(metric) 

    #metric_means_tr = {key: sum(values) / len(values) for key, values in metrics_tr.items()}
                
    #MAE_train[epoch] = np.append(MAE_train[epoch], metric_means_tr[metric_names[0]])
    #SSIM_train[epoch] = np.append(SSIM_train[epoch], metric_means_tr[metric_names[1]])
    #PCC_train[epoch] = np.append(PCC_train[epoch], metric_means_tr[metric_names[2]])
    #ECD_train[epoch] = np.append(ECD_train[epoch], metric_means_tr[metric_names[3]])
    #COS_train[epoch] = np.append(COS_train[epoch], metric_means_tr[metric_names[4]])
                
    #all_metrics[epoch] = {
    #    'MAE_train': MAE_train[epoch],
    #    'SSIM_train': SSIM_train[epoch],
    #    'PCC_train': PCC_train[epoch],
    #    'ECD_train': ECD_train[epoch],
    #    'COS_train': COS_train[epoch]
    #}
    
    #all_metrics[epoch] = metric_means_tr

    if epoch % 2 == 0:
        torch.save(state, f"{state_save_path}{data_name}_epoch_{epoch}_Unet.pth")

# # Save all metrics to a single .npy file
# save_path = f'eval/all_train_metrics_{modality}_{data_type}.npy'
# np.save(save_path, all_metrics)

# print("Training complete and metrics saved for all epochs.")
