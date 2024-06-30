import torch
from torch import nn

class GenLoss(nn.Module):
    def __init__(self):
        super(GenLoss, self).__init__()
        self.mae_loss = nn.L1Loss(reduction='none')  

    def forward(self, out_images, target_images, epoch):
        num_pairs = target_images.shape[0]
        num_channels = target_images.size(1)  
        
        pair_losses = []
        for pair_idx in range(num_pairs):								
													
            total_loss = 0	
            valid_channels = 0																	 											
            for channel in range(num_channels):	
                						
                mask = torch.any(target_images[pair_idx, channel, :, :] != 0) 

                if mask:
                    image_loss = self.mae_loss(out_images[pair_idx, channel, :, :], target_images[pair_idx, channel, :, :])
                    total_loss += image_loss.mean()
                    valid_channels += 1			

            if valid_channels > 0:
                total_loss /= valid_channels  
            pair_losses.append(total_loss)
            
        combined_loss = sum(pair_losses) / max(len(pair_losses), 1)  

        return combined_loss
        
if __name__ == "__main__":
    g_loss = GenLoss()

