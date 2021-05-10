
import torch
from torchvision.utils import make_grid            # makes image grid 
import matplotlib.pyplot as plt                    # plots graphs
import time

def get_noise(n_samples, z_dim, device='cpu'):
    
    return torch.randn(n_samples, z_dim, device=device)


def calculate_updated_noise(noise, weight):
    
    new_noise = noise + ( noise.grad * weight)
    return new_noise


def show_tensor_images(image_tensor, num_images=25, size=(3, 64, 64), nrow=5):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    fname='image_'+str(time.time())+'.png'
    plt.savefig('static//'+fname,bbox_inches='tight',pad_inches = 0)
    return fname
def get_score(current_classifications, original_classifications, target_indices, other_indices, penalty_weight):
    
    
    other_distances = current_classifications[:,other_indices] - original_classifications[:,other_indices]
    other_class_penalty = -torch.norm(other_distances, dim=1).mean() * penalty_weight
    target_score = current_classifications[:, target_indices].mean()

    return target_score + other_class_penalty    