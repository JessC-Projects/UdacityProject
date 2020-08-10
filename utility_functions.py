# PROGRAMMER: Jessica Costa
# DATE CREATED: 07 Aug 2020                                  
# REVISED DATE: 
# PURPOSE: These functions are used throughout the code and are listed below:
#     1. load_data(data_dir)
#     2. load_mappint(cat_name)
##

# Imports
import numpy as np
from collections import OrderedDict

from PIL import Image
import json

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

         
def load_data(data_dir):
    """
    Loads and transforms three sets of data (training, validation and testing).
            
    Parameters:
        data_dir - The path to the folder of images
    Returns:
        dataloaders - Dictionary with 'key' as type of data (train, valid, test)
    """   
    # Directories of each data set
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'   
    # Transform and normalize data for training, validation and testing sets
    data_transforms = {'train': transforms.Compose([transforms.RandomRotation(10),
                                                    transforms.RandomResizedCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                                         std = [0.229, 0.224, 0.225])]),
                       'valid-test': transforms.Compose([transforms.Resize(256),
                                                         transforms.CenterCrop(224),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                                              std = [0.229, 0.224, 0.225])])}
    # Load the datasets with ImageFolder
    image_datasets = {'train': datasets.ImageFolder(train_dir, transform = data_transforms['train']),
                      'valid': datasets.ImageFolder(valid_dir, transform = data_transforms['valid-test']),
                      'test': datasets.ImageFolder(test_dir, transform = data_transforms['valid-test'])}
    
    # Dataloaders using the image datasets and transforms
    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size = 64, shuffle=True), 
                   'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size = 64, shuffle=True),
                   'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size = 64, shuffle=True)}
    
    return image_datasets, dataloaders

def load_mapping(cat_name):
    
    # Load cat_to_name.json into cat_to_name
    with open(cat_name, 'r') as f:
        cat_to_name = json.load(f)
    
    return cat_to_name

def save_ckpt(model, image_datasets, arch, hidden_units, epochs, learn_rate, print_every, optimizer, cp_dir):
    model.class_to_idx = image_datasets['train'].class_to_idx
    
    checkpoint = {'arch': arch,
                  'hidden_layers': hidden_units,
                  'epochs': epochs,
                  'learn_rate': learn_rate,
                  'print_every': print_every,
                  'optimizer_state': optimizer.state_dict(),
                  'state_dict': model.state_dict()
                 }
    
    torch.save(checkpoint, cp_dir)
    
def load_ckpt(cp_dir):
    checkpoint = torch.load(cp_dir)
    
    arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_layers']
    epochs = checkpoint['epochs']
    learn_rate = checkpoint['learn_rate']
    print_every = checkpoint['print_every']
    #optimizer.state_dict = checkpoint['optimizer_state']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an torch Tensor (converted from Numpy array)
    '''
    # Process a PIL image for use in a PyTorch model 
    img = Image.open(image)
    
    # Get Thumbnail size
    if img.width <= img.height:
        thumbnail_size = (256, img.height)
    else:
        thumbnail_size = (img.width,256)
    
    # Resize image where shortest side is 256 pixels (keeping aspect ratio)
    img.thumbnail(thumbnail_size)
    
    # Crop image to 224x224
    cropped_size = (224, 224)
    
    # Calculating Crop Coordinates
    left_crop = int((img.width-cropped_size[0])/2)
    upper_crop = int((img.height-cropped_size[1])/2)
    right_crop = int((img.width+cropped_size[0])/2)
    lower_crop = int((img.height+cropped_size[1])/2)
    crop_coord = (left_crop, upper_crop, right_crop, lower_crop)
    
    cropped_img = img.crop(crop_coord)
    
    # Convert image to Numpy array with values ranging 0-1
    np_img = np.array(cropped_img)/256
    
    # Normalizing images
    means = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    np_img = (np_img - means)/std
    
    # Changing color channel to 1st dimmension from third. 
    np_img = np_img.transpose(2,0,1)
      
    # return tensor
    return torch.from_numpy(np_img)

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def get_flower_labels(cat_to_name, img_path, classes):
    """
    Return flower name and creates a dictionary that maps predicted categories to real flower names
    """
    flower_name = cat_to_name[img_path.split('/')[-2]]
    
    flower_names_top_k = [cat_to_name[k] for k in classes]
    
    return flower_name, flower_names_top_k
    
def plot_predict(flower_name, img_path, probs, flower_names_top_k):
    # Create two figures
    fig, (ax1, ax2) = plt.subplots(ncols = 1, nrows = 2, figsize = (6,10))
    
    # Plot Image
    ax1.set_title(flower_name)
    ax1.axis('off')
    ax1.imshow(Image.open(img_path))
    
    # Plot prediction
    ax2.barh(y = np.arange(len(probs)), width = probs, align = 'center')
    ax2.set_yticks(np.arange(len(probs)))
    ax2.set_yticklabels(flower_names_top_k)
    ax2.invert_yaxis() # labels top to bottom
    ax2.set_title('Class Probability')
    ax2.set_xlabel('Probability')
    ax2.set_xlim(0,1.1)
    
    return ax1, ax2
    
    



    
    
    
    
    
          
          
      
    
    
    
    


    