# PROGRAMMER: Jessica Costa
# DATE CREATED: 07 Aug 2020                                  
# REVISED DATE: 
# PURPOSE: model_functions.py contain functions and classes related to the model.
#          These functions are listed below:
#     1. device_sel(gpu)
#     2. new_classifier(arch, hidden_units, cat_to_name, learn_rate)
#     3. image_processing()
##

# Imports
import argparse
import numpy as np
from collections import OrderedDict
from PIL import Image

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# Model Options
resnet18 = models.resnet18(pretrained = True)
alexnet = models.alexnet(pretrained = True)
vgg16 = models.vgg16(pretrained = True)

models = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16}

def device_sel(gpu):
    '''
    Selects processor to be used (GPU - 'CUDA' or CPU)
    '''
    if gpu == True:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
        
    return device

def new_classifier(arch, hidden_units, cat_to_name):
    # Model Selection
    model = models[arch]
    
    # Freeze parameters to avoid backpropagation
    for param in model.parameters():
        param.requires_grad = False
        
    # Hyperparameters
    input_size = model.classifier[0].in_features
    output_size = len(cat_to_name)
    
    # This allows for various hidden_layers in classifier
    classifier_dict = []
    feature_in_out = [input_size] + hidden_units + [output_size]    
    feature_pairs = list(zip(feature_in_out, feature_in_out[1:]))
    
    for i, x in enumerate(feature_pairs):
        classifier_dict.append(('fc' + str(i+1), nn.Linear(*(feature_pairs[i]))))
        if i != len(feature_pairs) - 1:
            classifier_dict.append(('relu' + str(i+1), nn.ReLU()))
            classifier_dict.append(('dropout' + str(i+1), nn.Dropout(p=0.1)))
    classifier_dict.append(('output', nn.LogSoftmax(dim=1)))
    
    # Classifier
    classifier = nn.Sequential(OrderedDict(classifier_dict))
    
    # Uses built classifier in the pre-trained model
    model.classifier = classifier
    
    return model

def loss_optim(model, learn_rate):
    '''
    Function defines loss function and optimizer
    '''
    # Loss function
    criterion = nn.NLLLoss()
    # Freezes features parameters and only trains the classifier parameters
    optimizer = optim.Adam(model.classifier.parameters(), lr = learn_rate)
    
    return criterion, optimizer

def train_classifier(model, dataloaders, epochs, print_every, criterion, optimizer, device):
    '''
    Trains the classifier model
    '''    
    steps = 0
    model.to(device)
    model.train()    
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(dataloaders['train']):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            # Clearing gradients
            optimizer.zero_grad()            
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if steps % print_every == 0:
                # validading the classifier - calls on the classifier_validation function
                # mode changes to eval for inference - since not training model anymore
                model.eval()
                with torch.no_grad():
                    valid_loss, accuracy = classifier_validation(model, dataloaders, criterion, device)
                
                print('Epoch: {}/{}... '.format(e+1, epochs),
                      'Training Loss: {:.3f}.. '.format(running_loss/print_every),
                      'Validation Loss: {:.3f}..'.format(valid_loss/len(dataloaders['valid'])),
                      'Validation Accuracy: {:.3f}'.format(accuracy/len(dataloaders['valid'])))
                running_loss = 0
                model.train()
    return model

def classifier_validation(model, dataloaders, criterion, device):
    '''
    Function used to validate the classifier
    '''
    valid_loss = 0
    accuracy = 0
    for ii, (inputs, labels) in enumerate(dataloaders['valid']):
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        valid_loss += criterion(output, labels)
        
        ps = torch.exp(output)
        equality = labels.data == ps.max(dim=1)[1]
        accuracy += equality.type(torch.FloatTensor).mean()        
    return valid_loss, accuracy
    
def predict(img_path, device, model, top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Open and processes image
    image = process_image(image_path).type(torch.FloatTensor).unsqueeze_(0).to(device)
        
    # Invert dictionary
    model.idx_to_class = dict(map(reversed, model.class_to_idx.items()))
    
    # Change mode to eval
    model.eval()
    
    with torch.no_grad():
        outputs = model.forward(image)
        ps = torch.exp(outputs)
        probs, indices = ps.topk(topk)
        probs = probs.squeeze()
        classes = [model.idx_to_class[idx] for idx in indices[0].tolist()]
    
    return probs, classes    
    