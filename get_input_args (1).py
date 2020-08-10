# PROGRAMMER: Jessica Costa
# DATE CREATED: 08 Aug 2020                                  
# REVISED DATE: 
# PURPOSE: Create a function that retrieves the command line inputs 
#          from the user using the Argparse Python module. If the user fails to 
#          provide some or all of the inputs, then the default values are
#          used for the missing inputs. Command Line Arguments:
#     1. Image Folder as --data_dir with default value 'flowers'
#     2. CNN Model Architecture as --arch with default value 'vgg'
#     3. Learning Rate as --lr with default value 0.01
#     4. Hidden Units as --hidden_units with default value of 512
#     5. Epochs as --epochs with defaul value = 20
#     6. Processor Device used as --gpu with default value True
#     7. Checkpoint file path as --cp with default value 'checkpoint.pth'
#     8. Number of most likely classes as --top_k with default value 3
#     9. Json file mapping categories to names as --cat_names with default value 'cat_to_name.json'
#
##
# Imports python modules
import argparse

def get_input_args():
    """
    Retrieves and parses command line arguments provided by the user from a terminal window.
    
    Parameters:
        None
    Returns:
        parse_args() - data structure that stores the command line arguments objects
    """
    
    # Creates Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    # Creates arguments
    parser.add_argument('--data_dir', type = str, default = 'flowers', help = 'path to the data folder')
    parser.add_argument('--arch', type = str, default = 'vgg', help = 'CNN model architecture (resnet, alexnet, vgg)')
    parser.add_argument('--lr', type = float, default = 0.01, help = 'learning rate')
    parser.add_argument('--hidden_units', type = list, default = [1024,512] , help = 'number of nodes per layer in a list format: [a, b, c, ...]')
    parser.add_argument('--epochs', type = int, default = 20, help = 'number of epochs')
    parser.add_argument('--print_every', type = int, default = 60, help = 'print values every number of steps')
    parser.add_argument('--gpu', type = bool, default = True, help = 'Processor selection (True - GPU (default), False - CPU)')
    parser.add_argument('--cp_dir', type = str, default = 'checkpoint.pth', help = 'checkpoints directory')
    parser.add_argument('--img_path', type = str, default = 'flowers/test/1/image_06743.jpg', help = 'image path')
    parser.add_argument('--top_k', type = int, default = 3, help = 'number of most likely classes')
    parser.add_argument('--cat_names', type = str, default = 'cat_to_name.json', help = 'mapping of categories to real names - json file')
    
    return parser.parse_args()