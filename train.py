# PROGRAMMER: Jessica Costa
# DATE CREATED: 7 Aug 2020                               
# REVISED DATE: 
# PURPOSE: Trains a new network classifier and saves model as a checkpoint.
# 
##

# Imports functions created for this program
from get_input_args import get_input_args
from utility_functions import load_data, load_mapping, save_ckpt
from model_functions import device_sel, new_classifier, loss_optim, train_classifier

def main():
        
    # This function retrieves Command Line Arguments from user as input and returns
    # the collection of these command line argument as the variable in_arg
    in_arg = get_input_args()
    print(in_arg)

    # Loads the data
    image_datasets, dataloaders = load_data(in_arg.data_dir)
    
    # Loads mapping file
    cat_to_name = load_mapping(in_arg.cat_names)
    
    # Selects Processor Device (GPU/CPU)
    device = device_sel(in_arg.gpu)
    
    # Builds new, untrained feed-forward network as a classifier
    model = new_classifier(in_arg.arch, in_arg.hidden_units, cat_to_name)
    
    # Criterion and optimizer functions
    criterion, optimizer = loss_optim(model, in_arg.lr)
    
    # Trains Classifier
    train_classifier(model, dataloaders, in_arg.epochs, in_arg.print_every, criterion, optimizer, device)
    
    # Save trained Classifier
    save_ckpt(model, image_datasets, in_arg.arch, in_arg.hidden_units, in_arg.epochs, in_arg.lr, in_arg.print_every, optimizer, in_arg.cp_dir)
    
    
# Call to main
if __name__ == "__main__":
    main()    