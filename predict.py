# PROGRAMMER: Jessica Costa
# DATE CREATED: 7 Aug 2020
# REVISED DATE: 
# PURPOSE: Uses a trained network to predict the classes for an input image
# 
##

# Imports functions created for this program
from get_input_args import get_input_args
from utility_functions import load_ckpt, process_image, load_mapping, get_flower_labels, plot_predict
from model_functions import device_sel, predict

def main():
        
    # This function retrieves Command Line Arguments from user as input and returns
    # the collection of these command line argument as the variable in_arg
    in_arg = get_input_args()
    print(in_arg)

    # Selects Processor Device (GPU/CPU)
    device = device_sel(in_arg.gpu)
    
    # Loads the model and optimizer
    model = load_ckpt(in_arg.cp_dir)
        
    # Loads and processes image
    process_image = process_image(in_arg.img_path)
    
    # Predicts Class of image
    probs, classes = predict(in_arg.img_path, device, model, in_arg.top_k)
    
    # Classes mapping
    cat_to_name = load_mapping(in_arg.cat_names)
    flower_name, flower_names_top_k = get_flower_labels(cat_to_name, in_arg.img_path, classes)
    
    
    # Plots image and class probability
    plot_predict(flower_name, in_arg.img_path, probs, flower_names_top_k)
    
    
# Call to main
if __name__ == "__main__":
    main()
