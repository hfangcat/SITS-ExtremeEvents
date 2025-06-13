from test import main
import os
import json
import numpy as np

from main import load_config, deep_update

import argparse

# Input arguments
parser = argparse.ArgumentParser(description='Run the test script')
parser.add_argument('--default_config', type=str, help='Path to the default config file')
parser.add_argument('--subconfig', type=str, help='Path to the subconfig file')
parser.add_argument('--checkpoint_folder', type=str, help='Path to the folder containing the checkpoints')
parser.add_argument('--dataset', type=str, help='Name of the dataset')

args = parser.parse_args()

seed_list = [42, 43, 44]

# Identify the config folder
config_folder = os.path.dirname(args.default_config)

concat_dict = {}

for seed in seed_list:
    # Load the config files
    config = load_config(args.default_config)
    deep_update(config, load_config(args.subconfig))

    exp_name = config['exp_name']
    
    model_path = os.path.join(args.checkpoint_folder, \
                              'checkpoint_' + exp_name + '_' + str(seed) + '.pth')
    
    print("testing model: ", model_path)
    print("Experiment name: ", exp_name)
    print("Seed: ", seed)

    

    dict = main(args.default_config, args.subconfig, model_path)

    # Save the results in a dictionary
    concat_dict[seed] = dict

    # Save the results in a json file
    json_folder = os.path.join("results", args.dataset)
    if not os.path.exists(json_folder):
        os.makedirs(json_folder)

    json_file = os.path.join(json_folder, exp_name + '_' + str(seed) + '.json')

    with open(json_file, 'w') as f:
        json.dump(dict, f, indent=4)

    print("Results saved in: ", json_file)


# calculate the mean and std of the results
# Get list of metrics
metrics = concat_dict[42].keys()
aggregate_dict = {}

for metric in metrics:
    values = [concat_dict[seed][metric] for seed in seed_list]

    # Calculate mean and std, save in mean +- std format
    mean_value = np.mean(values)
    std_value = np.std(values)

    aggregate_dict[metric] = f'{mean_value:.4f} +- {std_value:.4f}'

# Save the aggregate results in a json file
json_file = os.path.join(json_folder, exp_name + '_aggregate.json')

with open(json_file, 'w') as f:
    json.dump(aggregate_dict, f, indent=4)

print("Aggregate results saved in: ", json_file)