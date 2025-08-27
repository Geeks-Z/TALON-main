import json
import argparse
from trainer import train
import os
def main():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args) # Converting argparse Namespace to a dict.
    args.update(param) # Add parameters from json

    train(args)

def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple pre-trained incremental learning algorthms.')
    parser.add_argument('--config', type=str, default='./scripts/talon_cub_B100_Inc10_debug.json',
                        help='Json file of settings.')
    return parser

if __name__ == '__main__':
    main()
