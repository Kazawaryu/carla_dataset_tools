import argparse
import glob
from pathlib import Path


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--yaml','-y',type=str,help='specify the yaml file')
    parser.add_argument('--ckpt','-c',type=str,help='specify the checkpoint file')
    parser.add_argument('--data','-d',type=str,help='specify the point cloud data file or directory')
    
    
    args = parser.parse_args()
  
    return args

def load_model(yaml_file,ckpt_file):

    return
    
def main():
    args = parse_config()
    load_model(args.yaml,args.ckpt)