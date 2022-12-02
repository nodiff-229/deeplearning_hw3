import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_model",type=str, default="")
args = parser.parse_args()

import os 
os.system(f"bash anylearn_starter/classifier_sample.sh {args.pretrained_model}")
