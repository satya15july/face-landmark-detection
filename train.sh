#!/bin/bash
set -e
### All of your tmp data will be saved in ./tmp folder

echo "Hello! I will prepare training data and starting to training step by step."

# 1. checking dataset if OK
if [ ! -d "./dataset/WIDER_train/images" ]; then
	echo "Error: The WIDER_train/images is not exist. Read dataset/README.md to get useful info."
	exit
fi
if [ ! -d "./dataset/lfw_5590" ]; then
	echo "Error: The lfw_5590 is not exist. Read dataset/README.md to get useful info."
	exit
fi
echo "Checking dataset pass."
if [ -d "./tmp" ]; then
	echo "Warning: The tmp folder is not empty. A good idea is to run ./clearAll.sh to clear it before training."
fi

### start to training P-Net
echo "Start to training P-Net"
python3.5 training/train.py --stage=pnet

### start to training R-Net
echo "Start to training R-Net"
python3.5 training/train.py --stage=rnet

### start to training O-Net
echo "Start to training O-Net"
python3.5 training/train.py --stage=onet

# 5. Done
echo "Congratulation! All stages had been done. Now you can going to testing and hope you enjoy your result."
echo "haha...bye bye"