# EfficientNet_LearningVisionIntelligence

This repository contains the implementation of training an EfficientNet-B0 model using the CIFAR-100 dataset. The model is trained with distributed data parallelism and includes features such as data augmentation, test-time augmentation, and learning rate scheduling.

## Requirements

To run this project, you need the following libraries:

- `torch`
- `torchvision`
- `albumentations`
- `matplotlib`
- `numpy`
- `tqdm`

Install the necessary libraries using:
  pip install torch torchvision albumentations matplotlib tqdm numpy


## Train Model:
To train the model, follow these steps:

1. Clone the repository:
    git clone <repository-url>
    cd <repository-directory>
    
2. Configure the following hyperparameters and configure relevant paths:

    Number of Epochs: 90
    Batch Size: 32
    Learning Rate: Automatically scaled based on the batch size, starting at 0.1 for 8 devices.
    Weight Decay: 1e-4
    Momentum: 0.9
    Test Time Augmentation (TTA): Enabled by default

All hyperparameters are located at the top of the script for easy modification.

3. Train the model:
  python EfficientNet.py

## Requirements
After training, the model is evaluated on the test set of CIFAR-100. The following metrics are computed:

    Top-1 Accuracy
    Top-5 Accuracy
    Superclass Accuracy

Run the evaluation by loading the best checkpoint, which is automatically saved to:
..............

## Random Seed
The code uses a fixed random seed for dataset splitting and transformations to ensure reproducibility:

Random Seed: 542
To change the random seed, modify the TRAIN_VAL_SPLIT_SEED variable in the script.

## Model Checkpoints
The best model is saved automatically during training to:
.....
