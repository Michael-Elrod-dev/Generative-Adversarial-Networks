# GAN Implementations for CIFAR-10

This repository contains implementations of three different Generative Adversarial Network (GAN) architectures trained on the CIFAR-10 dataset:
- DCGAN (Deep Convolutional GAN)
- WGAN (Wasserstein GAN)
- ACGAN (Auxiliary Classifier GAN)

## Overview

The project explores various GAN architectures and their effectiveness in generating realistic images from the CIFAR-10 dataset. Each implementation is contained in its own Jupyter notebook:

- `dcgan.ipynb`: Implementation of DCGAN with standard convolutional architecture
- `wgan.ipynb`: Implementation of WGAN with gradient penalty for improved stability
- `acgan.ipynb`: Implementation of ACGAN with conditional generation capabilities

## Requirements

- PyTorch
- torchvision
- numpy
- matplotlib
- scipy
- Jupyter Notebook

## Dataset

The project uses the CIFAR-10 dataset, which consists of 60,000 32x32 color images across 10 different classes. The images are preprocessed to 64x64 resolution and normalized to the range [-1, 1].

## Model Features

### DCGAN
- Transposed convolutions with batch normalization
- Adam optimizer
- Binary cross-entropy loss

### WGAN
- Wasserstein distance as loss function
- Gradient penalty for stability
- RMSprop optimizer
- Multiple critic updates per generator update

### ACGAN
- Auxiliary classifier in discriminator
- Class-conditional generation
- Combined adversarial and classification loss

## Results

The models generate sample images and track various metrics including:
- Generator and discriminator losses
- FID (Fréchet Inception Distance) scores
- Real vs. generated image comparisons
- Wasserstein distance (for WGAN)

Results are saved in the `Results` directory during training.

## Running the Notebooks

1. Install the required dependencies
2. Launch Jupyter Notebook
3. Open any of the implementation notebooks
4. Run all cells to train the model and generate results

Note: Training GANs can be computationally intensive. A GPU is recommended for reasonable training times.