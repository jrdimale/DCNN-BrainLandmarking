# DCNN-BrainLandmarking

Implementation of BMVC 2023 Paper - "Single-Landmark vs. Multi-Landmark Deep Learning Approaches to Brain MRI Landmarking: a Case Study with Healthy Controls and Down Syndrome Individuals"

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Citation](#citation)
- [Contact](#contact)

## Introduction

Code to train and test new models to perform 2D landmarking in Brain Images obtained from Magnetic Ressonance Imaging (MRI). Includes two python scripts, one to train models to predict a single landmark (a single 2D coordinate) and one to train models to predict N different landmarks simultaneously.

The presented paper compares both methods, using an ensemble of 8 single-landmark models, and a multi-landmark model that predicts 8 landmarks.

## Getting Started

The usage of conda is recommended for package installation. A requirements.txt file is included in the repository in order to create a conda environment with the required libraries. 

Dependencies:
- python >= 3.7
- pytorch (with suitable CUDA and CuDNN version)
- opencv
- scikit-image
- sklearn

Fast project setup:

- $ git clone https://github.com/jrdimale/DCNN-BrainLandmarking.git
- $ cd your-project
- $ conda create --name <your_new_env_name> --file requirements.txt

It is important to check compatibilities with your GPU devices, if using different Cuda - pytorch versions. The used versions are:

pytorch==1.5.1=py3.7_cuda10.1.243_cudnn7.6.3_0

## Usage

Modify the scripts accordingly to use them with your own data. There are some TODO's in the files to clarify where you have to add/modify the code based on your data format.

The scripts are first used to train different models. 

Training parameters are defined in the code.

- train
    python single_landmark.py train
    python multi_landmark.py train

- test
    python single_landmark.py test
    python multi_landmark.py test


## Citation

Please cite the paper accordingly if the code was used.

(Paper not published yet - Bibtex text available soon)

Single-Landmark vs. Multi-Landmark Deep Learning Approaches to Brain MRI Landmarking: a Case Study with Healthy Controls and Down Syndrome Individuals

## Contact

If you have any problem about our code, feel free to contact jordi.male@salle.url.edu
