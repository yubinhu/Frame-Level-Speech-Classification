# Frame-Level-Speech-Classification
Kaggle Competition
https://www.kaggle.com/c/idl-fall2021-hw1p2
(CMU Introduction to Deep Learning 11-785 HW1P2)

## Directories
Trained models are stored in "Models"  
Model and training settings are listed in Models.xlsx

## Data
- mel-spectrum data labeled at each time frame (10ms duration) by the phoneme (71 possible in total) pronounced at each time.   
- Training dataset size: 5GB

## Methods
- A deep neural network that reads in the mel-spectrum at a time frame (possibly with some "context" - earlier and later time frames) and predicts the phoneme category

## Technologies/Hardware
- PyTorch
- GPU (num: 1) instance with ~2GHz cpu

## Current Best Results
- 1MB model with testing accuracy 65% (context size: 10)