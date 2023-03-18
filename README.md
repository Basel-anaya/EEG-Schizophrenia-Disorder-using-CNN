# EEG-Schizophrenia-Disorder-using-CNN

### Introduction
This project focuses on using `Convolutional Neural Network (CNN)` to classify EEG signals and detect Schizophrenia disorder. EEG signals from `80+ patients` were collected, with multiple trial data for each patient. The aim is to train a model that can accurately classify EEG signals from patients with Schizophrenia and those without.

### Data
The training signals are stored in two folders named `../input/button-tone-sz` and `../input/buttontonesz2`. Each folder contains EEG signals from different patients. The data is in the form of .mat files, and each file contains a matrix of size `64x7696`, representing `64 channels` and `7696 time points`. The data has already been preprocessed and cleaned.

### Methodology
Preprocessing
Before training the model, the data was preprocessed by filtering the signals in the frequency range of **0.5 Hz** to **30 Hz** to remove any noise or artifacts. The signals were then downsampled to reduce the number of time points and speed up training.

### Model Architecture
The model used in this project is a Convolutional Neural Network (CNN). The input to the model is a `2D matrix of size 64x1924`, representing `64 channels` and `1924 time points`. The model consists of 4 convolutional layers, each followed by a max-pooling layer. The output of the last pooling layer is then flattened and fed into a fully connected layer, which outputs the final classification.

### Training
The model was trained on a `NVIDIA Tesla V100 GPU` using the `Adam optimizer` with a learning rate of **0.001**. The loss function used was binary cross-entropy, as this is a binary classification problem. The model was trained for 50 epochs, with early stopping applied if the validation loss did not improve for 5 consecutive epochs.

### Results
The trained model achieved an accuracy of 92% on the test set, which is a promising result for the detection of Schizophrenia using EEG signals. However, further validation and testing is required before the model can be used in a clinical setting.

### Conclusion
This project demonstrates the potential of using `Convolutional Neural Network (CNN)` to classify `EEG signals` and detect `Schizophrenia disorder`. The trained model achieved a high accuracy on the test set, indicating that the model has learned to distinguish between EEG signals from patients with Schizophrenia and those without. Further research is required to validate the model on larger datasets and in a clinical setting.
