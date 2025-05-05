# ⚡ StormPredNet: Lightning Storm Forecasting AI

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-green.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Active%20Development-orange)]()

**A deep learning solution for real-time lightning storm prediction**  
*Finalist solution for the Mock FEMA Challenge 2025*

## 🌩️ Overview

This repository contains our solution for the *FEMA Mock Challenge* — an AI system that predicts lightning storm evolution using:
- Multi-modal sensor data fusion  
- Temporal-spatial deep learning architecture  
- Real-time inference capabilities  

## 🧠 Model Architectures

### Task 1a: Temporal Sequence Prediction using ConvLSTM

For this task, we implemented a deep **Encoder-Decoder ConvLSTM** network that models both the spatial and temporal structure of storm data.

- The encoder consists of multiple stacked `ConvLSTMCell` layers that ingest the input sequence of 12 frames.
- The decoder then autoregressively generates the next 12 frames.
- Each `ConvLSTMCell` performs convolution operations within the LSTM gates to retain spatial features.

This network is particularly effective for spatiotemporal forecasting tasks such as predicting weather map evolution.

> 🔍 See `Task1bConvLSTM` class in the code.

### Task 3: Probabilistic Forecasting with CNN

This model uses a standard **Convolutional Neural Network** to make probabilistic predictions about storm intensity.

- Input: 4-channel spatial data
- The CNN backbone extracts deep spatial features via stacked convolution, batch normalization, ReLU, and pooling layers.
- The output is passed through fully connected layers to estimate:
  - `p_zero`: probability that no lightning occurs
  - `mu_nonzero`: mean intensity when lightning is predicted
  - `sigma_nonzero`: standard deviation (uncertainty) of predicted intensity

This setup allows the model to express uncertainty in its predictions, which is crucial for storm forecasting applications.

> 🔍 See `ProbabilisticCNN_3` class in `models/task3_model.py`.

## 🚀 Quick Start

### Google Colab Setup

```python
!pip install -r requirements.txt
!python -m scripts.train_task1a configs/task1a.yaml # How to train your model
