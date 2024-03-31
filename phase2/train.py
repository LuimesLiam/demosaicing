from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.io import read_image
import os
from torch import nn
from torch.nn import functional as F
from torch import optim
import torch.utils.data as td
import torchvision as tv
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import functional as TF
from torch.nn.functional import interpolate
import time
import gc
import cv2
import random
from sklearn.metrics import mean_squared_error
from math import sqrt
from model import Model, createUDnCNN
from dataset_prep import Create_Dataset, CFA_pattern



torch.cuda.empty_cache()
gc.collect()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if str(device) == 'cuda:0':
    print("GPU is available.")
else:
    print("GPU is not available, using CPU.")

def train_model_with_early_stopping(training_loader, validation_loader, epochs=500, early_stop_rounds=300, early_stop_delta=0.00000005):
    # Initialize the model, loss function, and optimizer
    # model = Model()

    # loss_function = nn.MSELoss()

    # optimizer = torch.optim.Adam(model.parameters(), lr=model.lr)
    model, loss_function, optimizer= createUDnCNN()

    model.to(device)

    best_validation_loss = float('inf')
    no_improvement_counter = 0

    historical_train_loss = []
    historical_val_loss = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        # Training phase
        for inputs, targets in training_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = loss_function(predictions, targets)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(training_loader)
        historical_train_loss.append(avg_train_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in validation_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                predictions = model(inputs)
                loss = loss_function(predictions, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(validation_loader)
        historical_val_loss.append(avg_val_loss)

        # Early stopping check
        if best_validation_loss - avg_val_loss > early_stop_delta:
            best_validation_loss = avg_val_loss
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1

        if no_improvement_counter >= early_stop_rounds:
            print(f"Early stopping after {epoch + 1} epochs")
            break

        # Log progress
        print(f'Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
    torch.save(model.state_dict(), 'model_state_dict_raw_raw.pth')

    return model, historical_train_loss, historical_val_loss


def train(path='images/'):
    dataSet_obj = Create_Dataset(path, CFA_path='sony/short', ground_truth_path='sony/long', pattern = 'grbg')

    train_loader, test_loader = dataSet_obj.make_dataset()

    X, y = next(iter(train_loader))
    print(X.shape)
    print(y.shape)
    print(len(train_loader))
    print(len(test_loader))

    X, y = next(iter(train_loader))
    print(X.shape)
    print(y.shape)
    print(len(train_loader))
    print(len(test_loader))


    trained_net, training_loss, validation_loss = train_model_with_early_stopping(train_loader,test_loader)


    plt.figure(figsize=(10, 5))
    plt.plot(training_loss, label='Training Loss')
    plt.plot(validation_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

train()