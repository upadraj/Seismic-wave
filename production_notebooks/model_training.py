#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import torch
import random
import torchmetrics

sys.path.append(os.path.abspath(os.path.join(r'../../Seismic-wave/')))

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from models.cnn import CNN, ImageCNN
from models.resnet import ResNet, PretrainedResNet
from models.loss import FocalLoss
from torchvision import datasets,transforms
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from torch.utils.data import DataLoader, TensorDataset, random_split
from utils.data_utils import count_unique_colum_and_vlaues, znorm


# In[2]:


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


# In[3]:


# Hyperparameters

batch_size = 32
initial_lr = 1e-4
base_lr = 1e-4
warmup_steps = 30
epochs = 10

model = ImageCNN()
model = model.to(device=device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
# criterion = FocalLoss()
optimizer = optim.Rprop(model.parameters(), lr=initial_lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# In[4]:


#  Time Series data loading.
# Uncomment everything on this code shell and commnet out the code shell after this cell to run as time series data.

# train_features = np.load("../data/processed/balance_16k/model_input/train_features.npy")
# train_labels = np.load("../data/processed/balance_16k/model_input/train_labels.npy")
# test_features = np.load("../data/processed/balance_16k/model_input/test_features.npy")
# test_labels= np.load("../data/processed/balance_16k/model_input/test_labels.npy")

# train_features = torch.from_numpy(train_features).float()
# train_features = train_features.to(device)
# train_labels = torch.from_numpy(train_labels)
# train_labels = train_labels.to(device)

# test_features = torch.from_numpy(test_features).float()
# test_features = test_features.to(device)
# test_labels = torch.from_numpy(test_labels)
# test_labels = test_labels.to(device)


# dataset = TensorDataset(train_features, train_labels)
# train_loader= DataLoader(dataset, batch_size=batch_size, shuffle=True)

# test_dataset = TensorDataset(test_features, test_labels)
# test_dataloader = DataLoader(test_dataset)


# In[5]:


# Image processing
# Uncomment everything on this code shell and commnet out the code shell before this cell to run as image data.

image_folder = "../spectrogram_images/"
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    
])
path = "../spectrogram_images/"
dataset = datasets.ImageFolder(root=image_folder, transform=transform)


dataset_size = len(dataset)
train_size = int(0.8 * dataset_size) 
test_size = dataset_size - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders for both training and testing
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size= batch_size, shuffle=False)

data_loader = DataLoader(dataset,batch_size=32, shuffle=True)


# In[6]:


# print(f"Train features: {train_features.shape}, Train label: {train_labels.shape} ")
# print(f"Test features: {test_features.shape}, Test label: {test_labels.shape} ")
# train_min, train_max = train_features.min(), train_features.max()
# print((train_min, train_max))

# unique_value, count = count_unique_colum_and_vlaues(test_labels.to(device='cpu'))
# for x in range(len(count)):
#     print(unique_value[x], count[x])

# print(train_features[0][0][0])
# # inputs = 2 * (inputs - x_min) / (x_max - x_min) -1


# In[7]:


def train_model(
    model=model,
    dataloader=train_loader,
    device=device,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    epochs=epochs,
    warmup_steps=warmup_steps,
    initial_lr=initial_lr,
    base_lr=base_lr,
):
    model = model.to(device=device)

    model.train()
    for epoch in range(epochs):
        if epoch < warmup_steps:
            lr = initial_lr + (base_lr - initial_lr) * (epoch / warmup_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        else:
            scheduler.step()
            lr = scheduler.get_last_lr()[0]

        batch_loss = 0.0
        for batch_inputs, batch_targets in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()

        avg_loss = batch_loss / len(train_dataset)
        print(
            f"Epoch {epoch+1}/{epochs}, Cross Entropy Loss: {avg_loss:.4f}, Learning Rate: {initial_lr:.6f}"
        )
        
def calculate_accuracy_and_probabilities(model, dataloader, num_classes):
    model.eval()
    total_correct = 0
    total_samples = 0
    class_correct = torch.zeros(num_classes)
    class_samples = torch.zeros(num_classes)

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            probs = F.softmax(outputs, dim=1)

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

            _, predictions = torch.max(outputs, 1)

            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            for i in range(len(labels)):
                label = labels[i]
                if predictions[i] == label:
                    class_correct[label] += 1
                class_samples[label] += 1

    y_prob = torch.cat(all_probs, dim=0)
    y_true = torch.cat(all_labels, dim=0)

    total_accuracy = total_correct / total_samples * 100
    class_accuracy = class_correct / class_samples * 100

    return total_accuracy, class_accuracy, y_prob, y_true


# In[8]:


model


# In[9]:


train_model()
num_classes = 6
total_accuracy, class_accuracy, y_prob, y_true = calculate_accuracy_and_probabilities(
    model, dataloader=test_loader, num_classes=num_classes
)

print(f"Total Accuracy: {total_accuracy:.2f}%")
for i in range(num_classes):
    print(f"Accuracy of class {i}: {class_accuracy[i]:.2f}%")


# In[10]:


# Saving the model
saved_model_pth = "../trained_models/Image_CNN.pth"
torch.save(model, saved_model_pth)

# loading the saved model
# model = torch.load(saved_model_pth)

# model.eval()

