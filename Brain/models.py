# in models.py
from django.db import models

class EEGData(models.Model):
    timestamp = models.DateTimeField()
    delta_tp9 = models.FloatField()
    delta_af7 = models.FloatField()
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18  # Import ResNet model

class ResNetModel(nn.Module):
    def __init__(self, num_classes):
        super(ResNetModel, self).__init__()
        # Load pre-trained ResNet-18 model
        self.resnet = resnet18(pretrained=True)
        # Change the output layer to have num_classes neurons
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Assuming num_classes is the number of classes in your classification task
num_classes = 10  # Change this according to your task
resnet_model = ResNetModel(num_classes)
