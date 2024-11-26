""" To be executed on your own PC, not on Jetson"""
import torch
from torchvision import models

# Load pretrained VGG16
vgg16 = models.vgg16(pretrained=True)

# Save the model's weights
torch.save(vgg16.state_dict(), 'vgg16_weights.pth')
print("VGG16 weights downloaded and saved as 'vgg16_weights.pth'")
