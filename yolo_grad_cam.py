import torch
import cv2
import urllib

import numpy as np

from PIL import Image
from torchvision import transforms

from torchinfo import summary as info
# from torchsummary import summary

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image

# Classify stuff
# import torch.nn as nn
# class Classify(nn.Module):
#     # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
#         super().__init__()
#         self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
#         self.flat = nn.Flatten()

#     def forward(self, x):
#         z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
#         return self.flat(self.conv(z))  # flatten to x(b,c2)

# def autopad(k, p=None):  # kernel, padding
#     # Pad to 'same'
#     if p is None:
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
#     return p

# YOLO
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True, autoshape=False)

# model.model = model.model[:8]

# Replace last layer with classify
# last_layer = model.model[-1]
# ch = last_layer.conv.in_channels if hasattr(last_layer, 'conv') else sum([x.in_channels for x in last_layer.m])
# c = Classify(ch, 80)  # Classify()
# c.i, c.f, c.type = last_layer.i, last_layer.f, 'models.common.Classify'  # index, from, type
# model.model[-1] = c  # replace

for p in model.parameters():
    p.requires_grad = True

# info(model, (32, 3, 1020, 1980), device='cpu')
# summary(model, (3, 640, 640))

# Read in image
image_path = './examples/both.png'
# image_path = './examples/dog_cat.jfif'
img = cv2.imread(image_path, 1)[:, :, ::-1]
input_image = cv2.resize(img, (224, 224))
input_image = np.float32(input_image) / 255
input_tensor = preprocess_image(input_image, mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])
# input_tensor = preprocess_image(input_image)
print(input_tensor.shape)

# Eval image
results = model(input_tensor)

target_layers = [model.model[7]]

model.to('cpu')
cam = GradCAM(model=model, target_layers=target_layers)

target_category = None
grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category, eigen_smooth=False, aug_smooth=False)

# Here grayscale_cam has only one image in the batch
grayscale_cam = grayscale_cam[0, :]

cam_image = show_cam_on_image(input_image, grayscale_cam)
cv2.imwrite('yolo_cam.jpg', cam_image)
