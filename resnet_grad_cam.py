import torch
from torchsummary import summary

from pytorch_grad_cam import GradCAM

from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50

model = resnet50(pretrained=True)
target_layers = [model.layer4[-1]]

# summary(model, (3, 224, 224))

import urllib
# url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
url, filename = ("https://raw.githubusercontent.com/jacobgil/pytorch-grad-cam/master/examples/both.png", "both.png")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

from PIL import Image
from torchvision import transforms
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

cam = GradCAM(model=model, target_layers=target_layers)
target_category = None

grayscale_cam = cam(input_tensor=input_batch, target_category=target_category)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]

rgb_image = input_image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])

import numpy as np

rgb_image = np.array(preprocess(rgb_image)).astype(np.float32)
rgb_image = rgb_image/255
visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=False)

import cv2
cv2.imwrite('resnet_cam.jpg', visualization)
