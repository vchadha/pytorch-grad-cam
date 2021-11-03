import torch
import argparse
import cv2
import numpy as np

from torchsummary import summary

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# summary(model, (3, 640, 640))

# img = 'https://ultralytics.com/images/zidane.jpg'
img = 'https://raw.githubusercontent.com/jacobgil/pytorch-grad-cam/master/examples/both.png'

results = model(img)

# Results
results.print()
results.show()

results.xyxy[0]


target_layers = [model.blocks[-1].norm1]

cam = GradCAM(model=model,
            target_layers=target_layers,
            use_cuda=False,
            reshape_transform=None)

# If None, returns the map for the highest scoring category.
# Otherwise, targets the requested category.
target_category = None

# AblationCAM and ScoreCAM have batched implementations.
# You can override the internal batch size for faster computation.
cam.batch_size = 32

input_tensor = img
grayscale_cam = cam(input_tensor=input_tensor,
                    target_category=target_category,
                    eigen_smooth=False,
                    aug_smooth=False)

# Here grayscale_cam has only one image in the batch
grayscale_cam = grayscale_cam[0, :]

cam_image = show_cam_on_image(rgb_img, grayscale_cam)
cv2.imwrite('yolo_cam.jpg', cam_image)
