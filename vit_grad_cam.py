import torch
import argparse
import cv2
import numpy as np

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image

# reshape for vit output?
def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

# Load Vit
model = torch.hub.load('facebookresearch/deit:main',
                          'deit_tiny_patch16_224', pretrained=True)

target_layers = [model.blocks[-1].norm1]

cam = GradCAM(model=model,
            target_layers=target_layers,
            use_cuda=False,
            # reshape_transform=None)
            reshape_transform=reshape_transform)

# image_path = './examples/both.png'
image_path = './examples/dog_cat.jfif'
rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
rgb_img = cv2.resize(rgb_img, (224, 224))
rgb_img = np.float32(rgb_img) / 255
input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])

# compute the predictions
out = model(input_tensor)

# and convert them into probabilities
scores = torch.nn.functional.softmax(out, dim=-1)[0]

# finally get the index of the prediction with highest score
topk_scores, topk_label = torch.topk(scores, k=5, dim=-1)

with open("imagenet_classes.txt", "r") as f:
    imagenet_categories = [s.strip() for s in f.readlines()]

for i in range(5):
  pred_name = imagenet_categories[topk_label[i]]
  print(f"Prediction index {i}: {pred_name:<25}, score: {topk_scores[i].item():.3f}")

# If None, returns the map for the highest scoring category.
# Otherwise, targets the requested category.
target_category = None

# AblationCAM and ScoreCAM have batched implementations.
# You can override the internal batch size for faster computation.
cam.batch_size = 32

grayscale_cam = cam(input_tensor=input_tensor,
                    target_category=target_category,
                    eigen_smooth=False,
                    aug_smooth=False)

# Here grayscale_cam has only one image in the batch
grayscale_cam = grayscale_cam[0, :]

cam_image = show_cam_on_image(rgb_img, grayscale_cam)
cv2.imwrite('vit_cam.jpg', cam_image)
