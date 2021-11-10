import torch
import cv2
import urllib

import numpy as np

from PIL import Image
from torchvision import transforms

from torchinfo import summary as info

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image


def extract_output(output, target_category):
    with torch.no_grad():
    #     bounding_boxes = output[0][0]
    #     classes = bounding_boxes[:, 5:]

    #     t1 = classes.max(axis=0)
    #     t2 = t1.values.argmax()
    #     t3 = t1.indices[t2]
        final_tensor = 2
        final_bouding_box = 2 # This is acutally the anchor right?
        final_i = 0
        final_j = 0
        final_c = 0
        max_value = -10000000

        # for t in range( len( output[1] ) ):
        tensor = output[1][final_tensor][0]

        # for b in range( tensor.shape[0] ):
        box = tensor[final_bouding_box]

        for i in range( box.shape[0] ):
            for j in range( box.shape[1] ):
                classes = box[i][j]
                classes = classes[5:]

                for c in range( len( classes ) ):
                    v = classes[c]

                    if v > max_value:
                        max_value = v
                        # final_tensor = t
                        # final_bouding_box = b
                        final_i = i
                        final_j = j
                        final_c = c + 5



    
    # loss = classes[t3][t2]

    return output[1][final_tensor][0][final_bouding_box][final_i][final_j][final_c]

def get_target_category(output):
    # target_category = np.unique(np.array(output.pandas().xyxy[0]['class']))
    # return target_category
    return None

# YOLO
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True, autoshape=False)


for p in model.parameters():
    p.requires_grad = True

# info(model, (32, 3, 1020, 1980), device='cpu')

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

# target_layers = [model.model[3].conv] # Small
# target_layers = [model.model[5].conv] # Medium
target_layers = [model.model[7].conv] # Large

model.to('cpu')
cam = GradCAM(model=model,
            target_layers=target_layers,
            use_cuda=False,
            reshape_transform=None,
            extract_output=extract_output,
            get_target_category=get_target_category)

target_category = 17
grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category, eigen_smooth=False, aug_smooth=False)

# Here grayscale_cam has only one image in the batch
grayscale_cam = grayscale_cam[0, :]

cam_image = show_cam_on_image(input_image, grayscale_cam, use_rgb=False)
cv2.imwrite('yolo_cam.jpg', cam_image)
