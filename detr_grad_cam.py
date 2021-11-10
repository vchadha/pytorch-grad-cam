import math
import cv2
import numpy as np

from PIL import Image
import requests
import matplotlib.pyplot as plt

import ipywidgets as widgets
from IPython.display import display, clear_output

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
# torch.set_grad_enabled(False);

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# reshape for vit output?
def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def extract_output(output, target_category):
    with torch.no_grad():
        logits_bbox_id = output['pred_logits'][0, :, target_category].argmax()

    loss = output['pred_logits'][0, logits_bbox_id, target_category]

    return loss

def get_target_category( output ):
    target_category = np.unique(np.argmax(output['pred_logits'].cpu().data.numpy(), axis=-1))
    return target_category

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)

for p in model.parameters():
    p.requires_grad = True

# print(model.transformer.decoder.norm)

# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# url = 'https://live-production.wcms.abc-cdn.net.au/09fc4d971d2ae450966dd606be4d3587?impolicy=wcms_crop_resize&cropH=1152&cropW=2048&xPos=0&yPos=417&width=862&height=485'
url = 'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.nme.com%2Fwp-content%2Fuploads%2F2019%2F07%2Fcats-2019-movie-taylor-swift-credit-universal-pictures%402000x1270.jpg&f=1&nofb=1'
im = Image.open(requests.get(url, stream=True).raw)

# mean-std normalize the input image (batch-size: 1)
img = transform(im).unsqueeze(0)

# propagate through the model
outputs = model(img)

# keep only predictions with 0.7+ confidence
probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
keep = probas.max(-1).values > 0.9

# convert boxes from [0; 1] to image scales
bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

plot_results(im, probas[keep], bboxes_scaled)

# Grad cam time!
# target_layers = [model.blocks[-1].norm1]
# target_layers = [model.transformer.decoder.norm]
target_layers = [model.input_proj]
# target_layers = [model.backbone[-2]]
# target_layers = [model.transformer.encoder.layers[-1].self_attn]
# target_layers = [model.transformer.decoder.layers[-1].multihead_attn]

print( img.shape )

cam = GradCAM(model=model,
            target_layers=target_layers,
            use_cuda=False,
            reshape_transform=None,
            extract_output=extract_output,
            get_target_category=get_target_category)
            # reshape_transform=reshape_transform)

# If None, returns the map for the highest scoring category.
# Otherwise, targets the requested category.
target_category = 17
# target_category = None

# AblationCAM and ScoreCAM have batched implementations.
# You can override the internal batch size for faster computation.
cam.batch_size = 32

grayscale_cam = cam(input_tensor=img,
                    target_category=target_category,
                    eigen_smooth=False,
                    aug_smooth=False)

# Here grayscale_cam has only one image in the batch
grayscale_cam = grayscale_cam[0, :]

rgb_image = im
preprocess = T.Compose([
    T.Resize(800),
])

rgb_image = np.array(preprocess(rgb_image)).astype(np.float32)
rgb_image = rgb_image/255
visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=False)

cv2.imwrite('detr_cam.jpg', visualization)
