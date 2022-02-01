from torch_snippets import *
from PIL import Image
from torchvision import transforms, models, datasets
from torch_snippets import Report
from torchvision.ops import nms
from io import BytesIO
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut, apply_modality_lut
import cv2
import torch
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
from torchvision import transforms, models, datasets
import matplotlib.pyplot as plt
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
PATH='ChestXR31.pt'

label2target = {'Aortic enlargement': 2,'Atelectasis': 8,'Calcification': 12,'Cardiomegaly': 1,'Consolidation': 13,
 'ILD': 4,'Infiltration': 10,'Lung Opacity': 7,'Nodule/Mass': 5,'Other lesion': 9,'Pleural effusion': 11,
 'Pleural thickening': 3,'Pneumothorax': 14,'Pulmonary fibrosis': 6,'background': 0}

target2label = {0: 'background',1: 'Cardiomegaly',2: 'Aortic enlargement',3: 'Pleural thickening',4: 'ILD',
 5: 'Nodule/Mass',6: 'Pulmonary fibrosis',7: 'Lung Opacity',8: 'Atelectasis',9: 'Other lesion',
 10: 'Infiltration',11: 'Pleural effusion',12: 'Calcification',13: 'Consolidation',14: 'Pneumothorax'}

SIZE = (1024, 1024)


def read_normal_image(image_encoded):
    pil_image = Image.open(BytesIO(image_encoded))
    return pil_image


def read_dcm_image(image_encoded, voi_lut = True, fix_monochrome = True):
    image_encoded = BytesIO(image_encoded)
    dicom = pydicom.read_file(image_encoded)
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data

def im_show_dcm(array, size=1024, keep_ratio=False, resample=Image.BILINEAR):
    im = Image.fromarray(array)
    if keep_ratio:
        im.thumbnail((size, size), resample)
    else:
        im = im.resize((size, size), resample)
    return im

def im_show_normal(image):
    image = image.resize(SIZE, Image.BILINEAR)
    return image

def preprocess_dcm(image):
    image = image/255.
    image = np.stack([image, image, image], axis=0)
    image = torch.tensor(image.astype('float32'))
    return image.to(device).float()

def preprocess_normal(image):
    image = image.resize(SIZE, Image.BILINEAR)
    image = np.asarray(image)
    print(image.shape)
    if image.shape[-1]==3:
        image = np.rollaxis(image,2)
        image = torch.tensor(image.astype('float32'))
        return image.to(device).float()
    elif len(image.shape) ==2:
        image = image / 255.
        image = np.stack([image, image, image], axis=0)
        image = torch.tensor(image.astype('float32'))
        return image.to(device).float()
    else:
        image = image[:,:,0] / 255.
        image = np.stack([image, image, image], axis=0)
        image = torch.tensor(image.astype('float32'))
        return image.to(device).float()

def get_model():
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 15)
        return model

model = get_model().to(device)
model.load_state_dict(torch.load(PATH, map_location=torch.device(device)))
#model = torch.load(PATH, map_location=torch.device(device))

from torchvision.ops import nms
def decode_output(output):
    'convert tensors to numpy arrays'
    bbs = output['boxes'].cpu().detach().numpy().astype(np.uint16)
    labels = np.array([target2label[i] for i in output['labels'].cpu().detach().numpy()])
    confs = output['scores'].cpu().detach().numpy()
    ixs = nms(torch.tensor(bbs.astype(np.float32)), torch.tensor(confs), 0.05)
    bbs, confs, labels = [tensor[ixs] for tensor in [bbs, confs, labels]]

    if len(ixs) == 1:
        bbs, confs, labels = [np.array([tensor]) for tensor in [bbs, confs, labels]]
    return bbs.tolist(), confs.tolist(), labels.tolist()

def predict(image):
    model.eval()
    with torch.no_grad():
        image.unsqueeze_(0)
        print(image.shape)
        outputs = model(image)
        for ix, output in enumerate(outputs):
            bbs, confs, labels = decode_output(output)
            info = [f'{l}:{c:.2f}' for l,c in zip(labels, confs)]
            return info, bbs, labels