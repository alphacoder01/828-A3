import torch 
import pickle
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
setup_logger(name="mask2former")

# import some common libraries
import numpy as np
import cv2
import torch
import os
import sys

from detectron2.utils.visualizer import Visualizer, ColorMode
import matplotlib.pyplot as plt

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config

from mask2former import add_maskformer2_config
from detectron2.data import DatasetCatalog

from train_net import Trainer
import os
import warnings
warnings.filterwarnings("ignore")
import albumentations as A


def get_data_dict(img_dir,anno_dir):
  img_files = os.listdir(img_dir)
  dataset_dicts = []
  for idx, img_file in enumerate(img_files):
    record = {}
    filename = os.path.join(img_dir,img_file)
    h,w = cv2.imread(filename).shape[:2]

    record["file_name"] = filename
    record["image_id"] = idx
    record["height"] = h
    record["width"] = w
    record["sem_seg_file_name"]  = os.path.join(anno_dir,img_file)
    dataset_dicts.append(record)

  return dataset_dicts




if __name__ == '__main__':
  from PIL import Image

  img_dir = sys.argv[1]
  anno_dir = sys.argv[2]
  question = int(sys.argv[3])

  if question == 2:
    os.makedirs('./temp_img_dir',exist_ok=True)
    for img in os.listdir(img_dir):
      path = os.path.join(img_dir,img)
      tf = A.Compose([
        A.CLAHE(always_apply=True),
        A.Sharpen(always_apply=True)
      ])

      timg = tf(image=np.array(Image.open(path).convert("RGB")))['image']
      cv2.imwrite(f"./temp_img_dir/{img}",timg[:,:,::-1])
    img_dir = './temp_img_dir'


  ckpt = "./ckpt/Q1.pth" if question == 1 else './ckpt/Q2.pth'
  print(ckpt)
  os.system(f'python train_net.py --config-file ./configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_small_bs16_90k.yaml@{img_dir}___{anno_dir} --eval-only MODEL.WEIGHTS {ckpt} MODEL.SEM_SEG_HEAD.NUM_CLASSES 7 TEST.AUG.ENABLED False')


