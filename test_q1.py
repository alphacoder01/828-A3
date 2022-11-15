import torch 
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



def get_data_dict(img_dir):
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
    record["sem_seg_file_name"]  = filename.replace("images","annotations")
    dataset_dicts.append(record)

  return dataset_dicts




if __name__ == '__main__':

  img_dir = sys.argv[1]

  classes =  ["background",
        "Bipolar_Forceps",
        "Prograsp_Forceps",
        "Large_Needle_Driver",
        "Monopolar_Curved_Scissors",
        "Ultrasound_Probe",
        "Suction_Instrument",
        "Clip_Applier"]

  test_data = []
  idx = 0

  for file_name in os.listdir(img_dir):
    path = os.path.join(img_dir, file_name)
    test_data.append({"file_name":path, "image_id":idx})
    idx += 1

  cfg = get_cfg()
  add_deeplab_config(cfg)
  add_maskformer2_config(cfg)
  cfg.merge_from_file("./configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_small_bs16_90k.yaml")
  cfg.MODEL.WEIGHTS = "./Outputs/Swin-S-20k/model_0019999.pth"
  cfg.DATASETS.TRAIN = ("train_Ashish",)
  cfg.DATASETS.TEST = ("val_Ashish",)
  cfg.TEST.AUG.ENABLED = True

  classes =  ["background",
    "Bipolar_Forceps",
    "Prograsp_Forceps",
    "Large_Needle_Driver",
    "Monopolar_Curved_Scissors",
    "Ultrasound_Probe",
    "Suction_Instrument",
    "Clip_Applier"]

  MetadataCatalog.get(f"val_Ashish").set(stuff_classes=classes, ignore_label=8,evaluator_type='sem_seg')
  meta = MetadataCatalog.get("val_Ashish")


  # predictor = DefaultPredictor(cfg);
  # # print(test_data[0]['file_name'])
  # im = cv2.cvtColor(cv2.imread(test_data[0]['file_name']),cv2.COLOR_BGR2RGB)/255.
  # outputs = predictor(im)
  # # print(outputs['sem_seg'].shape)
  # v = Visualizer(im[:, :, ::-1],
  #              metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
  #              scale=0.5)

  # semantic_result = v.draw_sem_seg(outputs["sem_seg"].argmax(0).to("cpu")).get_image()
  # img = cv2.cvtColor(semantic_result[:, :, ::-1], cv2.COLOR_RGBA2RGB)
  # cv2.imwrite("./test_img.png",img)


import random
predictor = DefaultPredictor(cfg)
dataset_dicts = get_data_dict("../../datasets/Surgical_Instrument_Seg/val/images/")
idx = 0
for d in random.sample(dataset_dicts, 3):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=meta, 
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_sem_seg(outputs["sem_seg"].argmax(0).to("cpu"))
    cv2.imwrite(f"./{idx}.png", out.get_image()[:, :, ::-1])