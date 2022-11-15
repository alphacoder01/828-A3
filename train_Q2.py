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

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
# coco_metadata = MetadataCatalog.get("coco_2017_val_panoptic")

# import Mask2Former project
from mask2former import add_maskformer2_config
from detectron2.data import DatasetCatalog


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



classes =  ["background",
    "Bipolar_Forceps",
    "Prograsp_Forceps",
    "Large_Needle_Driver",
    "Monopolar_Curved_Scissors",
    "Ultrasound_Probe",
    "Suction_Instrument",
    "Clip_Applier"]



for d in ["train", "val"]:
    DatasetCatalog.register(f"{d}_Ashish", lambda d=d: get_data_dict(f"./Modified_data/content/COL828_Data_for_Assignment_3/{d}/images"))
    MetadataCatalog.get(f"{d}_Ashish").set(stuff_classes=classes, ignore_label=8,evaluator_type='sem_seg')
# balloon_metadata = MetadataCatalog.get("train_sem2")



from train_net import Trainer
import os
import warnings
warnings.filterwarnings("ignore")

cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
#instance
# cfg.merge_from_file("/content/Mask2Former/configs/coco/instance-segmentation/swin/maskformer2_swin_small_bs16_50ep.yaml")


cfg.merge_from_file("./configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_small_bs16_90k.yaml")


cfg.DATASETS.TRAIN = ("train_Ashish",)
cfg.DATASETS.TEST = ("val_Ashish","train_Ashish",)
cfg.DATALOADER.NUM_WORKERS = 8
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES=7
#instance
# cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_small_bs16_50ep/model_final_1e7f22.pkl" 
cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/semantic/maskformer2_swin_small_bs16_90k/model_final_fa26ae.pkl"
cfg.TEST.AUG.ENABLED = True


cfg.SOLVER.IMS_PER_BATCH = 8  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 20000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

cfg.OUTPUT_DIR = "./Outputs/Q2-Swin-S-20k"
print(cfg.OUTPUT_DIR)
# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = Trainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()