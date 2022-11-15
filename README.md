# Installation Instructions

- conda create --name mask2former python=3.8 -y
- conda activate mask2former
- conda install pytorch==1.9.0 torchvision==0.10.0 
- cudatoolkit=11.1 -c pytorch -c nvidia
- pip install -U opencv-python

### under your working directory
- git clone https://github.com/facebookresearch/detectron2.git
- cd detectron2
- pip install -e .
- pip install git+https://github.com/cocodataset/panopticapi.git
- pip install git+https://github.com/mcordts/cityscapesScripts.git

- cd ..
- git https://github.com/facebookresearch/Mask2Former.git
- cd Mask2Former
- pip install -r requirements.txt
- conda install -c conda-forge cudatoolkit-dev
- cd mask2former/modeling/pixel_decoder/ops
- sh make.sh



# To Test:
- Download the checkpoints from [here](https://drive.google.com/drive/folders/1HOPkNu-PNNKSOTITbIfs6gE1IiINOigD?usp=share_link) and place them in ckpt(have to create this) directory in Mask2Former directory.
- python test_q1.py path2imageDir
- python test_q2.py path2imageDir
