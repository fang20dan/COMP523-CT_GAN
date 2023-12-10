# COMP523-CT_GAN
## Overview
Here you'll find our code for a GAN model that segments Brain MRI images for tumorous regions. This model was trained on the BraTS dataset and you can find the training code called `GAN4Seg_TrainingImpl.py` or the pretrained model called `model_x2_1.h5`.
## Instructions
Prior to training the model or running command line interface, be sure to install the necessary packages - can be found at the top of the `GAN4Seg_TrainingImpl.py` file. To train the model, run the `GAN4Seg_TrainingImpl.py` with desired inputs. To use trained models to segment brain images, run the `gan4seg_evalviz2.py` file in command line:
```
python gan4seg_evalviz2.py PATH_OF_NII_FILE_1.nii PATH_OF_NII_FILE_2.nii
```
This should return both a window view of segmented image as well as save a .jpeg file to your computer.
