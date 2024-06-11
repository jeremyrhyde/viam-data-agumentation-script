# Viam Data Augmentation Training Script

This repo contains a custom training script for Viam's ML pipeline that adds a data augmentation step/layer to an EfficentDet ML model (tflite).

For this script to run on a Viam dataset, the contents of `setup.py` and `model/training.py` should remain relatively static. `model/training_utils.py` contains additional functions for parsing and organizing data into the TensorFlow type definition. 

`training_custom.py` includes the custom data augmentation layer that has been added to the base EfficentDet model that is being run in this training script. 


## Data Augmentation

Data augmentation is a common method used in machine learning applications to increase the robustness of your model to changes in environmental conditions. These can include such things as changes to viewpoint and lighting conditions. 

For example, if training a model to detect dogs, you may need to augment your data to handle low and high levels of lighting, different levels of zoom, or the angle of image capture.