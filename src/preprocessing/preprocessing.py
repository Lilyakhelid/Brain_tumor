# %%
import os
import yaml
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from module_for_preprocessing import *



with open('../../config.yml', 'r') as file:
    config = yaml.safe_load(file)

# %%

subfolders = ['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']
size = config['données']['image']['size']
for subfolder in subfolders:
    resize_and_rename_images_of_folder('Training',subfolder,size)
    resize_and_rename_images_of_folder('Testing',subfolder,size)
# %%




