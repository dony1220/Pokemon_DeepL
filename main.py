#!/usr/bin/env python
# coding: utf-8

# In[3]:


from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from pathlib import Path
from PIL import Image
import os
import import_ipynb
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter('ignore')


# In[5]:


from FeatureExtractor import FeatureExtractor
from ImageProcessing import ImageProcessing
from PIL import Image

# img = Image.open("C:/Users/user/Desktop/DL image/pokemon/images/images/abomasnow.png")
dir_path = "C:/Users/user/Desktop/DL image/pokemon/images/images/"
ImageProcessing(dir_path)


# In[25]:





# In[ ]:




