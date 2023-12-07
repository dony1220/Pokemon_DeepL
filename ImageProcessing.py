# dir_path = "C:/Users/user/Desktop/DL image/pokemon/images/images/"

import os
from FeatureExtractor import FeatureExtractor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def ImageProcessing(dir_path):
    fe = FeatureExtractor()
    features = []
    img_paths = []
    for img_name in sorted(os.listdir(dir_path)):
    # Extract Features
        img_path = os.path.join(dir_path, img_name)
        feature = fe.extract(img=Image.open(img_path))
        img_paths.append(img_path)
        features.append(feature)
# Import the libraries

    img = Image.open("C:/Users/user/Desktop/DL image/pokemon/images/images/abomasnow.png")
# Extract its features
    query = fe.extract(img)

# # Calculate the similarity (distance) between images
    dists = np.linalg.norm(features - query, axis=1)

# Extract 30 images that have lowest distance
    ids = np.argsort(dists)[:30]

    scores = [(dists[id], img_paths[id], id) for id in ids]
# Visualize the result
    axes=[]
    fig=plt.figure(figsize=(8,8))
    for a in range(5*6):
        score = scores[a]
        axes.append(fig.add_subplot(5, 6, a+1))
        subplot_title=str(round(score[0],2)) + "/m" + str(score[2]+1)
        axes[-1].set_title(subplot_title)  
        plt.axis('off')
        plt.imshow(Image.open(score[1]))
    fig.tight_layout()
    plt.show()