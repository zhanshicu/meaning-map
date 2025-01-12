################################################################################
# This script generates the attentional map from eye movement data
# and requires the following data file:
# "HendersonHayes17-NHB_eyetracking_data.mat"; 
# "gonme.jpg" -> an example scene image

# the output will be the following data file:
# "gnome-attentional-map.png"

# Zhan Shi, Zhicheng Lin , Jan 5, 2025
################################################################################

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from antonioGaussian import antonio_gaussian
from scipy.ndimage import gaussian_filter

# read the data
mat = scipy.io.loadmat('HendersonHayes17-NHB_eyetracking_data.mat')

D = mat['D']

image_name = 'gnome.jpg'
subject_ids = [int(sbj[0][0]) for sbj in D['sbj'][0]]

# Construct a fixation matrix
fixation_matrix = np.zeros((768, 1024))

for sbj_id in subject_ids:
    subject_data = D[D['sbj'] == sbj_id]

    subject_images = [img[0][0] for img in subject_data['image'][0]]

    # Find the index of the image
    image_index = None
    for i, img in enumerate(subject_images):
        if img == image_name:
            image_index = i
            break

    # Extract fixation coordinates and durations
    subject_locs = [loc[0] for loc in subject_data['loc'][0]]
    subject_durs = [dur[0] for dur in subject_data['dur'][0]]

    scene_locs = subject_locs[image_index]
    scene_durs = subject_durs[image_index]
    scene_locs = np.array(scene_locs)
    scene_durs = np.array(scene_durs)
    for (x, y) in scene_locs:
        x = min(int(x), 767)  # Limit x to 767
        y = min(int(y), 1023)  # Limit y to 1023
        fixation_matrix[x, y] += 1
        # fixation_matrix[int(x), int(y)] += 1

# Filter the fixation matrix
filtered_fixation_matrix, _ = antonio_gaussian(fixation_matrix, 6)

save_path = "gnome-attentional-map.png"

if save_path:
    plt.imsave(save_path, filtered_fixation_matrix, cmap='viridis')
