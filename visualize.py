
import torch
from datamodule import DataModule

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np


dm = DataModule()
dm.prepare_data()
dm.setup()

train_dataloader = dm.train_dataloader()

# fix too many files open torch error
torch.multiprocessing.set_sharing_strategy('file_system')

# extract steering angles from dataset
steering_angles = []
for batch_idx, (image, steering_angle) in enumerate(train_dataloader):
    steering_angles.append(steering_angle)

# plot their distribution
sns.set_theme(); np.random.seed(0)
steering_angles = np.array(steering_angles)
ax = sns.histplot(steering_angles)

plt.xlabel("steering angle")  
plt.ylabel("count")  
plt.show()


