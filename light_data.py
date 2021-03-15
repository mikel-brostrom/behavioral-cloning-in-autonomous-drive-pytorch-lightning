import pickle
import os
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import CarDataset


class DataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.X_train, self.X_valid, self.y_train, self.y_valid = None, None, None, None

    def prepare_data(self):
        #reads CSV file into a single dataframe variable
        data_df = pd.read_csv(os.path.join(os.getcwd(), 
                                           './data', 
                                           'driving_log.csv'), 
                                           names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

        #yay dataframes, we can select rows and columns by their names
        #we'll store the camera images as our input data
        X = data_df[['center', 'left', 'right']].stack().values

        print('TOTAL number of input instances: ', X.shape[0])
        #and our steering commands as our output data
        y = data_df['steering'].repeat(3).values
        for idx, y_ in enumerate(y):
            # center images
            if idx % 3 == 0:
                y[idx] = y[idx]
            # left images, add positive offset to that it is steared to centrum
            if idx % 3 == 1:
                y[idx] = y[idx] + 0.1
            # right images, add negative offset to that it is steared to centrum
            if idx % 3 == 2:
                y[idx] = y[idx] - 0.1
        print('TOTAL number of target instances: ', y.shape[0])
        #now we can split the data into a training (80), testing(20), and validation set
        #thanks scikit learn
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)
        
    def setup(self):
        self.train = CarDataset(self.X_train, self.y_train, is_training=True)
        print('Number of target instances after split in TRAIN: ', len(self.train))
        self.valid = CarDataset(self.X_valid, self.y_valid, is_training=True)
        print('Number of target instances after split in VALID: ', len(self.valid))

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=32,
            shuffle=True,
            num_workers=8,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid,
            batch_size=32,
            shuffle=False,
            num_workers=8,
        )


def main():
    dm = DataModule()
    dm.prepare_data()
    dm.setup()


if __name__ == "__main__":
    main()