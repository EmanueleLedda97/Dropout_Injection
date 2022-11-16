from torch.utils.data import Dataset
import numpy as np
import constants as keys
import os

# Initializing a random seed
np.random.seed(42)


class UciDataset(Dataset):
    def __init__(self, dataset, mode='training', light_mode=False, split=None):

        self.proportions = (0.6, 0.2, 0.2)  # Split proportions for the data partition
        self.dataset = dataset
        self.mode = mode

        # Guarding the availability of mode and dataset
        if self.mode not in keys.AVAILABLE_DATASET_MODES:
            raise Exception("You must choice an available mode: {} is not available".format(mode))
        if self.dataset not in keys.UCI_DATASETS:
            raise Exception("You must choice an available UCI dataset: {} is not available".format(dataset))

        # Obtaining the data and indices of features and targets from the saved files for the chosen dataset
        dataset_path = os.path.join(keys.UCI_ROOT, self.dataset, 'data')
        data = np.loadtxt(os.path.join(dataset_path, 'data.txt'))
        index_features = np.loadtxt(os.path.join(dataset_path, 'index_features.txt'))
        index_target = np.loadtxt(os.path.join(dataset_path, 'index_target.txt'))

        # If light mode is active we restrict the dataset for a performing much lighter experiments
        if light_mode:
            data = data[:200, :]

        # Getting global normalized data (IID and OOD coincide because we do not have OOD samples)
        self.X_data, self.y_data = data[:, [int(i) for i in index_features]], data[:, int(index_target)]
        self.X_data = (self.X_data - np.mean(self.X_data, axis=0)) / np.std(self.X_data, axis=0)
        self.y_data = (self.y_data - np.mean(self.y_data, axis=0)) / np.std(self.y_data, axis=0)

        # Getting the ids and relative train, validation and test set
        self.ids_iid = np.arange(self.X_data.shape[0])

        # Selecting the indices from pre-saved file
        index_train = np.loadtxt(os.path.join(dataset_path, 'index_train_{}.txt'.format(int(split))))
        index_test = np.loadtxt(os.path.join(dataset_path, 'index_test_{}.txt'.format(int(split))))

        # We now take 1/5 of the training set for validating
        val_train_split = int(index_train.shape[0] / 5)
        self.ids_train = [int(i) for i in index_train[:-val_train_split]]  # From 0 to 4/5
        self.ids_validation = [int(i) for i in index_train[-val_train_split:]]  # From 4/5 to the end
        self.ids_test = [int(i) for i in index_test]

        # We use the computed indices for dividing the actual portions of data
        self.X_train, self.y_train = self.X_data[self.ids_train, :], self.y_data[self.ids_train]
        self.X_validation, self.y_validation = self.X_data[self.ids_validation, :], self.y_data[self.ids_validation]
        self.X_test, self.y_test = self.X_data[self.ids_test, :], self.y_data[self.ids_test]

    def __len__(self):

        dl = None
        if self.mode == 'training':
            dl = self.X_train.shape[0]
        elif self.mode == 'test':
            dl = self.X_test.shape[0]
        elif self.mode == 'validation':
            dl = self.X_validation.shape[0]

        return dl

    def __getitem__(self, idx):

        sample = None
        if self.mode == 'training':
            sample = {'x': self.X_train[idx], 'y': self.y_train[idx]}
        elif self.mode == 'test':
            sample = {'x': self.X_test[idx], 'y': self.y_test[idx]}
        elif self.mode == 'validation':
            sample = {'x': self.X_validation[idx], 'y': self.y_validation[idx]}

        return sample

    def switch_mode(self, new_mode):
        self.mode = new_mode

    def __get_split_ids(self, size):
        # Generating a random permutation
        perm = np.random.permutation(size)

        # Obtaining size of each set
        train_size = int(size * self.proportions[0])
        validation_size = int(size * self.proportions[1])

        # Indexing the permutations
        ids_training = perm[:train_size]
        ids_validation = perm[train_size:(train_size + validation_size)]
        ids_test = perm[(train_size + validation_size):]

        return ids_training, ids_validation, ids_test
