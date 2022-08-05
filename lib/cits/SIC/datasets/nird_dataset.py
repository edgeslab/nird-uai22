import pdb
import math
import torch
import numpy as np
import pandas as pd
import torch.utils.data



class NirdDataset(torch.utils.data.Dataset):
    """Dataset imported from NIRD experiment
    """

    def __init__(self, opt, shuffle_targets = False, X = None, Y = None, Z = None, data = None, targets = None, seed = 31):
        # Random number generator
        self.rng = np.random.RandomState(seed)

        self.shuffle_targets = shuffle_targets
        self.data            = data
        self.targets         = targets
        self.features        = ['x%d'%i for i in range(opt.Xdim)]
        self.fea_groundtruth = [0]

        if self.data is None:
            # x = pd.read_csv('lib/cits/SIC/datasets/nird/null/X.csv')
            # z = pd.read_csv('lib/cits/SIC/datasets/nird/null/Z.csv')
            if Z is not None:
                self.data = torch.Tensor(np.concatenate((X, Z), axis=1))
            else:
                self.data = torch.Tensor(X)

        if self.targets is None:
            # y = pd.read_csv('lib/cits/SIC/datasets/nird/null/Y.csv')
            self.targets = torch.Tensor(Y)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.shuffle_targets:
            y_index = self.rng.randint(len(self.targets))
        else:
            y_index = index
        return self.data[index], self.targets[y_index]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())

        fmt_str += '=== X === \n'
        t = self.data.data if isinstance(self.data, torch.Tensor) else self.data
        s = '{:8s} [{:.4f} , {:.4f}] m+-s = {:.4f} +- {:.4f}'
        si = 'x'.join(map(str, t.shape if isinstance(t, np.ndarray) else t.size()))
        fmt_str += s.format(si, t.min(), t.max(), t.mean(), t.std()) + '\n'

        fmt_str += '=== Y === \n'
        t = self.targets.data if isinstance(self.targets, torch.Tensor) else self.targets
        s = '{:8s} [{:.4f} , {:.4f}] m+-s = {:.4f} +- {:.4f}'
        si = 'x'.join(map(str, t.shape if isinstance(t, np.ndarray) else t.size()))
        fmt_str += s.format(si, t.min(), t.max(), t.mean(), t.std()) + '\n'

        return fmt_str

    def __len__(self):
        return len(self.data)

    def get_feature_names(self):
        return self.features

    def get_groundtruth_features(self):
        return self.fea_groundtruth