# pytorch imports
import torch
from torch.utils.data.sampler import SequentialSampler, Sampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from sklearn.base import TransformerMixin, BaseEstimator, RegressorMixin
import ts_transformer
import torch.nn as nn
import torch.utils.data as data_utils
import os
import glob
import pandas as pd
import numpy as np

# get the device available
_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def to_device(data, device=None):
    """Move tensor(s) to chosen device"""
    if not device:
        device = _device

    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def get_dataloader(dataset, val_pct=None, batch_size=1, 
                    num_workers=0, shuffle=False, device=None):
    """
    Return a DeviceDataLoader object
    """

    if val_pct:
        train_indicies, val_indicies = split_indicies(len(dataset), val_pct)
        print("train indicies:", train_indicies[:10])
        print("val indicies:", val_indicies[:10])
    else:
        train_indicies = np.arange(len(dataset))
        print("train indicies:", train_indicies[:10])

    train_sampler = SubsetSequentialSampler(train_indicies)
    train_loader = DataLoader(dataset, batch_size, sampler=train_sampler, num_workers=num_workers,
            shuffle=shuffle)
    train_loader = DeviceDataLoader(train_loader, device=device)

    if val_pct:
        val_sampler = SubsetSequentialSampler(val_indicies)
        val_loader = DataLoader(dataset, batch_size, sampler=val_sampler, num_workers=num_workers,
            shuffle=shuffle)
        val_loader = DeviceDataLoader(val_loader, device=device)
        
        return train_loader, val_loader

    return train_loader

def get_dataloader2(dataset, batch_size=1, 
                    num_workers=0, shuffle=False, device=None):
    """
    Return a DeviceDataLoader object
    """

    # if val_pct:
    #     train_indicies, val_indicies = split_indicies(len(dataset), val_pct)
    #     print("train indicies:", train_indicies[:10])
    #     print("val indicies:", val_indicies[:10])
    # else:
    #     train_indicies = np.arange(len(dataset))
    #     print("train indicies:", train_indicies[:10])

    #train_sampler = SubsetSequentialSampler(train_indicies)
    train_loader = DataLoader(dataset, batch_size, num_workers=num_workers,
            shuffle=shuffle)
    train_loader = DeviceDataLoader(train_loader, device=device)

    # if val_pct:
    #     #val_sampler = SubsetSequentialSampler(val_indicies)
    #     val_loader = DataLoader(dataset, batch_size, num_workers=num_workers,
    #         shuffle=shuffle)
    #     val_loader = DeviceDataLoader(val_loader, device=device)
        
    #     return train_loader, val_loader

    return train_loader


class SubsetSequentialSampler(Sampler):
    """
    Subeset a data set sequentiallyto
    """
    def __init__(self, indicies):
        self.indicies = indicies

    def __iter__(self):
        return (
            self.indicies[i] for i in torch.arange(0, len(self.indicies))
        )

    def __len__(self):
        return len(self.indicies)


class DeviceDataLoader():
    """
    Wrap a dataloader to move data to a device
    """
    def __init__(self, dl, device=None):
        self.dl = dl
        if device:
            device = device
        else:
            device = _device

        self.device = device

    def __iter__(self):
        """
        Yield a batch of data after moving it to device
        """
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """
        Number of batches
        """
        return len(self.dl)


class BejingAirDataset(Dataset):
    def __init__(
        self,
        root,
        train = True):

        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        
        self.root = root
        self.train = train
        self.features = ['wd__Wx', 'wd__Wy', 'X__PM10', 'X__SO2', 'X__NO2', 'X__CO',
            'X__O3', 'X__TEMP', 'X__PRES', 'X__DEWP', 'X__RAIN',
            'cos_sin__Hour sin', 'cos_sin__Hour cos', 'cos_sin__Day sin',
            'cos_sin__Day cos', 'cos_sin__Year sin', 'cos_sin__Year cos',
            'cat__x0_Aotizhongxin', 'cat__x0_Changping', 'cat__x0_Dingling',
            'cat__x0_Dongsi', 'cat__x0_Guanyuan', 'cat__x0_Gucheng',
            'cat__x0_Huairou', 'cat__x0_Nongzhanguan', 'cat__x0_Shunyi']
        self.target = ["y__PM2.5"]
        #self.cache = {}

        # Get the file paths
        if self.train:
            self.files = glob.glob(os.path.join(self.train_folder, "*.csv"))
        else:
            self.files = glob.glob(os.path.join(self.test_folder, "*.csv"))
        #print("self.files:", self.files)

    @property
    def train_folder(self):
        return os.path.join(self.root, "train")
    
    @property
    def test_folder(self):
        return os.path.join(self.root, "test")


    def __getitem__(self, index, index_col=0):
        """
        Fetch the file, split the data into X and y,
        convert to a tensor object to be passed to a model
        """
        #print(index)
        #self.data = self.cache.get(index, None)

        data_path = self.files[index]
        #print('fetching... {0}'.format(data_path))

        #if self.data is None:
        # Read in the data
        df = pd.read_csv(data_path)
        #df = df.iloc[index, :]
        # Pass it to a tensor through numpy and torch
        #print(os.path.basename(data_path))
        #print(df.columns)
        self.X = torch.from_numpy(df[self.features].values).float()
        self.y = torch.from_numpy(df[self.target].values).float()
        #self.data = torch.from_numpy(df.values)
        
        return self.X, self.y

    def __len__(self):
        return len(self.files)


class PytorchTransformer(BaseEstimator, RegressorMixin):
    """
    Pytorch transformer to be used in a sklearn pipeline
    """
    def __init__(self, device=None, batch_size=1, shuffle=False):
        
        if device is None:
            self.device =  torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = device

        self.shuffle = shuffle
        self._model = None
        self.batch_size = batch_size
        self._history = None
        self._losses = None
        self._running_loss = None

    def _build_model(self, feature_size=30, max_enc_len=5000):
        
        self._model = ts_transformer.TransAm(feature_size=feature_size, max_enc_len=max_enc_len, device=self.device)
        print("built model:", self.model)
        #return self._model.to(self.device)

    # def train(self, X, y, epochs=25):
    #     print("Creating optimizer and loss function")
    #     criterion = nn.MSELoss().to(self.device)
    #     optimizer = torch.optim.Adam(self._model.parameters())
    #     self._history = {"loss": [], "val_loss": [], "mse_loss": []}

    #     print("Dims: X {} | y {}".format(X.shape, y.shape))
    #     torch_x = torch.from_numpy(X).float().to(self.device)
    #     torch_y = torch.from_numpy(y).float().to(self.device)
    #     print("Dims: torch_X {} | torch_y {}".format(torch_x.size(), torch_y.shape))
    #     train = data_utils.TensorDataset(torch_x, torch_y)
    #     train_loader = data_utils.DataLoader(train, batch_size=self.batch_size,
    #                                          shuffle=self.shuffle)

    #     for epoch in range(epochs):
    #         running_loss = 0.0
    #         #losses = []
    #         for i, data in enumerate(train_loader):
    #             # get the inputs; data is a list of [inputs, labels]
    #             inputs, y = data.to(self.device), data.to(self.device)
    #             inputs = inputs.float()
    #             y = y.float()
    #             # zero the paramter gradients
    #             optimizer.zero_grad()

    #             # forward + backward + optimize
    #             outputs = self._model(inputs)
    #             loss = criterion(outputs, y)
    #             loss.backward()
    #             optimizer.step()

    #             self._history["loss"].append(loss.item())
    #             running_loss += loss.item()

    #             if i % 9 == (9 - 1):
    #                 print('[%d, %5d], total_loss %.3f,  lossL %.3f' %
    #                     (epoch + 1, i + 1, running_loss, running_loss / 9))
    #                 running_loss = 0.0
                    
                    

    def fit(self, X, y):
        self._build_model()
        #self.train(X, y)

        return self                
    
    def predict(self, X, y=None):

        torch_x = torch.from_numpy(X).float().to(self.device)
        #torch_y = torch.from_numpy(y).float().to(self.device)

        y_pred = self._model(torch_x)
        y_pred_formatted = y_pred.cpu().data.numpy() if self._gpu else y_pred.data.numpy()
        return y_pred_formatted