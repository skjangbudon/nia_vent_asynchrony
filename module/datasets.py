import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

class CustomDataset(Dataset): 
  def __init__(self, x_data, y_data, prt=False):
    if prt: print('x_data', x_data.shape)
    trans = transforms.Compose([transforms.ToTensor()])
    self.x_data = trans(x_data).permute(1, 0, 2).float()
    self.y_data = trans(y_data.reshape(1, 1, -1)).long()
    if prt: print('trans x_data', self.x_data.shape)
    if prt: print('trans y_data', self.y_data.shape)
  
  def __len__(self): 
    return len(self.x_data)

  def __getitem__(self, idx, prt=False): 
    if prt: print(self.x_data.shape)
    x = self.x_data[idx] 
    y = self.y_data[idx]
    return x, y

def set_dataloader(dataset, drop_last=False, shuffle=False, batch_size=128):
    num_workers = 5
    return DataLoader(dataset, batch_size=int(batch_size), drop_last=drop_last, shuffle=shuffle, 
            num_workers=num_workers)
            
def setup_data(X, y, make_valset=False):
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    if make_valset:
      X_train, X_val, y_train, y_val = train_test_split(
          X_trainval, y_trainval, test_size=0.125, random_state=42)
    else:
      X_train, y_train = X_trainval, y_trainval

    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    trainset = CustomDataset(X_train, y_train)
    testset = CustomDataset(X_test, y_test)

    train_dataloader = set_dataloader(trainset, drop_last=True, shuffle=True)
    val_dataloader = None
    test_dataloader = set_dataloader(testset, batch_size=1)

    if make_valset:
      X_val = scaler.transform(X_val)
      valset = CustomDataset(X_val, y_val)
      val_dataloader = set_dataloader(valset, batch_size=1)
      print('X.shape', X_train.shape, X_val.shape, X_test.shape)
      print('Y class distribution', sum(y_train==1)/len(y_train), sum(y_val==1)/len(y_val), sum(y_test==1)/len(y_test))
    else:
      print('X.shape', X_train.shape, X_test.shape)
      print('Y class distribution', sum(y_train==1)/len(y_train), sum(y_test==1)/len(y_test))

    return train_dataloader, val_dataloader, test_dataloader
