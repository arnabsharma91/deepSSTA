import torch, torch.nn as nn

hidden_dims2 = {'linear': 896, 'input_shape': [5,10]}
class GRU2(nn.Module):
    def __init__(self, dims_dict):
        super(GRU2, self).__init__()
        self.name = 'GRU'
        self.dims_dict = dims_dict
        self.gru = nn.GRU(dims_dict['input_shape'][1], 128, num_layers=2, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(dims_dict['linear'], 1024),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128))
        self.head = nn.Linear(128, 1)
        
        self.loss = nn.MSELoss()
        
    def forward(self, x, y=None):
        shape1, shape2 = self.dims_dict['input_shape']
        x = x.reshape(x.shape[0],shape1,shape2)
        if y is None:
            out, hn = self.gru(x)
            out = out.reshape(out.shape[0],-1)
            out = torch.cat([out, hn.reshape(hn.shape[1], -1)], dim=1)
            out = self.head(self.linear(out))
            return out
        else:
            out, hn = self.gru(x)
            out = out.reshape(out.shape[0],-1)
            out = torch.cat([out, hn.reshape(hn.shape[1], -1)], dim=1)
            out = self.head(self.linear(out))
            loss = self.loss(out, y)
            return loss

class LSTM2(nn.Module):
    def __init__(self, dims_dict):
        super(LSTM2, self).__init__()
        self.name = 'LSTM'
        self.dims_dict = dims_dict
        self.lstm = nn.LSTM(dims_dict['input_shape'][1], 128, num_layers=2, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(dims_dict['linear'], 1024),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128))
        self.head = nn.Linear(128, 1)
        
        self.loss = nn.MSELoss()
        
    def forward(self, x, y=None):
        shape1, shape2 = self.dims_dict['input_shape']
        x = x.reshape(x.shape[0],shape1,shape2)
        if y is None:
            out, (hn, cn) = self.lstm(x)
            out = out.reshape(out.shape[0],-1)
            out = torch.cat([out, hn.reshape(hn.shape[1], -1)], dim=1)
            out = self.head(self.linear(out))
            return out
        else:
            out, (hn, cn) = self.lstm(x)
            out = out.reshape(out.shape[0],-1)
            out = torch.cat([out, hn.reshape(hn.shape[1], -1)], dim=1)
            out = self.head(self.linear(out))
            loss = self.loss(out, y)
            return loss