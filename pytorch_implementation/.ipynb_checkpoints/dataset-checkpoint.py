class Dataset:
    def __init__(self, data_x, data_y=None):
        super(Dataset, self).__init__()
        self.data_x = data_x
        self.data_y = data_y

    def __len__(self):
        return len(self.data_x)
    
    def __getitem__(self, idx):
        if self.data_y is not None:
            return self.data_x[idx], self.data_y[idx]
        else:
            return self.data_x[idx]