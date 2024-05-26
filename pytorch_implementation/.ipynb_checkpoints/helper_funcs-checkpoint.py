import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import LSTM, GRU, hidden_dims

def inference_pytorch(model, dataloader):
    model.eval()
    preds = []
    for x in dataloader:
        if torch.cuda.is_available():
            model.cuda()
            x = x.cuda()
        pred = model(x).detach().cpu().numpy()
        preds.append(pred)
    model.to('cpu')
    torch.cuda.empty_cache()
    return np.concatenate(preds, axis=0)

def average_prediction(X_test, trained_models):
    all_preds = []
    test_dataloader = DataLoader(Dataset(torch.FloatTensor(X_test)), num_workers=4, batch_size=2048, shuffle=False)
    for i,model in enumerate(trained_models):
        current_pred = inference_pytorch(model, test_dataloader)
        all_preds.append(current_pred)
    return np.stack(all_preds, axis=1).mean(axis=1)

def load_trained(path):
    Models = []
    for fold in range(5):
        for Model in [GRU, LSTM]:
            model = Model(hidden_dims)
            for weights_path in os.listdir(path):
                if model.name in weights_path and f'fold{fold}' in weights_path:
                    model.load_state_dict(torch.load(f'{path}/{weights_path}', map_location='cpu'))
                    Models.append(model)
    return Models