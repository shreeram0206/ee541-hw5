import torch 
import torch.nn as nn
import torch.optim as optim
import os.path as osp
from utils import Config
from model import model
from data import get_dataloader
import pandas as pd

model_path = '/home/shreeram_narayanan26/shreeram/ee541-hw5-starter/pytorch/model.pth'

def test_model(dataloader, model, criterion, device, dataset_size):
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Model successfully loaded in eval mode")
    phase = 'test'
    predictions = []
    labels_array = []
    for data, labels in dataloader[phase]:
        # print("Labels variable is", labels.tolist())
        labels_array.append(labels.tolist())
        data = data.cuda()
        labels = labels.cuda()
        target = model(data)
        _, pred = torch.max(target, 1)
        # print("Pred variable is", pred.tolist())
        loss = criterion(target, labels)
        predictions.append(pred.tolist())
    print("The length of prediction array in test mode is ", len(predictions))
    df = pd.DataFrame(zip(labels_array, predictions), columns=['Ground Truth Labels', 'Predictions'])
    df.to_csv("output.csv", index=False)

if __name__=='__main__':
    dataloader, classes, dataset_size = get_dataloader(debug=Config['debug'], batch_size=Config['batch_size'], num_workers=Config['num_workers'])
    print("The test set data has following number of images/labels", len(dataloader['test']))
    device = torch.device('cuda:0' if torch.cuda.is_available() and Config['use_cuda'] else 'cpu')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, classes)
    criterion = nn.CrossEntropyLoss()
    test_model(dataloader, model, criterion, device, dataset_size=dataset_size)

