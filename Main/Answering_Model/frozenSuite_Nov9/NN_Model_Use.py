
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class Model(nn.Module):

    def __init__(self,in_features=30*10,h1=2048,h2=2048,h3=1024*2,h4=1024,h5=900,h6=900,h7=800,
                 h8=800,h9 = 800,h10=800,h11=800,h12=800,h13=800,h14=800,h15=800,out_features=30):
        
        # How many layers?
        # Input layer (# of features) --> hidden layer 1 (number of neurons N) --> h2 (N) --> output (346 of classes)
        super().__init__()
        self.fc1 = nn.Linear(in_features,h1)
        self.bn1 = nn.BatchNorm1d(num_features=h1,momentum=0.01)
        self.fc2 = nn.Linear(h1,h2)
        self.d2 =  nn.Dropout(0.25)
        self.bn2 = nn.BatchNorm1d(num_features=h2,momentum=0.01)
        self.fc3 = nn.Linear(h2,h3)
        self.bn3 = nn.BatchNorm1d(num_features=h3,momentum=0.01)
        self.d3 = nn.Dropout(0.4)
        self.fc4 = nn.Linear(h3,h4)
        self.bn4 = nn.BatchNorm1d(num_features=h4,momentum=0.01)
        self.d4 = nn.Dropout(0.3)
        self.fc5 = nn.Linear(h4,h5)
        self.bn5 = nn.BatchNorm1d(num_features=h5,momentum=0.01)
        self.d5 = nn.Dropout(0.25)
        self.fc6 = nn.Linear(h5,h6)
        self.bn6 = nn.BatchNorm1d(num_features=h6,momentum=0.01)
        self.d6 = nn.Dropout(0.35)
        
        self.fc7 = nn.Linear(h6,h7)
        self.bn7 = nn.BatchNorm1d(num_features=h7,momentum=0.01)
        self.d7 = nn.Dropout(0.4)
        
        self.fc8 = nn.Linear(h7,h8)
        self.bn8 = nn.BatchNorm1d(num_features=h8,momentum=0.01)
        self.d8 = nn.Dropout(0.35)
        
        self.fc9 = nn.Linear(h8,h9)
        self.bn9 = nn.BatchNorm1d(num_features=h9,momentum=0.01)
        self.d9 = nn.Dropout(0.2)
        
        self.fc10 = nn.Linear(h9,h10)
        self.bn10 = nn.BatchNorm1d(num_features=h10,momentum=0.01)
        self.d10 = nn.Dropout(0.25)
        
        self.fc11 = nn.Linear(h10,h11)
        self.bn11 = nn.BatchNorm1d(num_features=h11,momentum=0.01)
        self.d11 = nn.Dropout(0.2)

        self.fc12 = nn.Linear(h11,h12)
        self.bn12 = nn.BatchNorm1d(num_features=h12,momentum=0.01)
        self.d12 = nn.Dropout(0.2)

        self.fc13 = nn.Linear(h12,h13)
        self.bn13 = nn.BatchNorm1d(num_features=h13,momentum=0.01)
        self.d13 = nn.Dropout(0.2)

        self.fc14 = nn.Linear(h13,h14)
        self.bn14 = nn.BatchNorm1d(num_features=h14,momentum=0.01)
        self.d14 = nn.Dropout(0.2)

        self.fc15 = nn.Linear(h14,h15)
        self.bn15 = nn.BatchNorm1d(num_features=h15,momentum=0.01)
        self.d15 = nn.Dropout(0.2)

        self.out = nn.Linear(h15,out_features)
  
    def forward(self,x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.d2(self.fc2(x))))
        x = F.relu(self.bn3(self.d3(self.fc3(x))))
        x = F.relu(self.bn4(self.d4(self.fc4(x))))
        x = F.relu(self.bn5(self.d5(self.fc5(x))))
        x = F.relu(self.bn6(self.d6(self.fc6(x))))
        x = F.relu(self.bn7(self.d7(self.fc7(x))))
        x = F.relu(self.bn8(self.d8(self.fc8(x))))
        x = F.relu(self.bn9(self.d9(self.fc9(x))))
        x = F.relu(self.bn10(self.d10(self.fc10(x))))
        x = F.relu(self.bn11(self.d11(self.fc11(x))))
        x = F.relu(self.bn12(self.d12(self.fc12(x))))
        x = F.relu(self.bn13(self.d13(self.fc13(x))))
        x = F.relu(self.bn14(self.d14(self.fc14(x))))
        x = F.relu(self.bn15(self.d15(self.fc15(x))))

        x = torch.sigmoid(self.out(x))
        return x


model = Model()
model.load_state_dict(torch.load('MyModel_V5_epoch400.pt', map_location=torch.device('cpu')));
model.eval()
device = torch.device("cpu")
model.to(device)



def getProbability(feature_list):
    feature = torch.tensor(np.array([feature_list]))
    prob = model(feature.type(torch.float))
    return prob

import pickle

with open('XGBoost_43.pkl','rb') as f:
    XGB = pickle.load(f)

def XGBoost_Answer(feature_list):
    feature = np.array([feature_list])
    ans = XGB.predict(feature)
    return ans
    
