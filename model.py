import torch
import torch.nn as nn
import numpy as np

class CNNDTIModel(nn.Module):
    def __init__(self, drug_input_dim, target_input_dim, hidden_dim, output_dim):
        super(CNNDTIModel, self).__init__()
        
        self.drug_conv = nn.Sequential(
            nn.Conv1d(
                1024,256,
                 kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.target_conv = nn.Sequential(
            nn.Conv1d(
                320,128,
                kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, drug_x, target_x):
        drug_x = drug_x.unsqueeze(1)      # (B,1,1024)
        target_x = target_x.unsqueeze(1)  # (B,1,320)

        print(drug_x.shape, target_x.shape)

        drug_x = self.drug_conv(drug_x)
        target_x = self.target_conv(target_x)
        combined = torch.cat((drug_x.view(drug_x.size(0), -1), target_x.view(target_x.size(0), -1)), dim=1)
        output = self.fc(combined)
        return output
    
class MLPDTIModel(nn.Module):
    def __init__(self, drug_input_dim, target_input_dim, hidden_dim, output_dim):
        super(MLPDTIModel, self).__init__()
        self.fc1 = nn.Linear(drug_input_dim + target_input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, drug_x, target_x):
        combined = torch.cat((drug_x, target_x), dim=1)
        hidden = self.relu(self.fc1(combined))
        output = self.fc2(hidden)
        return output
    
