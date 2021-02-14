import torch
import torch.nn as nn



class Normalization(nn.Module):
    def __init__(self, device):
        super(Normalization, self).__init__()
        mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)


    def forward(self, img):
        return (img - self.mean) / self.std