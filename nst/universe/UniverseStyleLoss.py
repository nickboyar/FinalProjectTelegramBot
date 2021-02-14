import torch
import torch.nn as nn
import torch.nn.functional as F


class UniverseStyleLoss(nn.Module):
    def __init__(self, target1, target2):
        super(UniverseStyleLoss, self).__init__()
        self.g_target1 = self.gram_matrix(target1).detach()
        self.g_target2 = self.gram_matrix(target2).detach()
        self.loss1 = F.mse_loss(self.g_target1, self.g_target1)
        self.loss2 = F.mse_loss(self.g_target2, self.g_target2)
        self.loss = self.loss1 + self.loss2

    def forward(self, input):
        g_input1 = self.gram_matrix(input)
        g_input2 = self.gram_matrix(input)

        self.loss1 = F.mse_loss(g_input1, self.g_target1)
        self.loss2 = F.mse_loss(g_input2, self.g_target2)
        self.loss = self.loss1 + self.loss2

        return input

    def gram_matrix(self, input):
        batch_size, h, w, f_map_num = input.size()
        features = input.reshape(batch_size * h, w * f_map_num)
        G = torch.mm(features, features.t())
        return G.div(batch_size * h * w * f_map_num)