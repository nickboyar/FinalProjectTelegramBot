import torch
import torch.nn as nn
import torch.nn.functional as F


class TwiceStyleLoss(nn.Module):
    def __init__(self, target1, target2):
        super(TwiceStyleLoss, self).__init__()
        self.center = target1.shape[3] // 2
        self.g_target1 = self.gram_matrix(target1[:, :, :, :self.center]).detach()
        self.g_target2 = self.gram_matrix(target2[:, :, :, self.center:]).detach()

        self.loss1 = F.mse_loss(self.g_target1, self.g_target1)
        self.loss2 = F.mse_loss(self.g_target2, self.g_target2)
        self.loss = self.loss1 + self.loss2

    def forward(self, x):
        g_x1 = self.gram_matrix(x[:, :, :, :self.center])
        g_x2 = self.gram_matrix(x[:, :, :, self.center:])

        self.loss1 = F.mse_loss(g_x1, self.g_target1)
        self.loss2 = F.mse_loss(g_x2, self.g_target2)
        self.loss = self.loss1 + self.loss2

        return x

    def gram_matrix(self, x):
        batch_size, h, w, f_map_num = x.size()
        features = x.reshape(batch_size * h, w * f_map_num)
        G = torch.mm(features, features.t())

        return G.div(batch_size * h * w * f_map_num)
