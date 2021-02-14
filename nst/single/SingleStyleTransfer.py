from PIL import Image

import asyncio

import torch
import torch.nn as nn

import torch.optim as optim

import torchvision.transforms as transforms

import copy

from nst.ContentLoss import ContentLoss
from nst.Normalization import Normalization
from nst.single.SingleStyleLoss import SingleStyleLoss


class SingleStyleTransfer:
    def __init__(self, style_img, content_img, imsize=1024, num_steps=500,
                 style_weight=100000, content_weight=1):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.imsize = imsize
        self.style_img = self.image_loader(style_img)
        self.content_img = self.image_loader(content_img)
        self.input_img = self.content_img.clone()

        self.content_layers = ['conv_4']
        self.style_layers = ['conv_2', 'conv_3', 'conv_4', 'conv_5']

        self.num_steps = num_steps
        self.style_weight = style_weight
        self.content_weight = content_weight

    def image_loader(self, image_name):
        loader = transforms.Compose([
            transforms.Resize(self.imsize),
            transforms.CenterCrop(self.imsize),
            transforms.ToTensor()])

        image = Image.open(image_name)
        image = loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)

    def get_style_model_and_losses(self):
        cnn = copy.deepcopy(CNN)

        normalization = Normalization(self.device).to(self.device)

        content_losses = []
        style_losses = []

        model = nn.Sequential(normalization)

        i = 0
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in self.content_layers:
                target = model(self.content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in self.style_layers:
                target_feature = model(self.style_img).detach()
                style_loss = SingleStyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], SingleStyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses

    def get_input_optimizer(self):
        optimizer = optim.LBFGS([self.input_img.requires_grad_()])
        return optimizer


    async def transfer(self):
        global CNN

        CNN = torch.load('nets/st.pth', ).to(self.device).eval()

        model, style_losses, content_losses = self.get_style_model_and_losses()
        optimizer = self.get_input_optimizer()

        print('Optimizing..')
        run = [0]

        while run[0] <= self.num_steps:
            await asyncio.sleep(2)

            def closure():
                self.input_img.data.clamp_(0, 1)
                optimizer.zero_grad()
                model(self.input_img)
                style_score = 0
                content_score = 0
                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss
                style_score *= self.style_weight
                content_score *= self.content_weight
                loss = style_score + content_score
                loss.backward()
                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()
                return style_score + content_score

            optimizer.step(closure)
        self.input_img.data.clamp_(0, 1)
        return self.input_img