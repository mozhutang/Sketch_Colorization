import torch
import torch.nn as nn
import torch.optim as optim

import random
import time
import os

from torchvision import transforms

from PIL import Image

Norm = nn.BatchNorm2d

class PatchGAN(nn.Module):
    def __init__(self, dim=64, norm='batch', sigmoid=True):
        super(PatchGAN, self).__init__()
        self.norm = nn.BatchNorm2d
        
        self.dim = dim

        layers = nn.ModuleList()

        layers.append(self._building_block(6, self.dim))

        layers.append(self._building_block(self.dim, self.dim * 2))

        layers.append(self._building_block(self.dim * 2, self.dim * 8))

        layers.append(
            nn.Sequential(
                nn.Conv2d(self.dim * 8, 1, 4, 1, 1),
                nn.Sigmoid()))

        self.layers = nn.Sequential(*layers)

        for module in self.modules():
            nn.init.normal_(module.weight, 0, 0.02)
            
    def forward(self, image):
        return self.layers(image)

    def _building_block(self, in_channel, out_channel, norm=True, stride=2):
        layers = []
        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 4, stride=2, padding=1),
                nn.ReLU(0.2)))
        return self.layers

class AttentionBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 gate_channels,
                 inter_channels=None,
                 bias=True):
        super(AttentionBlock, self).__init__()
        inter_channels=in_channels // 2

        self.W = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1)

        for module in self.modules():
            nn.init.normal_(module.weight, 0, 0.02)

    def forward(self, x, g):
        g = self.W(g)
        q = self.W(torch.relu(g + x))
        q = torch.sigmoid(q)
        resampled = F.interpolate(q, size=x.shape[2:])
        result = resampled.expand_as(x) * x
        result = self.W(result)

        return result, resampled

class DeepUNetPaintGenerator(nn.Module):

    def __init__(self, bias=True):
        super(DeepUNetPaintGenerator, self).__init__()

        self.bias = bias
        self.dim = 64

        self.down_sampler = self._down_sample()
        self.up_sampler = self._up_sample()
        self.attentions = self._attention_blocks()

        self.first_layer = nn.Conv2d(15, self.dim, 3, 1, 1)
            
        self.gate_block = nn.Conv2d(self.dim * 8,self.dim * 8,kernel_size=1,stride=1)

        self.last_layer = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(self.dim, 3, 3, 1, 1, bias=bias)
        )

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0, 0.02)


    def forward(self, image, colors):

        image = self.first_layer(image)

        image = self.last_layer(image)

        return image, []

    def _attention_blocks(self):
        layers = nn.ModuleList()

        for i in range(6):
            layers.append(AttentionBlock(self.dim * 8, self.dim * 8, bias=self.bias))
        
        return layers

    def _down_sample(self):
        layers = nn.ModuleList()

        layers.append(DeepUNetDownSample(self.dim, self.dim * 2, self.bias))

        layers.append(
            DeepUNetDownSample(self.dim * 2, self.dim * 4, self.bias))

        layers.append(
            DeepUNetDownSample(self.dim * 4, self.dim * 8, self.bias))

        layers.append(
            DeepUNetDownSample(self.dim * 8, self.dim * 8, self.bias))

        layers.append(
            DeepUNetDownSample(self.dim * 8, self.dim * 8, self.bias))

        layers.append(
            DeepUNetDownSample(self.dim * 8, self.dim * 8, self.bias))

        return layers

    def _up_sample(self):
        layers = nn.ModuleList()
        layers.append(
            DeepUNetUpSample(self.dim * 8 * 2, self.dim * 8, self.bias, True))
        layers.append(
            DeepUNetUpSample(self.dim * 8 * 2, self.dim * 8, self.bias, True))
        layers.append(
            DeepUNetUpSample(self.dim * 8 * 2, self.dim * 8, self.bias, True))
        layers.append(
            DeepUNetUpSample(self.dim * 8 * 2, self.dim * 4, self.bias))
        layers.append(
            DeepUNetUpSample(self.dim * 4 * 2, self.dim * 2, self.bias))
        layers.append(DeepUNetUpSample(self.dim * 2 * 2, self.dim, self.bias))
        return layers


class DeepUNetDownSample(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(DeepUNetDownSample, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True)
        self.norm1 = Norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True)
        self.norm2 = Norm(out_channels)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)


        self.channel_map = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        feature = torch.relu(x)

        feature = self.conv1(feature)
        feature = self.norm1(feature)

        feature = torch.relu(feature)
        feature = self.conv2(feature)
        feature = self.norm2(feature)

        connection = feature + self.channel_map(x)
        feature, idx = self.pool(connection)
        return feature, connection, idx


class DeepUNetUpSample(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, dropout=False):
        super(DeepUNetUpSample, self).__init__()
        self.pool = nn.MaxUnpool2d(2, 2)

        self.dropout = nn.Dropout2d(0.5, True) 

    def forward(self, x, connection, idx):
        x = self.pool(x, idx)
        feature = torch.relu(x)
        feature = torch.relu(feature)
        feature = self.conv(feature)
        feature = feature + self.channel_map(x)

        feature = self.dropout(feature)

        return feature



class ModelTrainer:
    def __init__(self, args, data_loader, device):

        self.args = args
        self.data_loader = data_loader
        self.device = device
        self.resolution = 512

    def train(self):
        raise NotImplementedError

    def validate(self, dataset, epoch, samples=3):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def save_model(self, name, epoch):
        raise NotImplementedError

    def _set_optimizers(self):
        raise NotImplementedError

    def _set_losses(self):
        raise NotImplementedError

    def _update_generator(self):
        raise NotImplementedError

    def _update_discriminator(self):
        raise NotImplementedError

class DeepUNetTrainer(ModelTrainer):
    def __init__(self, *args):
        super(DeepUNetTrainer, self).__init__(*args)

        self.generator = DeepUNetPaintGenerator().to(self.device)
        self.discriminator = PatchGAN(sigmoid=self.args.no_mse).to(self.device)

        self.optimizers = self._set_optimizers()

        self.losses = self._set_losses()

        self.image_pool = nn.MaxUnpool2d(2, 2)

        self.imageA = None
        self.imageB = None
        self.fakeB = None

    def train(self, last_iteration):

        average_trackers = [
            self.loss_G_gan, self.loss_D_fake, self.loss_D_real, self.loss_G_l1
        ]
        self.generator.train()
        self.discriminator.train()
        for tracker in average_trackers:
            tracker.initialize()
        for i, datas in enumerate(self.data_loader, last_iteration):
            imageA, imageB, colors = datas
            if self.args.mode == 'B2A':
                imageA, imageB = imageB, imageA

            self.imageA = imageA.to(self.device)
            self.imageB = imageB.to(self.device)
            colors = colors.to(self.device)

            self.fakeB, _ = self.generator(
                self.imageA,
                colors,
            )

            self._update_discriminator()
            self._update_generator()

        return i

    def validate(self, dataset, epoch, samples=3):

        length = len(dataset)

        idxs_total = [
            random.sample(range(0, length - 1), samples * 2)
            for _ in range(epoch)
        ]

        for j, idxs in enumerate(idxs_total):
            styles = idxs[samples:]
            targets = idxs[0:samples]

            result = Image.new(
                'RGB', (5 * self.resolution, samples * self.resolution))

            toPIL = transforms.ToPILImage()

            G_loss_gan = []
            G_loss_l1 = []
            D_loss_real = []
            D_loss_fake = []
            l1_loss = self.losses['L1']
            gan_loss = self.losses['GAN']
            for i, (target, style) in enumerate(zip(targets, styles)):
                sub_result = Image.new('RGB',
                                       (5 * self.resolution, self.resolution))
                imageA, imageB, _ = dataset[target]
                styleA, styleB, colors = dataset[style]

                if self.args.mode == 'B2A':
                    imageA, imageB = imageB, imageA
                    styleA, styleB = styleB, styleA

                imageA = imageA.unsqueeze(0).to(self.device)
                imageB = imageB.unsqueeze(0).to(self.device)
                styleB = styleB.unsqueeze(0).to(self.device)
                colors = colors.unsqueeze(0).to(self.device)

                with torch.no_grad():
                    fakeB, _ = self.generator(
                        imageA,
                        colors,
                    )
                    fakeAB = torch.cat([imageA, fakeB], 1)
                    realAB = torch.cat([imageA, imageB], 1)

                    G_loss_l1.append(l1_loss(fakeB, imageB).item())
                    G_loss_gan.append(
                        gan_loss(self.discriminator(fakeAB), True).item())

                    D_loss_real.append(
                        gan_loss(self.discriminator(realAB), True).item())
                    D_loss_fake.append(
                        gan_loss(self.discriminator(fakeAB), False).item())

                styleB = styleB.squeeze()
                fakeB = fakeB.squeeze()
                imageA = imageA.squeeze()
                imageB = imageB.squeeze()
                colors = colors.squeeze()


                color1 = color1.rotate(90)
                color2 = color2.rotate(90)
                color3 = color3.rotate(90)
                color4 = color4.rotate(90)

                color_result = Image.new('RGB',
                                         (self.resolution, self.resolution))
                color_result.paste(
                    color1.crop((0, 0, self.resolution, self.resolution // 4)),
                    (0, 0))
                color_result.paste(
                    color2.crop((0, 0, self.resolution, self.resolution // 4)),
                    (0, self.resolution // 4))
                color_result.paste(
                    color3.crop((0, 0, self.resolution, self.resolution // 4)),
                    (0, self.resolution // 4 * 2))
                color_result.paste(
                    color4.crop((0, 0, self.resolution, self.resolution // 4)),
                    (0, self.resolution // 4 * 3))

                sub_result.paste(imageA, (0, 0))
                sub_result.paste(styleB, (self.resolution, 0))
                sub_result.paste(fakeB, (2 * self.resolution, 0))
                sub_result.paste(imageB, (3 * self.resolution, 0))
                sub_result.paste(color_result, (4 * self.resolution, 0))

                result.paste(sub_result, (0, 0 + self.resolution * i))

    def _set_optimizers(self):
        optimG = optim.Adam(
            self.generator.parameters(),
            lr=self.args.learning_rate,
            betas=(self.args.beta1, 0.999))
        optimD = optim.Adam(
            self.discriminator.parameters(),
            lr=self.args.learning_rate,
            betas=(self.args.beta1, 0.999))

        return {'G': optimG, 'D': optimD}

    def _set_losses(self):
        gan_loss = nn.MSELoss()
        l1_loss = nn.L1Loss().to(self.device)

        return {'GAN': gan_loss, 'L1': l1_loss}

    def _update_generator(self):
        optimG = self.optimizers['G']
        gan_loss = self.losses['GAN']
        l1_loss = self.losses['L1']
        batch_size = self.imageA.shape[0]

        optimG.zero_grad()
        fake_AB = torch.cat([self.imageA, self.fakeB], 1)
        logit_fake = self.discriminator(fake_AB)
        loss_G_gan = gan_loss(logit_fake, True)

        loss_G_l1 = l1_loss(self.fakeB, self.imageB) * self.args.lambd

        self.loss_G_gan.update(loss_G_gan.item(), batch_size)
        self.loss_G_l1.update(loss_G_l1.item(), batch_size)

        loss_G = loss_G_gan + loss_G_l1

        loss_G.backward()
        optimG.step()

    def _update_discriminator(self):
        optimD = self.optimizers['D']
        gan_loss = self.losses['GAN']
        batch_size = self.imageA.shape[0]

        optimD.zero_grad()

        real_AB = torch.cat([self.imageA, self.imageB], 1)
        logit_real = self.discriminator(real_AB)
        loss_D_real = gan_loss(logit_real, True)
        self.loss_D_real.update(loss_D_real.item(), batch_size)

        fake_AB = torch.cat([self.imageA, self.fakeB], 1)
        fake_AB = self.image_pool(fake_AB)
        logit_fake = self.discriminator(fake_AB.detach())
        loss_D_fake = gan_loss(logit_fake, False)
        self.loss_D_fake.update(loss_D_fake.item(), batch_size)

        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        optimD.step()