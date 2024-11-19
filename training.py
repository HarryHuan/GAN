from __future__ import division
import torch
import numpy as np
from torch.utils import data
from torch import optim
from einops import rearrange
from torch.autograd import Variable
import torchvision.utils as vutils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trainer(object):
    def __init__(self, model_G, model_D, TrainDataset, args):
        self.args = args
        self.model_G = model_G
        self.model_D = model_D
        if torch.cuda.is_available():
            self.model_G.cuda()
            self.model_D.cuda()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # HHH shuffle=False：这指定了在每个epoch开始时是否随机打乱数据。设置为 False 意味着数据将按照原始顺序进行加载
        # HHH num_workers=0：这指定了用于数据加载的子进程数量。如果设置为 0 意味着数据加载将在主进程中进行，而不是在子进程中。这通常用于调试目的，因为使用多个子进程可能会导致调试更加困难。在实际训练中，通常会设置一个大于0的值，以利用多核CPU加速数据加载
        self.Trainloader = data.DataLoader(dataset=TrainDataset, batch_size=args.batchSize, shuffle=False, num_workers=int(args.workers))

        # HHH 损失函数
        self.criterion = torch.nn.BCELoss()
        self.criterionMSE = torch.nn.MSELoss()
        # HHH 优化器
        self.optimizerG = optim.Adam(self.model_G.parameters(), lr=self.args.lr, betas=(self.args.beta1, 0.999))
        self.optimizerD = optim.Adam(self.model_D.parameters(), lr=self.args.lr, betas=(self.args.beta1, 0.999))

        self.epochs = args.epochs
        self.start_epoch = args.start_epoch

    def train_model(self):
        self.model_G.train()
        self.model_D.train()
        for epoch in range(self.start_epoch, self.epochs):
            for i, data in enumerate(self.Trainloader, 0):
                input_cropped, label, mask = data

                # HHH Variable 类用于表示张量（tensor）并附加自动微分的功能
                # HHH 需要注意的是，在PyTorch 0.4及以后的版本中，Variable 类和 Tensor 类被合并为 torch.Tensor 类。现在，所有的张量都是变量，这意味着它们都具备自动微分的功能。因此，直接使用 torch.tensor 创建的张量就已经包含了 Variable 的功能
                input = Variable(label)
                input = input.to(device)
                label = label.to(device)

                # train with real
                # HHH 用于清除（归零）模型 model_D 的梯度的方法。这通常在训练过程中的每次迭代开始时调用，以确保旧的梯度不会累积或影响新的梯度计算
                self.model_D.zero_grad()
                # HHH 真图只跑D
                output_D_real = self.model_D(input)
                output_D_real = output_D_real.to(device)
                # HHH label_real是output_D_real的形状 全 1 常量
                label_real = torch.full((output_D_real.size()), 1, dtype=torch.float)
                label_real = label_real.to(device)
                # HHH 损失  输入为真 返回必须为真
                loss_D_real = self.criterion(output_D_real, label_real)
                loss_D_real.backward()

                # train with fake
                input_cropped = input_cropped.to(device)
                # HHH 假图跑G又跑D
                fake = self.model_G(input_cropped)
                # HHH 对一个张量调用 .detach() 方法时，PyTorch会返回一个新的张量，这个张量与原始张量共享数据，但是不会在反向传播中跟踪梯度
                # HHH 对于model_D的传入参数，多乘以一个mask，将中间以外的值置零
                mask1 = rearrange(mask, "b h w c -> b c h w", c=mask.shape[3], h=mask.shape[1], w=mask.shape[2])
                mask1 = mask1.to(device)
                tmp = fake.detach() * mask1
                output_D_fake = self.model_D(fake.detach() * mask1)
                output_D_fake = output_D_fake.to(device)
                # HHH label_fake是output_D_fake的形状 全 0 常量
                label_fake = torch.full((output_D_fake.size()), 0, dtype=torch.float)
                label_fake = label_fake.to(device)
                # HHH 损失  输入为假 返回必须为假
                loss_D_fake = self.criterion(output_D_fake, label_fake)
                loss_D_fake.backward()

                lossD_total = (loss_D_real + loss_D_fake) * 0.5
                self.optimizerD.step()

                # 生成器的训练
                self.model_G.zero_grad()
                output_D_fake = self.model_D(fake)
                label_real = torch.full((output_D_fake.size()), 1, dtype=torch.float)
                label_real = label_real.to(device)
                loss_G = self.criterionMSE(output_D_fake, label_real)

                mask = rearrange(mask, "b h w c -> b c h w", c=mask.shape[3], h=mask.shape[1], w=mask.shape[2])
                mask = mask.to(device)

                lossG_recon = torch.mean(0.998 * ((fake - label) * mask).pow(2))
                total_loss_G = (1 - 0.998) * loss_G + 0.998 * lossG_recon
                total_loss_G.backward()
                self.optimizerG.step()

                print(f"Epoch [{epoch + 1}/{(self.epochs - self.start_epoch)}] Batch [{i + 1}/{len(self.Trainloader)}] "
                      f"Loss_D: {loss_D_real.item() + loss_D_fake.item()} Loss_G: {loss_G.item()}")

                if i % 80 == 0:
                    vutils.save_image(label, 'result/train/real/real_samples_epoch_%03d.png' % epoch)
                    vutils.save_image(input_cropped, 'result/train/cropped/cropped_samples_epoch_%03d.png' % epoch)
                    recon_image = input_cropped.clone()
                    recon_image.data[:, :, int(128 / 4):int(128 / 4 + 128 / 2), int(128 / 4):int(128 / 4 + 128 / 2)] = fake.data
                    vutils.save_image(recon_image.data, 'result/train/recon/recon_samples_epoch_%03d.png' % epoch)

        torch.save(self.model_G, 'model_G.pth')
        torch.save(self.model_D, 'model_D.pth')
