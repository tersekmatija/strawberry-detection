# Adapted from UFLD
import math

class CosineAnnealingLR:
    def __init__(self, optimizer, T_max , eta_min = 0, warmup = None, warmup_iters = None):
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min

        self.iters = 0
        self.base_lr = [group['lr'] for group in optimizer.param_groups]

    def step(self, external_iter = None):
        self.iters += 1
        if external_iter is not None:
            self.iters = external_iter
        if self.warmup == 'linear' and self.iters < self.warmup_iters:
            rate = self.iters / self.warmup_iters
            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = lr * rate
            return
        
        # cos policy

        for group, lr in zip(self.optimizer.param_groups, self.base_lr):
            group['lr'] = self.eta_min + (lr - self.eta_min) * (1 + math.cos(math.pi * self.iters / self.T_max)) / 2

class LinearLR:
    def __init__(self, optimizer, T_max, eta_min = 0.00001, warmup = None, warmup_iters = None):
        self.optimizer = optimizer
        self.eta_min = eta_min
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.T_max = T_max
        self.iters = 0        
        self.base_lr = [group['lr'] for group in optimizer.param_groups]


    def step(self, external_iter = None):
        self.iters += 1
        if external_iter is not None:
            self.iters = external_iter

        if self.warmup == 'linear' and self.iters < self.warmup_iters:
            rate = self.iters / self.warmup_iters
            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = lr * rate
            return
        
        for group, lr in zip(self.optimizer.param_groups, self.base_lr):
            k = (lr - self.eta_min) / (self.warmup_iters - self.T_max)
            group['lr'] = max((self.iters-self.warmup_iters)* k + lr, self.eta_min)


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import torch
    import torchvision

    model = torchvision.models.resnet18(pretrained=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.937, nesterov=True)
    scheduler = LinearLR(optimizer, T_max = 5000, eta_min = 0.0001, warmup="linear", warmup_iters=500)

    x = []
    y = []
    for i in range(7000):
        x.append(i)
        scheduler.step()
        y.append(optimizer.param_groups[0]["lr"])

    plt.plot(x, y)
    plt.show()