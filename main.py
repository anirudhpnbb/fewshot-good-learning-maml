import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import Omniglot
import numpy as np
import random

class ConvNet(nn.Module):
    def __init__(self, num_classes=5):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(64 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class MAML:
    def __init__(self, model, lr_inner=0.01, lr_outer=0.001, num_inner_steps=1):
        self.model = model
        self.lr_inner = lr_inner
        self.lr_outer = lr_outer
        self.num_inner_steps = num_inner_steps
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr_outer)
    
    def inner_update(self, loss):
        grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
        updated_weights = []
        for param, grad in zip(self.model.parameters(), grads):
            updated_weights.append(param - self.lr_inner * grad)
        return updated_weights
    
    def forward_with_weights(self, x, weights):
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = x.view(x.size(0), -1)
        x = nn.functional.linear(x, weights[-1], weights[-2])
        return x
    
    def meta_update(self, task_losses):
        meta_loss = torch.stack(task_losses).mean()
        self.optimizer.zero_grad()
        meta_loss.backward()
        self.optimizer.step()
    
    def train(self, tasks):
        task_losses = []
        for task in tasks:
            x_train, y_train, x_val, y_val = task
            for _ in range(self.num_inner_steps):
                y_pred = self.model(x_train)
                loss = nn.CrossEntropyLoss()(y_pred, y_train)
                updated_weights = self.inner_update(loss)
            y_pred_val = self.forward_with_weights(x_val, updated_weights)
            val_loss = nn.CrossEntropyLoss()(y_pred_val, y_val)
            task_losses.append(val_loss)
        self.meta_update(task_losses)


transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

omniglot_train = Omniglot(root='./data', background=True, transform=transform, download=True)
omniglot_test = Omniglot(root='./data', background=False, transform=transform, download=True)

def get_few_shot_tasks(dataset, num_classes=5, num_samples=5, num_tasks=32):
    tasks = []
    classes = list(set(target for _, target in dataset))
    for _ in range(num_tasks):
        sampled_classes = random.sample(classes, num_classes)
        x_train, y_train, x_val, y_val = [], [], [], []
        for i, cls in enumerate(sampled_classes):
            cls_images = [img for img, target in dataset if target == cls]
            images = random.sample(cls_images, num_samples + 1)
            x_train.extend(images[:num_samples])
            y_train.extend([i] * num_samples)
            x_val.append(images[num_samples])
            y_val.append(i)
        x_train = torch.stack(x_train)
        y_train = torch.tensor(y_train)
        x_val = torch.stack(x_val)
        y_val = torch.tensor(y_val)
        tasks.append((x_train, y_train, x_val, y_val))
    return tasks

model = ConvNet(num_classes=5)
maml = MAML(model)

num_epochs = 100
for epoch in range(num_epochs):
    tasks = get_few_shot_tasks(omniglot_train)
    maml.train(tasks)
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Meta Loss: {sum(maml.meta_loss) / len(maml.meta_loss):.4f}')
