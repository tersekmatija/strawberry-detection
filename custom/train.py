from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from utils.loss import CombinedLoss
from datasets.strawberrydi import StrawDIDataset
from models.model import Model

batch_size = 2
learning_rate = 0.0001
train_dataset = StrawDIDataset(split="train", root="/home/matija/Luxonis/people-detseg/data/StrawDI_Db1", transforms=torchvision.transforms.CenterCrop(768))
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, collate_fn=StrawDIDataset.collate_fn)

model = Model(1)
model.train()
model.cuda()

criterion = CombinedLoss(model.nc, model.gr, anchors = [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]])

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

epochs=30

for epoch in range(epochs):
    with tqdm(total=len(trainloader.dataset), desc ='Training - Epoch: '+str(epoch)+"/"+str(epochs), unit='chunks') as prog_bar:
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)

            #print(labels[0].shape)
            #labels[0] = torch.nn.functional.interpolate(labels[0], scale_factor=0.25, mode='nearest')
            labels[0] = labels[0].cuda()
            labels[1] = labels[1].cuda()
            #print(outputs[1].shape)
            #print(labels[1].shape)


            loss = criterion(outputs,labels)
            loss.backward()

            #torch.nn.utils.clip_grad_norm_(model.parameters(),
            #                          max_norm=10.0)
            
            optimizer.step()
            prog_bar.set_postfix(**{'run:': "model_name",'lr': learning_rate,
                                    'loss': loss.item()
                                    })
            prog_bar.update(batch_size)