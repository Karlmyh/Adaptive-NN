import torch
from model.model import MLP
from trainer.trainer import train
from data_loader.data_loader import simulation_data_loader


d=100
n_train=10000
n_test=20000
learning_rate=0.5






loss_fn=torch.nn.MSELoss()
model=MLP(d)
data_loader=simulation_data_loader(d,n_train,n_test)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[700,2100], gamma=0.5)


trained_model=train(model, data_loader,loss_fn,optimizer,scheduler)


