### from internet 
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

##local 
import LinearNN as lnn



# set divce and checking for GPUs
device  = ( 
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device.")



### params & vars ###
model = lnn.NeuralNetwork().to(device)
loss_fn = nn.MSELoss()  # from paper
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

batch_size = 64
epochs = 5


trainging_data = 
test_data = 
train_dataloader = DataLoader(training_data, batch_size = batch_size)
test_dataloader = DataLoader(test_data, batch_size = batch_size)




### functions ###

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)   #forward()
        loss = loss_fn(pred, y)

        loss.backward()  # compute gradient steps
        optimizer.step()  # update params
        optimizer.zero_grad()  # zero the gradients for the next iterations 

        if batch % 100 == 0:
            loss, current  = lss.item(), (batch + 1) * len(X)
            print(f"loss: {loss::>5f} [{current:5>d}/{size:>5d}]")