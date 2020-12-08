import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm
from data_loader import load_data

class SRNET(nn.Module):
  def __init__(self):
    super(SRNET, self).__init__()

    self.model = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=9//2),
      nn.ReLU(),
      nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0//2),
      nn.ReLU(),
      nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, stride=1, padding=5//2),
      nn.ReLU(),
    )

  def forward(self, input):
    return torch.clamp(self.model(input), min=1e-12, max=1-(1e-12))

def training(data, batch=16, epochs=5000, prevModel=None):
  if prevModel != None:
    model = torch.load(prevModel)
  else:
    model = SRNET()
    loss_function = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=1e-2)

  if torch.cuda.is_available():
    model = model.cuda()

  best_loss = 9999
  for epoch in range(epochs):
    print('epoch {}/{}'.format(epoch+1, epochs), end=' | ')
    x, y = load_data(data, batch)
    if torch.cuda.is_available():
      x = x.cuda()
      y = y.cuda()
    
    opt.zero_grad()
    y_pred = model.forward(x)
    train_loss = loss_function(y_pred, y)
    train_loss.backward()
    opt.step()
    print('Loss : {} | Best Loss : {}'.format(train_loss, best_loss))
    #print(torch.cuda.memory_allocated(0))
    
    if train_loss.item() < best_loss:
      best_loss = train_loss.item()
      torch.save(model.state_dict(), './model.pth')

    torch.cuda.empty_cache()
  print('Best Loss:', best_loss)